from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .instruction_matrix_grammar import INSTRUCTION_GRAMMAR_SCHEMA_VERSION
from .utils import (
    IGNORE_INDEX,
    canonicalize_foreground,
    label_diversity_on_fg,
    load_config,
    load_label_grid,
    resolve_manifest_path,
    save_json,
    save_jsonl,
)


CANONICAL_SIZE = 20
NUM_LABELS = 16
NUM_CELLS = CANONICAL_SIZE * CANONICAL_SIZE
LARGE_PENALTY = 1.0e12


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate instruction_matrix_grammar_prior as P(Y | true category).")
    parser.add_argument("--cache", type=Path, default=Path("knit_decode/struct_foreground_v1/cache/foreground_v1/foreground_cache_train.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/manifests/inverse_rendering_val_frontonly.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("knit_decode/struct_foreground_v1/configs/foreground_v1.yaml"))
    parser.add_argument("--split-name", type=str, default="val")
    parser.add_argument("--output-dir", type=Path, default=Path("output/struct_foreground_v1/instruction_grammar_likelihood/val"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--label-smoothing", type=float, default=1.0e-4)
    parser.add_argument("--mask-smoothing", type=float, default=1.0e-4)
    parser.add_argument("--w-mask", type=float, default=1.0)
    parser.add_argument("--w-label", type=float, default=1.0)
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--use-calibrated-label-prob", dest="use_calibrated_label_prob", action="store_true", default=None)
    parser.add_argument("--no-calibrated-label-prob", dest="use_calibrated_label_prob", action="store_false")
    parser.add_argument("--use-smoothed-mode-prior", dest="use_smoothed_mode_prior", action="store_true", default=None)
    parser.add_argument("--no-smoothed-mode-prior", dest="use_smoothed_mode_prior", action="store_false")
    parser.add_argument("--compare-raw-calibrated", action="store_true")
    parser.add_argument("--sweep-calibration", action="store_true")
    parser.add_argument("--sweep-categories", type=str, default="")
    parser.add_argument("--write-recommended-config", action="store_true")
    return parser


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required to load the foreground cache.") from error


def _load_manifest(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Manifest row {line_no} is not an object: {path}")
        rows.append(row)
        if max_samples is not None and len(rows) >= int(max_samples):
            break
    return rows


def _infer_manifest_root(manifest_path: Path, rows: list[dict[str, Any]]) -> Path:
    search_roots = [manifest_path.parent, *manifest_path.parents]
    sample = rows[: min(32, len(rows))]
    for candidate_root in search_roots:
        checked = 0
        ok = True
        for row in sample:
            raw = row.get("target_path")
            if not isinstance(raw, str):
                continue
            checked += 1
            if not (candidate_root / raw).exists():
                ok = False
                break
        if ok and checked > 0:
            return candidate_root
    return manifest_path.parent


def _to_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    if isinstance(value, tuple):
        return [_json_safe(child) for child in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def _clamp_prob(value: float, eps: float) -> float:
    return min(1.0 - eps, max(eps, float(value)))


def _smooth_mask_prob(value: float, smoothing: float, eps: float) -> float:
    p = float(value) * (1.0 - float(smoothing)) + 0.5 * float(smoothing)
    return _clamp_prob(p, eps)


def _smooth_label_prob(value: float, smoothing: float, eps: float) -> float:
    p = float(value) * (1.0 - float(smoothing)) + float(smoothing) / float(NUM_LABELS)
    return max(eps, min(1.0, p))


def _logsumexp(values: list[float]) -> float:
    if not values:
        return -math.inf
    max_value = max(values)
    if not math.isfinite(max_value):
        return max_value
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _entropy_from_counts(counts: Counter[int]) -> float:
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        prob = float(count) / float(total)
        if prob > 0.0:
            entropy -= prob * math.log(prob)
    return float(entropy)


def _config_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    cf = config.get("instruction_matrix_grammar_prior", {}) if isinstance(config.get("instruction_matrix_grammar_prior", {}), dict) else {}
    eval_cf = cf.get("evaluate_instruction_grammar_likelihood", {}) if isinstance(cf.get("evaluate_instruction_grammar_likelihood", {}), dict) else {}
    return bool(eval_cf.get(key, default))


def _mode_pruning_thresholds(config: dict[str, Any]) -> dict[str, float]:
    cf = config.get("instruction_matrix_grammar_prior", {}) if isinstance(config.get("instruction_matrix_grammar_prior", {}), dict) else {}
    calibration_cf = cf.get("calibration", {}) if isinstance(cf.get("calibration", {}), dict) else {}
    pruning_cf = calibration_cf.get("mode_pruning", {}) if isinstance(calibration_cf.get("mode_pruning", {}), dict) else {}
    return {
        "max_unused_fraction_warn": float(pruning_cf.get("max_unused_fraction_warn", 0.5)),
        "max_mode_fraction_warn": float(pruning_cf.get("max_mode_fraction_warn", 0.8)),
        "min_mode_usage_fraction": float(pruning_cf.get("min_mode_usage_fraction", 0.01)),
    }


def _mode_usage_cleanup_config(config: dict[str, Any]) -> dict[str, Any]:
    cf = config.get("instruction_matrix_grammar_prior", {}) if isinstance(config.get("instruction_matrix_grammar_prior", {}), dict) else {}
    calibration_cf = cf.get("calibration", {}) if isinstance(cf.get("calibration", {}), dict) else {}
    cleanup_cf = calibration_cf.get("mode_usage_cleanup", {}) if isinstance(calibration_cf.get("mode_usage_cleanup", {}), dict) else {}
    return cleanup_cf if isinstance(cleanup_cf, dict) else {}


def _load_y_for_row(row: dict[str, Any], manifest_root: Path, data_cf: dict[str, Any]) -> tuple[list[list[int]], list[list[int]], bool]:
    sample_id = str(row.get("sample_id", row.get("id", "unknown")))
    if not isinstance(row.get("target_path"), str):
        raise ValueError(f"Manifest row missing target_path for sample_id={sample_id}.")
    target_path = resolve_manifest_path(str(row["target_path"]), manifest_root, sample_id=sample_id, field_name="target_path")
    y_raw = load_label_grid(target_path, sample_id=sample_id)
    canonical = canonicalize_foreground(
        y_raw,
        background_class_id=int(data_cf.get("background_class_id", 0)),
        canonical_size=int(data_cf.get("canonical_size", CANONICAL_SIZE)),
        canonical_mode=str(data_cf.get("canonical_mode", "full_masked")),
        ignore_index=int(data_cf.get("ignore_index", IGNORE_INDEX)),
    )
    fg_y20 = _to_list(canonical["fg_y20"])
    fg_mask20 = _to_list(canonical["fg_mask20"])
    y20: list[list[int]] = []
    mask20: list[list[int]] = []
    for y_pos in range(CANONICAL_SIZE):
        y_row: list[int] = []
        m_row: list[int] = []
        for x_pos in range(CANONICAL_SIZE):
            active = int(fg_mask20[y_pos][x_pos]) > 0
            label = int(fg_y20[y_pos][x_pos]) if active else 0
            if not active:
                label = 0
            if not 0 <= label <= NUM_LABELS:
                raise ValueError(f"sample_id={sample_id} has invalid canonical label {label} at ({y_pos},{x_pos}).")
            y_row.append(label)
            m_row.append(1 if active else 0)
        y20.append(y_row)
        mask20.append(m_row)
    return y20, mask20, bool(canonical.get("is_empty_foreground", False))


def _priors_from_counts(entry: dict[str, Any], mode_count: int, beta: float) -> list[float]:
    mode_num_samples = _to_list(entry.get("mode_num_samples", []))
    if not isinstance(mode_num_samples, list) or len(mode_num_samples) != mode_count:
        return [1.0 / float(mode_count) for _ in range(mode_count)]
    counts = [max(0.0, _safe_float(value)) for value in mode_num_samples]
    total = sum(counts)
    beta = max(0.0, float(beta))
    denom = total + beta * float(mode_count)
    if denom <= 0.0:
        return [1.0 / float(mode_count) for _ in range(mode_count)]
    priors = [(count + beta) / denom for count in counts]
    prior_sum = sum(priors)
    return [float(value) / prior_sum for value in priors] if prior_sum > 0.0 else [1.0 / float(mode_count) for _ in range(mode_count)]


def _validate_modes(
    entry: dict[str, Any],
    category: str,
    source_name: str,
    *,
    use_calibrated_label_prob: bool = True,
    use_smoothed_mode_prior: bool = True,
) -> tuple[list[Any], list[Any], list[float]]:
    fg_prob = _to_list(entry.get("basis_fg_mask_prob"))
    label_key = "basis_label_prob_16_calibrated" if use_calibrated_label_prob and entry.get("basis_label_prob_16_calibrated") is not None else "basis_label_prob_16"
    if (not use_calibrated_label_prob) and entry.get("basis_label_prob_16_raw") is not None:
        label_key = "basis_label_prob_16_raw"
    label_prob = _to_list(entry.get(label_key))
    if not isinstance(fg_prob, list) or not isinstance(label_prob, list):
        raise ValueError(f"{source_name} category {category!r} is missing basis_fg_mask_prob/basis_label_prob_16.")
    if len(fg_prob) != len(label_prob):
        raise ValueError(f"{source_name} category {category!r} has mismatched mode counts.")
    mode_count = len(fg_prob)
    if mode_count <= 0:
        raise ValueError(f"{source_name} category {category!r} has no modes.")
    prior_key = "mode_prior_smoothed" if use_smoothed_mode_prior else "mode_prior_raw"
    stored_priors = _to_list(entry.get(prior_key))
    if isinstance(stored_priors, list) and len(stored_priors) == mode_count and sum(max(0.0, _safe_float(value)) for value in stored_priors) > 0.0:
        total = sum(max(0.0, _safe_float(value)) for value in stored_priors)
        priors = [max(0.0, _safe_float(value)) / total for value in stored_priors]
    else:
        beta = _safe_float(entry.get("mode_prior_smoothing_beta"), 1.0 if use_smoothed_mode_prior else 0.0) if use_smoothed_mode_prior else 0.0
        priors = _priors_from_counts(entry, mode_count, beta)
    return fg_prob, label_prob, priors


def _mode_fg_grid(fg_mode: Any, *, category: str, mode_index: int, source_name: str) -> list[list[float]]:
    grid = _to_list(fg_mode)
    if isinstance(grid, list) and len(grid) == 1 and isinstance(grid[0], list):
        grid = grid[0]
    if not isinstance(grid, list) or len(grid) != CANONICAL_SIZE:
        raise ValueError(f"{source_name} {category} mode={mode_index} fg mask must have shape [1,20,20] or [20,20].")
    out: list[list[float]] = []
    for row in grid:
        if not isinstance(row, list) or len(row) != CANONICAL_SIZE:
            raise ValueError(f"{source_name} {category} mode={mode_index} fg mask must have shape [20,20].")
        out.append([float(value) for value in row])
    return out


def _mode_label_tensor(label_mode: Any, *, category: str, mode_index: int, source_name: str) -> list[list[list[float]]]:
    tensor = _to_list(label_mode)
    if not isinstance(tensor, list) or len(tensor) != NUM_LABELS:
        raise ValueError(f"{source_name} {category} mode={mode_index} label prob must have shape [16,20,20].")
    out: list[list[list[float]]] = []
    for channel in tensor:
        if not isinstance(channel, list) or len(channel) != CANONICAL_SIZE:
            raise ValueError(f"{source_name} {category} mode={mode_index} label prob must have shape [16,20,20].")
        out_channel: list[list[float]] = []
        for row in channel:
            if not isinstance(row, list) or len(row) != CANONICAL_SIZE:
                raise ValueError(f"{source_name} {category} mode={mode_index} label prob must have shape [16,20,20].")
            out_channel.append([float(value) for value in row])
        out.append(out_channel)
    return out


def _score_modes(
    y20: list[list[int]],
    mask20: list[list[int]],
    fg_modes: list[Any],
    label_modes: list[Any],
    priors: list[float],
    *,
    category: str,
    source_name: str,
    eps: float,
    label_smoothing: float,
    mask_smoothing: float,
    w_mask: float,
    w_label: float,
    include_mask: bool = True,
    include_label: bool = True,
) -> dict[str, Any]:
    num_fg = sum(int(value) for row in mask20 for value in row)
    mode_scores: list[dict[str, float | int]] = []
    log_terms: list[float] = []
    nan_or_inf = 0
    best_index = 0
    best_energy = math.inf
    for mode_index, (fg_raw, label_raw) in enumerate(zip(fg_modes, label_modes)):
        fg_grid = _mode_fg_grid(fg_raw, category=category, mode_index=mode_index, source_name=source_name)
        label_tensor = _mode_label_tensor(label_raw, category=category, mode_index=mode_index, source_name=source_name)
        mask_sum = 0.0
        label_sum = 0.0
        true_prob_sum = 0.0
        for y_pos in range(CANONICAL_SIZE):
            for x_pos in range(CANONICAL_SIZE):
                active = int(mask20[y_pos][x_pos]) > 0
                if include_mask:
                    p_mask = _smooth_mask_prob(float(fg_grid[y_pos][x_pos]), mask_smoothing, eps)
                    mask_sum += -math.log(p_mask if active else (1.0 - p_mask))
                if include_label and active:
                    label = int(y20[y_pos][x_pos])
                    p_label = _smooth_label_prob(float(label_tensor[label - 1][y_pos][x_pos]), label_smoothing, eps)
                    label_sum += -math.log(p_label)
                    true_prob_sum += p_label
        mask_mean = mask_sum / float(NUM_CELLS) if include_mask else 0.0
        label_mean = label_sum / float(max(num_fg, 1)) if include_label else 0.0
        energy_mean = float(w_mask) * mask_mean + float(w_label) * label_mean
        energy_total = float(w_mask) * mask_sum + float(w_label) * label_sum
        if not math.isfinite(energy_mean) or not math.isfinite(energy_total):
            nan_or_inf += 1
            energy_mean = LARGE_PENALTY
            energy_total = LARGE_PENALTY
        prior = max(eps, float(priors[mode_index]) if mode_index < len(priors) else 0.0)
        log_terms.append(math.log(prior) - energy_total)
        mode_scores.append(
            {
                "mode_id": int(mode_index),
                "map_energy": float(energy_mean),
                "energy_total": float(energy_total),
                "mask_nll_sum": float(mask_sum),
                "label_nll_sum": float(label_sum),
                "mask_nll_per_cell": float(mask_mean),
                "label_nll_per_fg": float(label_mean),
                "mean_true_token_probability": float(true_prob_sum / float(max(num_fg, 1))) if include_label else 0.0,
            }
        )
        if energy_mean < best_energy:
            best_energy = float(energy_mean)
            best_index = int(mode_index)
    mix_log = _logsumexp(log_terms)
    mixture_total = -mix_log if math.isfinite(mix_log) else LARGE_PENALTY
    if not math.isfinite(mixture_total):
        nan_or_inf += 1
        mixture_total = LARGE_PENALTY
    posterior: list[float] = []
    if math.isfinite(mix_log):
        posterior = [float(math.exp(value - mix_log)) for value in log_terms]
    best = mode_scores[best_index]
    return {
        "best_mode_id": int(best_index),
        "map_energy": float(best["map_energy"]),
        "mixture_nll_per_cell": float(mixture_total / float(NUM_CELLS)),
        "mask_nll_per_cell": float(best["mask_nll_per_cell"]),
        "label_nll_per_fg": float(best["label_nll_per_fg"]),
        "label_nll_sum": float(best["label_nll_sum"]),
        "mask_nll_sum": float(best["mask_nll_sum"]),
        "mean_true_token_probability": float(best["mean_true_token_probability"]),
        "mode_posterior": posterior,
        "mode_scores": mode_scores,
        "nan_or_inf_count": int(nan_or_inf),
    }


def _uniform_label_modes(mode_count: int) -> list[list[list[list[float]]]]:
    return [[[[1.0 / float(NUM_LABELS) for _ in range(CANONICAL_SIZE)] for _ in range(CANONICAL_SIZE)] for _ in range(NUM_LABELS)] for _ in range(mode_count)]


def _constant_mask_mode(prob: float) -> list[list[list[float]]]:
    return [[[float(prob) for _ in range(CANONICAL_SIZE)] for _ in range(CANONICAL_SIZE)]]


def _category_mean_mask(fg_modes: list[Any], category: str, source_name: str) -> list[list[float]]:
    grids = [_mode_fg_grid(mode, category=category, mode_index=index, source_name=source_name) for index, mode in enumerate(fg_modes)]
    return [
        [sum(float(grid[y_pos][x_pos]) for grid in grids) / float(len(grids)) for x_pos in range(CANONICAL_SIZE)]
        for y_pos in range(CANONICAL_SIZE)
    ]


def _category_label_hist(label_modes: list[Any], priors: list[float], category: str, source_name: str) -> list[float]:
    hist = [0.0 for _ in range(NUM_LABELS)]
    for mode_index, mode in enumerate(label_modes):
        tensor = _mode_label_tensor(mode, category=category, mode_index=mode_index, source_name=source_name)
        prior = float(priors[mode_index]) if mode_index < len(priors) else 1.0 / float(len(label_modes))
        for label_index in range(NUM_LABELS):
            hist[label_index] += prior * sum(sum(float(value) for value in row) for row in tensor[label_index])
    total = sum(hist)
    if total <= 0.0:
        return [1.0 / float(NUM_LABELS) for _ in range(NUM_LABELS)]
    return [float(value) / total for value in hist]


def _hist_label_mode(hist: list[float]) -> list[list[list[float]]]:
    return [[[float(hist[label_index]) for _ in range(CANONICAL_SIZE)] for _ in range(CANONICAL_SIZE)] for label_index in range(NUM_LABELS)]


def _score_baselines(
    y20: list[list[int]],
    mask20: list[list[int]],
    *,
    category: str,
    fg_modes: list[Any],
    label_modes: list[Any],
    raw_label_modes: list[Any],
    calibrated_label_modes: list[Any],
    priors: list[float],
    global_fg_rate: float,
    dictionary_entry: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    uniform_mask = _constant_mask_mode(global_fg_rate if 0.0 < global_fg_rate < 1.0 else 0.5)
    uniform_label = _uniform_label_modes(1)
    out["uniform_baseline"] = _score_modes(
        y20,
        mask20,
        uniform_mask,
        uniform_label,
        [1.0],
        category=category,
        source_name="uniform_baseline",
        eps=float(args.eps),
        label_smoothing=float(args.label_smoothing),
        mask_smoothing=float(args.mask_smoothing),
        w_mask=float(args.w_mask),
        w_label=float(args.w_label),
    )
    cat_mask = _category_mean_mask(fg_modes, category, "instruction_matrix_grammar_prior")
    cat_hist = _category_label_hist(label_modes, priors, category, "instruction_matrix_grammar_prior")
    out["category_hist_baseline"] = _score_modes(
        y20,
        mask20,
        [cat_mask],
        [_hist_label_mode(cat_hist)],
        [1.0],
        category=category,
        source_name="category_hist_baseline",
        eps=float(args.eps),
        label_smoothing=float(args.label_smoothing),
        mask_smoothing=float(args.mask_smoothing),
        w_mask=float(args.w_mask),
        w_label=float(args.w_label),
    )
    out["mode_mask_only"] = _score_modes(
        y20,
        mask20,
        fg_modes,
        _uniform_label_modes(len(fg_modes)),
        priors,
        category=category,
        source_name="mode_mask_only",
        eps=float(args.eps),
        label_smoothing=float(args.label_smoothing),
        mask_smoothing=float(args.mask_smoothing),
        w_mask=float(args.w_mask),
        w_label=float(args.w_label),
        include_mask=True,
        include_label=True,
    )
    out["mode_label_only_raw"] = _score_modes(
        y20,
        mask20,
        [_constant_mask_mode(0.5)[0] for _ in fg_modes],
        raw_label_modes,
        priors,
        category=category,
        source_name="mode_label_only_raw",
        eps=float(args.eps),
        label_smoothing=float(args.label_smoothing),
        mask_smoothing=float(args.mask_smoothing),
        w_mask=0.0,
        w_label=float(args.w_label),
        include_mask=False,
        include_label=True,
    )
    out["mode_label_only_calibrated"] = _score_modes(
        y20,
        mask20,
        [_constant_mask_mode(0.5)[0] for _ in fg_modes],
        calibrated_label_modes,
        priors,
        category=category,
        source_name="mode_label_only_calibrated",
        eps=float(args.eps),
        label_smoothing=float(args.label_smoothing),
        mask_smoothing=float(args.mask_smoothing),
        w_mask=0.0,
        w_label=float(args.w_label),
        include_mask=False,
        include_label=True,
    )
    if dictionary_entry is not None:
        try:
            d_fg, d_label, d_priors = _validate_modes(
                dictionary_entry,
                category,
                "dictionary_bank",
                use_calibrated_label_prob=False,
                use_smoothed_mode_prior=False,
            )
            out["dictionary_fallback"] = _score_modes(
                y20,
                mask20,
                d_fg,
                d_label,
                d_priors,
                category=category,
                source_name="dictionary_bank",
                eps=float(args.eps),
                label_smoothing=float(args.label_smoothing),
                mask_smoothing=float(args.mask_smoothing),
                w_mask=float(args.w_mask),
                w_label=float(args.w_label),
            )
        except Exception as error:
            out["dictionary_fallback"] = {"skipped": True, "warning": f"{type(error).__name__}: {error}"}
    return out


def _calibration_rows(
    y20: list[list[int]],
    mask20: list[list[int]],
    fg_mode: Any,
    label_mode: Any,
    *,
    category: str,
    mode_index: int,
    eps: float,
    label_smoothing: float,
    mask_smoothing: float,
) -> tuple[list[tuple[int, float, int]], list[float]]:
    fg_grid = _mode_fg_grid(fg_mode, category=category, mode_index=mode_index, source_name="calibration")
    label_tensor = _mode_label_tensor(label_mode, category=category, mode_index=mode_index, source_name="calibration")
    mask_rows: list[tuple[int, float, int]] = []
    true_token_probs: list[float] = []
    for y_pos in range(CANONICAL_SIZE):
        for x_pos in range(CANONICAL_SIZE):
            p_mask = _smooth_mask_prob(float(fg_grid[y_pos][x_pos]), mask_smoothing, eps)
            bin_index = min(9, max(0, int(math.floor(p_mask * 10.0))))
            active = int(mask20[y_pos][x_pos]) > 0
            mask_rows.append((bin_index, p_mask, 1 if active else 0))
            if active:
                label = int(y20[y_pos][x_pos])
                p_label = _smooth_label_prob(float(label_tensor[label - 1][y_pos][x_pos]), label_smoothing, eps)
                true_token_probs.append(p_label)
    return mask_rows, true_token_probs


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key, "")) for key in fieldnames})


def _write_histogram_png(path: Path, values_by_name: dict[str, list[float]]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        path.with_suffix(".warning.txt").write_text(f"matplotlib unavailable: {type(error).__name__}: {error}\n", encoding="utf-8")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for name, values in values_by_name.items():
        if values:
            plt.hist(values, bins=40, alpha=0.45, label=name)
    plt.xlabel("NLL / energy")
    plt.ylabel("sample count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return True


def _write_mask_calibration_png(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        path.with_suffix(".warning.txt").write_text(f"matplotlib unavailable: {type(error).__name__}: {error}\n", encoding="utf-8")
        return False
    x = [float(row["predicted_probability"]) for row in rows]
    y = [float(row["empirical_foreground_rate"]) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.plot([0.0, 1.0], [0.0, 1.0], color="gray", linestyle="--", linewidth=1)
    plt.plot(x, y, marker="o")
    plt.xlabel("predicted foreground probability")
    plt.ylabel("empirical foreground rate")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return True


def _summarize_by_category(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    categories = sorted({str(row["category"]) for row in rows})
    for category in categories:
        cat_rows = [row for row in rows if str(row["category"]) == category]
        out.append(
            {
                "category": category,
                "support": len(cat_rows),
                "mean_map_energy": _mean([float(row["map_energy"]) for row in cat_rows]),
                "mean_mixture_nll_per_cell": _mean([float(row["mixture_nll_per_cell"]) for row in cat_rows]),
                "mean_mask_nll_per_cell": _mean([float(row["mask_nll_per_cell"]) for row in cat_rows]),
                "mean_label_nll_per_fg": _mean([float(row["label_nll_per_fg"]) for row in cat_rows]),
                "mean_num_foreground": _mean([float(row["num_foreground"]) for row in cat_rows]),
                "mean_label_diversity": _mean([float(row["label_diversity"]) for row in cat_rows]),
            }
        )
    return out


def _baseline_summary(primary_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], primary_name: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {primary_name: primary_rows}
    for row in baseline_rows:
        name = str(row["baseline"])
        if name == primary_name:
            continue
        grouped.setdefault(name, []).append(row)
    out: list[dict[str, Any]] = []
    primary_mean = _mean([float(row["mixture_nll_per_cell"]) for row in primary_rows])
    for name in sorted(grouped):
        rows = grouped[name]
        mean_mix = _mean([float(row["mixture_nll_per_cell"]) for row in rows if "mixture_nll_per_cell" in row])
        out.append(
            {
                "baseline": name,
                "support": len(rows),
                "mean_map_energy": _mean([float(row["map_energy"]) for row in rows if "map_energy" in row]),
                "mean_mixture_nll_per_cell": mean_mix,
                "mean_mask_nll_per_cell": _mean([float(row["mask_nll_per_cell"]) for row in rows if "mask_nll_per_cell" in row]),
                "mean_label_nll_per_fg": _mean([float(row["label_nll_per_fg"]) for row in rows if "label_nll_per_fg" in row]),
                "delta_vs_instruction_mixture_nll_per_cell": float(mean_mix - primary_mean) if name != primary_name else 0.0,
                "instruction_improves": bool(primary_mean < mean_mix) if name != primary_name else None,
            }
        )
    return out


def _mode_usage(rows: list[dict[str, Any]], prior_categories: dict[str, Any], thresholds: dict[str, float]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    csv_rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}
    for category in sorted({str(row["category"]) for row in rows}):
        cat_rows = [row for row in rows if str(row["category"]) == category]
        counts = Counter(int(row["best_mode_id"]) for row in cat_rows)
        mode_count = int(prior_categories.get(category, {}).get("effective_modes", 0)) if isinstance(prior_categories.get(category), dict) else 0
        total = sum(counts.values())
        entropy = _entropy_from_counts(counts)
        max_fraction = max((count / float(total) for count in counts.values()), default=0.0)
        unused = [mode_id for mode_id in range(mode_count) if counts.get(mode_id, 0) == 0]
        used_modes = int(mode_count - len(unused))
        unused_fraction = float(len(unused) / float(max(1, mode_count)))
        normalized_entropy = float(entropy / math.log(mode_count)) if mode_count > 1 else 0.0
        min_usage = float(thresholds.get("min_mode_usage_fraction", 0.01))
        nontrivial_modes = int(sum(1 for count in counts.values() if total > 0 and float(count) / float(total) >= min_usage))
        risks = []
        if max_fraction > float(thresholds.get("max_mode_fraction_warn", 0.8)):
            risks.append("max_mode_fraction_gt_0.8")
        if unused_fraction > float(thresholds.get("max_unused_fraction_warn", 0.5)):
            risks.append("unused_fraction_gt_warn")
        if used_modes < math.ceil(0.5 * float(mode_count)):
            risks.append("used_modes_lt_half_k")
        suggested_count = max(1, nontrivial_modes or used_modes)
        suggested_action = "inspect" if risks else "keep"
        if risks and suggested_count < mode_count:
            suggested_action = "reduce_modes"
        payload[category] = {
            "support": int(total),
            "mode_count": int(mode_count),
            "used_modes": int(used_modes),
            "counts": {str(key): int(value) for key, value in sorted(counts.items())},
            "mode_usage_entropy": float(entropy),
            "normalized_mode_usage_entropy": float(normalized_entropy),
            "max_mode_fraction": float(max_fraction),
            "unused_modes": unused,
            "unused_fraction": float(unused_fraction),
            "risk_flags": risks,
            "risks": risks,
            "nontrivial_modes": int(nontrivial_modes),
            "suggested_mode_count": int(suggested_count),
            "suggested_action": suggested_action,
        }
        csv_rows.append(
            {
                "category": category,
                "support": int(total),
                "mode_count": int(mode_count),
                "used_modes": int(used_modes),
                "mode_usage_entropy": float(entropy),
                "normalized_mode_usage_entropy": float(normalized_entropy),
                "max_mode_fraction": float(max_fraction),
                "unused_fraction": float(unused_fraction),
                "unused_modes": " ".join(str(value) for value in unused),
                "risk_flags": " ".join(risks),
                "suggested_mode_count": int(suggested_count),
                "suggested_action": suggested_action,
                "counts_json": json.dumps(payload[category]["counts"], ensure_ascii=False),
            }
        )
    return csv_rows, payload


def _write_markdown_table(path: Path, title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = [f"# {title}", ""]
    if rows:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join("---" for _ in columns) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(_json_safe(row.get(column, ""))) for column in columns) + " |")
    else:
        lines.append("No rows.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _score_variant(
    y20: list[list[int]],
    mask20: list[list[int]],
    entry: dict[str, Any],
    *,
    category: str,
    variant_name: str,
    use_calibrated_label_prob: bool,
    use_smoothed_mode_prior: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    fg_modes, label_modes, priors = _validate_modes(
        entry,
        category,
        "instruction_matrix_grammar_prior",
        use_calibrated_label_prob=use_calibrated_label_prob,
        use_smoothed_mode_prior=use_smoothed_mode_prior,
    )
    score = _score_modes(
        y20,
        mask20,
        fg_modes,
        label_modes,
        priors,
        category=category,
        source_name=variant_name,
        eps=float(args.eps),
        label_smoothing=float(args.label_smoothing),
        mask_smoothing=float(args.mask_smoothing),
        w_mask=float(args.w_mask),
        w_label=float(args.w_label),
    )
    return {**score, "fg_modes": fg_modes, "label_modes": label_modes, "priors": priors}


def _parse_category_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _candidate_values(cleanup_cf: dict[str, Any], section: str, category: str, default: list[float | int]) -> list[float | int]:
    section_cf = cleanup_cf.get(section, {}) if isinstance(cleanup_cf.get(section, {}), dict) else {}
    if not bool(section_cf.get("enabled", False)):
        return default
    by_category = section_cf.get("candidates_by_category", {}) if isinstance(section_cf.get("candidates_by_category", {}), dict) else {}
    values = by_category.get(category, default)
    if not isinstance(values, list) or not values:
        return default
    return [value for value in values if value is not None]


def _entry_alpha(entry: dict[str, Any]) -> float:
    return _safe_float(entry.get("label_hist_interpolation_alpha"), 0.0)


def _entry_beta(entry: dict[str, Any]) -> float:
    return _safe_float(entry.get("mode_prior_smoothing_beta"), 0.0)


def _calibrate_label_modes_list(label_modes: list[Any], category_hist: list[float], alpha: float) -> list[list[list[list[float]]]]:
    alpha = min(1.0, max(0.0, float(alpha)))
    hist = [max(0.0, float(value)) for value in category_hist[:NUM_LABELS]]
    if len(hist) != NUM_LABELS or sum(hist) <= 0.0:
        hist = [1.0 / float(NUM_LABELS) for _ in range(NUM_LABELS)]
    else:
        total = sum(hist)
        hist = [value / total for value in hist]
    calibrated: list[list[list[list[float]]]] = []
    for mode_index, raw_mode in enumerate(label_modes):
        tensor = _mode_label_tensor(raw_mode, category="sweep", mode_index=mode_index, source_name="calibration_sweep")
        out_mode = [[[0.0 for _ in range(CANONICAL_SIZE)] for _ in range(CANONICAL_SIZE)] for _ in range(NUM_LABELS)]
        for y_pos in range(CANONICAL_SIZE):
            for x_pos in range(CANONICAL_SIZE):
                denom = 0.0
                for label_index in range(NUM_LABELS):
                    value = (1.0 - alpha) * float(tensor[label_index][y_pos][x_pos]) + alpha * float(hist[label_index])
                    out_mode[label_index][y_pos][x_pos] = value
                    denom += value
                if denom <= 0.0:
                    for label_index in range(NUM_LABELS):
                        out_mode[label_index][y_pos][x_pos] = 1.0 / float(NUM_LABELS)
                else:
                    for label_index in range(NUM_LABELS):
                        out_mode[label_index][y_pos][x_pos] = float(out_mode[label_index][y_pos][x_pos]) / denom
        calibrated.append(out_mode)
    return calibrated


def _subset_by_indices(values: list[Any], indices: list[int]) -> list[Any]:
    return [values[index] for index in indices if 0 <= int(index) < len(values)]


def _subset_priors_from_counts(entry: dict[str, Any], subset: list[int], beta: float) -> list[float]:
    counts_raw = _to_list(entry.get("mode_num_samples", []))
    mode_count = int(entry.get("effective_modes", len(counts_raw) if isinstance(counts_raw, list) else 0))
    if not isinstance(counts_raw, list) or len(counts_raw) != mode_count:
        return [1.0 / float(max(1, len(subset))) for _ in subset]
    counts = [max(0.0, _safe_float(counts_raw[index])) for index in subset if 0 <= int(index) < len(counts_raw)]
    if not counts:
        return []
    beta = max(0.0, float(beta))
    denom = sum(counts) + beta * float(len(counts))
    if denom <= 0.0:
        return [1.0 / float(len(counts)) for _ in counts]
    priors = [(count + beta) / denom for count in counts]
    total = sum(priors)
    return [float(value) / total for value in priors] if total > 0.0 else [1.0 / float(len(counts)) for _ in counts]


def _top_used_subset(category_rows: list[dict[str, Any]], mode_count: int, reduced_count: int) -> list[int]:
    counts = Counter(int(row["best_mode_id"]) for row in category_rows)
    ranked = sorted(range(mode_count), key=lambda mode_id: (-int(counts.get(mode_id, 0)), mode_id))
    keep = max(1, min(int(reduced_count), mode_count))
    return sorted(ranked[:keep])


def _usage_stats_from_mode_ids(mode_ids: list[int], mode_count: int, cleanup_cf: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(int(mode_id) for mode_id in mode_ids)
    total = sum(counts.values())
    entropy = _entropy_from_counts(counts)
    max_fraction = max((count / float(total) for count in counts.values()), default=0.0)
    unused = [mode_id for mode_id in range(mode_count) if counts.get(mode_id, 0) == 0]
    used_modes = int(mode_count - len(unused))
    unused_fraction = float(len(unused) / float(max(1, mode_count)))
    normalized_entropy = float(entropy / math.log(mode_count)) if mode_count > 1 else 0.0
    max_mode_fraction_target = float(cleanup_cf.get("max_mode_fraction_target", 0.8))
    min_used_mode_fraction_target = float(cleanup_cf.get("min_used_mode_fraction_target", 0.5))
    used_mode_fraction = float(used_modes / float(max(1, mode_count)))
    risk_flags: list[str] = []
    if max_fraction > max_mode_fraction_target:
        risk_flags.append("max_mode_fraction_gt_0.8")
    if used_mode_fraction < min_used_mode_fraction_target:
        risk_flags.append("used_modes_lt_half_k")
    return {
        "support": int(total),
        "mode_count": int(mode_count),
        "used_modes": int(used_modes),
        "used_mode_fraction": float(used_mode_fraction),
        "unused_modes": unused,
        "unused_fraction": float(unused_fraction),
        "max_mode_fraction": float(max_fraction),
        "mode_usage_entropy": float(entropy),
        "normalized_mode_usage_entropy": float(normalized_entropy),
        "risk_flags": risk_flags,
        "counts": {str(key): int(value) for key, value in sorted(counts.items())},
    }


def _posterior_assignment(score: dict[str, Any]) -> int:
    posterior = score.get("mode_posterior", [])
    if isinstance(posterior, list) and posterior:
        return int(max(range(len(posterior)), key=lambda index: float(posterior[index])))
    return int(score.get("best_mode_id", 0))


def _candidate_summary(
    *,
    category: str,
    alpha: float,
    beta: float,
    mode_count: int,
    mode_subset: list[int],
    sample_scores: list[dict[str, Any]],
    cleanup_cf: dict[str, Any],
    baseline_mix: float,
    baseline_label: float,
) -> dict[str, Any]:
    mode_ids = [_posterior_assignment(row) for row in sample_scores]
    usage = _usage_stats_from_mode_ids(mode_ids, mode_count, cleanup_cf)
    mix = _mean([float(row["mixture_nll_per_cell"]) for row in sample_scores])
    label = _mean([float(row["label_nll_per_fg"]) for row in sample_scores])
    mask = _mean([float(row["mask_nll_per_cell"]) for row in sample_scores])
    used_mode_fraction = float(usage["used_mode_fraction"])
    max_mode_fraction = float(usage["max_mode_fraction"])
    passes_likelihood = bool(mix <= baseline_mix + 0.005 and label <= baseline_label + 0.01)
    passes_usage = bool(not usage["risk_flags"])
    score = float(mix + 0.05 * max(0.0, max_mode_fraction - 0.8) + 0.05 * max(0.0, 0.5 - used_mode_fraction))
    return {
        "category": category,
        "alpha": float(alpha),
        "beta": float(beta),
        "mode_count": int(mode_count),
        "mode_subset": " ".join(str(value) for value in mode_subset),
        "mixture_nll_per_cell": float(mix),
        "mask_nll_per_cell": float(mask),
        "label_nll_per_fg": float(label),
        "max_mode_fraction": max_mode_fraction,
        "used_modes": int(usage["used_modes"]),
        "used_mode_fraction": used_mode_fraction,
        "unused_fraction": float(usage["unused_fraction"]),
        "mode_usage_entropy": float(usage["mode_usage_entropy"]),
        "normalized_mode_usage_entropy": float(usage["normalized_mode_usage_entropy"]),
        "risk_flags": " ".join(str(flag) for flag in usage["risk_flags"]),
        "passes_likelihood_gate": passes_likelihood,
        "passes_usage_gate": passes_usage,
        "score": score,
        "assignment_rule": "posterior_argmax",
        "counts_json": json.dumps(usage["counts"], ensure_ascii=False),
        "mode_subset_list": mode_subset,
    }


def _classify_category_decision(category: str, selected: dict[str, Any] | None, before: dict[str, Any] | None, cleanup_cf: dict[str, Any]) -> dict[str, Any]:
    if selected is None:
        return {"decision": "needs_inspect", "reason": "no_candidate_selected"}
    if not bool(selected.get("passes_likelihood_gate", False)):
        return {"decision": "needs_inspect", "reason": "no_candidate_passes_likelihood_gate"}
    allow_unimodal = cleanup_cf.get("allow_justified_unimodal_by_category", {})
    allow_unimodal = allow_unimodal if isinstance(allow_unimodal, dict) else {}
    mode_count = int(selected.get("mode_count", 0))
    before_mix = float(before.get("mixture_nll_per_cell", LARGE_PENALTY)) if isinstance(before, dict) else LARGE_PENALTY
    selected_mix = float(selected.get("mixture_nll_per_cell", LARGE_PENALTY))
    selected_label = float(selected.get("label_nll_per_fg", LARGE_PENALTY))
    before_label = float(before.get("label_nll_per_fg", LARGE_PENALTY)) if isinstance(before, dict) else LARGE_PENALTY
    likelihood_stable = bool(selected_mix <= before_mix + 0.005 and selected_label <= before_label + 0.01)
    usage_ok = bool(selected.get("passes_usage_gate", False))
    if category == "Links1":
        if bool(allow_unimodal.get(category, False)) and mode_count <= 2 and likelihood_stable:
            return {
                "decision": "justify_unimodal",
                "justified_low_mode_diversity": True,
                "reason": "reduced_mode_count_1_or_2_preserves_likelihood",
            }
        if usage_ok:
            return {"decision": "keep_modes" if mode_count >= int(before.get("mode_count", mode_count)) else "reduce_modes", "reason": "usage_gate_passes"}
        if selected_mix > before_mix + 0.005:
            return {"decision": "needs_inspect", "reason": "reduced_modes_worsen_mixture_nll"}
        return {"decision": "keep_modes", "reason": "dominant_mode_requires_beta_or_smoothing"}
    if category in {"Cable1", "Tuck"}:
        before_count = int(before.get("mode_count", mode_count)) if isinstance(before, dict) else mode_count
        if mode_count < before_count and likelihood_stable:
            return {"decision": "reduce_modes", "reason": "reduced_mode_count_preserves_likelihood"}
        if not usage_ok and likelihood_stable:
            return {"decision": "keep_modes", "reason": "modes_are_rare_but_useful"}
        if usage_ok:
            return {"decision": "keep_modes" if mode_count == before_count else "reduce_modes", "reason": "usage_gate_passes"}
    return {"decision": "keep_modes" if usage_ok else "needs_inspect", "reason": "joint_likelihood_usage_selection"}


def _format_yaml_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_recommended_config_patch(path: Path, recommendations: dict[str, Any]) -> None:
    lines = [
        "# Diagnostic recommendation only. Do not apply without explicit review.",
        "instruction_matrix_grammar_prior:",
        "  mode_count_by_category:",
    ]
    mode_updates = {
        category: payload.get("mode_count")
        for category, payload in sorted(recommendations.items())
        if bool(payload.get("passes_likelihood_gate", False))
        and payload.get("recommended_mode_count") is not None
        and payload.get("decision") in {"reduce_modes", "justify_unimodal"}
    }
    if mode_updates:
        for category, value in mode_updates.items():
            lines.append(f"    {category}: {_format_yaml_value(value)}")
    else:
        lines.append("    {}")
    lines.extend(
        [
            "  calibration:",
            "    label_hist_interpolation:",
            "      alpha_by_category:",
        ]
    )
    alpha_updates = {
        category: payload.get("alpha")
        for category, payload in sorted(recommendations.items())
        if bool(payload.get("passes_likelihood_gate", False)) and payload.get("alpha") is not None and payload.get("decision") != "needs_inspect"
    }
    if alpha_updates:
        for category, value in alpha_updates.items():
            lines.append(f"        {category}: {_format_yaml_value(value)}")
    else:
        lines.append("        {}")
    lines.extend(["    mode_prior_smoothing:", "      beta_by_category:"])
    beta_updates = {
        category: payload.get("beta")
        for category, payload in sorted(recommendations.items())
        if bool(payload.get("passes_likelihood_gate", False)) and payload.get("beta") is not None and payload.get("decision") != "needs_inspect"
    }
    if beta_updates:
        for category, value in beta_updates.items():
            lines.append(f"        {category}: {_format_yaml_value(value)}")
    else:
        lines.append("        {}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_calibration_sweep(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    categories: dict[str, Any],
    eval_samples: list[dict[str, Any]],
    primary_rows: list[dict[str, Any]],
    mode_usage_payload: dict[str, Any],
    primary_summary: dict[str, Any],
) -> dict[str, Any]:
    cleanup_cf = _mode_usage_cleanup_config(config)
    requested = _parse_category_list(str(args.sweep_categories))
    if not requested:
        risk_categories = cleanup_cf.get("risk_categories", {}) if isinstance(cleanup_cf.get("risk_categories", {}), dict) else {}
        requested = [category for category, enabled in risk_categories.items() if bool(enabled)]
    requested = [category for category in requested if category in categories]
    results: list[dict[str, Any]] = []
    recommendations: dict[str, Any] = {}
    before_after_rows: list[dict[str, Any]] = []
    val_tuned_risk = str(args.split_name).lower() == "val"
    for category in requested:
        entry = categories.get(category)
        if not isinstance(entry, dict):
            continue
        cat_samples = [sample for sample in eval_samples if str(sample["category"]) == category]
        cat_primary_rows = [row for row in primary_rows if str(row["category"]) == category]
        if not cat_samples or not cat_primary_rows:
            continue
        raw_fg_modes, raw_label_modes, _ = _validate_modes(
            entry,
            category,
            "instruction_matrix_grammar_prior",
            use_calibrated_label_prob=False,
            use_smoothed_mode_prior=False,
        )
        mode_count_full = len(raw_fg_modes)
        category_hist = _to_list(entry.get("category_label_hist_16", []))
        if not isinstance(category_hist, list) or len(category_hist) != NUM_LABELS:
            category_hist = [1.0 / float(NUM_LABELS) for _ in range(NUM_LABELS)]
        current_alpha = _entry_alpha(entry)
        current_beta = _entry_beta(entry)
        alpha_values = [float(value) for value in _candidate_values(cleanup_cf, "alpha_sweep", category, [current_alpha])]
        beta_values = [float(value) for value in _candidate_values(cleanup_cf, "beta_sweep", category, [current_beta])]
        mode_values = [max(1, min(mode_count_full, int(value))) for value in _candidate_values(cleanup_cf, "mode_count_sweep", category, [mode_count_full])]
        if mode_count_full not in mode_values:
            mode_values.append(mode_count_full)
        mode_values = sorted(set(mode_values))
        cat_before = {
            "category": category,
            "stage": "before",
            "alpha": current_alpha,
            "beta": current_beta,
            "mode_count": mode_count_full,
            "mode_subset": " ".join(str(index) for index in range(mode_count_full)),
            "mixture_nll_per_cell": _mean([float(row["mixture_nll_per_cell"]) for row in cat_primary_rows]),
            "mask_nll_per_cell": _mean([float(row["mask_nll_per_cell"]) for row in cat_primary_rows]),
            "label_nll_per_fg": _mean([float(row["label_nll_per_fg"]) for row in cat_primary_rows]),
            "max_mode_fraction": float(mode_usage_payload.get(category, {}).get("max_mode_fraction", 0.0)),
            "used_modes": int(mode_usage_payload.get(category, {}).get("used_modes", 0)),
            "unused_fraction": float(mode_usage_payload.get(category, {}).get("unused_fraction", 0.0)),
            "risk_flags": " ".join(mode_usage_payload.get(category, {}).get("risk_flags", [])),
        }
        before_after_rows.append(cat_before)
        for alpha in sorted(set(alpha_values)):
            label_modes = _calibrate_label_modes_list(raw_label_modes, category_hist, alpha)
            for beta in sorted(set(beta_values)):
                for candidate_count in sorted(set(mode_values)):
                    subset = _top_used_subset(cat_primary_rows, mode_count_full, candidate_count)
                    if not subset:
                        continue
                    fg_subset = _subset_by_indices(raw_fg_modes, subset)
                    label_subset = _subset_by_indices(label_modes, subset)
                    priors = _subset_priors_from_counts(entry, subset, beta)
                    sample_scores: list[dict[str, Any]] = []
                    for sample in cat_samples:
                        score = _score_modes(
                            sample["y20"],
                            sample["mask20"],
                            fg_subset,
                            label_subset,
                            priors,
                            category=category,
                            source_name="calibration_sweep",
                            eps=float(args.eps),
                            label_smoothing=float(args.label_smoothing),
                            mask_smoothing=float(args.mask_smoothing),
                            w_mask=float(args.w_mask),
                            w_label=float(args.w_label),
                        )
                        sample_scores.append(score)
                    candidate = _candidate_summary(
                        category=category,
                        alpha=alpha,
                        beta=beta,
                        mode_count=len(subset),
                        mode_subset=subset,
                        sample_scores=sample_scores,
                        cleanup_cf=cleanup_cf,
                        baseline_mix=float(cat_before["mixture_nll_per_cell"]),
                        baseline_label=float(cat_before["label_nll_per_fg"]),
                    )
                    results.append(candidate)
        cat_results = [row for row in results if str(row["category"]) == category]
        feasible = [row for row in cat_results if bool(row["passes_likelihood_gate"])]
        usage_feasible = [row for row in feasible if bool(row["passes_usage_gate"])]
        pool = usage_feasible or feasible or cat_results
        selected = min(pool, key=lambda row: (float(row["score"]), float(row["mixture_nll_per_cell"]), float(row["label_nll_per_fg"]))) if pool else None
        decision = _classify_category_decision(category, selected, cat_before, cleanup_cf)
        if selected is not None:
            rec = {
                **{key: _json_safe(value) for key, value in selected.items() if key != "mode_subset_list"},
                **decision,
                "recommended_mode_count": int(selected["mode_count"]) if int(selected["mode_count"]) != mode_count_full else None,
                "val_tuned_risk": bool(val_tuned_risk),
            }
            recommendations[category] = rec
            before_after_rows.append(
                {
                    "category": category,
                    "stage": "after_recommended",
                    "alpha": float(selected["alpha"]),
                    "beta": float(selected["beta"]),
                    "mode_count": int(selected["mode_count"]),
                    "mode_subset": str(selected["mode_subset"]),
                    "mixture_nll_per_cell": float(selected["mixture_nll_per_cell"]),
                    "mask_nll_per_cell": float(selected["mask_nll_per_cell"]),
                    "label_nll_per_fg": float(selected["label_nll_per_fg"]),
                    "max_mode_fraction": float(selected["max_mode_fraction"]),
                    "used_modes": int(selected["used_modes"]),
                    "unused_fraction": float(selected["unused_fraction"]),
                    "risk_flags": str(selected["risk_flags"]),
                }
            )
    fieldnames = [
        "category",
        "alpha",
        "beta",
        "mode_count",
        "mode_subset",
        "mixture_nll_per_cell",
        "mask_nll_per_cell",
        "label_nll_per_fg",
        "max_mode_fraction",
        "used_modes",
        "unused_fraction",
        "mode_usage_entropy",
        "normalized_mode_usage_entropy",
        "risk_flags",
        "passes_likelihood_gate",
        "passes_usage_gate",
        "score",
        "assignment_rule",
        "counts_json",
    ]
    _write_csv(args.output_dir / "calibration_sweep_results.csv", [{key: row.get(key, "") for key in fieldnames} for row in results], fieldnames)
    _write_csv(
        args.output_dir / "mode_usage_before_after.csv",
        before_after_rows,
        ["category", "stage", "alpha", "beta", "mode_count", "mode_subset", "mixture_nll_per_cell", "mask_nll_per_cell", "label_nll_per_fg", "max_mode_fraction", "used_modes", "unused_fraction", "risk_flags"],
    )
    selected_by_category = {category: payload for category, payload in recommendations.items() if payload.get("decision") != "needs_inspect" and bool(payload.get("passes_likelihood_gate", False))}
    total_support = max(1, len(primary_rows))
    replaced_categories = set(selected_by_category)
    remaining_rows = [row for row in primary_rows if str(row["category"]) not in replaced_categories]
    estimated_mix_sum = sum(float(row["mixture_nll_per_cell"]) for row in remaining_rows)
    estimated_mask_sum = sum(float(row["mask_nll_per_cell"]) for row in remaining_rows)
    estimated_label_sum = sum(float(row["label_nll_per_fg"]) for row in remaining_rows)
    for category, payload in selected_by_category.items():
        support = len([row for row in primary_rows if str(row["category"]) == category])
        estimated_mix_sum += float(payload.get("mixture_nll_per_cell", 0.0)) * support
        estimated_mask_sum += float(payload.get("mask_nll_per_cell", 0.0)) * support
        estimated_label_sum += float(payload.get("label_nll_per_fg", 0.0)) * support
    estimated_global = {
        "mean_mixture_nll_per_cell": float(estimated_mix_sum / float(total_support)),
        "mean_mask_nll_per_cell": float(estimated_mask_sum / float(total_support)),
        "mean_label_nll_per_fg": float(estimated_label_sum / float(total_support)),
        "passes_current_run_likelihood_gate": bool(
            estimated_mix_sum / float(total_support) <= float(primary_summary.get("mean_mixture_nll_per_cell", LARGE_PENALTY)) + 0.005
            and estimated_label_sum / float(total_support) <= float(primary_summary.get("mean_label_nll_per_fg", LARGE_PENALTY)) + 0.01
        ),
    }
    sweep_summary = {
        "split_name": str(args.split_name),
        "diagnostic_only": bool(cleanup_cf.get("diagnostic_only", True)),
        "val_tuned_risk": bool(val_tuned_risk),
        "categories": requested,
        "num_candidates": int(len(results)),
        "global_current": {
            "mean_mixture_nll_per_cell": float(primary_summary.get("mean_mixture_nll_per_cell", 0.0)),
            "mean_label_nll_per_fg": float(primary_summary.get("mean_label_nll_per_fg", 0.0)),
        },
        "estimated_global_after_recommended": estimated_global,
        "recommendations": recommendations,
        "decisions": {category: {"decision": payload.get("decision"), "reason": payload.get("reason"), "justified_low_mode_diversity": payload.get("justified_low_mode_diversity", False)} for category, payload in recommendations.items()},
    }
    save_json(args.output_dir / "calibration_sweep_summary.json", _json_safe(sweep_summary))
    save_json(args.output_dir / "recommended_calibration_by_category.json", _json_safe(recommendations))
    if bool(args.write_recommended_config):
        _write_recommended_config_patch(args.output_dir / "recommended_config_patch.yaml", recommendations)
    lines = [
        "# Calibration Sweep Summary",
        "",
        f"- split_name: {args.split_name}",
        f"- diagnostic_only: {bool(cleanup_cf.get('diagnostic_only', True))}",
        f"- val_tuned_risk: {val_tuned_risk}",
        f"- num_candidates: {len(results)}",
        f"- estimated_global_after_recommended_mixture: {estimated_global['mean_mixture_nll_per_cell']:.6f}",
        f"- estimated_global_after_recommended_label: {estimated_global['mean_label_nll_per_fg']:.6f}",
        f"- estimated_global_passes_current_run_likelihood_gate: {estimated_global['passes_current_run_likelihood_gate']}",
        "",
        "## Decisions",
        "",
    ]
    for category in requested:
        payload = recommendations.get(category, {})
        lines.append(
            "- {category}: decision={decision} reason={reason} alpha={alpha} beta={beta} mode_count={mode_count} mix={mix} label={label} risks={risks}".format(
                category=category,
                decision=payload.get("decision", "none"),
                reason=payload.get("reason", "none"),
                alpha=payload.get("alpha", "n/a"),
                beta=payload.get("beta", "n/a"),
                mode_count=payload.get("mode_count", "n/a"),
                mix=payload.get("mixture_nll_per_cell", "n/a"),
                label=payload.get("label_nll_per_fg", "n/a"),
                risks=payload.get("risk_flags", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Candidates are post-hoc diagnostics over the cached grammar prior; they do not rebuild cache or alter the main config.",
            "- Likelihood gates use current per-category likelihood plus the requested tolerances before applying mode-usage scoring.",
            "- If split_name is val, recommendations are marked val_tuned_risk and should be verified on a separate split before treating them as final.",
        ]
    )
    (args.output_dir / "calibration_sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sweep_summary


def _comparison_summary(rows: list[dict[str, Any]], prior_categories: dict[str, Any], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for variant in sorted({str(row["variant"]) for row in rows}):
        variant_rows = [row for row in rows if str(row["variant"]) == variant]
        _, usage_payload = _mode_usage(variant_rows, prior_categories, thresholds)
        risk_categories = [category for category, payload in usage_payload.items() if payload.get("risk_flags")]
        out.append(
            {
                "variant": variant,
                "support": len(variant_rows),
                "mean_mixture_nll_per_cell": _mean([float(row["mixture_nll_per_cell"]) for row in variant_rows]),
                "mean_mask_nll_per_cell": _mean([float(row["mask_nll_per_cell"]) for row in variant_rows]),
                "mean_label_nll_per_fg": _mean([float(row["label_nll_per_fg"]) for row in variant_rows]),
                "mean_true_token_probability": _mean([float(row["mean_true_token_probability"]) for row in variant_rows]),
                "delta_label_nll_vs_raw_raw": 0.0,
                "delta_mixture_nll_vs_raw_raw": 0.0,
                "calibrated_label_nll_improves_vs_raw_raw": None,
                "calibrated_mixture_nll_improves_vs_raw_raw": None,
                "mode_usage_risk_categories": " ".join(risk_categories),
            }
        )
    lookup = {str(row["variant"]): row for row in out}
    raw = lookup.get("raw_mode_prior_raw_label_prob")
    calibrated = lookup.get("smoothed_prior_calibrated_label_prob")
    if raw is not None and calibrated is not None:
        for row in out:
            row["delta_label_nll_vs_raw_raw"] = float(row["mean_label_nll_per_fg"]) - float(raw["mean_label_nll_per_fg"])
            row["delta_mixture_nll_vs_raw_raw"] = float(row["mean_mixture_nll_per_cell"]) - float(raw["mean_mixture_nll_per_cell"])
        calibrated["calibrated_label_nll_improves_vs_raw_raw"] = bool(float(calibrated["mean_label_nll_per_fg"]) < float(raw["mean_label_nll_per_fg"]))
        calibrated["calibrated_mixture_nll_improves_vs_raw_raw"] = bool(float(calibrated["mean_mixture_nll_per_cell"]) < float(raw["mean_mixture_nll_per_cell"]))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data_cf = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    if args.use_calibrated_label_prob is None:
        args.use_calibrated_label_prob = _config_bool(config, "use_calibrated_label_prob", True)
    if args.use_smoothed_mode_prior is None:
        args.use_smoothed_mode_prior = _config_bool(config, "use_smoothed_mode_prior", True)
    mode_thresholds = _mode_pruning_thresholds(config)
    torch = _require_torch()
    cache = torch.load(args.cache, map_location="cpu")
    if not isinstance(cache, dict):
        raise ValueError(f"Cache payload must be a dict: {args.cache}")
    prior = cache.get("instruction_matrix_grammar_prior")
    if not isinstance(prior, dict):
        raise ValueError("Cache does not contain instruction_matrix_grammar_prior.")
    if prior.get("schema_version") != INSTRUCTION_GRAMMAR_SCHEMA_VERSION:
        raise ValueError(f"instruction_matrix_grammar_prior schema={prior.get('schema_version')!r}; expected {INSTRUCTION_GRAMMAR_SCHEMA_VERSION!r}.")
    categories = prior.get("categories", {})
    if not isinstance(categories, dict):
        raise ValueError("instruction_matrix_grammar_prior.categories is missing or invalid.")
    dictionary_bank = cache.get("dictionary_bank") if isinstance(cache.get("dictionary_bank"), dict) else None
    dictionary_categories = dictionary_bank.get("categories", {}) if isinstance(dictionary_bank, dict) and isinstance(dictionary_bank.get("categories", {}), dict) else {}

    rows = _load_manifest(args.manifest, args.max_samples)
    manifest_root = _infer_manifest_root(args.manifest, rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_stats = cache.get("category_foreground_area_stats", {}) if isinstance(cache.get("category_foreground_area_stats", {}), dict) else {}
    global_fg_values = [_safe_float(value.get("mean")) for value in global_stats.values() if isinstance(value, dict) and "mean" in value]
    global_fg_rate = _mean(global_fg_values) if global_fg_values else 0.5

    per_sample_rows: list[dict[str, Any]] = []
    baseline_sample_rows: list[dict[str, Any]] = []
    comparison_sample_rows: list[dict[str, Any]] = []
    eval_samples: list[dict[str, Any]] = []
    skipped_unknown_category = 0
    skipped_no_modes = 0
    empty_foreground_count = 0
    nan_or_inf_count = 0
    calibration_counts: dict[int, dict[str, float]] = {index: {"count": 0.0, "pred_sum": 0.0, "fg_sum": 0.0} for index in range(10)}
    label_true_probs: list[float] = []
    baseline_warnings: list[str] = []

    for index, row in enumerate(rows, start=1):
        if index == 1 or index % 50 == 0 or index == len(rows):
            print(f"[eval-igr-likelihood] {index}/{len(rows)}", flush=True)
        sample_id = str(row.get("sample_id", row.get("id", f"row_{index}")))
        category = str(row.get("category", ""))
        if category not in categories:
            skipped_unknown_category += 1
            continue
        entry = categories[category]
        if not isinstance(entry, dict):
            skipped_no_modes += 1
            continue
        try:
            fg_modes, label_modes, priors = _validate_modes(
                entry,
                category,
                "instruction_matrix_grammar_prior",
                use_calibrated_label_prob=bool(args.use_calibrated_label_prob),
                use_smoothed_mode_prior=bool(args.use_smoothed_mode_prior),
            )
        except Exception as error:
            skipped_no_modes += 1
            print(f"WARNING skip sample_id={sample_id} category={category}: {error}", flush=True)
            continue
        y20, mask20, is_empty = _load_y_for_row(row, manifest_root, data_cf)
        if is_empty:
            empty_foreground_count += 1
        eval_samples.append({"sample_id": sample_id, "category": category, "y20": y20, "mask20": mask20})
        primary = _score_variant(
            y20,
            mask20,
            entry,
            category=category,
            variant_name="instruction_matrix_grammar_prior_calibrated" if bool(args.use_calibrated_label_prob) else "instruction_matrix_grammar_prior_raw",
            use_calibrated_label_prob=bool(args.use_calibrated_label_prob),
            use_smoothed_mode_prior=bool(args.use_smoothed_mode_prior),
            args=args,
        )
        fg_modes = primary["fg_modes"]
        label_modes = primary["label_modes"]
        priors = primary["priors"]
        nan_or_inf_count += int(primary.get("nan_or_inf_count", 0))
        best_mode_id = int(primary["best_mode_id"])
        cal_rows, true_probs = _calibration_rows(
            y20,
            mask20,
            fg_modes[best_mode_id],
            label_modes[best_mode_id],
            category=category,
            mode_index=best_mode_id,
            eps=float(args.eps),
            label_smoothing=float(args.label_smoothing),
            mask_smoothing=float(args.mask_smoothing),
        )
        for bin_index, pred, fg in cal_rows:
            calibration_counts[bin_index]["count"] += 1.0
            calibration_counts[bin_index]["pred_sum"] += float(pred)
            calibration_counts[bin_index]["fg_sum"] += float(fg)
        label_true_probs.extend(true_probs)
        num_fg = sum(int(value) for row_mask in mask20 for value in row_mask)
        sample_row = {
            "sample_id": sample_id,
            "category": category,
            "best_mode_id": best_mode_id,
            "map_energy": float(primary["map_energy"]),
            "mixture_nll_per_cell": float(primary["mixture_nll_per_cell"]),
            "mask_nll_per_cell": float(primary["mask_nll_per_cell"]),
            "label_nll_per_fg": float(primary["label_nll_per_fg"]),
            "num_foreground": int(num_fg),
            "label_diversity": int(label_diversity_on_fg(y20, mask20)),
            "mean_true_token_probability": float(primary["mean_true_token_probability"]),
            "mode_posterior": [float(value) for value in primary.get("mode_posterior", [])],
        }
        per_sample_rows.append(sample_row)
        if args.compare_raw_calibrated:
            variants = [
                ("raw_mode_prior_raw_label_prob", False, False),
                ("smoothed_prior_raw_label_prob", False, True),
                ("raw_mode_prior_calibrated_label_prob", True, False),
                ("smoothed_prior_calibrated_label_prob", True, True),
            ]
            for variant_name, use_cal_label, use_smooth_prior in variants:
                variant = _score_variant(
                    y20,
                    mask20,
                    entry,
                    category=category,
                    variant_name=variant_name,
                    use_calibrated_label_prob=use_cal_label,
                    use_smoothed_mode_prior=use_smooth_prior,
                    args=args,
                )
                comparison_sample_rows.append(
                    {
                        "sample_id": sample_id,
                        "category": category,
                        "variant": variant_name,
                        "best_mode_id": int(variant["best_mode_id"]),
                        "map_energy": float(variant["map_energy"]),
                        "mixture_nll_per_cell": float(variant["mixture_nll_per_cell"]),
                        "mask_nll_per_cell": float(variant["mask_nll_per_cell"]),
                        "label_nll_per_fg": float(variant["label_nll_per_fg"]),
                        "mean_true_token_probability": float(variant["mean_true_token_probability"]),
                    }
                )
        if args.baselines:
            raw_fg_modes, raw_label_modes, raw_priors = _validate_modes(
                entry,
                category,
                "instruction_matrix_grammar_prior",
                use_calibrated_label_prob=False,
                use_smoothed_mode_prior=bool(args.use_smoothed_mode_prior),
            )
            cal_fg_modes, calibrated_label_modes, _ = _validate_modes(
                entry,
                category,
                "instruction_matrix_grammar_prior",
                use_calibrated_label_prob=True,
                use_smoothed_mode_prior=bool(args.use_smoothed_mode_prior),
            )
            baselines = _score_baselines(
                y20,
                mask20,
                category=category,
                fg_modes=fg_modes,
                label_modes=label_modes,
                raw_label_modes=raw_label_modes,
                calibrated_label_modes=calibrated_label_modes,
                priors=priors,
                global_fg_rate=global_fg_rate,
                dictionary_entry=dictionary_categories.get(category) if isinstance(dictionary_categories.get(category), dict) else None,
                args=args,
            )
            for baseline_name, baseline in baselines.items():
                if bool(baseline.get("skipped", False)):
                    baseline_warnings.append(f"{sample_id} {category} {baseline_name}: {baseline.get('warning')}")
                    continue
                nan_or_inf_count += int(baseline.get("nan_or_inf_count", 0))
                baseline_sample_rows.append(
                    {
                        "sample_id": sample_id,
                        "category": category,
                        "baseline": baseline_name,
                        "best_mode_id": int(baseline["best_mode_id"]),
                        "map_energy": float(baseline["map_energy"]),
                        "mixture_nll_per_cell": float(baseline["mixture_nll_per_cell"]),
                        "mask_nll_per_cell": float(baseline["mask_nll_per_cell"]),
                        "label_nll_per_fg": float(baseline["label_nll_per_fg"]),
                    }
                )

    per_category_rows = _summarize_by_category(per_sample_rows)
    primary_name = "instruction_matrix_grammar_prior_calibrated" if bool(args.use_calibrated_label_prob) else "instruction_matrix_grammar_prior_raw"
    if comparison_sample_rows:
        for primary_variant in ("raw_mode_prior_raw_label_prob", "smoothed_prior_calibrated_label_prob"):
            for row in comparison_sample_rows:
                if str(row["variant"]) == primary_variant:
                    baseline_name = "instruction_matrix_grammar_prior_calibrated" if primary_variant == "smoothed_prior_calibrated_label_prob" else "instruction_matrix_grammar_prior_raw"
                    baseline_sample_rows.append({**row, "baseline": baseline_name})
    baseline_comparison_rows = _baseline_summary(per_sample_rows, baseline_sample_rows, primary_name)
    mode_usage_rows, mode_usage_payload = _mode_usage(per_sample_rows, categories, mode_thresholds)
    calibration_comparison_rows = _comparison_summary(comparison_sample_rows, categories, mode_thresholds) if comparison_sample_rows else []
    calibration_rows = []
    for bin_index in range(10):
        bucket = calibration_counts[bin_index]
        count = int(bucket["count"])
        calibration_rows.append(
            {
                "bin": f"{bin_index / 10.0:.1f}-{(bin_index + 1) / 10.0:.1f}",
                "count": count,
                "predicted_probability": float(bucket["pred_sum"] / bucket["count"]) if bucket["count"] > 0 else 0.0,
                "empirical_foreground_rate": float(bucket["fg_sum"] / bucket["count"]) if bucket["count"] > 0 else 0.0,
            }
        )
    baseline_lookup = {row["baseline"]: row for row in baseline_comparison_rows}
    primary_mix = _mean([float(row["mixture_nll_per_cell"]) for row in per_sample_rows])
    improves_over_baselines = {
        str(name): bool(primary_mix < float(row.get("mean_mixture_nll_per_cell", LARGE_PENALTY)))
        for name, row in baseline_lookup.items()
        if str(name) != primary_name and not str(name).startswith("mode_label_only")
    }
    summary = {
        "split_name": str(args.split_name),
        "rows_read": int(len(rows)),
        "evaluated": int(len(per_sample_rows)),
        "skipped_unknown_category": int(skipped_unknown_category),
        "skipped_no_modes": int(skipped_no_modes),
        "empty_foreground_count": int(empty_foreground_count),
        "nan_or_inf_count": int(nan_or_inf_count),
        "mean_map_energy": _mean([float(row["map_energy"]) for row in per_sample_rows]),
        "mean_mixture_nll_per_cell": primary_mix,
        "mean_mask_nll_per_cell": _mean([float(row["mask_nll_per_cell"]) for row in per_sample_rows]),
        "mean_label_nll_per_fg": _mean([float(row["label_nll_per_fg"]) for row in per_sample_rows]),
        "mean_true_token_probability": _mean(label_true_probs),
        "per_category_means": per_category_rows,
        "baseline_comparison": baseline_comparison_rows,
        "calibration_comparison": calibration_comparison_rows,
        "improves_over_baselines": improves_over_baselines,
        "instruction_prior_improves_over_all_available_baselines": bool(improves_over_baselines and all(improves_over_baselines.values())),
        "mode_usage_risk_categories": {category: payload["risk_flags"] for category, payload in mode_usage_payload.items() if payload.get("risk_flags")},
        "suggested_mode_count_by_category": {category: int(payload["suggested_mode_count"]) for category, payload in mode_usage_payload.items()},
        "baseline_warnings": baseline_warnings[:100],
        "parameters": {
            "eps": float(args.eps),
            "label_smoothing": float(args.label_smoothing),
            "mask_smoothing": float(args.mask_smoothing),
            "w_mask": float(args.w_mask),
            "w_label": float(args.w_label),
            "use_calibrated_label_prob": bool(args.use_calibrated_label_prob),
            "use_smoothed_mode_prior": bool(args.use_smoothed_mode_prior),
        },
    }
    if args.sweep_calibration:
        sweep_summary = _run_calibration_sweep(
            args=args,
            config=config,
            categories=categories,
            eval_samples=eval_samples,
            primary_rows=per_sample_rows,
            mode_usage_payload=mode_usage_payload,
            primary_summary=summary,
        )
        summary["calibration_sweep"] = {
            "enabled": True,
            "summary_path": str(args.output_dir / "calibration_sweep_summary.json"),
            "recommended_config_patch_path": str(args.output_dir / "recommended_config_patch.yaml") if bool(args.write_recommended_config) else None,
            "val_tuned_risk": bool(sweep_summary.get("val_tuned_risk", False)),
            "decisions": sweep_summary.get("decisions", {}),
        }

    save_json(args.output_dir / "summary.json", _json_safe(summary))
    save_jsonl(args.output_dir / "per_sample_likelihoods.jsonl", [_json_safe(row) for row in per_sample_rows])
    _write_csv(
        args.output_dir / "per_category_likelihood.csv",
        per_category_rows,
        ["category", "support", "mean_map_energy", "mean_mixture_nll_per_cell", "mean_mask_nll_per_cell", "mean_label_nll_per_fg", "mean_num_foreground", "mean_label_diversity"],
    )
    _write_csv(
        args.output_dir / "baseline_comparison.csv",
        baseline_comparison_rows,
        ["baseline", "support", "mean_map_energy", "mean_mixture_nll_per_cell", "mean_mask_nll_per_cell", "mean_label_nll_per_fg", "delta_vs_instruction_mixture_nll_per_cell", "instruction_improves"],
    )
    _write_markdown_table(
        args.output_dir / "baseline_comparison.md",
        "Baseline Comparison",
        baseline_comparison_rows,
        ["baseline", "support", "mean_mixture_nll_per_cell", "mean_mask_nll_per_cell", "mean_label_nll_per_fg", "delta_vs_instruction_mixture_nll_per_cell", "instruction_improves"],
    )
    with (args.output_dir / "baseline_comparison.md").open("a", encoding="utf-8") as handle:
        handle.write("\n## Notes\n\n")
        handle.write("- calibrated prior is evaluated as full generative prior.\n")
        handle.write("- label_only rows are diagnostic only and do not model foreground support.\n")
        handle.write("- B0.6.4 decision should rely on likelihood and mode usage jointly.\n")
    if calibration_comparison_rows:
        _write_csv(
            args.output_dir / "calibration_comparison.csv",
            calibration_comparison_rows,
            [
                "variant",
                "support",
                "mean_mixture_nll_per_cell",
                "mean_mask_nll_per_cell",
                "mean_label_nll_per_fg",
                "mean_true_token_probability",
                "delta_label_nll_vs_raw_raw",
                "delta_mixture_nll_vs_raw_raw",
                "calibrated_label_nll_improves_vs_raw_raw",
                "calibrated_mixture_nll_improves_vs_raw_raw",
                "mode_usage_risk_categories",
            ],
        )
        _write_markdown_table(
            args.output_dir / "calibration_comparison.md",
            "Calibration Comparison",
            calibration_comparison_rows,
            [
                "variant",
                "support",
                "mean_mixture_nll_per_cell",
                "mean_mask_nll_per_cell",
                "mean_label_nll_per_fg",
                "mean_true_token_probability",
                "delta_label_nll_vs_raw_raw",
                "delta_mixture_nll_vs_raw_raw",
                "mode_usage_risk_categories",
            ],
        )
    _write_csv(
        args.output_dir / "mode_usage.csv",
        mode_usage_rows,
        [
            "category",
            "support",
            "mode_count",
            "used_modes",
            "unused_modes",
            "unused_fraction",
            "max_mode_fraction",
            "mode_usage_entropy",
            "normalized_mode_usage_entropy",
            "risk_flags",
            "suggested_mode_count",
            "suggested_action",
            "counts_json",
        ],
    )
    save_json(args.output_dir / "mode_usage_by_category.json", _json_safe(mode_usage_payload))
    save_json(args.output_dir / "mode_usage_risks.json", _json_safe({category: payload for category, payload in mode_usage_payload.items() if payload.get("risk_flags")}))
    _write_csv(args.output_dir / "mask_calibration.csv", calibration_rows, ["bin", "count", "predicted_probability", "empirical_foreground_rate"])
    _write_histogram_png(
        args.output_dir / "energy_histograms.png",
        {
            "map_energy": [float(row["map_energy"]) for row in per_sample_rows],
            "mixture_nll_per_cell": [float(row["mixture_nll_per_cell"]) for row in per_sample_rows],
            "label_nll_per_fg": [float(row["label_nll_per_fg"]) for row in per_sample_rows],
        },
    )
    _write_mask_calibration_png(args.output_dir / "mask_calibration.png", calibration_rows)
    summary_lines = [
        "# Instruction Grammar Likelihood Summary",
        "",
        f"split_name: {args.split_name}",
        f"rows_read: {len(rows)}",
        f"evaluated: {len(per_sample_rows)}",
        f"skipped_unknown_category: {skipped_unknown_category}",
        f"skipped_no_modes: {skipped_no_modes}",
        f"empty_foreground_count: {empty_foreground_count}",
        f"nan_or_inf_count: {nan_or_inf_count}",
        f"mean_map_energy: {summary['mean_map_energy']:.6f}",
        f"mean_mixture_nll_per_cell: {summary['mean_mixture_nll_per_cell']:.6f}",
        f"mean_mask_nll_per_cell: {summary['mean_mask_nll_per_cell']:.6f}",
        f"mean_label_nll_per_fg: {summary['mean_label_nll_per_fg']:.6f}",
        f"mean_true_token_probability: {summary['mean_true_token_probability']:.6f}",
        f"instruction_prior_improves_over_all_available_baselines: {summary['instruction_prior_improves_over_all_available_baselines']}",
        "",
        "## Mode Usage Risks",
        "",
        json.dumps(_json_safe(summary["mode_usage_risk_categories"]), ensure_ascii=False, indent=2),
        "",
        "## Baseline Comparison",
        "",
        "Note: calibrated prior is evaluated as full generative prior.",
        "Note: label_only rows are diagnostic only and do not model foreground support.",
        "Note: mode_usage risks should be interpreted jointly with likelihood.",
        "Note: unused_modes on small val can be benign; high max_mode_fraction can indicate natural unimodality or mode collapse.",
        "Note: B0.6.4 decision should rely on likelihood + usage jointly.",
        "",
    ]
    for row in baseline_comparison_rows:
        summary_lines.append(
            "- {baseline}: mixture={mix:.6f} mask={mask:.6f} label={label:.6f} improves={improves}".format(
                baseline=row["baseline"],
                mix=float(row["mean_mixture_nll_per_cell"]),
                mask=float(row["mean_mask_nll_per_cell"]),
                label=float(row["mean_label_nll_per_fg"]),
                improves=row["instruction_improves"],
            )
        )
    (args.output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(
        "[eval-igr-likelihood] done "
        f"evaluated={len(per_sample_rows)} "
        f"mean_mixture_nll_per_cell={summary['mean_mixture_nll_per_cell']:.6f} "
        f"output_dir={args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
