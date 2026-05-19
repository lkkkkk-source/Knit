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


def _validate_modes(entry: dict[str, Any], category: str, source_name: str) -> tuple[list[Any], list[Any], list[float]]:
    fg_prob = _to_list(entry.get("basis_fg_mask_prob"))
    label_prob = _to_list(entry.get("basis_label_prob_16"))
    if not isinstance(fg_prob, list) or not isinstance(label_prob, list):
        raise ValueError(f"{source_name} category {category!r} is missing basis_fg_mask_prob/basis_label_prob_16.")
    if len(fg_prob) != len(label_prob):
        raise ValueError(f"{source_name} category {category!r} has mismatched mode counts.")
    mode_count = len(fg_prob)
    if mode_count <= 0:
        raise ValueError(f"{source_name} category {category!r} has no modes.")
    mode_num_samples = _to_list(entry.get("mode_num_samples", []))
    priors: list[float]
    if isinstance(mode_num_samples, list) and len(mode_num_samples) == mode_count and sum(max(0.0, _safe_float(value)) for value in mode_num_samples) > 0.0:
        total = sum(max(0.0, _safe_float(value)) for value in mode_num_samples)
        priors = [max(0.0, _safe_float(value)) / total for value in mode_num_samples]
    else:
        priors = [1.0 / float(mode_count) for _ in range(mode_count)]
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
    out["mode_label_only"] = _score_modes(
        y20,
        mask20,
        [_constant_mask_mode(0.5)[0] for _ in fg_modes],
        label_modes,
        priors,
        category=category,
        source_name="mode_label_only",
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
            d_fg, d_label, d_priors = _validate_modes(dictionary_entry, category, "dictionary_bank")
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


def _baseline_summary(primary_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {"instruction_matrix_grammar_prior": primary_rows}
    for row in baseline_rows:
        grouped.setdefault(str(row["baseline"]), []).append(row)
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
                "delta_vs_instruction_mixture_nll_per_cell": float(mean_mix - primary_mean) if name != "instruction_matrix_grammar_prior" else 0.0,
                "instruction_improves": bool(primary_mean < mean_mix) if name != "instruction_matrix_grammar_prior" else None,
            }
        )
    return out


def _mode_usage(rows: list[dict[str, Any]], prior_categories: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
        risks = []
        if max_fraction > 0.8:
            risks.append("max_mode_fraction_gt_0.8")
        if mode_count > 1 and len(unused) >= max(1, mode_count // 2):
            risks.append("many_modes_unused")
        payload[category] = {
            "support": int(total),
            "mode_count": int(mode_count),
            "counts": {str(key): int(value) for key, value in sorted(counts.items())},
            "mode_usage_entropy": float(entropy),
            "max_mode_fraction": float(max_fraction),
            "unused_modes": unused,
            "risks": risks,
        }
        csv_rows.append(
            {
                "category": category,
                "support": int(total),
                "mode_count": int(mode_count),
                "mode_usage_entropy": float(entropy),
                "max_mode_fraction": float(max_fraction),
                "unused_modes": " ".join(str(value) for value in unused),
                "risk_flags": " ".join(risks),
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data_cf = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
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
            fg_modes, label_modes, priors = _validate_modes(entry, category, "instruction_matrix_grammar_prior")
        except Exception as error:
            skipped_no_modes += 1
            print(f"WARNING skip sample_id={sample_id} category={category}: {error}", flush=True)
            continue
        y20, mask20, is_empty = _load_y_for_row(row, manifest_root, data_cf)
        if is_empty:
            empty_foreground_count += 1
        primary = _score_modes(
            y20,
            mask20,
            fg_modes,
            label_modes,
            priors,
            category=category,
            source_name="instruction_matrix_grammar_prior",
            eps=float(args.eps),
            label_smoothing=float(args.label_smoothing),
            mask_smoothing=float(args.mask_smoothing),
            w_mask=float(args.w_mask),
            w_label=float(args.w_label),
        )
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
            "mode_posterior": [float(value) for value in primary.get("mode_posterior", [])],
        }
        per_sample_rows.append(sample_row)
        if args.baselines:
            baselines = _score_baselines(
                y20,
                mask20,
                category=category,
                fg_modes=fg_modes,
                label_modes=label_modes,
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
    baseline_comparison_rows = _baseline_summary(per_sample_rows, baseline_sample_rows) if args.baselines else _baseline_summary(per_sample_rows, [])
    mode_usage_rows, mode_usage_payload = _mode_usage(per_sample_rows, categories)
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
        if str(name) != "instruction_matrix_grammar_prior"
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
        "improves_over_baselines": improves_over_baselines,
        "instruction_prior_improves_over_all_available_baselines": bool(improves_over_baselines and all(improves_over_baselines.values())),
        "mode_usage_risk_categories": {category: payload["risks"] for category, payload in mode_usage_payload.items() if payload.get("risks")},
        "baseline_warnings": baseline_warnings[:100],
        "parameters": {
            "eps": float(args.eps),
            "label_smoothing": float(args.label_smoothing),
            "mask_smoothing": float(args.mask_smoothing),
            "w_mask": float(args.w_mask),
            "w_label": float(args.w_label),
        },
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
    _write_csv(
        args.output_dir / "mode_usage.csv",
        mode_usage_rows,
        ["category", "support", "mode_count", "mode_usage_entropy", "max_mode_fraction", "unused_modes", "risk_flags", "counts_json"],
    )
    save_json(args.output_dir / "mode_usage_by_category.json", _json_safe(mode_usage_payload))
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
