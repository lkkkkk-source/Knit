from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compose_foreground import compose_foreground
from .grammar_energy import GrammarEnergy, compute_candidate_descriptors
from .inspect_foreground_planner import _require_torch
from .utils import InstructionGrammarPriorError, bbox_from_mask, bbox_vector, build_planner_from_checkpoint_payload, checkpoint_get, cuda_diagnostics, finish_progress, foreground_area, foreground_descriptor, format_device_name, format_metric_line, label_diversity_on_fg, load_config, mask_component_stats, model_parameter_device, normalized_l2_between, print_progress, require_centroid_sketch_fields, require_foreground_cache_fields, resolve_canonical_mode, resolve_device, save_binary_map, save_json, save_jsonl, save_label_grid_mosaic, save_label_map, validate_instruction_grammar_prior_category


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample category-only foreground candidates.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--num-candidates", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None, help="Device override: auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--prior-source", type=str, default=None, choices=["category_kmeans", "instruction_matrix_grammar"])
    parser.add_argument("--prior-mode", type=str, default="sample", choices=["top", "sample", "all"])
    parser.add_argument("--fallback-if-missing", dest="fallback_if_missing", action="store_true", default=None)
    parser.add_argument("--no-fallback-if-missing", dest="fallback_if_missing", action="store_false")
    parser.add_argument("--use-grammar-rerank", action="store_true")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--save-raw-candidates", action="store_true")
    parser.add_argument("--energy-config", type=Path, default=None)
    parser.add_argument("--diverse-topk", dest="diverse_topk", action="store_true", default=None)
    parser.add_argument("--no-diverse-topk", dest="diverse_topk", action="store_false")
    parser.add_argument("--diversity-weight", type=float, default=0.25)
    parser.add_argument("--duplicate-mask-iou-threshold", type=float, default=0.95)
    parser.add_argument("--duplicate-label-hamming-threshold", type=float, default=0.05)
    parser.add_argument("--dump-logit-diagnostics", action="store_true", help="Write lightweight mask/label logit diagnostics without changing sampling.")
    parser.add_argument("--mask-thresholds", type=str, default="0.2,0.3,0.5", help="Comma-separated mask thresholds used only for diagnostics.")
    return parser


def _to_list(value: object) -> object:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _choose_mode(torch: object, priors: list[float], strategy: str, count_index: int) -> int:
    mode_count = len(priors)
    if mode_count <= 0:
        raise ValueError("Cannot choose prior mode from an empty mode list.")
    if strategy == "top":
        return int(max(range(mode_count), key=lambda index: float(priors[index])))
    if strategy == "all":
        return int(count_index % mode_count)
    weights = getattr(torch, "tensor")([max(0.0, float(value)) for value in priors], dtype=getattr(torch, "float32"))
    if float(weights.sum().item()) <= 0.0:
        weights = getattr(torch, "ones")((mode_count,), dtype=getattr(torch, "float32")) / float(mode_count)
    else:
        weights = weights / weights.sum()
    return int(getattr(torch, "multinomial")(weights, num_samples=1).item())


def _binary_mask_from_prob(fg_prob: list[list[list[float]]]) -> list[list[int]]:
    grid = fg_prob[0]
    mask = [[1 if float(grid[y][x]) >= 0.5 else 0 for x in range(20)] for y in range(20)]
    if sum(sum(row) for row in mask) <= 0:
        flat = sorted((float(grid[y][x]), y, x) for y in range(20) for x in range(20))
        for _, y, x in flat[-max(1, len(flat) // 20):]:
            mask[y][x] = 1
    return mask


def _argmax_labels(label_prob: list[list[list[float]]], mask: list[list[int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for y in range(20):
        row: list[int] = []
        for x in range(20):
            if not mask[y][x]:
                row.append(0)
            else:
                row.append(int(max(range(16), key=lambda idx: float(label_prob[idx][y][x])) + 1))
        out.append(row)
    return out


def _parse_mask_thresholds(text: str) -> list[float]:
    thresholds: list[float] = []
    for part in str(text).split(","):
        value_text = part.strip()
        if not value_text:
            continue
        value = float(value_text)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"--mask-thresholds values must be in [0,1], got {value}.")
        thresholds.append(value)
    return thresholds or [0.5]


def _dominant_label_ratio(label_grid: list[list[int]], mask_grid: list[list[int]]) -> float:
    counts: dict[int, int] = {}
    total = 0
    for y_pos in range(20):
        for x_pos in range(20):
            if int(mask_grid[y_pos][x_pos]) <= 0:
                continue
            label = int(label_grid[y_pos][x_pos])
            if label <= 0:
                continue
            counts[label] = counts.get(label, 0) + 1
            total += 1
    return float(max(counts.values(), default=0)) / float(max(1, total))


def _tensor_stats(tensor: object) -> dict[str, float]:
    return {
        "min": float(tensor.min().item()),
        "mean": float(tensor.mean().item()),
        "max": float(tensor.max().item()),
    }


def _label_entropy_mean(torch: object, prob: object, mask: object) -> float:
    mask_bool = mask.to(dtype=getattr(torch, "bool"))
    if int(mask_bool.sum().item()) <= 0:
        return 0.0
    eps = 1.0e-8
    entropy = -(prob.clamp_min(eps) * prob.clamp_min(eps).log()).sum(dim=0)
    return float(entropy[mask_bool].mean().item())


def _masked_top_prob_means(torch: object, prob: object, mask: object) -> tuple[float, float, float]:
    mask_bool = mask.to(dtype=getattr(torch, "bool"))
    if int(mask_bool.sum().item()) <= 0:
        return 0.0, 0.0, 0.0
    top2 = getattr(torch, "topk")(prob, k=2, dim=0).values
    top1_values = top2[0][mask_bool]
    top2_values = top2[1][mask_bool]
    return (
        float(top1_values.mean().item()),
        float(top2_values.mean().item()),
        float((top1_values - top2_values).mean().item()),
    )


def _candidate_logit_diagnostics(
    torch: object,
    fg_mask_logits: object,
    fg_label_logits: object,
    prior_fg_mask_prob: object,
    prior_label_prob_16: object,
    fg_mask: list[list[int]],
    fg_label: list[list[int]],
    composed_y20: list[list[int]],
    label_diversity: float,
    dominant_label_ratio: float,
    mask_thresholds: list[float],
) -> dict[str, object]:
    mask_prob = getattr(torch, "sigmoid")(fg_mask_logits.detach())
    label_logits = fg_label_logits.detach()
    label_prob = getattr(__import__("importlib").import_module("torch.nn.functional"), "softmax")(label_logits, dim=0)
    pred_mask_tensor = (mask_prob >= 0.5)
    prior_mask_tensor = (prior_fg_mask_prob.detach()[0] >= 0.5)
    label_argmax = (label_logits.argmax(dim=0) + 1).detach()
    prior_label_argmax = (prior_label_prob_16.detach().argmax(dim=0) + 1).detach()
    fg_mask_tensor = getattr(torch, "tensor")(fg_mask, dtype=getattr(torch, "bool"), device=fg_mask_logits.device)
    fg_label_argmax_unique_fg = sorted({int(value) for value in label_argmax[fg_mask_tensor].detach().cpu().tolist()}) if int(fg_mask_tensor.sum().item()) > 0 else []
    prior_unique_fg = sorted({int(value) for value in prior_label_argmax[prior_mask_tensor].detach().cpu().tolist()}) if int(prior_mask_tensor.sum().item()) > 0 else []
    prior_label_grid = [[int(prior_label_argmax[y, x].item()) if bool(prior_mask_tensor[y, x].item()) else 0 for x in range(20)] for y in range(20)]
    prior_mask_grid = prior_mask_tensor.to(dtype=getattr(torch, "long")).detach().cpu().tolist()
    top1_mean, top2_mean, margin_mean = _masked_top_prob_means(torch, label_prob, fg_mask_tensor)
    area_by_threshold = {
        f"{threshold:.6g}": float(((mask_prob >= threshold).to(dtype=getattr(torch, "float32")).mean()).item())
        for threshold in mask_thresholds
    }
    warnings: list[str] = []
    if len(prior_unique_fg) > 1 and len(fg_label_argmax_unique_fg) <= 1:
        warnings.append("model_not_using_label_prior")
    if len(prior_unique_fg) <= 1:
        warnings.append("selected_prior_label_collapse")
    if len(fg_label_argmax_unique_fg) <= 1:
        warnings.append("label_head_argmax_collapse")
    area_03 = area_by_threshold.get("0.3")
    area_05 = area_by_threshold.get("0.5")
    if area_03 is not None and area_05 is not None and abs(area_03 - area_05) >= 0.05:
        warnings.append("mask_threshold_sensitive")
    return {
        "mask": {
            **_tensor_stats(fg_mask_logits.detach()),
            "sigmoid_q50": float(getattr(torch, "quantile")(mask_prob.flatten(), 0.50).item()),
            "sigmoid_q90": float(getattr(torch, "quantile")(mask_prob.flatten(), 0.90).item()),
            "sigmoid_q95": float(getattr(torch, "quantile")(mask_prob.flatten(), 0.95).item()),
            "sigmoid_q99": float(getattr(torch, "quantile")(mask_prob.flatten(), 0.99).item()),
            "foreground_area_by_threshold": area_by_threshold,
        },
        "label": {
            **_tensor_stats(label_logits),
            "std": float(label_logits.std(unbiased=False).item()),
            "fg_label_argmax_unique_tokens_all": sorted({int(value) for value in label_argmax.detach().cpu().flatten().tolist()}),
            "fg_label_argmax_unique_tokens_pred_foreground": fg_label_argmax_unique_fg,
            "fg_label_softmax_entropy_mean_pred_foreground": _label_entropy_mean(torch, label_prob, fg_mask_tensor),
            "fg_label_softmax_entropy_mean_prior_foreground": _label_entropy_mean(torch, label_prob, prior_mask_tensor),
            "top1_label_prob_mean_foreground": top1_mean,
            "top2_label_prob_mean_foreground": top2_mean,
            "top1_minus_top2_margin_mean_foreground": margin_mean,
            "composed_y20_unique_values": sorted({int(value) for row in composed_y20 for value in row}),
            "label_diversity": float(label_diversity),
            "dominant_label_ratio": float(dominant_label_ratio),
        },
        "prior_label": {
            "prior_fg_area": float(prior_mask_tensor.to(dtype=getattr(torch, "float32")).mean().item()),
            "prior_label_argmax_unique_tokens_prior_foreground": prior_unique_fg,
            "prior_label_prob_16_entropy_mean_prior_foreground": _label_entropy_mean(torch, prior_label_prob_16.detach(), prior_mask_tensor),
            "prior_dominant_label_ratio": _dominant_label_ratio(prior_label_grid, prior_mask_grid),
            "prior_label_diversity": float(label_diversity_on_fg(prior_label_grid, prior_mask_grid)),
        },
        "model_prior_comparison": {
            "predicted_argmax_unique_vs_prior_argmax_unique": {
                "predicted": fg_label_argmax_unique_fg,
                "prior": prior_unique_fg,
            },
            "warnings": warnings,
        },
    }


def _label_diagnostics_summary(rows: list[dict[str, object]], mask_thresholds: list[float]) -> dict[str, object]:
    warnings_by_type: dict[str, int] = {}
    pred_hist: dict[str, int] = {}
    prior_hist: dict[str, int] = {}
    threshold_area_sum = {f"{threshold:.6g}": 0.0 for threshold in mask_thresholds}
    threshold_area_count = 0
    for row in rows:
        diag = row.get("label_diagnostics")
        if not isinstance(diag, dict):
            continue
        pred_unique = diag.get("label", {}).get("fg_label_argmax_unique_tokens_pred_foreground", [])
        prior_unique = diag.get("prior_label", {}).get("prior_label_argmax_unique_tokens_prior_foreground", [])
        pred_hist[str(len(pred_unique))] = pred_hist.get(str(len(pred_unique)), 0) + 1
        prior_hist[str(len(prior_unique))] = prior_hist.get(str(len(prior_unique)), 0) + 1
        for warning in diag.get("model_prior_comparison", {}).get("warnings", []):
            warnings_by_type[str(warning)] = warnings_by_type.get(str(warning), 0) + 1
        area_by_threshold = diag.get("mask", {}).get("foreground_area_by_threshold", {})
        if isinstance(area_by_threshold, dict):
            threshold_area_count += 1
            for key in threshold_area_sum:
                threshold_area_sum[key] += float(area_by_threshold.get(key, 0.0))
    return {
        "num_candidates": len(rows),
        "num_nonempty": sum(1 for row in rows if float(row.get("fg_area", 0.0)) > 0.0),
        "num_low_label_diversity": sum(1 for row in rows if "low_label_diversity" in row.get("invalid_reasons", [])),
        "predicted_unique_token_histogram": dict(sorted(pred_hist.items())),
        "prior_unique_token_histogram": dict(sorted(prior_hist.items())),
        "warnings_by_type": dict(sorted(warnings_by_type.items())),
        "mask_thresholds": mask_thresholds,
        "foreground_area_mean_by_threshold": {
            key: value / float(max(1, threshold_area_count)) for key, value in sorted(threshold_area_sum.items())
        },
    }


def _instruction_prior_batch(cache_payload: dict[str, object], category: str, torch: object, count: int, strategy: str, label_prob_key: str, mode_prior_key: str) -> dict[str, object]:
    validated = validate_instruction_grammar_prior_category(cache_payload, category, label_prob_key=label_prob_key, mode_prior_key=mode_prior_key)
    fg_modes = validated["fg_modes"]
    label_modes = validated["label_modes"]
    if not isinstance(fg_modes, list) or not isinstance(label_modes, list):
        raise InstructionGrammarPriorError("Invalid instruction_matrix_grammar_prior: validated mode tensors must be lists.")
    priors = [float(value) for value in validated["mode_prior"]]
    mode_ids = [_choose_mode(torch, priors, strategy, index) for index in range(count)]
    fg_probs = []
    label_probs = []
    label_hists = []
    row_proj = []
    col_proj = []
    adjacency = []
    transition = []
    bbox_stats = []
    for mode_id in mode_ids:
        fg_prob = _to_list(fg_modes[mode_id])
        label_prob = _to_list(label_modes[mode_id])
        if not isinstance(fg_prob, list) or not isinstance(label_prob, list):
            raise ValueError(f"instruction_matrix_grammar_prior category {category!r} mode={mode_id} invalid tensors.")
        fg_mask = _binary_mask_from_prob(fg_prob)
        fg_label = _argmax_labels(label_prob, fg_mask)
        bbox = bbox_from_mask([[bool(value) for value in row] for row in fg_mask])
        desc = foreground_descriptor(fg_label, fg_mask, bbox)
        fg_probs.append(fg_prob)
        label_probs.append(label_prob)
        label_hists.append(desc["label_hist_16"])
        row_proj.append(desc["row_projection"])
        col_proj.append(desc["col_projection"])
        adjacency.append(desc["adjacency_signature"])
        transition.append(desc["transition_2x2_stats"])
        bbox_stats.append(bbox_vector(bbox))
    return {
        "prior_source": "instruction_matrix_grammar",
        "mode_ids": mode_ids,
        "mode_count": int(validated["num_modes"]),
        "mode_prior_key": str(validated["mode_prior_key"]),
        "label_prob_key": str(validated["label_prob_key"]),
        "mode_prior": priors,
        "mode_prior_sum": float(validated["mode_prior_sum"]),
        "mode_prior_min": float(validated["mode_prior_min"]),
        "mode_prior_max": float(validated["mode_prior_max"]),
        "schema_version": str(validated["schema_version"]),
        "num_categories": int(validated["num_categories"]),
        "warnings": list(validated["warnings"]),
        "centroid_fg_mask_prob": getattr(torch, "tensor")(fg_probs, dtype=getattr(torch, "float32")),
        "centroid_label_prob_16": getattr(torch, "tensor")(label_probs, dtype=getattr(torch, "float32")),
        "centroid_label_hist": getattr(torch, "tensor")(label_hists, dtype=getattr(torch, "float32")),
        "centroid_row_projection": getattr(torch, "tensor")(row_proj, dtype=getattr(torch, "float32")),
        "centroid_col_projection": getattr(torch, "tensor")(col_proj, dtype=getattr(torch, "float32")),
        "centroid_adjacency": getattr(torch, "tensor")(adjacency, dtype=getattr(torch, "float32")),
        "centroid_transition_stats": getattr(torch, "tensor")(transition, dtype=getattr(torch, "float32")),
        "centroid_bbox_stats": getattr(torch, "tensor")(bbox_stats, dtype=getattr(torch, "float32")),
    }


def _descriptor_margin(descriptor: list[float], category: str, cache_payload: dict[str, object]) -> tuple[float, float, float]:
    if category not in cache_payload["descriptor_mean_by_category"] or category not in cache_payload["descriptor_std_by_category"]:
        raise ValueError(f"Foreground train cache is missing descriptor statistics for category {category!r}.")
    own_dist = normalized_l2_between(
        descriptor,
        cache_payload["descriptor_mean_by_category"].get(category, []),
        cache_payload["descriptor_global_mean"],
        cache_payload["descriptor_global_std"],
    )
    nearest_other = float("inf")
    for other_category, mean in cache_payload["descriptor_mean_by_category"].items():
        if other_category == category:
            continue
        value = normalized_l2_between(descriptor, mean, cache_payload["descriptor_global_mean"], cache_payload["descriptor_global_std"])
        nearest_other = min(nearest_other, value)
    margin = own_dist - nearest_other if nearest_other < float("inf") else own_dist
    return own_dist, nearest_other, margin


def _mean(rows: list[dict[str, object]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in rows) / float(len(rows))


def _mean_energy(rows: list[dict[str, object]], key: str) -> float:
    vals = []
    for row in rows:
        energy = row.get("grammar_energy")
        if isinstance(energy, dict):
            vals.append(float(energy.get(key, 0.0)))
    return sum(vals) / float(max(1, len(vals)))


def _summary_means(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {
            "dominant_label_ratio": 0.0,
            "label_diversity": 0.0,
            "total": 0.0,
            "trans": 0.0,
            "motif": 0.0,
            "div": 0.0,
            "dom": 0.0,
            "single_label_collapse": 0.0,
        }
    return {
        "dominant_label_ratio": sum(float(row["grammar_diagnostics"]["dominant_label_ratio"]) for row in rows) / float(len(rows)),
        "label_diversity": sum(float(row["grammar_diagnostics"]["label_diversity"]) for row in rows) / float(len(rows)),
        "total": _mean_energy(rows, "total"),
        "trans": _mean_energy(rows, "trans"),
        "motif": _mean_energy(rows, "motif"),
        "div": _mean_energy(rows, "div"),
        "dom": _mean_energy(rows, "dom"),
        "single_label_collapse": _mean_energy(rows, "single_label_collapse"),
    }


def _summary_improvement(selected: dict[str, float], raw: dict[str, float]) -> dict[str, float]:
    return {key: float(selected.get(key, 0.0)) - float(raw.get(key, 0.0)) for key in sorted(set(raw) | set(selected))}


def _load_energy_override(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as error:
            raise ImportError("PyYAML is required for non-JSON --energy-config files.") from error
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"--energy-config must contain a mapping, got {type(payload)}")
    return payload


def _mask_iou(a: list[list[int]], b: list[list[int]]) -> float:
    inter = 0
    union = 0
    for y in range(20):
        for x in range(20):
            av = int(a[y][x]) > 0
            bv = int(b[y][x]) > 0
            inter += int(av and bv)
            union += int(av or bv)
    return float(inter) / float(max(1, union))


def _y20_hamming(a: list[list[int]], b: list[list[int]]) -> float:
    diff = 0
    total = 0
    for y in range(20):
        for x in range(20):
            total += 1
            diff += int(int(a[y][x]) != int(b[y][x]))
    return float(diff) / float(max(1, total))


def _l1(a: list[float], b: list[float]) -> float:
    return sum(abs(float(x) - float(y)) for x, y in zip(a, b)) / float(max(1, min(len(a), len(b))))


def _candidate_distance(a: dict[str, object], b: dict[str, object]) -> float:
    ad = a["grammar_descriptor"]
    bd = b["grammar_descriptor"]
    return (
        _l1(ad["label_hist_norm_16"], bd["label_hist_norm_16"])
        + _l1(ad["row_fg_projection"], bd["row_fg_projection"])
        + _l1(ad["col_fg_projection"], bd["col_fg_projection"])
        + (1.0 - _mask_iou(ad["fg_mask"], bd["fg_mask"]))
        + _y20_hamming(a["composed_y20"], b["composed_y20"])
    ) / 5.0


def _single_label_collapse(row: dict[str, object]) -> float:
    energy = row.get("grammar_energy")
    if not isinstance(energy, dict):
        return 0.0
    return float(energy.get("single_label_collapse", 0.0))


def _select_diverse_topk(
    rows: list[dict[str, object]],
    energy_order: list[int],
    top_k: int,
    *,
    diversity_weight: float,
    duplicate_mask_iou_threshold: float,
    duplicate_label_hamming_threshold: float,
) -> tuple[list[int], list[int], list[int], bool, bool]:
    rows_by_index = {int(row["index"]): row for row in rows}
    selected: list[int] = []
    skipped: list[int] = []
    collapse_skipped: list[int] = []
    remaining = list(energy_order)
    has_noncollapse_candidate = any(_single_label_collapse(rows_by_index[idx]) <= 0.0 for idx in energy_order if idx in rows_by_index)
    while remaining and len(selected) < top_k:
        best_index = None
        best_score = float("inf")
        for idx in remaining:
            row = rows_by_index[idx]
            if has_noncollapse_candidate and _single_label_collapse(row) > 0.0:
                continue
            duplicate = False
            for selected_idx in selected:
                selected_row = rows_by_index[selected_idx]
                mask_iou = _mask_iou(row["grammar_descriptor"]["fg_mask"], selected_row["grammar_descriptor"]["fg_mask"])
                hamming = _y20_hamming(row["composed_y20"], selected_row["composed_y20"])
                if mask_iou > duplicate_mask_iou_threshold or hamming < duplicate_label_hamming_threshold:
                    duplicate = True
                    break
            if duplicate:
                continue
            min_dist = min((_candidate_distance(row, rows_by_index[s]) for s in selected), default=0.0)
            total = float((row.get("grammar_energy") or {}).get("total", 0.0))
            adjusted = total - float(diversity_weight) * min_dist
            if adjusted < best_score:
                best_score = adjusted
                best_index = idx
        if best_index is None:
            break
        selected.append(best_index)
        remaining = [idx for idx in remaining if idx != best_index]
    selected_set = set(selected)
    for idx in energy_order:
        if idx in selected_set:
            continue
        duplicate = False
        for selected_idx in selected:
            row = rows_by_index[idx]
            selected_row = rows_by_index[selected_idx]
            if _mask_iou(row["grammar_descriptor"]["fg_mask"], selected_row["grammar_descriptor"]["fg_mask"]) > duplicate_mask_iou_threshold or _y20_hamming(row["composed_y20"], selected_row["composed_y20"]) < duplicate_label_hamming_threshold:
                duplicate = True
                break
        if duplicate:
            skipped.append(idx)
        if idx not in selected_set and has_noncollapse_candidate and _single_label_collapse(rows_by_index[idx]) > 0.0:
            collapse_skipped.append(idx)
    duplicate_fill = False
    collapse_fill = False
    if len(selected) < top_k:
        duplicate_fill = True
        for idx in energy_order:
            if idx not in selected_set and _single_label_collapse(rows_by_index[idx]) <= 0.0:
                selected.append(idx)
                selected_set.add(idx)
            if len(selected) >= top_k:
                break
    if len(selected) < top_k:
        collapse_fill = True
        for idx in energy_order:
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
            if len(selected) >= top_k:
                break
    skipped = [idx for idx in skipped if idx not in selected_set]
    collapse_skipped = [idx for idx in collapse_skipped if idx not in selected_set]
    if not has_noncollapse_candidate and any(_single_label_collapse(rows_by_index[idx]) > 0.0 for idx in selected):
        collapse_fill = True
    return selected[:top_k], skipped, collapse_skipped, duplicate_fill, collapse_fill


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    mask_thresholds = _parse_mask_thresholds(args.mask_thresholds)
    config = load_config(args.config)
    canonical_mode = resolve_canonical_mode(config["data"])
    payload = _require_torch().load(args.checkpoint, map_location="cpu")
    checkpoint_compatibility = str(checkpoint_get(payload, "checkpoint_compatibility", required=False) or "")
    if checkpoint_compatibility and checkpoint_compatibility != "spatial-centroid-v2":
        raise ValueError(
            "Checkpoint is incompatible with the current spatial-centroid planner. "
            "Please retrain with the rebuilt full_masked cache and the current struct_foreground_v1 code."
        )
    checkpoint_canonical_mode = str(checkpoint_get(payload, "canonical_mode"))
    if checkpoint_canonical_mode != canonical_mode:
        raise ValueError(
            f"Canonical mode mismatch: checkpoint has {checkpoint_canonical_mode!r}, config expects {canonical_mode!r}."
        )
    metrics = payload.get("metrics", {})
    category_to_id = checkpoint_get(payload, "category_to_id")
    train_categories = list(checkpoint_get(payload, "train_categories"))
    if args.category not in train_categories:
        raise ValueError(
            f"category not available in trained foreground prior: {args.category}\n"
            f"available categories: {sorted(train_categories)}"
        )
    if args.category not in category_to_id:
        raise KeyError(f"Unknown category {args.category!r}. Available categories: {sorted(category_to_id)}")
    output_dir = Path(args.output_dir or (args.checkpoint.parent / "samples" / args.category))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch = _require_torch()
    _ = checkpoint_get(payload, "descriptor_slices")
    _ = checkpoint_get(payload, "descriptor_global_mean")
    _ = checkpoint_get(payload, "descriptor_global_std")
    _ = checkpoint_get(payload, "category_foreground_area_stats")
    cache_payload = torch.load(Path(str(checkpoint_get(payload, "train_cache_path"))), map_location="cpu")
    require_foreground_cache_fields(cache_payload, context="Foreground train cache")
    require_centroid_sketch_fields(cache_payload, context="Foreground train cache")
    if args.category not in cache_payload["category_foreground_area_stats"]:
        raise ValueError(f"Foreground train cache is missing category_foreground_area_stats for category {args.category!r}.")
    planner_prior_cf = (config.get("instruction_matrix_grammar_prior", {}) if isinstance(config.get("instruction_matrix_grammar_prior", {}), dict) else {}).get("planner_prior", {})
    planner_prior_cf = planner_prior_cf if isinstance(planner_prior_cf, dict) else {}
    sampling_cf = config.get("sampling", {}) if isinstance(config.get("sampling", {}), dict) else {}
    device = resolve_device(torch, str(args.device or sampling_cf.get("device", "auto")))
    prior_source = str(args.prior_source or sampling_cf.get("prior_source", "category_kmeans"))
    prior_mode_strategy = str(args.prior_mode)
    label_prob_key = str(planner_prior_cf.get("label_prob_key", "basis_label_prob_16"))
    mode_prior_key = str(planner_prior_cf.get("mode_prior_key", "mode_prior_smoothed"))
    fallback_if_missing = bool(planner_prior_cf.get("fallback_if_missing", True))
    if args.fallback_if_missing is not None:
        fallback_if_missing = bool(args.fallback_if_missing)
    requested_prior_source = prior_source
    actual_prior_source = prior_source
    fallback_reason = ""
    prior_debug: dict[str, object] = {
        "requested_prior_source": requested_prior_source,
        "actual_prior_source": actual_prior_source,
        "fallback_if_missing": fallback_if_missing,
        "fallback_reason": fallback_reason,
        "mode_prior_key": mode_prior_key,
        "label_prob_key": label_prob_key,
        "prior_mode_strategy": prior_mode_strategy,
        "num_modes": 0,
        "mode_prior_sum": 0.0,
        "mode_prior_min": 0.0,
        "mode_prior_max": 0.0,
        "schema_version": None,
        "warnings": [],
    }
    grammar_bank = cache_payload.get("grammar_bank")
    rerank_cf = config.get("grammar_rerank", {}) if isinstance(config.get("grammar_rerank", {}), dict) else {}
    energy_override = _load_energy_override(args.energy_config)
    if energy_override:
        rerank_cf = {**rerank_cf, **energy_override}
    rerank_enabled = bool(args.use_grammar_rerank or rerank_cf.get("enabled", False))
    if rerank_enabled and not isinstance(grammar_bank, dict):
        raise ValueError("Grammar rerank requested but cache does not contain grammar_bank. Rebuild the cache first.")
    grammar_energy = GrammarEnergy(grammar_bank, weights=rerank_cf.get("weights", {}), config={**config.get("grammar_bank", {}), **rerank_cf}) if rerank_enabled else None
    diverse_topk = bool(args.diverse_topk if args.diverse_topk is not None else rerank_enabled)
    category_to_num_modes = checkpoint_get(payload, "category_to_num_modes")
    if args.category not in category_to_num_modes:
        raise ValueError(
            f"category not available in trained foreground prior: {args.category}\n"
            f"available categories: {sorted(train_categories)}"
        )
    model, model_kwargs, load_debug = build_planner_from_checkpoint_payload(payload, config, device=device)
    diag = cuda_diagnostics(torch)
    print(
        format_metric_line(
            "sample-prior-init:",
            [
                ("requested_prior_source", requested_prior_source),
                ("fallback_if_missing", fallback_if_missing),
                ("prior_mode_strategy", prior_mode_strategy),
                ("mode_prior_key", mode_prior_key),
                ("label_prob_key", label_prob_key),
            ],
        )
    )
    print(
        format_metric_line(
            "sample-device:",
            [
                ("device", format_device_name(device, torch)),
                ("model-device", model_parameter_device(model)),
                ("cuda_available", diag["cuda_available"]),
                ("cuda_device_count", diag["cuda_device_count"]),
                ("cuda_device_name", diag["cuda_device_name"]),
                ("torch_version", diag["torch_version"]),
            ],
        )
    )
    print(
        format_metric_line(
            "sample-foreground-load:",
            [
                ("source", load_debug["source"]),
                ("checkpoint_grammar_head_shape", load_debug["checkpoint_grammar_head_weight_shape"]),
                ("constructed_grammar_head_shape", load_debug["constructed_grammar_head_weight_shape"]),
                ("grammar_dim", model_kwargs["grammar_dim"]),
                ("hidden_dim", model_kwargs["hidden_dim"]),
                ("strict_load", load_debug["strict_load_success"]),
            ],
        )
    )
    model.eval()
    oversample = int(config["sampling"]["planner_oversample"])
    num_valid = int(config["sampling"]["num_valid_plans"])
    category_ids = getattr(torch, "full")((oversample,), int(category_to_id[args.category]), dtype=getattr(torch, "long"), device=device)
    rows = []
    with getattr(torch, "no_grad")():
        num_modes = int(category_to_num_modes[args.category])
        centroid_source = cache_payload["centroid_sketch_by_category"].get(args.category, {})
        max_num_modes = int(model_kwargs["max_num_modes"])
        fallback_count = 0
        if prior_source == "instruction_matrix_grammar":
            try:
                prior_batch = _instruction_prior_batch(cache_payload, args.category, torch, oversample, prior_mode_strategy, label_prob_key, mode_prior_key)
                local_z = [int(value) for value in prior_batch["mode_ids"]]
                condition_num_modes = min(max_num_modes, int(prior_batch["mode_count"]))
                mode_mask = getattr(torch, "tensor")([[1 if k < condition_num_modes else 0 for k in range(max_num_modes)] for _ in range(oversample)], dtype=getattr(torch, "bool"), device=device)
                centroid_fg_mask_prob = prior_batch["centroid_fg_mask_prob"].to(device)
                centroid_label_prob_16_tensor = prior_batch["centroid_label_prob_16"].to(device)
                centroid_label_hist_tensor = prior_batch["centroid_label_hist"].to(device)
                centroid_row_projection_tensor = prior_batch["centroid_row_projection"].to(device)
                centroid_col_projection_tensor = prior_batch["centroid_col_projection"].to(device)
                centroid_adjacency_tensor = prior_batch["centroid_adjacency"].to(device)
                centroid_transition_stats_tensor = prior_batch["centroid_transition_stats"].to(device)
                centroid_bbox_stats_tensor = prior_batch["centroid_bbox_stats"].to(device)
                label_prob_key = str(prior_batch["label_prob_key"])
                mode_prior_key = str(prior_batch["mode_prior_key"])
                prior_debug.update({
                    "actual_prior_source": "instruction_matrix_grammar",
                    "fallback_reason": "",
                    "mode_prior_key": mode_prior_key,
                    "label_prob_key": label_prob_key,
                    "num_modes": int(prior_batch["mode_count"]),
                    "mode_prior_sum": float(prior_batch["mode_prior_sum"]),
                    "mode_prior_min": float(prior_batch["mode_prior_min"]),
                    "mode_prior_max": float(prior_batch["mode_prior_max"]),
                    "schema_version": str(prior_batch["schema_version"]),
                    "instruction_grammar_categories": int(prior_batch["num_categories"]),
                    "warnings": list(prior_batch["warnings"]),
                })
            except InstructionGrammarPriorError as error:
                fallback_reason = str(error)
                if not fallback_if_missing:
                    raise ValueError(
                        f"{error} Use --prior-source category_kmeans for old cache/checkpoint."
                    ) from error
                fallback_count += 1
                actual_prior_source = "category_kmeans"
                prior_source = "category_kmeans"
                prior_debug.update({
                    "actual_prior_source": actual_prior_source,
                    "fallback_reason": fallback_reason,
                })
        if prior_source == "category_kmeans":
            mode_mask = getattr(torch, "tensor")([[1 if k < num_modes else 0 for k in range(max_num_modes)] for _ in range(oversample)], dtype=getattr(torch, "bool"), device=device)
            category_embed = model.category_embed(category_ids)
            z_logits = model.local_z_head(category_embed).masked_fill(mode_mask.logical_not(), float("-inf"))
            z_probs = getattr(__import__("importlib").import_module("torch.nn.functional"), "softmax")(z_logits, dim=-1)
            sampled_local_z = getattr(torch, "multinomial")(z_probs, num_samples=1).squeeze(1)
            centroid_label_hist = []
            centroid_label_prob_16 = []
            centroid_row_projection = []
            centroid_col_projection = []
            centroid_adjacency = []
            centroid_transition_stats = []
            centroid_bbox_stats = []
            local_z = []
            for index in range(oversample):
                mode_index = int(sampled_local_z[index].item())
                local_z.append(mode_index)
                centroid = centroid_source.get(mode_index)
                if centroid is None:
                    raise ValueError(f"Missing centroid sketch for category={args.category!r} local_z={mode_index}.")
                if "centroid_fg_mask_prob" not in centroid or "centroid_label_prob_16" not in centroid:
                    raise ValueError("Foreground cache is missing centroid_fg_mask_prob / centroid_label_prob_16. Please rebuild the cache with the current build_foreground_cache.py.")
                centroid_label_prob_16.append(centroid["centroid_label_prob_16"])
                centroid_label_hist.append(centroid["centroid_label_hist"])
                centroid_row_projection.append(centroid["centroid_row_projection"])
                centroid_col_projection.append(centroid["centroid_col_projection"])
                centroid_adjacency.append(centroid["centroid_adjacency"])
                centroid_transition_stats.append(centroid["centroid_transition_stats"])
                centroid_bbox_stats.append(centroid["centroid_bbox_stats"])
            centroid_fg_mask_prob = getattr(torch, "stack")([centroid_source[int(z)]["centroid_fg_mask_prob"] for z in local_z]).to(device=device, dtype=getattr(torch, "float32"))
            centroid_label_prob_16_tensor = getattr(torch, "stack")(centroid_label_prob_16).to(device=device, dtype=getattr(torch, "float32"))
            centroid_label_hist_tensor = getattr(torch, "tensor")(centroid_label_hist, dtype=getattr(torch, "float32"), device=device)
            centroid_row_projection_tensor = getattr(torch, "tensor")(centroid_row_projection, dtype=getattr(torch, "float32"), device=device)
            centroid_col_projection_tensor = getattr(torch, "tensor")(centroid_col_projection, dtype=getattr(torch, "float32"), device=device)
            centroid_adjacency_tensor = getattr(torch, "tensor")(centroid_adjacency, dtype=getattr(torch, "float32"), device=device)
            centroid_transition_stats_tensor = getattr(torch, "tensor")(centroid_transition_stats, dtype=getattr(torch, "float32"), device=device)
            centroid_bbox_stats_tensor = getattr(torch, "tensor")(centroid_bbox_stats, dtype=getattr(torch, "float32"), device=device)
            actual_prior_source = "category_kmeans"
            prior_debug.update({
                "actual_prior_source": actual_prior_source,
                "mode_prior_key": "model_local_z",
                "label_prob_key": "centroid_label_prob_16",
                "num_modes": int(num_modes),
                "mode_prior_sum": 1.0,
                "mode_prior_min": 0.0,
                "mode_prior_max": 1.0,
                "schema_version": cache_payload.get("meta", {}).get("schema_version") if isinstance(cache_payload.get("meta"), dict) else None,
            })
        out = model(
            category_ids,
            centroid_fg_mask_prob,
            centroid_label_prob_16_tensor,
            centroid_label_hist_tensor,
            centroid_row_projection_tensor,
            centroid_col_projection_tensor,
            centroid_adjacency_tensor,
            centroid_transition_stats_tensor,
            centroid_bbox_stats_tensor,
            getattr(torch, "tensor")(local_z, dtype=getattr(torch, "long"), device=device),
            mode_mask,
        )
        for index in range(oversample):
            fg_mask = (getattr(torch, "sigmoid")(out["fg_mask_logits"][index, 0]) >= 0.5).to(dtype=getattr(torch, "long")).detach().cpu().tolist()
            fg_label = (out["fg_label_logits"][index].argmax(dim=0) + 1).detach().cpu().tolist()
            fg_area = foreground_area(fg_mask)
            invalid = []
            if fg_area <= 0.0:
                invalid.append("empty_foreground")
            if fg_area >= 0.99:
                invalid.append("full_foreground")
            area_stats = cache_payload["category_foreground_area_stats"].get(args.category, {})
            if area_stats:
                if fg_area < float(area_stats["valid_low"]):
                    invalid.append("fg_area_low")
                if fg_area > float(area_stats["valid_high"]):
                    invalid.append("fg_area_high")
            label_div = label_diversity_on_fg(fg_label, fg_mask)
            if label_div <= 1:
                invalid.append("low_label_diversity")
            component_stats = mask_component_stats(fg_mask)
            desc = foreground_descriptor(fg_label, fg_mask, bbox_from_mask([[bool(value) for value in row] for row in fg_mask]))
            own_distance, nearest_other_distance, margin = _descriptor_margin(desc["descriptor"], args.category, cache_payload)
            composed_y20 = compose_foreground(fg_mask, fg_label, out["bbox_pred"][index].detach().cpu().tolist(), output_dir=None, canonical_mode=canonical_mode)["composed_y20"]
            grammar_desc = compute_candidate_descriptors(composed_y20, config.get("grammar_bank", {}) if isinstance(config.get("grammar_bank", {}), dict) else {})
            energy = grammar_energy.score(composed_y20, args.category, mode_z=int(out["local_z"][index].item())) if grammar_energy is not None else None
            energy_diag = energy.get("diagnostics", {}) if isinstance(energy, dict) else {}
            label_diag = None
            if bool(args.dump_logit_diagnostics):
                label_diag = _candidate_logit_diagnostics(
                    torch,
                    out["fg_mask_logits"][index, 0],
                    out["fg_label_logits"][index],
                    centroid_fg_mask_prob[index],
                    centroid_label_prob_16_tensor[index],
                    fg_mask,
                    fg_label,
                    composed_y20,
                    float(label_div),
                    float(grammar_desc["dominant_label_ratio"]),
                    mask_thresholds,
                )
            row = {
                "index": index,
                "local_z": int(out["local_z"][index].item()),
                "prior_source": actual_prior_source,
                "prior_mode_id": int(local_z[index]),
                "candidate_index_raw": int(index),
                "fg_mask": fg_mask,
                "fg_label": fg_label,
                "bbox_pred": [float(value) for value in out["bbox_pred"][index].detach().cpu().tolist()],
                "fg_area": fg_area,
                "label_diversity": label_div,
                "num_components": float(component_stats["num_components"]),
                "largest_component_ratio": float(component_stats["largest_component_ratio"]),
                "tiny_component_count": float(component_stats["tiny_component_count"]),
                "is_valid_foreground": len(invalid) == 0,
                "invalid_reasons": invalid,
                "own_category_distance": own_distance,
                "nearest_other_category_distance": nearest_other_distance,
                "category_descriptor_margin": margin,
                "composed_y20": composed_y20,
                "grammar_descriptor": grammar_desc,
                "grammar_diagnostics": {
                    "foreground_area_ratio": float(grammar_desc["foreground_area_ratio"]),
                    "label_diversity": int(grammar_desc["label_diversity"]),
                    "dominant_label_ratio": float(grammar_desc["dominant_label_ratio"]),
                    "num_components": int(grammar_desc["num_connected_components"]),
                    "largest_component_ratio": float(grammar_desc["largest_component_ratio"]),
                    "tiny_island_count": int(grammar_desc["tiny_island_count"]),
                    **energy_diag,
                },
                "grammar_energy": energy,
            }
            if label_diag is not None:
                row["label_diagnostics"] = label_diag
            rows.append(row)
        valid_rows = [row for row in rows if row["is_valid_foreground"]]
        fallback_used = False
        duplicate_skipped_indices: list[int] = []
        collapse_skipped_indices: list[int] = []
        duplicate_fill = False
        collapse_fill = False
        ranking_energy_only: list[int] = []
        selected_topk_indices: list[int] = []
        if rerank_enabled:
            top_k = int(args.top_k or rerank_cf.get("top_k", num_valid))
            ranking_energy_only = [int(rows[i]["index"]) for i in sorted(range(len(rows)), key=lambda i: float((rows[i].get("grammar_energy") or {}).get("total", 0.0)))]
            if diverse_topk:
                selected_topk_indices, duplicate_skipped_indices, collapse_skipped_indices, duplicate_fill, collapse_fill = _select_diverse_topk(
                    rows,
                    ranking_energy_only,
                    top_k,
                    diversity_weight=float(args.diversity_weight),
                    duplicate_mask_iou_threshold=float(args.duplicate_mask_iou_threshold),
                    duplicate_label_hamming_threshold=float(args.duplicate_label_hamming_threshold),
                )
            else:
                selected_topk_indices = ranking_energy_only[:top_k]
            rows_by_index_for_select = {int(row["index"]): row for row in rows}
            selected = [rows_by_index_for_select[idx] for idx in selected_topk_indices if idx in rows_by_index_for_select]
        else:
            selected = valid_rows[:num_valid]
            if len(selected) < int(args.num_candidates):
                fallback_used = True
                selected.extend([row for row in rows if not row["is_valid_foreground"]][: int(args.num_candidates) - len(selected)])
            selected = selected[: int(args.num_candidates)]
    outputs = []
    raw_candidates = []
    for out_index, row in enumerate(selected):
        sample_dir = output_dir / f"candidate_{out_index:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        composed = row["composed_y20"]
        save_binary_map(row["fg_mask"], sample_dir / "fg_mask20.png", scale=12)
        save_label_map(row["fg_label"], sample_dir / "fg_label20.png", scale=12)
        save_label_map(composed, sample_dir / "composed_y20.png", scale=12)
        payload_row = {
            "index": int(row["index"]),
            "rank": int(out_index),
            "category": args.category,
            "local_z": row["local_z"],
            "prior_source": row.get("prior_source", actual_prior_source),
            "prior_mode_id": row.get("prior_mode_id", row["local_z"]),
            "candidate_index_raw": int(row.get("candidate_index_raw", row["index"])),
            "candidate_index_selected": int(out_index),
            "bbox_pred": row["bbox_pred"],
            "fg_area": row["fg_area"],
            "label_diversity": row["label_diversity"],
            "num_components": row["num_components"],
            "largest_component_ratio": row["largest_component_ratio"],
            "tiny_component_count": row["tiny_component_count"],
            "is_valid_foreground": row["is_valid_foreground"],
            "invalid_reasons": row["invalid_reasons"],
            "own_category_distance": row["own_category_distance"],
            "nearest_other_category_distance": row["nearest_other_category_distance"],
            "category_descriptor_margin": row["category_descriptor_margin"],
            "grammar_diagnostics": row["grammar_diagnostics"],
            "composed_y20": composed,
            "sample_dir": str(sample_dir),
            "grammar_energy": row.get("grammar_energy"),
        }
        if args.dump_logit_diagnostics and "label_diagnostics" in row:
            payload_row["label_diagnostics"] = row["label_diagnostics"]
        save_json(sample_dir / "meta.json", payload_row)
        outputs.append(payload_row)
        print_progress("sample-fg", out_index + 1, len(selected), f"z={row['local_z']} area={row['fg_area']:.4f}")
    if args.save_raw_candidates:
        raw_candidates = []
        for row in rows:
            raw_candidates.append({
                "index": row["index"],
                "local_z": row["local_z"],
                "prior_source": row.get("prior_source", actual_prior_source),
                "prior_mode_id": row.get("prior_mode_id", row["local_z"]),
                "candidate_index_raw": int(row.get("candidate_index_raw", row["index"])),
                "is_valid_foreground": row["is_valid_foreground"],
                "invalid_reasons": row["invalid_reasons"],
                "fg_area": row["fg_area"],
                "label_diversity": row["label_diversity"],
                "largest_component_ratio": row["largest_component_ratio"],
                "tiny_component_count": row["tiny_component_count"],
                "grammar_energy": row.get("grammar_energy"),
                "composed_y20": row["composed_y20"],
            })
            if args.dump_logit_diagnostics and "label_diagnostics" in row:
                raw_candidates[-1]["label_diagnostics"] = row["label_diagnostics"]
    finish_progress()
    ranking = ranking_energy_only if rerank_enabled else [int(row.get("index", i)) for i, row in enumerate(outputs)]
    summary = {
        "category": args.category,
        "num_candidates": len(outputs),
        "planner_oversample": oversample,
        "num_valid_plans": len(valid_rows),
        "fallback_used": fallback_used,
        "requested_prior_source": requested_prior_source,
        "actual_prior_source": actual_prior_source,
        "fallback_if_missing": fallback_if_missing,
        "fallback_reason": fallback_reason,
        "prior_source": actual_prior_source,
        "mode_ids_oversample": [int(value) for value in local_z],
        "mode_ids_selected": [int(row.get("prior_mode_id", row["local_z"])) for row in outputs],
        "prior_mode_strategy": prior_mode_strategy,
        "mode_prior_key": mode_prior_key if actual_prior_source == "instruction_matrix_grammar" else "model_local_z",
        "label_prob_key": label_prob_key if actual_prior_source == "instruction_matrix_grammar" else "centroid_label_prob_16",
        "fallback_count": fallback_count,
        "warning": ("valid candidates fewer than requested; included invalid fallback samples" if fallback_used else ""),
        "rerank_enabled": rerank_enabled,
        "top_k": int(args.top_k or rerank_cf.get("top_k", num_valid)),
        "weights": rerank_cf.get("weights", {}),
        "samples": outputs,
        "ranking": ranking,
        "selected_topk_indices": selected_topk_indices if rerank_enabled else ranking,
    }
    save_json(output_dir / "candidates.json", summary)
    if args.dump_logit_diagnostics:
        label_diag_rows = [
            {
                "index": int(row["index"]),
                "rank": int(index),
                "category": args.category,
                "local_z": int(row["local_z"]),
                "prior_source": row.get("prior_source", actual_prior_source),
                "prior_mode_id": int(row.get("prior_mode_id", row["local_z"])),
                "is_valid_foreground": bool(row["is_valid_foreground"]),
                "invalid_reasons": row["invalid_reasons"],
                "fg_area": float(row["fg_area"]),
                "label_diagnostics": row.get("label_diagnostics", {}),
            }
            for index, row in enumerate(rows)
        ]
        label_diag_summary = _label_diagnostics_summary(rows, mask_thresholds)
        save_json(output_dir / "label_diagnostics.json", {
            "category": args.category,
            "requested_prior_source": requested_prior_source,
            "actual_prior_source": actual_prior_source,
            "summary": label_diag_summary,
        })
        save_jsonl(output_dir / "per_candidate_label_diagnostics.jsonl", label_diag_rows)
        save_json(output_dir / "mask_threshold_sweep.json", {
            "category": args.category,
            "mask_thresholds": mask_thresholds,
            "foreground_area_mean_by_threshold": label_diag_summary["foreground_area_mean_by_threshold"],
            "candidates": [
                {
                    "index": int(row["index"]),
                    "local_z": int(row["local_z"]),
                    "prior_mode_id": int(row.get("prior_mode_id", row["local_z"])),
                    "fg_area": float(row["fg_area"]),
                    "foreground_area_by_threshold": (row.get("label_diagnostics", {}).get("mask", {}).get("foreground_area_by_threshold", {}) if isinstance(row.get("label_diagnostics"), dict) else {}),
                    "warnings": (row.get("label_diagnostics", {}).get("model_prior_comparison", {}).get("warnings", []) if isinstance(row.get("label_diagnostics"), dict) else []),
                }
                for row in rows
            ],
        })
    save_json(output_dir / "prior_mode_ids.json", {
        "category": args.category,
        "requested_prior_source": requested_prior_source,
        "actual_prior_source": actual_prior_source,
        "prior_mode_strategy": prior_mode_strategy,
        "mode_ids_oversample": [int(value) for value in local_z],
        "mode_ids_selected": [int(row.get("prior_mode_id", row["local_z"])) for row in outputs],
        "num_oversample": int(len(local_z)),
        "num_selected": int(len(outputs)),
        "note": "mode_ids_oversample correspond to pre-rerank/pre-selection candidates; mode_ids_selected correspond to final saved candidates when available.",
    })
    prior_debug.update({
        "category": args.category,
        "requested_prior_source": requested_prior_source,
        "actual_prior_source": actual_prior_source,
        "fallback_if_missing": fallback_if_missing,
        "fallback_reason": fallback_reason,
        "prior_mode_strategy": prior_mode_strategy,
        "mode_prior_key": mode_prior_key if actual_prior_source == "instruction_matrix_grammar" else "model_local_z",
        "label_prob_key": label_prob_key if actual_prior_source == "instruction_matrix_grammar" else "centroid_label_prob_16",
        "fallback_count": fallback_count,
        "mode_id_histogram": {str(mode_id): [int(value) for value in local_z].count(mode_id) for mode_id in sorted(set(int(value) for value in local_z))},
    })
    save_json(output_dir / "prior_debug.json", prior_debug)
    if rerank_enabled:
        rows_by_index = {int(row["index"]): row for row in rows}
        energy_top_rows = [rows_by_index[i] for i in ranking_energy_only[: int(args.top_k or rerank_cf.get("top_k", num_valid))] if i in rows_by_index]
        selected_top_rows = [rows_by_index[i] for i in selected_topk_indices if i in rows_by_index]
        raw_mean = _summary_means(rows)
        energy_topk_mean = _summary_means(energy_top_rows)
        selected_topk_mean = _summary_means(selected_top_rows)
        breakdown = {
            "category": args.category,
            "num_candidates": len(rows),
            "top_k": int(args.top_k or rerank_cf.get("top_k", num_valid)),
            "weights": rerank_cf.get("weights", {}),
            "diverse_topk": diverse_topk,
            "diversity_weight": float(args.diversity_weight),
            "duplicate_mask_iou_threshold": float(args.duplicate_mask_iou_threshold),
            "duplicate_label_hamming_threshold": float(args.duplicate_label_hamming_threshold),
            "summary": {
                "raw_mean": raw_mean,
                "energy_topk_mean": energy_topk_mean,
                "selected_topk_mean": selected_topk_mean,
                "improvement": _summary_improvement(selected_topk_mean, raw_mean),
            },
            "candidates": [
                {
                    "index": row["index"],
                    "local_z": row["local_z"],
                    "valid": bool((row.get("grammar_energy") or {}).get("valid", row["is_valid_foreground"])),
                    "invalid_reasons": (row.get("grammar_energy") or {}).get("invalid_reasons", row["invalid_reasons"]),
                    "energy": row.get("grammar_energy"),
                    "diagnostics": {
                        **row["grammar_diagnostics"],
                    },
                }
                for row in rows
            ],
            "ranking": ranking,
            "ranking_energy_only": ranking_energy_only,
            "selected_topk_indices": selected_topk_indices,
            "duplicate_skipped_indices": duplicate_skipped_indices,
            "collapse_skipped_indices": collapse_skipped_indices,
            "duplicate_fill": duplicate_fill,
            "collapse_fill": collapse_fill,
        }
        save_json(output_dir / "energy_breakdown.json", breakdown)
    if rerank_enabled and outputs:
        save_label_grid_mosaic([row["composed_y20"] for row in rows], output_dir / "raw_candidates_y20.png", columns=8, scale=8)
        save_label_grid_mosaic([row["composed_y20"] for row in outputs], output_dir / "reranked_topk_y20.png", columns=min(4, len(outputs)), scale=12)
    if args.save_raw_candidates:
        save_json(output_dir / "raw_candidates.json", raw_candidates)
    save_jsonl(output_dir / "per_sample.jsonl", outputs)
    print(format_metric_line("sample-foreground:", [("category", args.category), ("num_candidates", len(outputs)), ("valid_plans", len(valid_rows)), ("rerank", rerank_enabled)]))
    if rerank_enabled:
        raw_valid = rows
        rows_by_index = {int(row["index"]): row for row in rows}
        top_rows = [rows_by_index[i] for i in selected_topk_indices if i in rows_by_index]
        print(format_metric_line("grammar-rerank:", [
            ("best_total_energy", float(rows_by_index[ranking[0]]["grammar_energy"]["total"]) if ranking and ranking[0] in rows_by_index else 0.0),
            ("raw_valid_count", len(raw_valid)),
            ("reranked_valid_count", sum(1 for row in top_rows if row["is_valid_foreground"])),
            ("mean_dom_raw", sum(float(row["grammar_diagnostics"]["dominant_label_ratio"]) for row in raw_valid) / float(max(1, len(raw_valid)))),
            ("mean_dom_topk", sum(float(row["grammar_diagnostics"]["dominant_label_ratio"]) for row in top_rows) / float(max(1, len(top_rows)))),
            ("mean_div_raw", _mean(raw_valid, "label_diversity")),
            ("mean_div_topk", _mean(top_rows, "label_diversity")),
            ("mean_trans_raw", _mean_energy(raw_valid, "trans")),
            ("mean_trans_topk", _mean_energy(top_rows, "trans")),
            ("mean_motif_raw", _mean_energy(raw_valid, "motif")),
            ("mean_motif_topk", _mean_energy(top_rows, "motif")),
            ("mean_div_energy_raw", _mean_energy(raw_valid, "div")),
            ("mean_div_energy_topk", _mean_energy(top_rows, "div")),
            ("mean_dom_energy_raw", _mean_energy(raw_valid, "dom")),
            ("mean_dom_energy_topk", _mean_energy(top_rows, "dom")),
            ("mean_single_label_raw", _mean_energy(raw_valid, "single_label_collapse")),
            ("mean_single_label_topk", _mean_energy(top_rows, "single_label_collapse")),
        ]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
