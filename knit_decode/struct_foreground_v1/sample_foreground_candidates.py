from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compose_foreground import compose_foreground
from .grammar_energy import GrammarEnergy, compute_candidate_descriptors
from .inspect_foreground_planner import _require_torch
from .utils import bbox_from_mask, build_planner_from_checkpoint_payload, checkpoint_get, finish_progress, foreground_area, foreground_descriptor, format_metric_line, label_diversity_on_fg, load_config, mask_component_stats, normalized_l2_between, print_progress, require_centroid_sketch_fields, require_foreground_cache_fields, resolve_canonical_mode, save_binary_map, save_json, save_jsonl, save_label_grid_mosaic, save_label_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample category-only foreground candidates.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--num-candidates", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--use-grammar-rerank", action="store_true")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--save-raw-candidates", action="store_true")
    parser.add_argument("--energy-config", type=Path, default=None)
    parser.add_argument("--diverse-topk", dest="diverse_topk", action="store_true", default=None)
    parser.add_argument("--no-diverse-topk", dest="diverse_topk", action="store_false")
    parser.add_argument("--diversity-weight", type=float, default=0.25)
    parser.add_argument("--duplicate-mask-iou-threshold", type=float, default=0.95)
    parser.add_argument("--duplicate-label-hamming-threshold", type=float, default=0.05)
    return parser


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
        }
    return {
        "dominant_label_ratio": sum(float(row["grammar_diagnostics"]["dominant_label_ratio"]) for row in rows) / float(len(rows)),
        "label_diversity": sum(float(row["grammar_diagnostics"]["label_diversity"]) for row in rows) / float(len(rows)),
        "total": _mean_energy(rows, "total"),
        "trans": _mean_energy(rows, "trans"),
        "motif": _mean_energy(rows, "motif"),
        "div": _mean_energy(rows, "div"),
        "dom": _mean_energy(rows, "dom"),
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


def _select_diverse_topk(
    rows: list[dict[str, object]],
    energy_order: list[int],
    top_k: int,
    *,
    diversity_weight: float,
    duplicate_mask_iou_threshold: float,
    duplicate_label_hamming_threshold: float,
) -> tuple[list[int], list[int], bool]:
    rows_by_index = {int(row["index"]): row for row in rows}
    selected: list[int] = []
    skipped: list[int] = []
    remaining = list(energy_order)
    while remaining and len(selected) < top_k:
        best_index = None
        best_score = float("inf")
        for idx in remaining:
            row = rows_by_index[idx]
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
    duplicate_fill = False
    if len(selected) < top_k:
        duplicate_fill = True
        for idx in energy_order:
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
            if len(selected) >= top_k:
                break
    return selected[:top_k], skipped, duplicate_fill


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
    model, model_kwargs, load_debug = build_planner_from_checkpoint_payload(payload, config)
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
    category_ids = getattr(torch, "full")((oversample,), int(category_to_id[args.category]), dtype=getattr(torch, "long"))
    rows = []
    with getattr(torch, "no_grad")():
        num_modes = int(category_to_num_modes[args.category])
        centroid_source = cache_payload["centroid_sketch_by_category"].get(args.category, {})
        max_num_modes = int(model_kwargs["max_num_modes"])
        mode_mask = getattr(torch, "tensor")([[1 if k < num_modes else 0 for k in range(max_num_modes)] for _ in range(oversample)], dtype=getattr(torch, "bool"))
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
        out = model(
            category_ids,
            getattr(torch, "stack")([centroid_source[int(z)]["centroid_fg_mask_prob"] for z in local_z]).to(dtype=getattr(torch, "float32")),
            getattr(torch, "stack")(centroid_label_prob_16).to(dtype=getattr(torch, "float32")),
            getattr(torch, "tensor")(centroid_label_hist, dtype=getattr(torch, "float32")),
            getattr(torch, "tensor")(centroid_row_projection, dtype=getattr(torch, "float32")),
            getattr(torch, "tensor")(centroid_col_projection, dtype=getattr(torch, "float32")),
            getattr(torch, "tensor")(centroid_adjacency, dtype=getattr(torch, "float32")),
            getattr(torch, "tensor")(centroid_transition_stats, dtype=getattr(torch, "float32")),
            getattr(torch, "tensor")(centroid_bbox_stats, dtype=getattr(torch, "float32")),
            getattr(torch, "tensor")(local_z, dtype=getattr(torch, "long")),
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
            rows.append({
                "index": index,
                "local_z": int(out["local_z"][index].item()),
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
            })
        valid_rows = [row for row in rows if row["is_valid_foreground"]]
        fallback_used = False
        duplicate_skipped_indices: list[int] = []
        duplicate_fill = False
        ranking_energy_only: list[int] = []
        selected_topk_indices: list[int] = []
        if rerank_enabled:
            top_k = int(args.top_k or rerank_cf.get("top_k", num_valid))
            ranking_energy_only = [int(rows[i]["index"]) for i in sorted(range(len(rows)), key=lambda i: float((rows[i].get("grammar_energy") or {}).get("total", 0.0)))]
            if diverse_topk:
                selected_topk_indices, duplicate_skipped_indices, duplicate_fill = _select_diverse_topk(
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
        save_json(sample_dir / "meta.json", payload_row)
        outputs.append(payload_row)
        print_progress("sample-fg", out_index + 1, len(selected), f"z={row['local_z']} area={row['fg_area']:.4f}")
    if args.save_raw_candidates:
        raw_candidates = []
        for row in rows:
            raw_candidates.append({
                "index": row["index"],
                "local_z": row["local_z"],
                "is_valid_foreground": row["is_valid_foreground"],
                "invalid_reasons": row["invalid_reasons"],
                "fg_area": row["fg_area"],
                "label_diversity": row["label_diversity"],
                "largest_component_ratio": row["largest_component_ratio"],
                "tiny_component_count": row["tiny_component_count"],
                "grammar_energy": row.get("grammar_energy"),
                "composed_y20": row["composed_y20"],
            })
    finish_progress()
    ranking = ranking_energy_only if rerank_enabled else [int(row.get("index", i)) for i, row in enumerate(outputs)]
    summary = {
        "category": args.category,
        "num_candidates": len(outputs),
        "planner_oversample": oversample,
        "num_valid_plans": len(valid_rows),
        "fallback_used": fallback_used,
        "warning": ("valid candidates fewer than requested; included invalid fallback samples" if fallback_used else ""),
        "rerank_enabled": rerank_enabled,
        "top_k": int(args.top_k or rerank_cf.get("top_k", num_valid)),
        "weights": rerank_cf.get("weights", {}),
        "samples": outputs,
        "ranking": ranking,
        "selected_topk_indices": selected_topk_indices if rerank_enabled else ranking,
    }
    save_json(output_dir / "candidates.json", summary)
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
            "duplicate_fill": duplicate_fill,
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
        ]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
