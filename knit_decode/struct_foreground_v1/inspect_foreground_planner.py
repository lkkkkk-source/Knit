from __future__ import annotations

import argparse
from pathlib import Path

from .compose_foreground import compose_foreground
from .models.foreground_planner import ForegroundCanonicalPlanner
from .utils import bbox_from_mask, checkpoint_get, finish_progress, format_metric_line, foreground_area, foreground_descriptor, label_diversity_on_fg, load_config, normalized_l2_between, print_progress, require_foreground_cache_fields, save_binary_map, save_json, save_jsonl, save_label_grid_mosaic, save_label_map


def _require_torch() -> object:
    import importlib

    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required for inspect_foreground_planner.") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect foreground planner outputs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _checkpoint_dim(payload: dict[str, object], metrics: dict[str, object], metric_key: str, state_key: str) -> int:
    if metric_key in metrics:
        return int(metrics[metric_key])
    weight = payload["model_state_dict"].get(state_key)
    if weight is None:
        raise KeyError(f"Missing {metric_key} and state key {state_key} in checkpoint.")
    return int(weight.shape[0])


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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    payload = _require_torch().load(args.checkpoint, map_location="cpu")
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
    output_dir = Path(args.output_dir or (args.checkpoint.parent / "inspect" / args.category))
    output_dir.mkdir(parents=True, exist_ok=True)
    category_id = int(category_to_id[args.category])
    _ = checkpoint_get(payload, "descriptor_slices")
    _ = checkpoint_get(payload, "descriptor_global_mean")
    _ = checkpoint_get(payload, "descriptor_global_std")
    _ = checkpoint_get(payload, "category_foreground_area_stats")
    train_cache_path = Path(str(checkpoint_get(payload, "train_cache_path")))
    cache_payload = _require_torch().load(train_cache_path, map_location="cpu")
    require_foreground_cache_fields(cache_payload, context="Foreground train cache")
    if args.category not in cache_payload["category_foreground_area_stats"]:
        raise ValueError(f"Foreground train cache is missing category_foreground_area_stats for category {args.category!r}.")
    area_stats = cache_payload["category_foreground_area_stats"][args.category]
    category_to_num_modes = checkpoint_get(payload, "category_to_num_modes")
    if args.category not in category_to_num_modes:
        raise ValueError(
            f"category not available in trained foreground prior: {args.category}\n"
            f"available categories: {sorted(train_categories)}"
        )
    grammar_dim = _checkpoint_dim(payload, metrics, "grammar_signature_dim", "grammar_head.weight")
    adjacency_dim = _checkpoint_dim(payload, metrics, "adjacency_signature_dim", "adj_head.weight")
    bbox_dim = _checkpoint_dim(payload, metrics, "bbox_dim", "bbox_head.weight")
    max_num_modes = int(checkpoint_get(payload, "max_num_modes"))
    model = ForegroundCanonicalPlanner(
        num_categories=len(category_to_id),
        max_num_modes=max_num_modes,
        hidden_dim=int(config["planner"]["hidden_dim"]),
        category_embed_dim=int(config["planner"]["category_embed_dim"]),
        mode_embed_dim=int(config["planner"]["mode_embed_dim"]),
        grammar_dim=grammar_dim,
        adjacency_dim=adjacency_dim,
        bbox_dim=bbox_dim,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    torch = _require_torch()
    category_ids = getattr(torch, "full")((int(args.num_samples),), category_id, dtype=getattr(torch, "long"))
    rows: list[dict[str, object]] = []
    planner_grids: list[list[list[int]]] = []
    with getattr(torch, "no_grad")():
        num_modes = int(category_to_num_modes[args.category])
        centroid_source = cache_payload["centroid_sketch_by_category"].get(args.category, {})
        centroid_label_hist = []
        centroid_row_projection = []
        centroid_col_projection = []
        centroid_adjacency = []
        centroid_transition_stats = []
        centroid_bbox_stats = []
        mode_mask = []
        local_z = []
        for index in range(int(args.num_samples)):
            mode_index = index % max(1, num_modes)
            local_z.append(mode_index)
            mode_mask.append([1 if k < num_modes else 0 for k in range(max_num_modes)])
            centroid = centroid_source.get(mode_index)
            if centroid is None:
                raise ValueError(f"Missing centroid sketch for category={args.category!r} local_z={mode_index}.")
            if "centroid_fg_mask_prob" not in centroid:
                raise ValueError("Foreground cache is missing centroid_fg_mask_prob. Please rebuild the cache with the current build_foreground_cache.py.")
            centroid_label_hist.append(centroid["centroid_label_hist"])
            centroid_row_projection.append(centroid["centroid_row_projection"])
            centroid_col_projection.append(centroid["centroid_col_projection"])
            centroid_adjacency.append(centroid["centroid_adjacency"])
            centroid_transition_stats.append(centroid["centroid_transition_stats"])
            centroid_bbox_stats.append(centroid["centroid_bbox_stats"])
        out = model(
            category_ids,
            getattr(torch, "tensor")(centroid_label_hist, dtype=getattr(torch, "float32")),
            getattr(torch, "tensor")(centroid_row_projection, dtype=getattr(torch, "float32")),
            getattr(torch, "tensor")(centroid_col_projection, dtype=getattr(torch, "float32")),
            getattr(torch, "tensor")(centroid_adjacency, dtype=getattr(torch, "float32")),
            getattr(torch, "tensor")(centroid_transition_stats, dtype=getattr(torch, "float32")),
            getattr(torch, "tensor")(centroid_bbox_stats, dtype=getattr(torch, "float32")),
            getattr(torch, "tensor")(local_z, dtype=getattr(torch, "long")),
            getattr(torch, "tensor")(mode_mask, dtype=getattr(torch, "bool")),
        )
        probs = getattr(__import__("importlib").import_module("torch.nn.functional"), "softmax")(out["local_z_logits"], dim=-1)
        for index in range(int(args.num_samples)):
            sample_dir = output_dir / f"planner_{index:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            fg_mask = (getattr(torch, "sigmoid")(out["fg_mask_logits"][index, 0]) >= 0.5).to(dtype=getattr(torch, "long")).detach().cpu().tolist()
            fg_label = (out["fg_label_logits"][index].argmax(dim=0) + 1).detach().cpu().tolist()
            bbox_pred = [float(value) for value in out["bbox_pred"][index].detach().cpu().tolist()]
            composed = compose_foreground(fg_mask, fg_label, bbox_pred, sample_dir)["composed_y20"]
            save_binary_map(fg_mask, sample_dir / "fg_mask20.png", scale=12)
            save_label_map(fg_label, sample_dir / "fg_label20.png", scale=12)
            save_label_map(composed, sample_dir / "composed_y20.png", scale=12)
            desc = foreground_descriptor(fg_label, fg_mask, bbox_from_mask([[bool(value) for value in row] for row in fg_mask]))
            own_distance, nearest_other_distance, margin = _descriptor_margin(desc["descriptor"], args.category, cache_payload)
            fg_area = foreground_area(fg_mask)
            label_div = label_diversity_on_fg(fg_label, fg_mask)
            invalid = []
            if not any(any(v for v in row_fg) for row_fg in fg_mask):
                invalid.append("empty_foreground")
            if fg_area >= 0.99:
                invalid.append("full_foreground")
            if fg_area < float(area_stats["valid_low"]):
                invalid.append("fg_area_low")
            if fg_area > float(area_stats["valid_high"]):
                invalid.append("fg_area_high")
            if label_div <= 1:
                invalid.append("low_label_diversity")
            row = {
                "category": args.category,
                "sample_index": index,
                "local_z": int(out["local_z"][index].item()),
                "local_z_prob": float(probs[index].max().item()),
                "bbox_pred": bbox_pred,
                "is_valid_foreground": len(invalid) == 0,
                "mean_fg_area": fg_area,
                "label_diversity": label_div,
                "mean_vertical_continuity": float(desc["vertical_continuity"]),
                "mean_horizontal_continuity": float(desc["horizontal_continuity"]),
                "mean_center_band_score": float(desc["center_band_score"]),
                "mean_symmetry_score": float(desc["symmetry_score"]),
                "mean_stripe_score": float(desc["stripe_score"][0]),
                "own_category_distance": own_distance,
                "nearest_other_category_distance": nearest_other_distance,
                "category_descriptor_margin": margin,
                "invalid_reasons": invalid,
                "sample_dir": str(sample_dir),
            }
            save_json(sample_dir / "sample.json", row)
            rows.append(row)
            planner_grids.append(composed)
            print_progress("inspect-fg", index + 1, int(args.num_samples), f"z={row['local_z']} prob={row['local_z_prob']:.4f}")
    finish_progress()
    if planner_grids:
        save_label_grid_mosaic(planner_grids, output_dir / "planner_grid.png", columns=max(1, min(4, int(args.num_samples))), scale=12)
    summary = {
        "valid_foreground_rate": sum(1.0 if bool(row["is_valid_foreground"]) else 0.0 for row in rows) / float(max(1, len(rows))),
        "empty_foreground_rate": sum(1.0 for row in rows if row["mean_fg_area"] <= 0.0) / float(max(1, len(rows))),
        "full_foreground_rate": sum(1.0 for row in rows if row["mean_fg_area"] >= 0.99) / float(max(1, len(rows))),
        "mean_fg_area": sum(float(row["mean_fg_area"]) for row in rows) / float(max(1, len(rows))),
        "min_fg_area": min((float(row["mean_fg_area"]) for row in rows), default=0.0),
        "max_fg_area": max((float(row["mean_fg_area"]) for row in rows), default=0.0),
        "unique_local_z_count": len({int(row["local_z"]) for row in rows}),
        "effective_local_modes": len({int(row["local_z"]) for row in rows}),
        "foreground_label_diversity": sum(float(row["label_diversity"]) for row in rows) / float(max(1, len(rows))),
        "unique_foreground_layout_count": len({str(row["bbox_pred"]) for row in rows}),
        "mean_vertical_continuity": sum(float(row["mean_vertical_continuity"]) for row in rows) / float(max(1, len(rows))),
        "mean_horizontal_continuity": sum(float(row["mean_horizontal_continuity"]) for row in rows) / float(max(1, len(rows))),
        "mean_center_band_score": sum(float(row["mean_center_band_score"]) for row in rows) / float(max(1, len(rows))),
        "mean_symmetry_score": sum(float(row["mean_symmetry_score"]) for row in rows) / float(max(1, len(rows))),
        "mean_stripe_score": sum(float(row["mean_stripe_score"]) for row in rows) / float(max(1, len(rows))),
        "own_category_distance": sum(float(row["own_category_distance"]) for row in rows) / float(max(1, len(rows))),
        "nearest_other_category_distance": sum(float(row["nearest_other_category_distance"]) for row in rows) / float(max(1, len(rows))),
        "category_descriptor_margin": sum(float(row["category_descriptor_margin"]) for row in rows) / float(max(1, len(rows))),
        "invalid_reason_counts": {
            reason: sum(1 for row in rows if reason in row["invalid_reasons"])
            for reason in ["empty_foreground", "full_foreground", "fg_area_low", "fg_area_high", "low_label_diversity"]
        },
    }
    save_json(output_dir / "summary.json", summary)
    save_jsonl(output_dir / "per_sample.jsonl", rows)
    print(format_metric_line("inspect-foreground:", [("category", args.category), ("num_samples", int(args.num_samples)), ("valid_fg", summary["valid_foreground_rate"])]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
