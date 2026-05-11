from __future__ import annotations

import argparse
from pathlib import Path

from .compose_foreground import compose_foreground
from .inspect_foreground_planner import _require_torch
from .models.foreground_planner import ForegroundCanonicalPlanner
from .utils import bbox_from_mask, checkpoint_get, finish_progress, foreground_area, foreground_descriptor, format_metric_line, label_diversity_on_fg, load_config, normalized_l2_between, print_progress, require_foreground_cache_fields, save_binary_map, save_json, save_jsonl, save_label_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample category-only foreground candidates.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--num-candidates", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=None)
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
    output_dir = Path(args.output_dir or (args.checkpoint.parent / "samples" / args.category))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch = _require_torch()
    _ = checkpoint_get(payload, "descriptor_slices")
    _ = checkpoint_get(payload, "descriptor_global_mean")
    _ = checkpoint_get(payload, "descriptor_global_std")
    _ = checkpoint_get(payload, "category_foreground_area_stats")
    cache_payload = torch.load(Path(str(checkpoint_get(payload, "train_cache_path"))), map_location="cpu")
    require_foreground_cache_fields(cache_payload, context="Foreground train cache")
    if args.category not in cache_payload["category_foreground_area_stats"]:
        raise ValueError(f"Foreground train cache is missing category_foreground_area_stats for category {args.category!r}.")
    category_to_num_modes = checkpoint_get(payload, "category_to_num_modes")
    if args.category not in category_to_num_modes:
        raise ValueError(
            f"category not available in trained foreground prior: {args.category}\n"
            f"available categories: {sorted(train_categories)}"
        )
    grammar_dim = int(metrics.get("grammar_signature_dim", 17))
    adjacency_dim = int(metrics.get("adjacency_signature_dim", 256))
    bbox_dim = int(metrics.get("bbox_dim", 10))
    model = ForegroundCanonicalPlanner(
        num_categories=len(category_to_id),
        max_num_modes=int(config["planner"]["num_modes_per_category"]),
        hidden_dim=int(config["planner"]["hidden_dim"]),
        category_embed_dim=int(config["planner"]["category_embed_dim"]),
        mode_embed_dim=int(config["planner"]["mode_embed_dim"]),
        grammar_dim=grammar_dim,
        adjacency_dim=adjacency_dim,
        bbox_dim=bbox_dim,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    oversample = int(config["sampling"]["planner_oversample"])
    num_valid = int(config["sampling"]["num_valid_plans"])
    category_ids = getattr(torch, "full")((oversample,), int(category_to_id[args.category]), dtype=getattr(torch, "long"))
    rows = []
    with getattr(torch, "no_grad")():
        num_modes = int(category_to_num_modes[args.category])
        centroid_source = cache_payload["centroid_sketch_by_category"].get(args.category, {})
        max_num_modes = int(config["planner"]["num_modes_per_category"])
        mode_mask = getattr(torch, "tensor")([[1 if k < num_modes else 0 for k in range(max_num_modes)] for _ in range(oversample)], dtype=getattr(torch, "bool"))
        category_embed = model.category_embed(category_ids)
        z_logits = model.local_z_head(category_embed).masked_fill(mode_mask.logical_not(), float("-inf"))
        z_probs = getattr(__import__("importlib").import_module("torch.nn.functional"), "softmax")(z_logits, dim=-1)
        sampled_local_z = getattr(torch, "multinomial")(z_probs, num_samples=1).squeeze(1)
        centroid_label_hist = []
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
            desc = foreground_descriptor(fg_label, fg_mask, bbox_from_mask([[bool(value) for value in row] for row in fg_mask]))
            own_distance, nearest_other_distance, margin = _descriptor_margin(desc["descriptor"], args.category, cache_payload)
            rows.append({
                "index": index,
                "local_z": int(out["local_z"][index].item()),
                "fg_mask": fg_mask,
                "fg_label": fg_label,
                "bbox_pred": [float(value) for value in out["bbox_pred"][index].detach().cpu().tolist()],
                "fg_area": fg_area,
                "label_diversity": label_div,
                "is_valid_foreground": len(invalid) == 0,
                "invalid_reasons": invalid,
                "own_category_distance": own_distance,
                "nearest_other_category_distance": nearest_other_distance,
                "category_descriptor_margin": margin,
            })
        valid_rows = [row for row in rows if row["is_valid_foreground"]]
        selected = valid_rows[:num_valid]
        fallback_used = False
        if len(selected) < int(args.num_candidates):
            fallback_used = True
            selected.extend([row for row in rows if not row["is_valid_foreground"]][: int(args.num_candidates) - len(selected)])
        selected = selected[: int(args.num_candidates)]
    outputs = []
    for out_index, row in enumerate(selected):
        sample_dir = output_dir / f"candidate_{out_index:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        composed = compose_foreground(row["fg_mask"], row["fg_label"], row["bbox_pred"], sample_dir)["composed_y20"]
        save_binary_map(row["fg_mask"], sample_dir / "fg_mask20.png", scale=12)
        save_label_map(row["fg_label"], sample_dir / "fg_label20.png", scale=12)
        save_label_map(composed, sample_dir / "composed_y20.png", scale=12)
        payload_row = {
            "category": args.category,
            "local_z": row["local_z"],
            "bbox_pred": row["bbox_pred"],
            "fg_area": row["fg_area"],
            "label_diversity": row["label_diversity"],
            "is_valid_foreground": row["is_valid_foreground"],
            "invalid_reasons": row["invalid_reasons"],
            "own_category_distance": row["own_category_distance"],
            "nearest_other_category_distance": row["nearest_other_category_distance"],
            "category_descriptor_margin": row["category_descriptor_margin"],
            "composed_y20": composed,
            "sample_dir": str(sample_dir),
        }
        save_json(sample_dir / "meta.json", payload_row)
        outputs.append(payload_row)
        print_progress("sample-fg", out_index + 1, len(selected), f"z={row['local_z']} area={row['fg_area']:.4f}")
    finish_progress()
    save_json(output_dir / "candidates.json", {"category": args.category, "num_candidates": len(outputs), "planner_oversample": oversample, "num_valid_plans": len(valid_rows), "fallback_used": fallback_used, "warning": ("valid candidates fewer than requested; included invalid fallback samples" if fallback_used else ""), "samples": outputs})
    save_jsonl(output_dir / "per_sample.jsonl", outputs)
    print(format_metric_line("sample-foreground:", [("category", args.category), ("num_candidates", len(outputs)), ("valid_plans", len(valid_rows))]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
