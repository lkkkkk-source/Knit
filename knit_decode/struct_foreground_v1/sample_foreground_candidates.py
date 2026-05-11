from __future__ import annotations

import argparse
from pathlib import Path

from .compose_foreground import compose_foreground
from .inspect_foreground_planner import _require_torch
from .models.foreground_planner import ForegroundCanonicalPlanner
from .utils import finish_progress, foreground_descriptor, format_metric_line, load_config, normalized_l2, print_progress, save_binary_map, save_json, save_jsonl, save_label_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample category-only foreground candidates.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--num-candidates", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _descriptor_margin(descriptor: list[float], category: str, cache_payload: dict[str, object]) -> tuple[float, float, float]:
    if "descriptor_mean_by_category" not in cache_payload or "descriptor_std_by_category" not in cache_payload:
        raise ValueError(
            "Foreground cache is missing descriptor_mean_by_category / descriptor_std_by_category. "
            "Please rebuild the train foreground cache with the current build_foreground_cache.py."
        )
    own_dist = normalized_l2(
        descriptor,
        cache_payload["descriptor_mean_by_category"].get(category, []),
        cache_payload["descriptor_std_by_category"].get(category, []),
    )
    nearest_other = float("inf")
    for other_category, mean in cache_payload["descriptor_mean_by_category"].items():
        if other_category == category:
            continue
        value = normalized_l2(descriptor, mean, cache_payload["descriptor_std_by_category"][other_category])
        nearest_other = min(nearest_other, value)
    margin = own_dist - nearest_other if nearest_other < float("inf") else own_dist
    return own_dist, nearest_other, margin


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    payload = _require_torch().load(args.checkpoint, map_location="cpu")
    metrics = payload["metrics"]
    category_to_id = metrics["category_to_id"]
    if args.category not in category_to_id:
        raise KeyError(f"Unknown category {args.category!r}. Available categories: {sorted(category_to_id)}")
    output_dir = Path(args.output_dir or (args.checkpoint.parent / "samples" / args.category))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch = _require_torch()
    cache_payload = torch.load(Path(config["data"]["cache_dir"]) / "foreground_cache_train.pt", map_location="cpu")
    if "descriptor_global_mean" not in cache_payload or "descriptor_global_std" not in cache_payload:
        raise ValueError(
            "Foreground train cache is missing descriptor_global_mean / descriptor_global_std. "
            "Please rebuild the train foreground cache with the current build_foreground_cache.py."
        )
    if "category_foreground_area_stats" not in cache_payload:
        raise ValueError(
            "Foreground cache is missing category_foreground_area_stats. "
            "Please rebuild the train foreground cache with the current build_foreground_cache.py."
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
        dummy = getattr(torch, "zeros")((oversample, 16), dtype=getattr(torch, "float32"))
        out = model(category_ids, dummy, getattr(torch, "zeros")((oversample, 20), dtype=getattr(torch, "float32")), getattr(torch, "zeros")((oversample, 20), dtype=getattr(torch, "float32")), getattr(torch, "zeros")((oversample, 256), dtype=getattr(torch, "float32")), getattr(torch, "zeros")((oversample, 6), dtype=getattr(torch, "float32")), getattr(torch, "zeros")((oversample, 10), dtype=getattr(torch, "float32")), None)
        for index in range(oversample):
            fg_mask = (getattr(torch, "sigmoid")(out["fg_mask_logits"][index, 0]) >= 0.5).to(dtype=getattr(torch, "long")).detach().cpu().tolist()
            fg_label = (out["fg_label_logits"][index].argmax(dim=0) + 1).detach().cpu().tolist()
            fg_area = sum(int(value) for row in fg_mask for value in row) / 400.0
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
            label_div = len({int(fg_label[y][x]) for y in range(20) for x in range(20) if fg_mask[y][x]})
            if label_div <= 1:
                invalid.append("low_label_diversity")
            desc = foreground_descriptor(fg_label, fg_mask, {"area_ratio": fg_area, "aspect_ratio": 1.0, "center_x": 10.0, "center_y": 10.0})
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
        if len(selected) < int(args.num_candidates):
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
            "sample_dir": str(sample_dir),
        }
        save_json(sample_dir / "meta.json", payload_row)
        outputs.append(payload_row)
        print_progress("sample-fg", out_index + 1, len(selected), f"z={row['local_z']} area={row['fg_area']:.4f}")
    finish_progress()
    save_json(output_dir / "candidates.json", {"category": args.category, "num_candidates": len(outputs), "planner_oversample": oversample, "num_valid_plans": len(valid_rows), "samples": outputs})
    save_jsonl(output_dir / "per_sample.jsonl", outputs)
    print(format_metric_line("sample-foreground:", [("category", args.category), ("num_candidates", len(outputs)), ("valid_plans", len(valid_rows))]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
