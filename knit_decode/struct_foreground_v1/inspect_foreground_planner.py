from __future__ import annotations

import argparse
from pathlib import Path

from .compose_foreground import compose_foreground
from .models.foreground_planner import ForegroundCanonicalPlanner
from .utils import finish_progress, format_metric_line, foreground_descriptor, load_config, normalized_l2, print_progress, save_binary_map, save_json, save_jsonl, save_label_map


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
    output_dir = Path(args.output_dir or (args.checkpoint.parent / "inspect" / args.category))
    output_dir.mkdir(parents=True, exist_ok=True)
    category_id = int(category_to_id[args.category])
    train_cache_path = Path(config["data"]["cache_dir"]) / "foreground_cache_train.pt"
    cache_payload = _require_torch().load(train_cache_path, map_location="cpu")
    grammar_dim = _checkpoint_dim(payload, metrics, "grammar_signature_dim", "grammar_head.weight")
    adjacency_dim = _checkpoint_dim(payload, metrics, "adjacency_signature_dim", "adj_head.weight")
    bbox_dim = _checkpoint_dim(payload, metrics, "bbox_dim", "bbox_head.weight")
    max_num_modes = int(metrics.get("max_num_modes", config["planner"]["num_modes_per_category"]))
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
    with getattr(torch, "no_grad")():
        dummy = getattr(torch, "zeros")((int(args.num_samples), 16), dtype=getattr(torch, "float32"))
        out = model(category_ids, dummy, getattr(torch, "zeros")((int(args.num_samples), 20), dtype=getattr(torch, "float32")), getattr(torch, "zeros")((int(args.num_samples), 20), dtype=getattr(torch, "float32")), getattr(torch, "zeros")((int(args.num_samples), 256), dtype=getattr(torch, "float32")), getattr(torch, "zeros")((int(args.num_samples), 6), dtype=getattr(torch, "float32")), getattr(torch, "zeros")((int(args.num_samples), 10), dtype=getattr(torch, "float32")), None)
        probs = getattr(__import__("importlib").import_module("torch.nn.functional"), "softmax")(out["local_z_logits"], dim=-1)
        for index in range(int(args.num_samples)):
            sample_dir = output_dir / f"planner_{index:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            fg_mask = (getattr(torch, "sigmoid")(out["fg_mask_logits"][index, 0]) >= 0.5).to(dtype=getattr(torch, "long")).detach().cpu().tolist()
            fg_label = (out["fg_label_logits"][index].argmax(dim=0) + 1).detach().cpu().tolist()
            composed = compose_foreground(fg_mask, fg_label, [0.2, 0.2, 0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0.25, 1.0], sample_dir)["composed_y20"]
            save_binary_map(fg_mask, sample_dir / "fg_mask20.png", scale=12)
            save_label_map(fg_label, sample_dir / "fg_label20.png", scale=12)
            save_label_map(composed, sample_dir / "composed_y20.png", scale=12)
            desc = foreground_descriptor(fg_label, fg_mask, {"area_ratio": sum(int(v) for row in fg_mask for v in row) / 400.0, "aspect_ratio": 1.0, "center_x": 10.0, "center_y": 10.0})
            own_distance, nearest_other_distance, margin = _descriptor_margin(desc["descriptor"], args.category, cache_payload)
            label_div = len({int(fg_label[y][x]) for y in range(20) for x in range(20) if fg_mask[y][x]})
            invalid = []
            if not any(any(v for v in row_fg) for row_fg in fg_mask):
                invalid.append("empty_foreground")
            if sum(int(v) for row_fg in fg_mask for v in row_fg) >= 399:
                invalid.append("full_foreground")
            row = {
                "category": args.category,
                "sample_index": index,
                "local_z": int(out["local_z"][index].item()),
                "local_z_prob": float(probs[index].max().item()),
                "bbox_pred": [float(value) for value in out["bbox_pred"][index].detach().cpu().tolist()],
                "is_valid_foreground": len(invalid) == 0,
                "mean_fg_area": sum(int(value) for row_fg in fg_mask for value in row_fg) / 400.0,
                "label_diversity": label_div,
                "mean_vertical_continuity": float(desc["vertical_continuity"]),
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
            print_progress("inspect-fg", index + 1, int(args.num_samples), f"z={row['local_z']} prob={row['local_z_prob']:.4f}")
    finish_progress()
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
        "mean_horizontal_continuity": 0.0,
        "mean_center_band_score": sum(float(row["mean_center_band_score"]) for row in rows) / float(max(1, len(rows))),
        "mean_symmetry_score": sum(float(row["mean_symmetry_score"]) for row in rows) / float(max(1, len(rows))),
        "mean_stripe_score": sum(float(row["mean_stripe_score"]) for row in rows) / float(max(1, len(rows))),
        "own_category_distance": sum(float(row["own_category_distance"]) for row in rows) / float(max(1, len(rows))),
        "nearest_other_category_distance": sum(float(row["nearest_other_category_distance"]) for row in rows) / float(max(1, len(rows))),
        "category_descriptor_margin": sum(float(row["category_descriptor_margin"]) for row in rows) / float(max(1, len(rows))),
        "invalid_reason_counts": {
            "empty_foreground": sum(1 for row in rows if "empty_foreground" in row["invalid_reasons"]),
            "full_foreground": sum(1 for row in rows if "full_foreground" in row["invalid_reasons"]),
        },
    }
    save_json(output_dir / "summary.json", summary)
    save_jsonl(output_dir / "per_sample.jsonl", rows)
    print(format_metric_line("inspect-foreground:", [("category", args.category), ("num_samples", int(args.num_samples)), ("valid_fg", summary["valid_foreground_rate"])]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
