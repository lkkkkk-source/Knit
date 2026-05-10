from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models.planner import LatentPlanner
from .utils import compute_grammar_descriptor, finish_progress, format_metric_line, load_config, print_progress, save_binary_map, save_label_map, upsample_nearest


def _require_torch() -> object:
    import importlib

    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required for planner inspection. Install with `pip install -e .[train]`.") from error


def _resolve_device(torch: object, device_name: str) -> object:
    device_cls = getattr(torch, "device")
    if device_name == "cpu":
        return device_cls("cpu")
    if not getattr(torch, "cuda").is_available():
        raise RuntimeError(f"Requested device {device_name!r}, but CUDA is not available.")
    return device_cls(device_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect planner diversity for a single category.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--planner-checkpoint", type=Path, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def _checkpoint_head_dim(payload: dict[str, object], key: str) -> int:
    if key in payload:
        return int(payload[key])
    state_dict = payload["model_state_dict"]
    weight = state_dict.get(key.replace("_dim", "_head.weight"))
    if weight is None:
        raise KeyError(f"Unable to infer {key} from checkpoint payload.")
    return int(weight.shape[0])


def _plan_descriptor_from_outputs(c10: list[list[int]], o10: list[list[int]], background_class_id: int) -> tuple[list[float], dict[str, object]]:
    upsampled_labels = upsample_nearest(c10, 20)
    upsampled_occ = upsample_nearest(o10, 20)
    pseudo_y20 = []
    for y_pos in range(20):
        row = []
        for x_pos in range(20):
            if int(upsampled_occ[y_pos][x_pos]) <= 0:
                row.append(int(background_class_id))
            else:
                row.append(int(upsampled_labels[y_pos][x_pos]))
        pseudo_y20.append(row)
    grammar = compute_grammar_descriptor(pseudo_y20, background_class_id=background_class_id, coarse_threshold=0.25, num_classes=17)
    return grammar["descriptor"], grammar


def _distance_metrics(descriptor: list[float], category: str, cache_payload: dict[str, object]) -> tuple[float, float, float]:
    import torch

    descriptors = cache_payload["items"]
    own = [entry["descriptor"] for entry in descriptors if entry["category"] == category]
    others_by_category: dict[str, list[list[float]]] = {}
    for entry in descriptors:
        if entry["category"] == category:
            continue
        others_by_category.setdefault(entry["category"], []).append(entry["descriptor"])
    descriptor_tensor = getattr(torch, "tensor")(descriptor, dtype=getattr(torch, "float32"))
    own_tensor = getattr(torch, "tensor")(own, dtype=getattr(torch, "float32")) if own else None
    own_distance = float(((own_tensor - descriptor_tensor.unsqueeze(0)) ** 2).sum(dim=-1).sqrt().mean().item()) if own_tensor is not None else float("inf")
    nearest_other = float("inf")
    for other_desc in others_by_category.values():
        other_tensor = getattr(torch, "tensor")(other_desc, dtype=getattr(torch, "float32"))
        value = float(((other_tensor - descriptor_tensor.unsqueeze(0)) ** 2).sum(dim=-1).sqrt().mean().item())
        nearest_other = min(nearest_other, value)
    margin = own_distance - nearest_other if nearest_other < float("inf") else own_distance
    return own_distance, nearest_other, margin


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    planner_cf = config["planner"]
    data_cf = config["data"]
    output_dir = Path(args.output_dir or (args.planner_checkpoint.parent / "inspect" / args.category))
    output_dir.mkdir(parents=True, exist_ok=True)

    torch = _require_torch()
    if not args.planner_checkpoint.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {args.planner_checkpoint}\n"
            "Please run train_planner.py first and verify checkpoint path."
        )
    payload = getattr(torch, "load")(args.planner_checkpoint, map_location="cpu")
    metrics = payload["metrics"]
    category_to_id = metrics["category_to_id"]
    train_cache_path = Path(config["data"]["plan_cache_dir"]) / f"{Path(config['data']['train_manifest']).stem}.pt"
    cache_payload = getattr(torch, "load")(train_cache_path, map_location="cpu")
    if args.category not in category_to_id:
        raise KeyError(
            f"Unknown category {args.category!r}. "
            f"Available categories from checkpoint: {sorted(category_to_id)}"
        )
    device = _resolve_device(torch, args.device)
    ckpt_grammar_dim = _checkpoint_head_dim(payload, "grammar_signature_dim")
    ckpt_adj_dim = _checkpoint_head_dim(payload, "adjacency_signature_dim")
    config_grammar_dim = config.get("planner", {}).get("grammar_signature_dim", None)
    if config_grammar_dim is not None and int(config_grammar_dim) != ckpt_grammar_dim:
        print(
            f"warning: config grammar_signature_dim={config_grammar_dim} "
            f"checkpoint grammar_signature_dim={ckpt_grammar_dim}; "
            "using checkpoint grammar_signature_dim for inspection"
        )
    planner = LatentPlanner(
        num_categories=int(metrics["num_categories"]),
        num_modes=int(planner_cf["num_modes"]),
        coarse_size=int(data_cf["coarse_size"]),
        num_classes=int(data_cf["num_classes"]),
        category_embed_dim=int(planner_cf["category_embed_dim"]),
        mode_embed_dim=int(planner_cf["mode_embed_dim"]),
        hidden_dim=int(planner_cf["hidden_dim"]),
        num_layers=int(planner_cf["num_layers"]),
        coarse_size_10=10,
        grammar_dim=ckpt_grammar_dim,
        adjacency_dim=ckpt_adj_dim,
        max_num_modes_per_category=int(metrics.get("max_num_modes_per_category", planner_cf.get("max_num_modes_per_category", 16))),
    )
    planner.load_state_dict(payload["model_state_dict"])
    planner.to(device)
    planner.eval()
    category_ids = getattr(torch, "full")((int(args.num_samples),), int(category_to_id[args.category]), device=device, dtype=getattr(torch, "long"))
    rows: list[dict[str, object]] = []
    with getattr(torch, "no_grad")():
        outputs = planner(category_ids, z_ids=None, sample_mode="sample", z_temperature=1.2, z_top_p=0.9)
        probs = getattr(__import__("importlib").import_module("torch.nn.functional"), "softmax")(outputs["z_logits"], dim=-1)
        for index in range(int(args.num_samples)):
            sample_dir = output_dir / f"planner_{index:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            c5 = outputs["c5_logits"][index].argmax(dim=0).detach().cpu().tolist()
            o5 = (getattr(torch, "sigmoid")(outputs["o5_logits"][index, 0]) >= 0.5).detach().cpu().tolist()
            c10 = outputs["c10_logits"][index].argmax(dim=0).detach().cpu().tolist()
            o10 = (getattr(torch, "sigmoid")(outputs["o10_logits"][index, 0]) >= 0.5).detach().cpu().tolist()
            save_label_map(c5, sample_dir / "c5.png", scale=32)
            save_binary_map(o5, sample_dir / "o5.png", scale=32)
            save_label_map(c10, sample_dir / "c10.png", scale=16)
            save_binary_map(o10, sample_dir / "o10.png", scale=16)
            top_probs, top_ids = getattr(torch, "topk")(probs[index], k=min(5, probs.shape[-1]))
            row_projection = [float(value) for value in outputs["row_projection_pred"][index].detach().cpu().tolist()]
            col_projection = [float(value) for value in outputs["col_projection_pred"][index].detach().cpu().tolist()]
            grammar_signature = [float(value) for value in outputs["grammar_signature_pred"][index].detach().cpu().tolist()]
            adjacency_signature = [float(value) for value in outputs["adjacency_signature_pred"][index].detach().cpu().tolist()]
            plan_descriptor, grammar = _plan_descriptor_from_outputs(c10, o10, background_class_id=int(config["data"]["background_class_id"]))
            own_distance, nearest_other_distance, margin = _distance_metrics(plan_descriptor, args.category, cache_payload)
            row = {
                "category": args.category,
                "z_id": int(outputs["z_ids"][index].item()),
                "z_logprob": float(outputs["z_logprob"][index].item()),
                "top_z_ids": [int(value) for value in top_ids.detach().cpu().tolist()],
                "top_z_probs": [float(value) for value in top_probs.detach().cpu().tolist()],
                "fg_ratio_pred": float(outputs["fg_ratio_pred"][index].item()),
                "r17_pred": [float(value) for value in outputs["r17_pred"][index].detach().cpu().tolist()],
                "row_projection": row_projection,
                "col_projection": col_projection,
                "grammar_signature": grammar_signature,
                "adjacency_signature": adjacency_signature,
                "vertical_continuity": float(grammar["vertical_continuity"]),
                "horizontal_continuity": float(grammar["horizontal_continuity"]),
                "symmetry_score": float(grammar["symmetry_score"]),
                "center_band_score": float(grammar["center_band_score"]),
                "stripe_score": grammar["stripe_score"],
                "own_category_distance": own_distance,
                "nearest_other_category_distance": nearest_other_distance,
                "category_descriptor_margin": margin,
                "c5": c5,
                "o5": o5,
                "c10": c10,
                "o10": o10,
                "sample_dir": str(sample_dir),
            }
            (sample_dir / "meta.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
            rows.append(row)
            print_progress("inspect-planner", index + 1, int(args.num_samples), f"z={row['z_id']} prob={row['top_z_probs'][0]:.4f}")
    finish_progress()
    payload_out = {
        "category": args.category,
        "samples": rows,
        "summary": {
            "all_background_rate": sum(1.0 for row in rows if row["fg_ratio_pred"] <= 1e-6) / float(max(1, len(rows))),
            "mean_fg_ratio": sum(float(row["fg_ratio_pred"]) for row in rows) / float(max(1, len(rows))),
            "unique_o5_layout_count": len({json.dumps(row["o5"]) for row in rows}),
            "unique_o10_layout_count": len({json.dumps(row["o10"]) for row in rows}),
            "effective_local_modes": len({int(row["z_id"]) for row in rows}),
            "mean_vertical_continuity": sum(float(row["vertical_continuity"]) for row in rows) / float(max(1, len(rows))),
            "mean_center_band_score": sum(float(row["center_band_score"]) for row in rows) / float(max(1, len(rows))),
            "mean_symmetry_score": sum(float(row["symmetry_score"]) for row in rows) / float(max(1, len(rows))),
            "own_category_distance": sum(float(row["own_category_distance"]) for row in rows) / float(max(1, len(rows))),
            "nearest_other_category_distance": sum(float(row["nearest_other_category_distance"]) for row in rows) / float(max(1, len(rows))),
            "category_descriptor_margin": sum(float(row["category_descriptor_margin"]) for row in rows) / float(max(1, len(rows))),
        },
    }
    (output_dir / "planner_samples.json").write_text(json.dumps(payload_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(format_metric_line("inspect-planner:", [("category", args.category), ("num_samples", int(args.num_samples)), ("output", str(output_dir))]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
