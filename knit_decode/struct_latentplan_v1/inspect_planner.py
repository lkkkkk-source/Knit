from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models.planner import LatentPlanner
from .utils import finish_progress, format_metric_line, load_config, print_progress, save_binary_map, save_label_map


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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    planner_cf = config["planner"]
    data_cf = config["data"]
    output_dir = Path(args.output_dir or (args.planner_checkpoint.parent / "inspect" / args.category))
    output_dir.mkdir(parents=True, exist_ok=True)

    torch = _require_torch()
    payload = getattr(torch, "load")(args.planner_checkpoint, map_location="cpu")
    metrics = payload["metrics"]
    category_to_id = metrics["category_to_id"]
    if args.category not in category_to_id:
        raise KeyError(
            f"Unknown category {args.category!r}. "
            f"Available categories from checkpoint: {sorted(category_to_id)}"
        )
    device = _resolve_device(torch, args.device)
    planner = LatentPlanner(
        num_categories=int(metrics["num_categories"]),
        num_modes=int(planner_cf["num_modes"]),
        coarse_size=int(data_cf["coarse_size"]),
        num_classes=int(data_cf["num_classes"]),
        category_embed_dim=int(planner_cf["category_embed_dim"]),
        mode_embed_dim=int(planner_cf["mode_embed_dim"]),
        hidden_dim=int(planner_cf["hidden_dim"]),
        num_layers=int(planner_cf["num_layers"]),
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
            save_label_map(c5, sample_dir / "c5.png", scale=32)
            save_binary_map(o5, sample_dir / "o5.png", scale=32)
            top_probs, top_ids = getattr(torch, "topk")(probs[index], k=min(5, probs.shape[-1]))
            row = {
                "category": args.category,
                "z_id": int(outputs["z_ids"][index].item()),
                "z_logprob": float(outputs["z_logprob"][index].item()),
                "top_z_ids": [int(value) for value in top_ids.detach().cpu().tolist()],
                "top_z_probs": [float(value) for value in top_probs.detach().cpu().tolist()],
                "fg_ratio_pred": float(outputs["fg_ratio_pred"][index].item()),
                "r17_pred": [float(value) for value in outputs["r17_pred"][index].detach().cpu().tolist()],
                "sample_dir": str(sample_dir),
            }
            (sample_dir / "meta.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
            rows.append(row)
            print_progress("inspect-planner", index + 1, int(args.num_samples), f"z={row['z_id']} prob={row['top_z_probs'][0]:.4f}")
    finish_progress()
    payload_out = {"category": args.category, "samples": rows}
    (output_dir / "planner_samples.json").write_text(json.dumps(payload_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(format_metric_line("inspect-planner:", [("category", args.category), ("num_samples", int(args.num_samples)), ("output", str(output_dir))]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
