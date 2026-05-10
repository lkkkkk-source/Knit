from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import cast

from knit_decode.struct_ar_v1.train import _compute_metrics, _resolve_device

from .dataset import NUM_CLASSES, build_dataloader
from .model import MultiScaleMaskGitPrior
from .train import _sample_multiscale, _save_label_map


def _require_torch() -> object:
    import importlib

    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required for struct_maskgit_v1 evaluation. Install with `pip install -e .[train]`.") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample and evaluate a struct_maskgit_v1 checkpoint on a manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-choice-temperature", type=float, default=None)
    parser.add_argument("--sample-steps-5", type=int, default=None)
    parser.add_argument("--sample-steps-10", type=int, default=None)
    parser.add_argument("--sample-steps-20", type=int, default=None)
    parser.add_argument("--mask-scheduling-method", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch = _require_torch()
    checkpoint = cast(dict[str, object], getattr(torch, "load")(args.checkpoint, map_location="cpu"))
    metrics = cast(dict[str, object], checkpoint["metrics"])
    model_config = cast(dict[str, object], checkpoint["model_config"])
    category_to_id = cast(dict[str, int], metrics["category_to_id"])
    palette_path = Path(cast(str, metrics["palette"]))
    device = _resolve_device(torch, args.device)
    if str(device).startswith("cuda") and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    loader, dataset = build_dataloader(
        args.manifest,
        palette_path=palette_path,
        batch_size=args.batch_size,
        shuffle=False,
        category_to_id=category_to_id,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    model = MultiScaleMaskGitPrior(
        num_categories=int(model_config["num_categories"]),
        num_classes=int(model_config["num_classes"]),
        width=int(model_config["width"]),
        depth=int(model_config["depth"]),
        heads=int(model_config["heads"]),
        mlp_ratio=float(model_config["mlp_ratio"]),
        dropout=float(model_config["dropout"]),
    )
    model.load_state_dict(cast(dict[str, object], checkpoint["model_state_dict"]))
    model.to(device)
    model.eval()

    effective_args = argparse.Namespace(
        sample_choice_temperature=args.sample_choice_temperature if args.sample_choice_temperature is not None else metrics["sample_choice_temperature"],
        sample_steps_5=args.sample_steps_5 if args.sample_steps_5 is not None else metrics["sample_steps_5"],
        sample_steps_10=args.sample_steps_10 if args.sample_steps_10 is not None else metrics["sample_steps_10"],
        sample_steps_20=args.sample_steps_20 if args.sample_steps_20 is not None else metrics["sample_steps_20"],
        mask_scheduling_method=args.mask_scheduling_method if args.mask_scheduling_method is not None else metrics["mask_scheduling_method"],
    )

    selected_ids = None
    if args.num_samples is not None:
        rng = random.Random(args.seed)
        population = [str(sample["sample_id"]) for sample in dataset.samples]
        selected_ids = set(rng.sample(population, min(args.num_samples, len(population))))

    confusion = [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)]
    exact_matches = 0
    total_samples = 0
    rows: list[dict[str, object]] = []
    with getattr(torch, "no_grad")():
        for batch in loader:
            keep_indices = list(range(len(batch["sample_ids"])))
            if selected_ids is not None:
                keep_indices = [index for index, sample_id in enumerate(batch["sample_ids"]) if str(sample_id) in selected_ids]
                if not keep_indices:
                    continue
            index_tensor = getattr(torch, "tensor")(keep_indices, dtype=getattr(torch, "long"))
            category_ids = batch["category_ids"].index_select(0, index_tensor).to(device)
            grid20 = batch["grid20"].index_select(0, index_tensor).to(device)
            sample_ids = [str(batch["sample_ids"][index]) for index in keep_indices]
            categories = [str(batch["categories"][index]) for index in keep_indices]
            sampled = _sample_multiscale(model, category_ids, effective_args)
            pred20 = sampled["pred20"]
            exact_tensor = pred20.eq(grid20).reshape(pred20.shape[0], -1).all(dim=1)
            exact_matches += int(exact_tensor.sum().item())
            total_samples += int(pred20.shape[0])
            pred_list = pred20.detach().cpu().tolist()
            tgt_list = grid20.detach().cpu().tolist()
            for sample_index, pred_mask in enumerate(pred_list):
                target_mask = tgt_list[sample_index]
                sample_correct = 0
                for row_index, row in enumerate(target_mask):
                    for col_index, actual in enumerate(row):
                        predicted = pred_mask[row_index][col_index]
                        confusion[actual][predicted] += 1
                        sample_correct += int(predicted == actual)
                sample_dir = args.output_dir / sample_ids[sample_index].replace("/", "__")
                sample_dir.mkdir(parents=True, exist_ok=True)
                _save_label_map(pred_mask, sample_dir / "pred20.png")
                _save_label_map(target_mask, sample_dir / "target20.png")
                sample_metrics = {
                    "sample_id": sample_ids[sample_index],
                    "category": categories[sample_index],
                    "pixel_accuracy": sample_correct / float(len(target_mask) * len(target_mask[0])),
                    "exact_match": bool(exact_tensor[sample_index].item()),
                    "output_dir": str(sample_dir),
                }
                (sample_dir / "meta.json").write_text(json.dumps(sample_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
                rows.append(sample_metrics)

    summary = _compute_metrics(confusion)
    result = {
        "manifest": str(args.manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "num_samples": total_samples,
        "pixel_accuracy": summary["pixel_accuracy"],
        "mean_iou": summary["mean_iou"],
        "per_class_iou": summary["per_class_iou"],
        "exact_match": exact_matches / max(1, total_samples),
        "sample_choice_temperature": effective_args.sample_choice_temperature,
        "sample_steps_5": effective_args.sample_steps_5,
        "sample_steps_10": effective_args.sample_steps_10,
        "sample_steps_20": effective_args.sample_steps_20,
        "mask_scheduling_method": effective_args.mask_scheduling_method,
        "samples": rows,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
