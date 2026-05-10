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


def _largest_component_ratio(mask: list[list[bool]]) -> float:
    height = len(mask)
    width = len(mask[0]) if height else 0
    total = sum(1 for row in mask for value in row if value)
    if total <= 0:
        return 0.0
    visited = [[False for _ in range(width)] for _ in range(height)]
    best = 0
    for y_pos in range(height):
        for x_pos in range(width):
            if not mask[y_pos][x_pos] or visited[y_pos][x_pos]:
                continue
            stack = [(y_pos, x_pos)]
            visited[y_pos][x_pos] = True
            size = 0
            while stack:
                cur_y, cur_x = stack.pop()
                size += 1
                for next_y, next_x in ((cur_y - 1, cur_x), (cur_y + 1, cur_x), (cur_y, cur_x - 1), (cur_y, cur_x + 1)):
                    if 0 <= next_y < height and 0 <= next_x < width and mask[next_y][next_x] and not visited[next_y][next_x]:
                        visited[next_y][next_x] = True
                        stack.append((next_y, next_x))
            best = max(best, size)
    return best / float(total)


def _structure_metrics(pred_mask: list[list[int]], tgt_mask: list[list[int]], background_class_id: int, num_classes: int) -> dict[str, float]:
    pred_fg = [[value != background_class_id for value in row] for row in pred_mask]
    tgt_fg = [[value != background_class_id for value in row] for row in tgt_mask]
    pred_fg_count = sum(1 for row in pred_fg for value in row if value)
    tgt_fg_count = sum(1 for row in tgt_fg for value in row if value)
    intersection = 0
    union = 0
    for y_pos in range(len(pred_mask)):
        for x_pos in range(len(pred_mask[0])):
            pred_value = pred_fg[y_pos][x_pos]
            tgt_value = tgt_fg[y_pos][x_pos]
            if pred_value and tgt_value:
                intersection += 1
            if pred_value or tgt_value:
                union += 1
    pred_counts = [0 for _ in range(num_classes)]
    tgt_counts = [0 for _ in range(num_classes)]
    for row in pred_mask:
        for value in row:
            pred_counts[int(value)] += 1
    for row in tgt_mask:
        for value in row:
            tgt_counts[int(value)] += 1
    count_l1 = sum(abs(pred_value - tgt_value) for pred_value, tgt_value in zip(pred_counts, tgt_counts)) / float(max(1, len(pred_mask) * len(pred_mask[0])))
    return {
        "foreground_iou": 0.0 if union <= 0 else intersection / float(union),
        "foreground_ratio_error": abs(pred_fg_count - tgt_fg_count) / float(max(1, len(pred_mask) * len(pred_mask[0]))),
        "class_count_l1": count_l1,
        "largest_component_ratio": _largest_component_ratio(pred_fg),
    }


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
    total_samples = 0
    total_fg_iou = 0.0
    total_fg_ratio_error = 0.0
    total_count_l1 = 0.0
    total_largest_component_ratio = 0.0
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
            pred_list = pred20.detach().cpu().tolist()
            tgt_list = grid20.detach().cpu().tolist()
            for sample_index, pred_mask in enumerate(pred_list):
                target_mask = tgt_list[sample_index]
                structure = _structure_metrics(pred_mask, target_mask, background_class_id=0, num_classes=NUM_CLASSES)
                total_fg_iou += structure["foreground_iou"]
                total_fg_ratio_error += structure["foreground_ratio_error"]
                total_count_l1 += structure["class_count_l1"]
                total_largest_component_ratio += structure["largest_component_ratio"]
                total_samples += 1
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
                    "foreground_iou": structure["foreground_iou"],
                    "foreground_ratio_error": structure["foreground_ratio_error"],
                    "class_count_l1": structure["class_count_l1"],
                    "largest_component_ratio": structure["largest_component_ratio"],
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
        "foreground_iou": total_fg_iou / max(1, total_samples),
        "foreground_ratio_error": total_fg_ratio_error / max(1, total_samples),
        "class_count_l1": total_count_l1 / max(1, total_samples),
        "largest_component_ratio": total_largest_component_ratio / max(1, total_samples),
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
