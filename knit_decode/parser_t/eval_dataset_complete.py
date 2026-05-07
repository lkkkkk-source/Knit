from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import cast

from PIL import Image

from .dataset import (
    build_parser_dataloader,
    build_parser_manifest_from_dataset_complete,
    mask_to_image,
    read_vocabulary,
)
from .losses import segmentation_cross_entropy
from .model import build_parser_model


def _require_torch() -> object:
    import importlib

    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required for parser evaluation. Install with `pip install -e .[train]`.") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained parser on dataset_complete.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to dataset_complete")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained parser checkpoint.pt")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for evaluation outputs")
    parser.add_argument("--device", type=str, default="cpu", help="Evaluation device, for example `cpu`, `cuda`, or `cuda:1`.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-vis", type=int, default=16, help="Number of predictions to export for inspection")
    parser.add_argument("--manifest-output", type=Path, default=None, help="Optional path for the generated dataset_complete manifest")
    parser.add_argument("--vocabulary", type=Path, default=None, help="Optional override for vocabulary.json")
    parser.add_argument("--model", type=str, default=None, help="Optional override for parser model name")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--grid-size", type=int, nargs=2, default=None, metavar=("ROWS", "COLS"))
    return parser


def _resolve_device(torch: object, device_name: str) -> object:
    device_cls = getattr(torch, "device")
    if device_name == "cpu":
        return device_cls("cpu")
    if not getattr(torch, "cuda").is_available():
        raise RuntimeError(f"Requested device {device_name!r}, but CUDA is not available in the current environment.")
    return device_cls(device_name)


def _autocast_context(torch: object, enabled: bool) -> object:
    cuda_amp = getattr(getattr(torch, "cuda"), "amp")
    return getattr(cuda_amp, "autocast")(enabled=enabled)


def _print_progress(stage: str, current: int, total: int, extra: str = "") -> None:
    width = 30
    ratio = 0.0 if total <= 0 else current / total
    filled = min(width, int(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    suffix = f" {extra}" if extra else ""
    print(f"\r[{stage}] [{bar}] {current}/{total}{suffix}", end="", flush=True)


def _finish_progress() -> None:
    print(flush=True)


def _compute_segmentation_metrics(confusion: list[list[int]]) -> dict[str, object]:
    num_classes = len(confusion)
    correct = sum(confusion[index][index] for index in range(num_classes))
    total = sum(sum(row) for row in confusion)
    pixel_accuracy = correct / total if total else 0.0
    ious: list[float] = []
    per_class_iou: list[float | None] = []
    for class_id in range(num_classes):
        intersection = confusion[class_id][class_id]
        predicted = sum(confusion[row_id][class_id] for row_id in range(num_classes))
        actual = sum(confusion[class_id])
        union = actual + predicted - intersection
        if union > 0:
            value = intersection / union
            ious.append(value)
            per_class_iou.append(value)
        else:
            per_class_iou.append(None)
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    foreground_values = [value for value in per_class_iou[1:] if value is not None]
    foreground_mean_iou = sum(foreground_values) / len(foreground_values) if foreground_values else 0.0
    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "foreground_mean_iou": foreground_mean_iou,
        "per_class_iou": per_class_iou,
    }


def _histogram(values: list[list[int]], num_classes: int) -> list[int]:
    counts = [0 for _ in range(num_classes)]
    for row in values:
        for class_id in row:
            counts[class_id] += 1
    return counts


def _save_grayscale_tensor(torch: object, image_tensor: object, output_path: Path) -> None:
    image = image_tensor.detach().cpu().clamp(0.0, 1.0)
    image = (image * 255.0).round().to(dtype=getattr(torch, "uint8"))
    height = int(image.shape[-2])
    width = int(image.shape[-1])
    flat_pixels = image.reshape(-1).tolist()
    output = Image.new("L", (width, height))
    output.putdata(flat_pixels)
    output.save(output_path)


def _resolve_existing_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    candidates = [path, base_dir / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path.resolve()


def _load_checkpoint_metadata(checkpoint_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    torch = _require_torch()
    checkpoint = cast(dict[str, object], getattr(torch, "load")(checkpoint_path, map_location="cpu"))
    metrics = cast(dict[str, object], checkpoint.get("metrics", {}))
    return checkpoint, metrics


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch = _require_torch()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint, checkpoint_metrics = _load_checkpoint_metadata(checkpoint_path)

    image_size = tuple(int(value) for value in (args.image_size or checkpoint_metrics.get("image_size") or [160, 160]))
    grid_size = tuple(int(value) for value in (args.grid_size or checkpoint_metrics.get("grid_size") or [20, 20]))
    model_name = str(args.model or checkpoint_metrics.get("model") or "kaspar")
    manifest_output = args.manifest_output or (args.output_dir / "dataset_complete_manifest.jsonl")
    manifest_path = build_parser_manifest_from_dataset_complete(args.dataset_root, manifest_output)

    vocabulary_path = args.vocabulary
    if vocabulary_path is None:
        manifest_value = checkpoint_metrics.get("manifest")
        if not isinstance(manifest_value, str):
            raise ValueError("Checkpoint metrics do not contain a training manifest path. Pass --vocabulary explicitly.")
        train_manifest_path = _resolve_existing_path(manifest_value, checkpoint_path.parent)
        vocabulary_path = train_manifest_path.parent / "vocabulary.json"
    vocabulary = read_vocabulary(vocabulary_path)

    dataloader, dataset = build_parser_dataloader(
        manifest_path,
        batch_size=args.batch_size,
        shuffle=False,
        image_size=cast(tuple[int, int], image_size),
        grid_size=cast(tuple[int, int], grid_size),
        vocabulary=vocabulary,
        top_k_colors=max(1, len(vocabulary.top_colors)),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    sample_lookup = {sample.sample_id: sample for sample in dataset.samples}

    device = _resolve_device(torch, args.device)
    if str(device).startswith("cuda") and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    use_amp = bool(args.amp and str(device).startswith("cuda"))

    model = build_parser_model(model_name, num_classes=dataset.num_classes)
    model.load_state_dict(cast(dict[str, object], checkpoint["model_state_dict"]))
    model.to(device)
    model.eval()

    total_loss = 0.0
    batch_count = 0
    confusion = [[0 for _ in range(dataset.num_classes)] for _ in range(dataset.num_classes)]
    prediction_hist = [0 for _ in range(dataset.num_classes)]
    target_hist = [0 for _ in range(dataset.num_classes)]
    vis_dir = args.output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_written = 0

    no_grad = getattr(torch, "no_grad")
    argmax = getattr(torch, "argmax")
    total_batches = len(dataloader)
    with no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            with _autocast_context(torch, use_amp):
                logits = model(images)
                loss = segmentation_cross_entropy(logits, targets)
            total_loss += float(loss.item())
            batch_count += 1
            _print_progress("eval", batch_count, total_batches, f"loss={total_loss / max(1, batch_count):.6f}")

            predictions = argmax(logits, dim=1).detach().cpu().tolist()
            target_rows = targets.detach().cpu().tolist()
            image_rows = images.detach().cpu()
            sample_ids = [str(sample_id) for sample_id in batch["sample_ids"]]
            for sample_index, prediction_mask in enumerate(predictions):
                target_mask = target_rows[sample_index]
                pred_counts = _histogram(prediction_mask, dataset.num_classes)
                tgt_counts = _histogram(target_mask, dataset.num_classes)
                for class_id in range(dataset.num_classes):
                    prediction_hist[class_id] += pred_counts[class_id]
                    target_hist[class_id] += tgt_counts[class_id]
                for row_index, row in enumerate(target_mask):
                    for col_index, actual in enumerate(row):
                        predicted = prediction_mask[row_index][col_index]
                        confusion[actual][predicted] += 1
                if vis_written < args.num_vis:
                    sample_dir = vis_dir / sample_ids[sample_index].replace("/", "__")
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    sample = sample_lookup[sample_ids[sample_index]]
                    _save_grayscale_tensor(torch, image_rows[sample_index], sample_dir / "input.png")
                    mask_to_image(prediction_mask, vocabulary).save(sample_dir / "pred.png")
                    mask_to_image(target_mask, vocabulary).save(sample_dir / "target.png")
                    shutil.copy2(sample.image_path, sample_dir / "input_source.png")
                    shutil.copy2(sample.target_path, sample_dir / "gt_source.png")
                    (sample_dir / "paths.json").write_text(
                        json.dumps(
                            {
                                "sample_id": sample.sample_id,
                                "category": sample.category,
                                "input_source": str(sample.image_path),
                                "gt_source": str(sample.target_path),
                            },
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    vis_written += 1
    _finish_progress()

    metrics = _compute_segmentation_metrics(confusion)
    result = {
        "dataset_root": str(args.dataset_root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "checkpoint": str(checkpoint_path),
        "vocabulary": str(Path(vocabulary_path).resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "device": str(device),
        "amp": use_amp,
        "model": model_name,
        "image_size": list(image_size),
        "grid_size": list(grid_size),
        "num_samples": len(dataset),
        "batch_size": args.batch_size,
        "num_classes": dataset.num_classes,
        "class_names": list(dataset.class_names),
        "loss": total_loss / max(1, batch_count),
        "pixel_accuracy": metrics["pixel_accuracy"],
        "mean_iou": metrics["mean_iou"],
        "foreground_mean_iou": metrics["foreground_mean_iou"],
        "per_class_iou": metrics["per_class_iou"],
        "prediction_histogram": prediction_hist,
        "target_histogram": target_hist,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
