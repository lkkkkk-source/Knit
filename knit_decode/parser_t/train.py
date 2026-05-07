from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from .dataset import build_parser_dataloader, compute_class_pixel_counts, mask_to_image
from .losses import shift_tolerant_cross_entropy
from .model import build_parser_model


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        optim = importlib.import_module("torch.optim")
    except ImportError as error:
        raise ImportError("PyTorch is required for parser training. Install with `pip install -e .[train]`.") from error
    return torch, optim


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a first-pass simulation-image to structure parser.")
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL manifest mapping simulation images to structure-mask targets.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-shift", type=int, default=1, help="Global shift tolerance used by the Inverse-Knitting-style CE.")
    parser.add_argument("--image-size", type=int, nargs=2, default=(128, 128), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu", help="Training device, for example `cpu`, `cuda`, or `cuda:1`.")
    parser.add_argument("--val-manifest", type=Path, default=None, help="Optional validation manifest.")
    parser.add_argument("--num-vis", type=int, default=4, help="Number of validation predictions to export.")
    parser.add_argument("--model", type=str, default="unet", help="Parser backbone name. Default: unet.")
    return parser


def _resolve_device(torch: object, device_name: str) -> object:
    device_cls = getattr(torch, "device")
    if device_name == "cpu":
        return device_cls("cpu")
    if not getattr(torch, "cuda").is_available():
        raise RuntimeError(f"Requested device {device_name!r}, but CUDA is not available in the current environment.")
    return device_cls(device_name)


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
    foreground_mean_iou = per_class_iou[1] if len(per_class_iou) > 1 and per_class_iou[1] is not None else 0.0
    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "foreground_mean_iou": foreground_mean_iou,
        "per_class_iou": per_class_iou,
    }


def _evaluate_model(
    torch: object,
    model: object,
    dataloader: object,
    device: object,
    max_shift: int,
    output_dir: Path,
    num_vis: int,
    class_weights: object | None,
    num_classes: int,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    batch_count = 0
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    vis_dir = output_dir / "val_predictions"
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_written = 0

    no_grad = getattr(torch, "no_grad")
    argmax = getattr(torch, "argmax")
    with no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            logits = model(images)
            loss = shift_tolerant_cross_entropy(logits, targets, max_shift=max_shift, weight=class_weights)
            total_loss += float(loss.item())
            batch_count += 1

            predictions = argmax(logits, dim=1).detach().cpu().tolist()
            target_rows = targets.detach().cpu().tolist()
            for sample_index, prediction_mask in enumerate(predictions):
                target_mask = target_rows[sample_index]
                for row_index, row in enumerate(target_mask):
                    for col_index, actual in enumerate(row):
                        predicted = prediction_mask[row_index][col_index]
                        confusion[actual][predicted] += 1
                if vis_written < num_vis:
                    sample_id = str(batch["sample_ids"][sample_index]).replace("/", "__")
                    mask_to_image(prediction_mask).save(vis_dir / f"{sample_id}_pred.png")
                    mask_to_image(target_mask).save(vis_dir / f"{sample_id}_target.png")
                    vis_written += 1

    metrics = _compute_segmentation_metrics(confusion)
    metrics["loss"] = total_loss / max(1, batch_count)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch, optim = _require_torch()
    device = _resolve_device(torch, args.device)
    dataloader, dataset = build_parser_dataloader(
        args.manifest,
        batch_size=args.batch_size,
        shuffle=True,
        image_size=(int(args.image_size[0]), int(args.image_size[1])),
    )
    val_dataloader = None
    if args.val_manifest is not None:
        val_dataloader, _ = build_parser_dataloader(
            args.val_manifest,
            batch_size=args.batch_size,
            shuffle=False,
            image_size=(int(args.image_size[0]), int(args.image_size[1])),
        )

    model = build_parser_model(args.model, num_classes=dataset.num_classes)
    model.to(device)
    optimizer = getattr(optim, "Adam")(model.parameters(), lr=args.learning_rate)

    class_pixel_counts = compute_class_pixel_counts(dataset)
    total_pixels = max(1, sum(class_pixel_counts))
    weights: list[float] = []
    for count in class_pixel_counts:
        if count <= 0:
            weights.append(0.0)
        else:
            weights.append(total_pixels / float(count))
    max_weight = max((value for value in weights if value > 0), default=1.0)
    normalized_weights = [value / max_weight if value > 0 else 0.0 for value in weights]
    class_weights = getattr(torch, "tensor")(normalized_weights, dtype=getattr(torch, "float32")).to(device)

    history: list[dict[str, object]] = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for batch in dataloader:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            logits = model(images)
            loss = shift_tolerant_cross_entropy(logits, targets, max_shift=args.max_shift, weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1

        mean_loss = total_loss / max(1, batch_count)
        epoch_metrics: dict[str, object] = {"epoch": epoch + 1, "train_loss": mean_loss}
        if val_dataloader is not None:
            val_metrics = _evaluate_model(
                torch,
                model,
                val_dataloader,
                device,
                args.max_shift,
                args.output_dir,
                args.num_vis,
                class_weights,
                dataset.num_classes,
            )
            epoch_metrics["val_loss"] = val_metrics["loss"]
            epoch_metrics["val_pixel_accuracy"] = val_metrics["pixel_accuracy"]
            epoch_metrics["val_mean_iou"] = val_metrics["mean_iou"]
            epoch_metrics["val_foreground_mean_iou"] = val_metrics["foreground_mean_iou"]
            epoch_metrics["val_per_class_iou"] = val_metrics["per_class_iou"]
            print(
                f"epoch={epoch + 1} train_loss={mean_loss:.6f} "
                f"val_loss={cast(float, val_metrics['loss']):.6f} "
                f"val_acc={cast(float, val_metrics['pixel_accuracy']):.4f} "
                f"val_miou={cast(float, val_metrics['mean_iou']):.4f} "
                f"val_fg_miou={cast(float, val_metrics['foreground_mean_iou']):.4f}"
            )
        else:
            print(f"epoch={epoch + 1} train_loss={mean_loss:.6f}")
        history.append(epoch_metrics)

    metrics = {
        "manifest": str(args.manifest),
        "val_manifest": str(args.val_manifest) if args.val_manifest is not None else None,
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_shift": args.max_shift,
        "image_size": [int(args.image_size[0]), int(args.image_size[1])],
        "learning_rate": args.learning_rate,
        "device": str(device),
        "model": args.model,
        "num_classes": dataset.num_classes,
        "num_samples": len(dataset),
        "num_val_samples": 0 if val_dataloader is None else len(cast(object, val_dataloader).dataset),
        "class_pixel_counts": class_pixel_counts,
        "class_weights": normalized_weights,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    getattr(torch, "save")(
        {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        },
        args.output_dir / "checkpoint.pt",
    )
    print(f"saved metrics: {args.output_dir / 'metrics.json'}")
    print(f"saved checkpoint: {args.output_dir / 'checkpoint.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
