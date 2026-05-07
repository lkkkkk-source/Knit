from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from .dataset import build_parser_dataloader, build_topk_color_vocabulary, compute_class_pixel_counts, mask_to_image
from .losses import segmentation_cross_entropy
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
    parser = argparse.ArgumentParser(description="Train an Inverse-Knitting-style simulation-image to structure-grid parser.")
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL manifest mapping simulation images to structure-grid targets.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-shift", type=int, default=1, help="Global shift tolerance used by the Inverse-Knitting-style CE.")
    parser.add_argument("--image-size", type=int, nargs=2, default=(160, 160), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--grid-size", type=int, nargs=2, default=(20, 20), metavar=("ROWS", "COLS"))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu", help="Training device, for example `cpu`, `cuda`, or `cuda:1`.")
    parser.add_argument("--val-manifest", type=Path, default=None, help="Optional validation manifest.")
    parser.add_argument("--num-vis", type=int, default=4, help="Number of validation predictions to export.")
    parser.add_argument("--model", type=str, default="kaspar", help="Parser backbone name. Default: kaspar.")
    parser.add_argument("--top-k-colors", type=int, default=4, help="Number of most frequent grid colors to keep as explicit classes.")
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
    output_dir: Path,
    num_vis: int,
    num_classes: int,
    vocabulary: object,
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
            loss = segmentation_cross_entropy(logits, targets)
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
                    mask_to_image(prediction_mask, vocabulary).save(vis_dir / f"{sample_id}_pred.png")
                    mask_to_image(target_mask, vocabulary).save(vis_dir / f"{sample_id}_target.png")
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
    _, train_dataset = build_parser_dataloader(
        args.manifest,
        batch_size=args.batch_size,
        shuffle=True,
        image_size=(int(args.image_size[0]), int(args.image_size[1])),
        grid_size=(int(args.grid_size[0]), int(args.grid_size[1])),
        vocabulary=None,
        top_k_colors=args.top_k_colors,
    )
    vocabulary = train_dataset.vocabulary
    dataloader, dataset = build_parser_dataloader(
        args.manifest,
        batch_size=args.batch_size,
        shuffle=True,
        image_size=(int(args.image_size[0]), int(args.image_size[1])),
        grid_size=(int(args.grid_size[0]), int(args.grid_size[1])),
        vocabulary=vocabulary,
        top_k_colors=args.top_k_colors,
    )
    val_dataloader = None
    if args.val_manifest is not None:
        val_dataloader, _ = build_parser_dataloader(
            args.val_manifest,
            batch_size=args.batch_size,
            shuffle=False,
            image_size=(int(args.image_size[0]), int(args.image_size[1])),
            grid_size=(int(args.grid_size[0]), int(args.grid_size[1])),
            vocabulary=vocabulary,
            top_k_colors=args.top_k_colors,
        )

    model = build_parser_model(args.model, num_classes=dataset.num_classes)
    model.to(device)
    optimizer = getattr(optim, "Adam")(model.parameters(), lr=args.learning_rate)

    class_pixel_counts = compute_class_pixel_counts(dataset)

    history: list[dict[str, object]] = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for batch in dataloader:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            logits = model(images)
            loss = segmentation_cross_entropy(logits, targets)
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
                args.output_dir,
                args.num_vis,
                dataset.num_classes,
                vocabulary,
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
        "grid_size": [int(args.grid_size[0]), int(args.grid_size[1])],
        "learning_rate": args.learning_rate,
        "device": str(device),
        "model": args.model,
        "top_k_colors": args.top_k_colors,
        "num_classes": dataset.num_classes,
        "class_names": dataset.class_names,
        "num_samples": len(dataset),
        "num_val_samples": 0 if val_dataloader is None else len(cast(object, val_dataloader).dataset),
        "class_pixel_counts": class_pixel_counts,
        "class_weights": None,
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
