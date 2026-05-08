from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from .dataset import NUM_CLASSES, build_dataloader, compute_class_counts
from .losses import build_syntax_penalties, syntax_loss, weighted_cross_entropy
from .model import InverseImg2Prog
from .palette import infer_palette_mapping


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        optim = importlib.import_module("torch.optim")
    except ImportError as error:
        raise ImportError("PyTorch is required for parser_t_inverse training. Install with `pip install -e .[train]`.") from error
    return torch, optim


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an inverse-style 17-class parser on dataset2.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--palette", type=Path, default=None, help="Optional precomputed color->class mapping JSON")
    parser.add_argument("--syntax-dir", type=Path, default=None, help="Optional syntax directory override")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, nargs=2, default=(160, 160), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--syntax-weight", type=float, default=0.1)
    return parser


def _resolve_device(torch: object, device_name: str) -> object:
    device_cls = getattr(torch, "device")
    if device_name == "cpu":
        return device_cls("cpu")
    if not getattr(torch, "cuda").is_available():
        raise RuntimeError(f"Requested device {device_name!r}, but CUDA is not available.")
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


def _compute_metrics(confusion: list[list[int]]) -> dict[str, object]:
    num_classes = len(confusion)
    correct = sum(confusion[index][index] for index in range(num_classes))
    total = sum(sum(row) for row in confusion)
    pixel_accuracy = correct / total if total else 0.0
    ious: list[float] = []
    per_class_iou: list[float | None] = []
    for class_id in range(num_classes):
        intersection = confusion[class_id][class_id]
        predicted = sum(confusion[row][class_id] for row in range(num_classes))
        actual = sum(confusion[class_id])
        union = actual + predicted - intersection
        if union > 0:
            value = intersection / union
            ious.append(value)
            per_class_iou.append(value)
        else:
            per_class_iou.append(None)
    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "per_class_iou": per_class_iou,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch, optim = _require_torch()
    device = _resolve_device(torch, args.device)
    use_amp = bool(args.amp and str(device).startswith("cuda"))
    if str(device).startswith("cuda") and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    palette_path = args.palette or (args.output_dir / "palette_mapping.json")
    if args.palette is None:
        infer_palette_mapping(args.manifest, palette_path)

    syntax_dir = args.syntax_dir or ((args.manifest.parent.parent / "dataset2" / "syntax") if (args.manifest.parent.parent / "dataset2" / "syntax").exists() else (Path("dataset2") / "syntax"))
    syntax_penalties = build_syntax_penalties(syntax_dir, NUM_CLASSES)

    train_loader, train_dataset = build_dataloader(
        args.manifest,
        palette_path=palette_path,
        batch_size=args.batch_size,
        shuffle=True,
        image_size=(int(args.image_size[0]), int(args.image_size[1])),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    val_loader = None
    if args.val_manifest is not None:
        val_loader, _ = build_dataloader(
            args.val_manifest,
            palette_path=palette_path,
            batch_size=args.batch_size,
            shuffle=False,
            image_size=(int(args.image_size[0]), int(args.image_size[1])),
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
        )

    model = InverseImg2Prog(num_classes=NUM_CLASSES)
    model.to(device)
    optimizer = getattr(optim, "Adam")(model.parameters(), lr=args.learning_rate)
    scaler = getattr(getattr(torch, "cuda"), "amp").GradScaler(enabled=use_amp)

    class_weights = None
    if args.use_class_weights:
        counts = compute_class_counts(train_dataset)
        total = max(1, sum(counts))
        weights = []
        for count in counts:
            weights.append(0.0 if count <= 0 else (total / float(count)) ** 0.5)
        max_weight = max((value for value in weights if value > 0), default=1.0)
        normalized = [value / max_weight if value > 0 else 0.0 for value in weights]
        class_weights = getattr(torch, "tensor")(normalized, dtype=getattr(torch, "float32"), device=device)

    history: list[dict[str, object]] = []
    for epoch in range(args.epochs):
        print(f"epoch {epoch + 1}/{args.epochs}")
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_syntax = 0.0
        batch_count = 0
        for batch in train_loader:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            with _autocast_context(torch, use_amp):
                logits = model(images)
                ce_loss = weighted_cross_entropy(logits, targets, weight=class_weights)
                syn_loss = syntax_loss(logits, syntax_penalties) if args.syntax_weight > 0 else getattr(torch, "tensor")(0.0, device=device)
                loss = ce_loss + args.syntax_weight * syn_loss
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            total_ce += float(ce_loss.item())
            total_syntax += float(syn_loss.item()) if hasattr(syn_loss, "item") else 0.0
            batch_count += 1
            _print_progress("train", batch_count, len(train_loader), f"loss={total_loss / batch_count:.6f} ce={total_ce / batch_count:.6f} syntax={total_syntax / batch_count:.6f}")
        _finish_progress()

        epoch_metrics: dict[str, object] = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, batch_count),
            "train_ce": total_ce / max(1, batch_count),
            "train_syntax": total_syntax / max(1, batch_count),
        }

        if val_loader is not None:
            model.eval()
            val_total = 0.0
            val_confusion = [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)]
            with getattr(torch, "no_grad")():
                val_batches = 0
                for batch in val_loader:
                    images = batch["images"].to(device)
                    targets = batch["targets"].to(device)
                    with _autocast_context(torch, use_amp):
                        logits = model(images)
                        val_loss = weighted_cross_entropy(logits, targets, weight=class_weights)
                    predictions = getattr(torch, "argmax")(logits, dim=1).detach().cpu().tolist()
                    target_rows = targets.detach().cpu().tolist()
                    for sample_index, pred_mask in enumerate(predictions):
                        tgt_mask = target_rows[sample_index]
                        for row_index, row in enumerate(tgt_mask):
                            for col_index, actual in enumerate(row):
                                val_confusion[actual][pred_mask[row_index][col_index]] += 1
                    val_total += float(val_loss.item())
                    val_batches += 1
                    _print_progress("val", val_batches, len(val_loader), f"loss={val_total / val_batches:.6f}")
            _finish_progress()
            metrics = _compute_metrics(val_confusion)
            epoch_metrics["val_loss"] = val_total / max(1, len(val_loader))
            epoch_metrics["val_pixel_accuracy"] = metrics["pixel_accuracy"]
            epoch_metrics["val_mean_iou"] = metrics["mean_iou"]
            epoch_metrics["val_per_class_iou"] = metrics["per_class_iou"]
            print(
                f"epoch={epoch + 1} train_loss={cast(float, epoch_metrics['train_loss']):.6f} "
                f"val_loss={cast(float, epoch_metrics['val_loss']):.6f} "
                f"val_acc={cast(float, epoch_metrics['val_pixel_accuracy']):.4f} "
                f"val_miou={cast(float, epoch_metrics['val_mean_iou']):.4f}"
            )
        else:
            print(f"epoch={epoch + 1} train_loss={cast(float, epoch_metrics['train_loss']):.6f}")
        history.append(epoch_metrics)

    metrics = {
        "manifest": str(args.manifest),
        "val_manifest": str(args.val_manifest) if args.val_manifest is not None else None,
        "palette": str(palette_path),
        "syntax_dir": str(syntax_dir),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": [int(args.image_size[0]), int(args.image_size[1])],
        "learning_rate": args.learning_rate,
        "device": str(device),
        "amp": use_amp,
        "use_class_weights": bool(args.use_class_weights),
        "syntax_weight": args.syntax_weight,
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
