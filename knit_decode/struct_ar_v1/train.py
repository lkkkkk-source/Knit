from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from PIL import Image

from knit_decode.parser_t_inverse.losses import build_syntax_penalties, syntax_loss, weighted_cross_entropy
from knit_decode.parser_t_inverse.palette import OFFICIAL_PALETTE, infer_palette_mapping

from .dataset import NUM_CLASSES, build_dataloader, compute_class_counts
from .model import MultiScaleStructureTransformer


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        optim = importlib.import_module("torch.optim")
    except ImportError as error:
        raise ImportError("PyTorch is required for struct_ar_v1 training. Install with `pip install -e .[train]`.") from error
    return torch, optim


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a multi-scale 2D AR transformer for category->instruction17.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--palette", type=Path, default=None)
    parser.add_argument("--syntax-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--syntax-weight", type=float, default=0.1)
    parser.add_argument("--loss-weight-5", type=float, default=0.5)
    parser.add_argument("--loss-weight-10", type=float, default=0.75)
    parser.add_argument("--loss-weight-20", type=float, default=1.0)
    parser.add_argument("--teacher-forcing", action="store_true")
    parser.add_argument("--num-vis", type=int, default=16)
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


def _save_label_map(mask: list[list[int]], output_path: Path) -> None:
    height = len(mask)
    width = len(mask[0]) if height else 0
    image = Image.new("P", (width, height))
    for y_pos, row in enumerate(mask):
        for x_pos, class_id in enumerate(row):
            image.putpixel((x_pos, y_pos), int(class_id))
    palette = []
    for color in OFFICIAL_PALETTE:
        palette.extend(color)
    palette.extend([0] * (768 - len(palette)))
    image.putpalette(palette)
    image.save(output_path)


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
    syntax_dir = args.syntax_dir or (Path("dataset2") / "syntax")
    syntax_penalties = build_syntax_penalties(syntax_dir, NUM_CLASSES)

    train_loader, train_dataset = build_dataloader(
        args.manifest,
        palette_path=palette_path,
        batch_size=args.batch_size,
        shuffle=True,
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
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
        )

    model = MultiScaleStructureTransformer(num_categories=len(train_dataset.category_to_id), num_classes=NUM_CLASSES)
    model.to(device)
    optimizer = getattr(optim, "AdamW")(model.parameters(), lr=args.learning_rate)
    scaler = getattr(getattr(torch, "cuda"), "amp").GradScaler(enabled=use_amp)

    class_weights = None
    if args.use_class_weights:
        counts = compute_class_counts(train_dataset.base)
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
        total_ce5 = 0.0
        total_ce10 = 0.0
        total_ce20 = 0.0
        total_syntax = 0.0
        batch_count = 0
        for batch in train_loader:
            category_ids = batch["category_ids"].to(device)
            grid5 = batch["grid5"].to(device)
            grid10 = batch["grid10"].to(device)
            grid20 = batch["grid20"].to(device)
            with _autocast_context(torch, use_amp):
                outputs = model(category_ids, grid5, grid10, grid20, teacher_forcing=bool(args.teacher_forcing))
                ce5 = weighted_cross_entropy(outputs["logits5"], grid5, weight=class_weights)
                ce10 = weighted_cross_entropy(outputs["logits10"], grid10, weight=class_weights)
                ce20 = weighted_cross_entropy(outputs["logits20"], grid20, weight=class_weights)
                syn = syntax_loss(outputs["logits20"], syntax_penalties) if args.syntax_weight > 0 else getattr(torch, "tensor")(0.0, device=device)
                loss = args.loss_weight_5 * ce5 + args.loss_weight_10 * ce10 + args.loss_weight_20 * ce20 + args.syntax_weight * syn
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            total_ce5 += float(ce5.item())
            total_ce10 += float(ce10.item())
            total_ce20 += float(ce20.item())
            total_syntax += float(syn.item()) if hasattr(syn, "item") else 0.0
            batch_count += 1
            _print_progress(
                "train",
                batch_count,
                len(train_loader),
                (
                    f"loss={total_loss / batch_count:.6f} "
                    f"ce5={total_ce5 / batch_count:.6f} "
                    f"ce10={total_ce10 / batch_count:.6f} "
                    f"ce20={total_ce20 / batch_count:.6f} "
                    f"syntax={total_syntax / batch_count:.6f}"
                ),
            )
        _finish_progress()

        epoch_metrics: dict[str, object] = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, batch_count),
            "train_ce5": total_ce5 / max(1, batch_count),
            "train_ce10": total_ce10 / max(1, batch_count),
            "train_ce20": total_ce20 / max(1, batch_count),
            "train_syntax": total_syntax / max(1, batch_count),
        }

        if val_loader is not None:
            model.eval()
            val_total = 0.0
            val_confusion = [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)]
            vis_dir = args.output_dir / "val_predictions" / f"epoch_{epoch + 1:03d}"
            vis_dir.mkdir(parents=True, exist_ok=True)
            vis_written = 0
            with getattr(torch, "no_grad")():
                val_batches = 0
                for batch in val_loader:
                    category_ids = batch["category_ids"].to(device)
                    grid5 = batch["grid5"].to(device)
                    grid10 = batch["grid10"].to(device)
                    grid20 = batch["grid20"].to(device)
                    sample_ids = [str(value) for value in batch["sample_ids"]]
                    with _autocast_context(torch, use_amp):
                        outputs = model(category_ids, grid5, grid10, grid20, teacher_forcing=False)
                        val_loss = weighted_cross_entropy(outputs["logits20"], grid20, weight=class_weights)
                    predictions = outputs["pred20"].detach().cpu().tolist()
                    targets = grid20.detach().cpu().tolist()
                    for sample_index, pred_mask in enumerate(predictions):
                        tgt_mask = targets[sample_index]
                        for row_index, row in enumerate(tgt_mask):
                            for col_index, actual in enumerate(row):
                                val_confusion[actual][pred_mask[row_index][col_index]] += 1
                        if vis_written < args.num_vis:
                            sample_dir = vis_dir / sample_ids[sample_index].replace("/", "__")
                            sample_dir.mkdir(parents=True, exist_ok=True)
                            _save_label_map(pred_mask, sample_dir / "pred20.png")
                            _save_label_map(tgt_mask, sample_dir / "target20.png")
                            vis_written += 1
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

    result = {
        "manifest": str(args.manifest),
        "val_manifest": str(args.val_manifest) if args.val_manifest is not None else None,
        "palette": str(palette_path),
        "syntax_dir": str(syntax_dir),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "amp": use_amp,
        "num_classes": NUM_CLASSES,
        "num_categories": len(train_dataset.category_to_id),
        "category_to_id": train_dataset.category_to_id,
        "syntax_weight": args.syntax_weight,
        "loss_weight_5": args.loss_weight_5,
        "loss_weight_10": args.loss_weight_10,
        "loss_weight_20": args.loss_weight_20,
        "teacher_forcing": bool(args.teacher_forcing),
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    getattr(torch, "save")(
        {
            "model_state_dict": model.state_dict(),
            "metrics": result,
        },
        args.output_dir / "checkpoint.pt",
    )
    print(f"saved metrics: {args.output_dir / 'metrics.json'}")
    print(f"saved checkpoint: {args.output_dir / 'checkpoint.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
