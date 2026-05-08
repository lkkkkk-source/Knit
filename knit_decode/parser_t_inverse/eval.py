from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from PIL import Image

from .dataset import NUM_CLASSES, build_dataloader
from .losses import build_syntax_penalties, syntax_loss, weighted_cross_entropy
from .model import InverseImg2Prog
from .palette import OFFICIAL_PALETTE


def _require_torch() -> object:
    import importlib

    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required for parser_t_inverse evaluation. Install with `pip install -e .[train]`.") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an inverse-style parser checkpoint.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--palette", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-vis", type=int, default=16)
    parser.add_argument("--syntax-dir", type=Path, default=None)
    parser.add_argument("--syntax-weight", type=float, default=0.0)
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


def _save_grayscale(tensor: object, output_path: Path) -> None:
    torch = _require_torch()
    image = tensor.detach().cpu().clamp(0.0, 1.0)
    image = (image * 255.0).round().to(dtype=getattr(torch, "uint8"))
    height = int(image.shape[-2])
    width = int(image.shape[-1])
    flat = image.reshape(-1).tolist()
    output = Image.new("L", (width, height))
    output.putdata(flat)
    output.save(output_path)


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

    torch = _require_torch()
    device = _resolve_device(torch, args.device)
    use_amp = bool(args.amp and str(device).startswith("cuda"))
    if str(device).startswith("cuda") and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    checkpoint = cast(dict[str, object], getattr(torch, "load")(args.checkpoint.resolve(), map_location="cpu"))
    checkpoint_metrics = cast(dict[str, object], checkpoint.get("metrics", {}))
    image_size = tuple(int(value) for value in checkpoint_metrics.get("image_size", [160, 160]))
    syntax_dir = args.syntax_dir or Path(str(checkpoint_metrics.get("syntax_dir", "dataset2/syntax")))
    syntax_penalties = build_syntax_penalties(syntax_dir, NUM_CLASSES) if args.syntax_weight > 0 else None

    dataloader, _ = build_dataloader(
        args.manifest,
        palette_path=args.palette,
        batch_size=args.batch_size,
        shuffle=False,
        image_size=cast(tuple[int, int], image_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )

    model = InverseImg2Prog(num_classes=NUM_CLASSES)
    model.load_state_dict(cast(dict[str, object], checkpoint["model_state_dict"]))
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_syntax = 0.0
    batch_count = 0
    confusion = [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)]
    vis_dir = args.output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_written = 0

    with getattr(torch, "no_grad")():
        for batch in dataloader:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            with _autocast_context(torch, use_amp):
                logits = model(images)
                ce_loss = weighted_cross_entropy(logits, targets)
                syn_loss = syntax_loss(logits, syntax_penalties) if syntax_penalties is not None else getattr(torch, "tensor")(0.0, device=device)
                loss = ce_loss + args.syntax_weight * syn_loss
            total_loss += float(loss.item())
            total_syntax += float(syn_loss.item()) if hasattr(syn_loss, "item") else 0.0
            batch_count += 1
            _print_progress("eval", batch_count, len(dataloader), f"loss={total_loss / batch_count:.6f}")

            predictions = getattr(torch, "argmax")(logits, dim=1).detach().cpu().tolist()
            target_rows = targets.detach().cpu().tolist()
            sample_ids = [str(value) for value in batch["sample_ids"]]
            for sample_index, pred_mask in enumerate(predictions):
                tgt_mask = target_rows[sample_index]
                for row_index, row in enumerate(tgt_mask):
                    for col_index, actual in enumerate(row):
                        confusion[actual][pred_mask[row_index][col_index]] += 1
                if vis_written < args.num_vis:
                    sample_dir = vis_dir / sample_ids[sample_index].replace("/", "__")
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    _save_grayscale(images[sample_index], sample_dir / "input.png")
                    _save_label_map(pred_mask, sample_dir / "pred.png")
                    _save_label_map(tgt_mask, sample_dir / "target.png")
                    vis_written += 1
    _finish_progress()

    metrics = _compute_metrics(confusion)
    result = {
        "manifest": str(args.manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "palette": str(args.palette.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "device": str(device),
        "image_size": list(image_size),
        "loss": total_loss / max(1, batch_count),
        "syntax_loss": total_syntax / max(1, batch_count),
        "pixel_accuracy": metrics["pixel_accuracy"],
        "mean_iou": metrics["mean_iou"],
        "per_class_iou": metrics["per_class_iou"],
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
