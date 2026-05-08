from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from .dataset import build_refiner_dataloader
from .model import RefinerUNet


def _require_torch() -> tuple[object, object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        optim = importlib.import_module("torch.optim")
        functional = importlib.import_module("torch.nn.functional")
    except ImportError as error:
        raise ImportError("PyTorch is required for refiner-v1 training. Install with `pip install -e .[train]`.") from error
    return torch, optim, functional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a minimal paired real-to-rendering refiner.")
    parser.add_argument("--manifest", type=Path, required=True, help="Training JSONL manifest")
    parser.add_argument("--val-manifest", type=Path, default=None, help="Optional validation manifest")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, nargs=2, default=(160, 160), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--l1-weight", type=float, default=100.0, help="Weight for pixel L1 reconstruction loss")
    parser.add_argument("--num-vis", type=int, default=8, help="Number of validation predictions to export")
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch, optim, functional = _require_torch()
    device = _resolve_device(torch, args.device)
    use_amp = bool(args.amp and str(device).startswith("cuda"))
    if str(device).startswith("cuda"):
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True

    train_loader, train_dataset = build_refiner_dataloader(
        args.manifest,
        batch_size=args.batch_size,
        shuffle=True,
        image_size=(int(args.image_size[0]), int(args.image_size[1])),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    val_loader = None
    if args.val_manifest is not None:
        val_loader, _ = build_refiner_dataloader(
            args.val_manifest,
            batch_size=args.batch_size,
            shuffle=False,
            image_size=(int(args.image_size[0]), int(args.image_size[1])),
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
        )

    model = RefinerUNet()
    model.to(device)
    optimizer = getattr(optim, "Adam")(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    scaler = getattr(getattr(torch, "cuda"), "amp").GradScaler(enabled=use_amp)

    history: list[dict[str, object]] = []
    best_val = None
    for epoch in range(args.epochs):
        print(f"epoch {epoch + 1}/{args.epochs}")
        model.train()
        total_loss = 0.0
        batch_count = 0
        total_batches = len(train_loader)
        for batch in train_loader:
            sources = batch["sources"].to(device)
            targets = batch["targets"].to(device)
            with _autocast_context(torch, use_amp):
                predictions = model(sources)
                l1_loss = functional.l1_loss(predictions, targets)
                loss = args.l1_weight * l1_loss
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1
            _print_progress("train", batch_count, total_batches, f"loss={total_loss / batch_count:.6f}")
        _finish_progress()

        epoch_metrics: dict[str, object] = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, batch_count),
        }

        if val_loader is not None:
            model.eval()
            val_total = 0.0
            val_batches = 0
            vis_dir = args.output_dir / "val_predictions"
            vis_dir.mkdir(parents=True, exist_ok=True)
            vis_written = 0
            with getattr(torch, "no_grad")():
                for batch in val_loader:
                    sources = batch["sources"].to(device)
                    targets = batch["targets"].to(device)
                    with _autocast_context(torch, use_amp):
                        predictions = model(sources)
                        val_loss = args.l1_weight * functional.l1_loss(predictions, targets)
                    val_total += float(val_loss.item())
                    val_batches += 1
                    _print_progress("val", val_batches, len(val_loader), f"loss={val_total / val_batches:.6f}")
                    if vis_written < args.num_vis:
                        sample_count = min(args.num_vis - vis_written, predictions.shape[0])
                        for sample_index in range(sample_count):
                            sample_id = str(batch["sample_ids"][sample_index]).replace("/", "__")
                            sample_dir = vis_dir / sample_id
                            sample_dir.mkdir(parents=True, exist_ok=True)
                            for name, tensor in {
                                "input.png": sources[sample_index],
                                "pred.png": predictions[sample_index],
                                "target.png": targets[sample_index],
                            }.items():
                                image = tensor.detach().cpu().clamp(-1.0, 1.0)
                                image = ((image + 1.0) * 127.5).round().to(dtype=getattr(torch, "uint8")).permute(1, 2, 0)
                                from PIL import Image
                                Image.fromarray(image.numpy(), mode="RGB").save(sample_dir / name)
                            vis_written += 1
            _finish_progress()
            epoch_metrics["val_loss"] = val_total / max(1, val_batches)
            current_val = cast(float, epoch_metrics["val_loss"])
            if best_val is None or current_val < best_val:
                best_val = current_val
                getattr(torch, "save")({"model_state_dict": model.state_dict(), "metrics": epoch_metrics}, args.output_dir / "best.pt")
            print(
                f"epoch={epoch + 1} train_loss={cast(float, epoch_metrics['train_loss']):.6f} "
                f"val_loss={current_val:.6f}"
            )
        else:
            print(f"epoch={epoch + 1} train_loss={cast(float, epoch_metrics['train_loss']):.6f}")

        history.append(epoch_metrics)

    metrics = {
        "manifest": str(args.manifest),
        "val_manifest": str(args.val_manifest) if args.val_manifest is not None else None,
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": [int(args.image_size[0]), int(args.image_size[1])],
        "learning_rate": args.learning_rate,
        "device": str(device),
        "amp": use_amp,
        "l1_weight": args.l1_weight,
        "num_train_samples": len(train_dataset),
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
