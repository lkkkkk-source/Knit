from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from knit_decode.parser_t_inverse.losses import build_syntax_penalties, syntax_loss

from .dataset import build_dataloader
from .losses import gan_hinge_discriminator_loss, gan_hinge_generator_loss, l1_loss, parser_cross_entropy
from .model import ConditionalPatchDiscriminator, RefinerTransformer
from .teacher import FrozenInverseParser


def _require_torch() -> tuple[object, object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        optim = importlib.import_module("torch.optim")
        functional = importlib.import_module("torch.nn.functional")
    except ImportError as error:
        raise ImportError("PyTorch is required for refiner_inverse_v1 training. Install with `pip install -e .[train]`.") from error
    return torch, optim, functional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an inverse-style real-to-regularized refiner with frozen parser supervision.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--parser-checkpoint", type=Path, required=True, help="Frozen parser_t_inverse checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, nargs=2, default=(160, 160), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--recon-weight", type=float, default=100.0)
    parser.add_argument("--parser-weight", type=float, default=2.0)
    parser.add_argument("--adv-weight", type=float, default=0.2)
    parser.add_argument("--syntax-weight", type=float, default=0.05)
    parser.add_argument("--num-vis", type=int, default=8)
    parser.add_argument("--syntax-dir", type=Path, default=Path("dataset2/syntax"))
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


def _to_gray(images: object) -> object:
    return images.mean(dim=1, keepdim=True)


def _save_gray_tensor(tensor: object, output_path: Path) -> None:
    torch, _, _ = _require_torch()
    image = tensor.detach().cpu().clamp(-0.5, 0.5)
    image = ((image + 0.5) * 255.0).round().to(dtype=getattr(torch, "uint8"))
    height = int(image.shape[-2])
    width = int(image.shape[-1])
    flat = image.reshape(-1).tolist()
    from PIL import Image
    output = Image.new("L", (width, height))
    output.putdata(flat)
    output.save(output_path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch, optim, functional = _require_torch()
    device = _resolve_device(torch, args.device)
    use_amp = bool(args.amp and str(device).startswith("cuda"))
    if str(device).startswith("cuda") and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    train_loader, train_dataset = build_dataloader(
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
        val_loader, _ = build_dataloader(
            args.val_manifest,
            batch_size=args.batch_size,
            shuffle=False,
            image_size=(int(args.image_size[0]), int(args.image_size[1])),
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
        )

    generator = RefinerTransformer()
    discriminator = ConditionalPatchDiscriminator()
    generator.to(device)
    discriminator.to(device)
    parser_model = FrozenInverseParser(args.parser_checkpoint, device=device)
    syntax_penalties = build_syntax_penalties(args.syntax_dir, 17)

    optimizer_g = getattr(optim, "Adam")(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_d = getattr(optim, "Adam")(discriminator.parameters(), lr=args.learning_rate * 0.5, betas=(0.5, 0.999))
    scaler_g = getattr(getattr(torch, "cuda"), "amp").GradScaler(enabled=use_amp)
    scaler_d = getattr(getattr(torch, "cuda"), "amp").GradScaler(enabled=use_amp)

    history: list[dict[str, object]] = []
    best_val = None
    for epoch in range(args.epochs):
        print(f"epoch {epoch + 1}/{args.epochs}")
        generator.train()
        discriminator.train()
        total_g = 0.0
        total_d = 0.0
        total_recon = 0.0
        total_parser = 0.0
        total_syntax = 0.0
        batch_count = 0
        total_batches = len(train_loader)

        for batch in train_loader:
            sources_rgb = batch["sources"].to(device)
            targets_rgb = batch["targets"].to(device)
            source_gray = _to_gray(sources_rgb)
            target_gray = _to_gray(targets_rgb)

            with getattr(torch, "no_grad")():
                target_logits = parser_model(target_gray)
                target_instruction = getattr(torch, "argmax")(target_logits, dim=1)
                target_onehot = functional.one_hot(target_instruction, num_classes=17).permute(0, 3, 1, 2).float()

            # discriminator step
            optimizer_d.zero_grad()
            with _autocast_context(torch, use_amp):
                fake_gray = generator(source_gray).detach()
                real_logits = discriminator(target_gray, target_onehot)
                fake_logits = discriminator(fake_gray, target_onehot)
                loss_d = gan_hinge_discriminator_loss(real_logits, fake_logits)
            if use_amp:
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()
            else:
                loss_d.backward()
                optimizer_d.step()

            # generator step
            optimizer_g.zero_grad()
            with _autocast_context(torch, use_amp):
                fake_gray = generator(source_gray)
                fake_disc_logits = discriminator(fake_gray, target_onehot)
                adv_loss = gan_hinge_generator_loss(fake_disc_logits)
                recon_loss = l1_loss(fake_gray, target_gray)
                parser_logits = parser_model(fake_gray)
                parser_loss = parser_cross_entropy(parser_logits, target_instruction)
                syn_loss = syntax_loss(parser_logits, syntax_penalties) if args.syntax_weight > 0 else getattr(torch, "tensor")(0.0, device=device)
                loss_g = (
                    args.recon_weight * recon_loss
                    + args.parser_weight * parser_loss
                    + args.adv_weight * adv_loss
                    + args.syntax_weight * syn_loss
                )
            if use_amp:
                scaler_g.scale(loss_g).backward()
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                loss_g.backward()
                optimizer_g.step()

            total_g += float(loss_g.item())
            total_d += float(loss_d.item())
            total_recon += float(recon_loss.item())
            total_parser += float(parser_loss.item())
            total_syntax += float(syn_loss.item()) if hasattr(syn_loss, "item") else 0.0
            batch_count += 1
            _print_progress(
                "train",
                batch_count,
                total_batches,
                f"g={total_g / batch_count:.4f} d={total_d / batch_count:.4f} recon={total_recon / batch_count:.4f} parser={total_parser / batch_count:.4f}",
            )
        _finish_progress()

        epoch_metrics: dict[str, object] = {
            "epoch": epoch + 1,
            "train_generator_loss": total_g / max(1, batch_count),
            "train_discriminator_loss": total_d / max(1, batch_count),
            "train_recon_loss": total_recon / max(1, batch_count),
            "train_parser_loss": total_parser / max(1, batch_count),
            "train_syntax_loss": total_syntax / max(1, batch_count),
        }

        if val_loader is not None:
            generator.eval()
            val_total = 0.0
            vis_dir = args.output_dir / "val_predictions"
            vis_dir.mkdir(parents=True, exist_ok=True)
            vis_written = 0
            with getattr(torch, "no_grad")():
                val_batches = 0
                for batch in val_loader:
                    sources_rgb = batch["sources"].to(device)
                    targets_rgb = batch["targets"].to(device)
                    source_gray = _to_gray(sources_rgb)
                    target_gray = _to_gray(targets_rgb)
                    fake_gray = generator(source_gray)
                    recon_loss = l1_loss(fake_gray, target_gray)
                    target_logits = parser_model(target_gray)
                    target_instruction = getattr(torch, "argmax")(target_logits, dim=1)
                    parser_logits = parser_model(fake_gray)
                    parser_loss = parser_cross_entropy(parser_logits, target_instruction)
                    val_loss = args.recon_weight * recon_loss + args.parser_weight * parser_loss
                    val_total += float(val_loss.item())
                    val_batches += 1
                    _print_progress("val", val_batches, len(val_loader), f"loss={val_total / val_batches:.4f}")
                    if vis_written < args.num_vis:
                        count = min(args.num_vis - vis_written, fake_gray.shape[0])
                        for sample_index in range(count):
                            sample_id = str(batch["sample_ids"][sample_index]).replace("/", "__")
                            sample_dir = vis_dir / sample_id
                            sample_dir.mkdir(parents=True, exist_ok=True)
                            _save_gray_tensor(source_gray[sample_index], sample_dir / "input.png")
                            _save_gray_tensor(fake_gray[sample_index], sample_dir / "pred.png")
                            _save_gray_tensor(target_gray[sample_index], sample_dir / "target.png")
                            vis_written += 1
            _finish_progress()
            epoch_metrics["val_loss"] = val_total / max(1, len(val_loader))
            current_val = cast(float, epoch_metrics["val_loss"])
            if best_val is None or current_val < best_val:
                best_val = current_val
                getattr(torch, "save")(
                    {
                        "generator_state_dict": generator.state_dict(),
                        "discriminator_state_dict": discriminator.state_dict(),
                        "metrics": epoch_metrics,
                    },
                    args.output_dir / "best.pt",
                )
            print(
                f"epoch={epoch + 1} train_g={cast(float, epoch_metrics['train_generator_loss']):.4f} "
                f"train_d={cast(float, epoch_metrics['train_discriminator_loss']):.4f} "
                f"val_loss={current_val:.4f}"
            )
        else:
            print(
                f"epoch={epoch + 1} train_g={cast(float, epoch_metrics['train_generator_loss']):.4f} "
                f"train_d={cast(float, epoch_metrics['train_discriminator_loss']):.4f}"
            )

        history.append(epoch_metrics)

    metrics = {
        "manifest": str(args.manifest),
        "val_manifest": str(args.val_manifest) if args.val_manifest is not None else None,
        "parser_checkpoint": str(args.parser_checkpoint),
        "syntax_dir": str(args.syntax_dir),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": [int(args.image_size[0]), int(args.image_size[1])],
        "learning_rate": args.learning_rate,
        "device": str(device),
        "amp": use_amp,
        "recon_weight": args.recon_weight,
        "parser_weight": args.parser_weight,
        "adv_weight": args.adv_weight,
        "syntax_weight": args.syntax_weight,
        "num_train_samples": len(train_dataset),
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    getattr(torch, "save")(
        {
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "metrics": metrics,
        },
        args.output_dir / "checkpoint.pt",
    )
    print(f"saved metrics: {args.output_dir / 'metrics.json'}")
    print(f"saved checkpoint: {args.output_dir / 'checkpoint.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
