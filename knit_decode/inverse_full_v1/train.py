from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from knit_decode.parser_t_inverse.losses import build_syntax_penalties, syntax_loss

from .dataset import build_dataloader
from .losses import (
    cross_entropy_loss,
    gan_hinge_discriminator_loss,
    gan_hinge_generator_loss,
    l1_loss,
    mil_cross_entropy_loss,
)
from .model import InverseImg2Prog, ResidualConditionalPatchDiscriminator, ResidualRefiner
from .perceptual import VGGFeatureExtractor, gram_matrix


def _require_torch() -> tuple[object, object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        optim = importlib.import_module("torch.optim")
        functional = importlib.import_module("torch.nn.functional")
    except ImportError as error:
        raise ImportError("PyTorch is required for inverse_full_v1 training. Install with `pip install -e .[train]`.") from error
    return torch, optim, functional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyTorch attempt at a closer original inverse-style joint system.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--parser-init", type=Path, default=None)
    parser.add_argument("--palette", type=Path, default=Path("parser_t_inverse/palette_mapping.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, nargs=2, default=(160, 160), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-best-crop", action="store_true")
    parser.add_argument("--augment-scale", type=float, default=4.0)
    parser.add_argument("--use-transfer", action="store_true")
    parser.add_argument("--transfer-root", type=Path, default=Path("dataset2/transfer/Cable1_019_0_19/gray"))
    parser.add_argument("--mean-value", type=float, default=0.5)
    parser.add_argument("--ce-weight", type=float, default=2.0)
    parser.add_argument("--mil-weight", type=float, default=2.0)
    parser.add_argument("--gan-weight", type=float, default=0.2)
    parser.add_argument("--syntax-weight", type=float, default=0.1)
    parser.add_argument("--perceptual-weight", type=float, default=0.02 / (128.0 * 128.0))
    parser.add_argument("--style-weight", type=float, default=0.2)
    parser.add_argument("--transfer-weight", type=float, default=1.0)
    parser.add_argument("--vgg-variant", type=str, default="16", choices=("16", "19"))
    parser.add_argument("--syntax-dir", type=Path, default=Path("dataset2/syntax"))
    parser.add_argument("--num-vis", type=int, default=8)
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


def _save_gray(tensor: object, output_path: Path, residual: bool = False, mean_value: float = 0.5) -> None:
    torch, _, _ = _require_torch()
    image = tensor.detach().cpu()
    if residual:
        image = image + mean_value
    image = image.clamp(0.0, 1.0)
    image = (image * 255.0).round().to(dtype=getattr(torch, "uint8"))
    from PIL import Image
    output = Image.new("L", (int(image.shape[-1]), int(image.shape[-2])))
    output.putdata(image.reshape(-1).tolist())
    output.save(output_path)


def _init_parser(parser_model: object, checkpoint_path: Path | None) -> None:
    if checkpoint_path is None:
        return
    torch, _, _ = _require_torch()
    checkpoint = cast(dict[str, object], getattr(torch, "load")(checkpoint_path, map_location="cpu"))
    parser_model.load_state_dict(cast(dict[str, object], checkpoint["model_state_dict"]))


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
        palette_path=args.palette,
        batch_size=args.batch_size,
        shuffle=True,
        image_size=(int(args.image_size[0]), int(args.image_size[1])),
        transfer_root=args.transfer_root if args.use_transfer else None,
        use_best_crop=args.use_best_crop,
        augment_scale=args.augment_scale,
        mean_value=args.mean_value,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    val_loader = None
    if args.val_manifest is not None:
        val_loader, _ = build_dataloader(
            args.val_manifest,
            palette_path=args.palette,
            batch_size=args.batch_size,
            shuffle=False,
            image_size=(int(args.image_size[0]), int(args.image_size[1])),
            transfer_root=args.transfer_root if args.use_transfer else None,
            use_best_crop=args.use_best_crop,
            augment_scale=0.0,
            mean_value=args.mean_value,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
        )

    refiner = ResidualRefiner()
    parser_head = InverseImg2Prog(num_classes=17)
    discriminator = ResidualConditionalPatchDiscriminator()
    _init_parser(parser_head, args.parser_init)
    refiner.to(device)
    parser_head.to(device)
    discriminator.to(device)
    syntax_penalties = build_syntax_penalties(args.syntax_dir, 17)
    vgg = VGGFeatureExtractor(variant=args.vgg_variant, device=device)

    optimizer_g = getattr(optim, "Adam")(list(refiner.parameters()) + list(parser_head.parameters()), lr=args.learning_rate, betas=(0.5, 0.999), eps=1e-4)
    optimizer_d = getattr(optim, "Adam")(discriminator.parameters(), lr=args.learning_rate * 0.5, betas=(0.5, 0.999), eps=1e-4)
    scaler_g = getattr(getattr(torch, "cuda"), "amp").GradScaler(enabled=use_amp)
    scaler_d = getattr(getattr(torch, "cuda"), "amp").GradScaler(enabled=use_amp)

    history: list[dict[str, object]] = []
    best_val = None
    for epoch in range(args.epochs):
        print(f"epoch {epoch + 1}/{args.epochs}")
        refiner.train()
        parser_head.train()
        discriminator.train()
        totals = {
            "g": 0.0,
            "d": 0.0,
            "ce": 0.0,
            "mil": 0.0,
            "gan": 0.0,
            "perc": 0.0,
            "style": 0.0,
            "syntax": 0.0,
            "transfer": 0.0,
        }
        batch_count = 0
        for batch in train_loader:
            source_res = batch["source_residual"].to(device)
            target_res = batch["target_residual"].to(device)
            instruction_target = batch["instruction_target"].to(device)
            target_onehot = functional.one_hot(instruction_target, num_classes=17).permute(0, 3, 1, 2).float()

            optimizer_d.zero_grad()
            with _autocast_context(torch, use_amp):
                fake_res = refiner(source_res).detach()
                real_logits = discriminator(target_res, target_onehot)
                fake_logits = discriminator(fake_res, target_onehot)
                loss_d = gan_hinge_discriminator_loss(real_logits, fake_logits)
            if use_amp:
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()
            else:
                loss_d.backward()
                optimizer_d.step()

            optimizer_g.zero_grad()
            with _autocast_context(torch, use_amp):
                fake_res = refiner(source_res)
                fake_disc_logits = discriminator(fake_res, target_onehot)
                gan_loss = gan_hinge_generator_loss(fake_disc_logits)
                logits_real = parser_head(fake_res)
                ce_loss = cross_entropy_loss(logits_real, instruction_target)
                mil_loss = mil_cross_entropy_loss(logits_real, instruction_target)
                syntax_reg = syntax_loss(logits_real, syntax_penalties)
                fake_img = (fake_res + args.mean_value).clamp(0.0, 1.0)
                target_img = (target_res + args.mean_value).clamp(0.0, 1.0)
                fake_feats = vgg(fake_img, ["pool3", "conv1_2", "conv2_2", "conv3_3"])
                target_feats = vgg(target_img, ["pool3", "conv1_2", "conv2_2", "conv3_3"])
                perc_loss = functional.mse_loss(fake_feats["pool3"] / 128.0, target_feats["pool3"] / 128.0)
                style_terms = []
                for layer_name, layer_weight in [("conv1_2", 0.3), ("conv2_2", 0.5), ("conv3_3", 1.0)]:
                    style_terms.append(layer_weight * functional.l1_loss(gram_matrix(fake_feats[layer_name] / 128.0), gram_matrix(target_feats[layer_name] / 128.0)))
                style_loss = sum(style_terms) / max(1, len(style_terms))
                transfer_loss = getattr(torch, "tensor")(0.0, device=device)
                if args.use_transfer and batch["transfer_residual"] is not None and batch["transfer_mask"] is not None:
                    transfer_mask = batch["transfer_mask"].to(device)
                    if bool(getattr(transfer_mask, "any")().item()):
                        fake_transfer = fake_res[transfer_mask]
                        target_transfer = batch["transfer_residual"].to(device)[transfer_mask]
                        transfer_loss = l1_loss(fake_transfer, target_transfer)
                loss_g = (
                    args.ce_weight * ce_loss
                    + args.mil_weight * mil_loss
                    + args.gan_weight * gan_loss
                    + args.syntax_weight * syntax_reg
                    + args.perceptual_weight * perc_loss
                    + args.style_weight * style_loss
                    + args.transfer_weight * transfer_loss
                )
            if use_amp:
                scaler_g.scale(loss_g).backward()
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                loss_g.backward()
                optimizer_g.step()

            totals["g"] += float(loss_g.item())
            totals["d"] += float(loss_d.item())
            totals["ce"] += float(ce_loss.item())
            totals["mil"] += float(mil_loss.item())
            totals["gan"] += float(gan_loss.item())
            totals["perc"] += float(perc_loss.item())
            totals["style"] += float(style_loss.item())
            totals["syntax"] += float(syntax_reg.item())
            totals["transfer"] += float(transfer_loss.item()) if hasattr(transfer_loss, "item") else 0.0
            batch_count += 1
            _print_progress("train", batch_count, len(train_loader), f"g={totals['g']/batch_count:.4f} d={totals['d']/batch_count:.4f} ce={totals['ce']/batch_count:.4f} mil={totals['mil']/batch_count:.4f}")
        _finish_progress()

        epoch_metrics: dict[str, object] = {
            "epoch": epoch + 1,
            "train_generator_loss": totals["g"] / max(1, batch_count),
            "train_discriminator_loss": totals["d"] / max(1, batch_count),
            "train_ce_loss": totals["ce"] / max(1, batch_count),
            "train_mil_loss": totals["mil"] / max(1, batch_count),
            "train_gan_loss": totals["gan"] / max(1, batch_count),
            "train_perceptual_loss": totals["perc"] / max(1, batch_count),
            "train_style_loss": totals["style"] / max(1, batch_count),
            "train_syntax_loss": totals["syntax"] / max(1, batch_count),
            "train_transfer_loss": totals["transfer"] / max(1, batch_count),
        }

        if val_loader is not None:
            refiner.eval()
            parser_head.eval()
            val_total = 0.0
            vis_dir = args.output_dir / "val_predictions"
            vis_dir.mkdir(parents=True, exist_ok=True)
            vis_written = 0
            with getattr(torch, "no_grad")():
                val_batches = 0
                for batch in val_loader:
                    source_res = batch["source_residual"].to(device)
                    instruction_target = batch["instruction_target"].to(device)
                    fake_res = refiner(source_res)
                    logits = parser_head(fake_res)
                    ce_loss = cross_entropy_loss(logits, instruction_target)
                    mil_loss = mil_cross_entropy_loss(logits, instruction_target)
                    syntax_reg = syntax_loss(logits, syntax_penalties)
                    val_loss = args.ce_weight * ce_loss + args.mil_weight * mil_loss + args.syntax_weight * syntax_reg
                    val_total += float(val_loss.item())
                    val_batches += 1
                    _print_progress("val", val_batches, len(val_loader), f"loss={val_total / val_batches:.4f}")
                    if vis_written < args.num_vis:
                        count = min(args.num_vis - vis_written, fake_res.shape[0])
                        for sample_index in range(count):
                            sample_id = str(batch["sample_ids"][sample_index]).replace("/", "__")
                            sample_dir = vis_dir / sample_id
                            sample_dir.mkdir(parents=True, exist_ok=True)
                            _save_gray(batch["source_gray"][sample_index], sample_dir / "input.png", residual=False, mean_value=args.mean_value)
                            _save_gray(fake_res[sample_index], sample_dir / "pred.png", residual=True, mean_value=args.mean_value)
                            _save_gray(batch["target_gray"][sample_index], sample_dir / "target.png", residual=False, mean_value=args.mean_value)
                            vis_written += 1
            _finish_progress()
            epoch_metrics["val_loss"] = val_total / max(1, len(val_loader))
            current_val = cast(float, epoch_metrics["val_loss"])
            if best_val is None or current_val < best_val:
                best_val = current_val
                getattr(torch, "save")(
                    {
                        "refiner_state_dict": refiner.state_dict(),
                        "parser_state_dict": parser_head.state_dict(),
                        "discriminator_state_dict": discriminator.state_dict(),
                        "metrics": epoch_metrics,
                    },
                    args.output_dir / "best.pt",
                )
            print(f"epoch={epoch + 1} train_g={cast(float, epoch_metrics['train_generator_loss']):.4f} train_d={cast(float, epoch_metrics['train_discriminator_loss']):.4f} val_loss={current_val:.4f}")
        else:
            print(f"epoch={epoch + 1} train_g={cast(float, epoch_metrics['train_generator_loss']):.4f} train_d={cast(float, epoch_metrics['train_discriminator_loss']):.4f}")
        history.append(epoch_metrics)

    metrics = {
        "manifest": str(args.manifest),
        "val_manifest": str(args.val_manifest) if args.val_manifest is not None else None,
        "parser_init": str(args.parser_init) if args.parser_init is not None else None,
        "palette": str(args.palette),
        "syntax_dir": str(args.syntax_dir),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": [int(args.image_size[0]), int(args.image_size[1])],
        "learning_rate": args.learning_rate,
        "device": str(device),
        "amp": use_amp,
        "use_best_crop": bool(args.use_best_crop),
        "augment_scale": args.augment_scale,
        "use_transfer": bool(args.use_transfer),
        "transfer_root": str(args.transfer_root) if args.use_transfer else None,
        "mean_value": args.mean_value,
        "ce_weight": args.ce_weight,
        "mil_weight": args.mil_weight,
        "gan_weight": args.gan_weight,
        "syntax_weight": args.syntax_weight,
        "perceptual_weight": args.perceptual_weight,
        "style_weight": args.style_weight,
        "transfer_weight": args.transfer_weight,
        "vgg_variant": args.vgg_variant,
        "num_train_samples": len(train_dataset),
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    getattr(torch, "save")(
        {
            "refiner_state_dict": refiner.state_dict(),
            "parser_state_dict": parser_head.state_dict(),
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
