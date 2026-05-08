from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from .dataset import build_render_dataloader, load_render_manifest
from .diffusion import DiffusionSchedule
from .model import CategoryConditionalUNet


def _require_torch() -> tuple[object, object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        optim = importlib.import_module("torch.optim")
        functional = importlib.import_module("torch.nn.functional")
    except ImportError as error:
        raise ImportError("PyTorch is required for render-v1 training. Install with `pip install -e .[train]`.") from error
    return torch, optim, functional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a minimal category-to-rendering diffusion baseline.")
    parser.add_argument("--manifest", type=Path, required=True, help="Training JSONL manifest")
    parser.add_argument("--val-manifest", type=Path, default=None, help="Optional validation manifest")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, nargs=2, default=(160, 160), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--num-diffusion-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--teacher-mode", type=str, choices=("coarse", "inverse17"), default="coarse", help="Teacher type to use when teacher loss is enabled")
    parser.add_argument("--teacher-checkpoint", type=Path, default=None, help="Optional frozen parser teacher checkpoint")
    parser.add_argument("--teacher-target-manifest", type=Path, default=None, help="Optional cached parser-target manifest aligned with the training manifest")
    parser.add_argument("--teacher-loss-weight", type=float, default=0.0, help="Weight for frozen teacher structure loss")
    parser.add_argument("--teacher-max-timestep", type=int, default=None, help="Only apply teacher loss when timestep <= this threshold")
    parser.add_argument("--teacher-palette", type=Path, default=None, help="Palette mapping for inverse17 teacher targets")
    parser.add_argument("--teacher-syntax-dir", type=Path, default=None, help="Syntax directory for inverse17 teacher")
    parser.add_argument("--teacher-syntax-weight", type=float, default=0.0, help="Extra syntax weight for inverse17 teacher logits")
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


def _build_category_mapping(train_manifest: Path, val_manifest: Path | None) -> dict[str, int]:
    categories = {sample["category"] for sample in load_render_manifest(train_manifest)}
    if val_manifest is not None:
        categories.update(sample["category"] for sample in load_render_manifest(val_manifest))
    return {category: index for index, category in enumerate(sorted(categories))}


def _sample_id_from_entry(sample: object) -> str:
    if isinstance(sample, dict):
        value = sample.get("sample_id")
        if isinstance(value, str):
            return value
    value = getattr(sample, "sample_id", None)
    if isinstance(value, str):
        return value
    raise ValueError(f"Unable to read sample_id from sample entry: {sample!r}")


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

    category_to_id = _build_category_mapping(args.manifest, args.val_manifest)
    train_loader, train_dataset = build_render_dataloader(
        args.manifest,
        batch_size=args.batch_size,
        shuffle=True,
        image_size=(int(args.image_size[0]), int(args.image_size[1])),
        category_to_id=category_to_id,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    val_loader = None
    if args.val_manifest is not None:
        val_loader, _ = build_render_dataloader(
            args.val_manifest,
            batch_size=args.batch_size,
            shuffle=False,
            image_size=(int(args.image_size[0]), int(args.image_size[1])),
            category_to_id=category_to_id,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
        )

    teacher = None
    parser_target_dataset = None
    parser_target_index = None
    if args.teacher_checkpoint is not None:
        if args.teacher_target_manifest is None:
            raise ValueError("When using --teacher-checkpoint, pass --teacher-target-manifest with cached parser targets.")
        if args.teacher_mode == "coarse":
            from .teacher import FrozenParserTeacher
            from knit_decode.parser_t.dataset import SimulationTopologyDataset

            teacher = FrozenParserTeacher(args.teacher_checkpoint, device=device)
            parser_target_dataset = SimulationTopologyDataset(
                args.teacher_target_manifest,
                image_size=teacher.image_size,
                grid_size=teacher.grid_size,
                vocabulary=None,
                top_k_colors=max(1, teacher.num_classes - 1),
            )
        else:
            from .inverse_teacher import FrozenInverseTeacher
            from knit_decode.parser_t_inverse.dataset import ParserInverseDataset
            from knit_decode.parser_t_inverse.palette import infer_palette_mapping

            if args.teacher_palette is None:
                inferred_palette = args.output_dir / "teacher_palette_mapping.json"
                infer_palette_mapping(args.teacher_target_manifest, inferred_palette)
                palette_path = inferred_palette
            else:
                palette_path = args.teacher_palette
            syntax_dir = args.teacher_syntax_dir or (Path("dataset2") / "syntax")
            teacher = FrozenInverseTeacher(args.teacher_checkpoint, syntax_dir=syntax_dir, device=device)
            parser_target_dataset = ParserInverseDataset(
                args.teacher_target_manifest,
                palette_path=palette_path,
                image_size=teacher.image_size,
            )
        parser_target_index = {_sample_id_from_entry(sample): index for index, sample in enumerate(parser_target_dataset.samples)}

    model = CategoryConditionalUNet(num_categories=len(train_dataset.category_to_id))
    model.to(device)
    optimizer = getattr(optim, "AdamW")(model.parameters(), lr=args.learning_rate)
    scaler = getattr(getattr(torch, "cuda"), "amp").GradScaler(enabled=use_amp)
    diffusion = DiffusionSchedule(args.num_diffusion_steps)

    history: list[dict[str, object]] = []
    for epoch in range(args.epochs):
        print(f"epoch {epoch + 1}/{args.epochs}")
        model.train()
        total_loss = 0.0
        total_denoise_loss = 0.0
        total_teacher_loss = 0.0
        batch_count = 0
        total_batches = len(train_loader)
        for batch in train_loader:
            images = batch["images"].to(device)
            category_ids = batch["category_ids"].to(device)
            sample_ids = [str(value) for value in batch["sample_ids"]]
            timesteps = getattr(torch, "randint")(0, args.num_diffusion_steps, (images.shape[0],), device=device, dtype=getattr(torch, "long"))
            noise = getattr(torch, "randn_like")(images)
            noisy_images = diffusion.q_sample(images, timesteps, noise)

            with _autocast_context(torch, use_amp):
                pred_noise = model(noisy_images, timesteps, category_ids)
                denoise_loss = functional.mse_loss(pred_noise, noise)
                teacher_loss = getattr(torch, "tensor")(0.0, device=device)
                use_teacher = (
                    teacher is not None
                    and parser_target_dataset is not None
                    and parser_target_index is not None
                    and args.teacher_loss_weight > 0.0
                )
                if use_teacher and args.teacher_max_timestep is not None:
                    teacher_mask = timesteps <= int(args.teacher_max_timestep)
                    use_teacher = bool(getattr(teacher_mask, "any")().item())
                else:
                    teacher_mask = None
                if use_teacher:
                    target_tensors = []
                    selected_ids = sample_ids
                    selected_noisy = noisy_images
                    selected_pred_noise = pred_noise
                    selected_timesteps = timesteps
                    if teacher_mask is not None:
                        selected_ids = [sample_id for sample_id, keep in zip(sample_ids, teacher_mask.tolist()) if keep]
                        selected_noisy = noisy_images[teacher_mask]
                        selected_pred_noise = pred_noise[teacher_mask]
                        selected_timesteps = timesteps[teacher_mask]
                    for sample_id in selected_ids:
                        target_index = parser_target_index.get(sample_id)
                        if target_index is None:
                            raise KeyError(f"Missing parser target for sample_id={sample_id!r} in {args.teacher_target_manifest}")
                        parser_item = parser_target_dataset[target_index]
                        target_tensors.append(parser_item["target"])
                    target_grid = getattr(torch, "stack")(target_tensors).to(device)
                    predicted_clean = (
                        selected_noisy
                        - getattr(torch, "sqrt")(1.0 - diffusion.alpha_bars.to(device)[selected_timesteps]).view(-1, 1, 1, 1) * selected_pred_noise
                    )
                    predicted_clean = predicted_clean / getattr(torch, "sqrt")(diffusion.alpha_bars.to(device)[selected_timesteps]).view(-1, 1, 1, 1)
                    predicted_clean = predicted_clean.clamp(-1.0, 1.0)
                    teacher_logits = teacher.logits(predicted_clean) if hasattr(teacher, "logits") else teacher(predicted_clean)
                    teacher_loss = functional.cross_entropy(teacher_logits, target_grid)
                    if args.teacher_mode == "inverse17" and args.teacher_syntax_weight > 0.0:
                        teacher_loss = teacher_loss + args.teacher_syntax_weight * teacher.syntax_loss(teacher_logits)
                loss = denoise_loss + args.teacher_loss_weight * teacher_loss

            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            total_denoise_loss += float(denoise_loss.item())
            total_teacher_loss += float(teacher_loss.item()) if hasattr(teacher_loss, "item") else 0.0
            batch_count += 1
            _print_progress(
                "train",
                batch_count,
                total_batches,
                f"loss={total_loss / batch_count:.6f} denoise={total_denoise_loss / batch_count:.6f} teacher={total_teacher_loss / batch_count:.6f}",
            )
        _finish_progress()

        epoch_metrics: dict[str, object] = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, batch_count),
            "train_denoise_loss": total_denoise_loss / max(1, batch_count),
            "train_teacher_loss": total_teacher_loss / max(1, batch_count),
        }

        if val_loader is not None:
            model.eval()
            val_total = 0.0
            val_batches = 0
            with getattr(torch, "no_grad")():
                for batch in val_loader:
                    images = batch["images"].to(device)
                    category_ids = batch["category_ids"].to(device)
                    timesteps = getattr(torch, "randint")(0, args.num_diffusion_steps, (images.shape[0],), device=device, dtype=getattr(torch, "long"))
                    noise = getattr(torch, "randn_like")(images)
                    noisy_images = diffusion.q_sample(images, timesteps, noise)
                    with _autocast_context(torch, use_amp):
                        pred_noise = model(noisy_images, timesteps, category_ids)
                        val_loss = functional.mse_loss(pred_noise, noise)
                    val_total += float(val_loss.item())
                    val_batches += 1
                    _print_progress("val", val_batches, len(val_loader), f"loss={val_total / val_batches:.6f}")
            _finish_progress()
            epoch_metrics["val_loss"] = val_total / max(1, val_batches)
            print(
                f"epoch={epoch + 1} train_loss={cast(float, epoch_metrics['train_loss']):.6f} "
                f"train_denoise={cast(float, epoch_metrics['train_denoise_loss']):.6f} "
                f"train_teacher={cast(float, epoch_metrics['train_teacher_loss']):.6f} "
                f"val_loss={cast(float, epoch_metrics['val_loss']):.6f}"
            )
        else:
            print(
                f"epoch={epoch + 1} train_loss={cast(float, epoch_metrics['train_loss']):.6f} "
                f"train_denoise={cast(float, epoch_metrics['train_denoise_loss']):.6f} "
                f"train_teacher={cast(float, epoch_metrics['train_teacher_loss']):.6f}"
            )

        history.append(epoch_metrics)

    metadata = {
        "manifest": str(args.manifest),
        "val_manifest": str(args.val_manifest) if args.val_manifest is not None else None,
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": [int(args.image_size[0]), int(args.image_size[1])],
        "num_diffusion_steps": args.num_diffusion_steps,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "amp": use_amp,
        "num_categories": len(train_dataset.category_to_id),
        "category_to_id": train_dataset.category_to_id,
        "teacher_checkpoint": str(args.teacher_checkpoint) if args.teacher_checkpoint is not None else None,
        "teacher_target_manifest": str(args.teacher_target_manifest) if args.teacher_target_manifest is not None else None,
        "teacher_mode": args.teacher_mode,
        "teacher_loss_weight": args.teacher_loss_weight,
        "teacher_max_timestep": args.teacher_max_timestep,
        "teacher_palette": str(args.teacher_palette) if args.teacher_palette is not None else None,
        "teacher_syntax_dir": str(args.teacher_syntax_dir) if args.teacher_syntax_dir is not None else None,
        "teacher_syntax_weight": args.teacher_syntax_weight,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    getattr(torch, "save")(
        {
            "model_state_dict": model.state_dict(),
            "metrics": metadata,
        },
        args.output_dir / "checkpoint.pt",
    )
    print(f"saved metrics: {args.output_dir / 'metrics.json'}")
    print(f"saved checkpoint: {args.output_dir / 'checkpoint.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
