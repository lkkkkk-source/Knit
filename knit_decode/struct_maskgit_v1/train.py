from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from PIL import Image

from knit_decode.parser_t_inverse.losses import build_syntax_penalties, syntax_loss
from knit_decode.parser_t_inverse.palette import OFFICIAL_PALETTE, infer_palette_mapping
from knit_decode.struct_ar_v1.train import _compute_metrics, _resolve_device

from .dataset import NUM_CLASSES, build_dataloader, compute_class_counts, load_manifest
from .mask_schedule import schedule
from .model import MultiScaleMaskGitPrior
from .parallel_decode import decode


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        optim = importlib.import_module("torch.optim")
    except ImportError as error:
        raise ImportError("PyTorch is required for struct_maskgit_v1 training. Install with `pip install -e .[train]`.") from error
    return torch, optim


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a MaskGIT-style multi-scale prior for category->instruction17.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--palette", type=Path, default=None)
    parser.add_argument("--syntax-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=4.5e-2)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.96)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--compute-loss-for-all", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mask-scheduling-method", type=str, default="cosine")
    parser.add_argument("--min-masking-rate", type=float, default=0.5)
    parser.add_argument("--fg-occ-weight", type=float, default=1.0)
    parser.add_argument("--fg-class-weight", type=float, default=1.0)
    parser.add_argument("--all-ce-weight", type=float, default=0.25)
    parser.add_argument("--fg-dice-weight", type=float, default=0.5)
    parser.add_argument("--bg-loss-weight", type=float, default=0.25)
    parser.add_argument("--syntax-weight", type=float, default=0.0)
    parser.add_argument("--count-weight", type=float, default=0.0)
    parser.add_argument("--mask-weight", type=float, default=0.0)
    parser.add_argument("--background-class-id", type=int, default=0)
    parser.add_argument("--loss-weight-5", type=float, default=0.5)
    parser.add_argument("--loss-weight-10", type=float, default=0.75)
    parser.add_argument("--loss-weight-20", type=float, default=1.0)
    parser.add_argument("--sample-choice-temperature", type=float, default=4.5)
    parser.add_argument("--sample-steps-5", type=int, default=6)
    parser.add_argument("--sample-steps-10", type=int, default=8)
    parser.add_argument("--sample-steps-20", type=int, default=10)
    parser.add_argument("--model-width", type=int, default=512)
    parser.add_argument("--model-depth", type=int, default=12)
    parser.add_argument("--model-heads", type=int, default=8)
    parser.add_argument("--model-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--model-dropout", type=float, default=0.1)
    parser.add_argument("--num-vis", type=int, default=16)
    return parser


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
    categories = {sample["category"] for sample in load_manifest(train_manifest)}
    if val_manifest is not None:
        categories.update(sample["category"] for sample in load_manifest(val_manifest))
    return {category: index for index, category in enumerate(sorted(categories))}


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


def _jsonable_config(config: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in config.items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def _sample_mask(torch: object, targets: object, min_masking_rate: float, method: str) -> object:
    batch_size, height, width = targets.shape
    seq_len = height * width
    ratios = getattr(torch, "rand")((batch_size,), device=targets.device, dtype=getattr(torch, "float32"))
    flat_mask = getattr(torch, "zeros")((batch_size, seq_len), dtype=getattr(torch, "bool"), device=targets.device)
    for batch_index in range(batch_size):
        ratio = max(float(min_masking_rate), schedule(float(ratios[batch_index].item()), seq_len, method=method))
        mask_count = max(1, min(seq_len, int(round(seq_len * ratio))))
        indices = getattr(torch, "randperm")(seq_len, device=targets.device)[:mask_count]
        flat_mask[batch_index, indices] = True
    return flat_mask.reshape(batch_size, height, width)


def _smoothed_cross_entropy(
    logits: object,
    targets: object,
    mask: object,
    label_smoothing: float,
    compute_loss_for_all: bool,
    class_weights: object | None,
) -> object:
    torch, _ = _require_torch()
    functional = __import__("importlib").import_module("torch.nn.functional")
    num_classes = int(logits.shape[1])
    logits_hw = logits.permute(0, 2, 3, 1).reshape(-1, num_classes)
    targets_flat = targets.reshape(-1)
    selected = getattr(torch, "ones_like")(targets_flat, dtype=getattr(torch, "bool")) if compute_loss_for_all else mask.reshape(-1)
    selected_logits = logits_hw[selected]
    selected_targets = targets_flat[selected]
    if int(selected_targets.numel()) == 0:
        return getattr(torch, "tensor")(0.0, device=logits.device)
    log_probs = functional.log_softmax(selected_logits, dim=-1)
    off_value = float(label_smoothing) / float(num_classes)
    on_value = 1.0 - float(label_smoothing) + off_value
    soft_targets = getattr(torch, "full_like")(selected_logits, off_value)
    soft_targets.scatter_(1, selected_targets.unsqueeze(1), on_value)
    token_loss = -(soft_targets * log_probs).sum(dim=-1)
    if class_weights is not None:
        token_weight = class_weights.gather(0, selected_targets)
        token_loss = token_loss * token_weight
        return token_loss.sum() / token_weight.sum().clamp_min(1e-6)
    return token_loss.mean()


def _weighted_smoothed_cross_entropy(
    logits: object,
    targets: object,
    mask: object,
    label_smoothing: float,
    compute_loss_for_all: bool,
    background_class_id: int,
    bg_loss_weight: float,
    class_weights: object | None,
) -> object:
    torch, _ = _require_torch()
    functional = __import__("importlib").import_module("torch.nn.functional")
    num_classes = int(logits.shape[1])
    logits_hw = logits.permute(0, 2, 3, 1).reshape(-1, num_classes)
    targets_flat = targets.reshape(-1)
    selected = getattr(torch, "ones_like")(targets_flat, dtype=getattr(torch, "bool")) if compute_loss_for_all else mask.reshape(-1)
    selected_logits = logits_hw[selected]
    selected_targets = targets_flat[selected]
    if int(selected_targets.numel()) == 0:
        return getattr(torch, "tensor")(0.0, device=logits.device)
    log_probs = functional.log_softmax(selected_logits, dim=-1)
    off_value = float(label_smoothing) / float(num_classes)
    on_value = 1.0 - float(label_smoothing) + off_value
    soft_targets = getattr(torch, "full_like")(selected_logits, off_value)
    soft_targets.scatter_(1, selected_targets.unsqueeze(1), on_value)
    token_loss = -(soft_targets * log_probs).sum(dim=-1)
    sample_weights = getattr(torch, "ones_like")(token_loss)
    sample_weights = getattr(torch, "where")(
        selected_targets == int(background_class_id),
        getattr(torch, "full_like")(sample_weights, float(bg_loss_weight)),
        sample_weights,
    )
    if class_weights is not None:
        sample_weights = sample_weights * class_weights.gather(0, selected_targets)
    return (token_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)


def _foreground_occupancy_loss(logits: object, grid20: object, background_class_id: int, dice_weight: float) -> tuple[object, object]:
    torch, _ = _require_torch()
    functional = __import__("importlib").import_module("torch.nn.functional")
    probs = functional.softmax(logits, dim=1)
    fg_prob = 1.0 - probs[:, background_class_id]
    fg_target = (grid20 != background_class_id).to(dtype=fg_prob.dtype)
    bce = functional.binary_cross_entropy(fg_prob.clamp(1e-6, 1.0 - 1e-6), fg_target)
    intersection = (fg_prob * fg_target).sum(dim=(1, 2))
    denom = fg_prob.sum(dim=(1, 2)) + fg_target.sum(dim=(1, 2))
    dice = 1.0 - ((2.0 * intersection + 1e-6) / (denom + 1e-6))
    dice_loss = dice.mean()
    return bce + float(dice_weight) * dice_loss, dice_loss


def _foreground_class_loss(
    logits: object,
    grid20: object,
    mask20: object,
    label_smoothing: float,
    compute_loss_for_all: bool,
    background_class_id: int,
) -> object:
    torch, _ = _require_torch()
    functional = __import__("importlib").import_module("torch.nn.functional")
    fg_targets = grid20 != background_class_id
    selected = fg_targets if compute_loss_for_all else (fg_targets & mask20)
    if int(selected.sum().item()) == 0:
        return getattr(torch, "tensor")(0.0, device=logits.device)
    keep_class_ids = [index for index in range(int(logits.shape[1])) if index != int(background_class_id)]
    class_index = getattr(torch, "tensor")(keep_class_ids, device=logits.device, dtype=getattr(torch, "long"))
    fg_logits = logits.index_select(1, class_index).permute(0, 2, 3, 1).reshape(-1, len(keep_class_ids))
    fg_targets_full = grid20.reshape(-1)
    selected_flat = selected.reshape(-1)
    fg_logits = fg_logits[selected_flat]
    fg_targets_full = fg_targets_full[selected_flat]
    adjusted_targets = fg_targets_full.clone()
    adjusted_targets = getattr(torch, "where")(adjusted_targets > background_class_id, adjusted_targets - 1, adjusted_targets)
    num_classes = int(fg_logits.shape[-1])
    log_probs = functional.log_softmax(fg_logits, dim=-1)
    off_value = float(label_smoothing) / float(num_classes)
    on_value = 1.0 - float(label_smoothing) + off_value
    soft_targets = getattr(torch, "full_like")(fg_logits, off_value)
    soft_targets.scatter_(1, adjusted_targets.unsqueeze(1), on_value)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def _count_ratio_loss(logits: object, count_vectors: object) -> object:
    functional = __import__("importlib").import_module("torch.nn.functional")
    probs = functional.softmax(logits, dim=1)
    pred_counts = probs.sum(dim=(2, 3))
    gt_counts = count_vectors.to(dtype=pred_counts.dtype)
    pred_ratio = pred_counts / pred_counts.sum(dim=1, keepdim=True).clamp_min(1e-6)
    gt_ratio = gt_counts / gt_counts.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return functional.smooth_l1_loss(pred_ratio, gt_ratio)


def _foreground_mask_loss(logits: object, grid20: object, background_class_id: int) -> object:
    functional = __import__("importlib").import_module("torch.nn.functional")
    fg_logit = -logits[:, background_class_id]
    target = (grid20 != background_class_id).to(dtype=logits.dtype)
    return functional.binary_cross_entropy_with_logits(fg_logit, target)


def _sample_stage(
    model: object,
    stage_name: str,
    category_ids: object,
    size: int,
    cond_grid: object | None,
    num_steps: int,
    choice_temperature: float,
    mask_scheduling_method: str,
) -> object:
    torch, _ = _require_torch()
    seq_len = size * size
    init_tokens = getattr(torch, "full")(
        (int(category_ids.shape[0]), seq_len),
        int(model.mask_token_id),
        device=category_ids.device,
        dtype=getattr(torch, "long"),
    )

    def tokens_to_logits(tokens: object) -> object:
        return model.stage_logits(stage_name, category_ids, tokens, cond_grid=cond_grid)

    sampled = decode(
        init_tokens,
        tokens_to_logits=tokens_to_logits,
        mask_token_id=int(model.mask_token_id),
        num_iter=num_steps,
        choice_temperature=choice_temperature,
        mask_scheduling_method=mask_scheduling_method,
    )
    return sampled.reshape(-1, size, size)


def _sample_multiscale(model: object, category_ids: object, args: argparse.Namespace) -> dict[str, object]:
    pred5 = _sample_stage(
        model,
        stage_name="stage5",
        category_ids=category_ids,
        size=5,
        cond_grid=None,
        num_steps=args.sample_steps_5,
        choice_temperature=args.sample_choice_temperature,
        mask_scheduling_method=args.mask_scheduling_method,
    )
    pred10 = _sample_stage(
        model,
        stage_name="stage10",
        category_ids=category_ids,
        size=10,
        cond_grid=pred5,
        num_steps=args.sample_steps_10,
        choice_temperature=args.sample_choice_temperature,
        mask_scheduling_method=args.mask_scheduling_method,
    )
    pred20 = _sample_stage(
        model,
        stage_name="stage20",
        category_ids=category_ids,
        size=20,
        cond_grid=pred10,
        num_steps=args.sample_steps_20,
        choice_temperature=args.sample_choice_temperature,
        mask_scheduling_method=args.mask_scheduling_method,
    )
    return {"pred5": pred5, "pred10": pred10, "pred20": pred20}


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
    category_to_id = _build_category_mapping(args.manifest, args.val_manifest)

    train_loader, train_dataset = build_dataloader(
        args.manifest,
        palette_path=palette_path,
        batch_size=args.batch_size,
        shuffle=True,
        category_to_id=category_to_id,
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
            category_to_id=category_to_id,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
        )

    model = MultiScaleMaskGitPrior(
        num_categories=len(train_dataset.category_to_id),
        num_classes=NUM_CLASSES,
        width=args.model_width,
        depth=args.model_depth,
        heads=args.model_heads,
        mlp_ratio=args.model_mlp_ratio,
        dropout=args.model_dropout,
    )
    model.to(device)
    optimizer = getattr(optim, "AdamW")(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
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
        total_fg_occ = 0.0
        total_fg_dice = 0.0
        total_fg_class = 0.0
        total_all_ce = 0.0
        total_syntax = 0.0
        total_count = 0.0
        total_mask = 0.0
        batch_count = 0
        for batch in train_loader:
            category_ids = batch["category_ids"].to(device)
            grid5 = batch["grid5"].to(device)
            grid10 = batch["grid10"].to(device)
            grid20 = batch["grid20"].to(device)
            count_vectors = batch["count_vectors"].to(device)
            mask5 = _sample_mask(torch, grid5, args.min_masking_rate, args.mask_scheduling_method)
            mask10 = _sample_mask(torch, grid10, args.min_masking_rate, args.mask_scheduling_method)
            mask20 = _sample_mask(torch, grid20, args.min_masking_rate, args.mask_scheduling_method)
            with _autocast_context(torch, use_amp):
                outputs = model(category_ids, grid5, grid10, grid20, mask5, mask10, mask20)
                ce5 = _smoothed_cross_entropy(outputs["logits5"], grid5, mask5, args.label_smoothing, bool(args.compute_loss_for_all), class_weights)
                ce10 = _smoothed_cross_entropy(outputs["logits10"], grid10, mask10, args.label_smoothing, bool(args.compute_loss_for_all), class_weights)
                all_ce20 = _weighted_smoothed_cross_entropy(
                    outputs["logits20"],
                    grid20,
                    mask20,
                    args.label_smoothing,
                    bool(args.compute_loss_for_all),
                    args.background_class_id,
                    args.bg_loss_weight,
                    class_weights,
                )
                fg_occ, fg_dice = _foreground_occupancy_loss(outputs["logits20"], grid20, args.background_class_id, args.fg_dice_weight)
                fg_class = _foreground_class_loss(
                    outputs["logits20"],
                    grid20,
                    mask20,
                    args.label_smoothing,
                    bool(args.compute_loss_for_all),
                    args.background_class_id,
                )
                ce20 = (
                    args.fg_occ_weight * fg_occ
                    + args.fg_class_weight * fg_class
                    + args.all_ce_weight * all_ce20
                )
                syn = syntax_loss(outputs["logits20"], syntax_penalties) if args.syntax_weight > 0 else getattr(torch, "tensor")(0.0, device=device)
                count_loss = _count_ratio_loss(outputs["logits20"], count_vectors) if args.count_weight > 0 else getattr(torch, "tensor")(0.0, device=device)
                mask_loss = _foreground_mask_loss(outputs["logits20"], grid20, args.background_class_id) if args.mask_weight > 0 else getattr(torch, "tensor")(0.0, device=device)
                loss = (
                    args.loss_weight_5 * ce5
                    + args.loss_weight_10 * ce10
                    + args.loss_weight_20 * ce20
                    + args.syntax_weight * syn
                    + args.count_weight * count_loss
                    + args.mask_weight * mask_loss
                )
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
            total_fg_occ += float(fg_occ.item())
            total_fg_dice += float(fg_dice.item())
            total_fg_class += float(fg_class.item())
            total_all_ce += float(all_ce20.item())
            total_syntax += float(syn.item()) if hasattr(syn, "item") else 0.0
            total_count += float(count_loss.item()) if hasattr(count_loss, "item") else 0.0
            total_mask += float(mask_loss.item()) if hasattr(mask_loss, "item") else 0.0
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
                    f"fg_occ={total_fg_occ / batch_count:.6f} "
                    f"fg_cls={total_fg_class / batch_count:.6f} "
                    f"all_ce={total_all_ce / batch_count:.6f} "
                    f"syntax={total_syntax / batch_count:.6f} "
                    f"count={total_count / batch_count:.6f} "
                    f"mask={total_mask / batch_count:.6f}"
                ),
            )
        _finish_progress()

        epoch_metrics: dict[str, object] = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, batch_count),
            "train_ce5": total_ce5 / max(1, batch_count),
            "train_ce10": total_ce10 / max(1, batch_count),
            "train_ce20": total_ce20 / max(1, batch_count),
            "train_fg_occ": total_fg_occ / max(1, batch_count),
            "train_fg_dice": total_fg_dice / max(1, batch_count),
            "train_fg_class": total_fg_class / max(1, batch_count),
            "train_all_ce": total_all_ce / max(1, batch_count),
            "train_syntax": total_syntax / max(1, batch_count),
            "train_count": total_count / max(1, batch_count),
            "train_mask": total_mask / max(1, batch_count),
        }

        if val_loader is not None:
            model.eval()
            val_confusion = [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)]
            vis_dir = args.output_dir / "val_predictions" / f"epoch_{epoch + 1:03d}"
            vis_dir.mkdir(parents=True, exist_ok=True)
            vis_written = 0
            val_exact = 0
            val_samples = 0
            with getattr(torch, "no_grad")():
                val_batches = 0
                for batch in val_loader:
                    category_ids = batch["category_ids"].to(device)
                    grid20 = batch["grid20"].to(device)
                    sample_ids = [str(value) for value in batch["sample_ids"]]
                    sampled = _sample_multiscale(model, category_ids, args)
                    pred20 = sampled["pred20"]
                    exact_matches = pred20.eq(grid20).reshape(pred20.shape[0], -1).all(dim=1)
                    val_exact += int(exact_matches.sum().item())
                    val_samples += int(pred20.shape[0])
                    predictions = pred20.detach().cpu().tolist()
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
                    val_batches += 1
                    metrics = _compute_metrics(val_confusion)
                    _print_progress(
                        "val",
                        val_batches,
                        len(val_loader),
                        f"pixacc={cast(float, metrics['pixel_accuracy']):.6f} miou={cast(float, metrics['mean_iou']):.6f}",
                    )
            _finish_progress()
            metrics = _compute_metrics(val_confusion)
            epoch_metrics["val_pixel_accuracy"] = metrics["pixel_accuracy"]
            epoch_metrics["val_mean_iou"] = metrics["mean_iou"]
            epoch_metrics["val_per_class_iou"] = metrics["per_class_iou"]
            epoch_metrics["val_exact_match"] = val_exact / max(1, val_samples)
            print(
                f"epoch={epoch + 1} train_loss={cast(float, epoch_metrics['train_loss']):.6f} "
                f"val_acc={cast(float, epoch_metrics['val_pixel_accuracy']):.4f} "
                f"val_miou={cast(float, epoch_metrics['val_mean_iou']):.4f} "
                f"val_exact={cast(float, epoch_metrics['val_exact_match']):.4f}"
            )
        else:
            print(f"epoch={epoch + 1} train_loss={cast(float, epoch_metrics['train_loss']):.6f}")
        history.append(epoch_metrics)

    model_config = {
        "num_categories": len(category_to_id),
        "num_classes": NUM_CLASSES,
        "width": args.model_width,
        "depth": args.model_depth,
        "heads": args.model_heads,
        "mlp_ratio": args.model_mlp_ratio,
        "dropout": args.model_dropout,
    }
    result = {
        "manifest": str(args.manifest),
        "val_manifest": str(args.val_manifest) if args.val_manifest is not None else None,
        "palette": str(palette_path),
        "syntax_dir": str(syntax_dir),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "device": str(device),
        "amp": use_amp,
        "num_classes": NUM_CLASSES,
        "num_categories": len(category_to_id),
        "category_to_id": category_to_id,
        "label_smoothing": args.label_smoothing,
        "compute_loss_for_all": bool(args.compute_loss_for_all),
        "mask_scheduling_method": args.mask_scheduling_method,
        "min_masking_rate": args.min_masking_rate,
        "fg_occ_weight": args.fg_occ_weight,
        "fg_class_weight": args.fg_class_weight,
        "all_ce_weight": args.all_ce_weight,
        "fg_dice_weight": args.fg_dice_weight,
        "bg_loss_weight": args.bg_loss_weight,
        "syntax_weight": args.syntax_weight,
        "count_weight": args.count_weight,
        "mask_weight": args.mask_weight,
        "background_class_id": args.background_class_id,
        "sample_choice_temperature": args.sample_choice_temperature,
        "sample_steps_5": args.sample_steps_5,
        "sample_steps_10": args.sample_steps_10,
        "sample_steps_20": args.sample_steps_20,
        "model_config": model_config,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    getattr(torch, "save")(
        {
            "model_state_dict": model.state_dict(),
            "metrics": result,
            "model_config": model_config,
            "config": _jsonable_config(vars(args)),
        },
        args.output_dir / "checkpoint.pt",
    )
    print(f"saved metrics: {args.output_dir / 'metrics.json'}")
    print(f"saved checkpoint: {args.output_dir / 'checkpoint.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
