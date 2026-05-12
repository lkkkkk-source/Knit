from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import cast

from .compose_foreground import compose_foreground
from .dataset import build_dataloader, load_manifest
from .models.foreground_planner import ForegroundCanonicalPlanner
from .utils import finish_progress, foreground_area, format_metric_line, label_diversity_on_fg, load_config, mask_component_stats, print_progress, require_centroid_sketch_fields, require_foreground_cache_fields, resolve_canonical_mode


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        optim = importlib.import_module("torch.optim")
    except ImportError as error:
        raise ImportError("PyTorch is required for train_foreground_planner.") from error
    return torch, optim


def _resolve_device(torch: object, device_name: str) -> object:
    device_cls = getattr(torch, "device")
    if device_name == "cpu":
        return device_cls("cpu")
    if not getattr(torch, "cuda").is_available():
        raise RuntimeError(f"Requested device {device_name!r}, but CUDA is not available.")
    return device_cls(device_name)


def _build_category_mapping(train_manifest: Path, val_manifest: Path | None) -> dict[str, object]:
    train_categories = sorted({sample["category"] for sample in load_manifest(train_manifest)})
    categories = set(train_categories)
    val_categories: list[str] = []
    if val_manifest is not None:
        val_categories = sorted({sample["category"] for sample in load_manifest(val_manifest)})
        categories.update(val_categories)
    category_to_id = {category: index for index, category in enumerate(sorted(categories))}
    return {
        "category_to_id": category_to_id,
        "id_to_category": {index: category for category, index in category_to_id.items()},
        "train_categories": train_categories,
        "val_categories": val_categories,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train foreground-canonical planner.")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def _safe_entropy(counts: list[int]) -> float:
    total = float(sum(counts))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        prob = count / total
        entropy -= prob * math.log(prob + 1e-12)
    return entropy


def _effective_modes(counts: list[int]) -> float:
    total = float(sum(counts))
    if total <= 0.0:
        return 0.0
    denom = sum((count / total) ** 2 for count in counts if count > 0)
    return 1.0 / max(denom, 1e-12)


def _soft_dice_loss(torch: object, pred_prob: object, target: object, eps: float = 1e-6) -> object:
    intersection = (pred_prob * target).sum(dim=(1, 2))
    denom = pred_prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def _soft_iou_loss(torch: object, pred_prob: object, target: object, eps: float = 1e-6) -> object:
    intersection = (pred_prob * target).sum(dim=(1, 2))
    union = (pred_prob + target - pred_prob * target).sum(dim=(1, 2))
    iou = (intersection + eps) / (union + eps)
    return 1.0 - iou.mean()


def _init_generated_metric_state() -> dict[str, object]:
    return {
        "sample_count": 0,
        "fg_mask_iou_sum": 0.0,
        "fg_label_acc_on_fg_sum": 0.0,
        "raw_fg_ce_sum": 0.0,
        "weighted_fg_ce_sum": 0.0,
        "bbox_l1_sum": 0.0,
        "centroid_mask_loss_sum": 0.0,
        "mask_bce_loss_sum": 0.0,
        "mask_dice_loss_sum": 0.0,
        "mask_soft_iou_loss_sum": 0.0,
        "centroid_label_loss_sum": 0.0,
        "row_projection_l1_sum": 0.0,
        "col_projection_l1_sum": 0.0,
        "adjacency_l1_sum": 0.0,
        "grammar_l1_sum": 0.0,
        "empty_count": 0,
        "full_count": 0,
        "fg_area_low_count": 0,
        "fg_area_high_count": 0,
        "low_label_diversity_count": 0,
        "invalid_bbox_count": 0,
        "valid_foreground_count": 0,
        "fg_area_sum": 0.0,
        "fg_area_min": float("inf"),
        "fg_area_max": 0.0,
        "fg_area_valid_low_sum": 0.0,
        "fg_area_valid_high_sum": 0.0,
        "label_diversity_sum": 0.0,
        "label_diversity_min": float("inf"),
        "label_diversity_max": 0.0,
        "num_components_sum": 0.0,
        "largest_component_ratio_sum": 0.0,
        "tiny_component_count_sum": 0.0,
        "fragmented_count": 0,
        "local_mode_counts": {},
    }


def _bbox_is_valid(bbox_pred: list[float]) -> bool:
    if len(bbox_pred) < 10:
        return False
    x0, y0, x1, y1, w, h, center_x, center_y, area_ratio, aspect_ratio = bbox_pred[:10]
    normalized_values = [x0, y0, x1, y1, w, h, center_x, center_y, area_ratio]
    if not all(math.isfinite(float(value)) for value in normalized_values + [aspect_ratio]):
        return False
    if any(float(value) < 0.0 or float(value) > 1.0 for value in normalized_values):
        return False
    if float(w) <= 0.0 or float(h) <= 0.0:
        return False
    if float(aspect_ratio) <= 0.0:
        return False
    return True


def _accumulate_generated_metrics(
    torch: object,
    metric_state: dict[str, object],
    outputs: dict[str, object],
    batch: dict[str, object],
    losses: dict[str, float],
    category_area_stats: dict[str, dict[str, float]],
) -> None:
    fg_mask_prob = getattr(torch, "sigmoid")(outputs["fg_mask_logits"].detach().squeeze(1))
    fg_mask_pred = (fg_mask_prob >= 0.5).to(dtype=getattr(torch, "long"))
    fg_mask_target = batch["fg_mask20"].detach().to(dtype=getattr(torch, "long"))
    fg_target = batch["fg_y20"].detach()
    fg_label_pred = outputs["fg_label_logits"].detach().argmax(dim=1) + 1
    inter = (fg_mask_pred * fg_mask_target).sum(dim=(1, 2)).to(dtype=getattr(torch, "float32"))
    union = ((fg_mask_pred + fg_mask_target) > 0).sum(dim=(1, 2)).to(dtype=getattr(torch, "float32"))
    fg_positions = fg_mask_target > 0
    fg_correct = ((fg_label_pred == fg_target) & fg_positions).sum().item()
    fg_total = fg_positions.sum().item()
    fg_mask_iou = float((inter / union.clamp_min(1.0)).mean().item())
    label_acc = float(fg_correct) / float(max(1, fg_total))
    batch_size = int(batch["category_ids"].shape[0])
    metric_state["sample_count"] = int(metric_state["sample_count"]) + batch_size
    metric_state["fg_mask_iou_sum"] = float(metric_state["fg_mask_iou_sum"]) + fg_mask_iou * batch_size
    metric_state["fg_label_acc_on_fg_sum"] = float(metric_state["fg_label_acc_on_fg_sum"]) + label_acc * batch_size
    metric_state["raw_fg_ce_sum"] = float(metric_state["raw_fg_ce_sum"]) + float(losses["raw_fg_ce"]) * batch_size
    metric_state["weighted_fg_ce_sum"] = float(metric_state["weighted_fg_ce_sum"]) + float(losses["weighted_fg_ce"]) * batch_size
    metric_state["bbox_l1_sum"] = float(metric_state["bbox_l1_sum"]) + float(losses["bbox"]) * batch_size
    metric_state["centroid_mask_loss_sum"] = float(metric_state["centroid_mask_loss_sum"]) + float(losses["centroid_mask"]) * batch_size
    metric_state["mask_bce_loss_sum"] = float(metric_state["mask_bce_loss_sum"]) + float(losses["mask_bce"]) * batch_size
    metric_state["mask_dice_loss_sum"] = float(metric_state["mask_dice_loss_sum"]) + float(losses["mask_dice"]) * batch_size
    metric_state["mask_soft_iou_loss_sum"] = float(metric_state["mask_soft_iou_loss_sum"]) + float(losses["mask_soft_iou"]) * batch_size
    metric_state["centroid_label_loss_sum"] = float(metric_state["centroid_label_loss_sum"]) + float(losses["centroid_label"]) * batch_size
    metric_state["row_projection_l1_sum"] = float(metric_state["row_projection_l1_sum"]) + float(losses["row"]) * batch_size
    metric_state["col_projection_l1_sum"] = float(metric_state["col_projection_l1_sum"]) + float(losses["col"]) * batch_size
    metric_state["adjacency_l1_sum"] = float(metric_state["adjacency_l1_sum"]) + float(losses["adj"]) * batch_size
    metric_state["grammar_l1_sum"] = float(metric_state["grammar_l1_sum"]) + float(losses["grammar"]) * batch_size
    local_counts = cast(dict[int, int], metric_state["local_mode_counts"])
    for index in range(batch_size):
        local_z = int(outputs["local_z"][index].item())
        local_counts[local_z] = local_counts.get(local_z, 0) + 1
        pred_mask = fg_mask_pred[index].cpu().tolist()
        pred_label = fg_label_pred[index].cpu().tolist()
        bbox_pred = [float(value) for value in outputs["bbox_pred"][index].detach().cpu().tolist()]
        _ = compose_foreground(pred_mask, pred_label, bbox_pred)["composed_y20"]
        category = str(batch["categories"][index])
        area_stats = category_area_stats.get(category, {})
        valid_low = float(area_stats.get("valid_low", 0.0))
        valid_high = float(area_stats.get("valid_high", 1.0))
        area = foreground_area(pred_mask)
        label_diversity = float(label_diversity_on_fg(pred_label, pred_mask))
        component_stats = mask_component_stats(pred_mask)
        bbox_valid = _bbox_is_valid(bbox_pred)
        is_empty = area <= 0.0
        is_full = area >= 0.99
        is_area_low = area < valid_low
        is_area_high = area > valid_high
        is_low_diversity = label_diversity <= 1.0
        is_fragmented = component_stats["num_components"] > 1.0
        is_valid = not any([is_empty, is_full, is_area_low, is_area_high, is_low_diversity, not bbox_valid])
        metric_state["fg_area_sum"] = float(metric_state["fg_area_sum"]) + area
        metric_state["fg_area_min"] = min(float(metric_state["fg_area_min"]), area)
        metric_state["fg_area_max"] = max(float(metric_state["fg_area_max"]), area)
        metric_state["fg_area_valid_low_sum"] = float(metric_state["fg_area_valid_low_sum"]) + valid_low
        metric_state["fg_area_valid_high_sum"] = float(metric_state["fg_area_valid_high_sum"]) + valid_high
        metric_state["label_diversity_sum"] = float(metric_state["label_diversity_sum"]) + label_diversity
        metric_state["label_diversity_min"] = min(float(metric_state["label_diversity_min"]), label_diversity)
        metric_state["label_diversity_max"] = max(float(metric_state["label_diversity_max"]), label_diversity)
        metric_state["num_components_sum"] = float(metric_state["num_components_sum"]) + float(component_stats["num_components"])
        metric_state["largest_component_ratio_sum"] = float(metric_state["largest_component_ratio_sum"]) + float(component_stats["largest_component_ratio"])
        metric_state["tiny_component_count_sum"] = float(metric_state["tiny_component_count_sum"]) + float(component_stats["tiny_component_count"])
        if area <= 0.0:
            metric_state["empty_count"] = int(metric_state["empty_count"]) + 1
        if area >= 0.99:
            metric_state["full_count"] = int(metric_state["full_count"]) + 1
        if is_area_low:
            metric_state["fg_area_low_count"] = int(metric_state["fg_area_low_count"]) + 1
        if is_area_high:
            metric_state["fg_area_high_count"] = int(metric_state["fg_area_high_count"]) + 1
        if is_low_diversity:
            metric_state["low_label_diversity_count"] = int(metric_state["low_label_diversity_count"]) + 1
        if not bbox_valid:
            metric_state["invalid_bbox_count"] = int(metric_state["invalid_bbox_count"]) + 1
        if is_fragmented:
            metric_state["fragmented_count"] = int(metric_state["fragmented_count"]) + 1
        if is_valid:
            metric_state["valid_foreground_count"] = int(metric_state["valid_foreground_count"]) + 1


def _finalize_generated_metrics(metric_state: dict[str, object]) -> dict[str, float]:
    sample_count = int(metric_state["sample_count"])
    denom = float(max(1, sample_count))
    local_counts = cast(dict[int, int], metric_state["local_mode_counts"])
    mode_hist = [local_counts[key] for key in sorted(local_counts)]
    metrics = {
        "fg_mask_iou": float(metric_state["fg_mask_iou_sum"]) / denom,
        "fg_label_acc_on_fg": float(metric_state["fg_label_acc_on_fg_sum"]) / denom,
        "raw_fg_ce": float(metric_state["raw_fg_ce_sum"]) / denom,
        "weighted_fg_ce": float(metric_state["weighted_fg_ce_sum"]) / denom,
        "bbox_l1": float(metric_state["bbox_l1_sum"]) / denom,
        "centroid_mask_loss": float(metric_state["centroid_mask_loss_sum"]) / denom,
        "mask_bce_loss": float(metric_state["mask_bce_loss_sum"]) / denom,
        "mask_dice_loss": float(metric_state["mask_dice_loss_sum"]) / denom,
        "mask_soft_iou_loss": float(metric_state["mask_soft_iou_loss_sum"]) / denom,
        "centroid_label_loss": float(metric_state["centroid_label_loss_sum"]) / denom,
        "row_projection_l1": float(metric_state["row_projection_l1_sum"]) / denom,
        "col_projection_l1": float(metric_state["col_projection_l1_sum"]) / denom,
        "adjacency_l1": float(metric_state["adjacency_l1_sum"]) / denom,
        "grammar_l1": float(metric_state["grammar_l1_sum"]) / denom,
        "empty_foreground_rate": float(metric_state["empty_count"]) / denom,
        "full_foreground_rate": float(metric_state["full_count"]) / denom,
        "fg_area_low_rate": float(metric_state["fg_area_low_count"]) / denom,
        "fg_area_high_rate": float(metric_state["fg_area_high_count"]) / denom,
        "low_label_diversity_rate": float(metric_state["low_label_diversity_count"]) / denom,
        "invalid_bbox_rate": float(metric_state["invalid_bbox_count"]) / denom,
        "fragmented_rate": float(metric_state["fragmented_count"]) / denom,
        "valid_foreground_rate": float(metric_state["valid_foreground_count"]) / denom,
        "local_z_entropy": float(_safe_entropy(mode_hist)),
        "effective_local_modes": float(_effective_modes(mode_hist)),
        "sampled_unique_local_z_count": float(len(local_counts)),
        "fg_area_mean": float(metric_state["fg_area_sum"]) / denom,
        "fg_area_min": 0.0 if sample_count <= 0 else float(metric_state["fg_area_min"]),
        "fg_area_max": 0.0 if sample_count <= 0 else float(metric_state["fg_area_max"]),
        "fg_area_valid_low_mean": float(metric_state["fg_area_valid_low_sum"]) / denom,
        "fg_area_valid_high_mean": float(metric_state["fg_area_valid_high_sum"]) / denom,
        "label_diversity_mean": float(metric_state["label_diversity_sum"]) / denom,
        "label_diversity_min": 0.0 if sample_count <= 0 else float(metric_state["label_diversity_min"]),
        "label_diversity_max": 0.0 if sample_count <= 0 else float(metric_state["label_diversity_max"]),
        "num_components_mean": float(metric_state["num_components_sum"]) / denom,
        "largest_component_ratio_mean": float(metric_state["largest_component_ratio_sum"]) / denom,
        "tiny_component_count_mean": float(metric_state["tiny_component_count_sum"]) / denom,
    }
    for key, value in list(metrics.items()):
        if key in {
            "empty_foreground_rate",
            "full_foreground_rate",
            "fg_area_low_rate",
            "fg_area_high_rate",
            "low_label_diversity_rate",
            "invalid_bbox_rate",
            "fragmented_rate",
            "valid_foreground_rate",
            "fg_area_mean",
            "fg_area_min",
            "fg_area_max",
            "fg_area_valid_low_mean",
            "fg_area_valid_high_mean",
            "label_diversity_mean",
            "label_diversity_min",
            "label_diversity_max",
            "num_components_mean",
            "largest_component_ratio_mean",
            "tiny_component_count_mean",
        }:
            metrics[f"sampled_{key}"] = value
    return metrics


def _resolve_loss_weights(train_cf: dict[str, object]) -> dict[str, float]:
    loss_weights = cast(dict[str, float], train_cf.get("loss_weights", {}))
    resolved = {
        "local_z": float(loss_weights.get("local_z", 1.0)),
        "mask_bce_weight": float(loss_weights.get("fg_mask", 1.0)),
        "mask_dice_weight": float(train_cf.get("mask_dice_weight", loss_weights.get("mask_dice", 0.5))),
        "mask_soft_iou_weight": float(train_cf.get("mask_soft_iou_weight", loss_weights.get("mask_soft_iou", 0.5))),
        "fg_label_weight": float(loss_weights.get("fg_label", 1.0)),
        "centroid_mask_weight": float(loss_weights.get("centroid_mask_weight", 0.05)),
        "centroid_label_weight": float(train_cf.get("centroid_label_weight", loss_weights.get("centroid_label", 0.05))),
        "bbox": float(loss_weights.get("bbox", 1.0)),
        "row_projection": float(loss_weights.get("row_projection", 1.0)),
        "col_projection": float(loss_weights.get("col_projection", 1.0)),
        "grammar_signature": float(loss_weights.get("grammar_signature", 1.0)),
        "adjacency": float(loss_weights.get("adjacency", 1.0)),
    }
    if "mask_dice_weight" not in train_cf or "mask_soft_iou_weight" not in train_cf or "centroid_label_weight" not in train_cf:
        _warn("train.mask_dice_weight / train.mask_soft_iou_weight / train.centroid_label_weight not fully set; falling back to train.loss_weights for compatibility")
    return resolved


def _evaluate_loader(model: object, loader: object, device: object, torch: object, functional: object, ignore_index: int, fg_label_weight: float) -> dict[str, float]:
    model.eval()
    metric_state = _init_generated_metric_state()
    category_area_stats = loader.dataset.cache_payload["category_foreground_area_stats"]
    with getattr(torch, "no_grad")():
        for batch in loader:
            local_z = batch["local_z"].to(device)
            mode_mask = batch["mode_mask"].to(device)
            if bool((local_z < 0).any().item()) or bool((local_z >= mode_mask.sum(dim=-1)).any().item()):
                raise ValueError("Validation batch contains invalid local_z relative to mode_mask.")
            outputs = model(
                batch["category_ids"].to(device),
                centroid_fg_mask_prob=batch["centroid_fg_mask_prob"].to(device),
                centroid_label_prob_16=batch["centroid_label_prob_16"].to(device),
                centroid_label_hist=batch["centroid_label_hist"].to(device),
                centroid_row_projection=batch["centroid_row_projection"].to(device),
                centroid_col_projection=batch["centroid_col_projection"].to(device),
                centroid_adjacency=batch["centroid_adjacency"].to(device),
                centroid_transition_stats=batch["centroid_transition_stats"].to(device),
                centroid_bbox_stats=batch["centroid_bbox_stats"].to(device),
                local_z=local_z,
                mode_mask=mode_mask,
            )
            fg_mask_target = batch["fg_mask20"].to(device)
            fg_mask_prob = getattr(torch, "sigmoid")(outputs["fg_mask_logits"].squeeze(1))
            fg_target = batch["fg_y20"].to(device)
            fg_label_target = getattr(torch, "where")(fg_target >= 1, fg_target - 1, fg_target)
            losses = {
                "mask_bce": float(functional.binary_cross_entropy_with_logits(outputs["fg_mask_logits"].squeeze(1), fg_mask_target).item()),
                "mask_dice": float(_soft_dice_loss(torch, fg_mask_prob, fg_mask_target).item()),
                "mask_soft_iou": float(_soft_iou_loss(torch, fg_mask_prob, fg_mask_target).item()),
                "raw_fg_ce": float(functional.cross_entropy(outputs["fg_label_logits"], fg_label_target, ignore_index=ignore_index).item()),
                "bbox": float(functional.l1_loss(outputs["bbox_pred"], batch["bbox_stats"].to(device)).item()),
                "centroid_mask": float(functional.binary_cross_entropy_with_logits(outputs["fg_mask_logits"].squeeze(1), batch["centroid_fg_mask_prob"].to(device).squeeze(1)).item()),
                "centroid_label": float(functional.mse_loss(outputs["centroid_label_prob_pred"], batch["centroid_label_prob_16"].to(device)).item()),
                "row": float(functional.l1_loss(outputs["row_projection_pred"], batch["row_projection"].to(device)).item()),
                "col": float(functional.l1_loss(outputs["col_projection_pred"], batch["col_projection"].to(device)).item()),
                "grammar": float(functional.l1_loss(outputs["grammar_signature_pred"], batch["grammar_signature"].to(device)).item()),
                "adj": float(functional.l1_loss(outputs["adjacency_signature_pred"], batch["adjacency_signature"].to(device)).item()),
            }
            losses["weighted_fg_ce"] = losses["raw_fg_ce"] * fg_label_weight
            _accumulate_generated_metrics(torch, metric_state, outputs, batch, losses, category_area_stats)
    return _finalize_generated_metrics(metric_state)


def _require_cache_payload_fields(cache_payload: dict[str, object], *, context: str) -> None:
    require_foreground_cache_fields(cache_payload, context=context)
    require_centroid_sketch_fields(cache_payload, context=context)
    for key in ["centroid_sketch_by_category", "category_to_num_modes", "descriptor_slices"]:
        if key not in cache_payload:
            raise ValueError(f"{context} is missing required field {key!r}.")


def _warn(message: str) -> None:
    print(f"warning: {message}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data_cf = config["data"]
    planner_cf = config["planner"]
    train_cf = config["train"]
    canonical_mode = resolve_canonical_mode(data_cf)
    ignore_index = int(data_cf.get("ignore_index", -100))
    resolved_loss_weights = _resolve_loss_weights(train_cf)
    train_cf["fg_label_weight"] = resolved_loss_weights["fg_label_weight"]
    train_manifest = Path(data_cf["train_manifest"])
    val_manifest = Path(data_cf["val_manifest"])
    train_cache = Path(data_cf["cache_dir"]) / "foreground_cache_train.pt"
    val_cache = Path(data_cf["cache_dir"]) / "foreground_cache_val.pt"
    mapping = _build_category_mapping(train_manifest, val_manifest)
    category_to_id = cast(dict[str, int], mapping["category_to_id"])

    torch, optim = _require_torch()
    device = _resolve_device(torch, str(train_cf["device"]))
    train_loader, train_dataset = build_dataloader(train_manifest, train_cache, batch_size=int(train_cf["batch_size"]), shuffle=True, category_to_id=category_to_id, num_workers=int(train_cf["num_workers"]), pin_memory=bool(train_cf["pin_memory"]), persistent_workers=bool(train_cf["persistent_workers"]), exclude_unseen_categories=False)
    val_loader, val_dataset = build_dataloader(val_manifest, val_cache, batch_size=int(train_cf["batch_size"]), shuffle=False, category_to_id=category_to_id, num_workers=int(train_cf["num_workers"]), pin_memory=bool(train_cf["pin_memory"]), persistent_workers=bool(train_cf["persistent_workers"]), exclude_unseen_categories=True)
    _require_cache_payload_fields(train_dataset.cache_payload, context="Train foreground cache")
    _require_cache_payload_fields(val_dataset.cache_payload, context="Val foreground cache")
    unseen_val_categories = list(val_dataset.skipped_unseen_categories)
    val_total_items = len(load_manifest(val_manifest))
    print(
        format_metric_line(
            "foreground-train-init:",
            [
                ("train_cache_path", str(train_cache)),
                ("val_cache_path", str(val_cache)),
                ("canonical_mode", canonical_mode),
                ("train_items", len(train_dataset)),
                ("val_items_total", val_total_items),
                ("val_seen_items", len(val_dataset)),
                ("val_skipped_unseen", val_dataset.skipped_unseen_count),
                ("categories", len(category_to_id)),
                ("unseen_val_categories", unseen_val_categories),
                ("category_to_num_modes", train_dataset.cache_payload["category_to_num_modes"]),
                ("mask_bce_weight", resolved_loss_weights["mask_bce_weight"]),
                ("mask_dice_weight", resolved_loss_weights["mask_dice_weight"]),
                ("mask_soft_iou_weight", resolved_loss_weights["mask_soft_iou_weight"]),
                ("fg_label_weight", resolved_loss_weights["fg_label_weight"]),
                ("centroid_label_weight", resolved_loss_weights["centroid_label_weight"]),
            ],
        )
    )
    if len(val_dataset) == 0:
        _warn("validation seen split has 0 items after excluding unseen categories; best metric will fallback to train_valid_foreground_rate")
    model_kwargs = {
        "num_categories": len(category_to_id),
        "max_num_modes": int(planner_cf["num_modes_per_category"]),
        "hidden_dim": int(planner_cf["hidden_dim"]),
        "category_embed_dim": int(planner_cf["category_embed_dim"]),
        "mode_embed_dim": int(planner_cf["mode_embed_dim"]),
        "grammar_dim": len(train_dataset[0]["grammar_signature"]),
        "adjacency_dim": len(train_dataset[0]["adjacency_signature"]),
        "bbox_dim": len(train_dataset[0]["bbox_stats"]),
    }
    model = ForegroundCanonicalPlanner(
        **model_kwargs,
    )
    model.to(device)
    optimizer = getattr(optim, "AdamW")(model.parameters(), lr=float(train_cf["learning_rate"]), weight_decay=float(train_cf["weight_decay"]))
    functional = __import__("importlib").import_module("torch.nn.functional")
    history: list[dict[str, object]] = []
    output_dir = Path(train_cf["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_metric_name = "val_sampled_valid_foreground_rate"
    best_metric_value = float("-inf")

    for epoch in range(int(train_cf["epochs"])):
        print(f"\nepoch {epoch + 1}/{int(train_cf['epochs'])}")
        model.train()
        total_loss = 0.0
        metric_state = _init_generated_metric_state()
        batch_step_count = 0
        for batch in train_loader:
            local_z = batch["local_z"].to(device)
            mode_mask = batch["mode_mask"].to(device)
            valid_mode_counts = mode_mask.sum(dim=-1)
            if bool((local_z < 0).any().item()) or bool((local_z >= valid_mode_counts).any().item()):
                raise ValueError("Train batch contains invalid local_z relative to mode_mask.")
            outputs = model(
                batch["category_ids"].to(device),
                centroid_fg_mask_prob=batch["centroid_fg_mask_prob"].to(device),
                centroid_label_prob_16=batch["centroid_label_prob_16"].to(device),
                centroid_label_hist=batch["centroid_label_hist"].to(device),
                centroid_row_projection=batch["centroid_row_projection"].to(device),
                centroid_col_projection=batch["centroid_col_projection"].to(device),
                centroid_adjacency=batch["centroid_adjacency"].to(device),
                centroid_transition_stats=batch["centroid_transition_stats"].to(device),
                centroid_bbox_stats=batch["centroid_bbox_stats"].to(device),
                local_z=local_z,
                mode_mask=mode_mask,
            )
            fg_mask_target = batch["fg_mask20"].to(device)
            fg_mask_prob = getattr(torch, "sigmoid")(outputs["fg_mask_logits"].squeeze(1))
            fg_mask_bce = functional.binary_cross_entropy_with_logits(outputs["fg_mask_logits"].squeeze(1), fg_mask_target)
            fg_mask_dice = _soft_dice_loss(torch, fg_mask_prob, fg_mask_target)
            fg_mask_soft_iou = _soft_iou_loss(torch, fg_mask_prob, fg_mask_target)
            fg_target = batch["fg_y20"].to(device)
            fg_label_target = getattr(torch, "where")(fg_target >= 1, fg_target - 1, fg_target)
            raw_fg_ce = functional.cross_entropy(outputs["fg_label_logits"], fg_label_target, ignore_index=ignore_index)
            weighted_fg_ce = resolved_loss_weights["fg_label_weight"] * raw_fg_ce
            centroid_mask_loss = functional.binary_cross_entropy_with_logits(outputs["fg_mask_logits"].squeeze(1), batch["centroid_fg_mask_prob"].to(device).squeeze(1))
            centroid_label_loss = functional.mse_loss(outputs["centroid_label_prob_pred"], batch["centroid_label_prob_16"].to(device))
            bbox_loss = functional.l1_loss(outputs["bbox_pred"], batch["bbox_stats"].to(device))
            row_loss = functional.l1_loss(outputs["row_projection_pred"], batch["row_projection"].to(device))
            col_loss = functional.l1_loss(outputs["col_projection_pred"], batch["col_projection"].to(device))
            grammar_loss = functional.l1_loss(outputs["grammar_signature_pred"], batch["grammar_signature"].to(device))
            adj_loss = functional.l1_loss(outputs["adjacency_signature_pred"], batch["adjacency_signature"].to(device))
            z_loss = functional.cross_entropy(outputs["local_z_logits"].masked_fill(mode_mask.logical_not(), float("-inf")), local_z)
            loss = (
                resolved_loss_weights["local_z"] * z_loss
                + resolved_loss_weights["mask_bce_weight"] * fg_mask_bce
                + resolved_loss_weights["mask_dice_weight"] * fg_mask_dice
                + resolved_loss_weights["mask_soft_iou_weight"] * fg_mask_soft_iou
                + weighted_fg_ce
                + resolved_loss_weights["centroid_mask_weight"] * centroid_mask_loss
                + resolved_loss_weights["centroid_label_weight"] * centroid_label_loss
                + resolved_loss_weights["bbox"] * bbox_loss
                + resolved_loss_weights["row_projection"] * row_loss
                + resolved_loss_weights["col_projection"] * col_loss
                + resolved_loss_weights["grammar_signature"] * grammar_loss
                + resolved_loss_weights["adjacency"] * adj_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batch_step_count += 1
            losses = {
                "mask_bce": float(fg_mask_bce.item()),
                "mask_dice": float(fg_mask_dice.item()),
                "mask_soft_iou": float(fg_mask_soft_iou.item()),
                "raw_fg_ce": float(raw_fg_ce.item()),
                "weighted_fg_ce": float(weighted_fg_ce.item()),
                "bbox": float(bbox_loss.item()),
                "centroid_mask": float(centroid_mask_loss.item()),
                "centroid_label": float(centroid_label_loss.item()),
                "row": float(row_loss.item()),
                "col": float(col_loss.item()),
                "grammar": float(grammar_loss.item()),
                "adj": float(adj_loss.item()),
            }
            _accumulate_generated_metrics(torch, metric_state, outputs, batch, losses, train_dataset.cache_payload["category_foreground_area_stats"])
            current_count = int(metric_state["sample_count"])
            batch_count = current_count / max(1, int(train_cf["batch_size"]))
            print_progress(
                "fg-train",
                int(min(len(train_loader), math.ceil(batch_count))),
                len(train_loader),
                f"loss={total_loss / max(1, batch_step_count):.4f} raw_fg_ce={losses['raw_fg_ce']:.4f} weighted_fg_ce={losses['weighted_fg_ce']:.4f}",
            )
        finish_progress()
        train_summary = _finalize_generated_metrics(metric_state)
        val_summary = _evaluate_loader(model, val_loader, device, torch, functional, ignore_index, resolved_loss_weights["fg_label_weight"]) if len(val_dataset) > 0 else {}
        summary = {
            "epoch": epoch + 1,
            "train_loss": total_loss / float(max(1, len(train_loader))),
            **{f"train_{key}": value for key, value in train_summary.items()},
            **{f"val_{key}": value for key, value in val_summary.items()},
            "val_seen_count": len(val_dataset),
            "val_unseen_skipped_count": val_dataset.skipped_unseen_count,
            "unseen_val_categories": unseen_val_categories,
            "requires_cache_rebuild": True,
            "checkpoint_compatibility": "spatial-centroid-v2",
        }
        history.append(summary)
        metric_items = [
            ("train_loss", cast(float, summary["train_loss"])),
            ("train_fg_iou", cast(float, summary["train_fg_mask_iou"])),
            ("train_raw_fg_ce", cast(float, summary["train_raw_fg_ce"])),
            ("train_weighted_fg_ce", cast(float, summary["train_weighted_fg_ce"])),
            ("train_empty_rate", cast(float, summary["train_empty_foreground_rate"])),
            ("train_full_rate", cast(float, summary["train_full_foreground_rate"])),
            ("train_area_low_rate", cast(float, summary["train_fg_area_low_rate"])),
            ("train_area_high_rate", cast(float, summary["train_fg_area_high_rate"])),
            ("train_low_div_rate", cast(float, summary["train_low_label_diversity_rate"])),
            ("train_num_comp", cast(float, summary["train_num_components_mean"])),
            ("train_largest_comp", cast(float, summary["train_largest_component_ratio_mean"])),
            ("train_tiny_comp", cast(float, summary["train_tiny_component_count_mean"])),
        ]
        if "val_valid_foreground_rate" in summary:
            metric_items.extend(
                [
                    ("val_valid_fg", cast(float, summary["val_valid_foreground_rate"])),
                    ("val_raw_fg_ce", cast(float, summary["val_raw_fg_ce"])),
                    ("val_weighted_fg_ce", cast(float, summary["val_weighted_fg_ce"])),
                    ("val_empty_rate", cast(float, summary["val_empty_foreground_rate"])),
                    ("val_full_rate", cast(float, summary["val_full_foreground_rate"])),
                    ("val_area_low_rate", cast(float, summary["val_fg_area_low_rate"])),
                    ("val_area_high_rate", cast(float, summary["val_fg_area_high_rate"])),
                    ("val_low_div_rate", cast(float, summary["val_low_label_diversity_rate"])),
                    ("val_num_comp", cast(float, summary["val_num_components_mean"])),
                    ("val_largest_comp", cast(float, summary["val_largest_component_ratio_mean"])),
                    ("val_tiny_comp", cast(float, summary["val_tiny_component_count_mean"])),
                ]
            )
        else:
            metric_items.append(("val_valid_fg", "n/a"))
        print(format_metric_line("summary foreground:", metric_items))

        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "config": config,
            "canonical_mode": canonical_mode,
            "metrics": summary,
            "category_to_id": category_to_id,
            "id_to_category": mapping["id_to_category"],
            "category_to_num_modes": train_dataset.cache_payload["category_to_num_modes"],
            "train_categories": mapping["train_categories"],
            "val_categories": mapping["val_categories"],
            "unseen_val_categories": unseen_val_categories,
            "val_unseen_skipped_count": val_dataset.skipped_unseen_count,
            "descriptor_slices": train_dataset.cache_payload["descriptor_slices"],
            "descriptor_global_mean": train_dataset.cache_payload["descriptor_global_mean"],
            "descriptor_global_std": train_dataset.cache_payload["descriptor_global_std"],
            "category_foreground_area_stats": train_dataset.cache_payload["category_foreground_area_stats"],
            "train_cache_path": str(train_cache),
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
            "history": history,
            "grammar_signature_dim": len(train_dataset[0]["grammar_signature"]),
            "adjacency_signature_dim": len(train_dataset[0]["adjacency_signature"]),
            "bbox_dim": len(train_dataset[0]["bbox_stats"]),
            "max_num_modes": int(planner_cf["num_modes_per_category"]),
            "spatial_condition_channels": 17,
            "model_kwargs": model_kwargs,
            "checkpoint_compatibility": "spatial-centroid-v2",
            "requires_cache_rebuild": True,
        }
        checkpoint_last_path = output_dir / "checkpoint_last.pt"
        getattr(torch, "save")(checkpoint_payload, checkpoint_last_path)
        print(f"saved checkpoint_last: {checkpoint_last_path}")
        current_metric = float(summary.get("val_sampled_valid_foreground_rate", summary["train_sampled_valid_foreground_rate"]))
        if current_metric >= best_metric_value:
            best_metric_value = current_metric
            checkpoint_payload["best_metric_value"] = best_metric_value
            checkpoint_payload["best_metric_name"] = best_metric_name
            best_path = output_dir / "checkpoint.pt"
            getattr(torch, "save")(checkpoint_payload, best_path)
            print(f"saved best checkpoint: {best_path}")

    metrics = {
        "history": history,
        "category_to_id": category_to_id,
        "id_to_category": mapping["id_to_category"],
        "category_to_num_modes": train_dataset.cache_payload["category_to_num_modes"],
        "train_categories": mapping["train_categories"],
        "val_categories": mapping["val_categories"],
        "unseen_val_categories": unseen_val_categories,
        "val_unseen_skipped_count": val_dataset.skipped_unseen_count,
        "descriptor_slices": train_dataset.cache_payload["descriptor_slices"],
        "descriptor_global_mean": train_dataset.cache_payload["descriptor_global_mean"],
        "descriptor_global_std": train_dataset.cache_payload["descriptor_global_std"],
        "category_foreground_area_stats": train_dataset.cache_payload["category_foreground_area_stats"],
        "train_cache_path": str(train_cache),
        "grammar_signature_dim": len(train_dataset[0]["grammar_signature"]),
        "adjacency_signature_dim": len(train_dataset[0]["adjacency_signature"]),
        "bbox_dim": len(train_dataset[0]["bbox_stats"]),
        "max_num_modes": int(planner_cf["num_modes_per_category"]),
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "config": config,
        "canonical_mode": canonical_mode,
        "spatial_condition_channels": 17,
        "checkpoint_compatibility": "spatial-centroid-v2",
        "requires_cache_rebuild": True,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
