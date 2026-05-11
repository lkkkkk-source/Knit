from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from .dataset import build_dataloader, load_manifest
from .models.planner import LatentPlanner
from .utils import ensure_palette_path, finish_progress, format_metric_line, load_config, print_progress


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        optim = importlib.import_module("torch.optim")
    except ImportError as error:
        raise ImportError("PyTorch is required for planner training. Install with `pip install -e .[train]`.") from error
    return torch, optim


def _resolve_device(torch: object, device_name: str) -> object:
    device_cls = getattr(torch, "device")
    if device_name == "cpu":
        return device_cls("cpu")
    if not getattr(torch, "cuda").is_available():
        raise RuntimeError(f"Requested device {device_name!r}, but CUDA is not available.")
    return device_cls(device_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train latent global planner for category-only structure prior.")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def _build_category_mapping(train_manifest: Path, val_manifest: Path | None) -> dict[str, int]:
    train_categories = sorted({sample["category"] for sample in load_manifest(train_manifest)})
    categories = set(train_categories)
    val_categories: list[str] = []
    if val_manifest is not None:
        val_categories = sorted({sample["category"] for sample in load_manifest(val_manifest)})
        categories.update(val_categories)
    category_to_id = {category: index for index, category in enumerate(sorted(categories))}
    unseen_val_categories = sorted(set(val_categories) - set(train_categories))
    return {
        "category_to_id": category_to_id,
        "train_categories": train_categories,
        "val_categories": val_categories,
        "unseen_val_categories": unseen_val_categories,
    }


def _batch_losses(functional: object, outputs: dict[str, object], z: object, c5: object, o5: object, r17: object, fg_ratio: object) -> dict[str, object]:
    z_loss = functional.cross_entropy(outputs["z_logits"], z, reduction="none")
    c5_map = functional.cross_entropy(outputs["c5_logits"], c5, reduction="none")
    c5_loss = c5_map.mean(dim=(1, 2))
    o5_map = functional.binary_cross_entropy_with_logits(outputs["o5_logits"].squeeze(1), o5, reduction="none")
    o5_loss = o5_map.mean(dim=(1, 2))
    r17_loss = (outputs["r17_pred"] - r17).abs().mean(dim=1)
    fg_loss = (outputs["fg_ratio_pred"] - fg_ratio).abs()
    total = z_loss + c5_loss + o5_loss + 0.1 * r17_loss + 0.1 * fg_loss
    return {
        "total": total,
        "z": z_loss,
        "c5": c5_loss,
        "o5": o5_loss,
        "count": r17_loss,
        "fg": fg_loss,
    }


def _planner_losses_v11(
    functional: object,
    outputs: dict[str, object],
    local_z: object,
    mode_mask: object,
    c5: object,
    o5: object,
    c10: object,
    o10: object,
    r17: object,
    fg_ratio: object,
    row_projection: object,
    col_projection: object,
    grammar_signature: object,
    adjacency_signature: object,
) -> dict[str, object]:
    invalid_local = local_z >= mode_mask.sum(dim=-1).to(dtype=local_z.dtype)
    if bool(invalid_local.any().item()):
        bad_index = int(invalid_local.nonzero(as_tuple=False)[0].item())
        raise ValueError(
            f"local_z out of range for mode_mask at batch_index={bad_index}: "
            f"local_z={int(local_z[bad_index].item())} valid_modes={int(mode_mask[bad_index].sum().item())}"
        )
    masked_logits = outputs["z_logits"].masked_fill(mode_mask.logical_not(), float("-inf"))
    z_loss = functional.cross_entropy(masked_logits, local_z, reduction="none")
    c5_loss = functional.cross_entropy(outputs["c5_logits"], c5, reduction="none").mean(dim=(1, 2))
    o5_loss = functional.binary_cross_entropy_with_logits(outputs["o5_logits"].squeeze(1), o5, reduction="none").mean(dim=(1, 2))
    c10_loss = functional.cross_entropy(outputs["c10_logits"], c10, reduction="none").mean(dim=(1, 2))
    o10_loss = functional.binary_cross_entropy_with_logits(outputs["o10_logits"].squeeze(1), o10, reduction="none").mean(dim=(1, 2))
    r17_loss = (outputs["r17_pred"] - r17).abs().mean(dim=1)
    fg_loss = (outputs["fg_ratio_pred"] - fg_ratio).abs()
    row_loss = (outputs["row_projection_pred"] - row_projection).abs().mean(dim=1)
    col_loss = (outputs["col_projection_pred"] - col_projection).abs().mean(dim=1)
    sig_loss = (outputs["grammar_signature_pred"] - grammar_signature).abs().mean(dim=1)
    adj_loss = (outputs["adjacency_signature_pred"] - adjacency_signature).abs().mean(dim=1)
    total = z_loss + c5_loss + o5_loss + c10_loss + o10_loss + 0.1 * r17_loss + 0.1 * fg_loss + 0.2 * row_loss + 0.2 * col_loss + 0.1 * sig_loss + 0.1 * adj_loss
    return {
        "total": total,
        "z": z_loss,
        "masked_z_logits": masked_logits,
        "c5": c5_loss,
        "o5": o5_loss,
        "c10": c10_loss,
        "o10": o10_loss,
        "count": r17_loss,
        "fg": fg_loss,
        "row": row_loss,
        "col": col_loss,
        "sig": sig_loss,
        "adj": adj_loss,
    }


def _category_barrier(
    fg_ratio_raw: object,
    o5_logits: object,
    o10_logits: object,
    fg_valid_low: object,
    fg_valid_high: object,
    o5_valid_low_ratio: object,
    o5_valid_high_ratio: object,
    o10_valid_low_ratio: object,
    o10_valid_high_ratio: object,
) -> tuple[object, object, dict[str, float]]:
    torch, _ = _require_torch()
    functional = __import__("importlib").import_module("torch.nn.functional")
    fg_ratio_prob = fg_ratio_raw
    mean_o5 = getattr(torch, "sigmoid")(o5_logits).mean(dim=(1, 2, 3))
    mean_o10 = getattr(torch, "sigmoid")(o10_logits).mean(dim=(1, 2, 3))
    for name, tensor in {
        "fg_valid_low": fg_valid_low,
        "fg_valid_high": fg_valid_high,
        "o5_valid_low_ratio": o5_valid_low_ratio,
        "o5_valid_high_ratio": o5_valid_high_ratio,
        "o10_valid_low_ratio": o10_valid_low_ratio,
        "o10_valid_high_ratio": o10_valid_high_ratio,
    }.items():
        if tensor is None:
            raise ValueError(f"{name} is missing from batch diagnostics")
    if bool((fg_valid_low < 0).any().item()) or bool((fg_valid_high <= 0).all().item()):
        raise ValueError("Invalid fg validity bounds in batch")
    fg_barrier = (
        functional.softplus(fg_valid_low - fg_ratio_prob)
        + functional.softplus(fg_ratio_prob - fg_valid_high)
    ).mean()
    occ_barrier = (
        functional.softplus(o5_valid_low_ratio - mean_o5)
        + functional.softplus(mean_o5 - o5_valid_high_ratio)
        + functional.softplus(o10_valid_low_ratio - mean_o10)
        + functional.softplus(mean_o10 - o10_valid_high_ratio)
    ).mean()
    all_bg_mask = fg_ratio_prob <= 0.02
    all_fg_mask = fg_ratio_prob >= 0.98
    valid_mask = (fg_ratio_prob >= fg_valid_low) & (fg_ratio_prob <= fg_valid_high)
    rates = {
        "pred_all_background_rate": float(all_bg_mask.to(dtype=getattr(torch, "float32")).mean().item()),
        "pred_all_foreground_rate": float(all_fg_mask.to(dtype=getattr(torch, "float32")).mean().item()),
        "pred_valid_fg_rate": float(valid_mask.to(dtype=getattr(torch, "float32")).mean().item()),
        "pred_fg_ratio_mean": float(fg_ratio_prob.mean().item()),
        "pred_fg_ratio_min": float(fg_ratio_prob.min().item()),
        "pred_fg_ratio_max": float(fg_ratio_prob.max().item()),
        "valid_low_mean": float(fg_valid_low.mean().item()),
        "valid_high_mean": float(fg_valid_high.mean().item()),
        "mean_o5_occ": float(mean_o5.mean().item()),
        "mean_o10_occ": float(mean_o10.mean().item()),
        "o5_low_mean": float(o5_valid_low_ratio.mean().item()),
        "o5_high_mean": float(o5_valid_high_ratio.mean().item()),
        "o10_low_mean": float(o10_valid_low_ratio.mean().item()),
        "o10_high_mean": float(o10_valid_high_ratio.mean().item()),
    }
    return fg_barrier, occ_barrier, rates


def _diag_update(store: dict[str, object], probs: object, sampled_z: object, categories: list[str]) -> None:
    torch, _ = _require_torch()
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
    top_probs, top_ids = getattr(torch, "topk")(probs, k=min(5, probs.shape[-1]), dim=-1)
    sampled_list = sampled_z.detach().cpu().tolist()
    category_stats = cast(dict[str, dict[str, object]], store.setdefault("category_stats", {}))
    for index, category in enumerate(categories):
        slot = category_stats.setdefault(category, {"count": 0, "entropy_sum": 0.0, "top1_sum": 0.0, "top5_sum": 0.0, "sampled_z": [], "top_z_hist": {}})
        slot["count"] = int(slot["count"]) + 1
        slot["entropy_sum"] = float(slot["entropy_sum"]) + float(entropy[index].item())
        slot["top1_sum"] = float(slot["top1_sum"]) + float(top_probs[index, 0].item())
        slot["top5_sum"] = float(slot["top5_sum"]) + float(top_probs[index].sum().item())
        slot["sampled_z"].append(int(sampled_list[index]))
        top_id = int(top_ids[index, 0].item())
        top_hist = cast(dict[int, int], slot["top_z_hist"])
        top_hist[top_id] = top_hist.get(top_id, 0) + 1
    store.setdefault("entropy", []).extend(float(value) for value in entropy.detach().cpu().tolist())
    store.setdefault("top1", []).extend(float(value) for value in top_probs[:, 0].detach().cpu().tolist())
    store.setdefault("top5", []).extend(float(value) for value in top_probs.sum(dim=-1).detach().cpu().tolist())
    store.setdefault("sampled_z", []).extend(int(value) for value in sampled_list)


def _diag_finalize(store: dict[str, object], num_modes: int) -> dict[str, object]:
    import math

    entropy_values = cast(list[float], store.get("entropy", []))
    top1_values = cast(list[float], store.get("top1", []))
    top5_values = cast(list[float], store.get("top5", []))
    sampled_z = cast(list[int], store.get("sampled_z", []))
    mean_entropy = sum(entropy_values) / max(1, len(entropy_values))
    unique_z = sorted(set(sampled_z))
    category_stats = cast(dict[str, dict[str, object]], store.get("category_stats", {}))
    z_by_category: dict[str, dict[str, object]] = {}
    for category, slot in category_stats.items():
        count = int(slot["count"])
        top_hist = cast(dict[int, int], slot["top_z_hist"])
        top_z_ids = [item[0] for item in sorted(top_hist.items(), key=lambda item: (-item[1], item[0]))[:5]]
        mean_cat_entropy = float(slot["entropy_sum"]) / max(1, count)
        z_by_category[category] = {
            "count": count,
            "mean_entropy": mean_cat_entropy,
            "mean_top1_prob": float(slot["top1_sum"]) / max(1, count),
            "effective_num_modes": math.exp(mean_cat_entropy),
            "top_z_ids": top_z_ids,
        }
    return {
        "z_entropy": mean_entropy,
        "z_top1_prob": sum(top1_values) / max(1, len(top1_values)),
        "z_top5_prob_sum": sum(top5_values) / max(1, len(top5_values)),
        "effective_num_modes": math.exp(mean_entropy),
        "sampled_unique_z_count": len(unique_z),
        "sampled_unique_z_ratio": len(unique_z) / float(max(1, num_modes)),
        "z_by_category": z_by_category,
    }


def _masked_mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data_cf = config["data"]
    planner_cf = config["planner"]
    train_cf = config["train_planner"]
    cache_cf = config.get("cache", {})
    loss_cf = config["loss"]
    manifest_path = Path(data_cf["train_manifest"])
    val_manifest_path = Path(data_cf["val_manifest"])
    plan_cache_path = Path(data_cf["plan_cache_dir"]) / f"{manifest_path.stem}.pt"
    val_plan_cache_path = Path(data_cf["plan_cache_dir"]) / f"{val_manifest_path.stem}.pt"
    output_dir = Path(train_cf["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    torch, optim = _require_torch()
    device = _resolve_device(torch, str(train_cf["device"]))
    palette_path = ensure_palette_path(manifest_path, data_cf.get("palette_path"))
    mapping_info = _build_category_mapping(manifest_path, val_manifest_path)
    category_to_id = cast(dict[str, int], mapping_info["category_to_id"])
    train_categories = cast(list[str], mapping_info["train_categories"])
    val_categories = cast(list[str], mapping_info["val_categories"])
    unseen_val_categories = cast(list[str], mapping_info["unseen_val_categories"])
    if unseen_val_categories:
        print(f"warning: unseen validation categories detected: {unseen_val_categories}")
    if val_categories and len(unseen_val_categories) == len(val_categories):
        print("warning: validation split contains only unseen categories")
    train_loader, train_dataset = build_dataloader(
        manifest_path,
        palette_path=palette_path,
        plan_cache_path=plan_cache_path,
        batch_size=int(train_cf["batch_size"]),
        shuffle=True,
        category_to_id=category_to_id,
        num_workers=int(train_cf["num_workers"]),
        pin_memory=bool(train_cf["pin_memory"]),
        persistent_workers=bool(train_cf["persistent_workers"]),
    )
    val_loader, _ = build_dataloader(
        val_manifest_path,
        palette_path=palette_path,
        plan_cache_path=val_plan_cache_path,
        batch_size=int(train_cf["batch_size"]),
        shuffle=False,
        category_to_id=category_to_id,
        num_workers=int(train_cf["num_workers"]),
        pin_memory=bool(train_cf["pin_memory"]),
        persistent_workers=bool(train_cf["persistent_workers"]),
    )
    model = LatentPlanner(
        num_categories=len(category_to_id),
        num_modes=int(planner_cf["num_modes"]),
        coarse_size=int(data_cf["coarse_size"]),
        num_classes=int(data_cf["num_classes"]),
        category_embed_dim=int(planner_cf["category_embed_dim"]),
        mode_embed_dim=int(planner_cf["mode_embed_dim"]),
        hidden_dim=int(planner_cf["hidden_dim"]),
        num_layers=int(planner_cf["num_layers"]),
        coarse_size_10=10,
        grammar_dim=len(train_dataset[0]["grammar_signature"]),
        adjacency_dim=len(train_dataset[0]["adjacency_signature"]),
        max_num_modes_per_category=int(planner_cf.get("max_num_modes_per_category", 16)),
    )
    model.to(device)
    optimizer = getattr(optim, "AdamW")(model.parameters(), lr=float(train_cf["learning_rate"]), weight_decay=float(train_cf["weight_decay"]))
    functional = __import__("importlib").import_module("torch.nn.functional")
    history: list[dict[str, object]] = []
    z_category_history: dict[str, dict[str, object]] = {}
    best_metric_name = "val_seen_loss"
    best_metric_value = float("inf")
    category_fg_stats = train_dataset.cache_payload.get("category_fg_stats", {})
    category_occ_stats = train_dataset.cache_payload.get("category_occ_stats", {})
    print(format_metric_line("categories:", [("num_categories", len(category_to_id)), ("train_categories", len(train_categories)), ("val_categories", len(val_categories)), ("unseen_val_categories", unseen_val_categories)]))
    for epoch in range(int(train_cf["epochs"])):
        print(f"\nepoch {epoch + 1}/{int(train_cf['epochs'])}")
        model.train()
        total_loss = 0.0
        total_z = 0.0
        total_c5 = 0.0
        total_o5 = 0.0
        total_c10 = 0.0
        total_o10 = 0.0
        total_r17 = 0.0
        total_fg = 0.0
        total_sig = 0.0
        total_adj = 0.0
        total_fg_barrier = 0.0
        total_occ_barrier = 0.0
        batch_count = 0
        train_diag: dict[str, object] = {}
        train_all_bg = 0.0
        train_all_fg = 0.0
        train_valid_fg = 0.0
        train_fg_ratio_mean = 0.0
        train_fg_ratio_min = 0.0
        train_fg_ratio_max = 0.0
        train_valid_low_mean = 0.0
        train_valid_high_mean = 0.0
        train_mean_o5_occ = 0.0
        train_mean_o10_occ = 0.0
        train_o5_low_mean = 0.0
        train_o5_high_mean = 0.0
        train_o10_low_mean = 0.0
        train_o10_high_mean = 0.0
        for batch in train_loader:
            category_ids = batch["category_ids"].to(device)
            z = batch["local_z"].to(device)
            mode_mask = batch["mode_mask"].to(device)
            c5 = batch["c5"].to(device)
            o5 = batch["o5"].to(device)
            c10 = batch["c10"].to(device)
            o10 = batch["o10"].to(device)
            r17 = batch["r17"].to(device)
            fg_ratio = batch["fg_ratio"].to(device)
            row_projection = batch["row_projection"].to(device)
            col_projection = batch["col_projection"].to(device)
            grammar_signature = batch["grammar_signature"].to(device)
            adjacency_signature = batch["adjacency_signature"].to(device)
            fg_valid_low = batch["fg_valid_low"].to(device)
            fg_valid_high = batch["fg_valid_high"].to(device)
            o5_valid_low_ratio = batch["o5_valid_low_ratio"].to(device)
            o5_valid_high_ratio = batch["o5_valid_high_ratio"].to(device)
            o10_valid_low_ratio = batch["o10_valid_low_ratio"].to(device)
            o10_valid_high_ratio = batch["o10_valid_high_ratio"].to(device)
            outputs = model(category_ids, z_ids=z, mode_mask=mode_mask, sample_mode="teacher")
            losses = _planner_losses_v11(functional, outputs, z, mode_mask, c5, o5, c10, o10, r17, fg_ratio, row_projection, col_projection, grammar_signature, adjacency_signature)
            z_loss = losses["z"].mean()
            c5_loss = losses["c5"].mean()
            o5_loss = losses["o5"].mean()
            c10_loss = losses["c10"].mean()
            o10_loss = losses["o10"].mean()
            r17_loss = losses["count"].mean()
            fg_loss = losses["fg"].mean()
            sig_loss = losses["sig"].mean()
            adj_loss = losses["adj"].mean()
            fg_barrier, occ_barrier, fg_rates = _category_barrier(
                outputs["fg_ratio_pred"],
                outputs["o5_logits"],
                outputs["o10_logits"],
                fg_valid_low,
                fg_valid_high,
                o5_valid_low_ratio,
                o5_valid_high_ratio,
                o10_valid_low_ratio,
                o10_valid_high_ratio,
            )
            loss = losses["total"].mean() + float(loss_cf["fg_barrier"]) * fg_barrier + float(loss_cf["occ_barrier"]) * occ_barrier + float(loss_cf["anti_empty_full"]) * (fg_barrier + occ_barrier)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            probs = functional.softmax(losses["masked_z_logits"], dim=-1)
            sampled_z = getattr(torch, "distributions").Categorical(probs=probs).sample()
            _diag_update(train_diag, probs, sampled_z, batch["categories"])
            total_loss += float(loss.item())
            total_z += float(z_loss.item())
            total_c5 += float(c5_loss.item())
            total_o5 += float(o5_loss.item())
            total_c10 += float(c10_loss.item())
            total_o10 += float(o10_loss.item())
            total_r17 += float(r17_loss.item())
            total_fg += float(fg_loss.item())
            total_sig += float(sig_loss.item())
            total_adj += float(adj_loss.item())
            total_fg_barrier += float(fg_barrier.item())
            total_occ_barrier += float(occ_barrier.item())
            train_all_bg += fg_rates["pred_all_background_rate"]
            train_all_fg += fg_rates["pred_all_foreground_rate"]
            train_valid_fg += fg_rates["pred_valid_fg_rate"]
            train_fg_ratio_mean += fg_rates["pred_fg_ratio_mean"]
            train_fg_ratio_min += fg_rates["pred_fg_ratio_min"]
            train_fg_ratio_max += fg_rates["pred_fg_ratio_max"]
            train_valid_low_mean += fg_rates["valid_low_mean"]
            train_valid_high_mean += fg_rates["valid_high_mean"]
            train_mean_o5_occ += fg_rates["mean_o5_occ"]
            train_mean_o10_occ += fg_rates["mean_o10_occ"]
            train_o5_low_mean += fg_rates["o5_low_mean"]
            train_o5_high_mean += fg_rates["o5_high_mean"]
            train_o10_low_mean += fg_rates["o10_low_mean"]
            train_o10_high_mean += fg_rates["o10_high_mean"]
            batch_count += 1
            print_progress("planner-train", batch_count, len(train_loader), f"loss={total_loss / batch_count:.4f} z={total_z / batch_count:.4f} c5={total_c5 / batch_count:.4f} c10={total_c10 / batch_count:.4f} fg_bar={total_fg_barrier / batch_count:.6f} occ_bar={total_occ_barrier / batch_count:.6f}")
        finish_progress()

        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_seen_total: list[float] = []
        val_unseen_total: list[float] = []
        val_seen_z: list[float] = []
        val_unseen_z: list[float] = []
        val_seen_c5: list[float] = []
        val_unseen_c5: list[float] = []
        val_seen_o5: list[float] = []
        val_unseen_o5: list[float] = []
        val_seen_count: list[float] = []
        val_unseen_count: list[float] = []
        val_fg_barrier = 0.0
        val_occ_barrier = 0.0
        val_all_bg = 0.0
        val_all_fg = 0.0
        val_valid_fg = 0.0
        val_fg_ratio_mean = 0.0
        val_fg_ratio_min = 0.0
        val_fg_ratio_max = 0.0
        val_valid_low_mean = 0.0
        val_valid_high_mean = 0.0
        val_mean_o5_occ = 0.0
        val_mean_o10_occ = 0.0
        val_o5_low_mean = 0.0
        val_o5_high_mean = 0.0
        val_o10_low_mean = 0.0
        val_o10_high_mean = 0.0
        val_diag: dict[str, object] = {}
        train_category_set = set(train_categories)
        with getattr(torch, "no_grad")():
            for batch in val_loader:
                category_ids = batch["category_ids"].to(device)
                z = batch["local_z"].to(device)
                mode_mask = batch["mode_mask"].to(device)
                c5 = batch["c5"].to(device)
                o5 = batch["o5"].to(device)
                c10 = batch["c10"].to(device)
                o10 = batch["o10"].to(device)
                r17 = batch["r17"].to(device)
                fg_ratio = batch["fg_ratio"].to(device)
                row_projection = batch["row_projection"].to(device)
                col_projection = batch["col_projection"].to(device)
                grammar_signature = batch["grammar_signature"].to(device)
                adjacency_signature = batch["adjacency_signature"].to(device)
                fg_valid_low = batch["fg_valid_low"].to(device)
                fg_valid_high = batch["fg_valid_high"].to(device)
                o5_valid_low_ratio = batch["o5_valid_low_ratio"].to(device)
                o5_valid_high_ratio = batch["o5_valid_high_ratio"].to(device)
                o10_valid_low_ratio = batch["o10_valid_low_ratio"].to(device)
                o10_valid_high_ratio = batch["o10_valid_high_ratio"].to(device)
                outputs = model(category_ids, z_ids=z, mode_mask=mode_mask, sample_mode="teacher")
                losses = _planner_losses_v11(functional, outputs, z, mode_mask, c5, o5, c10, o10, r17, fg_ratio, row_projection, col_projection, grammar_signature, adjacency_signature)
                fg_barrier, occ_barrier, fg_rates = _category_barrier(
                    outputs["fg_ratio_pred"],
                    outputs["o5_logits"],
                    outputs["o10_logits"],
                    fg_valid_low,
                    fg_valid_high,
                    o5_valid_low_ratio,
                    o5_valid_high_ratio,
                    o10_valid_low_ratio,
                    o10_valid_high_ratio,
                )
                val_loss += float((losses["total"].mean() + float(loss_cf["fg_barrier"]) * fg_barrier + float(loss_cf["occ_barrier"]) * occ_barrier + float(loss_cf["anti_empty_full"]) * (fg_barrier + occ_barrier)).item())
                val_batches += 1
                probs = functional.softmax(losses["masked_z_logits"], dim=-1)
                sampled_z = getattr(torch, "distributions").Categorical(probs=probs).sample()
                _diag_update(val_diag, probs, sampled_z, batch["categories"])
                val_fg_barrier += float(fg_barrier.item())
                val_occ_barrier += float(occ_barrier.item())
                val_all_bg += fg_rates["pred_all_background_rate"]
                val_all_fg += fg_rates["pred_all_foreground_rate"]
                val_valid_fg += fg_rates["pred_valid_fg_rate"]
                val_fg_ratio_mean += fg_rates["pred_fg_ratio_mean"]
                val_fg_ratio_min += fg_rates["pred_fg_ratio_min"]
                val_fg_ratio_max += fg_rates["pred_fg_ratio_max"]
                val_valid_low_mean += fg_rates["valid_low_mean"]
                val_valid_high_mean += fg_rates["valid_high_mean"]
                val_mean_o5_occ += fg_rates["mean_o5_occ"]
                val_mean_o10_occ += fg_rates["mean_o10_occ"]
                val_o5_low_mean += fg_rates["o5_low_mean"]
                val_o5_high_mean += fg_rates["o5_high_mean"]
                val_o10_low_mean += fg_rates["o10_low_mean"]
                val_o10_high_mean += fg_rates["o10_high_mean"]
                for sample_index, category in enumerate(batch["categories"]):
                    target_total = val_seen_total if category in train_category_set else val_unseen_total
                    target_z = val_seen_z if category in train_category_set else val_unseen_z
                    target_c5 = val_seen_c5 if category in train_category_set else val_unseen_c5
                    target_o5 = val_seen_o5 if category in train_category_set else val_unseen_o5
                    target_count = val_seen_count if category in train_category_set else val_unseen_count
                    target_total.append(float(losses["total"][sample_index].item()))
                    target_z.append(float(losses["z"][sample_index].item()))
                    target_c5.append(float(losses["c5"][sample_index].item()))
                    target_o5.append(float(losses["o5"][sample_index].item()))
                    target_count.append(float(losses["count"][sample_index].item()))
        summary = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, batch_count),
            "train_z": total_z / max(1, batch_count),
            "train_c5": total_c5 / max(1, batch_count),
            "train_o5": total_o5 / max(1, batch_count),
            "train_c10": total_c10 / max(1, batch_count),
            "train_o10": total_o10 / max(1, batch_count),
            "train_signature": total_sig / max(1, batch_count),
            "train_adj": total_adj / max(1, batch_count),
            "train_fg_barrier": total_fg_barrier / max(1, batch_count),
            "train_occ_barrier": total_occ_barrier / max(1, batch_count),
            "train_pred_all_background_rate": train_all_bg / max(1, batch_count),
            "train_pred_all_foreground_rate": train_all_fg / max(1, batch_count),
            "train_pred_valid_fg_rate": train_valid_fg / max(1, batch_count),
            "train_pred_fg_ratio_mean": train_fg_ratio_mean / max(1, batch_count),
            "train_pred_fg_ratio_min": train_fg_ratio_min / max(1, batch_count),
            "train_pred_fg_ratio_max": train_fg_ratio_max / max(1, batch_count),
            "train_valid_low_mean": train_valid_low_mean / max(1, batch_count),
            "train_valid_high_mean": train_valid_high_mean / max(1, batch_count),
            "train_mean_o5_occ": train_mean_o5_occ / max(1, batch_count),
            "train_mean_o10_occ": train_mean_o10_occ / max(1, batch_count),
            "train_o5_low_mean": train_o5_low_mean / max(1, batch_count),
            "train_o5_high_mean": train_o5_high_mean / max(1, batch_count),
            "train_o10_low_mean": train_o10_low_mean / max(1, batch_count),
            "train_o10_high_mean": train_o10_high_mean / max(1, batch_count),
            "val_loss": val_loss / max(1, val_batches),
            "val_seen_loss": _masked_mean(val_seen_total),
            "val_unseen_loss": _masked_mean(val_unseen_total),
            "val_seen_z": _masked_mean(val_seen_z),
            "val_unseen_z": _masked_mean(val_unseen_z),
            "val_seen_c5": _masked_mean(val_seen_c5),
            "val_unseen_c5": _masked_mean(val_unseen_c5),
            "val_seen_o5": _masked_mean(val_seen_o5),
            "val_unseen_o5": _masked_mean(val_unseen_o5),
            "val_seen_count": _masked_mean(val_seen_count),
            "val_unseen_count": _masked_mean(val_unseen_count),
            "val_fg_barrier": val_fg_barrier / max(1, val_batches),
            "val_occ_barrier": val_occ_barrier / max(1, val_batches),
            "val_pred_all_background_rate": val_all_bg / max(1, val_batches),
            "val_pred_all_foreground_rate": val_all_fg / max(1, val_batches),
            "val_pred_valid_fg_rate": val_valid_fg / max(1, val_batches),
            "val_pred_fg_ratio_mean": val_fg_ratio_mean / max(1, val_batches),
            "val_pred_fg_ratio_min": val_fg_ratio_min / max(1, val_batches),
            "val_pred_fg_ratio_max": val_fg_ratio_max / max(1, val_batches),
            "val_valid_low_mean": val_valid_low_mean / max(1, val_batches),
            "val_valid_high_mean": val_valid_high_mean / max(1, val_batches),
            "val_mean_o5_occ": val_mean_o5_occ / max(1, val_batches),
            "val_mean_o10_occ": val_mean_o10_occ / max(1, val_batches),
            "val_o5_low_mean": val_o5_low_mean / max(1, val_batches),
            "val_o5_high_mean": val_o5_high_mean / max(1, val_batches),
            "val_o10_low_mean": val_o10_low_mean / max(1, val_batches),
            "val_o10_high_mean": val_o10_high_mean / max(1, val_batches),
        }
        summary.update({f"train_{key}": value for key, value in _diag_finalize(train_diag, int(planner_cf["num_modes"])).items() if key != "z_by_category"})
        summary.update({f"val_{key}": value for key, value in _diag_finalize(val_diag, int(planner_cf["num_modes"])).items() if key != "z_by_category"})
        z_category_history[f"epoch_{epoch + 1:03d}"] = {
            "train": _diag_finalize(train_diag, int(planner_cf["num_modes"]))["z_by_category"],
            "val": _diag_finalize(val_diag, int(planner_cf["num_modes"]))["z_by_category"],
        }
        history.append(summary)
        print(format_metric_line("summary planner:", [("train_loss", cast(float, summary["train_loss"])), ("train_z", cast(float, summary["train_z"])), ("train_c5", cast(float, summary["train_c5"])), ("train_o5", cast(float, summary["train_o5"])), ("train_c10", cast(float, summary["train_c10"])), ("train_o10", cast(float, summary["train_o10"])), ("train_sig", cast(float, summary["train_signature"])), ("train_adj", cast(float, summary["train_adj"])), ("val_loss", cast(float, summary["val_loss"]))]))
        print(format_metric_line("summary seen  :", [("val_seen_loss", cast(float, summary["val_seen_loss"])), ("val_seen_z", cast(float, summary["val_seen_z"])), ("val_seen_c5", cast(float, summary["val_seen_c5"])), ("val_seen_o5", cast(float, summary["val_seen_o5"])), ("val_seen_count", cast(float, summary["val_seen_count"]))]))
        print(format_metric_line("summary unseen:", [("val_unseen_loss", cast(float, summary["val_unseen_loss"])), ("val_unseen_z", cast(float, summary["val_unseen_z"])), ("val_unseen_c5", cast(float, summary["val_unseen_c5"])), ("val_unseen_o5", cast(float, summary["val_unseen_o5"])), ("val_unseen_count", cast(float, summary["val_unseen_count"]))]))
        print(format_metric_line("zdiag train  :", [("entropy", cast(float, summary["train_z_entropy"])), ("top1", cast(float, summary["train_z_top1_prob"])), ("top5", cast(float, summary["train_z_top5_prob_sum"])), ("eff_modes", cast(float, summary["train_effective_num_modes"])), ("uniq_z", int(summary["train_sampled_unique_z_count"])), ("uniq_ratio", cast(float, summary["train_sampled_unique_z_ratio"]))]))
        print(format_metric_line("zdiag val    :", [("entropy", cast(float, summary["val_z_entropy"])), ("top1", cast(float, summary["val_z_top1_prob"])), ("top5", cast(float, summary["val_z_top5_prob_sum"])), ("eff_modes", cast(float, summary["val_effective_num_modes"])), ("uniq_z", int(summary["val_sampled_unique_z_count"])), ("uniq_ratio", cast(float, summary["val_sampled_unique_z_ratio"]))]))
        print(format_metric_line("validity     :", [("train_valid_fg", cast(float, summary["train_pred_valid_fg_rate"])), ("train_all_bg", cast(float, summary["train_pred_all_background_rate"])), ("train_all_fg", cast(float, summary["train_pred_all_foreground_rate"])), ("val_valid_fg", cast(float, summary["val_pred_valid_fg_rate"])), ("val_all_bg", cast(float, summary["val_pred_all_background_rate"])), ("val_all_fg", cast(float, summary["val_pred_all_foreground_rate"]))]))
        print(format_metric_line("validity-train:", [("fg_bar", f"{cast(float, summary['train_fg_barrier']):.6f}"), ("occ_bar", f"{cast(float, summary['train_occ_barrier']):.6f}"), ("pred_fg_mean", cast(float, summary["train_pred_fg_ratio_mean"])), ("pred_fg_min", cast(float, summary["train_pred_fg_ratio_min"])), ("pred_fg_max", cast(float, summary["train_pred_fg_ratio_max"])), ("valid_low", cast(float, summary["train_valid_low_mean"])), ("valid_high", cast(float, summary["train_valid_high_mean"])), ("mean_o5", cast(float, summary["train_mean_o5_occ"])), ("mean_o10", cast(float, summary["train_mean_o10_occ"])), ("o5_low", cast(float, summary["train_o5_low_mean"])), ("o5_high", cast(float, summary["train_o5_high_mean"])), ("o10_low", cast(float, summary["train_o10_low_mean"])), ("o10_high", cast(float, summary["train_o10_high_mean"]))]))
        print(format_metric_line("validity-val  :", [("fg_bar", f"{cast(float, summary['val_fg_barrier']):.6f}"), ("occ_bar", f"{cast(float, summary['val_occ_barrier']):.6f}"), ("pred_fg_mean", cast(float, summary["val_pred_fg_ratio_mean"])), ("pred_fg_min", cast(float, summary["val_pred_fg_ratio_min"])), ("pred_fg_max", cast(float, summary["val_pred_fg_ratio_max"])), ("valid_low", cast(float, summary["val_valid_low_mean"])), ("valid_high", cast(float, summary["val_valid_high_mean"])), ("mean_o5", cast(float, summary["val_mean_o5_occ"])), ("mean_o10", cast(float, summary["val_mean_o10_occ"])), ("o5_low", cast(float, summary["val_o5_low_mean"])), ("o5_high", cast(float, summary["val_o5_high_mean"])), ("o10_low", cast(float, summary["val_o10_low_mean"])), ("o10_high", cast(float, summary["val_o10_high_mean"]))]))

        if val_seen_total:
            current_metric_name = "val_seen_loss"
            current_metric_value = cast(float, summary["val_seen_loss"])
        else:
            current_metric_name = "val_loss"
            current_metric_value = cast(float, summary["val_loss"])

        metrics = {
            "history": history,
            "category_to_id": category_to_id,
            "id_to_category": {index: category for category, index in category_to_id.items()},
            "num_categories": len(category_to_id),
            "train_categories": train_categories,
            "val_categories": val_categories,
            "unseen_val_categories": unseen_val_categories,
            "z_by_category": z_category_history,
            "category_to_num_modes": train_loader.dataset.cache_meta.get("category_to_num_modes", {}),
            "max_num_modes_per_category": int(planner_cf.get("max_num_modes_per_category", 16)),
            "descriptor_slices": train_loader.dataset.cache_meta.get("descriptor_slices", {}),
            "grammar_signature_dim": len(train_dataset[0]["grammar_signature"]),
            "adjacency_signature_dim": len(train_dataset[0]["adjacency_signature"]),
            "planner_config": planner_cf,
            "model_config": {
                "num_categories": len(category_to_id),
                "num_modes": int(planner_cf["num_modes"]),
                "coarse_size": int(data_cf["coarse_size"]),
                "num_classes": int(data_cf["num_classes"]),
                "category_embed_dim": int(planner_cf["category_embed_dim"]),
                "mode_embed_dim": int(planner_cf["mode_embed_dim"]),
                "hidden_dim": int(planner_cf["hidden_dim"]),
                "num_layers": int(planner_cf["num_layers"]),
                "coarse_size_10": 10,
                "grammar_signature_dim": len(train_dataset[0]["grammar_signature"]),
                "adjacency_signature_dim": len(train_dataset[0]["adjacency_signature"]),
                "max_num_modes_per_category": int(planner_cf.get("max_num_modes_per_category", 16)),
            },
            "config": config,
        }
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "config": config,
            "metrics": metrics,
            "category_to_id": category_to_id,
            "id_to_category": {index: category for category, index in category_to_id.items()},
            "num_categories": len(category_to_id),
            "train_categories": train_categories,
            "val_categories": val_categories,
            "unseen_val_categories": unseen_val_categories,
            "grammar_signature_dim": len(train_dataset[0]["grammar_signature"]),
            "adjacency_signature_dim": len(train_dataset[0]["adjacency_signature"]),
            "descriptor_slices": train_loader.dataset.cache_meta.get("descriptor_slices", {}),
            "planner_config": planner_cf,
            "model_config": metrics["model_config"],
            "best_metric_name": current_metric_name if current_metric_value < best_metric_value else best_metric_name,
            "best_metric_value": current_metric_value if current_metric_value < best_metric_value else best_metric_value,
        }
        last_path = output_dir / "checkpoint_last.pt"
        getattr(torch, "save")(checkpoint_payload, last_path)
        print(f"saved checkpoint_last: {last_path}")

        if current_metric_value < best_metric_value:
            best_metric_name = current_metric_name
            best_metric_value = current_metric_value
            checkpoint_payload["best_metric_name"] = best_metric_name
            checkpoint_payload["best_metric_value"] = best_metric_value
            best_path = output_dir / "checkpoint.pt"
            getattr(torch, "save")(checkpoint_payload, best_path)
            print(f"saved best checkpoint: {best_path}")
    metrics = {
        "history": history,
        "category_to_id": category_to_id,
        "id_to_category": {index: category for category, index in category_to_id.items()},
        "num_categories": len(category_to_id),
        "train_categories": train_categories,
        "val_categories": val_categories,
        "unseen_val_categories": unseen_val_categories,
        "z_by_category": z_category_history,
        "config": config,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
