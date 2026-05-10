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
    )
    model.to(device)
    optimizer = getattr(optim, "AdamW")(model.parameters(), lr=float(train_cf["learning_rate"]), weight_decay=float(train_cf["weight_decay"]))
    functional = __import__("importlib").import_module("torch.nn.functional")
    history: list[dict[str, object]] = []
    z_category_history: dict[str, dict[str, object]] = {}
    print(format_metric_line("categories:", [("num_categories", len(category_to_id)), ("train_categories", len(train_categories)), ("val_categories", len(val_categories)), ("unseen_val_categories", unseen_val_categories)]))
    for epoch in range(int(train_cf["epochs"])):
        print(f"\nepoch {epoch + 1}/{int(train_cf['epochs'])}")
        model.train()
        total_loss = 0.0
        total_z = 0.0
        total_c5 = 0.0
        total_o5 = 0.0
        total_r17 = 0.0
        total_fg = 0.0
        batch_count = 0
        train_diag: dict[str, object] = {}
        for batch in train_loader:
            category_ids = batch["category_ids"].to(device)
            z = batch["z"].to(device)
            c5 = batch["c5"].to(device)
            o5 = batch["o5"].to(device)
            r17 = batch["r17"].to(device)
            fg_ratio = batch["fg_ratio"].to(device)
            outputs = model(category_ids, z_ids=z, sample_mode="teacher")
            losses = _batch_losses(functional, outputs, z, c5, o5, r17, fg_ratio)
            z_loss = losses["z"].mean()
            c5_loss = losses["c5"].mean()
            o5_loss = losses["o5"].mean()
            r17_loss = losses["count"].mean()
            fg_loss = losses["fg"].mean()
            loss = losses["total"].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            probs = functional.softmax(outputs["z_logits"], dim=-1)
            sampled_z = getattr(torch, "distributions").Categorical(probs=probs).sample()
            _diag_update(train_diag, probs, sampled_z, batch["categories"])
            total_loss += float(loss.item())
            total_z += float(z_loss.item())
            total_c5 += float(c5_loss.item())
            total_o5 += float(o5_loss.item())
            total_r17 += float(r17_loss.item())
            total_fg += float(fg_loss.item())
            batch_count += 1
            print_progress("planner-train", batch_count, len(train_loader), f"loss={total_loss / batch_count:.4f} z={total_z / batch_count:.4f} c5={total_c5 / batch_count:.4f} o5={total_o5 / batch_count:.4f}")
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
        val_diag: dict[str, object] = {}
        train_category_set = set(train_categories)
        with getattr(torch, "no_grad")():
            for batch in val_loader:
                category_ids = batch["category_ids"].to(device)
                z = batch["z"].to(device)
                c5 = batch["c5"].to(device)
                o5 = batch["o5"].to(device)
                r17 = batch["r17"].to(device)
                fg_ratio = batch["fg_ratio"].to(device)
                outputs = model(category_ids, z_ids=z, sample_mode="teacher")
                losses = _batch_losses(functional, outputs, z, c5, o5, r17, fg_ratio)
                val_loss += float(losses["total"].mean().item())
                val_batches += 1
                probs = functional.softmax(outputs["z_logits"], dim=-1)
                sampled_z = getattr(torch, "distributions").Categorical(probs=probs).sample()
                _diag_update(val_diag, probs, sampled_z, batch["categories"])
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
        }
        summary.update({f"train_{key}": value for key, value in _diag_finalize(train_diag, int(planner_cf["num_modes"])).items() if key != "z_by_category"})
        summary.update({f"val_{key}": value for key, value in _diag_finalize(val_diag, int(planner_cf["num_modes"])).items() if key != "z_by_category"})
        z_category_history[f"epoch_{epoch + 1:03d}"] = {
            "train": _diag_finalize(train_diag, int(planner_cf["num_modes"]))["z_by_category"],
            "val": _diag_finalize(val_diag, int(planner_cf["num_modes"]))["z_by_category"],
        }
        history.append(summary)
        print(format_metric_line("summary planner:", [("train_loss", cast(float, summary["train_loss"])), ("train_z", cast(float, summary["train_z"])), ("train_c5", cast(float, summary["train_c5"])), ("train_o5", cast(float, summary["train_o5"])), ("val_loss", cast(float, summary["val_loss"]))]))
        print(format_metric_line("summary seen  :", [("val_seen_loss", cast(float, summary["val_seen_loss"])), ("val_seen_z", cast(float, summary["val_seen_z"])), ("val_seen_c5", cast(float, summary["val_seen_c5"])), ("val_seen_o5", cast(float, summary["val_seen_o5"])), ("val_seen_count", cast(float, summary["val_seen_count"]))]))
        print(format_metric_line("summary unseen:", [("val_unseen_loss", cast(float, summary["val_unseen_loss"])), ("val_unseen_z", cast(float, summary["val_unseen_z"])), ("val_unseen_c5", cast(float, summary["val_unseen_c5"])), ("val_unseen_o5", cast(float, summary["val_unseen_o5"])), ("val_unseen_count", cast(float, summary["val_unseen_count"]))]))
        print(format_metric_line("zdiag train  :", [("entropy", cast(float, summary["train_z_entropy"])), ("top1", cast(float, summary["train_z_top1_prob"])), ("top5", cast(float, summary["train_z_top5_prob_sum"])), ("eff_modes", cast(float, summary["train_effective_num_modes"])), ("uniq_z", int(summary["train_sampled_unique_z_count"])), ("uniq_ratio", cast(float, summary["train_sampled_unique_z_ratio"]))]))
        print(format_metric_line("zdiag val    :", [("entropy", cast(float, summary["val_z_entropy"])), ("top1", cast(float, summary["val_z_top1_prob"])), ("top5", cast(float, summary["val_z_top5_prob_sum"])), ("eff_modes", cast(float, summary["val_effective_num_modes"])), ("uniq_z", int(summary["val_sampled_unique_z_count"])), ("uniq_ratio", cast(float, summary["val_sampled_unique_z_ratio"]))]))

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
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    getattr(torch, "save")({"model_state_dict": model.state_dict(), "metrics": metrics}, output_dir / "checkpoint.pt")
    print(f"saved checkpoint: {output_dir / 'checkpoint.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
