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
        for batch in train_loader:
            category_ids = batch["category_ids"].to(device)
            z = batch["z"].to(device)
            c5 = batch["c5"].to(device)
            o5 = batch["o5"].to(device)
            r17 = batch["r17"].to(device)
            fg_ratio = batch["fg_ratio"].to(device)
            outputs = model(category_ids, z_ids=z, sample_mode="teacher")
            z_loss = functional.cross_entropy(outputs["z_logits"], z)
            c5_loss = functional.cross_entropy(outputs["c5_logits"], c5)
            o5_loss = functional.binary_cross_entropy_with_logits(outputs["o5_logits"].squeeze(1), o5)
            r17_loss = functional.l1_loss(outputs["r17_pred"], r17)
            fg_loss = functional.l1_loss(outputs["fg_ratio_pred"], fg_ratio)
            loss = z_loss + c5_loss + o5_loss + 0.1 * r17_loss + 0.1 * fg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
        with getattr(torch, "no_grad")():
            for batch in val_loader:
                category_ids = batch["category_ids"].to(device)
                z = batch["z"].to(device)
                c5 = batch["c5"].to(device)
                o5 = batch["o5"].to(device)
                r17 = batch["r17"].to(device)
                fg_ratio = batch["fg_ratio"].to(device)
                outputs = model(category_ids, z_ids=z, sample_mode="teacher")
                loss = (
                    functional.cross_entropy(outputs["z_logits"], z)
                    + functional.cross_entropy(outputs["c5_logits"], c5)
                    + functional.binary_cross_entropy_with_logits(outputs["o5_logits"].squeeze(1), o5)
                    + 0.1 * functional.l1_loss(outputs["r17_pred"], r17)
                    + 0.1 * functional.l1_loss(outputs["fg_ratio_pred"], fg_ratio)
                )
                val_loss += float(loss.item())
                val_batches += 1
        summary = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, batch_count),
            "train_z": total_z / max(1, batch_count),
            "train_c5": total_c5 / max(1, batch_count),
            "train_o5": total_o5 / max(1, batch_count),
            "val_loss": val_loss / max(1, val_batches),
        }
        history.append(summary)
        print(format_metric_line("summary planner:", [("train_loss", cast(float, summary["train_loss"])), ("train_z", cast(float, summary["train_z"])), ("train_c5", cast(float, summary["train_c5"])), ("train_o5", cast(float, summary["train_o5"])), ("val_loss", cast(float, summary["val_loss"]))]))

    metrics = {
        "history": history,
        "category_to_id": category_to_id,
        "id_to_category": {index: category for category, index in category_to_id.items()},
        "num_categories": len(category_to_id),
        "train_categories": train_categories,
        "val_categories": val_categories,
        "unseen_val_categories": unseen_val_categories,
        "config": config,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    getattr(torch, "save")({"model_state_dict": model.state_dict(), "metrics": metrics}, output_dir / "checkpoint.pt")
    print(f"saved checkpoint: {output_dir / 'checkpoint.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
