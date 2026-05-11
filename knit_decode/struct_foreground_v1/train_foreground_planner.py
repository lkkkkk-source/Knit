from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from .dataset import build_dataloader, load_manifest
from .models.foreground_planner import ForegroundCanonicalPlanner
from .utils import IGNORE_INDEX, finish_progress, format_metric_line, load_config, print_progress


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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data_cf = config["data"]
    planner_cf = config["planner"]
    train_cf = config["train"]
    train_manifest = Path(data_cf["train_manifest"])
    val_manifest = Path(data_cf["val_manifest"])
    train_cache = Path(data_cf["cache_dir"]) / "foreground_cache_train.pt"
    val_cache = Path(data_cf["cache_dir"]) / "foreground_cache_val.pt"
    mapping = _build_category_mapping(train_manifest, val_manifest)
    category_to_id = cast(dict[str, int], mapping["category_to_id"])

    torch, optim = _require_torch()
    device = _resolve_device(torch, str(train_cf["device"]))
    train_loader, train_dataset = build_dataloader(train_manifest, train_cache, batch_size=int(train_cf["batch_size"]), shuffle=True, category_to_id=category_to_id, num_workers=int(train_cf["num_workers"]), pin_memory=bool(train_cf["pin_memory"]), persistent_workers=bool(train_cf["persistent_workers"]))
    val_loader, _ = build_dataloader(val_manifest, val_cache, batch_size=int(train_cf["batch_size"]), shuffle=False, category_to_id=category_to_id, num_workers=int(train_cf["num_workers"]), pin_memory=bool(train_cf["pin_memory"]), persistent_workers=bool(train_cf["persistent_workers"]))
    model = ForegroundCanonicalPlanner(
        num_categories=len(category_to_id),
        max_num_modes=int(planner_cf["num_modes_per_category"]),
        hidden_dim=int(planner_cf["hidden_dim"]),
        category_embed_dim=int(planner_cf["category_embed_dim"]),
        mode_embed_dim=int(planner_cf["mode_embed_dim"]),
        grammar_dim=len(train_dataset[0]["grammar_signature"]),
        adjacency_dim=len(train_dataset[0]["adjacency_signature"]),
        bbox_dim=len(train_dataset[0]["bbox_stats"]),
    )
    model.to(device)
    optimizer = getattr(optim, "AdamW")(model.parameters(), lr=float(train_cf["learning_rate"]), weight_decay=float(train_cf["weight_decay"]))
    functional = __import__("importlib").import_module("torch.nn.functional")
    history: list[dict[str, object]] = []
    output_dir = Path(train_cf["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(int(train_cf["epochs"])):
        print(f"\nepoch {epoch + 1}/{int(train_cf['epochs'])}")
        model.train()
        total_loss = 0.0
        total_fg_mask = 0.0
        total_fg_ce = 0.0
        batch_count = 0
        for batch in train_loader:
            outputs = model(
                batch["category_ids"].to(device),
                centroid_label_hist=batch["centroid_label_hist"].to(device),
                centroid_row_projection=batch["centroid_row_projection"].to(device),
                centroid_col_projection=batch["centroid_col_projection"].to(device),
                centroid_adjacency=batch["centroid_adjacency"].to(device),
                centroid_transition_stats=batch["centroid_transition_stats"].to(device),
                centroid_bbox_stats=batch["centroid_bbox_stats"].to(device),
                local_z=batch["local_z"].to(device),
                mode_mask=batch["mode_mask"].to(device),
            )
            fg_mask_target = batch["fg_mask20"].to(device)
            fg_mask_loss = functional.binary_cross_entropy_with_logits(outputs["fg_mask_logits"].squeeze(1), fg_mask_target)
            fg_target = batch["fg_y20"].to(device)
            fg_label_target = fg_target.clone()
            fg_label_target = getattr(torch, "where")(fg_label_target >= 1, fg_label_target - 1, fg_label_target)
            fg_ce = functional.cross_entropy(outputs["fg_label_logits"], fg_label_target, ignore_index=IGNORE_INDEX)
            bbox_loss = functional.l1_loss(outputs["bbox_pred"], batch["bbox_stats"].to(device))
            row_loss = functional.l1_loss(outputs["row_projection_pred"], batch["row_projection"].to(device))
            col_loss = functional.l1_loss(outputs["col_projection_pred"], batch["col_projection"].to(device))
            grammar_loss = functional.l1_loss(outputs["grammar_signature_pred"], batch["grammar_signature"].to(device))
            adj_loss = functional.l1_loss(outputs["adjacency_signature_pred"], batch["adjacency_signature"].to(device))
            z_loss = functional.cross_entropy(outputs["local_z_logits"].masked_fill(batch["mode_mask"].to(device).logical_not(), float("-inf")), batch["local_z"].to(device))
            loss = z_loss + fg_mask_loss + fg_ce + bbox_loss + row_loss + col_loss + grammar_loss + adj_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_fg_mask += float(fg_mask_loss.item())
            total_fg_ce += float(fg_ce.item())
            batch_count += 1
            print_progress("fg-train", batch_count, len(train_loader), f"loss={total_loss / batch_count:.4f} fg_mask={total_fg_mask / batch_count:.4f} fg_ce={total_fg_ce / batch_count:.4f}")
        finish_progress()
        summary = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, batch_count),
            "fg_mask_iou": 0.0,
            "fg_label_acc_on_fg": 0.0,
            "foreground_label_ce": total_fg_ce / max(1, batch_count),
            "empty_foreground_rate": 0.0,
            "full_foreground_rate": 0.0,
            "local_z_entropy": 0.0,
            "effective_local_modes": 0.0,
            "bbox_l1": 0.0,
            "adjacency_l1": 0.0,
            "grammar_l1": 0.0,
        }
        history.append(summary)
        print(format_metric_line("summary foreground:", [("train_loss", cast(float, summary["train_loss"])), ("fg_ce", cast(float, summary["foreground_label_ce"]))]))

    metrics = {
        "history": history,
        "category_to_id": category_to_id,
        "id_to_category": mapping["id_to_category"],
        "category_to_num_modes": train_dataset.cache_payload["category_to_num_modes"],
        "descriptor_slices": train_dataset.cache_payload["descriptor_slices"],
        "grammar_signature_dim": len(train_dataset[0]["grammar_signature"]),
        "adjacency_signature_dim": len(train_dataset[0]["adjacency_signature"]),
        "bbox_dim": len(train_dataset[0]["bbox_stats"]),
        "max_num_modes": int(planner_cf["num_modes_per_category"]),
        "config": config,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    getattr(torch, "save")({"model_state_dict": model.state_dict(), "metrics": metrics}, output_dir / "checkpoint.pt")
    print(f"saved checkpoint: {output_dir / 'checkpoint.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
