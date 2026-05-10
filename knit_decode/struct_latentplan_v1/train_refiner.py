from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from .dataset import build_dataloader
from .models.planner import LatentPlanner
from .models.refiner import PlanConditionedMaskRefiner
from .utils import compute_plan_statistics, ensure_palette_path, finish_progress, format_metric_line, load_config, print_progress


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        optim = importlib.import_module("torch.optim")
    except ImportError as error:
        raise ImportError("PyTorch is required for refiner training. Install with `pip install -e .[train]`.") from error
    return torch, optim


def _resolve_device(torch: object, device_name: str) -> object:
    device_cls = getattr(torch, "device")
    if device_name == "cpu":
        return device_cls("cpu")
    if not getattr(torch, "cuda").is_available():
        raise RuntimeError(f"Requested device {device_name!r}, but CUDA is not available.")
    return device_cls(device_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train plan-conditioned MaskGIT refiner.")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def _sample_mask(torch: object, targets: object, min_masking_rate: float) -> object:
    batch_size, height, width = targets.shape
    seq_len = height * width
    flat_mask = getattr(torch, "zeros")((batch_size, seq_len), dtype=getattr(torch, "bool"), device=targets.device)
    ratios = getattr(torch, "rand")((batch_size,), device=targets.device)
    for batch_index in range(batch_size):
        mask_count = max(1, min(seq_len, int(round(seq_len * max(float(min_masking_rate), float(ratios[batch_index].item()))))))
        indices = getattr(torch, "randperm")(seq_len, device=targets.device)[:mask_count]
        flat_mask[batch_index, indices] = True
    return flat_mask.reshape(batch_size, height, width)


def _build_masked_tokens(torch: object, targets: object, mask: object, mask_token_id: int) -> object:
    mask_token = getattr(torch, "full_like")(targets, mask_token_id)
    return getattr(torch, "where")(mask, mask_token, targets)


def _coarse_plan_adherence_loss(logits: object, o5: object, c5: object, background_class_id: int) -> object:
    functional = __import__("importlib").import_module("torch.nn.functional")
    probs = functional.softmax(logits, dim=1)
    fg_prob = 1.0 - probs[:, background_class_id : background_class_id + 1]
    fg_coarse = functional.avg_pool2d(fg_prob, kernel_size=4, stride=4).squeeze(1)
    o5_loss = functional.binary_cross_entropy(fg_coarse.clamp(1e-6, 1.0 - 1e-6), o5)
    coarse_logits = functional.avg_pool2d(logits, kernel_size=4, stride=4)
    c5_loss = functional.cross_entropy(coarse_logits, c5)
    return o5_loss + c5_loss


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data_cf = config["data"]
    planner_cf = config["planner"]
    refiner_cf = config["refiner"]
    train_cf = config["train_refiner"]
    loss_cf = config["loss"]
    planner_run_dir = Path(config["train_planner"]["output_dir"])
    planner_checkpoint = planner_run_dir / "checkpoint.pt"
    manifest_path = Path(data_cf["train_manifest"])
    val_manifest_path = Path(data_cf["val_manifest"])
    plan_cache_path = Path(data_cf["plan_cache_dir"]) / f"{manifest_path.stem}.pt"
    val_plan_cache_path = Path(data_cf["plan_cache_dir"]) / f"{val_manifest_path.stem}.pt"
    output_dir = Path(train_cf["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    torch, optim = _require_torch()
    device = _resolve_device(torch, str(train_cf["device"]))
    palette_path = ensure_palette_path(manifest_path, data_cf.get("palette_path"))
    train_loader, train_dataset = build_dataloader(
        manifest_path,
        palette_path=palette_path,
        plan_cache_path=plan_cache_path,
        batch_size=int(train_cf["batch_size"]),
        shuffle=True,
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
        category_to_id=train_dataset.category_to_id,
        num_workers=int(train_cf["num_workers"]),
        pin_memory=bool(train_cf["pin_memory"]),
        persistent_workers=bool(train_cf["persistent_workers"]),
    )
    planner_model = LatentPlanner(
        num_categories=len(train_dataset.category_to_id),
        num_modes=int(planner_cf["num_modes"]),
        coarse_size=int(data_cf["coarse_size"]),
        num_classes=int(data_cf["num_classes"]),
        category_embed_dim=int(planner_cf["category_embed_dim"]),
        mode_embed_dim=int(planner_cf["mode_embed_dim"]),
        hidden_dim=int(planner_cf["hidden_dim"]),
        num_layers=int(planner_cf["num_layers"]),
    )
    planner_payload = cast(dict[str, object], getattr(torch, "load")(planner_checkpoint, map_location="cpu"))
    planner_model.load_state_dict(planner_payload["model_state_dict"])
    planner_model.to(device)
    planner_model.eval()
    for parameter in planner_model.parameters():
        parameter.requires_grad_(False)

    model = PlanConditionedMaskRefiner(
        num_categories=len(train_dataset.category_to_id),
        num_modes=int(planner_cf["num_modes"]),
        num_classes=int(data_cf["num_classes"]),
        grid_size=int(data_cf["label_size"]),
        hidden_dim=int(refiner_cf["hidden_dim"]),
        num_layers=int(refiner_cf["num_layers"]),
        num_heads=int(refiner_cf["num_heads"]),
        use_2d_rope=bool(refiner_cf["use_2d_rope"]),
    )
    model.to(device)
    optimizer = getattr(optim, "AdamW")(model.parameters(), lr=float(train_cf["learning_rate"]), weight_decay=float(train_cf["weight_decay"]))
    functional = __import__("importlib").import_module("torch.nn.functional")
    history: list[dict[str, object]] = []
    background_class_id = int(data_cf["background_class_id"])

    for epoch in range(int(train_cf["epochs"])):
        print(f"\nepoch {epoch + 1}/{int(train_cf['epochs'])}")
        model.train()
        total_loss = 0.0
        total_token = 0.0
        total_occ = 0.0
        total_count = 0.0
        total_plan = 0.0
        batch_count = 0
        for batch in train_loader:
            category_ids = batch["category_ids"].to(device)
            y20 = batch["y20"].to(device)
            z = batch["z"].to(device)
            c5 = batch["c5"].to(device)
            o5 = batch["o5"].to(device)
            r17 = batch["r17"].to(device)
            fg_ratio = batch["fg_ratio"].to(device)
            mask20 = _sample_mask(torch, y20, float(train_cf["min_masking_rate"]))
            masked_tokens = _build_masked_tokens(torch, y20, mask20, int(model.mask_token_id))
            outputs = model(masked_tokens, category_ids, z, c5, o5, r17, fg_ratio)
            merged_logits = outputs["merged_logits"]
            token_loss_map = functional.cross_entropy(merged_logits, y20, reduction="none")
            token_loss = (token_loss_map * mask20.to(dtype=token_loss_map.dtype)).sum() / mask20.sum().clamp_min(1.0)
            occupancy_target = (y20 != background_class_id).to(dtype=getattr(torch, "long"))
            occupancy_loss = functional.cross_entropy(outputs["occupancy_logits"], occupancy_target)
            probs = functional.softmax(merged_logits, dim=1)
            pred_counts = probs.sum(dim=(2, 3))
            count_loss = functional.l1_loss(pred_counts / pred_counts.sum(dim=1, keepdim=True).clamp_min(1e-6), r17)
            plan_loss = _coarse_plan_adherence_loss(merged_logits, o5, c5, background_class_id=background_class_id)
            loss = (
                float(loss_cf["token_ce"]) * token_loss
                + float(loss_cf["occupancy"]) * occupancy_loss
                + float(loss_cf["count"]) * count_loss
                + float(loss_cf["coarse_plan"]) * plan_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_token += float(token_loss.item())
            total_occ += float(occupancy_loss.item())
            total_count += float(count_loss.item())
            total_plan += float(plan_loss.item())
            batch_count += 1
            print_progress("refiner-train", batch_count, len(train_loader), f"loss={total_loss / batch_count:.4f} token={total_token / batch_count:.4f} occ={total_occ / batch_count:.4f} plan={total_plan / batch_count:.4f}")
        finish_progress()

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with getattr(torch, "no_grad")():
            for batch in val_loader:
                category_ids = batch["category_ids"].to(device)
                y20 = batch["y20"].to(device)
                z = batch["z"].to(device)
                c5 = batch["c5"].to(device)
                o5 = batch["o5"].to(device)
                r17 = batch["r17"].to(device)
                fg_ratio = batch["fg_ratio"].to(device)
                mask20 = _sample_mask(torch, y20, float(train_cf["min_masking_rate"]))
                masked_tokens = _build_masked_tokens(torch, y20, mask20, int(model.mask_token_id))
                outputs = model(masked_tokens, category_ids, z, c5, o5, r17, fg_ratio)
                merged_logits = outputs["merged_logits"]
                token_loss_map = functional.cross_entropy(merged_logits, y20, reduction="none")
                token_loss = (token_loss_map * mask20.to(dtype=token_loss_map.dtype)).sum() / mask20.sum().clamp_min(1.0)
                occupancy_target = (y20 != background_class_id).to(dtype=getattr(torch, "long"))
                occupancy_loss = functional.cross_entropy(outputs["occupancy_logits"], occupancy_target)
                probs = functional.softmax(merged_logits, dim=1)
                pred_counts = probs.sum(dim=(2, 3))
                count_loss = functional.l1_loss(pred_counts / pred_counts.sum(dim=1, keepdim=True).clamp_min(1e-6), r17)
                plan_loss = _coarse_plan_adherence_loss(merged_logits, o5, c5, background_class_id=background_class_id)
                val_loss += float((float(loss_cf["token_ce"]) * token_loss + float(loss_cf["occupancy"]) * occupancy_loss + float(loss_cf["count"]) * count_loss + float(loss_cf["coarse_plan"]) * plan_loss).item())
                val_batches += 1
        summary = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, batch_count),
            "train_token": total_token / max(1, batch_count),
            "train_occ": total_occ / max(1, batch_count),
            "train_count": total_count / max(1, batch_count),
            "train_plan": total_plan / max(1, batch_count),
            "val_loss": val_loss / max(1, val_batches),
        }
        history.append(summary)
        print(format_metric_line("summary refiner:", [("train_loss", cast(float, summary["train_loss"])), ("token", cast(float, summary["train_token"])), ("occ", cast(float, summary["train_occ"])), ("count", cast(float, summary["train_count"])), ("plan", cast(float, summary["train_plan"])), ("val_loss", cast(float, summary["val_loss"]))]))

    metrics = {
        "history": history,
        "category_to_id": train_dataset.category_to_id,
        "config": config,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    getattr(torch, "save")({"model_state_dict": model.state_dict(), "metrics": metrics}, output_dir / "checkpoint.pt")
    print(f"saved checkpoint: {output_dir / 'checkpoint.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
