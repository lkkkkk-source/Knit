from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
import math
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .pixelcnn_dataset import (
    GridTokenMap,
    KnitGridDataset,
    build_knit_grid_dataloader,
    build_knit_grid_dataloader_from_dataset,
    split_dataset_indices,
    subset_knit_grid_dataset,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _external_pixelcnn_paths() -> tuple[Path, Path]:
    root = _repo_root() / "external" / "pixel_models"
    return root / "pixelcnn.py", root / "optim.py"


def _load_external_module(module_name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pixelcnn_module() -> Any:
    pixelcnn_path, _ = _external_pixelcnn_paths()
    return _load_external_module("external_pixelcnn", pixelcnn_path)


def load_optim_module() -> Any:
    _, optim_path = _external_pixelcnn_paths()
    return _load_external_module("external_pixelcnn_optim", optim_path)


def grid_vocab_n_bits(grid_vocab_size: int) -> int:
    if grid_vocab_size <= 1:
        return 1
    return max(1, math.ceil(math.log2(grid_vocab_size)))


def _pad_normalized_value(grid_vocab_size: int) -> float:
    return 0.0 if grid_vocab_size <= 1 else -1.0


def pixelcnn_grid_loss(logits: Tensor, targets: Tensor, grid_vocab_size: int, ignore_index: int = -100) -> Tensor:
    sliced_logits = logits[:, :grid_vocab_size, 0, :, :]
    squeezed_targets = targets[:, 0, :, :]
    return F.cross_entropy(sliced_logits, squeezed_targets, reduction="mean", ignore_index=ignore_index)


def sample_pixelcnn_grid(
    model: Any,
    n_samples: int,
    rows: int,
    columns: int,
    device: torch.device,
    grid_vocab_size: int,
    valid_class_ids: tuple[int, ...],
) -> Tensor:
    normalized_pad = _pad_normalized_value(grid_vocab_size)
    with torch.no_grad():
        inputs = torch.full((n_samples, 1, rows, columns), normalized_pad, dtype=torch.float32, device=device)
        generated = torch.full((n_samples, rows, columns), 0, dtype=torch.long, device=device)
        for y_pos in range(rows):
            for x_pos in range(columns):
                logits = model(inputs)
                cell_logits = logits[:, :grid_vocab_size, 0, y_pos, x_pos]
                invalid_mask = torch.ones(grid_vocab_size, dtype=torch.bool, device=device)
                invalid_mask[list(valid_class_ids)] = False
                masked_logits = cell_logits.masked_fill(invalid_mask.unsqueeze(0), float("-inf"))
                probs = F.softmax(masked_logits, dim=1)
                sample = torch.multinomial(probs, num_samples=1).squeeze(1)
                generated[:, y_pos, x_pos] = sample
                if grid_vocab_size > 1:
                    normalized = sample.float().div(grid_vocab_size - 1).mul(2.0).add(-1.0)
                else:
                    normalized = sample.float().mul(0.0)
                inputs[:, 0, y_pos, x_pos] = normalized
    return generated


def decode_contiguous_grid(grid: list[list[int]], token_map: GridTokenMap) -> list[list[int | None]]:
    decoded: list[list[int | None]] = []
    for row in grid:
        decoded.append([
            token_map.contiguous_to_raw[token] if 0 <= token < len(token_map.contiguous_to_raw) else None
            for token in row
        ])
    return decoded


@dataclass(frozen=True)
class PixelCnnBaselineConfig:
    export_root: str
    output_dir: str
    batch_size: int = 2
    val_fraction: float = 0.1
    n_epochs: int = 1
    learning_rate: float = 5e-4
    learning_rate_decay: float = 0.999995
    polyak: float = 0.9995
    n_channels: int = 16
    n_out_conv_channels: int = 32
    kernel_size: int = 3
    n_res_layers: int = 2
    num_workers: int = 0
    ignore_index: int = -100
    seed: int = 0
    cuda: int | None = None
    n_samples: int = 2
    train: bool = False
    evaluate: bool = False
    generate: bool = False
    restore_file: str | None = None


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _set_seed(config: PixelCnnBaselineConfig) -> None:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)


def _resolve_device(config: PixelCnnBaselineConfig) -> torch.device:
    if config.cuda is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{config.cuda}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model_and_optimizer(
    config: PixelCnnBaselineConfig,
    export_root: Path,
) -> tuple[Any, Any, Any, GridTokenMap, KnitGridDataset]:
    dataset = KnitGridDataset(export_root)
    token_map = dataset.token_map
    n_bits = grid_vocab_n_bits(token_map.vocab_size)
    pixelcnn_module = load_pixelcnn_module()
    optim_module = load_optim_module()
    device = _resolve_device(config)
    model = pixelcnn_module.PixelCNN(
        image_dims=(1, 1, 1),
        n_bits=n_bits,
        n_channels=config.n_channels,
        n_out_conv_channels=config.n_out_conv_channels,
        kernel_size=config.kernel_size,
        n_res_layers=config.n_res_layers,
        n_cond_classes=None,
        norm_layer=True,
    ).to(device)
    optimizer = optim_module.Adam(model.parameters(), lr=config.learning_rate, betas=(0.95, 0.9995), polyak=config.polyak)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config.learning_rate_decay)
    return model, optimizer, scheduler, token_map, dataset


def _build_split_dataloaders(
    config: PixelCnnBaselineConfig,
    export_root: Path,
    dataset: KnitGridDataset,
) -> tuple[Any, Any | None]:
    if config.train and config.evaluate:
        train_indices, val_indices = split_dataset_indices(len(dataset), config.val_fraction, config.seed)
        train_dataset = subset_knit_grid_dataset(dataset, train_indices)
        val_dataset = subset_knit_grid_dataset(dataset, val_indices)
        train_loader = build_knit_grid_dataloader_from_dataset(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            ignore_index=config.ignore_index,
        )
        val_loader = build_knit_grid_dataloader_from_dataset(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            ignore_index=config.ignore_index,
        )
        return train_loader, val_loader

    dataloader = build_knit_grid_dataloader(
        export_root,
        batch_size=config.batch_size,
        shuffle=config.train,
        num_workers=config.num_workers,
        ignore_index=config.ignore_index,
    )
    return dataloader, None


def _run_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    scheduler: Any | None,
    device: torch.device,
    grid_vocab_size: int,
    train: bool,
    ignore_index: int,
) -> float:
    total_loss = 0.0
    batches = 0
    if train:
        model.train()
        iterator = dataloader
    else:
        model.eval()
        iterator = dataloader

    with torch.no_grad() if not train else torch.enable_grad():
        for batch in iterator:
            input_grid: Tensor = batch["input_grid"].to(device)
            target_grid: Tensor = batch["target_grid"].to(device)
            logits: Tensor = model(input_grid)
            loss = pixelcnn_grid_loss(logits, target_grid, grid_vocab_size, ignore_index=ignore_index)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += float(loss.item())
            batches += 1
    return total_loss / max(1, batches)


def _swap_ema_if_available(optimizer: Any) -> bool:
    if hasattr(optimizer, "swap_ema"):
        optimizer.swap_ema()
        return True
    return False


def _save_checkpoint(path: Path, model: Any, optimizer: Any, scheduler: Any | None, config: PixelCnnBaselineConfig, grid_vocab_size: int) -> None:
    payload = {
        "config": asdict(config),
        "grid_vocab_size": grid_vocab_size,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_checkpoint_weights(path: Path, model: Any, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def run_pixelcnn_baseline(config: PixelCnnBaselineConfig) -> dict[str, object]:
    _set_seed(config)
    export_root = Path(config.export_root)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(output_dir / "config.json", asdict(config))

    device = _resolve_device(config)
    model, optimizer, scheduler, token_map, dataset = _build_model_and_optimizer(config, export_root)
    model.to(device)

    if config.restore_file:
        checkpoint = torch.load(config.restore_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    train_dataloader, val_dataloader = _build_split_dataloaders(config, export_root, dataset)

    summary: dict[str, object] = {
        "export_root": str(export_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "grid_vocab_size": token_map.vocab_size,
        "dataset_size": len(dataset),
        "train_dataset_size": len(train_dataloader.dataset),
        "val_dataset_size": len(val_dataloader.dataset) if val_dataloader is not None else 0,
        "val_fraction": config.val_fraction,
        "train_loss": None,
        "eval_loss": None,
        "epochs_completed": 0,
        "best_epoch": None,
        "best_metric_name": None,
        "best_metric_value": None,
        "best_checkpoint_path": None,
        "metrics_history_path": None,
        "generated_samples_path": None,
    }
    metrics_history: list[dict[str, object]] = []

    if config.train:
        train_loss: float | None = None
        best_metric_name = "eval_loss" if config.evaluate else "train_loss"
        best_metric_value = float("inf")
        best_epoch: int | None = None
        best_checkpoint_path = output_dir / "best_checkpoint.pt"
        latest_checkpoint_path = output_dir / "checkpoint.pt"
        for epoch_index in range(config.n_epochs):
            train_loss = _run_epoch(model, train_dataloader, optimizer, scheduler, device, token_map.vocab_size, True, config.ignore_index)
            epoch_eval_loss: float | None = None
            if config.evaluate:
                eval_dataloader = val_dataloader if val_dataloader is not None else train_dataloader
                used_ema = _swap_ema_if_available(optimizer)
                try:
                    epoch_eval_loss = _run_epoch(model, eval_dataloader, optimizer, None, device, token_map.vocab_size, False, config.ignore_index)
                finally:
                    if used_ema:
                        _swap_ema_if_available(optimizer)
            metric_value = epoch_eval_loss if epoch_eval_loss is not None else train_loss
            metrics_history.append(
                {
                    "epoch": epoch_index + 1,
                    "train_loss": train_loss,
                    "eval_loss": epoch_eval_loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "is_best": metric_value < best_metric_value,
                }
            )
            _save_checkpoint(latest_checkpoint_path, model, optimizer, scheduler, config, token_map.vocab_size)
            if metric_value < best_metric_value:
                used_ema = False
                if config.evaluate and epoch_eval_loss is not None:
                    used_ema = _swap_ema_if_available(optimizer)
                try:
                    _save_checkpoint(best_checkpoint_path, model, optimizer, scheduler, config, token_map.vocab_size)
                finally:
                    if used_ema:
                        _swap_ema_if_available(optimizer)
                best_metric_value = metric_value
                best_epoch = epoch_index + 1
        summary["train_loss"] = train_loss
        summary["epochs_completed"] = config.n_epochs
        summary["best_epoch"] = best_epoch
        summary["best_metric_name"] = best_metric_name
        summary["best_metric_value"] = best_metric_value if best_epoch is not None else None
        summary["best_checkpoint_path"] = str(best_checkpoint_path) if best_epoch is not None else None
        history_path = output_dir / "metrics_history.json"
        _save_json(history_path, metrics_history)
        summary["metrics_history_path"] = str(history_path)

    if config.evaluate:
        eval_dataloader = val_dataloader if val_dataloader is not None else train_dataloader
        if config.train and summary["best_checkpoint_path"]:
            _load_checkpoint_weights(Path(str(summary["best_checkpoint_path"])), model, device)
            summary["eval_loss"] = _run_epoch(model, eval_dataloader, optimizer, None, device, token_map.vocab_size, False, config.ignore_index)
        else:
            used_ema = _swap_ema_if_available(optimizer)
            try:
                summary["eval_loss"] = _run_epoch(model, eval_dataloader, optimizer, None, device, token_map.vocab_size, False, config.ignore_index)
            finally:
                if used_ema:
                    _swap_ema_if_available(optimizer)

    if config.generate:
        first_item = dataset[0]
        model.eval()
        if config.train and summary["best_checkpoint_path"]:
            _load_checkpoint_weights(Path(str(summary["best_checkpoint_path"])), model, device)
        generated = sample_pixelcnn_grid(
            model,
            n_samples=config.n_samples,
            rows=first_item["rows"],
            columns=first_item["columns"],
            device=device,
            grid_vocab_size=token_map.vocab_size,
            valid_class_ids=token_map.action_class_ids,
        )
        generated_list = generated.detach().cpu().tolist()
        generated_raw = [decode_contiguous_grid(sample, token_map) for sample in generated_list]
        generated_path = output_dir / "generated_grids.json"
        _save_json(
            generated_path,
            {
                "rows": first_item["rows"],
                "columns": first_item["columns"],
                "contiguous_samples": generated_list,
                "raw_action_samples": generated_raw,
            },
        )
        summary["generated_samples_path"] = str(generated_path)

    _save_json(output_dir / "metrics.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a PixelCNN baseline on exported knit action grids")
    _ = parser.add_argument("--export-root", type=Path, required=True, help="AR export root containing manifest/vocab/grid files")
    _ = parser.add_argument("--output-dir", type=Path, required=True, help="Directory for baseline artifacts")
    _ = parser.add_argument("--batch-size", type=int, default=2)
    _ = parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation split fraction used when both --train and --evaluate are enabled.")
    _ = parser.add_argument("--n-epochs", type=int, default=10)
    _ = parser.add_argument("--learning-rate", type=float, default=5e-4)
    _ = parser.add_argument("--learning-rate-decay", type=float, default=0.999995)
    _ = parser.add_argument("--polyak", type=float, default=0.9995)
    _ = parser.add_argument("--n-channels", type=int, default=16)
    _ = parser.add_argument("--n-out-conv-channels", type=int, default=32)
    _ = parser.add_argument("--kernel-size", type=int, default=3)
    _ = parser.add_argument("--n-res-layers", type=int, default=2)
    _ = parser.add_argument("--num-workers", type=int, default=0)
    _ = parser.add_argument("--ignore-index", type=int, default=-100)
    _ = parser.add_argument("--seed", type=int, default=0)
    _ = parser.add_argument("--cuda", type=int, default=None)
    _ = parser.add_argument("--n-samples", type=int, default=2)
    _ = parser.add_argument("--restore-file", type=Path, default=None)
    _ = parser.add_argument("--train", action="store_true")
    _ = parser.add_argument("--evaluate", action="store_true")
    _ = parser.add_argument("--generate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = PixelCnnBaselineConfig(
        export_root=str(args.export_root),
        output_dir=str(args.output_dir),
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        learning_rate_decay=args.learning_rate_decay,
        polyak=args.polyak,
        n_channels=args.n_channels,
        n_out_conv_channels=args.n_out_conv_channels,
        kernel_size=args.kernel_size,
        n_res_layers=args.n_res_layers,
        num_workers=args.num_workers,
        ignore_index=args.ignore_index,
        seed=args.seed,
        cuda=args.cuda,
        n_samples=args.n_samples,
        train=args.train,
        evaluate=args.evaluate,
        generate=args.generate,
        restore_file=str(args.restore_file) if args.restore_file else None,
    )
    summary = run_pixelcnn_baseline(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
