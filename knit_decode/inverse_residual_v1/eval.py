from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from PIL import Image

from .dataset import build_dataloader
from .model import ResidualRefiner


def _require_torch() -> object:
    import importlib

    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required for inverse_residual_v1 eval. Install with `pip install -e .[train]`.") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate residual-space inverse joint model.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--palette", type=Path, default=Path("parser_t_inverse/palette_mapping.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--use-best-crop", action="store_true")
    parser.add_argument("--use-transfer", action="store_true")
    parser.add_argument("--transfer-root", type=Path, default=Path("dataset2/transfer/Cable1_019_0_19/gray"))
    parser.add_argument("--mean-value", type=float, default=0.5)
    parser.add_argument("--num-vis", type=int, default=16)
    return parser


def _resolve_device(torch: object, device_name: str) -> object:
    device_cls = getattr(torch, "device")
    if device_name == "cpu":
        return device_cls("cpu")
    if not getattr(torch, "cuda").is_available():
        raise RuntimeError(f"Requested device {device_name!r}, but CUDA is not available.")
    return device_cls(device_name)


def _save_gray(tensor: object, output_path: Path, residual: bool = False, mean_value: float = 0.5) -> None:
    torch = _require_torch()
    image = tensor.detach().cpu()
    if residual:
        image = image + mean_value
    image = image.clamp(0.0, 1.0)
    image = (image * 255.0).round().to(dtype=getattr(torch, "uint8"))
    output = Image.new("L", (int(image.shape[-1]), int(image.shape[-2])))
    output.putdata(image.reshape(-1).tolist())
    output.save(output_path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch = _require_torch()
    checkpoint = cast(dict[str, object], getattr(torch, "load")(args.checkpoint.resolve(), map_location="cpu"))
    metrics = cast(dict[str, object], checkpoint.get("metrics", {}))
    image_size = tuple(int(value) for value in metrics.get("image_size", [160, 160]))
    device = _resolve_device(torch, args.device)

    dataloader, _ = build_dataloader(
        args.manifest,
        palette_path=args.palette,
        batch_size=args.batch_size,
        shuffle=False,
        image_size=cast(tuple[int, int], image_size),
        transfer_root=args.transfer_root if args.use_transfer else None,
        use_best_crop=args.use_best_crop,
        augment_scale=0.0,
        mean_value=args.mean_value,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )

    refiner = ResidualRefiner()
    state_dict = checkpoint.get("refiner_state_dict", checkpoint.get("generator_state_dict"))
    if state_dict is None:
        raise ValueError("Checkpoint missing refiner_state_dict")
    refiner.load_state_dict(cast(dict[str, object], state_dict))
    refiner.to(device)
    refiner.eval()

    total_l1 = 0.0
    vis_written = 0
    with getattr(torch, "no_grad")():
        for batch in dataloader:
            source_res = batch["source_residual"].to(device)
            target_res = batch["target_residual"].to(device)
            pred_res = refiner(source_res)
            total_l1 += float((pred_res - target_res).abs().mean().item())
            if vis_written < args.num_vis:
                count = min(args.num_vis - vis_written, pred_res.shape[0])
                for sample_index in range(count):
                    sample_id = str(batch["sample_ids"][sample_index]).replace("/", "__")
                    sample_dir = args.output_dir / "samples" / sample_id
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    _save_gray(batch["source_gray"][sample_index], sample_dir / "input.png", residual=False, mean_value=args.mean_value)
                    _save_gray(pred_res[sample_index], sample_dir / "pred.png", residual=True, mean_value=args.mean_value)
                    _save_gray(batch["target_gray"][sample_index], sample_dir / "target.png", residual=False, mean_value=args.mean_value)
                    vis_written += 1

    result = {
        "manifest": str(args.manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "device": str(device),
        "image_size": list(image_size),
        "avg_residual_l1": total_l1 / max(1, len(dataloader)),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
