from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import cast

from PIL import Image

from .dataset import PairedRefinerDataset
from .model import RefinerUNet


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        functional = importlib.import_module("torch.nn.functional")
    except ImportError as error:
        raise ImportError("PyTorch is required for refiner-v1 evaluation. Install with `pip install -e .[train]`.") from error
    return torch, functional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained refiner-v1 checkpoint on paired data.")
    parser.add_argument("--manifest", type=Path, required=True, help="Validation or test manifest")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained refiner checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for evaluation outputs")
    parser.add_argument("--device", type=str, default="cpu", help="Evaluation device, for example `cpu`, `cuda`, or `cuda:1`.")
    parser.add_argument("--num-samples", type=int, default=16, help="How many rows to sample from the manifest")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling rows")
    return parser


def _resolve_device(torch: object, device_name: str) -> object:
    device_cls = getattr(torch, "device")
    if device_name == "cpu":
        return device_cls("cpu")
    if not getattr(torch, "cuda").is_available():
        raise RuntimeError(f"Requested device {device_name!r}, but CUDA is not available in the current environment.")
    return device_cls(device_name)


def _save_tensor_image(torch: object, image_tensor: object, output_path: Path) -> None:
    image = image_tensor.detach().cpu().clamp(-1.0, 1.0)
    image = ((image + 1.0) * 127.5).round().to(dtype=getattr(torch, "uint8")).permute(1, 2, 0)
    Image.fromarray(image.numpy(), mode="RGB").save(output_path)


def _load_checkpoint(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    torch, _ = _require_torch()
    checkpoint = cast(dict[str, object], getattr(torch, "load")(path, map_location="cpu"))
    metrics = cast(dict[str, object], checkpoint.get("metrics", {}))
    return checkpoint, metrics


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch, functional = _require_torch()
    device = _resolve_device(torch, args.device)
    checkpoint, checkpoint_metrics = _load_checkpoint(args.checkpoint.resolve())
    image_size = tuple(int(value) for value in checkpoint_metrics.get("image_size", [160, 160]))

    dataset = PairedRefinerDataset(args.manifest, image_size=cast(tuple[int, int], image_size))
    sample_count = min(args.num_samples, len(dataset))
    rng = random.Random(args.seed)
    sample_indices = rng.sample(list(range(len(dataset))), sample_count)

    model = RefinerUNet()
    model.load_state_dict(cast(dict[str, object], checkpoint["model_state_dict"]))
    model.to(device)
    model.eval()

    l1_values: list[float] = []
    mse_values: list[float] = []
    index_rows: list[dict[str, object]] = []

    with getattr(torch, "no_grad")():
        for out_index, data_index in enumerate(sample_indices, start=1):
            item = dataset[data_index]
            source = cast(object, item["source"]).unsqueeze(0).to(device)
            target = cast(object, item["target"]).to(device)
            pred = model(source)[0]
            l1 = float(functional.l1_loss(pred, target).item())
            mse = float(functional.mse_loss(pred, target).item())
            l1_values.append(l1)
            mse_values.append(mse)

            sample_dir = args.output_dir / "samples" / str(item["sample_id"]).replace("/", "__")
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_tensor_image(torch, cast(object, item["source"]), sample_dir / "input.png")
            _save_tensor_image(torch, pred, sample_dir / "pred.png")
            _save_tensor_image(torch, cast(object, item["target"]), sample_dir / "target.png")
            shutil.copy2(str(item["input_path"]), sample_dir / "input_source.jpg")
            shutil.copy2(str(item["target_path"]), sample_dir / "target_source.jpg")
            meta = {
                "sample_id": item["sample_id"],
                "paired_rendering_id": item["paired_rendering_id"],
                "category": item["category"],
                "input_path": item["input_path"],
                "target_path": item["target_path"],
                "l1": l1,
                "mse": mse,
            }
            (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            index_rows.append(meta)
            print(f"[{out_index}/{sample_count}] {item['sample_id']} l1={l1:.6f} mse={mse:.6f}")

    result = {
        "manifest": str(args.manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "device": str(device),
        "image_size": list(image_size),
        "num_samples": sample_count,
        "seed": args.seed,
        "avg_l1": sum(l1_values) / max(1, len(l1_values)),
        "avg_mse": sum(mse_values) / max(1, len(mse_values)),
        "samples": index_rows,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
