from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import cast

from PIL import Image

from .dataset import load_render_manifest, load_rgb_image, resize_image
from .model import CategoryConditionalUNet


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        functional = importlib.import_module("torch.nn.functional")
    except ImportError as error:
        raise ImportError("PyTorch is required for render-v1 evaluation. Install with `pip install -e .[train]`.") from error
    return torch, functional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample and evaluate a render-v1 checkpoint on a manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Validation or test manifest")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained render-v1 checkpoint.pt")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for evaluation outputs")
    parser.add_argument("--device", type=str, default="cpu", help="Evaluation device, for example `cpu`, `cuda`, or `cuda:1`.")
    parser.add_argument("--num-samples", type=int, default=16, help="How many rows to sample from the manifest")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling rows and noise")
    parser.add_argument("--sampling-steps", type=int, default=None, help="Override reverse diffusion steps; defaults to training steps")
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
    image = ((image + 1.0) * 127.5).round().to(dtype=getattr(torch, "uint8"))
    image = image.permute(1, 2, 0)
    output = Image.fromarray(image.numpy(), mode="RGB")
    output.save(output_path)


def _infer_root(manifest_path: Path) -> Path:
    direct_root = manifest_path.parent
    rows = load_render_manifest(manifest_path)
    if not rows:
        return direct_root
    direct_candidate = direct_root / rows[0]["image_path"]
    if direct_candidate.exists():
        return direct_root
    return manifest_path.parent.parent


def _load_checkpoint(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    torch, _ = _require_torch()
    checkpoint = cast(dict[str, object], getattr(torch, "load")(path, map_location="cpu"))
    metrics = cast(dict[str, object], checkpoint.get("metrics", {}))
    return checkpoint, metrics


def _ddpm_sample(model: object, category_ids: object, num_steps: int, image_size: tuple[int, int], device: object) -> object:
    torch, _ = _require_torch()
    betas = getattr(torch, "linspace")(1e-4, 2e-2, num_steps, dtype=getattr(torch, "float32"), device=device)
    alphas = 1.0 - betas
    alpha_bars = getattr(torch, "cumprod")(alphas, dim=0)
    batch_size = int(category_ids.shape[0])
    x = getattr(torch, "randn")((batch_size, 3, image_size[1], image_size[0]), device=device)
    for step in reversed(range(num_steps)):
        t = getattr(torch, "full")((batch_size,), step, device=device, dtype=getattr(torch, "long"))
        pred_noise = model(x, t, category_ids)
        alpha = alphas[step]
        alpha_bar = alpha_bars[step]
        beta = betas[step]
        x = (x - ((1 - alpha) / getattr(torch, "sqrt")(1 - alpha_bar)) * pred_noise) / getattr(torch, "sqrt")(alpha)
        if step > 0:
            noise = getattr(torch, "randn_like")(x)
            x = x + getattr(torch, "sqrt")(beta) * noise
    return x.clamp(-1.0, 1.0)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch, functional = _require_torch()
    device = _resolve_device(torch, args.device)
    if str(device).startswith("cuda") and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    checkpoint, checkpoint_metrics = _load_checkpoint(args.checkpoint.resolve())
    image_size = tuple(int(value) for value in checkpoint_metrics.get("image_size", [160, 160]))
    category_to_id = cast(dict[str, int], checkpoint_metrics.get("category_to_id", {}))
    if not category_to_id:
        raise ValueError("Checkpoint metrics do not contain category_to_id.")
    num_steps = int(args.sampling_steps or checkpoint_metrics.get("num_diffusion_steps", 1000))

    model = CategoryConditionalUNet(num_categories=len(category_to_id))
    model.load_state_dict(cast(dict[str, object], checkpoint["model_state_dict"]))
    model.to(device)
    model.eval()

    rows = load_render_manifest(args.manifest)
    if not rows:
        raise ValueError(f"No rows found in manifest: {args.manifest}")
    rng = random.Random(args.seed)
    sample_count = min(args.num_samples, len(rows))
    sampled_rows = rng.sample(rows, sample_count)
    manifest_root = _infer_root(args.manifest)

    mse_values: list[float] = []
    l1_values: list[float] = []
    psnr_values: list[float] = []
    index_rows: list[dict[str, object]] = []

    with getattr(torch, "no_grad")():
        for index, row in enumerate(sampled_rows, start=1):
            category = row["category"]
            category_id = category_to_id.get(category)
            if category_id is None:
                raise KeyError(f"Category {category!r} missing from checkpoint category_to_id.")
            category_tensor = getattr(torch, "tensor")([category_id], device=device, dtype=getattr(torch, "long"))
            generated = _ddpm_sample(model, category_tensor, num_steps=num_steps, image_size=cast(tuple[int, int], image_size), device=device)[0]

            image_path = (manifest_root / row["image_path"]).resolve()
            gt_image = resize_image(load_rgb_image(image_path), cast(tuple[int, int], image_size))
            gt_tensor = getattr(torch, "tensor")(list(gt_image.getdata()), dtype=getattr(torch, "float32")).reshape(gt_image.height, gt_image.width, 3)
            gt_tensor = gt_tensor.permute(2, 0, 1) / 127.5 - 1.0

            mse = float(functional.mse_loss(generated, gt_tensor.to(device)).item())
            l1 = float(functional.l1_loss(generated, gt_tensor.to(device)).item())
            psnr = 99.0 if mse <= 0 else float(10.0 * math.log10(4.0 / mse))
            mse_values.append(mse)
            l1_values.append(l1)
            psnr_values.append(psnr)

            sample_dir = args.output_dir / "samples" / row["sample_id"].replace("/", "__")
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_tensor_image(torch, generated, sample_dir / "pred.png")
            gt_image.save(sample_dir / "gt_resized.png")
            shutil.copy2(image_path, sample_dir / "gt_source.png")
            (sample_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "category": category,
                        "category_id": category_id,
                        "image_path": str(image_path),
                        "mse": mse,
                        "l1": l1,
                        "psnr": psnr,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            index_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "category": category,
                    "image_path": str(image_path),
                    "output_dir": str(sample_dir),
                    "mse": mse,
                    "l1": l1,
                    "psnr": psnr,
                }
            )
            print(f"[{index}/{sample_count}] {row['sample_id']} mse={mse:.6f} l1={l1:.6f} psnr={psnr:.3f}")

    result = {
        "manifest": str(args.manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "device": str(device),
        "image_size": list(image_size),
        "num_samples": sample_count,
        "seed": args.seed,
        "sampling_steps": num_steps,
        "avg_mse": sum(mse_values) / max(1, len(mse_values)),
        "avg_l1": sum(l1_values) / max(1, len(l1_values)),
        "avg_psnr": sum(psnr_values) / max(1, len(psnr_values)),
        "samples": index_rows,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
