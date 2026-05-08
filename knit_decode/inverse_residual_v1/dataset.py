from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TypedDict

from PIL import Image

from knit_decode.parser_t_inverse.dataset import instruction_to_class_grid, read_palette_mapping


class InverseResidualSample(TypedDict):
    sample_id: str
    paired_rendering_id: str
    category: str
    input_path: str
    target_path: str


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        data = importlib.import_module("torch.utils.data")
    except ImportError as error:
        raise ImportError("PyTorch is required for inverse_residual_v1. Install with `pip install -e .[train]`.") from error
    return torch, data


def load_manifest(path: str | Path) -> list[InverseResidualSample]:
    manifest_path = Path(path)
    rows: list[InverseResidualSample] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {manifest_path}")
        sample_id = payload.get("sample_id")
        paired_rendering_id = payload.get("paired_rendering_id")
        category = payload.get("category")
        input_path = payload.get("input_path")
        target_path = payload.get("target_path")
        if not all(isinstance(value, str) for value in [sample_id, paired_rendering_id, category, input_path, target_path]):
            raise ValueError(f"Invalid row in {manifest_path}: {payload!r}")
        rows.append(
            {
                "sample_id": sample_id,
                "paired_rendering_id": paired_rendering_id,
                "category": category,
                "input_path": input_path,
                "target_path": target_path,
            }
        )
    return rows


def load_rgb_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    except OSError as error:
        raise OSError(f"Failed to load image {path}: {error}") from error


def resize_image(image: Image.Image, size: tuple[int, int], nearest: bool = False) -> Image.Image:
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return image.resize(size, resample=resample)


def load_points(path: Path) -> list[tuple[float, float]]:
    points = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        x_val, y_val = line.split(",")[:2]
        points.append((float(x_val), float(y_val)))
    if len(points) != 4:
        raise ValueError(f"Expected 4 points in {path}, got {len(points)}")
    return points


def crop_with_points(
    image: Image.Image,
    points: list[tuple[float, float]],
    output_size: tuple[int, int],
    normalized: bool = False,
    augment_scale: float = 0.0,
) -> Image.Image:
    width, height = image.size
    scaled = []
    for x_val, y_val in points:
        if normalized:
            x_val = x_val * width
            y_val = y_val * height
        else:
            x_val = x_val - 1.0
            y_val = y_val - 1.0
        if augment_scale > 0:
            x_val += (random.random() - 0.5) * augment_scale
            y_val += (random.random() - 0.5) * augment_scale
        scaled.append((x_val, y_val))
    quad = [
        scaled[0][0], scaled[0][1],
        scaled[1][0], scaled[1][1],
        scaled[2][0], scaled[2][1],
        scaled[3][0], scaled[3][1],
    ]
    return image.transform(output_size, Image.Transform.QUAD, quad, resample=Image.Resampling.BILINEAR)


class ResidualJointDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        palette_path: str | Path,
        image_size: tuple[int, int] = (160, 160),
        transfer_root: str | Path | None = None,
        use_best_crop: bool = False,
        augment_scale: float = 0.0,
        mean_value: float = 0.5,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self._infer_root(self.manifest_path)
        self.samples = load_manifest(self.manifest_path)
        self.palette_mapping = read_palette_mapping(palette_path)
        self.image_size = image_size
        self.transfer_root = None if transfer_root is None else Path(transfer_root)
        self.use_best_crop = use_best_crop
        self.augment_scale = augment_scale
        self.mean_value = mean_value

    @staticmethod
    def _infer_root(manifest_path: Path) -> Path:
        direct_root = manifest_path.parent
        rows = load_manifest(manifest_path)
        if not rows:
            return direct_root
        probe = direct_root / rows[0]["input_path"]
        if probe.exists():
            return direct_root
        return manifest_path.parent.parent

    def __len__(self) -> int:
        return len(self.samples)

    def _instruction_path(self, paired_rendering_id: str) -> Path:
        return (self.root / "dataset2" / "instruction" / f"{paired_rendering_id}.png").resolve()

    def _best_paths(self, sample_id: str) -> tuple[Path, Path | None, bool]:
        rgb_path = (self.root / "dataset2" / "real" / "best" / "rgb" / f"{sample_id}.jpg").resolve()
        points_path = (self.root / "dataset2" / "real" / "best" / "points" / f"{sample_id}.txt").resolve()
        if points_path.exists():
            return rgb_path, points_path, False
        npoints_path = (self.root / "dataset2" / "real" / "best" / "npoints" / f"{sample_id}.txt").resolve()
        if npoints_path.exists():
            return rgb_path, npoints_path, True
        return rgb_path, None, False

    def __getitem__(self, index: int) -> dict[str, object]:
        torch, _ = _require_torch()
        sample = self.samples[index]
        input_path = (self.root / sample["input_path"]).resolve()
        target_path = (self.root / sample["target_path"]).resolve()

        source = load_rgb_image(input_path)
        if self.use_best_crop:
            best_rgb, point_path, normalized = self._best_paths(sample["sample_id"])
            if best_rgb.exists() and point_path is not None and point_path.exists():
                source = load_rgb_image(best_rgb)
                source = crop_with_points(source, load_points(point_path), self.image_size, normalized=normalized, augment_scale=self.augment_scale)
            else:
                source = resize_image(source, self.image_size)
        else:
            source = resize_image(source, self.image_size)

        target = resize_image(load_rgb_image(target_path), self.image_size)
        instruction = load_rgb_image(self._instruction_path(sample["paired_rendering_id"]))
        instruction_grid = instruction_to_class_grid(instruction, self.palette_mapping)

        transfer_tensor = None
        has_transfer = False
        if self.transfer_root is not None:
            candidate = (self.transfer_root / f"{sample['paired_rendering_id']}.jpg").resolve()
            if candidate.exists():
                transfer = resize_image(load_rgb_image(candidate), self.image_size)
                transfer_tensor = getattr(torch, "tensor")(list(transfer.getdata()), dtype=getattr(torch, "float32")).reshape(transfer.height, transfer.width, 3)
                transfer_tensor = transfer_tensor.permute(2, 0, 1).mean(dim=0, keepdim=True) / 255.0 - self.mean_value
                has_transfer = True

        source_tensor = getattr(torch, "tensor")(list(source.getdata()), dtype=getattr(torch, "float32")).reshape(source.height, source.width, 3)
        target_tensor = getattr(torch, "tensor")(list(target.getdata()), dtype=getattr(torch, "float32")).reshape(target.height, target.width, 3)
        source_gray = source_tensor.permute(2, 0, 1).mean(dim=0, keepdim=True) / 255.0
        target_gray = target_tensor.permute(2, 0, 1).mean(dim=0, keepdim=True) / 255.0
        source_residual = source_gray - self.mean_value
        target_residual = target_gray - self.mean_value

        instruction_tensor = getattr(torch, "tensor")(instruction_grid, dtype=getattr(torch, "long"))
        return {
            "sample_id": sample["sample_id"],
            "paired_rendering_id": sample["paired_rendering_id"],
            "category": sample["category"],
            "source_gray": source_gray,
            "target_gray": target_gray,
            "source_residual": source_residual,
            "target_residual": target_residual,
            "instruction_target": instruction_tensor,
            "transfer_residual": transfer_tensor,
            "has_transfer": has_transfer,
            "input_path": str(input_path),
            "target_path": str(target_path),
        }


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    torch, _ = _require_torch()
    source_gray = getattr(torch, "stack")([sample["source_gray"] for sample in batch])
    target_gray = getattr(torch, "stack")([sample["target_gray"] for sample in batch])
    source_residual = getattr(torch, "stack")([sample["source_residual"] for sample in batch])
    target_residual = getattr(torch, "stack")([sample["target_residual"] for sample in batch])
    instruction_target = getattr(torch, "stack")([sample["instruction_target"] for sample in batch])
    has_transfer = [bool(sample["has_transfer"]) for sample in batch]
    if any(has_transfer):
        fallback = getattr(torch, "zeros_like")(batch[0]["source_residual"])
        transfer_residual = getattr(torch, "stack")([sample["transfer_residual"] if sample["transfer_residual"] is not None else fallback for sample in batch])
        transfer_mask = getattr(torch, "tensor")(has_transfer, dtype=getattr(torch, "bool"))
    else:
        transfer_residual = None
        transfer_mask = None
    return {
        "sample_ids": [str(sample["sample_id"]) for sample in batch],
        "paired_rendering_ids": [str(sample["paired_rendering_id"]) for sample in batch],
        "categories": [str(sample["category"]) for sample in batch],
        "source_gray": source_gray,
        "target_gray": target_gray,
        "source_residual": source_residual,
        "target_residual": target_residual,
        "instruction_target": instruction_target,
        "transfer_residual": transfer_residual,
        "transfer_mask": transfer_mask,
        "input_paths": [str(sample["input_path"]) for sample in batch],
        "target_paths": [str(sample["target_path"]) for sample in batch],
    }


def build_dataloader(
    manifest_path: str | Path,
    palette_path: str | Path,
    batch_size: int,
    shuffle: bool,
    image_size: tuple[int, int] = (160, 160),
    transfer_root: str | Path | None = None,
    use_best_crop: bool = False,
    augment_scale: float = 0.0,
    mean_value: float = 0.5,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> tuple[object, ResidualJointDataset]:
    _, data = _require_torch()
    dataset = ResidualJointDataset(
        manifest_path,
        palette_path=palette_path,
        image_size=image_size,
        transfer_root=transfer_root,
        use_best_crop=use_best_crop,
        augment_scale=augment_scale,
        mean_value=mean_value,
    )
    dataloader_cls = getattr(data, "DataLoader")
    worker_persistent = persistent_workers if num_workers > 0 else False
    return dataloader_cls(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=worker_persistent,
    ), dataset
