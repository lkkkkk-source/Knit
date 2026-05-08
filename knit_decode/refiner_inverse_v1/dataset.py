from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from PIL import Image

from knit_decode.parser_t_inverse.dataset import instruction_to_class_grid, read_palette_mapping


class RefinerInverseSample(TypedDict):
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
        raise ImportError("PyTorch is required for refiner_inverse_v1 training. Install with `pip install -e .[train]`.") from error
    return torch, data


def load_manifest(path: str | Path) -> list[RefinerInverseSample]:
    manifest_path = Path(path)
    rows: list[RefinerInverseSample] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object line in {manifest_path}")
        sample_id = payload.get("sample_id")
        paired_rendering_id = payload.get("paired_rendering_id")
        category = payload.get("category")
        input_path = payload.get("input_path")
        target_path = payload.get("target_path")
        if not all(isinstance(value, str) for value in [sample_id, paired_rendering_id, category, input_path, target_path]):
            raise ValueError(f"Invalid refiner sample entry in {manifest_path}: {payload!r}")
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


def _load_points(path: Path, normalized: bool = False) -> list[tuple[float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        x_val, y_val = line.split(",")[:2]
        rows.append((float(x_val), float(y_val)))
    if len(rows) != 4:
        raise ValueError(f"Expected 4 corner points in {path}, got {len(rows)}")
    if normalized:
        return rows
    # original inverse repo points are matlab-style 1-indexed
    return [(x - 1.0, y - 1.0) for x, y in rows]


def _crop_with_points(image: Image.Image, points: list[tuple[float, float]], output_size: tuple[int, int], augment_scale: float = 0.0) -> Image.Image:
    import random

    src = []
    for x_val, y_val in points:
        jitter_x = 0.0
        jitter_y = 0.0
        if augment_scale > 0:
            jitter_x = (random.random() - 0.5) * augment_scale
            jitter_y = (random.random() - 0.5) * augment_scale
        src.append((x_val + jitter_x, y_val + jitter_y))
    quad = [
        src[0][0], src[0][1],
        src[1][0], src[1][1],
        src[2][0], src[2][1],
        src[3][0], src[3][1],
    ]
    return image.transform(output_size, Image.Transform.QUAD, quad, resample=Image.Resampling.BILINEAR)


class PairedInverseRefinerDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        palette_path: str | Path,
        image_size: tuple[int, int] = (160, 160),
        transfer_root: str | Path | None = None,
        use_best_crop: bool = False,
        augment_scale: float = 0.0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self._infer_root(self.manifest_path)
        self.image_size = image_size
        self.samples = load_manifest(self.manifest_path)
        self.palette_mapping = read_palette_mapping(palette_path)
        self.transfer_root = None if transfer_root is None else Path(transfer_root)
        self.use_best_crop = use_best_crop
        self.augment_scale = augment_scale

    @staticmethod
    def _infer_root(manifest_path: Path) -> Path:
        direct_root = manifest_path.parent
        rows = load_manifest(manifest_path)
        if not rows:
            return direct_root
        direct_candidate = direct_root / rows[0]["input_path"]
        if direct_candidate.exists():
            return direct_root
        return manifest_path.parent.parent

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_instruction_path(self, paired_rendering_id: str) -> Path:
        return (self.root / "dataset2" / "instruction" / f"{paired_rendering_id}.png").resolve()

    def _resolve_best_assets(self, sample_id: str) -> tuple[Path, Path | None]:
        rgb_path = (self.root / "dataset2" / "real" / "best" / "rgb" / f"{sample_id}.jpg").resolve()
        points_path = (self.root / "dataset2" / "real" / "best" / "points" / f"{sample_id}.txt").resolve()
        npoints_path = (self.root / "dataset2" / "real" / "best" / "npoints" / f"{sample_id}.txt").resolve()
        if points_path.exists():
            return rgb_path, points_path
        if npoints_path.exists():
            return rgb_path, npoints_path
        return rgb_path, None

    def __getitem__(self, index: int) -> dict[str, object]:
        torch, _ = _require_torch()
        sample = self.samples[index]
        input_path = (self.root / sample["input_path"]).resolve()
        target_path = (self.root / sample["target_path"]).resolve()
        source_image = load_rgb_image(input_path)
        if self.use_best_crop:
            best_rgb, points_path = self._resolve_best_assets(sample["sample_id"])
            if best_rgb.exists() and points_path is not None and points_path.exists():
                source_image = load_rgb_image(best_rgb)
                normalized = "npoints" in str(points_path.parent)
                points = _load_points(points_path, normalized=normalized)
                if normalized:
                    width, height = source_image.size
                    points = [(x * width, y * height) for x, y in points]
                source_image = _crop_with_points(source_image, points, self.image_size, augment_scale=self.augment_scale)
            else:
                source_image = resize_image(source_image, self.image_size, nearest=False)
        else:
            source_image = resize_image(source_image, self.image_size, nearest=False)

        target_image = resize_image(load_rgb_image(target_path), self.image_size, nearest=False)
        instruction_image = load_rgb_image(self._resolve_instruction_path(sample["paired_rendering_id"]))
        instruction_grid = instruction_to_class_grid(instruction_image, self.palette_mapping)

        transfer_tensor = None
        has_transfer = False
        if self.transfer_root is not None:
            candidate = (self.transfer_root / f"{sample['paired_rendering_id']}.jpg").resolve()
            if candidate.exists():
                transfer_image = resize_image(load_rgb_image(candidate), self.image_size, nearest=False)
                transfer_tensor = getattr(torch, "tensor")(list(transfer_image.getdata()), dtype=getattr(torch, "float32")).reshape(transfer_image.height, transfer_image.width, 3)
                transfer_tensor = transfer_tensor.permute(2, 0, 1) / 127.5 - 1.0
                has_transfer = True

        source_tensor = getattr(torch, "tensor")(list(source_image.getdata()), dtype=getattr(torch, "float32")).reshape(source_image.height, source_image.width, 3)
        target_tensor = getattr(torch, "tensor")(list(target_image.getdata()), dtype=getattr(torch, "float32")).reshape(target_image.height, target_image.width, 3)
        source_tensor = source_tensor.permute(2, 0, 1) / 127.5 - 1.0
        target_tensor = target_tensor.permute(2, 0, 1) / 127.5 - 1.0
        instruction_tensor = getattr(torch, "tensor")(instruction_grid, dtype=getattr(torch, "long"))

        return {
            "sample_id": sample["sample_id"],
            "paired_rendering_id": sample["paired_rendering_id"],
            "category": sample["category"],
            "source": source_tensor,
            "target": target_tensor,
            "instruction_target": instruction_tensor,
            "transfer": transfer_tensor,
            "has_transfer": has_transfer,
            "input_path": str(input_path),
            "target_path": str(target_path),
        }


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    torch, _ = _require_torch()
    sources = getattr(torch, "stack")([sample["source"] for sample in batch])
    targets = getattr(torch, "stack")([sample["target"] for sample in batch])
    instruction_targets = getattr(torch, "stack")([sample["instruction_target"] for sample in batch])
    has_transfer = [bool(sample["has_transfer"]) for sample in batch]
    if any(has_transfer):
        fallback = getattr(torch, "zeros_like")(batch[0]["source"])
        transfers = getattr(torch, "stack")([sample["transfer"] if sample["transfer"] is not None else fallback for sample in batch])
        transfer_mask = getattr(torch, "tensor")(has_transfer, dtype=getattr(torch, "bool"))
    else:
        transfers = None
        transfer_mask = None
    return {
        "sample_ids": [str(sample["sample_id"]) for sample in batch],
        "paired_rendering_ids": [str(sample["paired_rendering_id"]) for sample in batch],
        "categories": [str(sample["category"]) for sample in batch],
        "sources": sources,
        "targets": targets,
        "instruction_targets": instruction_targets,
        "transfers": transfers,
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
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> tuple[object, PairedInverseRefinerDataset]:
    _, data = _require_torch()
    dataset = PairedInverseRefinerDataset(
        manifest_path,
        palette_path=palette_path,
        image_size=image_size,
        transfer_root=transfer_root,
        use_best_crop=use_best_crop,
        augment_scale=augment_scale,
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
