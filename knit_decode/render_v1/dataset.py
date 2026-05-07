from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

from PIL import Image


class RenderSample(TypedDict):
    sample_id: str
    category: str
    image_path: str


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        data = importlib.import_module("torch.utils.data")
    except ImportError as error:
        raise ImportError("PyTorch is required for render-v1 training. Install with `pip install -e .[train]`.") from error
    return torch, data


def load_render_manifest(path: str | Path) -> list[RenderSample]:
    manifest_path = Path(path)
    rows: list[RenderSample] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object line in {manifest_path}")
        sample_id = payload.get("sample_id")
        category = payload.get("category")
        image_path = payload.get("image_path", payload.get("input_path"))
        if category is None and isinstance(sample_id, str):
            category = sample_id.split("_", 1)[0]
        if not isinstance(sample_id, str) or not isinstance(category, str) or not isinstance(image_path, str):
            raise ValueError(f"Invalid render sample entry in {manifest_path}: {payload!r}")
        rows.append({"sample_id": sample_id, "category": category, "image_path": image_path})
    return rows


def load_rgb_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    except OSError as error:
        raise OSError(f"Failed to load image {path}: {error}") from error


def resize_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.resize(size, resample=Image.Resampling.BILINEAR)


class CategoryRenderingDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        image_size: tuple[int, int] = (160, 160),
        category_to_id: dict[str, int] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self._infer_root(self.manifest_path)
        self.image_size = image_size
        self.samples = load_render_manifest(self.manifest_path)
        categories = sorted({sample["category"] for sample in self.samples})
        self.category_to_id = category_to_id or {category: idx for idx, category in enumerate(categories)}
        self.id_to_category = {idx: category for category, idx in self.category_to_id.items()}

    @staticmethod
    def _infer_root(manifest_path: Path) -> Path:
        direct_root = manifest_path.parent
        probe = load_render_manifest(manifest_path)
        if not probe:
            return direct_root
        sample = probe[0]
        direct_candidate = direct_root / sample["image_path"]
        if direct_candidate.exists():
            return direct_root
        return manifest_path.parent.parent

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        torch, _ = _require_torch()
        sample = self.samples[index]
        image_path = (self.root / sample["image_path"]).resolve()
        image = resize_image(load_rgb_image(image_path), self.image_size)
        pixels = list(image.getdata())
        flat = []
        for pixel in pixels:
            flat.extend(pixel)
        image_tensor = getattr(torch, "tensor")(flat, dtype=getattr(torch, "float32")).reshape(image.height, image.width, 3)
        image_tensor = image_tensor.permute(2, 0, 1) / 127.5 - 1.0
        return {
            "sample_id": sample["sample_id"],
            "category": sample["category"],
            "category_id": self.category_to_id[sample["category"]],
            "image": image_tensor,
            "image_path": str(image_path),
        }


def collate_render_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    torch, _ = _require_torch()
    images = getattr(torch, "stack")([sample["image"] for sample in batch])
    category_ids = getattr(torch, "tensor")([int(sample["category_id"]) for sample in batch], dtype=getattr(torch, "long"))
    return {
        "sample_ids": [str(sample["sample_id"]) for sample in batch],
        "categories": [str(sample["category"]) for sample in batch],
        "category_ids": category_ids,
        "images": images,
        "image_paths": [str(sample["image_path"]) for sample in batch],
    }


def build_render_dataloader(
    manifest_path: str | Path,
    batch_size: int,
    shuffle: bool,
    image_size: tuple[int, int] = (160, 160),
    category_to_id: dict[str, int] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> tuple[object, CategoryRenderingDataset]:
    _, data = _require_torch()
    dataset = CategoryRenderingDataset(manifest_path, image_size=image_size, category_to_id=category_to_id)
    dataloader_cls = getattr(data, "DataLoader")
    worker_persistent = persistent_workers if num_workers > 0 else False
    return dataloader_cls(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_render_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=worker_persistent,
    ), dataset
