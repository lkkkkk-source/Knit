from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from PIL import Image


class RefinerSample(TypedDict):
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
        raise ImportError("PyTorch is required for refiner-v1 training. Install with `pip install -e .[train]`.") from error
    return torch, data


def load_refiner_manifest(path: str | Path) -> list[RefinerSample]:
    manifest_path = Path(path)
    rows: list[RefinerSample] = []
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


def resize_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.resize(size, resample=Image.Resampling.BILINEAR)


class PairedRefinerDataset:
    def __init__(self, manifest_path: str | Path, image_size: tuple[int, int] = (160, 160)) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self._infer_root(self.manifest_path)
        self.image_size = image_size
        self.samples = load_refiner_manifest(self.manifest_path)

    @staticmethod
    def _infer_root(manifest_path: Path) -> Path:
        direct_root = manifest_path.parent
        rows = load_refiner_manifest(manifest_path)
        if not rows:
            return direct_root
        direct_candidate = direct_root / rows[0]["input_path"]
        if direct_candidate.exists():
            return direct_root
        return manifest_path.parent.parent

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        torch, _ = _require_torch()
        sample = self.samples[index]
        input_path = (self.root / sample["input_path"]).resolve()
        target_path = (self.root / sample["target_path"]).resolve()
        source = resize_image(load_rgb_image(input_path), self.image_size)
        target = resize_image(load_rgb_image(target_path), self.image_size)
        source_tensor = getattr(torch, "tensor")(list(source.getdata()), dtype=getattr(torch, "float32")).reshape(source.height, source.width, 3)
        target_tensor = getattr(torch, "tensor")(list(target.getdata()), dtype=getattr(torch, "float32")).reshape(target.height, target.width, 3)
        source_tensor = source_tensor.permute(2, 0, 1) / 127.5 - 1.0
        target_tensor = target_tensor.permute(2, 0, 1) / 127.5 - 1.0
        return {
            "sample_id": sample["sample_id"],
            "paired_rendering_id": sample["paired_rendering_id"],
            "category": sample["category"],
            "source": source_tensor,
            "target": target_tensor,
            "input_path": str(input_path),
            "target_path": str(target_path),
        }


def collate_refiner_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    torch, _ = _require_torch()
    sources = getattr(torch, "stack")([sample["source"] for sample in batch])
    targets = getattr(torch, "stack")([sample["target"] for sample in batch])
    return {
        "sample_ids": [str(sample["sample_id"]) for sample in batch],
        "paired_rendering_ids": [str(sample["paired_rendering_id"]) for sample in batch],
        "categories": [str(sample["category"]) for sample in batch],
        "sources": sources,
        "targets": targets,
        "input_paths": [str(sample["input_path"]) for sample in batch],
        "target_paths": [str(sample["target_path"]) for sample in batch],
    }


def build_refiner_dataloader(
    manifest_path: str | Path,
    batch_size: int,
    shuffle: bool,
    image_size: tuple[int, int] = (160, 160),
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> tuple[object, PairedRefinerDataset]:
    _, data = _require_torch()
    dataset = PairedRefinerDataset(manifest_path, image_size=image_size)
    dataloader_cls = getattr(data, "DataLoader")
    worker_persistent = persistent_workers if num_workers > 0 else False
    return dataloader_cls(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_refiner_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=worker_persistent,
    ), dataset
