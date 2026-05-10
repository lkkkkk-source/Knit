from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TypedDict, cast

from PIL import Image


NUM_CLASSES = 17


class ParserInverseSample(TypedDict):
    sample_id: str
    category: str
    input_path: str
    target_path: str
    index_path: str


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        data = importlib.import_module("torch.utils.data")
    except ImportError as error:
        raise ImportError("PyTorch is required for parser_t_inverse training. Install with `pip install -e .[train]`.") from error
    return torch, data


def load_manifest(path: str | Path) -> list[ParserInverseSample]:
    manifest_path = Path(path)
    rows: list[ParserInverseSample] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object line in {manifest_path}")
        row = {}
        for key in ["sample_id", "category", "input_path", "target_path", "index_path"]:
            value = payload.get(key)
            if not isinstance(value, str):
                raise ValueError(f"Missing {key!r} in manifest row: {payload!r}")
            row[key] = value
        rows.append(cast(ParserInverseSample, row))
    return rows


def read_palette_mapping(path: str | Path) -> dict[tuple[int, int, int], int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid palette mapping: {path}")
    mapping: dict[tuple[int, int, int], int] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, int):
            raise ValueError(f"Invalid palette entry: {key!r} -> {value!r}")
        rgb = tuple(int(part) for part in key.split(","))
        if len(rgb) != 3:
            raise ValueError(f"Invalid RGB key: {key!r}")
        mapping[cast(tuple[int, int, int], rgb)] = int(value) - 1
    return mapping


def load_rgb_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    except OSError as error:
        raise OSError(f"Failed to load image {path}: {error}") from error


def load_grayscale_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            image.load()
            return image.convert("L")
    except OSError as error:
        raise OSError(f"Failed to load image {path}: {error}") from error


def resize_image(image: Image.Image, size: tuple[int, int], nearest: bool = False) -> Image.Image:
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return image.resize(size, resample=resample)


def instruction_to_class_grid(image: Image.Image, palette_mapping: dict[tuple[int, int, int], int]) -> list[list[int]]:
    width, height = image.size
    if width != 20 or height != 20:
        raise ValueError(f"Expected 20x20 instruction image, got {width}x{height}")
    grid: list[list[int]] = []
    for y_pos in range(height):
        row: list[int] = []
        for x_pos in range(width):
            color = cast(tuple[int, int, int], tuple(int(channel) for channel in image.getpixel((x_pos, y_pos))))
            if color not in palette_mapping:
                raise KeyError(f"Color {color} missing from palette mapping")
            row.append(palette_mapping[color])
        grid.append(row)
    return grid


def load_index_counts(path: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label, count = line.strip().split()[:2]
        counts[int(label) - 1] = int(count)
    return counts


def _infer_manifest_root(manifest_path: Path, rows: list[ParserInverseSample]) -> Path:
    search_roots = [manifest_path.parent, *manifest_path.parents]
    for candidate_root in search_roots:
        if all((candidate_root / row["target_path"]).exists() for row in rows[: min(4, len(rows))]):
            return candidate_root
    raise FileNotFoundError(
        f"Unable to resolve manifest root for {manifest_path}. "
        "Checked manifest directory and its parents against target_path entries."
    )


class ParserInverseDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        palette_path: str | Path,
        image_size: tuple[int, int] = (160, 160),
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self._infer_root(self.manifest_path)
        self.image_size = image_size
        self.samples = load_manifest(self.manifest_path)
        self.palette_mapping = read_palette_mapping(palette_path)
        self.class_names = [f"class_{index + 1}" for index in range(NUM_CLASSES)]

    @staticmethod
    def _infer_root(manifest_path: Path) -> Path:
        rows = load_manifest(manifest_path)
        if not rows:
            return manifest_path.parent
        return _infer_manifest_root(manifest_path, rows)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        torch, _ = _require_torch()
        sample = self.samples[index]
        input_path = (self.root / sample["input_path"]).resolve()
        target_path = (self.root / sample["target_path"]).resolve()
        index_path = (self.root / sample["index_path"]).resolve()
        grayscale = resize_image(load_grayscale_image(input_path), self.image_size, nearest=False)
        instruction = load_rgb_image(target_path)
        class_grid = instruction_to_class_grid(instruction, self.palette_mapping)
        image_tensor = getattr(torch, "tensor")(list(grayscale.getdata()), dtype=getattr(torch, "float32")).reshape(1, grayscale.height, grayscale.width) / 255.0
        target_tensor = getattr(torch, "tensor")(class_grid, dtype=getattr(torch, "long"))
        class_counts = load_index_counts(index_path)
        count_vector = [class_counts.get(class_id, 0) for class_id in range(NUM_CLASSES)]
        count_tensor = getattr(torch, "tensor")(count_vector, dtype=getattr(torch, "float32"))
        return {
            "sample_id": sample["sample_id"],
            "category": sample["category"],
            "image": image_tensor,
            "target": target_tensor,
            "count_vector": count_tensor,
        }


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    torch, _ = _require_torch()
    images = getattr(torch, "stack")([sample["image"] for sample in batch])
    targets = getattr(torch, "stack")([sample["target"] for sample in batch])
    counts = getattr(torch, "stack")([sample["count_vector"] for sample in batch])
    return {
        "sample_ids": [str(sample["sample_id"]) for sample in batch],
        "categories": [str(sample["category"]) for sample in batch],
        "images": images,
        "targets": targets,
        "count_vectors": counts,
    }


def build_dataloader(
    manifest_path: str | Path,
    palette_path: str | Path,
    batch_size: int,
    shuffle: bool,
    image_size: tuple[int, int] = (160, 160),
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> tuple[object, ParserInverseDataset]:
    _, data = _require_torch()
    dataset = ParserInverseDataset(manifest_path, palette_path=palette_path, image_size=image_size)
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


def compute_class_counts(dataset: ParserInverseDataset) -> list[int]:
    counts = [0 for _ in range(NUM_CLASSES)]
    for sample in dataset.samples:
        index_path = (dataset.root / sample["index_path"]).resolve()
        row_counts = load_index_counts(index_path)
        for class_id, value in row_counts.items():
            counts[class_id] += value
    return counts
