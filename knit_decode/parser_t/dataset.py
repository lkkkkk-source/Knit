from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import TypedDict, cast

from PIL import Image


JsonObject = dict[str, object]
RGB = tuple[int, int, int]


class ParserSample(TypedDict):
    sample_id: str
    image_path: str
    target_path: str
    category: str


class ParserBatch(TypedDict):
    sample_ids: list[str]
    categories: list[str]
    images: object
    targets: object


@dataclass(frozen=True)
class SegmentationTarget:
    sample_id: str
    category: str
    image_path: Path
    target_path: Path


@dataclass(frozen=True)
class Palette:
    colors: tuple[RGB, ...]
    color_to_index: dict[RGB, int]

    @property
    def num_classes(self) -> int:
        return len(self.colors)


_HEM_PREFIX_RE = re.compile(r"^(\d+[A-Za-z]*)")


def _normalize_sim_stem(category: str, stem: str) -> str:
    if category == "Hem":
        match = _HEM_PREFIX_RE.match(stem)
        if match:
            return match.group(1)
    return stem


def _require_torch() -> tuple[object, object]:
    import importlib

    try:
        torch = importlib.import_module("torch")
        data = importlib.import_module("torch.utils.data")
    except ImportError as error:
        raise ImportError("PyTorch is required for parser training. Install with `pip install -e .[train]`.") from error
    return torch, data


def load_parser_manifest(path: str | Path) -> list[ParserSample]:
    manifest_path = Path(path)
    rows: list[ParserSample] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object line in {manifest_path}")
        required = ("sample_id", "image_path", "target_path", "category")
        if any(not isinstance(payload.get(key), str) for key in required):
            raise ValueError(f"Invalid parser sample entry in {manifest_path}: {payload!r}")
        rows.append(
            {
                "sample_id": str(payload["sample_id"]),
                "image_path": str(payload["image_path"]),
                "target_path": str(payload["target_path"]),
                "category": str(payload["category"]),
            }
        )
    return rows


def build_parser_manifest_from_dataset_complete(dataset_root: str | Path, output_path: str | Path) -> Path:
    dataset_root = Path(dataset_root)
    output_path = Path(output_path)
    manifest_root = output_path.parent.resolve()
    simulation_root = dataset_root / "simulation images"
    stitch_root = dataset_root / "stitch code patterns"
    manifest_rows: list[dict[str, object]] = []

    for category_dir in sorted(simulation_root.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        stitch_dir = stitch_root / category
        if not stitch_dir.exists():
            continue
        stitch_index = {path.stem: path for path in stitch_dir.glob("*.png")}
        for image_path in sorted(category_dir.glob("*.png")):
            normalized = _normalize_sim_stem(category, image_path.stem)
            stitch_candidate = stitch_index.get(f"{normalized}_resized")
            if stitch_candidate is None:
                continue
            manifest_rows.append(
                {
                    "sample_id": f"{category}/{image_path.stem}",
                    "category": category,
                    "image_path": os.path.relpath(image_path.resolve(), manifest_root).replace("\\", "/"),
                    "target_path": os.path.relpath(stitch_candidate.resolve(), manifest_root).replace("\\", "/"),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in manifest_rows)
    if text:
        text += "\n"
    output_path.write_text(text, encoding="utf-8")
    return output_path


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def build_palette(samples: Sequence[SegmentationTarget]) -> Palette:
    unique_colors: set[RGB] = set()
    for sample in samples:
        image = load_rgb_image(sample.target_path)
        unique_colors.update(cast(set[RGB], set(image.getdata())))
    sorted_colors = tuple(sorted(unique_colors))
    return Palette(colors=sorted_colors, color_to_index={color: idx for idx, color in enumerate(sorted_colors)})


def image_to_class_mask(image: Image.Image, palette: Palette) -> list[list[int]]:
    rows: list[list[int]] = []
    for y_pos in range(image.height):
        row: list[int] = []
        for x_pos in range(image.width):
            color = cast(RGB, image.getpixel((x_pos, y_pos)))
            if color not in palette.color_to_index:
                raise KeyError(f"Color {color!r} is not in the parser palette")
            row.append(palette.color_to_index[color])
        rows.append(row)
    return rows


def resize_image(image: Image.Image, size: tuple[int, int], nearest: bool = False) -> Image.Image:
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return image.resize(size, resample=resample)


class SimulationTopologyDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        root: str | Path | None = None,
        image_size: tuple[int, int] = (128, 128),
        palette: Palette | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = Path(root) if root is not None else self.manifest_path.parent
        self.image_size = image_size
        raw_samples = load_parser_manifest(self.manifest_path)
        self.samples = [
            SegmentationTarget(
                sample_id=sample["sample_id"],
                category=sample["category"],
                image_path=self.root / sample["image_path"],
                target_path=self.root / sample["target_path"],
            )
            for sample in raw_samples
        ]
        self.palette = palette if palette is not None else build_palette(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        torch, _ = _require_torch()
        sample = self.samples[index]
        image = resize_image(load_rgb_image(sample.image_path), self.image_size, nearest=False)
        target_image = resize_image(load_rgb_image(sample.target_path), self.image_size, nearest=True)
        image_data = list(image.getdata())
        channels = [
            [pixel[channel] / 255.0 for pixel in image_data]
            for channel in range(3)
        ]
        height, width = image.height, image.width
        image_tensor = getattr(torch, "tensor")(channels, dtype=getattr(torch, "float32")).reshape(3, height, width)
        target_mask = image_to_class_mask(target_image, self.palette)
        target_tensor = getattr(torch, "tensor")(target_mask, dtype=getattr(torch, "long"))
        return {
            "sample_id": sample.sample_id,
            "category": sample.category,
            "image": image_tensor,
            "target": target_tensor,
        }


def collate_parser_batch(batch: Sequence[dict[str, object]]) -> ParserBatch:
    torch, _ = _require_torch()
    images = getattr(torch, "stack")([sample["image"] for sample in batch])
    targets = getattr(torch, "stack")([sample["target"] for sample in batch])
    return {
        "sample_ids": [str(sample["sample_id"]) for sample in batch],
        "categories": [str(sample["category"]) for sample in batch],
        "images": images,
        "targets": targets,
    }


def build_parser_dataloader(
    manifest_path: str | Path,
    batch_size: int,
    shuffle: bool,
    image_size: tuple[int, int] = (128, 128),
    palette: Palette | None = None,
) -> object:
    _, data = _require_torch()
    dataset = SimulationTopologyDataset(manifest_path, image_size=image_size, palette=palette)
    dataloader_cls = getattr(data, "DataLoader")
    return dataloader_cls(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_parser_batch), dataset
