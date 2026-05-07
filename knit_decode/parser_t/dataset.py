from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import TypedDict, cast

from PIL import Image, ImageFile


RGB = tuple[int, int, int]
ImageFile.LOAD_TRUNCATED_IMAGES = True

SEMANTIC_CLASS_NAMES = (
    "background",
    "knit",
    "tuck",
    "transfer",
    "cable",
    "other",
)
SEMANTIC_CLASS_COLORS: dict[int, RGB] = {
    0: (0, 0, 0),
    1: (255, 255, 255),
    2: (0, 255, 0),
    3: (255, 0, 0),
    4: (0, 255, 255),
    5: (255, 0, 255),
}


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
class CropBox:
    left: int
    top: int
    right: int
    bottom: int


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


def _hex_to_rgb(value: str) -> RGB:
    normalized = value.strip().lstrip("#")
    if len(normalized) != 6:
        raise ValueError(f"Expected 6-digit hex color, received {value!r}")
    return (int(normalized[0:2], 16), int(normalized[2:4], 16), int(normalized[4:6], 16))


def _coarse_class_name(label: str) -> str:
    if any(token in label for token in ("空针", "无选针", "不织")):
        return "background"
    if "吊目" in label:
        return "tuck"
    if "索骨" in label:
        return "cable"
    if any(token in label for token in ("翻", "移")):
        return "transfer"
    if any(token in label for token in ("编织", "落布")):
        return "knit"
    return "other"


def _load_semantic_color_map(all_info_path: Path) -> dict[RGB, int]:
    payload = json.loads(all_info_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {all_info_path}")
    class_name_to_id = {name: idx for idx, name in enumerate(SEMANTIC_CLASS_NAMES)}
    color_to_class: dict[RGB, int] = {}
    for label, raw_value in payload.items():
        entries = raw_value if isinstance(raw_value, list) else [raw_value]
        if not isinstance(entries, list):
            continue
        coarse_id = class_name_to_id[_coarse_class_name(str(label))]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            color_value = entry.get("color")
            if not isinstance(color_value, str):
                continue
            rgb = _hex_to_rgb(color_value)
            existing = color_to_class.get(rgb)
            if existing is None or existing == coarse_id:
                color_to_class[rgb] = coarse_id
            else:
                color_to_class[rgb] = class_name_to_id["other"]
    return color_to_class


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
    try:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    except OSError as error:
        raise OSError(f"Failed to load image {path}: {error}") from error


def infer_background_color(image: Image.Image) -> RGB:
    width, height = image.size
    border_pixels: list[RGB] = []
    for x_pos in range(width):
        border_pixels.append(cast(RGB, image.getpixel((x_pos, 0))))
        border_pixels.append(cast(RGB, image.getpixel((x_pos, height - 1))))
    for y_pos in range(1, height - 1):
        border_pixels.append(cast(RGB, image.getpixel((0, y_pos))))
        border_pixels.append(cast(RGB, image.getpixel((width - 1, y_pos))))
    return Counter(border_pixels).most_common(1)[0][0]


def infer_active_crop(image: Image.Image, padding: int = 2) -> CropBox:
    width, height = image.size
    background = infer_background_color(image)
    active_x: list[int] = []
    active_y: list[int] = []
    for y_pos in range(height):
        for x_pos in range(width):
            if cast(RGB, image.getpixel((x_pos, y_pos))) != background:
                active_x.append(x_pos)
                active_y.append(y_pos)
    if not active_x or not active_y:
        return CropBox(0, 0, width, height)
    left = max(0, min(active_x) - padding)
    top = max(0, min(active_y) - padding)
    right = min(width, max(active_x) + 1 + padding)
    bottom = min(height, max(active_y) + 1 + padding)
    return CropBox(left, top, right, bottom)


def crop_image(image: Image.Image, crop_box: CropBox) -> Image.Image:
    return image.crop((crop_box.left, crop_box.top, crop_box.right, crop_box.bottom))


def resize_image(image: Image.Image, size: tuple[int, int], nearest: bool = False) -> Image.Image:
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return image.resize(size, resample=resample)


def downsample_semantic_grid(
    image: Image.Image,
    grid_size: tuple[int, int],
    color_to_class: dict[RGB, int],
) -> list[list[int]]:
    rows = image.height
    cols = image.width
    grid_rows, grid_cols = grid_size
    output: list[list[int]] = []
    for grid_row in range(grid_rows):
        row_values: list[int] = []
        y0 = (grid_row * rows) // grid_rows
        y1 = ((grid_row + 1) * rows) // grid_rows
        for grid_col in range(grid_cols):
            x0 = (grid_col * cols) // grid_cols
            x1 = ((grid_col + 1) * cols) // grid_cols
            counts = Counter()
            for y_pos in range(y0, max(y0 + 1, y1)):
                for x_pos in range(x0, max(x0 + 1, x1)):
                    color = cast(RGB, image.getpixel((x_pos, y_pos)))
                    counts[color_to_class.get(color, len(SEMANTIC_CLASS_NAMES) - 1)] += 1
            row_values.append(counts.most_common(1)[0][0])
        output.append(row_values)
    return output


def mask_to_image(mask: list[list[int]]) -> Image.Image:
    height = len(mask)
    width = len(mask[0]) if height else 0
    image = Image.new("RGB", (width, height))
    for y_pos, row in enumerate(mask):
        for x_pos, class_id in enumerate(row):
            image.putpixel((x_pos, y_pos), SEMANTIC_CLASS_COLORS[class_id])
    return image


class SimulationTopologyDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        root: str | Path | None = None,
        image_size: tuple[int, int] = (160, 160),
        grid_size: tuple[int, int] = (20, 20),
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = Path(root) if root is not None else self._infer_root(self.manifest_path)
        self.image_size = image_size
        self.grid_size = grid_size
        raw_samples = load_parser_manifest(self.manifest_path)
        self.samples = [
            SegmentationTarget(
                sample_id=sample["sample_id"],
                category=sample["category"],
                image_path=(self.root / sample["image_path"]).resolve(),
                target_path=(self.root / sample["target_path"]).resolve(),
            )
            for sample in raw_samples
        ]
        self.num_classes = len(SEMANTIC_CLASS_NAMES)
        self.class_names = list(SEMANTIC_CLASS_NAMES)
        self.color_to_class = self._infer_color_map()

    @staticmethod
    def _infer_root(manifest_path: Path) -> Path:
        direct_root = manifest_path.parent
        split_root = manifest_path.parent.parent
        probe = load_parser_manifest(manifest_path)
        if not probe:
            return direct_root
        sample = probe[0]
        direct_candidate = direct_root / sample["target_path"]
        if direct_candidate.exists():
            return direct_root
        split_candidate = split_root / sample["target_path"]
        if split_candidate.exists():
            return split_root
        return direct_root

    def _infer_color_map(self) -> dict[RGB, int]:
        for sample in self.samples:
            for parent in sample.target_path.parents:
                legend_path = parent / "all_info.json"
                if legend_path.exists():
                    return _load_semantic_color_map(legend_path)
        raise FileNotFoundError("Could not locate all_info.json for semantic color mapping.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        torch, _ = _require_torch()
        sample = self.samples[index]
        source_image = load_rgb_image(sample.image_path)
        source_target = load_rgb_image(sample.target_path)
        crop_box = infer_active_crop(source_target)
        image = resize_image(crop_image(source_image, crop_box), self.image_size, nearest=False).convert("L")
        target_image = resize_image(crop_image(source_target, crop_box), self.image_size, nearest=True)
        image_data = list(image.getdata())
        height, width = image.height, image.width
        image_tensor = getattr(torch, "tensor")([pixel / 255.0 for pixel in image_data], dtype=getattr(torch, "float32")).reshape(1, height, width)
        target_grid = downsample_semantic_grid(target_image, self.grid_size, self.color_to_class)
        target_tensor = getattr(torch, "tensor")(target_grid, dtype=getattr(torch, "long"))
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
    image_size: tuple[int, int] = (160, 160),
    grid_size: tuple[int, int] = (20, 20),
) -> object:
    _, data = _require_torch()
    dataset = SimulationTopologyDataset(manifest_path, image_size=image_size, grid_size=grid_size)
    dataloader_cls = getattr(data, "DataLoader")
    return dataloader_cls(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_parser_batch), dataset


def compute_class_pixel_counts(dataset: SimulationTopologyDataset) -> list[int]:
    counts = [0 for _ in range(dataset.num_classes)]
    for sample in dataset.samples:
        target_image = resize_image(
            crop_image(load_rgb_image(sample.target_path), infer_active_crop(load_rgb_image(sample.target_path))),
            dataset.image_size,
            nearest=True,
        )
        grid = downsample_semantic_grid(target_image, dataset.grid_size, dataset.color_to_class)
        for row in grid:
            for class_id in row:
                counts[class_id] += 1
    return counts
