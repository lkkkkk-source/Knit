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
OTHER_COLOR: RGB = (255, 0, 255)


class ParserSample(TypedDict):
    sample_id: str
    category: str
    image_path: str
    target_path: str
    shard_path: str
    item_index: int


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


@dataclass(frozen=True)
class ColorVocabulary:
    top_colors: tuple[RGB, ...]
    color_to_class: dict[RGB, int]
    class_names: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def other_class_id(self) -> int:
        return len(self.top_colors)

    def render_color(self, class_id: int) -> RGB:
        if class_id < len(self.top_colors):
            return self.top_colors[class_id]
        return OTHER_COLOR

    def to_jsonable(self) -> dict[str, object]:
        return {
            "top_colors": [list(color) for color in self.top_colors],
            "class_names": list(self.class_names),
        }

    @staticmethod
    def from_jsonable(payload: dict[str, object]) -> "ColorVocabulary":
        raw_colors = payload.get("top_colors")
        raw_names = payload.get("class_names")
        if not isinstance(raw_colors, list) or not isinstance(raw_names, list):
            raise ValueError("Invalid vocabulary payload")
        top_colors = tuple(
            cast(RGB, tuple(int(channel) for channel in color))
            for color in raw_colors
            if isinstance(color, list) and len(color) == 3
        )
        class_names = tuple(str(name) for name in raw_names)
        return ColorVocabulary(
            top_colors=top_colors,
            color_to_class={color: idx for idx, color in enumerate(top_colors)},
            class_names=class_names,
        )


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
        sample_id = payload.get("sample_id")
        input_path = payload.get("image_path", payload.get("input_path"))
        target_path = payload.get("target_path")
        shard_path = payload.get("shard_path")
        item_index = payload.get("item_index")
        category = payload.get("category")
        if category is None and isinstance(sample_id, str):
            category = sample_id.split("/", 1)[0] if "/" in sample_id else sample_id.split("_", 1)[0]
        uses_paths = isinstance(input_path, str) and isinstance(target_path, str)
        uses_shard = isinstance(shard_path, str) and isinstance(item_index, int)
        if not isinstance(sample_id, str) or not isinstance(category, str) or (not uses_paths and not uses_shard):
            raise ValueError(f"Invalid parser sample entry in {manifest_path}: {payload!r}")
        rows.append(
            {
                "sample_id": sample_id,
                "category": category,
                "image_path": "" if not uses_paths else input_path,
                "target_path": "" if not uses_paths else target_path,
                "shard_path": "" if not uses_shard else shard_path,
                "item_index": -1 if not uses_shard else item_index,
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


def load_grayscale_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            image.load()
            return image.convert("L")
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


def downsample_color_grid(image: Image.Image, grid_size: tuple[int, int]) -> list[list[RGB]]:
    rows = image.height
    cols = image.width
    grid_rows, grid_cols = grid_size
    output: list[list[RGB]] = []
    for grid_row in range(grid_rows):
        row_values: list[RGB] = []
        y0 = (grid_row * rows) // grid_rows
        y1 = ((grid_row + 1) * rows) // grid_rows
        for grid_col in range(grid_cols):
            x0 = (grid_col * cols) // grid_cols
            x1 = ((grid_col + 1) * cols) // grid_cols
            counts = Counter()
            for y_pos in range(y0, max(y0 + 1, y1)):
                for x_pos in range(x0, max(x0 + 1, x1)):
                    counts[cast(RGB, image.getpixel((x_pos, y_pos)))] += 1
            row_values.append(counts.most_common(1)[0][0])
        output.append(row_values)
    return output


def build_topk_color_vocabulary(samples: Sequence[SegmentationTarget], image_size: tuple[int, int], grid_size: tuple[int, int], top_k: int) -> ColorVocabulary:
    counter: Counter[RGB] = Counter()
    for sample in samples:
        target_image = load_rgb_image(sample.target_path)
        crop_box = infer_active_crop(target_image)
        resized = resize_image(crop_image(target_image, crop_box), image_size, nearest=True)
        color_grid = downsample_color_grid(resized, grid_size)
        for row in color_grid:
            counter.update(row)
    top_colors = tuple(color for color, _ in counter.most_common(top_k))
    color_to_class = {color: idx for idx, color in enumerate(top_colors)}
    class_names = tuple([f"color_{index}" for index in range(len(top_colors))] + ["other"])
    return ColorVocabulary(top_colors=top_colors, color_to_class=color_to_class, class_names=class_names)


def color_grid_to_class_grid(color_grid: list[list[RGB]], vocabulary: ColorVocabulary) -> list[list[int]]:
    rows: list[list[int]] = []
    for row in color_grid:
        rows.append([vocabulary.color_to_class.get(color, vocabulary.other_class_id) for color in row])
    return rows


def mask_to_image(mask: list[list[int]], vocabulary: ColorVocabulary) -> Image.Image:
    height = len(mask)
    width = len(mask[0]) if height else 0
    image = Image.new("RGB", (width, height))
    for y_pos, row in enumerate(mask):
        for x_pos, class_id in enumerate(row):
            image.putpixel((x_pos, y_pos), vocabulary.render_color(class_id))
    return image


def write_vocabulary(path: Path, vocabulary: ColorVocabulary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocabulary.to_jsonable(), indent=2, ensure_ascii=False), encoding="utf-8")


def read_vocabulary(path: Path) -> ColorVocabulary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid vocabulary file: {path}")
    return ColorVocabulary.from_jsonable(payload)


def write_grid_json(path: Path, grid: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": len(grid),
        "cols": len(grid[0]) if grid else 0,
        "grid": grid,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_grid_json(path: Path) -> list[list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid cached grid file: {path}")
    grid = payload.get("grid")
    if not isinstance(grid, list):
        raise ValueError(f"Missing grid in cached grid file: {path}")
    rows: list[list[int]] = []
    for row in grid:
        if not isinstance(row, list):
            raise ValueError(f"Invalid row in cached grid file: {path}")
        rows.append([int(value) for value in row])
    return rows


class SimulationTopologyDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        root: str | Path | None = None,
        image_size: tuple[int, int] = (160, 160),
        grid_size: tuple[int, int] = (20, 20),
        vocabulary: ColorVocabulary | None = None,
        top_k_colors: int = 4,
        crop_input: bool = False,
        crop_padding: int = 2,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = Path(root) if root is not None else self._infer_root(self.manifest_path)
        self.image_size = image_size
        self.grid_size = grid_size
        self.crop_input = crop_input
        self.crop_padding = crop_padding
        raw_samples = load_parser_manifest(self.manifest_path)
        self.samples = [
            SegmentationTarget(
                sample_id=sample["sample_id"],
                category=sample["category"],
                image_path=Path() if not sample["image_path"] else (self.root / sample["image_path"]).resolve(),
                target_path=Path() if not sample["target_path"] else (self.root / sample["target_path"]).resolve(),
            )
            for sample in raw_samples
        ]
        self.raw_samples = raw_samples
        self.cached_mode = all(sample.target_path.suffix.lower() == ".json" for sample in self.samples) if self.samples else False
        self.shard_mode = all(bool(sample["shard_path"]) for sample in raw_samples) if raw_samples else False
        self._shard_cache: dict[Path, dict[str, object]] = {}
        if vocabulary is not None:
            self.vocabulary = vocabulary
        elif self.cached_mode or self.shard_mode:
            vocab_path = self.manifest_path.parent / "vocabulary.json"
            self.vocabulary = read_vocabulary(vocab_path)
        else:
            self.vocabulary = build_topk_color_vocabulary(
                self.samples,
                image_size=self.image_size,
                grid_size=self.grid_size,
                top_k=top_k_colors,
            )
        self.num_classes = self.vocabulary.num_classes
        self.class_names = list(self.vocabulary.class_names)

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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        torch, _ = _require_torch()
        sample = self.samples[index]
        raw_sample = self.raw_samples[index]
        if self.shard_mode:
            shard_path = (self.manifest_path.parent / raw_sample["shard_path"]).resolve()
            shard = self._shard_cache.get(shard_path)
            if shard is None:
                shard = getattr(torch, "load")(shard_path, map_location="cpu")
                self._shard_cache[shard_path] = cast(dict[str, object], shard)
            item_index = raw_sample["item_index"]
            image_tensor = cast(object, shard["images"])[item_index].to(dtype=getattr(torch, "float32")) / 255.0
            target_tensor = cast(object, shard["targets"])[item_index].to(dtype=getattr(torch, "long"))
            return {
                "sample_id": sample.sample_id,
                "category": sample.category,
                "image": image_tensor,
                "target": target_tensor,
            }
        if self.cached_mode:
            image = load_grayscale_image(sample.image_path)
            image_data = list(image.getdata())
            height, width = image.height, image.width
            image_tensor = getattr(torch, "tensor")([pixel / 255.0 for pixel in image_data], dtype=getattr(torch, "float32")).reshape(1, height, width)
            class_grid = read_grid_json(sample.target_path)
            target_tensor = getattr(torch, "tensor")(class_grid, dtype=getattr(torch, "long"))
        else:
            source_image = load_rgb_image(sample.image_path)
            source_target = load_rgb_image(sample.target_path)
            if self.crop_input:
                crop_box = infer_active_crop(source_image, padding=self.crop_padding)
                source_image = crop_image(source_image, crop_box)
            image = resize_image(source_image, self.image_size, nearest=False).convert("L")
            target_image = resize_image(source_target, self.image_size, nearest=True)
            image_data = list(image.getdata())
            height, width = image.height, image.width
            image_tensor = getattr(torch, "tensor")([pixel / 255.0 for pixel in image_data], dtype=getattr(torch, "float32")).reshape(1, height, width)
            color_grid = downsample_color_grid(target_image, self.grid_size)
            class_grid = color_grid_to_class_grid(color_grid, self.vocabulary)
            target_tensor = getattr(torch, "tensor")(class_grid, dtype=getattr(torch, "long"))
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
    vocabulary: ColorVocabulary | None = None,
    top_k_colors: int = 4,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    crop_input: bool = False,
    crop_padding: int = 2,
) -> object:
    _, data = _require_torch()
    dataset = SimulationTopologyDataset(
        manifest_path,
        image_size=image_size,
        grid_size=grid_size,
        vocabulary=vocabulary,
        top_k_colors=top_k_colors,
        crop_input=crop_input,
        crop_padding=crop_padding,
    )
    dataloader_cls = getattr(data, "DataLoader")
    worker_persistent = persistent_workers if num_workers > 0 else False
    return dataloader_cls(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_parser_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=worker_persistent,
    ), dataset


def compute_class_pixel_counts(dataset: SimulationTopologyDataset) -> list[int]:
    metadata_path = dataset.manifest_path.parent / "metadata.json"
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        counts = payload.get("class_pixel_counts")
        if isinstance(counts, list) and all(isinstance(value, int) for value in counts):
            return [int(value) for value in counts]

    counts = [0 for _ in range(dataset.num_classes)]
    if dataset.cached_mode:
        for sample in dataset.samples:
            class_grid = read_grid_json(sample.target_path)
            for row in class_grid:
                for class_id in row:
                    counts[class_id] += 1
        return counts

    if dataset.shard_mode:
        torch, _ = _require_torch()
        seen_shards: set[Path] = set()
        for raw_sample in dataset.raw_samples:
            shard_path = (dataset.manifest_path.parent / raw_sample["shard_path"]).resolve()
            if shard_path in seen_shards:
                continue
            seen_shards.add(shard_path)
            shard = getattr(torch, "load")(shard_path, map_location="cpu")
            targets = cast(object, shard["targets"])
            bincount = getattr(torch, "bincount")(targets.reshape(-1), minlength=dataset.num_classes).tolist()
            for class_id, value in enumerate(bincount):
                counts[class_id] += int(value)
        return counts

    for sample in dataset.samples:
        target_image = resize_image(load_rgb_image(sample.target_path), dataset.image_size, nearest=True)
        color_grid = downsample_color_grid(target_image, dataset.grid_size)
        class_grid = color_grid_to_class_grid(color_grid, dataset.vocabulary)
        for row in class_grid:
            for class_id in row:
                counts[class_id] += 1
    return counts
