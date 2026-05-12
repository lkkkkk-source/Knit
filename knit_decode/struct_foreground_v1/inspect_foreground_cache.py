from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import cast

from .utils import EXPECTED_DESCRIPTOR_DIM, IGNORE_INDEX, REQUIRED_FOREGROUND_CACHE_SCHEMA_VERSION, assert_no_forbidden_cache_fields, format_metric_line, foreground_area, label_diversity_on_fg, resolve_canonical_mode, save_json, save_jsonl


REQUIRED_CACHE_KEYS = (
    "items",
    "centroid_sketch_by_category",
    "category_to_num_modes",
    "category_foreground_area_stats",
)

FALLBACK_PALETTE = [
    (255, 0, 0),
    (34, 139, 34),
    (255, 215, 0),
    (30, 144, 255),
    (255, 105, 180),
    (255, 140, 0),
    (148, 0, 211),
    (0, 206, 209),
    (154, 205, 50),
    (220, 20, 60),
    (255, 255, 255),
    (139, 69, 19),
    (70, 130, 180),
    (240, 230, 140),
    (199, 21, 133),
    (95, 158, 160),
    (255, 99, 71),
]
BACKGROUND_RED = (255, 0, 0)
FOREGROUND_GREEN = (0, 255, 0)
PLACEHOLDER_GRAY = (160, 160, 160)


def _require_torch() -> object:
    import importlib

    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required for inspect_foreground_cache.") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize foreground_v1 cache contents for a single category.")
    parser.add_argument("--cache", type=Path, default=Path("knit_decode/struct_foreground_v1/cache/foreground_v1/foreground_cache_train.pt"))
    parser.add_argument("--category", type=str, default="Cable1")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--cell-size", type=int, default=18)
    parser.add_argument("--label-mass-threshold", type=float, default=None)
    return parser


def _require_cache_fields(cache_payload: dict[str, object]) -> None:
    meta = cache_payload.get("meta")
    schema_version = meta.get("schema_version") if isinstance(meta, dict) else None
    if schema_version != REQUIRED_FOREGROUND_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Foreground cache has incompatible schema_version={schema_version!r}; "
            f"expected {REQUIRED_FOREGROUND_CACHE_SCHEMA_VERSION!r}. "
            "Please rebuild the foreground cache with the current build_foreground_cache.py."
        )
    assert_no_forbidden_cache_fields(cache_payload, context="Foreground cache")
    missing = [key for key in REQUIRED_CACHE_KEYS if key not in cache_payload]
    if missing:
        raise ValueError(f"Foreground cache is missing required fields: {', '.join(missing)}")


def _to_python_grid(value: object, *, context: str) -> list[list[int]]:
    if value is None:
        raise ValueError(f"{context} is missing.")
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError(f"{context} must be list-like, got {type(value)}")
    if len(value) != 20:
        raise ValueError(f"{context} must have 20 rows, got {len(value)}")
    rows: list[list[int]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 20:
            raise ValueError(f"{context} row {row_index} must have 20 columns.")
        rows.append([int(part) for part in row])
    return rows


def _centroid_mask_grid(entry: object, *, context: str) -> tuple[list[list[int]] | None, bool]:
    if entry is None:
        return None, True
    if isinstance(entry, dict):
        for key in ["centroid_fg_mask", "fg_mask20", "fg_mask", "mask"]:
            if key in entry and entry[key] is not None:
                return _to_python_grid(entry[key], context=f"{context}.{key}"), False
        return None, True
    return _to_python_grid(entry, context=context), False


def _centroid_prob_grid(entry: object, *, context: str) -> tuple[list[list[float]] | None, bool]:
    if entry is None:
        return None, True
    if isinstance(entry, dict):
        value = entry.get("centroid_fg_mask_prob")
        if value is None:
            return None, True
    else:
        return None, True
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], list) or len(value[0]) != 20 or any(not isinstance(row, list) or len(row) != 20 for row in value[0]):
        raise ValueError(f"{context}.centroid_fg_mask_prob must have shape [1,20,20].")
    return [[float(part) for part in row] for row in value[0]], False


def _centroid_label_prob_grid(entry: object, *, context: str) -> list[list[list[float]]]:
    if not isinstance(entry, dict) or "centroid_label_prob_16" not in entry:
        raise ValueError(f"{context}.centroid_label_prob_16 is required.")
    value = entry["centroid_label_prob_16"]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or len(value) != 16:
        raise ValueError(f"{context}.centroid_label_prob_16 must have shape [16,20,20].")
    channels: list[list[list[float]]] = []
    for channel_index, channel in enumerate(value):
        if not isinstance(channel, list) or len(channel) != 20 or any(not isinstance(row, list) or len(row) != 20 for row in channel):
            raise ValueError(f"{context}.centroid_label_prob_16[{channel_index}] must have shape [20,20].")
        channels.append([[float(part) for part in row] for row in channel])
    return channels


def _label_grid_for_display(fg_y20: list[list[int]], fg_mask20: list[list[int]]) -> list[list[int]]:
    grid: list[list[int]] = []
    for y_pos in range(20):
        row: list[int] = []
        for x_pos in range(20):
            label = int(fg_y20[y_pos][x_pos])
            if label == IGNORE_INDEX or int(fg_mask20[y_pos][x_pos]) <= 0:
                row.append(0)
            else:
                row.append(max(0, min(16, label)))
        grid.append(row)
    return grid


def _mask_grid_for_display(fg_mask20: list[list[int]]) -> list[list[int]]:
    return [[1 if int(value) > 0 else 0 for value in row] for row in fg_mask20]


def _overlay_grid_for_display(fg_y20: list[list[int]], fg_mask20: list[list[int]]) -> list[list[int]]:
    return _label_grid_for_display(fg_y20, fg_mask20)


def _original_y20_grid_for_display(original_y20: list[list[int]]) -> list[list[int]]:
    return [[max(0, min(16, int(value))) for value in row] for row in original_y20]


def _official_palette() -> list[tuple[int, int, int]]:
    try:
        from knit_decode.parser_t_inverse.palette import OFFICIAL_PALETTE

        return [tuple(int(channel) for channel in color) for color in OFFICIAL_PALETTE]
    except Exception:
        return FALLBACK_PALETTE


def _grid_to_rgb(grid: list[list[int]], *, mode: str, missing_mask: bool = False) -> list[list[tuple[int, int, int]]]:
    palette = _official_palette()
    rgb: list[list[tuple[int, int, int]]] = []
    for row in grid:
        rgb_row: list[tuple[int, int, int]] = []
        for value in row:
            if missing_mask:
                rgb_row.append(PLACEHOLDER_GRAY)
            elif mode == "mask":
                rgb_row.append(FOREGROUND_GREEN if int(value) > 0 else BACKGROUND_RED)
            else:
                class_id = int(value)
                if class_id <= 0:
                    rgb_row.append(BACKGROUND_RED)
                else:
                    rgb_row.append(palette[min(class_id, len(palette) - 1)])
        rgb.append(rgb_row)
    return rgb


def _prob_grid_to_rgb(grid: list[list[float]], *, missing_mask: bool = False) -> list[list[tuple[int, int, int]]]:
    rgb: list[list[tuple[int, int, int]]] = []
    for row in grid:
        rgb_row: list[tuple[int, int, int]] = []
        for value in row:
            if missing_mask:
                rgb_row.append(PLACEHOLDER_GRAY)
            else:
                shade = max(0, min(255, int(round(float(value) * 255.0))))
                rgb_row.append((shade, shade, shade))
        rgb.append(rgb_row)
    return rgb


def _label_prob_summary_grids(label_prob_16: list[list[list[float]]], *, label_mass_threshold: float) -> tuple[list[list[float]], list[list[int]], list[list[float]], dict[str, float]]:
    mass_grid = [[0.0 for _ in range(20)] for _ in range(20)]
    argmax_grid = [[0 for _ in range(20)] for _ in range(20)]
    confidence_grid = [[0.0 for _ in range(20)] for _ in range(20)]
    entropy_values: list[float] = []
    confidence_values: list[float] = []
    dominant_ratio_hits = 0
    label_diversity_values: list[float] = []
    active_count = 0
    for y_pos in range(20):
        for x_pos in range(20):
            values = [float(label_prob_16[channel][y_pos][x_pos]) for channel in range(16)]
            mass_value = sum(values)
            mass_grid[y_pos][x_pos] = mass_value
            best_index = max(range(16), key=lambda index: values[index])
            best_value = values[best_index]
            confidence_grid[y_pos][x_pos] = round(best_value, 3)
            if mass_value > label_mass_threshold:
                argmax_grid[y_pos][x_pos] = best_index + 1
                active_count += 1
                confidence_values.append(best_value)
                label_diversity_values.append(float(sum(1 for value in values if value > 0.0)))
                if best_value / max(mass_value, 1e-12) >= 0.5:
                    dominant_ratio_hits += 1
                normalized = [value / mass_value for value in values if value > 0.0]
                entropy = -sum(prob * math.log(prob + 1e-12) for prob in normalized)
                entropy_values.append(entropy)
    quality = {
        "label_mass_mean": sum(sum(row) for row in mass_grid) / 400.0,
        "label_mass_max": max(max(row) for row in mass_grid),
        "label_confidence_mean_on_mass": (sum(confidence_values) / float(len(confidence_values))) if confidence_values else 0.0,
        "label_entropy_mean_on_mass": (sum(entropy_values) / float(len(entropy_values))) if entropy_values else 0.0,
        "dominant_label_ratio_on_mass": float(dominant_ratio_hits) / float(max(1, active_count)),
        "label_diversity_on_mass": (sum(label_diversity_values) / float(len(label_diversity_values))) if label_diversity_values else 0.0,
        "active_mass_pixel_count": float(active_count),
    }
    return mass_grid, argmax_grid, confidence_grid, quality


def _save_tiled_grid(
    tiles: list[object],
    labels: list[str],
    output_path: Path,
    *,
    cols: int,
    cell_size: int,
) -> None:
    try:
        from PIL import Image, ImageDraw
        import numpy as np
    except Exception:
        save_json(output_path.with_suffix(output_path.suffix + ".json"), {"labels": labels, "tiles": tiles, "cols": cols, "cell_size": cell_size})
        return

    def tile_to_rgb_array(tile: object) -> object:
        array = np.asarray(tile)
        if array.ndim == 2:
            if np.issubdtype(array.dtype, np.floating):
                gray = (np.clip(array.astype(np.float32), 0.0, 1.0) * 255.0).round().astype(np.uint8)
                return np.repeat(gray[:, :, None], 3, axis=2)
            palette = np.asarray(_official_palette(), dtype=np.uint8)
            label_array = np.clip(array.astype(np.int64), 0, len(palette) - 1)
            return palette[label_array]
        if array.ndim == 3 and array.shape[2] == 3:
            if np.issubdtype(array.dtype, np.floating):
                max_value = float(np.nanmax(array)) if array.size else 1.0
                array = array * 255.0 if max_value <= 1.0 else array
            return np.clip(array, 0, 255).round().astype(np.uint8)
        raise ValueError(f"Tile must be [H,W] or [H,W,3], got shape={array.shape!r}.")

    if not tiles:
        raise ValueError("No tiles to render.")
    cols = max(1, cols)
    rows = (len(tiles) + cols - 1) // cols
    text_height = max(18, cell_size)
    canvas = Image.new("RGB", (cols * 20 * cell_size, rows * (20 * cell_size + text_height)), BACKGROUND_RED)
    draw = ImageDraw.Draw(canvas)
    print(f"saving {output_path}", flush=True)
    for index, tile in enumerate(tiles):
        x0 = (index % cols) * 20 * cell_size
        y0 = (index // cols) * (20 * cell_size + text_height)
        tile_rgb = tile_to_rgb_array(tile)
        tile_img = Image.fromarray(tile_rgb, mode="RGB")
        tile_img = tile_img.resize((20 * cell_size, 20 * cell_size), resample=Image.Resampling.NEAREST)
        canvas.paste(tile_img, (x0, y0 + text_height))
        draw.text((x0 + 2, y0 + 1), labels[index], fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"saved {output_path}", flush=True)


def _descriptor_norm(descriptor: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in descriptor))


def _projection_summary(values: object) -> dict[str, float]:
    if values is None:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, list) or not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    floats = [float(value) for value in values]
    return {
        "mean": sum(floats) / float(len(floats)),
        "min": min(floats),
        "max": max(floats),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cache_path = Path(args.cache)
    output_dir = Path(args.output_dir or Path("outputs") / f"foreground_cache_vis_{args.category}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _require_torch().load(cache_path, map_location="cpu")
    _require_cache_fields(payload)
    config = payload.get("config", {})
    data_cf = config.get("data", {})
    clustering_cf = config.get("clustering", {}) if isinstance(config.get("clustering", {}), dict) else {}
    canonical_mode = resolve_canonical_mode(data_cf)
    label_mass_threshold = float(args.label_mass_threshold if args.label_mass_threshold is not None else clustering_cf.get("label_mass_threshold", 0.05))
    if args.category not in payload["category_to_num_modes"]:
        raise ValueError(f"Category {args.category!r} not found in category_to_num_modes.")
    rng = random.Random(int(args.seed))
    items = [item for item in payload["items"] if str(item.get("category")) == args.category and not bool(item.get("is_empty_foreground", False))]
    if not items:
        raise ValueError(f"No non-empty foreground items found for category {args.category!r}.")
    sample_count = min(int(args.num_samples), len(items))
    samples = rng.sample(items, sample_count) if sample_count < len(items) else list(items)

    original_tiles = []
    y20_tiles = []
    mask_tiles = []
    overlay_tiles = []
    sample_rows: list[dict[str, object]] = []
    fg_areas: list[float] = []
    label_diversities: list[int] = []
    local_z_histogram: dict[str, int] = {}
    for index, item in enumerate(samples):
        fg_y20 = _to_python_grid(item["fg_y20"], context=f"item[{item['sample_id']}].fg_y20")
        fg_mask20 = _to_python_grid(item["fg_mask20"], context=f"item[{item['sample_id']}].fg_mask20")
        original_y20 = _to_python_grid(item["original_y20"], context=f"item[{item['sample_id']}].original_y20")
        descriptor = item.get("descriptor")
        if hasattr(descriptor, "tolist"):
            descriptor = descriptor.tolist()
        if not isinstance(descriptor, list) or len(descriptor) != EXPECTED_DESCRIPTOR_DIM:
            raise ValueError(f"item[{item['sample_id']}] descriptor must have length {EXPECTED_DESCRIPTOR_DIM}.")
        label_grid = _label_grid_for_display(fg_y20, fg_mask20)
        mask_grid = _mask_grid_for_display(fg_mask20)
        overlay_grid = _overlay_grid_for_display(fg_y20, fg_mask20)
        original_tiles.append(_grid_to_rgb(_original_y20_grid_for_display(original_y20), mode="label"))
        y20_tiles.append(_grid_to_rgb(label_grid, mode="label"))
        mask_tiles.append(_grid_to_rgb(mask_grid, mode="mask"))
        overlay_tiles.append(_grid_to_rgb(overlay_grid, mode="label"))
        local_z = int(item.get("local_z", -1))
        local_z_histogram[str(local_z)] = local_z_histogram.get(str(local_z), 0) + 1
        fg_area = float(item.get("fg_area", foreground_area(fg_mask20)))
        label_diversity = label_diversity_on_fg(label_grid, mask_grid)
        fg_areas.append(fg_area)
        label_diversities.append(label_diversity)
        sample_rows.append(
            {
                "sample_id": str(item["sample_id"]),
                "category": str(item["category"]),
                "local_z": local_z,
                "fg_area": fg_area,
                "label_diversity": label_diversity,
                "bbox_stats": item.get("bbox_stats"),
                "descriptor_norm": _descriptor_norm(cast(list[float], descriptor)),
                "output_row": index // max(1, int(args.cols)),
                "output_col": index % max(1, int(args.cols)),
            }
        )
    sample_labels = [str(row["sample_id"]) for row in sample_rows]
    _save_tiled_grid(original_tiles, sample_labels, output_dir / f"{args.category}_real_full_y20_grid.png", cols=int(args.cols), cell_size=int(args.cell_size))
    _save_tiled_grid(mask_tiles, sample_labels, output_dir / f"{args.category}_real_full_fg_mask_grid.png", cols=int(args.cols), cell_size=int(args.cell_size))
    _save_tiled_grid(overlay_tiles, sample_labels, output_dir / f"{args.category}_real_full_fg_y20_ignore_background_grid.png", cols=int(args.cols), cell_size=int(args.cell_size))

    centroid_entries = payload["centroid_sketch_by_category"].get(args.category)
    if centroid_entries is None:
        raise ValueError(f"Category {args.category!r} missing from centroid_sketch_by_category.")
    centroid_prob_tiles = []
    centroid_bin_tiles = []
    centroid_label_mass_tiles = []
    centroid_label_argmax_tiles = []
    centroid_label_confidence_tiles = []
    centroid_labels = []
    centroid_rows: list[dict[str, object]] = []
    if isinstance(centroid_entries, dict):
        iterable = sorted(centroid_entries.items(), key=lambda item: int(item[0]))
    elif isinstance(centroid_entries, list):
        iterable = list(enumerate(centroid_entries))
    else:
        raise ValueError(f"centroid_sketch_by_category[{args.category!r}] must be dict or list.")
    for local_z, entry in iterable:
        prob_grid, missing_prob = _centroid_prob_grid(entry, context=f"centroid[{args.category}][{local_z}]")
        mask_grid, missing_mask = _centroid_mask_grid(entry, context=f"centroid[{args.category}][{local_z}]")
        if prob_grid is None:
            prob_grid = [[0.0 for _ in range(20)] for _ in range(20)]
        if mask_grid is None:
            mask_grid = [[0 for _ in range(20)] for _ in range(20)]
        label_prob_16 = _centroid_label_prob_grid(entry, context=f"centroid[{args.category}][{local_z}]")
        label_mass_grid, label_argmax_grid, label_confidence_grid, label_quality = _label_prob_summary_grids(label_prob_16, label_mass_threshold=label_mass_threshold)
        centroid_prob_tiles.append(_prob_grid_to_rgb(prob_grid, missing_mask=missing_prob))
        centroid_bin_tiles.append(_grid_to_rgb(mask_grid, mode="mask", missing_mask=missing_mask))
        centroid_label_mass_tiles.append(_prob_grid_to_rgb(label_mass_grid, missing_mask=missing_prob))
        centroid_label_argmax_tiles.append(_grid_to_rgb(label_argmax_grid, mode="label", missing_mask=missing_prob))
        centroid_label_confidence_tiles.append(_prob_grid_to_rgb(label_confidence_grid, missing_mask=missing_prob))
        centroid_labels.append(f"z={local_z}")
        if isinstance(entry, dict):
            label_hist = entry.get("centroid_label_hist")
            if hasattr(label_hist, "tolist"):
                label_hist = label_hist.tolist()
            centroid_label_diversity = 0
            if isinstance(label_hist, list):
                centroid_label_diversity = sum(1 for value in label_hist if float(value) > 0.0)
            centroid_bbox_stats = entry.get("centroid_bbox_stats")
            if hasattr(centroid_bbox_stats, "tolist"):
                centroid_bbox_stats = centroid_bbox_stats.tolist()
            centroid_rows.append(
                {
                    "local_z": int(local_z),
                    "missing_mask": bool(missing_mask),
                    "missing_prob": bool(missing_prob),
                    "centroid_fg_area": foreground_area(mask_grid),
                    "centroid_fg_area_prob_mean": sum(sum(float(value) for value in row) for row in prob_grid) / 400.0,
                    "centroid_fg_area_prob_max": max(max(float(value) for value in row) for row in prob_grid),
                    "foreground_area_mean": foreground_area(mask_grid),
                    "label_mass_threshold": label_mass_threshold,
                    **label_quality,
                    "possible_label_collapse": bool(label_quality["dominant_label_ratio_on_mass"] > 0.8),
                    "low_label_diversity_centroid": bool(label_quality["label_diversity_on_mass"] <= 2.0),
                    "centroid_fg_mask_threshold": float(entry.get("centroid_fg_mask_threshold", 0.5)),
                    "fallback_used": bool(entry.get("fallback_used", False)),
                    "num_samples": int(entry.get("num_samples", 0)),
                    "centroid_label_diversity": centroid_label_diversity,
                    "centroid_bbox_stats": centroid_bbox_stats,
                    "centroid_row_projection_summary": _projection_summary(entry.get("centroid_row_projection")),
                    "centroid_col_projection_summary": _projection_summary(entry.get("centroid_col_projection")),
                }
            )
        else:
            centroid_rows.append(
                {
                    "local_z": int(local_z),
                    "missing_mask": bool(missing_mask),
                    "missing_prob": bool(missing_prob),
                    "centroid_fg_area": foreground_area(mask_grid),
                    "centroid_fg_area_prob_mean": 0.0,
                    "centroid_fg_area_prob_max": 0.0,
                    "foreground_area_mean": foreground_area(mask_grid),
                    "label_mass_threshold": label_mass_threshold,
                    "label_mass_mean": 0.0,
                    "label_mass_max": 0.0,
                    "label_confidence_mean_on_mass": 0.0,
                    "label_entropy_mean_on_mass": 0.0,
                    "dominant_label_ratio_on_mass": 0.0,
                    "label_diversity_on_mass": 0.0,
                    "active_mass_pixel_count": 0.0,
                    "possible_label_collapse": False,
                    "low_label_diversity_centroid": False,
                    "centroid_fg_mask_threshold": 0.0,
                    "fallback_used": False,
                    "num_samples": 0,
                    "centroid_label_diversity": 0,
                    "centroid_bbox_stats": None,
                    "centroid_row_projection_summary": {"mean": 0.0, "min": 0.0, "max": 0.0},
                    "centroid_col_projection_summary": {"mean": 0.0, "min": 0.0, "max": 0.0},
                }
            )
    _save_tiled_grid(centroid_prob_tiles, centroid_labels, output_dir / f"{args.category}_centroid_mask_prob_grid.png", cols=int(args.cols), cell_size=int(args.cell_size))
    _save_tiled_grid(centroid_bin_tiles, centroid_labels, output_dir / f"{args.category}_centroid_mask_bin_grid.png", cols=int(args.cols), cell_size=int(args.cell_size))
    _save_tiled_grid(centroid_bin_tiles, centroid_labels, output_dir / f"{args.category}_centroid_masks_grid.png", cols=int(args.cols), cell_size=int(args.cell_size))
    _save_tiled_grid(centroid_label_mass_tiles, centroid_labels, output_dir / f"{args.category}_centroid_label_mass_grid.png", cols=int(args.cols), cell_size=int(args.cell_size))
    _save_tiled_grid(centroid_label_argmax_tiles, centroid_labels, output_dir / f"{args.category}_centroid_label_argmax_grid.png", cols=int(args.cols), cell_size=int(args.cell_size))
    _save_tiled_grid(centroid_label_confidence_tiles, centroid_labels, output_dir / f"{args.category}_centroid_label_confidence_grid.png", cols=int(args.cols), cell_size=int(args.cell_size))

    sample_stats = {
        "category": args.category,
        "canonical_mode": canonical_mode,
        "num_total_items": len(items),
        "num_visualized": len(samples),
        "fg_area_mean": sum(fg_areas) / float(max(1, len(fg_areas))),
        "fg_area_min": min(fg_areas) if fg_areas else 0.0,
        "fg_area_max": max(fg_areas) if fg_areas else 0.0,
        "label_diversity_mean": sum(label_diversities) / float(max(1, len(label_diversities))),
        "label_diversity_min": min(label_diversities) if label_diversities else 0,
        "label_diversity_max": max(label_diversities) if label_diversities else 0,
        "local_z_histogram": local_z_histogram,
        "category_foreground_area_stats": payload["category_foreground_area_stats"].get(args.category),
    }
    save_json(output_dir / f"{args.category}_sample_stats.json", sample_stats)
    save_json(output_dir / f"{args.category}_centroid_stats.json", {"note": "centroid_label_argmax_grid is meaningful mainly where centroid_label_mass_grid / centroid_fg_mask_prob is high.", "label_mass_threshold": label_mass_threshold, "rows": centroid_rows})
    save_jsonl(output_dir / f"{args.category}_samples.jsonl", sample_rows)
    print(
        format_metric_line(
            "inspect-foreground-cache:",
            [("cache", str(cache_path)), ("category", args.category), ("num_samples", len(samples)), ("num_centroids", len(centroid_rows))],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
