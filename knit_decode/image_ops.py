from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
from typing import cast

from PIL import Image, ImageChops, ImageDraw

from .legend import Legend, RGB, rgb_to_hex


@dataclass(frozen=True)
class CroppedImage:
    image: Image.Image
    crop_box: tuple[int, int, int, int]
    background_rgb: RGB


@dataclass(frozen=True)
class QuantizedImage:
    image: Image.Image
    mean_distance: float
    max_distance: float


@dataclass(frozen=True)
class GridSpec:
    cell_width: int
    cell_height: int
    columns: int
    rows: int
    usable_width: int
    usable_height: int


@dataclass(frozen=True)
class DecodedCell:
    row: int
    column: int
    color_rgb: RGB
    color_hex: str
    action_id: int | None
    candidate_ids: tuple[int, ...]
    purity: float
    pixel_count: int


@dataclass(frozen=True)
class DecodedGrid:
    grid_spec: GridSpec
    cells: tuple[tuple[DecodedCell, ...], ...]
    ambiguous_cell_count: int
    average_cell_purity: float


def _get_rgb(image: Image.Image, x_pos: int, y_pos: int) -> RGB:
    value = image.getpixel((x_pos, y_pos))
    if (
        isinstance(value, tuple)
        and len(value) == 3
        and all(isinstance(channel, int) for channel in value)
    ):
        red, green, blue = value
        return (red, green, blue)
    raise ValueError(f"Expected RGB pixel, received {value!r}")


def load_rgb_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _squared_distance(left: RGB, right: RGB) -> int:
    return sum((left[index] - right[index]) ** 2 for index in range(3))


def infer_background_color(image: Image.Image) -> RGB:
    width, height = image.size
    border_pixels: list[RGB] = []
    for x_pos in range(width):
        border_pixels.append(_get_rgb(image, x_pos, 0))
        border_pixels.append(_get_rgb(image, x_pos, height - 1))
    for y_pos in range(1, height - 1):
        border_pixels.append(_get_rgb(image, 0, y_pos))
        border_pixels.append(_get_rgb(image, width - 1, y_pos))
    return Counter(border_pixels).most_common(1)[0][0]


def crop_active_region(image: Image.Image, tolerance: int = 36) -> CroppedImage:
    width, height = image.size
    background_rgb = infer_background_color(image)
    threshold = tolerance * tolerance
    active_x: list[int] = []
    active_y: list[int] = []
    for y_pos in range(height):
        for x_pos in range(width):
            if _squared_distance(_get_rgb(image, x_pos, y_pos), background_rgb) > threshold:
                active_x.append(x_pos)
                active_y.append(y_pos)

    if not active_x or not active_y:
        return CroppedImage(image=image.copy(), crop_box=(0, 0, width, height), background_rgb=background_rgb)

    crop_box = (min(active_x), min(active_y), max(active_x) + 1, max(active_y) + 1)
    return CroppedImage(image=image.crop(crop_box), crop_box=crop_box, background_rgb=background_rgb)


def quantize_to_legend(image: Image.Image, legend: Legend) -> QuantizedImage:
    palette = legend.unique_colors
    cache: dict[RGB, tuple[RGB, int]] = {}
    output = Image.new("RGB", image.size)
    total_distance = 0
    max_distance = 0

    for y_pos in range(image.height):
        for x_pos in range(image.width):
            source_rgb = _get_rgb(image, x_pos, y_pos)
            if source_rgb not in cache:
                best_rgb = min(palette, key=lambda palette_rgb: _squared_distance(source_rgb, palette_rgb))
                best_distance = _squared_distance(source_rgb, best_rgb)
                cache[source_rgb] = (best_rgb, best_distance)
            target_rgb, best_distance = cache[source_rgb]
            output.putpixel((x_pos, y_pos), target_rgb)
            total_distance += best_distance
            max_distance = max(max_distance, best_distance)

    pixel_count = max(1, image.width * image.height)
    return QuantizedImage(
        image=output,
        mean_distance=math.sqrt(total_distance / pixel_count),
        max_distance=math.sqrt(max_distance),
    )


def _cell_majority_ratio(image: Image.Image, cell_width: int, cell_height: int) -> tuple[float, int, int]:
    columns = image.width // cell_width
    rows = image.height // cell_height
    if columns < 1 or rows < 1:
        return 0.0, 0, 0

    ratios: list[float] = []
    for row_index in range(rows):
        for column_index in range(columns):
            counts: Counter[RGB] = Counter()
            for y_pos in range(row_index * cell_height, (row_index + 1) * cell_height):
                for x_pos in range(column_index * cell_width, (column_index + 1) * cell_width):
                    counts[_get_rgb(image, x_pos, y_pos)] += 1
            top_count = counts.most_common(1)[0][1]
            ratios.append(top_count / float(cell_width * cell_height))

    return sum(ratios) / len(ratios), columns, rows


def _collect_run_lengths(image: Image.Image, axis: str) -> Counter[int]:
    run_lengths: Counter[int] = Counter()
    if axis == "horizontal":
        for y_pos in range(image.height):
            current_length = 1
            for x_pos in range(1, image.width):
                if _get_rgb(image, x_pos, y_pos) == _get_rgb(image, x_pos - 1, y_pos):
                    current_length += 1
                else:
                    run_lengths[current_length] += 1
                    current_length = 1
            run_lengths[current_length] += 1
        return run_lengths

    if axis == "vertical":
        for x_pos in range(image.width):
            current_length = 1
            for y_pos in range(1, image.height):
                if _get_rgb(image, x_pos, y_pos) == _get_rgb(image, x_pos, y_pos - 1):
                    current_length += 1
                else:
                    run_lengths[current_length] += 1
                    current_length = 1
            run_lengths[current_length] += 1
        return run_lengths

    raise ValueError(f"Unsupported axis: {axis}")


def _dominant_run_length(image: Image.Image, axis: str) -> int | None:
    run_lengths = _collect_run_lengths(image, axis)
    if not run_lengths:
        return None

    candidates = [
        (length, frequency)
        for length, frequency in run_lengths.items()
        if length > 0 and frequency > 0
    ]
    if not candidates:
        return None

    best_length, _ = max(candidates, key=lambda item: (item[1], -item[0]))
    return best_length


def infer_grid_spec(
    image: Image.Image,
    cell_width: int | None = None,
    cell_height: int | None = None,
) -> GridSpec:
    inferred_width = cell_width or _dominant_run_length(image, "horizontal")
    inferred_height = cell_height or _dominant_run_length(image, "vertical")

    if inferred_width and inferred_height:
        mean_purity, columns, rows = _cell_majority_ratio(image, inferred_width, inferred_height)
        if columns >= 1 and rows >= 1 and mean_purity >= 0.95:
            return GridSpec(
                cell_width=inferred_width,
                cell_height=inferred_height,
                columns=columns,
                rows=rows,
                usable_width=columns * inferred_width,
                usable_height=rows * inferred_height,
            )

    width_candidates = [cell_width] if cell_width else list(range(1, min(64, image.width) + 1))
    height_candidates = [cell_height] if cell_height else list(range(1, min(64, image.height) + 1))

    best_spec: GridSpec | None = None
    best_score = float("-inf")
    for candidate_width in width_candidates:
        if candidate_width is None or candidate_width <= 0:
            continue
        for candidate_height in height_candidates:
            if candidate_height is None or candidate_height <= 0:
                continue
            mean_purity, columns, rows = _cell_majority_ratio(image, candidate_width, candidate_height)
            if columns < 1 or rows < 1:
                continue
            if cell_width is None and columns < 2:
                continue
            if cell_height is None and rows < 2:
                continue
            score = mean_purity - 0.001 * (candidate_width + candidate_height)
            if score > best_score:
                best_score = score
                best_spec = GridSpec(
                    cell_width=candidate_width,
                    cell_height=candidate_height,
                    columns=columns,
                    rows=rows,
                    usable_width=columns * candidate_width,
                    usable_height=rows * candidate_height,
                )

    if best_spec is not None:
        return best_spec

    fallback_width = cell_width or max(1, image.width)
    fallback_height = cell_height or max(1, image.height)
    columns = max(1, image.width // fallback_width)
    rows = max(1, image.height // fallback_height)
    return GridSpec(
        cell_width=fallback_width,
        cell_height=fallback_height,
        columns=columns,
        rows=rows,
        usable_width=columns * fallback_width,
        usable_height=rows * fallback_height,
    )


def decode_grid(image: Image.Image, grid_spec: GridSpec, legend: Legend) -> DecodedGrid:
    usable_image = image.crop((0, 0, grid_spec.usable_width, grid_spec.usable_height))
    rows: list[tuple[DecodedCell, ...]] = []
    purities: list[float] = []
    ambiguous_cells = 0
    for row_index in range(grid_spec.rows):
        row_cells: list[DecodedCell] = []
        for column_index in range(grid_spec.columns):
            counts: Counter[RGB] = Counter()
            for y_pos in range(row_index * grid_spec.cell_height, (row_index + 1) * grid_spec.cell_height):
                for x_pos in range(column_index * grid_spec.cell_width, (column_index + 1) * grid_spec.cell_width):
                    counts[_get_rgb(usable_image, x_pos, y_pos)] += 1
            top_rgb, top_count = counts.most_common(1)[0]
            candidate_ids = tuple(legend.candidate_ids_for_color(top_rgb))
            purity = top_count / float(grid_spec.cell_width * grid_spec.cell_height)
            action_id = candidate_ids[0] if len(candidate_ids) == 1 else None
            if action_id is None:
                ambiguous_cells += 1
            purities.append(purity)
            row_cells.append(
                DecodedCell(
                    row=row_index,
                    column=column_index,
                    color_rgb=top_rgb,
                    color_hex=rgb_to_hex(top_rgb),
                    action_id=action_id,
                    candidate_ids=candidate_ids,
                    purity=purity,
                    pixel_count=grid_spec.cell_width * grid_spec.cell_height,
                )
            )
        rows.append(tuple(row_cells))
    return DecodedGrid(
        grid_spec=grid_spec,
        cells=tuple(rows),
        ambiguous_cell_count=ambiguous_cells,
        average_cell_purity=sum(purities) / len(purities) if purities else 0.0,
    )


def reconstruct_grid(decoded_grid: DecodedGrid) -> Image.Image:
    width = decoded_grid.grid_spec.usable_width
    height = decoded_grid.grid_spec.usable_height
    image = Image.new("RGB", (width, height))
    for row in decoded_grid.cells:
        for cell in row:
            for y_pos in range(
                cell.row * decoded_grid.grid_spec.cell_height,
                (cell.row + 1) * decoded_grid.grid_spec.cell_height,
            ):
                for x_pos in range(
                    cell.column * decoded_grid.grid_spec.cell_width,
                    (cell.column + 1) * decoded_grid.grid_spec.cell_width,
                ):
                    image.putpixel((x_pos, y_pos), cell.color_rgb)
    return image


def render_grid_overlay(image: Image.Image, grid_spec: GridSpec) -> Image.Image:
    overlay = image.crop((0, 0, grid_spec.usable_width, grid_spec.usable_height)).copy()
    draw = ImageDraw.Draw(overlay)
    for x_pos in range(0, grid_spec.usable_width + 1, grid_spec.cell_width):
        draw.line((x_pos, 0, x_pos, grid_spec.usable_height), fill=(255, 255, 255), width=1)
    for y_pos in range(0, grid_spec.usable_height + 1, grid_spec.cell_height):
        draw.line((0, y_pos, grid_spec.usable_width, y_pos), fill=(255, 255, 255), width=1)
    return overlay


def build_diff_image(reference: Image.Image, reconstruction: Image.Image) -> Image.Image:
    usable_reference = reference.crop((0, 0, reconstruction.width, reconstruction.height))
    diff = ImageChops.difference(usable_reference, reconstruction)
    output = Image.new("RGB", reconstruction.size)
    for y_pos in range(reconstruction.height):
        for x_pos in range(reconstruction.width):
            if _get_rgb(diff, x_pos, y_pos) == (0, 0, 0):
                source_rgb = _get_rgb(usable_reference, x_pos, y_pos)
                dimmed_rgb = cast(RGB, tuple(max(24, channel // 3) for channel in source_rgb))
                output.putpixel((x_pos, y_pos), dimmed_rgb)
            else:
                output.putpixel((x_pos, y_pos), (255, 0, 0))
    return output


def format_action_grid(decoded_grid: DecodedGrid) -> str:
    lines: list[str] = []
    for row in decoded_grid.cells:
        tokens: list[str] = []
        for cell in row:
            if cell.action_id is not None:
                tokens.append(str(cell.action_id))
            else:
                tokens.append("|".join(str(candidate_id) for candidate_id in cell.candidate_ids))
        lines.append("\t".join(tokens))
    return "\n".join(lines)
