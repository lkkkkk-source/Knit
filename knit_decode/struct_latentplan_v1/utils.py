from __future__ import annotations

import json
from pathlib import Path

from knit_decode.parser_t_inverse.palette import OFFICIAL_PALETTE, infer_palette_mapping


def load_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    try:
        import yaml
    except ImportError as error:
        raise ImportError("PyYAML is required to load latentplan YAML configs. Install with `pip install pyyaml`.") from error
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config in {config_path}")
    return payload


def _require_torch() -> object:
    import importlib

    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required. Install with `pip install -e .[train]`.") from error


def ensure_palette_path(manifest_path: str | Path, palette_path: str | Path | None) -> Path:
    if palette_path is not None:
        return Path(palette_path)
    manifest_path = Path(manifest_path)
    inferred = manifest_path.parent / "palette_mapping.json"
    if not inferred.exists():
        infer_palette_mapping(manifest_path, inferred)
    return inferred


def save_label_map(mask: list[list[int]], output_path: Path, palette: tuple[tuple[int, int, int], ...] = OFFICIAL_PALETTE, scale: int = 1) -> None:
    try:
        from PIL import Image
    except Exception:
        output_path = output_path.with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(mask, ensure_ascii=False), encoding="utf-8")
        return

    height = len(mask)
    width = len(mask[0]) if height else 0
    image = Image.new("P", (width, height))
    for y_pos, row in enumerate(mask):
        for x_pos, class_id in enumerate(row):
            image.putpixel((x_pos, y_pos), int(class_id))
    flat_palette: list[int] = []
    for color in palette:
        flat_palette.extend(color)
    flat_palette.extend([0] * (768 - len(flat_palette)))
    image.putpalette(flat_palette)
    if scale > 1:
        image = image.resize((width * scale, height * scale), resample=Image.Resampling.NEAREST)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_binary_map(mask: list[list[int]] | list[list[bool]], output_path: Path, scale: int = 1) -> None:
    try:
        from PIL import Image
    except Exception:
        output_path = output_path.with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps([[1 if bool(value) else 0 for value in row] for row in mask], ensure_ascii=False), encoding="utf-8")
        return

    height = len(mask)
    width = len(mask[0]) if height else 0
    image = Image.new("L", (width, height))
    for y_pos, row in enumerate(mask):
        for x_pos, value in enumerate(row):
            image.putpixel((x_pos, y_pos), 255 if bool(value) else 0)
    if scale > 1:
        image = image.resize((width * scale, height * scale), resample=Image.Resampling.NEAREST)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _connected_components(mask: list[list[bool]]) -> tuple[int, int, tuple[int, int, int, int]]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    visited = [[False for _ in range(width)] for _ in range(height)]
    num_components = 0
    largest_area = 0
    largest_bbox = (0, 0, 0, 0)
    for y_pos in range(height):
        for x_pos in range(width):
            if not mask[y_pos][x_pos] or visited[y_pos][x_pos]:
                continue
            num_components += 1
            stack = [(y_pos, x_pos)]
            visited[y_pos][x_pos] = True
            area = 0
            min_y = max_y = y_pos
            min_x = max_x = x_pos
            while stack:
                cur_y, cur_x = stack.pop()
                area += 1
                min_y = min(min_y, cur_y)
                max_y = max(max_y, cur_y)
                min_x = min(min_x, cur_x)
                max_x = max(max_x, cur_x)
                for next_y, next_x in ((cur_y - 1, cur_x), (cur_y + 1, cur_x), (cur_y, cur_x - 1), (cur_y, cur_x + 1)):
                    if 0 <= next_y < height and 0 <= next_x < width and mask[next_y][next_x] and not visited[next_y][next_x]:
                        visited[next_y][next_x] = True
                        stack.append((next_y, next_x))
            if area > largest_area:
                largest_area = area
                largest_bbox = (min_x, min_y, max_x, max_y)
    return num_components, largest_area, largest_bbox


def compute_plan_statistics(grid20: list[list[int]], background_class_id: int, coarse_size: int, coarse_threshold: float) -> dict[str, object]:
    size = len(grid20)
    block = size // coarse_size
    fg20 = [[value != background_class_id for value in row] for row in grid20]
    fg_count = sum(1 for row in fg20 for value in row if value)
    total = max(1, size * size)
    fg_ratio = fg_count / float(total)

    o5: list[list[int]] = []
    c5: list[list[int]] = []
    coarse_hist = [0 for _ in range(17)]
    for y_pos in range(coarse_size):
        occ_row: list[int] = []
        cls_row: list[int] = []
        for x_pos in range(coarse_size):
            values: list[int] = []
            fg_values: list[int] = []
            for inner_y in range(y_pos * block, min(size, (y_pos + 1) * block)):
                for inner_x in range(x_pos * block, min(size, (x_pos + 1) * block)):
                    value = int(grid20[inner_y][inner_x])
                    values.append(value)
                    if value != background_class_id:
                        fg_values.append(value)
            fg_local_ratio = len(fg_values) / float(max(1, len(values)))
            if fg_local_ratio >= float(coarse_threshold) and fg_values:
                occ_row.append(1)
                counts: dict[int, int] = {}
                for value in fg_values:
                    counts[value] = counts.get(value, 0) + 1
                dominant = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
                cls_row.append(int(dominant))
            else:
                occ_row.append(0)
                cls_row.append(int(background_class_id))
            coarse_hist[int(cls_row[-1])] += 1
        o5.append(occ_row)
        c5.append(cls_row)

    class_counts = [0 for _ in range(17)]
    for row in grid20:
        for value in row:
            class_counts[int(value)] += 1
    r17 = [count / float(total) for count in class_counts]
    coarse_hist_norm = [count / float(max(1, coarse_size * coarse_size)) for count in coarse_hist]

    num_components, largest_area, largest_bbox = _connected_components(fg20)
    if largest_area > 0:
        min_x, min_y, max_x, max_y = largest_bbox
        bbox_w = max_x - min_x + 1
        bbox_h = max_y - min_y + 1
        cx = (min_x + max_x + 1) / (2.0 * size)
        cy = (min_y + max_y + 1) / (2.0 * size)
        bbox_stats = [
            min_x / float(size),
            min_y / float(size),
            max_x / float(size),
            max_y / float(size),
            cx,
            cy,
            largest_area / float(total),
            bbox_w / float(max(1, bbox_h)),
        ]
    else:
        bbox_stats = [0.0 for _ in range(8)]
    component_stats = [
        num_components / float(total),
        largest_area / float(max(1, fg_count)),
    ]
    descriptor = (
        [float(value) for row in o5 for value in row]
        + coarse_hist_norm
        + r17
        + [fg_ratio]
        + bbox_stats
        + component_stats
    )
    return {
        "fg20": fg20,
        "o5": o5,
        "c5": c5,
        "r17": r17,
        "fg_ratio": fg_ratio,
        "bbox_stats": bbox_stats,
        "component_stats": component_stats,
        "descriptor": descriptor,
        "num_components": num_components,
        "largest_component_ratio": component_stats[1],
    }


def _coarse_plan(grid20: list[list[int]], background_class_id: int, coarse_size: int, coarse_threshold: float) -> tuple[list[list[int]], list[list[int]]]:
    size = len(grid20)
    block = max(1, size // coarse_size)
    o: list[list[int]] = []
    c: list[list[int]] = []
    for y_pos in range(coarse_size):
        occ_row: list[int] = []
        cls_row: list[int] = []
        for x_pos in range(coarse_size):
            values: list[int] = []
            fg_values: list[int] = []
            for inner_y in range(y_pos * block, min(size, (y_pos + 1) * block)):
                for inner_x in range(x_pos * block, min(size, (x_pos + 1) * block)):
                    value = int(grid20[inner_y][inner_x])
                    values.append(value)
                    if value != background_class_id:
                        fg_values.append(value)
            fg_local_ratio = len(fg_values) / float(max(1, len(values)))
            if fg_local_ratio >= float(coarse_threshold) and fg_values:
                occ_row.append(1)
                counts: dict[int, int] = {}
                for value in fg_values:
                    counts[value] = counts.get(value, 0) + 1
                dominant = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
                cls_row.append(int(dominant))
            else:
                occ_row.append(0)
                cls_row.append(int(background_class_id))
        o.append(occ_row)
        c.append(cls_row)
    return o, c


def _projection_stats(values: list[float]) -> list[float]:
    if not values:
        return [0.0, 0.0, 0.0]
    mean_value = sum(values) / float(len(values))
    variance = sum((value - mean_value) ** 2 for value in values) / float(len(values))
    peaks = 0
    for index, value in enumerate(values):
        left = values[index - 1] if index > 0 else value
        right = values[index + 1] if index + 1 < len(values) else value
        if value > mean_value and value >= left and value >= right:
            peaks += 1
    return [mean_value, variance, peaks / float(max(1, len(values)))]


def _run_stats(lines: list[list[bool]]) -> list[float]:
    num_runs_list: list[float] = []
    mean_len_list: list[float] = []
    max_len_list: list[float] = []
    for line in lines:
        runs: list[int] = []
        current = 0
        for value in line:
            if value:
                current += 1
            elif current > 0:
                runs.append(current)
                current = 0
        if current > 0:
            runs.append(current)
        if runs:
            num_runs_list.append(float(len(runs)))
            mean_len_list.append(sum(runs) / float(len(runs)))
            max_len_list.append(float(max(runs)))
        else:
            num_runs_list.append(0.0)
            mean_len_list.append(0.0)
            max_len_list.append(0.0)
    return [
        sum(num_runs_list) / float(max(1, len(num_runs_list))),
        sum(mean_len_list) / float(max(1, len(mean_len_list))),
        sum(max_len_list) / float(max(1, len(max_len_list))),
    ]


def _adjacency_signature(grid20: list[list[int]], num_classes: int) -> list[float]:
    counts = [[0.0 for _ in range(num_classes)] for _ in range(num_classes)]
    height = len(grid20)
    width = len(grid20[0]) if height else 0
    total = 0.0
    for y_pos in range(height):
        for x_pos in range(width):
            src = int(grid20[y_pos][x_pos])
            for next_y, next_x in ((y_pos + 1, x_pos), (y_pos, x_pos + 1)):
                if 0 <= next_y < height and 0 <= next_x < width:
                    dst = int(grid20[next_y][next_x])
                    counts[src][dst] += 1.0
                    counts[dst][src] += 1.0
                    total += 2.0
    if total <= 0:
        return [0.0 for _ in range(num_classes * num_classes)]
    return [value / total for row in counts for value in row]


def _transition_2x2_stats(grid20: list[list[int]], background_class_id: int) -> list[float]:
    height = len(grid20)
    width = len(grid20[0]) if height else 0
    total = 0
    all_same = 0
    has_background = 0
    fg_mixed = 0
    unique_mean = 0.0
    diag_change = 0
    vertical_change = 0
    horizontal_change = 0
    for y_pos in range(max(0, height - 1)):
        for x_pos in range(max(0, width - 1)):
            block = [
                int(grid20[y_pos][x_pos]),
                int(grid20[y_pos][x_pos + 1]),
                int(grid20[y_pos + 1][x_pos]),
                int(grid20[y_pos + 1][x_pos + 1]),
            ]
            total += 1
            unique = set(block)
            unique_mean += len(unique)
            if len(unique) == 1:
                all_same += 1
            if background_class_id in unique:
                has_background += 1
            fg_unique = {value for value in unique if value != background_class_id}
            if len(fg_unique) >= 2:
                fg_mixed += 1
            if block[0] != block[3] or block[1] != block[2]:
                diag_change += 1
            if block[0] != block[2] or block[1] != block[3]:
                vertical_change += 1
            if block[0] != block[1] or block[2] != block[3]:
                horizontal_change += 1
    denom = float(max(1, total))
    return [
        all_same / denom,
        has_background / denom,
        fg_mixed / denom,
        unique_mean / denom,
        diag_change / denom,
        vertical_change / denom,
        horizontal_change / denom,
    ]


def compute_grammar_descriptor(grid20: list[list[int]], background_class_id: int, coarse_threshold: float, num_classes: int = 17) -> dict[str, object]:
    size = len(grid20)
    fg20 = [[value != background_class_id for value in row] for row in grid20]
    o5, c5 = _coarse_plan(grid20, background_class_id, coarse_size=5, coarse_threshold=coarse_threshold)
    o10, c10 = _coarse_plan(grid20, background_class_id, coarse_size=10, coarse_threshold=coarse_threshold)
    row_projection = [sum(1 for value in row if value != background_class_id) / float(max(1, len(row))) for row in grid20]
    col_projection = [
        sum(1 for y_pos in range(size) if grid20[y_pos][x_pos] != background_class_id) / float(max(1, size))
        for x_pos in range(size)
    ]
    row_run_stats = _run_stats(fg20)
    col_run_stats = _run_stats([[fg20[y_pos][x_pos] for y_pos in range(size)] for x_pos in range(size)])
    adjacency_signature = _adjacency_signature(grid20, num_classes=num_classes)
    transition_2x2_stats = _transition_2x2_stats(grid20, background_class_id=background_class_id)
    vertical_pairs = 0
    vertical_same = 0
    horizontal_pairs = 0
    horizontal_same = 0
    for y_pos in range(size):
        for x_pos in range(size):
            if y_pos + 1 < size:
                vertical_pairs += 1
                if fg20[y_pos][x_pos] and fg20[y_pos + 1][x_pos]:
                    vertical_same += 1
            if x_pos + 1 < size:
                horizontal_pairs += 1
                if fg20[y_pos][x_pos] and fg20[y_pos][x_pos + 1]:
                    horizontal_same += 1
    vertical_continuity = vertical_same / float(max(1, vertical_pairs))
    horizontal_continuity = horizontal_same / float(max(1, horizontal_pairs))
    left = sum(1 for y_pos in range(size) for x_pos in range(size // 2) if fg20[y_pos][x_pos] == fg20[y_pos][size - 1 - x_pos])
    symmetry_score = left / float(max(1, size * (size // 2)))
    center_start = int(size * 0.3)
    center_end = int(size * 0.7)
    center_fg = sum(1 for y_pos in range(size) for x_pos in range(center_start, center_end) if fg20[y_pos][x_pos])
    total_fg = sum(1 for row in fg20 for value in row if value)
    center_band_score = center_fg / float(max(1, total_fg))
    stripe_score = _projection_stats(col_projection)
    grammar_signature = row_run_stats + col_run_stats + transition_2x2_stats + [
        vertical_continuity,
        horizontal_continuity,
        symmetry_score,
        center_band_score,
    ] + stripe_score
    descriptor = (
        [float(value) for row in o5 for value in row]
        + [float(value) for row in o10 for value in row]
        + row_projection
        + col_projection
        + row_run_stats
        + col_run_stats
        + adjacency_signature
        + transition_2x2_stats
        + [vertical_continuity, horizontal_continuity, symmetry_score, center_band_score]
        + stripe_score
    )
    descriptor_slices = {
        "o5": [0, 25],
        "o10": [25, 125],
        "row_projection": [125, 145],
        "col_projection": [145, 165],
        "row_run_stats": [165, 168],
        "col_run_stats": [168, 171],
        "adjacency_signature": [171, 460],
        "transition_2x2_stats": [460, 467],
        "grammar_signature_tail": [467, len(descriptor)],
    }
    return {
        "o5": o5,
        "c5": c5,
        "o10": o10,
        "c10": c10,
        "row_projection": row_projection,
        "col_projection": col_projection,
        "row_run_stats": row_run_stats,
        "col_run_stats": col_run_stats,
        "adjacency_signature": adjacency_signature,
        "transition_2x2_stats": transition_2x2_stats,
        "vertical_continuity": vertical_continuity,
        "horizontal_continuity": horizontal_continuity,
        "symmetry_score": symmetry_score,
        "center_band_score": center_band_score,
        "stripe_score": stripe_score,
        "grammar_signature": grammar_signature,
        "descriptor": descriptor,
        "descriptor_slices": descriptor_slices,
    }


def compute_category_fg_stats(items: list[dict[str, object]], categories: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for category in categories:
        values = sorted(float(item["fg_ratio"]) for item in items if item["category"] == category)
        if not values:
            continue
        def _q(q: float) -> float:
            index = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
            return values[index]
        mean_value = sum(values) / float(len(values))
        std_value = (sum((value - mean_value) ** 2 for value in values) / float(max(1, len(values)))) ** 0.5
        q05 = _q(0.05)
        q95 = _q(0.95)
        stats[category] = {
            "count": float(len(values)),
            "mean": mean_value,
            "std": std_value,
            "q01": _q(0.01),
            "q05": q05,
            "q10": _q(0.10),
            "q50": _q(0.50),
            "q90": _q(0.90),
            "q95": q95,
            "q99": _q(0.99),
            "valid_low": max(0.02, q05 - 0.02),
            "valid_high": min(0.95, q95 + 0.02),
        }
    return stats


def compute_category_occ_stats(items: list[dict[str, object]], categories: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for category in categories:
        o5_counts = sorted(sum(int(value) for row in item["o5"] for value in row) for item in items if item["category"] == category)
        o10_counts = sorted(sum(int(value) for row in item["o10"] for value in row) for item in items if item["category"] == category)
        if not o5_counts or not o10_counts:
            continue
        def _q(values: list[int], q: float) -> float:
            index = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
            return float(values[index])
        q05_o5 = _q(o5_counts, 0.05)
        q95_o5 = _q(o5_counts, 0.95)
        q05_o10 = _q(o10_counts, 0.05)
        q95_o10 = _q(o10_counts, 0.95)
        stats[category] = {
            "o5_fg_cells_q05": q05_o5,
            "o5_fg_cells_q95": q95_o5,
            "o10_fg_cells_q05": q05_o10,
            "o10_fg_cells_q95": q95_o10,
            "valid_o5_min_cells": max(0.0, q05_o5 - 1.0),
            "valid_o5_max_cells": min(25.0, q95_o5 + 1.0),
            "valid_o10_min_cells": max(0.0, q05_o10 - 2.0),
            "valid_o10_max_cells": min(100.0, q95_o10 + 2.0),
        }
    return stats


def largest_component_ratio(mask: list[list[bool]]) -> float:
    total = sum(1 for row in mask for value in row if value)
    if total <= 0:
        return 0.0
    _, largest_area, _ = _connected_components(mask)
    return largest_area / float(total)


def connected_components_count(mask: list[list[bool]]) -> int:
    count, _, _ = _connected_components(mask)
    return count


def count_tiny_islands(mask: list[list[bool]], max_area: int = 2) -> int:
    height = len(mask)
    width = len(mask[0]) if height else 0
    visited = [[False for _ in range(width)] for _ in range(height)]
    tiny = 0
    for y_pos in range(height):
        for x_pos in range(width):
            if not mask[y_pos][x_pos] or visited[y_pos][x_pos]:
                continue
            stack = [(y_pos, x_pos)]
            visited[y_pos][x_pos] = True
            area = 0
            while stack:
                cur_y, cur_x = stack.pop()
                area += 1
                for next_y, next_x in ((cur_y - 1, cur_x), (cur_y + 1, cur_x), (cur_y, cur_x - 1), (cur_y, cur_x + 1)):
                    if 0 <= next_y < height and 0 <= next_x < width and mask[next_y][next_x] and not visited[next_y][next_x]:
                        visited[next_y][next_x] = True
                        stack.append((next_y, next_x))
            if area <= max_area:
                tiny += 1
    return tiny


def structure_metrics(pred_mask: list[list[int]], tgt_mask: list[list[int]], background_class_id: int, num_classes: int) -> dict[str, float]:
    pred_fg = [[value != background_class_id for value in row] for row in pred_mask]
    tgt_fg = [[value != background_class_id for value in row] for row in tgt_mask]
    pred_fg_count = sum(1 for row in pred_fg for value in row if value)
    tgt_fg_count = sum(1 for row in tgt_fg for value in row if value)
    intersection = 0
    union = 0
    agreement = 0
    total = max(1, len(pred_mask) * len(pred_mask[0]))
    pred_counts = [0 for _ in range(num_classes)]
    tgt_counts = [0 for _ in range(num_classes)]
    for y_pos in range(len(pred_mask)):
        for x_pos in range(len(pred_mask[0])):
            pred_value = int(pred_mask[y_pos][x_pos])
            tgt_value = int(tgt_mask[y_pos][x_pos])
            pred_counts[pred_value] += 1
            tgt_counts[tgt_value] += 1
            if pred_value == tgt_value:
                agreement += 1
            pred_fg_value = pred_value != background_class_id
            tgt_fg_value = tgt_value != background_class_id
            if pred_fg_value and tgt_fg_value:
                intersection += 1
            if pred_fg_value or tgt_fg_value:
                union += 1
    return {
        "pixel_accuracy": agreement / float(total),
        "foreground_iou": 0.0 if union <= 0 else intersection / float(union),
        "foreground_ratio_error": abs(pred_fg_count - tgt_fg_count) / float(total),
        "class_count_l1": sum(abs(pred - tgt) for pred, tgt in zip(pred_counts, tgt_counts)) / float(total),
        "connected_components": float(connected_components_count(pred_fg)),
        "largest_component_ratio": largest_component_ratio(pred_fg),
        "tiny_island_count": float(count_tiny_islands(pred_fg)),
        "all_background": 1.0 if pred_fg_count <= 0 else 0.0,
    }


def upsample_nearest(grid: list[list[int]], target_size: int) -> list[list[int]]:
    src_h = len(grid)
    src_w = len(grid[0]) if src_h else 0
    out: list[list[int]] = []
    for y_pos in range(target_size):
        row: list[int] = []
        src_y = min(src_h - 1, (y_pos * src_h) // target_size)
        for x_pos in range(target_size):
            src_x = min(src_w - 1, (x_pos * src_w) // target_size)
            row.append(int(grid[src_y][src_x]))
        out.append(row)
    return out


def print_progress(stage: str, current: int, total: int, extra: str = "") -> None:
    width = 30
    ratio = 0.0 if total <= 0 else current / total
    filled = min(width, int(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    suffix = f" {extra}" if extra else ""
    print(f"\r[{stage}] [{bar}] {current}/{total}{suffix}", end="", flush=True)


def finish_progress() -> None:
    print(flush=True)


def format_metric_line(prefix: str, items: list[tuple[str, object]]) -> str:
    parts = [prefix]
    for key, value in items:
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def sample_top_p(logits: object, temperature: float = 1.0, top_p: float = 0.9) -> object:
    torch = _require_torch()
    functional = __import__("importlib").import_module("torch.nn.functional")
    scaled = logits / max(float(temperature), 1e-6)
    probs = functional.softmax(scaled, dim=-1)
    sorted_probs, sorted_indices = getattr(torch, "sort")(probs, dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    keep = cumulative <= float(top_p)
    keep[..., 0] = True
    filtered = getattr(torch, "where")(keep, sorted_probs, getattr(torch, "zeros_like")(sorted_probs))
    filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    sampled_sorted = getattr(torch, "multinomial")(filtered, num_samples=1).squeeze(-1)
    return sorted_indices.gather(-1, sampled_sorted.unsqueeze(-1)).squeeze(-1)
