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
