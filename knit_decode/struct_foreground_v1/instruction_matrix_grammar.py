from __future__ import annotations

import importlib
import math
from collections import Counter
from pathlib import Path
from typing import Any


INSTRUCTION_GRAMMAR_SCHEMA_VERSION = "foreground_v1_instruction_matrix_grammar_prior_v1"
CANONICAL_SIZE = 20
NUM_LABELS = 16
IGNORE_INDEX = -100


DEFAULT_INSTRUCTION_GRAMMAR_CONFIG: dict[str, Any] = {
    "enabled": False,
    "schema_version": INSTRUCTION_GRAMMAR_SCHEMA_VERSION,
    "mode_count_default": 8,
    "mode_count_by_category": {},
    "descriptor_weights": {
        "support": 0.5,
        "label_hist": 0.8,
        "transition": 1.2,
        "motif2": 1.2,
        "motif3": 0.6,
        "row_program": 1.0,
        "col_program": 1.0,
        "directional_shift": 1.5,
        "occupancy": 0.8,
        "component": 0.4,
    },
    "motif2": {"enabled": True, "top_k": 128, "include_background": True},
    "motif3": {"enabled": True, "top_k": 128, "include_background": True},
    "row_program": {"enabled": True, "top_k": 128, "max_program_len": 20},
    "col_program": {"enabled": True, "top_k": 128, "max_program_len": 20},
    "directional_shift": {
        "enabled": True,
        "shift_clip": 3,
        "labels": "all",
        "apply_to_categories": ["Move1", "Move2", "Cable1", "Cable2"],
    },
    "clustering": {
        "method": "kmeans",
        "random_state": 0,
        "n_init": 10,
        "max_iter": 300,
        "min_samples_per_mode": 2,
        "reduce_modes_if_needed": True,
        "normalize_descriptor_blocks": True,
    },
    "aggregation": {"fg_threshold": 0.05, "eps": 1.0e-8, "save_float16": False},
    "quality": {
        "min_effective_modes_default": 2,
        "min_effective_mode_fraction_default": 0.5,
        "min_mode_samples": 1,
        "min_mean_fg_area": 0.005,
        "max_empty_mode_fraction": 0.5,
        "max_label_collapse_fraction_default": 0.75,
        "require_directional_non_degenerate_for": ["Move1"],
    },
    "strategy_by_category": {},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _merged_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = _deep_merge(DEFAULT_INSTRUCTION_GRAMMAR_CONFIG, {})
    if isinstance(config, dict):
        merged = _deep_merge(merged, config)
    merged["schema_version"] = str(merged.get("schema_version", INSTRUCTION_GRAMMAR_SCHEMA_VERSION))
    return merged


def _require_numpy() -> Any:
    try:
        return importlib.import_module("numpy")
    except ImportError as error:
        raise ImportError("NumPy is required for instruction matrix grammar prior building.") from error


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required to serialize instruction matrix grammar prior tensors.") from error


def _require_sklearn_cluster() -> Any:
    try:
        return importlib.import_module("sklearn.cluster")
    except ImportError as error:
        raise ImportError("scikit-learn is required for instruction matrix grammar mode clustering.") from error


def _as_grid(value: Any, *, context: str) -> list[list[int]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or len(value) != CANONICAL_SIZE:
        raise ValueError(f"{context} must have shape [20,20].")
    grid: list[list[int]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != CANONICAL_SIZE:
            raise ValueError(f"{context} row {row_index} must have 20 columns.")
        out_row = []
        for value in row:
            label = int(value)
            if label == IGNORE_INDEX:
                label = 0
            if not 0 <= label <= NUM_LABELS:
                raise ValueError(f"{context} contains invalid label {label}; expected 0..16 or ignore.")
            out_row.append(label)
        grid.append(out_row)
    return grid


def _raw_from_item(item: dict[str, Any]) -> list[list[int]]:
    return _as_grid(item.get("fg_y20"), context=f"item[{item.get('sample_id')}].fg_y20")


def _mask_from_raw(raw: list[list[int]]) -> list[list[int]]:
    return [[1 if 1 <= int(value) <= NUM_LABELS else 0 for value in row] for row in raw]


def _entropy_from_counts(counts: Counter[Any]) -> float:
    total = float(sum(int(value) for value in counts.values()))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        prob = float(value) / total
        if prob > 0.0:
            entropy -= prob * math.log(prob + 1.0e-12)
    return float(entropy)


def _entropy_from_probs(values: list[float]) -> float:
    return float(-sum(float(value) * math.log(float(value) + 1.0e-12) for value in values if float(value) > 0.0))


def _normalize_distribution(values: list[float], eps: float = 1.0e-8) -> list[float]:
    total = float(sum(max(0.0, float(value)) for value in values))
    if total <= eps:
        return [0.0 for _ in values]
    return [max(0.0, float(value)) / total for value in values]


def _l2_normalize(values: list[float], eps: float = 1.0e-8) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in values))
    if norm <= eps:
        return [0.0 for _ in values]
    return [float(value) / norm for value in values]


def _vector_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(sum(values) / len(values)), "min": float(min(values)), "max": float(max(values))}


def _top_counter(counter: Counter[Any], top_k: int) -> list[dict[str, Any]]:
    rows = []
    for key, count in counter.most_common(max(0, int(top_k))):
        rows.append({"program": list(key) if isinstance(key, tuple) else key, "count": int(count)})
    return rows


def _quantile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(float(value) for value in values)
    if len(vals) == 1:
        return vals[0]
    pos = float(p) * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def _bbox(mask: list[list[int]]) -> tuple[int, int, int, int] | None:
    coords = [(y, x) for y in range(CANONICAL_SIZE) for x in range(CANONICAL_SIZE) if int(mask[y][x]) > 0]
    if not coords:
        return None
    ys = [coord[0] for coord in coords]
    xs = [coord[1] for coord in coords]
    return min(ys), max(ys), min(xs), max(xs)


def _components(mask: list[list[int]]) -> tuple[int, float, int, float, float]:
    visited = [[False for _ in range(CANONICAL_SIZE)] for _ in range(CANONICAL_SIZE)]
    sizes: list[int] = []
    for y_pos in range(CANONICAL_SIZE):
        for x_pos in range(CANONICAL_SIZE):
            if visited[y_pos][x_pos] or int(mask[y_pos][x_pos]) <= 0:
                continue
            stack = [(y_pos, x_pos)]
            visited[y_pos][x_pos] = True
            size = 0
            while stack:
                y_cur, x_cur = stack.pop()
                size += 1
                for y_next, x_next in ((y_cur - 1, x_cur), (y_cur + 1, x_cur), (y_cur, x_cur - 1), (y_cur, x_cur + 1)):
                    if 0 <= y_next < CANONICAL_SIZE and 0 <= x_next < CANONICAL_SIZE and not visited[y_next][x_next] and int(mask[y_next][x_next]) > 0:
                        visited[y_next][x_next] = True
                        stack.append((y_next, x_next))
            sizes.append(size)
    if not sizes:
        return 0, 0.0, 0, 0.0, 0.0
    total = float(sum(sizes))
    return int(len(sizes)), float(max(sizes) / total), int(sum(1 for size in sizes if size <= 2)), float(total / len(sizes)), float(max(sizes))


def _patch_key(raw: list[list[int]], y_pos: int, x_pos: int, size: int) -> tuple[int, ...]:
    return tuple(int(raw[y_pos + dy][x_pos + dx]) for dy in range(size) for dx in range(size))


def _motif_counts(raw: list[list[int]], size: int, include_background: bool) -> Counter[tuple[int, ...]]:
    counts: Counter[tuple[int, ...]] = Counter()
    for y_pos in range(CANONICAL_SIZE - size + 1):
        for x_pos in range(CANONICAL_SIZE - size + 1):
            key = _patch_key(raw, y_pos, x_pos, size)
            if include_background or any(value > 0 for value in key):
                counts[key] += 1
    return counts


def _program_counts(raw: list[list[int]], axis: str) -> tuple[Counter[tuple[int, ...]], list[float], list[float], list[float]]:
    counts: Counter[tuple[int, ...]] = Counter()
    nonzero: list[float] = []
    span_start: list[float] = []
    span_end: list[float] = []
    for index in range(CANONICAL_SIZE):
        seq = [int(raw[index][x]) for x in range(CANONICAL_SIZE)] if axis == "row" else [int(raw[y][index]) for y in range(CANONICAL_SIZE)]
        nz = [value for value in seq if value > 0]
        nonzero.append(float(len(nz)) / float(CANONICAL_SIZE))
        if nz:
            active = [pos for pos, value in enumerate(seq) if value > 0]
            span_start.append(float(min(active)) / float(CANONICAL_SIZE - 1))
            span_end.append(float(max(active)) / float(CANONICAL_SIZE - 1))
            counts[tuple(nz)] += 1
        else:
            span_start.append(0.0)
            span_end.append(0.0)
            counts[tuple()] += 1
    return counts, nonzero, span_start, span_end


def _transition_descriptor(raw: list[list[int]]) -> dict[str, Any]:
    h = [[0.0 for _ in range(NUM_LABELS)] for _ in range(NUM_LABELS)]
    v = [[0.0 for _ in range(NUM_LABELS)] for _ in range(NUM_LABELS)]
    for y_pos in range(CANONICAL_SIZE):
        for x_pos in range(CANONICAL_SIZE):
            a = int(raw[y_pos][x_pos])
            if not 1 <= a <= NUM_LABELS:
                continue
            if x_pos + 1 < CANONICAL_SIZE:
                b = int(raw[y_pos][x_pos + 1])
                if 1 <= b <= NUM_LABELS:
                    h[a - 1][b - 1] += 1.0
            if y_pos + 1 < CANONICAL_SIZE:
                b = int(raw[y_pos + 1][x_pos])
                if 1 <= b <= NUM_LABELS:
                    v[a - 1][b - 1] += 1.0
    h_flat = [value for row in h for value in row]
    v_flat = [value for row in v for value in row]
    h_norm = _normalize_distribution(h_flat)
    v_norm = _normalize_distribution(v_flat)
    same_h = sum(h_norm[index * NUM_LABELS + index] for index in range(NUM_LABELS))
    same_v = sum(v_norm[index * NUM_LABELS + index] for index in range(NUM_LABELS))
    h_off = [h_norm[a * NUM_LABELS + b] for a in range(NUM_LABELS) for b in range(NUM_LABELS) if a != b]
    v_off = [v_norm[a * NUM_LABELS + b] for a in range(NUM_LABELS) for b in range(NUM_LABELS) if a != b]
    return {
        "transition_h": h_norm,
        "transition_v": v_norm,
        "transition_h_off": _normalize_distribution(h_off),
        "transition_v_off": _normalize_distribution(v_off),
        "same_label_h": float(same_h),
        "same_label_v": float(same_v),
        "diff_label_h": float(1.0 - same_h),
        "diff_label_v": float(1.0 - same_v),
    }


def _occupancy_descriptor(mask: list[list[int]]) -> dict[str, Any]:
    h = [[0.0, 0.0], [0.0, 0.0]]
    v = [[0.0, 0.0], [0.0, 0.0]]
    for y_pos in range(CANONICAL_SIZE):
        for x_pos in range(CANONICAL_SIZE):
            a = 1 if int(mask[y_pos][x_pos]) > 0 else 0
            if x_pos + 1 < CANONICAL_SIZE:
                b = 1 if int(mask[y_pos][x_pos + 1]) > 0 else 0
                h[a][b] += 1.0
            if y_pos + 1 < CANONICAL_SIZE:
                b = 1 if int(mask[y_pos + 1][x_pos]) > 0 else 0
                v[a][b] += 1.0
    return {"occupancy_h": _normalize_distribution([value for row in h for value in row]), "occupancy_v": _normalize_distribution([value for row in v for value in row])}


def _directional_descriptor(raw: list[list[int]], mask: list[list[int]], shift_clip: int) -> dict[str, Any]:
    diag_dr = [[0.0 for _ in range(NUM_LABELS)] for _ in range(NUM_LABELS)]
    diag_dl = [[0.0 for _ in range(NUM_LABELS)] for _ in range(NUM_LABELS)]
    for y_pos in range(CANONICAL_SIZE - 1):
        for x_pos in range(CANONICAL_SIZE):
            a = int(raw[y_pos][x_pos])
            if not 1 <= a <= NUM_LABELS:
                continue
            if x_pos + 1 < CANONICAL_SIZE:
                b = int(raw[y_pos + 1][x_pos + 1])
                if 1 <= b <= NUM_LABELS:
                    diag_dr[a - 1][b - 1] += 1.0
            if x_pos - 1 >= 0:
                b = int(raw[y_pos + 1][x_pos - 1])
                if 1 <= b <= NUM_LABELS:
                    diag_dl[a - 1][b - 1] += 1.0
    clip = max(1, int(shift_clip))
    bins = list(range(-clip, clip + 1))
    shift_counter = Counter({value: 0 for value in bins})
    row_centers: list[float | None] = []
    for y_pos in range(CANONICAL_SIZE):
        xs = [x_pos for x_pos in range(CANONICAL_SIZE) if int(mask[y_pos][x_pos]) > 0]
        row_centers.append(float(sum(xs) / len(xs)) if xs else None)
    shifts: list[float] = []
    for y_pos in range(CANONICAL_SIZE - 1):
        if row_centers[y_pos] is None or row_centers[y_pos + 1] is None:
            continue
        shift = float(row_centers[y_pos + 1]) - float(row_centers[y_pos])
        shifts.append(shift)
        shift_bin = int(round(max(-clip, min(clip, shift))))
        shift_counter[shift_bin] += 1
    label_shift_hist: list[float] = []
    label_shift_mean: list[float] = []
    label_shift_std: list[float] = []
    for label in range(1, NUM_LABELS + 1):
        centers: list[float | None] = []
        for y_pos in range(CANONICAL_SIZE):
            xs = [x_pos for x_pos in range(CANONICAL_SIZE) if int(raw[y_pos][x_pos]) == label]
            centers.append(float(sum(xs) / len(xs)) if xs else None)
        label_shifts = []
        label_counter = Counter({value: 0 for value in bins})
        for y_pos in range(CANONICAL_SIZE - 1):
            if centers[y_pos] is None or centers[y_pos + 1] is None:
                continue
            shift = float(centers[y_pos + 1]) - float(centers[y_pos])
            label_shifts.append(shift)
            label_counter[int(round(max(-clip, min(clip, shift))))] += 1
        label_shift_hist.extend(_normalize_distribution([float(label_counter[value]) for value in bins]))
        if label_shifts:
            mean = sum(label_shifts) / len(label_shifts)
            var = sum((value - mean) ** 2 for value in label_shifts) / len(label_shifts)
            label_shift_mean.append(float(mean / max(1, clip)))
            label_shift_std.append(float(math.sqrt(var) / max(1, clip)))
        else:
            label_shift_mean.append(0.0)
            label_shift_std.append(0.0)
    dr_total = sum(value for row in diag_dr for value in row)
    dl_total = sum(value for row in diag_dl for value in row)
    shift_hist = _normalize_distribution([float(shift_counter[value]) for value in bins])
    return {
        "diag_down_right": _normalize_distribution([value for row in diag_dr for value in row]),
        "diag_down_left": _normalize_distribution([value for row in diag_dl for value in row]),
        "shift_hist": shift_hist,
        "label_shift_hist": label_shift_hist,
        "label_shift_mean": label_shift_mean,
        "label_shift_std": label_shift_std,
        "shift_mean": float(sum(shifts) / len(shifts)) if shifts else 0.0,
        "shift_std": float(math.sqrt(sum((value - (sum(shifts) / len(shifts))) ** 2 for value in shifts) / len(shifts))) if shifts else 0.0,
        "shift_nonzero_fraction": float(sum(1 for value in shifts if abs(value) >= 0.5) / max(1, len(shifts))),
        "slant_right_score": float(dr_total),
        "slant_left_score": float(dl_total),
        "directional_asymmetry": float((dr_total - dl_total) / max(1.0, dr_total + dl_total)),
        "support_shift_entropy": _entropy_from_probs(shift_hist),
    }


def _hist_for_vocab(counter: Counter[Any], vocab: list[Any]) -> list[float]:
    total = float(sum(counter.values()))
    values = [float(counter.get(key, 0)) for key in vocab]
    other = max(0.0, total - sum(values))
    return _normalize_distribution([*values, other])


def _sample_descriptor(raw: list[list[int]], config: dict[str, Any], vocab: dict[str, list[Any]]) -> dict[str, Any]:
    weights = config.get("descriptor_weights", {}) if isinstance(config.get("descriptor_weights"), dict) else {}
    clustering_cf = config.get("clustering", {}) if isinstance(config.get("clustering"), dict) else {}
    normalize_blocks = bool(clustering_cf.get("normalize_descriptor_blocks", True))
    eps = float((config.get("aggregation", {}) if isinstance(config.get("aggregation"), dict) else {}).get("eps", 1.0e-8))
    mask = _mask_from_raw(raw)
    area = sum(sum(row) for row in mask) / float(CANONICAL_SIZE * CANONICAL_SIZE)
    row_proj = [sum(int(mask[y][x]) for x in range(CANONICAL_SIZE)) / float(CANONICAL_SIZE) for y in range(CANONICAL_SIZE)]
    col_proj = [sum(int(mask[y][x]) for y in range(CANONICAL_SIZE)) / float(CANONICAL_SIZE) for x in range(CANONICAL_SIZE)]
    bbox = _bbox(mask)
    if bbox is None:
        bbox_vec = [0.0, 0.0, 0.0, 0.0]
        hole_ratio = 0.0
    else:
        y_min, y_max, x_min, x_max = bbox
        bbox_vec = [y_min / 20.0, y_max / 20.0, x_min / 20.0, x_max / 20.0]
        bbox_area = float((y_max - y_min + 1) * (x_max - x_min + 1))
        hole_ratio = sum(1 for y in range(y_min, y_max + 1) for x in range(x_min, x_max + 1) if int(mask[y][x]) <= 0) / bbox_area
    label_counts = [0.0 for _ in range(NUM_LABELS)]
    for y_pos in range(CANONICAL_SIZE):
        for x_pos in range(CANONICAL_SIZE):
            label = int(raw[y_pos][x_pos])
            if 1 <= label <= NUM_LABELS:
                label_counts[label - 1] += 1.0
    label_hist = _normalize_distribution(label_counts, eps)
    label_diversity = sum(1 for value in label_hist if value > eps)
    dominant_label_ratio = max(label_hist) if label_hist else 0.0
    label_entropy = _entropy_from_probs(label_hist)
    normalized_entropy = label_entropy / math.log(NUM_LABELS)
    trans = _transition_descriptor(raw)
    occ = _occupancy_descriptor(mask)
    motif2_counts = _motif_counts(raw, 2, bool((config.get("motif2", {}) if isinstance(config.get("motif2"), dict) else {}).get("include_background", True)))
    motif3_counts = _motif_counts(raw, 3, bool((config.get("motif3", {}) if isinstance(config.get("motif3"), dict) else {}).get("include_background", True)))
    row_counts, row_nonzero, row_start, row_end = _program_counts(raw, "row")
    col_counts, col_nonzero, col_start, col_end = _program_counts(raw, "col")
    num_components, largest_component_ratio, tiny_island_count, component_area_mean, component_area_max = _components(mask)
    shift_clip = int((config.get("directional_shift", {}) if isinstance(config.get("directional_shift"), dict) else {}).get("shift_clip", 3))
    directional = _directional_descriptor(raw, mask, shift_clip)
    motif2_hist = _hist_for_vocab(motif2_counts, vocab.get("motif2", []))
    motif3_hist = _hist_for_vocab(motif3_counts, vocab.get("motif3", []))
    row_program_hist = _hist_for_vocab(row_counts, vocab.get("row_program", []))
    col_program_hist = _hist_for_vocab(col_counts, vocab.get("col_program", []))
    blocks: list[tuple[str, list[float]]] = [
        ("support", [area, *row_proj, *col_proj, *bbox_vec, hole_ratio]),
        ("label_hist", [*label_hist, float(label_diversity) / NUM_LABELS, dominant_label_ratio, label_entropy, normalized_entropy]),
        ("transition", [*trans["transition_h"], *trans["transition_v"], *trans["transition_h_off"], *trans["transition_v_off"], trans["same_label_h"], trans["same_label_v"], trans["diff_label_h"], trans["diff_label_v"]]),
        ("motif2", [*motif2_hist, _entropy_from_counts(motif2_counts)]),
        ("motif3", [*motif3_hist, _entropy_from_counts(motif3_counts)]),
        ("row_program", [*row_nonzero, *row_start, *row_end, *row_program_hist, _entropy_from_counts(row_counts)]),
        ("col_program", [*col_nonzero, *col_start, *col_end, *col_program_hist, _entropy_from_counts(col_counts)]),
        (
            "directional_shift",
            [
                *directional["diag_down_right"],
                *directional["diag_down_left"],
                *directional["shift_hist"],
                *directional["label_shift_hist"],
                *directional["label_shift_mean"],
                *directional["label_shift_std"],
                directional["shift_mean"],
                directional["shift_std"],
                directional["shift_nonzero_fraction"],
                directional["slant_right_score"],
                directional["slant_left_score"],
                directional["directional_asymmetry"],
                directional["support_shift_entropy"],
            ],
        ),
        ("occupancy", [*occ["occupancy_h"], *occ["occupancy_v"]]),
        ("component", [float(num_components) / 20.0, largest_component_ratio, float(tiny_island_count) / 20.0, component_area_mean / 400.0, component_area_max / 400.0]),
    ]
    vector: list[float] = []
    slices: dict[str, list[int]] = {}
    offset = 0
    for name, block in blocks:
        values = _l2_normalize(block, eps) if normalize_blocks else [float(value) for value in block]
        weight = float(weights.get(name, 1.0))
        weighted = [float(value) * weight for value in values]
        vector.extend(weighted)
        slices[name] = [offset, offset + len(weighted)]
        offset += len(weighted)
    vector = _l2_normalize(vector, eps)
    return {
        "descriptor": vector,
        "descriptor_slices": slices,
        "raw_y20": raw,
        "fg_mask": mask,
        "foreground_area_ratio": float(area),
        "label_hist_16": label_hist,
        "label_diversity": int(label_diversity),
        "dominant_label_ratio": float(dominant_label_ratio),
        "label_entropy": float(label_entropy),
        "motif2_entropy": float(_entropy_from_counts(motif2_counts)),
        "motif3_entropy": float(_entropy_from_counts(motif3_counts)),
        "row_program_counts": row_counts,
        "col_program_counts": col_counts,
        "transition_h": trans["transition_h"],
        "transition_v": trans["transition_v"],
        "directional": directional,
        "component": {
            "num_components": int(num_components),
            "largest_component_ratio": float(largest_component_ratio),
            "tiny_island_count": int(tiny_island_count),
            "component_area_mean": float(component_area_mean),
            "component_area_max": float(component_area_max),
        },
    }


def _build_vocab(samples_by_category: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, list[Any]]:
    counters = {"motif2": Counter(), "motif3": Counter(), "row_program": Counter(), "col_program": Counter()}
    motif2_cf = config.get("motif2", {}) if isinstance(config.get("motif2"), dict) else {}
    motif3_cf = config.get("motif3", {}) if isinstance(config.get("motif3"), dict) else {}
    row_cf = config.get("row_program", {}) if isinstance(config.get("row_program"), dict) else {}
    col_cf = config.get("col_program", {}) if isinstance(config.get("col_program"), dict) else {}
    for samples in samples_by_category.values():
        for sample in samples:
            raw = sample["raw_y20"]
            if bool(motif2_cf.get("enabled", True)):
                counters["motif2"].update(_motif_counts(raw, 2, bool(motif2_cf.get("include_background", True))))
            if bool(motif3_cf.get("enabled", True)):
                counters["motif3"].update(_motif_counts(raw, 3, bool(motif3_cf.get("include_background", True))))
            if bool(row_cf.get("enabled", True)):
                row_counts, _, _, _ = _program_counts(raw, "row")
                counters["row_program"].update(row_counts)
            if bool(col_cf.get("enabled", True)):
                col_counts, _, _, _ = _program_counts(raw, "col")
                counters["col_program"].update(col_counts)
    return {
        "motif2": [key for key, _ in counters["motif2"].most_common(int(motif2_cf.get("top_k", 128)))],
        "motif3": [key for key, _ in counters["motif3"].most_common(int(motif3_cf.get("top_k", 128)))],
        "row_program": [key for key, _ in counters["row_program"].most_common(int(row_cf.get("top_k", 128)))],
        "col_program": [key for key, _ in counters["col_program"].most_common(int(col_cf.get("top_k", 128)))],
    }


def _mode_count(category: str, config: dict[str, Any], sample_count: int) -> tuple[int, int]:
    by_category = config.get("mode_count_by_category", {}) if isinstance(config.get("mode_count_by_category"), dict) else {}
    requested = int(by_category.get(category, config.get("mode_count_default", 8)))
    clustering_cf = config.get("clustering", {}) if isinstance(config.get("clustering"), dict) else {}
    if bool(clustering_cf.get("reduce_modes_if_needed", True)):
        return requested, max(1, min(requested, int(sample_count)))
    if int(sample_count) < requested:
        raise ValueError(f"Category {category!r} has {sample_count} samples, fewer than requested grammar modes {requested}.")
    return requested, requested


def _cluster_descriptors(descriptors: list[list[float]], category: str, config: dict[str, Any]) -> tuple[list[int], list[list[float]], int, int]:
    requested, effective = _mode_count(category, config, len(descriptors))
    if effective <= 1:
        return [0 for _ in descriptors], [descriptors[0] if descriptors else []], requested, 1
    np = _require_numpy()
    clustering_cf = config.get("clustering", {}) if isinstance(config.get("clustering"), dict) else {}
    x = np.asarray(descriptors, dtype=np.float32)
    method = str(clustering_cf.get("method", "kmeans"))
    try:
        sklearn_cluster = _require_sklearn_cluster()
        cls = getattr(sklearn_cluster, "MiniBatchKMeans" if method == "minibatch_kmeans" else "KMeans")
        kwargs = {
            "n_clusters": int(effective),
            "random_state": int(clustering_cf.get("random_state", 0)),
            "max_iter": int(clustering_cf.get("max_iter", 300)),
        }
        n_init = clustering_cf.get("n_init", 10)
        kwargs["n_init"] = int(n_init) if isinstance(n_init, int) or str(n_init).isdigit() else n_init
        if method == "minibatch_kmeans":
            kwargs["batch_size"] = min(1024, max(16, len(descriptors)))
        model = cls(**kwargs)
        assigned = model.fit_predict(x).tolist()
        centers = [[float(value) for value in row] for row in model.cluster_centers_.tolist()]
        return [int(value) for value in assigned], centers, requested, effective
    except ImportError:
        return _numpy_kmeans(x, effective, requested, int(clustering_cf.get("random_state", 0)), int(clustering_cf.get("max_iter", 300)), np)


def _numpy_kmeans(x: Any, effective: int, requested: int, random_state: int, max_iter: int, np: Any) -> tuple[list[int], list[list[float]], int, int]:
    rng = np.random.default_rng(int(random_state))
    sample_count = int(x.shape[0])
    first = int(rng.integers(0, sample_count))
    centers = [x[first].copy()]
    while len(centers) < int(effective):
        current = np.stack(centers, axis=0)
        dist = ((x[:, None, :] - current[None, :, :]) ** 2).sum(axis=2)
        nearest = dist.min(axis=1)
        next_index = int(np.argmax(nearest))
        centers.append(x[next_index].copy())
    centers_arr = np.stack(centers, axis=0).astype(np.float32)
    assigned = np.zeros((sample_count,), dtype=np.int64)
    for _ in range(max(1, int(max_iter))):
        dist = ((x[:, None, :] - centers_arr[None, :, :]) ** 2).sum(axis=2)
        next_assigned = dist.argmin(axis=1).astype(np.int64)
        if np.array_equal(next_assigned, assigned):
            break
        assigned = next_assigned
        for mode_index in range(int(effective)):
            members = x[assigned == mode_index]
            if int(members.shape[0]) > 0:
                centers_arr[mode_index] = members.mean(axis=0)
    return [int(value) for value in assigned.tolist()], [[float(value) for value in row] for row in centers_arr.tolist()], int(requested), int(effective)


def _aggregate_mode(samples: list[dict[str, Any]], mode_index: int, config: dict[str, Any]) -> dict[str, Any]:
    np = _require_numpy()
    torch = _require_torch()
    aggregation_cf = config.get("aggregation", {}) if isinstance(config.get("aggregation"), dict) else {}
    eps = float(aggregation_cf.get("eps", 1.0e-8))
    fg_threshold = float(aggregation_cf.get("fg_threshold", 0.05))
    dtype = getattr(torch, "float16") if bool(aggregation_cf.get("save_float16", False)) else getattr(torch, "float32")
    raw_stack = np.asarray([sample["raw_y20"] for sample in samples], dtype=np.int64)
    mask_stack = (raw_stack > 0).astype(np.float32)
    fg_prob = mask_stack.mean(axis=0).astype(np.float32)
    fg_count = mask_stack.sum(axis=0).astype(np.float32)
    label_counts = np.zeros((NUM_LABELS, CANONICAL_SIZE, CANONICAL_SIZE), dtype=np.float32)
    for label_index in range(NUM_LABELS):
        label_counts[label_index] = (raw_stack == (label_index + 1)).sum(axis=0).astype(np.float32)
    label_prob = np.where(fg_count[None, :, :] > 0.0, label_counts / np.maximum(fg_count[None, :, :], eps), 0.0).astype(np.float32)
    label_confidence = label_prob.max(axis=0).astype(np.float32)
    label_mass = (fg_prob * label_confidence).astype(np.float32)
    label_argmax = np.where(fg_prob >= fg_threshold, label_prob.argmax(axis=0).astype(np.int64) + 1, 0).astype(np.int64)
    row_counter: Counter[tuple[int, ...]] = Counter()
    col_counter: Counter[tuple[int, ...]] = Counter()
    for sample in samples:
        row_counter.update(sample["row_program_counts"])
        col_counter.update(sample["col_program_counts"])
    directional_values = [sample["directional"] for sample in samples]
    mode_stats = {
        "mode_index": int(mode_index),
        "mode_num_samples": int(len(samples)),
        "mode_mean_area": float(sum(float(sample["foreground_area_ratio"]) for sample in samples) / max(1, len(samples))),
        "mode_label_diversity_mean": float(sum(float(sample["label_diversity"]) for sample in samples) / max(1, len(samples))),
        "mode_dominant_label_ratio_mean": float(sum(float(sample["dominant_label_ratio"]) for sample in samples) / max(1, len(samples))),
        "mode_label_entropy_mean": float(sum(float(sample["label_entropy"]) for sample in samples) / max(1, len(samples))),
        "mode_motif2_entropy_mean": float(sum(float(sample["motif2_entropy"]) for sample in samples) / max(1, len(samples))),
        "mode_motif3_entropy_mean": float(sum(float(sample["motif3_entropy"]) for sample in samples) / max(1, len(samples))),
        "mode_transition_h_mean": [float(sum(float(sample["transition_h"][index]) for sample in samples) / max(1, len(samples))) for index in range(NUM_LABELS * NUM_LABELS)],
        "mode_transition_v_mean": [float(sum(float(sample["transition_v"][index]) for sample in samples) / max(1, len(samples))) for index in range(NUM_LABELS * NUM_LABELS)],
        "mode_directional_shift_stats": _directional_stats(directional_values),
        "mode_active_pixels": int((fg_prob >= fg_threshold).sum().item() if hasattr((fg_prob >= fg_threshold).sum(), "item") else (fg_prob >= fg_threshold).sum()),
    }
    return {
        "basis_fg_mask_prob": torch.tensor(fg_prob[None, :, :], dtype=dtype),
        "basis_label_prob_16": torch.tensor(label_prob, dtype=dtype),
        "basis_label_mass": torch.tensor(label_mass, dtype=dtype),
        "basis_label_argmax": [[int(label_argmax[y_pos, x_pos]) for x_pos in range(CANONICAL_SIZE)] for y_pos in range(CANONICAL_SIZE)],
        "basis_label_confidence": torch.tensor(label_confidence, dtype=dtype),
        "mode_stats": mode_stats,
        "top_row_programs": _top_counter(row_counter, 32),
        "top_col_programs": _top_counter(col_counter, 32),
    }


def _directional_stats(directional_values: list[dict[str, Any]]) -> dict[str, Any]:
    if not directional_values:
        return {
            "shift_hist_mean": [],
            "shift_entropy_mean": 0.0,
            "shift_nonzero_fraction_mean": 0.0,
            "shift_mean": 0.0,
            "shift_std": 0.0,
            "slant_right_score_mean": 0.0,
            "slant_left_score_mean": 0.0,
            "directional_asymmetry_mean": 0.0,
            "directional_non_degenerate": False,
        }
    dim = len(directional_values[0].get("shift_hist", []))
    shift_hist_mean = [float(sum(float(value.get("shift_hist", [0.0] * dim)[index]) for value in directional_values) / max(1, len(directional_values))) for index in range(dim)]
    slant_right = [float(value.get("slant_right_score", 0.0)) for value in directional_values]
    slant_left = [float(value.get("slant_left_score", 0.0)) for value in directional_values]
    nonzero = [float(value.get("shift_nonzero_fraction", 0.0)) for value in directional_values]
    asym = [float(value.get("directional_asymmetry", 0.0)) for value in directional_values]
    entropy = [float(value.get("support_shift_entropy", 0.0)) for value in directional_values]
    shift_means = [float(value.get("shift_mean", 0.0)) for value in directional_values]
    shift_stds = [float(value.get("shift_std", 0.0)) for value in directional_values]
    directional_mass = max(sum(slant_right), sum(slant_left))
    non_degenerate = bool(directional_mass > 0.0 and (sum(nonzero) / max(1, len(nonzero)) > 0.0 or abs(sum(asym) / max(1, len(asym))) > 0.01))
    return {
        "shift_hist_mean": shift_hist_mean,
        "shift_entropy_mean": float(sum(entropy) / max(1, len(entropy))),
        "shift_nonzero_fraction_mean": float(sum(nonzero) / max(1, len(nonzero))),
        "shift_mean": float(sum(shift_means) / max(1, len(shift_means))),
        "shift_std": float(sum(shift_stds) / max(1, len(shift_stds))),
        "slant_right_score_mean": float(sum(slant_right) / max(1, len(slant_right))),
        "slant_left_score_mean": float(sum(slant_left) / max(1, len(slant_left))),
        "directional_asymmetry_mean": float(sum(asym) / max(1, len(asym))),
        "directional_non_degenerate": bool(non_degenerate),
    }


def _category_quality(category: str, requested: int, mode_stats: list[dict[str, Any]], category_descs: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    quality_cf = config.get("quality", {}) if isinstance(config.get("quality"), dict) else {}
    min_effective = int(quality_cf.get("min_effective_modes_default", 2))
    min_fraction = float(quality_cf.get("min_effective_mode_fraction_default", 0.5))
    min_mode_samples = int(quality_cf.get("min_mode_samples", 1))
    min_mean_area = float(quality_cf.get("min_mean_fg_area", 0.005))
    max_empty_fraction = float(quality_cf.get("max_empty_mode_fraction", 0.5))
    max_collapse_fraction = float(quality_cf.get("max_label_collapse_fraction_default", 0.75))
    fg_threshold = float((config.get("aggregation", {}) if isinstance(config.get("aggregation"), dict) else {}).get("fg_threshold", 0.05))
    true_div_q10 = _quantile([float(desc["label_diversity"]) for desc in category_descs], 0.10)
    effective_indices = []
    empty_indices = []
    collapsed_indices = []
    warnings: list[str] = []
    for stat in mode_stats:
        mode_index = int(stat["mode_index"])
        mode_empty = bool(int(stat.get("mode_num_samples", 0)) < min_mode_samples or float(stat.get("mode_mean_area", 0.0)) < min_mean_area or int(stat.get("mode_active_pixels", 0)) <= 0)
        if mode_empty:
            empty_indices.append(mode_index)
        collapsed = bool(true_div_q10 > 1.0 and float(stat.get("mode_label_diversity_mean", 0.0)) <= 1.0)
        collapsed = bool(collapsed or (true_div_q10 > 1.0 and float(stat.get("mode_dominant_label_ratio_mean", 0.0)) >= 0.995))
        if collapsed:
            collapsed_indices.append(mode_index)
        if not mode_empty and not collapsed:
            effective_indices.append(mode_index)
    effective_modes = len(effective_indices)
    empty_fraction = len(empty_indices) / float(max(1, len(mode_stats)))
    collapse_fraction = len(collapsed_indices) / float(max(1, len(mode_stats)))
    unusable_reasons: list[str] = []
    if effective_modes < min_effective:
        unusable_reasons.append("insufficient_effective_modes")
    if effective_modes / float(max(1, requested)) < min_fraction:
        unusable_reasons.append("effective_mode_fraction_too_low")
    if empty_fraction > max_empty_fraction:
        unusable_reasons.append("too_many_empty_modes")
    if collapse_fraction > max_collapse_fraction:
        unusable_reasons.append("too_many_label_collapsed_modes")
    directional = _directional_stats([desc["directional"] for desc in category_descs])
    required_directional = [str(value) for value in quality_cf.get("require_directional_non_degenerate_for", [])] if isinstance(quality_cf.get("require_directional_non_degenerate_for", []), list) else []
    if category in required_directional and not bool(directional.get("directional_non_degenerate", False)):
        unusable_reasons.append(f"{category.lower()}_directional_degenerate")
    if category in required_directional and effective_modes <= 0:
        unusable_reasons.append(f"{category.lower()}_no_effective_modes")
    if empty_indices:
        warnings.append(f"empty_mode_indices={empty_indices}")
    if collapsed_indices:
        warnings.append(f"label_collapsed_mode_indices={collapsed_indices}")
    return {
        "category_usable": bool(not unusable_reasons),
        "unusable_reasons": unusable_reasons,
        "warnings": warnings,
        "requested_modes": int(requested),
        "effective_modes": int(effective_modes),
        "effective_mode_ratio": float(effective_modes / max(1, requested)),
        "effective_mode_indices": effective_indices,
        "empty_mode_indices": empty_indices,
        "empty_mode_fraction": float(empty_fraction),
        "label_collapsed_mode_indices": collapsed_indices,
        "label_collapse_fraction": float(collapse_fraction),
        "label_diversity_q10": float(true_div_q10),
        "fg_threshold": float(fg_threshold),
        "directional_shift_stats": directional,
    }


def _quality_summary(categories: dict[str, dict[str, Any]]) -> dict[str, Any]:
    usable = [category for category, entry in categories.items() if bool(entry.get("category_usable", False))]
    unusable = [category for category, entry in categories.items() if not bool(entry.get("category_usable", False))]
    return {
        "categories_total": int(len(categories)),
        "categories_usable": int(len(usable)),
        "categories_unusable": int(len(unusable)),
        "usable_categories": sorted(usable),
        "unusable_categories": sorted(unusable),
        "warnings_by_category": {category: entry.get("quality_report", {}).get("warnings", []) for category, entry in categories.items() if entry.get("quality_report", {}).get("warnings")},
        "unusable_reasons_by_category": {category: entry.get("quality_report", {}).get("unusable_reasons", []) for category, entry in categories.items() if entry.get("quality_report", {}).get("unusable_reasons")},
        "effective_modes_by_category": {category: int(entry.get("effective_modes", 0)) for category, entry in categories.items()},
        "strategy_by_category": {category: str(entry.get("strategy", "")) for category, entry in categories.items()},
    }


def build_instruction_matrix_grammar_prior(items: list[dict[str, Any]], config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = _merged_config(config)
    if not bool(cfg.get("enabled", False)):
        return {"enabled": False, "schema_version": INSTRUCTION_GRAMMAR_SCHEMA_VERSION, "categories": {}, "quality_summary": {}}
    if str(cfg.get("schema_version")) != INSTRUCTION_GRAMMAR_SCHEMA_VERSION:
        raise ValueError(f"instruction_matrix_grammar_prior.schema_version must be {INSTRUCTION_GRAMMAR_SCHEMA_VERSION!r}.")
    samples_by_category: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if bool(item.get("is_empty_foreground", False)):
            continue
        category = str(item.get("category"))
        samples_by_category.setdefault(category, []).append({"sample_id": str(item.get("sample_id")), "raw_y20": _raw_from_item(item)})
    vocab = _build_vocab(samples_by_category, cfg)
    categories: dict[str, dict[str, Any]] = {}
    descriptor_slices: dict[str, list[int]] | None = None
    strategy_by_category = cfg.get("strategy_by_category", {}) if isinstance(cfg.get("strategy_by_category"), dict) else {}
    for category in sorted(samples_by_category):
        raw_samples = samples_by_category[category]
        descriptors = [_sample_descriptor(sample["raw_y20"], cfg, vocab) for sample in raw_samples]
        if descriptors:
            descriptor_slices = descriptors[0]["descriptor_slices"]
        assigned, centers, requested_modes, effective_modes = _cluster_descriptors([desc["descriptor"] for desc in descriptors], category, cfg)
        mode_payloads: list[dict[str, Any]] = []
        mode_stats: list[dict[str, Any]] = []
        nonempty_mode_indices: list[int] = []
        top_row_counter: Counter[tuple[int, ...]] = Counter()
        top_col_counter: Counter[tuple[int, ...]] = Counter()
        for mode_index in range(effective_modes):
            mode_descs = [desc for desc, label in zip(descriptors, assigned) if int(label) == mode_index]
            if not mode_descs:
                continue
            mode_payload = _aggregate_mode(mode_descs, mode_index, cfg)
            mode_payloads.append(mode_payload)
            mode_stats.append(mode_payload["mode_stats"])
            nonempty_mode_indices.append(int(mode_index))
            for desc in mode_descs:
                top_row_counter.update(desc["row_program_counts"])
                top_col_counter.update(desc["col_program_counts"])
        torch = _require_torch()
        if mode_payloads:
            basis_fg_mask_prob = torch.stack([payload["basis_fg_mask_prob"] for payload in mode_payloads], dim=0)
            basis_label_prob_16 = torch.stack([payload["basis_label_prob_16"] for payload in mode_payloads], dim=0)
            basis_label_mass = torch.stack([payload["basis_label_mass"] for payload in mode_payloads], dim=0)
            basis_label_confidence = torch.stack([payload["basis_label_confidence"] for payload in mode_payloads], dim=0)
            basis_label_argmax = [payload["basis_label_argmax"] for payload in mode_payloads]
        else:
            basis_fg_mask_prob = torch.zeros((0, 1, CANONICAL_SIZE, CANONICAL_SIZE), dtype=torch.float32)
            basis_label_prob_16 = torch.zeros((0, NUM_LABELS, CANONICAL_SIZE, CANONICAL_SIZE), dtype=torch.float32)
            basis_label_mass = torch.zeros((0, CANONICAL_SIZE, CANONICAL_SIZE), dtype=torch.float32)
            basis_label_confidence = torch.zeros((0, CANONICAL_SIZE, CANONICAL_SIZE), dtype=torch.float32)
            basis_label_argmax = []
        directional_shift_stats = _directional_stats([desc["directional"] for desc in descriptors])
        quality_report = _category_quality(category, requested_modes, mode_stats, descriptors, cfg)
        strategy = "instruction_matrix_grammar_modes_directional" if str(strategy_by_category.get(category, "")).endswith("directional") else "instruction_matrix_grammar_modes"
        actual_effective_modes = int(len(mode_payloads))
        categories[category] = {
            "strategy": strategy,
            "requested_modes": int(requested_modes),
            "effective_modes": actual_effective_modes,
            "category_usable": bool(quality_report["category_usable"]),
            "unusable_reasons": quality_report["unusable_reasons"],
            "num_samples": int(len(raw_samples)),
            "basis_fg_mask_prob": basis_fg_mask_prob,
            "basis_label_prob_16": basis_label_prob_16,
            "basis_label_mass": basis_label_mass,
            "basis_label_argmax": basis_label_argmax,
            "basis_label_confidence": basis_label_confidence,
            "mode_stats": mode_stats,
            "mode_centers": [centers[index] for index in nonempty_mode_indices],
            "mode_original_indices": nonempty_mode_indices,
            "mode_num_samples": [int(sum(1 for label in assigned if int(label) == mode_index)) for mode_index in nonempty_mode_indices],
            "top_row_programs": _top_counter(top_row_counter, 64),
            "top_col_programs": _top_counter(top_col_counter, 64),
            "directional_shift_stats": directional_shift_stats,
            "quality_report": quality_report,
        }
    return {
        "enabled": True,
        "schema_version": INSTRUCTION_GRAMMAR_SCHEMA_VERSION,
        "config": cfg,
        "vocab_summary": {key: int(len(value)) for key, value in vocab.items()},
        "instruction_grammar_descriptor_slices": descriptor_slices or {},
        "categories": categories,
        "quality_summary": _quality_summary(categories),
    }


def instruction_grammar_prior_summary(prior: object) -> dict[str, Any]:
    if not isinstance(prior, dict):
        return {"enabled": False, "schema_version": None, "categories": [], "usable_categories": [], "unusable_categories": [], "total_modes": 0}
    categories = prior.get("categories", {}) if isinstance(prior.get("categories", {}), dict) else {}
    quality = prior.get("quality_summary", {}) if isinstance(prior.get("quality_summary", {}), dict) else {}
    return {
        "enabled": bool(prior.get("enabled", False)),
        "schema_version": prior.get("schema_version"),
        "categories": sorted(str(category) for category in categories.keys()),
        "usable_categories": quality.get("usable_categories", []),
        "unusable_categories": quality.get("unusable_categories", []),
        "total_modes": int(sum(int(entry.get("effective_modes", 0)) for entry in categories.values() if isinstance(entry, dict))),
        "warnings_by_category": quality.get("warnings_by_category", {}),
        "unusable_reasons_by_category": quality.get("unusable_reasons_by_category", {}),
        "effective_modes_by_category": quality.get("effective_modes_by_category", {}),
        "strategy_by_category": quality.get("strategy_by_category", {}),
    }


def _to_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _masked_argmax(label_prob: Any, fg_prob: Any, threshold: float) -> tuple[list[list[int]], list[list[float]]]:
    label_prob = _to_list(label_prob)
    fg_prob = _to_list(fg_prob)
    if isinstance(fg_prob, list) and len(fg_prob) == 1 and isinstance(fg_prob[0], list):
        fg_prob = fg_prob[0]
    argmax_grid: list[list[int]] = []
    confidence_grid: list[list[float]] = []
    for y_pos in range(CANONICAL_SIZE):
        arg_row: list[int] = []
        conf_row: list[float] = []
        for x_pos in range(CANONICAL_SIZE):
            values = [float(label_prob[label_index][y_pos][x_pos]) for label_index in range(NUM_LABELS)]
            best_index = max(range(NUM_LABELS), key=lambda index: values[index])
            best_value = float(values[best_index])
            conf_row.append(best_value)
            arg_row.append(best_index + 1 if float(fg_prob[y_pos][x_pos]) >= threshold else 0)
        argmax_grid.append(arg_row)
        confidence_grid.append(conf_row)
    return argmax_grid, confidence_grid


def inspect_instruction_grammar_prior_category(
    prior: dict[str, Any],
    category: str,
    output_dir: Path,
    *,
    basis_mass_threshold: float,
    num_basis_samples: int,
    cols: int,
    cell_size: int,
    save_tiled_grid: Any,
    grid_to_rgb: Any,
    prob_grid_to_rgb: Any,
    save_json: Any,
) -> dict[str, Any]:
    categories = prior.get("categories", {}) if isinstance(prior.get("categories", {}), dict) else {}
    if category not in categories:
        raise ValueError(f"Category {category!r} not found in instruction_matrix_grammar_prior.categories.")
    entry = categories[category]
    if not isinstance(entry, dict):
        raise ValueError(f"instruction_matrix_grammar_prior.categories[{category!r}] must be a dict.")
    output_dir.mkdir(parents=True, exist_ok=True)
    fg_prob = _to_list(entry.get("basis_fg_mask_prob"))
    label_prob = _to_list(entry.get("basis_label_prob_16"))
    label_mass = _to_list(entry.get("basis_label_mass"))
    label_confidence = _to_list(entry.get("basis_label_confidence"))
    if fg_prob is None or label_prob is None or label_mass is None:
        raise ValueError(f"Instruction grammar prior category {category!r} is missing basis tensors.")
    max_tiles = min(int(entry.get("effective_modes", len(fg_prob))), max(1, int(num_basis_samples)))
    labels = [f"m={index}" for index in range(max_tiles)]
    fg_tiles = []
    argmax_tiles = []
    confidence_tiles = []
    sampled_tiles = []
    visual_rows: list[dict[str, Any]] = []
    for mode_index in range(max_tiles):
        fg_grid = fg_prob[mode_index][0] if isinstance(fg_prob[mode_index], list) and len(fg_prob[mode_index]) == 1 else fg_prob[mode_index]
        argmax_grid, confidence_grid = _masked_argmax(label_prob[mode_index], fg_grid, basis_mass_threshold)
        mass_grid = label_mass[mode_index]
        if label_confidence is not None:
            confidence_grid = label_confidence[mode_index]
        fg_tiles.append(prob_grid_to_rgb(fg_grid))
        argmax_tiles.append(grid_to_rgb(argmax_grid, mode="label"))
        confidence_tiles.append(prob_grid_to_rgb(confidence_grid))
        sampled_tiles.append(grid_to_rgb(argmax_grid, mode="label"))
        active_pixels = sum(1 for y_pos in range(CANONICAL_SIZE) for x_pos in range(CANONICAL_SIZE) if float(fg_grid[y_pos][x_pos]) >= basis_mass_threshold)
        visual_rows.append(
            {
                "mode_index": int(mode_index),
                "active_pixels": int(active_pixels),
                "mean_fg_prob": float(sum(sum(float(value) for value in row) for row in fg_grid) / 400.0),
                "mean_label_mass": float(sum(sum(float(value) for value in row) for row in mass_grid) / 400.0),
            }
        )
    save_tiled_grid(fg_tiles, labels, output_dir / f"{category}_grammar_mode_fg_mask_grid.png", cols=int(cols), cell_size=int(cell_size))
    save_tiled_grid(argmax_tiles, labels, output_dir / f"{category}_grammar_mode_label_argmax_grid.png", cols=int(cols), cell_size=int(cell_size))
    save_tiled_grid(confidence_tiles, labels, output_dir / f"{category}_grammar_mode_label_confidence_grid.png", cols=int(cols), cell_size=int(cell_size))
    save_tiled_grid(sampled_tiles, labels, output_dir / f"{category}_grammar_mode_sampled_prior_grid.png", cols=int(cols), cell_size=int(cell_size))
    save_json(output_dir / f"{category}_top_row_programs.json", {"category": category, "top_row_programs": entry.get("top_row_programs", [])})
    save_json(output_dir / f"{category}_top_col_programs.json", {"category": category, "top_col_programs": entry.get("top_col_programs", [])})
    save_json(output_dir / f"{category}_directional_shift_stats.json", {"category": category, "directional_shift_stats": entry.get("directional_shift_stats", {})})
    save_json(output_dir / f"{category}_grammar_mode_stats.json", {"category": category, "quality_report": entry.get("quality_report", {}), "mode_stats": entry.get("mode_stats", []), "visualized_modes": visual_rows})
    return {
        "fg_mask": output_dir / f"{category}_grammar_mode_fg_mask_grid.png",
        "label_argmax": output_dir / f"{category}_grammar_mode_label_argmax_grid.png",
        "label_confidence": output_dir / f"{category}_grammar_mode_label_confidence_grid.png",
        "sampled_prior": output_dir / f"{category}_grammar_mode_sampled_prior_grid.png",
        "top_row_programs": output_dir / f"{category}_top_row_programs.json",
        "top_col_programs": output_dir / f"{category}_top_col_programs.json",
        "directional_shift_stats": output_dir / f"{category}_directional_shift_stats.json",
        "mode_stats": output_dir / f"{category}_grammar_mode_stats.json",
    }


def inspect_all_instruction_grammar_prior_categories(
    prior: dict[str, Any],
    output_dir: Path,
    *,
    basis_mass_threshold: float,
    num_basis_samples: int,
    cols: int,
    cell_size: int,
    save_tiled_grid: Any,
    grid_to_rgb: Any,
    prob_grid_to_rgb: Any,
    save_json: Any,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = prior.get("categories", {}) if isinstance(prior.get("categories", {}), dict) else {}
    category_paths: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for category in sorted(str(key) for key in categories.keys()):
        entry = categories[category]
        category_dir = output_dir / category
        category_paths[category] = inspect_instruction_grammar_prior_category(
            prior,
            category,
            category_dir,
            basis_mass_threshold=basis_mass_threshold,
            num_basis_samples=num_basis_samples,
            cols=cols,
            cell_size=cell_size,
            save_tiled_grid=save_tiled_grid,
            grid_to_rgb=grid_to_rgb,
            prob_grid_to_rgb=prob_grid_to_rgb,
            save_json=save_json,
        )
        mode_stats = entry.get("mode_stats", []) if isinstance(entry.get("mode_stats", []), list) else []
        quality = entry.get("quality_report", {}) if isinstance(entry.get("quality_report", {}), dict) else {}
        directional = entry.get("directional_shift_stats", {}) if isinstance(entry.get("directional_shift_stats", {}), dict) else {}
        rows.append(
            {
                "category": category,
                "strategy": entry.get("strategy"),
                "usable": bool(entry.get("category_usable", False)),
                "requested_modes": int(entry.get("requested_modes", 0)),
                "effective_modes": int(entry.get("effective_modes", 0)),
                "effective_mode_ratio": quality.get("effective_mode_ratio", 0.0),
                "mean_mode_area": float(sum(float(stat.get("mode_mean_area", 0.0)) for stat in mode_stats) / max(1, len(mode_stats))),
                "mean_label_entropy": float(sum(float(stat.get("mode_label_entropy_mean", 0.0)) for stat in mode_stats) / max(1, len(mode_stats))),
                "mean_directional_shift_entropy": float(directional.get("shift_entropy_mean", 0.0)),
                "warnings": quality.get("warnings", []),
                "unusable_reasons": quality.get("unusable_reasons", []),
            }
        )
    summary = {
        "schema_version": prior.get("schema_version"),
        "quality_summary": prior.get("quality_summary", {}),
        "categories": rows,
    }
    json_path = output_dir / "all_categories_instruction_grammar_summary.json"
    md_path = output_dir / "all_categories_instruction_grammar_summary.md"
    save_json(json_path, summary)
    lines = [
        "# Instruction Grammar Prior Summary",
        "",
        f"schema_version: {prior.get('schema_version')}",
        "",
        "| category | strategy | usable | requested_modes | effective_modes | effective_mode_ratio | mean_mode_area | mean_label_entropy | mean_directional_shift_entropy | warnings | unusable_reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {category} | {strategy} | {usable} | {requested} | {effective} | {ratio} | {area} | {entropy} | {shift_entropy} | {warnings} | {reasons} |".format(
                category=row["category"],
                strategy=row["strategy"],
                usable=row["usable"],
                requested=row["requested_modes"],
                effective=row["effective_modes"],
                ratio=row["effective_mode_ratio"],
                area=row["mean_mode_area"],
                entropy=row["mean_label_entropy"],
                shift_entropy=row["mean_directional_shift_entropy"],
                warnings=", ".join(str(value) for value in row["warnings"]),
                reasons=", ".join(str(value) for value in row["unusable_reasons"]),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"categories": category_paths, "summary": {"json": json_path, "markdown": md_path}}
