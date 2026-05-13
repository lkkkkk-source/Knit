from __future__ import annotations

import argparse
import csv
import json
import math
import random
import struct
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import IGNORE_INDEX, bbox_from_mask, foreground_area, label_diversity_on_fg, load_label_grid, mask_component_stats, save_json, save_jsonl, validate_foreground_labels


CANONICAL_SIZE = 20
NUM_LABELS = 16


@dataclass
class SampleRecord:
    sample_id: str
    category: str
    input_path: str
    target_path: str
    index_path: str
    raw_y20: list[list[int]]
    fg_y20: list[list[int]]
    fg_mask20: list[list[int]]
    local_z: int | None = None
    source_path: str | None = None
    descriptor: dict[str, Any] | None = None


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required for export_pro_matrix_packet.") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export cross-category pro matrix packet from real train_frontonly instruction17 data.")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--samples-per-category", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection", type=str, default="mixed", choices=["random", "largest_fg", "diverse_labels", "compact", "mixed"])
    parser.add_argument("--max-total-samples", type=int, default=80)
    parser.add_argument("--save-onehot", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true", help="Run a tiny self-contained export smoke test without real manifest/cache inputs.")
    return parser


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _infer_manifest_root(manifest_path: Path, rows: list[dict[str, Any]]) -> Path:
    candidates = [manifest_path.parent, *manifest_path.parents]
    for root in candidates:
        ok = True
        for row in rows[: min(32, len(rows))]:
            for key in ("input_path", "target_path", "index_path"):
                value = row.get(key)
                if not isinstance(value, str):
                    continue
                if not (root / value).exists():
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return root
    return manifest_path.parent


def _resolve_path(raw_path: str, root: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (root / path).resolve()


def _grid_to_list(value: Any) -> list[list[int]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or len(value) != CANONICAL_SIZE or any(not isinstance(row, list) or len(row) != CANONICAL_SIZE for row in value):
        raise ValueError("grid must be [20,20].")
    return [[int(v) for v in row] for row in value]


def _canonicalize_full_masked(raw_y20: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
    fg_mask = [[1 if int(v) != 0 else 0 for v in row] for row in raw_y20]
    fg_y20 = [[int(v) if int(v) != 0 else IGNORE_INDEX for v in row] for row in raw_y20]
    validate_foreground_labels(fg_y20, fg_mask, canonical_size=CANONICAL_SIZE, context="export_pro_matrix_packet")
    return fg_y20, fg_mask


def _neighbors(y: int, x: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if y > 0:
        out.append((y - 1, x))
    if y + 1 < CANONICAL_SIZE:
        out.append((y + 1, x))
    if x > 0:
        out.append((y, x - 1))
    if x + 1 < CANONICAL_SIZE:
        out.append((y, x + 1))
    return out


def _connected_components(mask: list[list[int]]) -> list[list[tuple[int, int]]]:
    visited = [[False for _ in range(CANONICAL_SIZE)] for _ in range(CANONICAL_SIZE)]
    components: list[list[tuple[int, int]]] = []
    for y in range(CANONICAL_SIZE):
        for x in range(CANONICAL_SIZE):
            if visited[y][x] or not int(mask[y][x]):
                continue
            stack = [(y, x)]
            visited[y][x] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                comp.append((cy, cx))
                for ny, nx in _neighbors(cy, cx):
                    if not visited[ny][nx] and int(mask[ny][nx]):
                        visited[ny][nx] = True
                        stack.append((ny, nx))
            components.append(comp)
    return components


def _mask_to_labels(mask: list[list[int]], labels: list[list[int]]) -> list[int]:
    return [int(labels[y][x]) for y in range(CANONICAL_SIZE) for x in range(CANONICAL_SIZE) if int(mask[y][x]) and 1 <= int(labels[y][x]) <= 16]


def _hist16_from_labels(label_values: list[int]) -> list[int]:
    hist = [0 for _ in range(NUM_LABELS)]
    for label in label_values:
        if 1 <= label <= NUM_LABELS:
            hist[label - 1] += 1
    return hist


def _normalized(values: list[float]) -> list[float]:
    s = float(sum(values))
    if s <= 0.0:
        return [0.0 for _ in values]
    return [float(v) / s for v in values]


def _label_projections(fg_y20: list[list[int]]) -> tuple[list[float], list[float], list[list[float]], list[list[float]]]:
    row_proj = [sum(1 for x in range(CANONICAL_SIZE) if 1 <= int(fg_y20[y][x]) <= NUM_LABELS) / float(CANONICAL_SIZE) for y in range(CANONICAL_SIZE)]
    col_proj = [sum(1 for y in range(CANONICAL_SIZE) if 1 <= int(fg_y20[y][x]) <= NUM_LABELS) / float(CANONICAL_SIZE) for x in range(CANONICAL_SIZE)]
    row_label = [[0.0 for _ in range(CANONICAL_SIZE)] for _ in range(NUM_LABELS)]
    col_label = [[0.0 for _ in range(CANONICAL_SIZE)] for _ in range(NUM_LABELS)]
    for y in range(CANONICAL_SIZE):
        for x in range(CANONICAL_SIZE):
            label = int(fg_y20[y][x])
            if 1 <= label <= NUM_LABELS:
                row_label[label - 1][y] += 1.0
                col_label[label - 1][x] += 1.0
    for i in range(NUM_LABELS):
        row_label[i] = [v / float(CANONICAL_SIZE) for v in row_label[i]]
        col_label[i] = [v / float(CANONICAL_SIZE) for v in col_label[i]]
    return row_proj, col_proj, row_label, col_label


def _transition_matrices(fg_y20: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
    th = [[0 for _ in range(NUM_LABELS)] for _ in range(NUM_LABELS)]
    tv = [[0 for _ in range(NUM_LABELS)] for _ in range(NUM_LABELS)]
    for y in range(CANONICAL_SIZE):
        for x in range(CANONICAL_SIZE):
            a = int(fg_y20[y][x])
            if not (1 <= a <= NUM_LABELS):
                continue
            if x + 1 < CANONICAL_SIZE:
                b = int(fg_y20[y][x + 1])
                if 1 <= b <= NUM_LABELS:
                    th[a - 1][b - 1] += 1
            if y + 1 < CANONICAL_SIZE:
                b = int(fg_y20[y + 1][x])
                if 1 <= b <= NUM_LABELS:
                    tv[a - 1][b - 1] += 1
    return th, tv


def _top_pairs(matrix: list[list[int]], top_k: int = 20) -> list[dict[str, Any]]:
    pairs: list[tuple[int, int, int]] = []
    for i in range(NUM_LABELS):
        for j in range(NUM_LABELS):
            if matrix[i][j] > 0:
                pairs.append((matrix[i][j], i + 1, j + 1))
    pairs.sort(reverse=True)
    return [{"from": a, "to": b, "count": c} for c, a, b in pairs[:top_k]]


def _top_label_indices(hist: list[int], top_k: int = 5) -> list[int]:
    return [index + 1 for index, _ in sorted(enumerate(hist), key=lambda kv: kv[1], reverse=True)[:top_k] if _ > 0]


def _top_label_probs(label_probs: list[float], top_k: int = 5) -> list[dict[str, float | int]]:
    ranked = sorted(enumerate(label_probs), key=lambda kv: float(kv[1]), reverse=True)
    return [{"label": index + 1, "mean_prob": float(prob)} for index, prob in ranked[:top_k] if float(prob) > 0.0]


def _top_transition_probs(matrix: list[list[float]], top_k: int = 10) -> list[dict[str, float | int]]:
    pairs: list[tuple[float, int, int]] = []
    for i, row in enumerate(matrix):
        for j, prob in enumerate(row):
            if float(prob) > 0.0:
                pairs.append((float(prob), i + 1, j + 1))
    pairs.sort(reverse=True)
    return [{"from": a, "to": b, "prob": prob} for prob, a, b in pairs[:top_k]]


def _patch2x2_stats(fg_y20: list[list[int]]) -> tuple[list[dict[str, Any]], float, list[dict[str, Any]]]:
    hist: Counter[tuple[int, int, int, int]] = Counter()
    unique_hist: Counter[int] = Counter()
    trans = 0
    total = 0
    for y in range(CANONICAL_SIZE - 1):
        for x in range(CANONICAL_SIZE - 1):
            p = (int(fg_y20[y][x]), int(fg_y20[y][x + 1]), int(fg_y20[y + 1][x]), int(fg_y20[y + 1][x + 1]))
            if any(v == IGNORE_INDEX for v in p):
                continue
            hist[p] += 1
            unique_hist[len(set(p))] += 1
            total += 1
            if len(set(p)) > 1:
                trans += 1
    top = [{"pattern": list(key), "count": value} for key, value in hist.most_common(20)]
    unique_hist_list = [{"unique_label_count": key, "count": value} for key, value in sorted(unique_hist.items())]
    return [{"pattern": item["pattern"], "count": item["count"]} for item in top], (float(trans) / float(max(1, total))), unique_hist_list


def _symmetry_score(mask: list[list[int]], axis: str) -> float:
    hits = 0
    total = 0
    if axis == "lr":
        for y in range(CANONICAL_SIZE):
            for x in range(CANONICAL_SIZE // 2):
                total += 1
                hits += int(int(mask[y][x]) == int(mask[y][CANONICAL_SIZE - 1 - x]))
    else:
        for y in range(CANONICAL_SIZE // 2):
            for x in range(CANONICAL_SIZE):
                total += 1
                hits += int(int(mask[y][x]) == int(mask[CANONICAL_SIZE - 1 - y][x]))
    return float(hits) / float(max(1, total))


def _continuity(mask: list[list[int]], axis: str) -> float:
    same = 0
    total = 0
    for y in range(CANONICAL_SIZE):
        for x in range(CANONICAL_SIZE):
            if axis == "v" and y + 1 < CANONICAL_SIZE:
                total += 1
                same += int(int(mask[y][x]) and int(mask[y + 1][x]))
            if axis == "h" and x + 1 < CANONICAL_SIZE:
                total += 1
                same += int(int(mask[y][x]) and int(mask[y][x + 1]))
    return float(same) / float(max(1, total))


def _center_band_score(mask: list[list[int]]) -> float:
    xs = range(6, 14)
    total = sum(int(v) for row in mask for v in row)
    if total <= 0:
        return 0.0
    center = sum(int(mask[y][x]) for y in range(CANONICAL_SIZE) for x in xs)
    return float(center) / float(total)


def _stripe_scores(mask: list[list[int]]) -> tuple[float, float]:
    row_proj = [sum(int(v) for v in row) / float(CANONICAL_SIZE) for row in mask]
    col_proj = [sum(int(mask[y][x]) for y in range(CANONICAL_SIZE)) / float(CANONICAL_SIZE) for x in range(CANONICAL_SIZE)]
    def score(proj: list[float]) -> float:
        mean = sum(proj) / float(len(proj))
        var = sum((v - mean) ** 2 for v in proj) / float(len(proj))
        return math.sqrt(var)
    return score(col_proj), score(row_proj)


def _graph_descriptors(fg_y20: list[list[int]], fg_mask20: list[list[int]]) -> dict[str, Any]:
    nodes = [(y, x) for y in range(CANONICAL_SIZE) for x in range(CANONICAL_SIZE) if int(fg_mask20[y][x])]
    edges = 0
    for y, x in nodes:
        if y + 1 < CANONICAL_SIZE and int(fg_mask20[y + 1][x]):
            edges += 1
        if x + 1 < CANONICAL_SIZE and int(fg_mask20[y][x + 1]):
            edges += 1
    components = _connected_components(fg_mask20)
    largest = max((len(comp) for comp in components), default=0)
    total_nodes = len(nodes)
    density = float(edges) / float(max(1, 2 * total_nodes))
    eigvals: list[float] | None = None
    try:
        import numpy as np

        if total_nodes > 0:
            index = {node: i for i, node in enumerate(nodes)}
            lap = np.zeros((total_nodes, total_nodes), dtype=np.float32)
            for y, x in nodes:
                i = index[(y, x)]
                for ny, nx in _neighbors(y, x):
                    if int(fg_mask20[ny][nx]):
                        j = index[(ny, nx)]
                        lap[i, i] += 1.0
                        lap[i, j] -= 1.0
            eigvals = [float(v) for v in np.linalg.eigvalsh(lap)[:10].tolist()]
    except Exception:
        eigvals = None
    return {
        "graph_num_nodes": int(total_nodes),
        "graph_num_edges": int(edges),
        "graph_density_4neighbor": float(density),
        "graph_num_components": int(len(components)),
        "graph_largest_component_ratio": float(largest) / float(max(1, total_nodes)),
        "skeleton_like_score": float((largest / float(max(1, total_nodes))) * _continuity(fg_mask20, "v")),
        "graph_laplacian_eigenvalues_smallest_10": eigvals,
    }


def _full_descriptor(raw_y20: list[list[int]], fg_y20: list[list[int]], fg_mask20: list[list[int]], category: str, sample_id: str, source_path: str, local_z: int | None) -> dict[str, Any]:
    label_values = _mask_to_labels(fg_mask20, fg_y20)
    hist = _hist16_from_labels(label_values)
    hist_norm = _normalized([float(v) for v in hist])
    row_proj, col_proj, row_label, col_label = _label_projections(fg_y20)
    th, tv = _transition_matrices(fg_y20)
    total_h = sum(sum(row) for row in th)
    total_v = sum(sum(row) for row in tv)
    same_h = sum(th[i][i] for i in range(NUM_LABELS))
    same_v = sum(tv[i][i] for i in range(NUM_LABELS))
    patch2x2, patch_ratio, patch_unique_hist = _patch2x2_stats(fg_y20)
    comp = _connected_components(fg_mask20)
    comp_sizes = [len(c) for c in comp]
    largest = max(comp_sizes, default=0)
    comp_stats = mask_component_stats(fg_mask20)
    dominant_label = max(range(1, NUM_LABELS + 1), key=lambda i: hist[i - 1]) if any(hist) else 0
    dominant_ratio = float(max(hist)) / float(max(1, sum(hist)))
    com_y = sum(y for y in range(CANONICAL_SIZE) for x in range(CANONICAL_SIZE) if int(fg_mask20[y][x])) / float(max(1, sum(sum(r) for r in fg_mask20)))
    com_x = sum(x for y in range(CANONICAL_SIZE) for x in range(CANONICAL_SIZE) if int(fg_mask20[y][x])) / float(max(1, sum(sum(r) for r in fg_mask20)))
    descriptor = {
        "category": category,
        "sample_id": sample_id,
        "source_path": source_path,
        "local_z": local_z,
        "foreground_area": float(foreground_area(fg_mask20)),
        "foreground_area_ratio": float(foreground_area(fg_mask20)),
        "bbox": {
            "x_min": float(bbox_from_mask([[bool(v) for v in row] for row in fg_mask20])["x0"]),
            "y_min": float(bbox_from_mask([[bool(v) for v in row] for row in fg_mask20])["y0"]),
            "x_max": float(bbox_from_mask([[bool(v) for v in row] for row in fg_mask20])["x1"]),
            "y_max": float(bbox_from_mask([[bool(v) for v in row] for row in fg_mask20])["y1"]),
            "width": float(bbox_from_mask([[bool(v) for v in row] for row in fg_mask20])["w"]),
            "height": float(bbox_from_mask([[bool(v) for v in row] for row in fg_mask20])["h"]),
        },
        "center_of_mass_yx": [float(com_y), float(com_x)],
        "num_connected_components": int(comp_stats["num_components"]),
        "largest_component_ratio": float(comp_stats["largest_component_ratio"]),
        "tiny_component_count": int(comp_stats["tiny_component_count"]),
        "label_diversity": int(label_diversity_on_fg(fg_y20, fg_mask20)),
        "dominant_label": int(dominant_label),
        "dominant_label_ratio": float(dominant_ratio),
        "label_hist_16": hist,
        "label_hist_norm_16": hist_norm,
        "row_fg_projection": row_proj,
        "col_fg_projection": col_proj,
        "row_label_projection_16x20": row_label,
        "col_label_projection_16x20": col_label,
        "transition_h_16x16": th,
        "transition_v_16x16": tv,
        "transition_h_norm_16x16": [[float(v) / float(max(1, total_h)) for v in row] for row in th],
        "transition_v_norm_16x16": [[float(v) / float(max(1, total_v)) for v in row] for row in tv],
        "same_label_h_ratio": float(same_h) / float(max(1, total_h)),
        "same_label_v_ratio": float(same_v) / float(max(1, total_v)),
        "diff_label_h_ratio": float(max(0, total_h - same_h)) / float(max(1, total_h)),
        "diff_label_v_ratio": float(max(0, total_v - same_v)) / float(max(1, total_v)),
        "top_horizontal_transitions": _top_pairs(th, 20),
        "top_vertical_transitions": _top_pairs(tv, 20),
        "patch2x2_unique_label_hist": patch_unique_hist,
        "patch2x2_transition_ratio": float(patch_ratio),
        "top_patch2x2_patterns": patch2x2,
        "vertical_continuity_score": float(_continuity(fg_mask20, "v")),
        "horizontal_continuity_score": float(_continuity(fg_mask20, "h")),
        "left_right_symmetry_score": float(_symmetry_score(fg_mask20, "lr")),
        "top_bottom_symmetry_score": float(_symmetry_score(fg_mask20, "tb")),
        "center_band_score": float(_center_band_score(fg_mask20)),
        "stripe_score_vertical": float(_stripe_scores(fg_mask20)[0]),
        "stripe_score_horizontal": float(_stripe_scores(fg_mask20)[1]),
    }
    descriptor.update(_graph_descriptors(fg_y20, fg_mask20))
    return descriptor


def _score_sample(record: SampleRecord, selection: str) -> dict[str, float]:
    desc = record.descriptor or {}
    score = {
        "random": random.random(),
        "largest_fg": float(desc.get("foreground_area_ratio", 0.0)),
        "diverse_labels": float(desc.get("label_diversity", 0.0)),
        "compact": float(desc.get("largest_component_ratio", 0.0)) - 0.05 * float(desc.get("tiny_component_count", 0.0)),
    }
    return score


def _mixed_select(records: list[SampleRecord], k: int, rng: random.Random) -> list[SampleRecord]:
    if len(records) <= k:
        return records[:]
    picks: list[SampleRecord] = []
    seen: set[str] = set()
    buckets = {
        "random": sorted(records, key=lambda _: rng.random())[:2],
        "largest_fg": sorted(records, key=lambda r: float((r.descriptor or {}).get("foreground_area_ratio", 0.0)), reverse=True)[:1],
        "diverse_labels": sorted(records, key=lambda r: float((r.descriptor or {}).get("label_diversity", 0.0)), reverse=True)[:1],
        "compact": sorted(records, key=lambda r: float((r.descriptor or {}).get("largest_component_ratio", 0.0)) - 0.05 * float((r.descriptor or {}).get("tiny_component_count", 0.0)), reverse=True)[:1],
    }
    if any(r.local_z is not None for r in records):
        z_groups: dict[int, list[SampleRecord]] = defaultdict(list)
        for r in records:
            if r.local_z is not None:
                z_groups[int(r.local_z)].append(r)
        for z in sorted(z_groups):
            if z_groups[z]:
                buckets.setdefault("z", []).append(z_groups[z][0])
                break
    for key in ("random", "largest_fg", "diverse_labels", "compact"):
        for rec in buckets.get(key, []):
            if rec.sample_id not in seen:
                picks.append(rec)
                seen.add(rec.sample_id)
    if "z" in buckets:
        for rec in buckets["z"]:
            if rec.sample_id not in seen:
                picks.append(rec)
                seen.add(rec.sample_id)
    if len(picks) < k:
        for rec in sorted(records, key=lambda _: rng.random()):
            if rec.sample_id not in seen:
                picks.append(rec)
                seen.add(rec.sample_id)
            if len(picks) >= k:
                break
    return picks[:k]


def _pick_samples(records: list[SampleRecord], k: int, selection: str, rng: random.Random) -> list[SampleRecord]:
    if selection == "random":
        return sorted(records, key=lambda _: rng.random())[:k]
    if selection == "largest_fg":
        return sorted(records, key=lambda r: float((r.descriptor or {}).get("foreground_area_ratio", 0.0)), reverse=True)[:k]
    if selection == "diverse_labels":
        return sorted(records, key=lambda r: float((r.descriptor or {}).get("label_diversity", 0.0)), reverse=True)[:k]
    if selection == "compact":
        return sorted(records, key=lambda r: float((r.descriptor or {}).get("largest_component_ratio", 0.0)) - 0.05 * float((r.descriptor or {}).get("tiny_component_count", 0.0)), reverse=True)[:k]
    return _mixed_select(records, k, rng)


def _mean_matrix(mats: list[list[list[float]]]) -> list[list[float]]:
    if not mats:
        return [[0.0 for _ in range(NUM_LABELS)] for _ in range(NUM_LABELS)]
    out = [[0.0 for _ in range(len(mats[0][0]))] for _ in range(len(mats[0]))]
    for mat in mats:
        for i, row in enumerate(mat):
            for j, value in enumerate(row):
                out[i][j] += float(value)
    scale = float(len(mats))
    return [[value / scale for value in row] for row in out]


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    out = [0.0 for _ in range(len(vectors[0]))]
    for vec in vectors:
        for i, value in enumerate(vec):
            out[i] += float(value)
    return [v / float(len(vectors)) for v in out]


def _safe_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / float(len(values))
    return math.sqrt(sum((v - mean) ** 2 for v in values) / float(len(values)))


def _descriptor_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    def flatten(x: Any) -> list[float]:
        if isinstance(x, dict):
            vals: list[float] = []
            for v in x.values():
                vals.extend(flatten(v))
            return vals
        if isinstance(x, list):
            vals: list[float] = []
            for v in x:
                vals.extend(flatten(v))
            return vals
        if isinstance(x, (int, float)):
            return [float(x)]
        return []
    av = flatten({k: a[k] for k in ("label_hist_norm_16", "transition_h_norm_16x16", "transition_v_norm_16x16", "row_fg_projection", "col_fg_projection", "largest_component_ratio", "tiny_component_count", "vertical_continuity_score", "horizontal_continuity_score", "center_band_score") if k in a})
    bv = flatten({k: b[k] for k in ("label_hist_norm_16", "transition_h_norm_16x16", "transition_v_norm_16x16", "row_fg_projection", "col_fg_projection", "largest_component_ratio", "tiny_component_count", "vertical_continuity_score", "horizontal_continuity_score", "center_band_score") if k in b})
    dim = min(len(av), len(bv))
    if dim == 0:
        return 0.0
    return math.sqrt(sum((av[i] - bv[i]) ** 2 for i in range(dim)) / float(dim))


def _write_heatmap_csv(path: Path, matrix: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(matrix)


def _grid_to_str(grid: list[list[int]]) -> str:
    return "\n".join(" ".join(f"{int(v):2d}" for v in row) for row in grid)


def _make_grid_image(grids: list[list[list[int]]], labels: list[str], output_path: Path, *, cell_size: int = 18, cols: int = 3, mode: str = "labels") -> None:
    try:
        import numpy as np
    except Exception:
        save_json(output_path.with_suffix(".json"), {"labels": labels, "grids": grids})
        return
    cols = max(1, cols)
    rows = (len(grids) + cols - 1) // cols
    canvas = np.zeros((rows * CANONICAL_SIZE * cell_size, cols * CANONICAL_SIZE * cell_size, 3), dtype=np.uint8)
    canvas[:, :] = np.asarray((255, 0, 0), dtype=np.uint8)
    palette = [
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
    for idx, grid in enumerate(grids):
        arr = np.asarray(grid, dtype=np.int32)
        rgb = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE, 3), dtype=np.uint8)
        for y in range(CANONICAL_SIZE):
            for x in range(CANONICAL_SIZE):
                val = int(arr[y, x])
                if mode == "mask":
                    rgb[y, x] = (0, 255, 0) if val > 0 else (255, 0, 0)
                else:
                    if val <= 0:
                        rgb[y, x] = (255, 0, 0)
                    else:
                        rgb[y, x] = palette[min(val, len(palette) - 1)]
        x0 = (idx % cols) * CANONICAL_SIZE * cell_size
        y0 = (idx // cols) * CANONICAL_SIZE * cell_size
        tile = np.repeat(np.repeat(rgb, cell_size, axis=0), cell_size, axis=1)
        canvas[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1], :] = tile
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_png_rgb(output_path, canvas)


def _save_heatmap_png(matrix: list[list[float]], output_path: Path, title: str) -> None:
    try:
        import numpy as np
    except Exception:
        save_json(output_path.with_suffix(".json"), {"title": title, "matrix": matrix})
        return
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.size == 0:
        arr = np.zeros((NUM_LABELS, NUM_LABELS), dtype=np.float32)
    mn = float(arr.min()) if arr.size else 0.0
    mx = float(arr.max()) if arr.size else 1.0
    denom = max(1e-8, mx - mn)
    norm = (arr - mn) / denom
    heat = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    heat[..., 0] = np.clip((norm * 255.0).round(), 0, 255).astype(np.uint8)
    heat[..., 1] = np.clip(((1.0 - np.abs(norm - 0.5) * 2.0) * 255.0).round(), 0, 255).astype(np.uint8)
    heat[..., 2] = np.clip(((1.0 - norm) * 255.0).round(), 0, 255).astype(np.uint8)
    heat_big = np.repeat(np.repeat(heat, 12, axis=0), 12, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_png_rgb(output_path, heat_big)


def _write_png_rgb(output_path: Path, array: Any) -> None:
    import numpy as np

    arr = np.asarray(array, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("PNG writer expects HxWx3 uint8 array.")
    height, width, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(height))
    png = bytearray()
    png.extend(b"\x89PNG\r\n\x1a\n")

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png.extend(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
    png.extend(chunk(b"IDAT", zlib.compress(raw, level=9)))
    png.extend(chunk(b"IEND", b""))
    output_path.write_bytes(bytes(png))


def _format_matrix_section(title: str, matrices: list[list[list[int]]], labels: list[str]) -> str:
    parts = [f"## {title}"]
    for label, grid in zip(labels, matrices):
        parts.append(f"### {label}")
        parts.append("```text")
        parts.append(_grid_to_str(grid))
        parts.append("```")
    return "\n".join(parts)


def _flatten_numeric(value: Any) -> list[float]:
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_numeric(item))
        return out
    if isinstance(value, (int, float)):
        return [float(value)]
    return []


def _make_sample_row(rec: SampleRecord, onehot_npz_path: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": rec.sample_id,
        "category": rec.category,
        "source_path": rec.source_path,
        "local_z": rec.local_z,
        "raw_y20": rec.raw_y20,
        "fg_y20": rec.fg_y20,
        "fg_mask": rec.fg_mask20,
        "fg_mask20": rec.fg_mask20,
        "onehot_npz_path": onehot_npz_path,
    }
    if rec.descriptor:
        row.update(rec.descriptor)
        row["descriptor"] = rec.descriptor
    return row


def _problem_statement_lines() -> list[str]:
    return [
        "## Problem statement",
        "- Y in {0..16}^{20x20}",
        "- 0 is background",
        "- 1..16 are foreground knitting structure tokens",
        "- The target task is category-only generation.",
        "- no prototype / instruction / rendering / pattern-viz / real at inference",
        "- At inference time, do not use prototype, instruction, rendering, pattern-viz, or real inputs.",
        "- instruction17 may be used only for training, supervision, cache statistics, exported descriptors, and evaluation.",
        "- The goal is to infer a generalizable mathematical structure prior across categories.",
    ]


def _questions_for_gpt_pro_lines() -> list[str]:
    return [
        "## Questions for GPT Pro",
        "Given these cross-category 20x20 instruction17 matrices and descriptors, infer a mathematical structure prior that can guide category-only foreground generation without using prototype, instruction, rendering, pattern-viz, or real inputs at inference.",
        "",
        "Please address:",
        "- recommended general prior representation",
        "- category-specific parameters",
        "- exact mathematical definitions",
        "- energy function E(Y | category,z)",
        "- rerank score for generated candidates",
        "- how to estimate the prior from training data",
        "- how to inject or use it in current foreground planner",
        "- what should replace or supplement KMeans local_z",
        "- how to keep inference category-only",
        "- implementation steps in struct_foreground_v1",
        "- ablation plan across categories",
    ]


def _save_onehot_npz(rec: SampleRecord, path: Path) -> None:
    import numpy as np

    raw = np.asarray(rec.raw_y20, dtype=np.int16)
    fg = np.asarray(rec.fg_y20, dtype=np.int16)
    mask = np.asarray(rec.fg_mask20, dtype=np.uint8)
    onehot = np.zeros((NUM_LABELS, CANONICAL_SIZE, CANONICAL_SIZE), dtype=np.uint8)
    for label in range(1, NUM_LABELS + 1):
        onehot[label - 1] = (raw == label).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, raw_y20=raw, fg_y20=fg, fg_mask=mask, onehot_16x20x20=onehot)


def _synthetic_grid(category: str, variant: int) -> list[list[int]]:
    grid = [[0 for _ in range(CANONICAL_SIZE)] for _ in range(CANONICAL_SIZE)]
    if category == "CableLike":
        for y in range(3, 18):
            width = 2 + ((y + variant) % 2)
            left = 7 + ((y // 4 + variant) % 2)
            right = 11 - ((y // 5 + variant) % 2)
            for dx in range(width):
                grid[y][left + dx] = 3 + ((y + dx + variant) % 4)
                grid[y][right + dx] = 7 + ((y + dx + variant) % 4)
            if y % 5 == variant % 5:
                for x in range(left + width, right):
                    grid[y][x] = 5 + ((x + y) % 3)
    else:
        for y in range(4, 17, 3):
            for x in range(4, 17):
                grid[y][x] = 2 + ((x + variant) % 5)
        for x in range(4 + variant % 2, 17, 3):
            for y in range(4, 17):
                grid[y][x] = 9 + ((y + variant) % 5)
        for y in range(5, 16, 4):
            for x in range(5, 16, 4):
                grid[y][x] = 15
    return grid


def _synthetic_records() -> dict[str, list[SampleRecord]]:
    out: dict[str, list[SampleRecord]] = {"CableLike": [], "MeshLike": []}
    for category in out:
        for idx in range(2):
            sample_id = f"{category}_{idx:02d}"
            raw_y20 = _synthetic_grid(category, idx)
            fg_y20, fg_mask20 = _canonicalize_full_masked(raw_y20)
            rec = SampleRecord(
                sample_id=sample_id,
                category=category,
                input_path="synthetic",
                target_path="synthetic",
                index_path="synthetic",
                raw_y20=raw_y20,
                fg_y20=fg_y20,
                fg_mask20=fg_mask20,
                local_z=idx,
                source_path=f"synthetic://{sample_id}",
            )
            rec.descriptor = _full_descriptor(raw_y20, fg_y20, fg_mask20, category, sample_id, str(rec.source_path), rec.local_z)
            out[category].append(rec)
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rng = random.Random(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    skipped: list[dict[str, str]] = []
    warnings: list[str] = []
    category_filter = [c.strip() for c in args.categories.split(",") if c.strip()] if args.categories else None
    if args.synthetic_smoke:
        export_records = _synthetic_records()
        categories_available = sorted(export_records)
        categories = category_filter or categories_available
        export_records = {cat: export_records.get(cat, []) for cat in categories}
    else:
        if args.manifest is None:
            raise SystemExit("--manifest is required unless --synthetic-smoke is set.")
        rows = _load_manifest(Path(args.manifest))
        manifest_root = _infer_manifest_root(Path(args.manifest), rows)
        cache_by_id: dict[str, dict[str, Any]] = {}
        if args.cache is not None and Path(args.cache).exists():
            torch = _require_torch()
            cache_payload = torch.load(Path(args.cache), map_location="cpu")
            cache_by_id = {str(item["sample_id"]): item for item in cache_payload.get("items", [])}
        categories_available = sorted({str(row["category"]) for row in rows})
        categories = category_filter or categories_available
        export_records = {cat: [] for cat in categories}
        for row in rows:
            sample_id = str(row["sample_id"])
            category = str(row["category"])
            if category not in export_records:
                continue
            try:
                target_path = _resolve_path(str(row["target_path"]), manifest_root)
                cache_item = cache_by_id.get(sample_id, {})
                raw_y20: list[list[int]] | None = None
                fg_y20: list[list[int]] | None = None
                fg_mask20: list[list[int]] | None = None
                if cache_item:
                    if "original_y20" in cache_item:
                        raw_y20 = _grid_to_list(cache_item["original_y20"])
                    if "fg_y20" in cache_item:
                        fg_y20 = _grid_to_list(cache_item["fg_y20"])
                    if "fg_mask20" in cache_item:
                        fg_mask20 = _grid_to_list(cache_item["fg_mask20"])
                if raw_y20 is None:
                    raw_y20 = _grid_to_list(load_label_grid(target_path, sample_id=sample_id))
                if len(raw_y20) != CANONICAL_SIZE or any(len(r) != CANONICAL_SIZE for r in raw_y20):
                    raise ValueError("raw_y20 must be [20,20].")
                if any(v < 0 or v > 16 for rowv in raw_y20 for v in rowv):
                    raise ValueError("raw_y20 value out of range 0..16.")
                if fg_y20 is None or fg_mask20 is None:
                    fg_y20, fg_mask20 = _canonicalize_full_masked(raw_y20)
                else:
                    validate_foreground_labels(fg_y20, fg_mask20, canonical_size=CANONICAL_SIZE, context=f"cache[{sample_id}]")
                record = SampleRecord(
                    sample_id=sample_id,
                    category=category,
                    input_path=str(row["input_path"]),
                    target_path=str(row["target_path"]),
                    index_path=str(row["index_path"]),
                    raw_y20=raw_y20,
                    fg_y20=fg_y20,
                    fg_mask20=fg_mask20,
                    local_z=int(cache_by_id.get(sample_id, {}).get("local_z")) if sample_id in cache_by_id and str(cache_by_id[sample_id].get("local_z", "")) != "" else None,
                    source_path=str(target_path),
                )
                record.descriptor = _full_descriptor(raw_y20, fg_y20, fg_mask20, category, sample_id, str(target_path), record.local_z)
                export_records[category].append(record)
            except Exception as exc:
                skipped.append({"sample_id": sample_id, "category": category, "reason": str(exc)})
                warnings.append(f"{sample_id}: {exc}")
    selected_by_category: dict[str, list[SampleRecord]] = {}
    total_limit = int(args.max_total_samples)
    per_category = max(1, int(args.samples_per_category))
    if categories:
        per_category = min(per_category, max(1, total_limit // max(1, len(categories))))
    for category in categories:
        selected_by_category[category] = _pick_samples(export_records.get(category, []), per_category, str(args.selection), rng)
    total_selected = sum(len(v) for v in selected_by_category.values())
    if total_selected > total_limit:
        flat = [(cat, rec) for cat, recs in selected_by_category.items() for rec in recs]
        flat = flat[:total_limit]
        new_map: dict[str, list[SampleRecord]] = {cat: [] for cat in categories}
        for cat, rec in flat:
            new_map[cat].append(rec)
        selected_by_category = new_map
        total_selected = total_limit
    # category summaries
    category_summaries: dict[str, dict[str, Any]] = {}
    samples_index: list[dict[str, Any]] = []
    compact_dir = output_dir / "compact_for_gpt"
    categories_dir = output_dir / "categories"
    categories_dir.mkdir(parents=True, exist_ok=True)
    compact_dir.mkdir(parents=True, exist_ok=True)
    for category, records in selected_by_category.items():
        rows_desc = [rec.descriptor or {} for rec in records]
        if not rows_desc:
            continue
        summary = {
            "num_samples_available": len(export_records.get(category, [])),
            "num_samples_exported": len(records),
            "category_label_hist_norm_16_mean": _mean_vector([r["label_hist_norm_16"] for r in rows_desc if "label_hist_norm_16" in r]),
            "category_transition_h_norm_16x16_mean": _mean_matrix([r["transition_h_norm_16x16"] for r in rows_desc if "transition_h_norm_16x16" in r]),
            "category_transition_v_norm_16x16_mean": _mean_matrix([r["transition_v_norm_16x16"] for r in rows_desc if "transition_v_norm_16x16" in r]),
            "category_row_fg_projection_mean": _mean_vector([r["row_fg_projection"] for r in rows_desc if "row_fg_projection" in r]),
            "category_col_fg_projection_mean": _mean_vector([r["col_fg_projection"] for r in rows_desc if "col_fg_projection" in r]),
            "category_foreground_area_ratio_mean": sum(float(r.get("foreground_area_ratio", 0.0)) for r in rows_desc) / float(len(rows_desc)),
            "category_foreground_area_ratio_std": _safe_std([float(r.get("foreground_area_ratio", 0.0)) for r in rows_desc]),
            "category_label_diversity_mean": sum(float(r.get("label_diversity", 0.0)) for r in rows_desc) / float(len(rows_desc)),
            "category_label_diversity_std": _safe_std([float(r.get("label_diversity", 0.0)) for r in rows_desc]),
            "category_dominant_label_ratio_mean": sum(float(r.get("dominant_label_ratio", 0.0)) for r in rows_desc) / float(len(rows_desc)),
            "category_dominant_label_ratio_std": _safe_std([float(r.get("dominant_label_ratio", 0.0)) for r in rows_desc]),
            "category_vertical_continuity_score_mean": sum(float(r.get("vertical_continuity_score", 0.0)) for r in rows_desc) / float(len(rows_desc)),
            "category_vertical_continuity_score_std": _safe_std([float(r.get("vertical_continuity_score", 0.0)) for r in rows_desc]),
            "category_horizontal_continuity_score_mean": sum(float(r.get("horizontal_continuity_score", 0.0)) for r in rows_desc) / float(len(rows_desc)),
            "category_horizontal_continuity_score_std": _safe_std([float(r.get("horizontal_continuity_score", 0.0)) for r in rows_desc]),
            "category_largest_component_ratio_mean": sum(float(r.get("largest_component_ratio", 0.0)) for r in rows_desc) / float(len(rows_desc)),
            "category_largest_component_ratio_std": _safe_std([float(r.get("largest_component_ratio", 0.0)) for r in rows_desc]),
            "category_tiny_component_count_mean": sum(float(r.get("tiny_component_count", 0.0)) for r in rows_desc) / float(len(rows_desc)),
            "category_tiny_component_count_std": _safe_std([float(r.get("tiny_component_count", 0.0)) for r in rows_desc]),
            "representative_sample_ids": [rec.sample_id for rec in records],
        }
        summary["top_labels"] = _top_label_probs(summary["category_label_hist_norm_16_mean"], top_k=5)
        summary["top_horizontal_transitions"] = _top_transition_probs(summary["category_transition_h_norm_16x16_mean"], top_k=10)
        summary["top_vertical_transitions"] = _top_transition_probs(summary["category_transition_v_norm_16x16_mean"], top_k=10)
        if any(rec.local_z is not None for rec in records):
            local_z_vals = [int(rec.local_z) for rec in records if rec.local_z is not None]
            summary["unique_local_z_count"] = len(set(local_z_vals))
            summary["selected_local_z_list"] = sorted(set(local_z_vals))
            summary["per_z_sample_count"] = dict(Counter(local_z_vals))
            z_summary: dict[str, Any] = {}
            for z in sorted(set(local_z_vals)):
                z_rows = [rec.descriptor or {} for rec in records if rec.local_z == z]
                z_summary[str(z)] = {
                    "num_samples": len(z_rows),
                    "foreground_area_ratio_mean": sum(float(r.get("foreground_area_ratio", 0.0)) for r in z_rows) / float(max(1, len(z_rows))),
                    "label_diversity_mean": sum(float(r.get("label_diversity", 0.0)) for r in z_rows) / float(max(1, len(z_rows))),
                    "vertical_continuity_mean": sum(float(r.get("vertical_continuity_score", 0.0)) for r in z_rows) / float(max(1, len(z_rows))),
                    "horizontal_continuity_mean": sum(float(r.get("horizontal_continuity_score", 0.0)) for r in z_rows) / float(max(1, len(z_rows))),
                }
            summary["z_level_descriptor_summary"] = z_summary
        category_summaries[category] = summary
        category_dir = categories_dir / category
        csv_dir = category_dir / "csv"
        vis_dir = category_dir / "visual"
        category_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)
        selected_raw = [rec.raw_y20 for rec in records]
        selected_labels = [rec.sample_id for rec in records]
        _make_grid_image(selected_raw, selected_labels, vis_dir / f"{category}_selected_raw_grid.png", mode="labels")
        selected_fg_masks = [rec.fg_mask20 for rec in records]
        _make_grid_image(selected_fg_masks, selected_labels, vis_dir / f"{category}_fg_mask_grid.png", mode="mask")
        _write_heatmap_csv(csv_dir / f"{category}_mean_transition_h.csv", summary["category_transition_h_norm_16x16_mean"])
        _write_heatmap_csv(csv_dir / f"{category}_mean_transition_v.csv", summary["category_transition_v_norm_16x16_mean"])
        _save_heatmap_png(summary["category_transition_h_norm_16x16_mean"], vis_dir / f"{category}_transition_h_heatmap.png", f"{category} transition_h")
        _save_heatmap_png(summary["category_transition_v_norm_16x16_mean"], vis_dir / f"{category}_transition_v_heatmap.png", f"{category} transition_v")
        sample_rows = []
        for rec in records:
            onehot_npz_path = None
            if args.save_onehot:
                npz_path = category_dir / "samples_npz" / f"{rec.sample_id}.npz"
                _save_onehot_npz(rec, npz_path)
                onehot_npz_path = str(npz_path)
            sample_rows.append(_make_sample_row(rec, onehot_npz_path=onehot_npz_path))
            samples_index.append({"sample_id": rec.sample_id, "category": rec.category, "local_z": rec.local_z, "source_path": rec.source_path, "onehot_npz_path": onehot_npz_path})
        save_json(category_dir / f"{category}_samples.json", sample_rows)
        save_json(category_dir / f"{category}_summary.json", summary)
        save_jsonl(category_dir / f"{category}_samples.jsonl", sample_rows)
        compact_md = [
            f"# {category}",
            "",
            f"- exported_samples: {len(records)}",
            f"- foreground_area_ratio_mean: {summary['category_foreground_area_ratio_mean']:.4f}",
            f"- label_diversity_mean: {summary['category_label_diversity_mean']:.4f}",
            f"- dominant_label_ratio_mean: {summary['category_dominant_label_ratio_mean']:.4f}",
            f"- vertical_continuity_mean: {summary['category_vertical_continuity_score_mean']:.4f}",
            f"- horizontal_continuity_mean: {summary['category_horizontal_continuity_score_mean']:.4f}",
            f"- largest_component_ratio_mean: {summary['category_largest_component_ratio_mean']:.4f}",
            f"- top_labels: {summary['top_labels']}",
            f"- top_horizontal_transitions: {summary['top_horizontal_transitions'][:5]}",
            f"- top_vertical_transitions: {summary['top_vertical_transitions'][:5]}",
            "",
            _format_matrix_section("Representative matrices", [rec.raw_y20 for rec in records[: min(6, len(records))]], [rec.sample_id for rec in records[: min(6, len(records))]]),
        ]
        (compact_dir / f"{category}_compact.md").write_text("\n".join(compact_md), encoding="utf-8")
        packet_md = [f"# {category} pro packet", "", "## Compact descriptor table"]
        for rec in records[: min(6, len(records))]:
            d = rec.descriptor or {}
            packet_md.append(f"- {rec.sample_id}: area={d.get('foreground_area_ratio', 0.0):.4f}, div={d.get('label_diversity', 0)}, dom={d.get('dominant_label', 0)}")
        packet_md.append("")
        packet_md.append(_format_matrix_section("Raw y20", [rec.raw_y20 for rec in records[: min(6, len(records))]], [rec.sample_id for rec in records[: min(6, len(records))]]))
        packet_md.append("")
        packet_md.append(_format_matrix_section("Foreground y20", [rec.fg_y20 for rec in records[: min(6, len(records))]], [rec.sample_id for rec in records[: min(6, len(records))]]))
        packet_md.append("")
        packet_md.append("## Top transitions")
        if records and records[0].descriptor:
            packet_md.append("### Horizontal")
            for item in records[0].descriptor["top_horizontal_transitions"][:10]:
                packet_md.append(f"- {item}")
            packet_md.append("### Vertical")
            for item in records[0].descriptor["top_vertical_transitions"][:10]:
                packet_md.append(f"- {item}")
        packet_md.append("")
        packet_md.append("## Top 2x2 motifs")
        if records and records[0].descriptor:
            for item in records[0].descriptor["top_patch2x2_patterns"][:10]:
                packet_md.append(f"- {item}")
        (category_dir / f"{category}_pro_packet.md").write_text("\n".join(packet_md), encoding="utf-8")
    all_categories_compact = [
        "# All categories compact packet",
        "",
        *_problem_statement_lines(),
        "",
        "## Categories included",
        *[f"- {cat}: {len(selected_by_category.get(cat, []))} samples" for cat in categories],
        "",
        "## Per-category compact summary table",
    ]
    for cat in categories:
        s = category_summaries.get(cat, {})
        all_categories_compact.append(
            f"- {cat}: foreground_area_ratio_mean={s.get('category_foreground_area_ratio_mean', 0.0):.4f}, label_diversity_mean={s.get('category_label_diversity_mean', 0.0):.4f}, dominant_label_ratio_mean={s.get('category_dominant_label_ratio_mean', 0.0):.4f}, vertical_continuity_score_mean={s.get('category_vertical_continuity_score_mean', 0.0):.4f}, horizontal_continuity_score_mean={s.get('category_horizontal_continuity_score_mean', 0.0):.4f}, largest_component_ratio_mean={s.get('category_largest_component_ratio_mean', 0.0):.4f}, top_labels={s.get('top_labels', [])}, top_horizontal_transitions={s.get('top_horizontal_transitions', [])[:5]}, top_vertical_transitions={s.get('top_vertical_transitions', [])[:5]}"
        )
    all_categories_compact.extend([
        "",
        "## Representative matrices",
    ])
    for cat in categories:
        recs = selected_by_category.get(cat, [])
        if not recs:
            continue
        all_categories_compact.append(f"### {cat}")
        for rec in recs[:2]:
            all_categories_compact.append(f"#### {rec.sample_id}")
            all_categories_compact.append("```text")
            all_categories_compact.append(_grid_to_str(rec.raw_y20))
            all_categories_compact.append("```")
    all_categories_compact.extend([
        "",
        "## Cross-category observations",
        "- This section reports only objective exported descriptor summaries.",
        "- Compare label histograms, horizontal and vertical transition probabilities, row and column projections, connected components, 2x2 motifs, continuity scores, and graph descriptors across categories.",
        "- Treat differences as empirical evidence to be modeled; do not assume they are causal without validation.",
        "",
        *_questions_for_gpt_pro_lines(),
    ])
    (compact_dir / "all_categories_compact_packet.md").write_text("\n".join(all_categories_compact), encoding="utf-8")
    distance_matrix: dict[str, dict[str, float]] = {a: {} for a in categories}
    for a in categories:
        for b in categories:
            if a == b:
                distance_matrix[a][b] = 0.0
            else:
                distance_matrix[a][b] = _descriptor_distance(category_summaries.get(a, {}), category_summaries.get(b, {}))
    pairs = sorted(((distance_matrix[a][b], a, b) for i, a in enumerate(categories) for b in categories[i + 1 :]), key=lambda t: t[0])
    discriminative = sorted(
        (
            (abs(float(category_summaries[a].get(metric, 0.0)) - float(category_summaries[b].get(metric, 0.0))), metric, a, b)
            for metric in ("category_foreground_area_ratio_mean", "category_label_diversity_mean", "category_vertical_continuity_score_mean", "category_horizontal_continuity_score_mean", "category_largest_component_ratio_mean")
            for a in categories
            for b in categories
            if a < b and a in category_summaries and b in category_summaries
        ),
        reverse=True,
    )[:20]
    cross_summary = {
        "categories": categories,
        "per_category": category_summaries,
        "distance_matrix": distance_matrix,
        "most_similar_pairs": [{"category_a": a, "category_b": b, "distance": d} for d, a, b in pairs[:10]],
        "most_different_pairs": [{"category_a": a, "category_b": b, "distance": d} for d, a, b in pairs[-10:]][::-1],
        "descriptor_dimensions": {
            "label_hist_norm_16": 16,
            "transition_h_norm_16x16": 256,
            "transition_v_norm_16x16": 256,
            "row_fg_projection": 20,
            "col_fg_projection": 20,
            "shape_components": 5,
        },
        "descriptor_definitions": [
            "foreground_area_ratio: foreground pixels / 400",
            "label_hist_norm_16: normalized foreground label histogram over labels 1..16",
            "transition_h/v_norm_16x16: normalized adjacent foreground label transitions",
            "row/col_fg_projection: foreground density per row/column",
            "connected_component descriptors: component count, largest component ratio, tiny component count",
        ],
        "discriminative_descriptor_pairs": [{"metric": metric, "category_a": a, "category_b": b, "gap": gap} for gap, metric, a, b in discriminative],
        "skipped_samples": skipped,
        "notes": [
            "This packet is read-only and derived from real instruction17 / fg_y20 data.",
            "full_masked is preserved: no crop, no resize, no padding, original 20x20 coordinates remain intact.",
        ],
    }
    save_json(output_dir / "cross_category_summary.json", cross_summary)
    cross_md = [
        "# Cross-category summary",
        "",
        "## Categories",
        *[f"- {cat}" for cat in categories],
        "",
        "## Similar pairs",
        *[f"- {p['category_a']} <-> {p['category_b']}: {p['distance']:.6f}" for p in cross_summary["most_similar_pairs"]],
        "",
        "## Different pairs",
        *[f"- {p['category_a']} <-> {p['category_b']}: {p['distance']:.6f}" for p in cross_summary["most_different_pairs"]],
        "",
        "## Discriminative descriptors",
        *[f"- {item['metric']}: {item['category_a']} vs {item['category_b']} gap={item['gap']:.6f}" for item in cross_summary["discriminative_descriptor_pairs"][:10]],
    ]
    (output_dir / "cross_category_brief.md").write_text("\n".join(cross_md), encoding="utf-8")
    main_md = [
        "# Pro Matrix Packet",
        "",
        *_problem_statement_lines(),
        "",
        "## Categories included",
        *[f"- {cat}: exported {len(selected_by_category.get(cat, []))} / available {len(export_records.get(cat, []))}" for cat in categories],
        "",
        "## Per-category compact summary table",
    ]
    for cat in categories:
        s = category_summaries.get(cat, {})
        main_md.append(
            f"- {cat}: area={s.get('category_foreground_area_ratio_mean', 0.0):.4f}, div={s.get('category_label_diversity_mean', 0.0):.4f}, dom={s.get('category_dominant_label_ratio_mean', 0.0):.4f}, v={s.get('category_vertical_continuity_score_mean', 0.0):.4f}, h={s.get('category_horizontal_continuity_score_mean', 0.0):.4f}, lcr={s.get('category_largest_component_ratio_mean', 0.0):.4f}"
        )
    main_md.extend([
        "",
        "## Representative matrices",
    ])
    for cat in categories:
        recs = selected_by_category.get(cat, [])
        if not recs:
            continue
        main_md.append(f"### {cat}")
        for rec in recs[:2]:
            main_md.append(f"- {rec.sample_id}")
            main_md.append("```text")
            main_md.append(_grid_to_str(rec.raw_y20))
            main_md.append("```")
    main_md.extend([
        "",
        "## Cross-category observations",
        "- The packet intentionally reports objective descriptor summaries only.",
        "- Use cross-category distances, label diversity, transition entropy, continuity, and component ratios to derive a general prior.",
        "",
        "## Questions for GPT Pro",
        "请基于以上多 category 的 20x20 instruction17 matrices 和 descriptors，从矩阵数学角度推导一种比 KMeans centroid 更泛化的 category-specific knitting grammar prior。请重点考虑：",
        "1. label transition matrices A_h, A_v",
        "2. local 2x2 motif distribution",
        "3. graph Laplacian / spectral descriptors",
        "4. NMF / dictionary basis over one-hot matrices",
        "5. row/col projection and center-band constraints",
        "6. connected component constraints",
        "7. category-only inference 下如何采样 diverse structure modes",
        "",
        "请输出：",
        "- recommended general prior representation",
        "- category-specific parameters",
        "- exact mathematical definitions",
        "- energy function E(Y | category,z)",
        "- rerank score for generated candidates",
        "- how to estimate the prior from training data",
        "- how to inject or use it in current foreground planner",
        "- what should replace or supplement KMeans local_z",
        "- how to keep inference category-only",
        "- implementation steps in struct_foreground_v1",
        "- ablation plan across Cable1/Cable2/Mesh/Tuck/Hem/etc.",
    ])
    if "## Questions for GPT Pro" in main_md:
        question_index = main_md.index("## Questions for GPT Pro")
        prefix_end = question_index - 1 if question_index > 0 and main_md[question_index - 1] == "" else question_index
        main_md = main_md[:prefix_end] + ["", *_questions_for_gpt_pro_lines()]
    (output_dir / "all_categories_pro_packet.md").write_text("\n".join(main_md), encoding="utf-8")
    save_json(output_dir / "samples_index.json", samples_index)
    readme_md = [
        "# Pro Matrix Packet",
        "",
        f"- output_dir: {output_dir}",
        f"- categories: {categories}",
        f"- samples_per_category: {per_category}",
        f"- total_samples_exported: {total_selected}",
        f"- skipped_samples: {len(skipped)}",
        "",
        "This directory contains only read-only exports from real instruction17 / fg_y20 data.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_md), encoding="utf-8")
    save_json(output_dir / "README.md.json", {
        "output_dir": str(output_dir),
        "categories": categories,
        "samples_per_category": per_category,
        "total_samples_exported": total_selected,
        "skipped_samples": skipped,
        "all_categories_pro_packet": str(output_dir / "all_categories_pro_packet.md"),
        "compact_for_gpt": str(compact_dir / "all_categories_compact_packet.md"),
    })
    print(f"output_dir: {output_dir}")
    print(f"categories: {categories}")
    print(f"samples_per_category: {per_category}")
    print(f"total samples exported: {total_selected}")
    print(f"skipped samples: {len(skipped)}")
    print(f"all_categories_pro_packet path: {output_dir / 'all_categories_pro_packet.md'}")
    print(f"compact_for_gpt path: {compact_dir / 'all_categories_compact_packet.md'}")
    if warnings:
        print("warnings:")
        for item in warnings[:20]:
            print(f"- {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
