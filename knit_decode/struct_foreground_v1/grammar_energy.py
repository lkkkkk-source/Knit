from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .utils import IGNORE_INDEX, bbox_from_mask, foreground_area, save_json


CANONICAL_SIZE = 20
NUM_LABELS = 16
GRAMMAR_BANK_SCHEMA_VERSION = "foreground_v1_grammar_bank_v1"


DEFAULT_GRAMMAR_BANK_CONFIG = {
    "enabled": True,
    "motif_top_k": 200,
    "smoothing_alpha": 0.001,
    "tiny_component_threshold": 2,
    "min_mode_samples": 3,
    "occupancy_transition_condition_nonempty": True,
}


DEFAULT_RERANK_WEIGHTS = {
    "area": 0.8,
    "conn": 0.7,
    "hist": 0.8,
    "rowcol": 0.7,
    "trans": 1.2,
    "occ_trans": 0.8,
    "motif": 1.3,
    "div": 0.6,
    "dom": 0.8,
    "graph": 0.0,
    "nmf": 0.0,
}


def _cfg(config: dict[str, Any] | None, key: str) -> Any:
    merged = dict(DEFAULT_GRAMMAR_BANK_CONFIG)
    if isinstance(config, dict):
        merged.update(config)
    return merged[key]


def _merged_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    merged.update(DEFAULT_GRAMMAR_BANK_CONFIG)
    if isinstance(config, dict):
        merged.update(config)
    return merged


def _as_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _validate_y20(y20: list[list[int]]) -> None:
    if len(y20) != CANONICAL_SIZE or any(len(row) != CANONICAL_SIZE for row in y20):
        raise ValueError("Expected y20 shape [20,20].")
    bad = [int(v) for row in y20 for v in row if not (0 <= int(v) <= NUM_LABELS)]
    if bad:
        raise ValueError(f"Expected y20 values in 0..16, found {bad[:5]}.")


def _mask_from_raw(y20: list[list[int]]) -> list[list[int]]:
    return [[1 if int(v) != 0 else 0 for v in row] for row in y20]


def _fg_from_raw(y20: list[list[int]]) -> list[list[int]]:
    return [[int(v) if int(v) != 0 else IGNORE_INDEX for v in row] for row in y20]


def _raw_from_fg(fg_y20: list[list[int]]) -> list[list[int]]:
    return [[int(v) if 1 <= int(v) <= NUM_LABELS else 0 for v in row] for row in fg_y20]


def _components(mask: list[list[int]], tiny_threshold: int) -> tuple[int, float, int]:
    visited = [[False for _ in range(CANONICAL_SIZE)] for _ in range(CANONICAL_SIZE)]
    sizes: list[int] = []
    for y in range(CANONICAL_SIZE):
        for x in range(CANONICAL_SIZE):
            if visited[y][x] or not int(mask[y][x]):
                continue
            stack = [(y, x)]
            visited[y][x] = True
            size = 0
            while stack:
                cy, cx = stack.pop()
                size += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < CANONICAL_SIZE and 0 <= nx < CANONICAL_SIZE and not visited[ny][nx] and int(mask[ny][nx]):
                        visited[ny][nx] = True
                        stack.append((ny, nx))
            sizes.append(size)
    if not sizes:
        return 0, 0.0, 0
    return len(sizes), float(max(sizes)) / float(max(1, sum(sizes))), sum(1 for size in sizes if size <= tiny_threshold)


def normalize_distribution(values: list[float], alpha: float = 0.0) -> list[float]:
    adjusted = [max(0.0, float(v)) + float(alpha) for v in values]
    total = sum(adjusted)
    if total <= 0.0:
        return [0.0 for _ in adjusted]
    return [v / total for v in adjusted]


def _normalize_matrix(matrix: list[list[float]], alpha: float = 0.0) -> list[list[float]]:
    flat = [float(v) for row in matrix for v in row]
    norm = normalize_distribution(flat, alpha=alpha)
    width = len(matrix[0]) if matrix else 0
    return [norm[i : i + width] for i in range(0, len(norm), width)]


def _sum_matrix(mats: list[list[list[float]]], h: int, w: int) -> list[list[float]]:
    out = [[0.0 for _ in range(w)] for _ in range(h)]
    for mat in mats:
        for y in range(h):
            for x in range(w):
                out[y][x] += float(mat[y][x])
    return out


def _mean_vector(vectors: list[list[float]], dim: int) -> list[float]:
    if not vectors:
        return [0.0 for _ in range(dim)]
    out = [0.0 for _ in range(dim)]
    for vec in vectors:
        for i in range(dim):
            out[i] += float(vec[i])
    return [v / float(len(vectors)) for v in out]


def _mean_matrix(mats: list[list[list[float]]], h: int, w: int) -> list[list[float]]:
    if not mats:
        return [[0.0 for _ in range(w)] for _ in range(h)]
    summed = _sum_matrix(mats, h, w)
    return [[v / float(len(mats)) for v in row] for row in summed]


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {key: 0.0 for key in ("q005", "q01", "q10", "q50", "q90", "q99", "q995")}
    vals = sorted(float(v) for v in values)

    def q(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = p * (len(vals) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)

    return {"q005": q(0.005), "q01": q(0.01), "q10": q(0.10), "q50": q(0.50), "q90": q(0.90), "q99": q(0.99), "q995": q(0.995)}


def _transition_counts(raw_y20: list[list[int]]) -> tuple[list[list[int]], list[list[int]], float, float, float, float]:
    h = [[0 for _ in range(NUM_LABELS)] for _ in range(NUM_LABELS)]
    v = [[0 for _ in range(NUM_LABELS)] for _ in range(NUM_LABELS)]
    for y in range(CANONICAL_SIZE):
        for x in range(CANONICAL_SIZE):
            a = int(raw_y20[y][x])
            if not (1 <= a <= NUM_LABELS):
                continue
            if x + 1 < CANONICAL_SIZE:
                b = int(raw_y20[y][x + 1])
                if 1 <= b <= NUM_LABELS:
                    h[a - 1][b - 1] += 1
            if y + 1 < CANONICAL_SIZE:
                b = int(raw_y20[y + 1][x])
                if 1 <= b <= NUM_LABELS:
                    v[a - 1][b - 1] += 1
    total_h = sum(sum(row) for row in h)
    total_v = sum(sum(row) for row in v)
    same_h = sum(h[i][i] for i in range(NUM_LABELS))
    same_v = sum(v[i][i] for i in range(NUM_LABELS))
    return (
        h,
        v,
        float(same_h) / float(max(1, total_h)),
        float(same_v) / float(max(1, total_v)),
        float(max(0, total_h - same_h)) / float(max(1, total_h)),
        float(max(0, total_v - same_v)) / float(max(1, total_v)),
    )


def _occupancy_transition_counts(mask: list[list[int]], condition_nonempty: bool) -> tuple[list[list[int]], list[list[int]]]:
    h = [[0, 0], [0, 0]]
    v = [[0, 0], [0, 0]]
    for y in range(CANONICAL_SIZE):
        for x in range(CANONICAL_SIZE):
            a = 1 if int(mask[y][x]) else 0
            if x + 1 < CANONICAL_SIZE:
                b = 1 if int(mask[y][x + 1]) else 0
                if (not condition_nonempty) or (a + b > 0):
                    h[a][b] += 1
            if y + 1 < CANONICAL_SIZE:
                b = 1 if int(mask[y + 1][x]) else 0
                if (not condition_nonempty) or (a + b > 0):
                    v[a][b] += 1
    return h, v


def _motif2_counts(raw_y20: list[list[int]]) -> tuple[Counter[int], float, list[dict[str, int]]]:
    counts: Counter[int] = Counter()
    unique_hist: Counter[int] = Counter()
    transition = 0
    total = 0
    for y in range(CANONICAL_SIZE - 1):
        for x in range(CANONICAL_SIZE - 1):
            patch = [int(raw_y20[y][x]), int(raw_y20[y][x + 1]), int(raw_y20[y + 1][x]), int(raw_y20[y + 1][x + 1])]
            if all(v == 0 for v in patch):
                continue
            key = patch[0] + 17 * patch[1] + (17**2) * patch[2] + (17**3) * patch[3]
            counts[key] += 1
            total += 1
            unique_count = len(set(patch))
            unique_hist[unique_count] += 1
            if unique_count > 1:
                transition += 1
    return counts, float(transition) / float(max(1, total)), [{"unique_value_count": int(k), "count": int(v)} for k, v in sorted(unique_hist.items())]


def _entropy_from_counts(counts: dict[str, int] | Counter[int]) -> float:
    total = float(sum(int(v) for v in counts.values()))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        prob = float(value) / total
        if prob > 0.0:
            entropy -= prob * math.log(prob, 2)
    return float(entropy)


def compute_candidate_descriptors(y20: list[list[int]], config: dict[str, Any] | None = None) -> dict[str, Any]:
    raw_y20 = [[int(v) for v in row] for row in _as_list(y20)]
    _validate_y20(raw_y20)
    mask = _mask_from_raw(raw_y20)
    fg_y20 = _fg_from_raw(raw_y20)
    labels = [int(raw_y20[y][x]) for y in range(CANONICAL_SIZE) for x in range(CANONICAL_SIZE) if 1 <= int(raw_y20[y][x]) <= NUM_LABELS]
    hist = [0 for _ in range(NUM_LABELS)]
    for label in labels:
        hist[label - 1] += 1
    hist_norm = normalize_distribution([float(v) for v in hist])
    row_fg = [sum(int(mask[y][x]) for x in range(CANONICAL_SIZE)) / float(CANONICAL_SIZE) for y in range(CANONICAL_SIZE)]
    col_fg = [sum(int(mask[y][x]) for y in range(CANONICAL_SIZE)) / float(CANONICAL_SIZE) for x in range(CANONICAL_SIZE)]
    row_label = [[0.0 for _ in range(CANONICAL_SIZE)] for _ in range(NUM_LABELS)]
    col_label = [[0.0 for _ in range(CANONICAL_SIZE)] for _ in range(NUM_LABELS)]
    for y in range(CANONICAL_SIZE):
        for x in range(CANONICAL_SIZE):
            label = int(raw_y20[y][x])
            if 1 <= label <= NUM_LABELS:
                row_label[label - 1][y] += 1.0 / float(CANONICAL_SIZE)
                col_label[label - 1][x] += 1.0 / float(CANONICAL_SIZE)
    trans_h, trans_v, same_h, same_v, diff_h, diff_v = _transition_counts(raw_y20)
    occ_h, occ_v = _occupancy_transition_counts(mask, bool(_cfg(config, "occupancy_transition_condition_nonempty")))
    motif_counts, motif_ratio, motif_unique_hist = _motif2_counts(raw_y20)
    num_components, largest_component_ratio, tiny_count = _components(mask, int(_cfg(config, "tiny_component_threshold")))
    bbox = bbox_from_mask([[bool(v) for v in row] for row in mask])
    area_ratio = foreground_area(mask)
    dominant_label = max(range(1, NUM_LABELS + 1), key=lambda label: hist[label - 1]) if any(hist) else 0
    dominant_ratio = float(max(hist)) / float(max(1, sum(hist)))
    return {
        "raw_y20": raw_y20,
        "fg_y20": fg_y20,
        "fg_mask": mask,
        "foreground_area": int(sum(sum(row) for row in mask)),
        "foreground_area_ratio": float(area_ratio),
        "bbox": bbox,
        "label_hist_16": hist,
        "label_hist_norm_16": hist_norm,
        "label_diversity": int(sum(1 for v in hist if v > 0)),
        "dominant_label": int(dominant_label),
        "dominant_label_ratio": float(dominant_ratio),
        "row_fg_projection": row_fg,
        "col_fg_projection": col_fg,
        "row_label_projection_16x20": row_label,
        "col_label_projection_16x20": col_label,
        "transition_h_16x16": trans_h,
        "transition_v_16x16": trans_v,
        "transition_h_norm_16x16": _normalize_matrix([[float(v) for v in row] for row in trans_h]),
        "transition_v_norm_16x16": _normalize_matrix([[float(v) for v in row] for row in trans_v]),
        "same_label_h_ratio": same_h,
        "same_label_v_ratio": same_v,
        "diff_label_h_ratio": diff_h,
        "diff_label_v_ratio": diff_v,
        "occupancy_transition_h_2x2": occ_h,
        "occupancy_transition_v_2x2": occ_v,
        "occupancy_transition_h_norm_2x2": _normalize_matrix([[float(v) for v in row] for row in occ_h]),
        "occupancy_transition_v_norm_2x2": _normalize_matrix([[float(v) for v in row] for row in occ_v]),
        "motif2_counts": {str(k): int(v) for k, v in motif_counts.items()},
        "motif2_total_count": int(sum(motif_counts.values())),
        "motif2_entropy": _entropy_from_counts(motif_counts),
        "patch2x2_transition_ratio": motif_ratio,
        "patch2x2_unique_label_hist": motif_unique_hist,
        "num_connected_components": int(num_components),
        "largest_component_ratio": float(largest_component_ratio),
        "tiny_island_count": int(tiny_count),
    }


def _top_motifs(counter: Counter[int], top_k: int) -> tuple[list[dict[str, int]], list[dict[str, float | int]], int, int]:
    total = int(sum(counter.values()))
    top = counter.most_common(top_k)
    top_count = int(sum(v for _, v in top))
    counts_top = [{"hash": int(k), "count": int(v)} for k, v in top]
    prob_top = [{"hash": int(k), "prob": float(v) / float(max(1, total))} for k, v in top]
    return counts_top, prob_top, total, total - top_count


def aggregate_descriptor_stats(descs: list[dict[str, Any]], config: dict[str, Any] | None = None) -> dict[str, Any]:
    alpha = float(_cfg(config, "smoothing_alpha"))
    motif_top_k = int(_cfg(config, "motif_top_k"))
    trans_h_sum = _sum_matrix([d["transition_h_16x16"] for d in descs], NUM_LABELS, NUM_LABELS)
    trans_v_sum = _sum_matrix([d["transition_v_16x16"] for d in descs], NUM_LABELS, NUM_LABELS)
    occ_h_sum = _sum_matrix([d["occupancy_transition_h_2x2"] for d in descs], 2, 2)
    occ_v_sum = _sum_matrix([d["occupancy_transition_v_2x2"] for d in descs], 2, 2)
    motif_counter: Counter[int] = Counter()
    for d in descs:
        for key, value in d["motif2_counts"].items():
            motif_counter[int(key)] += int(value)
    motif_counts_top, motif_prob_top, motif_total, motif_other = _top_motifs(motif_counter, motif_top_k)
    label_hist_sum = [sum(int(d["label_hist_16"][i]) for d in descs) for i in range(NUM_LABELS)]
    return {
        "num_samples": int(len(descs)),
        "label_hist_16_sum": label_hist_sum,
        "label_hist_norm_16_mean": normalize_distribution([float(v) for v in label_hist_sum], alpha=alpha),
        "transition_h_16x16_sum": trans_h_sum,
        "transition_v_16x16_sum": trans_v_sum,
        "transition_h_norm_16x16": _normalize_matrix(trans_h_sum, alpha=alpha),
        "transition_v_norm_16x16": _normalize_matrix(trans_v_sum, alpha=alpha),
        "occupancy_transition_h_2x2_sum": occ_h_sum,
        "occupancy_transition_v_2x2_sum": occ_v_sum,
        "occupancy_transition_h_norm_2x2": _normalize_matrix(occ_h_sum, alpha=alpha),
        "occupancy_transition_v_norm_2x2": _normalize_matrix(occ_v_sum, alpha=alpha),
        "motif2_counts_top": motif_counts_top,
        "motif2_prob_top": motif_prob_top,
        "motif2_total_count": int(motif_total),
        "motif2_other_count": int(motif_other),
        "row_fg_projection_mean": _mean_vector([d["row_fg_projection"] for d in descs], CANONICAL_SIZE),
        "col_fg_projection_mean": _mean_vector([d["col_fg_projection"] for d in descs], CANONICAL_SIZE),
        "row_label_projection_mean_16x20": _mean_matrix([d["row_label_projection_16x20"] for d in descs], NUM_LABELS, CANONICAL_SIZE),
        "col_label_projection_mean_16x20": _mean_matrix([d["col_label_projection_16x20"] for d in descs], NUM_LABELS, CANONICAL_SIZE),
        "foreground_area_ratio_quantiles": _quantiles([d["foreground_area_ratio"] for d in descs]),
        "num_components_quantiles": _quantiles([d["num_connected_components"] for d in descs]),
        "largest_component_ratio_quantiles": _quantiles([d["largest_component_ratio"] for d in descs]),
        "tiny_island_count_quantiles": _quantiles([d["tiny_island_count"] for d in descs]),
        "label_diversity_quantiles": _quantiles([d["label_diversity"] for d in descs]),
        "dominant_label_ratio_quantiles": _quantiles([d["dominant_label_ratio"] for d in descs]),
        "motif2_entropy_quantiles": _quantiles([d["motif2_entropy"] for d in descs]),
        "same_label_h_ratio_mean": sum(float(d["same_label_h_ratio"]) for d in descs) / float(max(1, len(descs))),
        "same_label_v_ratio_mean": sum(float(d["same_label_v_ratio"]) for d in descs) / float(max(1, len(descs))),
        "diff_label_h_ratio_mean": sum(float(d["diff_label_h_ratio"]) for d in descs) / float(max(1, len(descs))),
        "diff_label_v_ratio_mean": sum(float(d["diff_label_v_ratio"]) for d in descs) / float(max(1, len(descs))),
    }


def build_grammar_bank(items: list[dict[str, Any]], config: dict[str, Any] | None = None) -> dict[str, Any]:
    if config is not None and not bool(config.get("enabled", True)):
        return {"schema_version": GRAMMAR_BANK_SCHEMA_VERSION, "enabled": False, "categories": {}}
    by_category: dict[str, list[dict[str, Any]]] = {}
    by_mode: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for item in items:
        category = str(item["category"])
        raw_y20 = _as_list(item.get("original_y20"))
        if raw_y20 is None:
            raw_y20 = _raw_from_fg(_as_list(item["fg_y20"]))
        desc = compute_candidate_descriptors(raw_y20, config)
        by_category.setdefault(category, []).append(desc)
        local_z = int(item.get("local_z", -1))
        if local_z >= 0:
            by_mode.setdefault(category, {}).setdefault(local_z, []).append(desc)
    min_mode_samples = int(_cfg(config, "min_mode_samples"))
    categories: dict[str, Any] = {}
    for category, descs in sorted(by_category.items()):
        modes: dict[str, Any] = {}
        for local_z, mode_descs in sorted(by_mode.get(category, {}).items()):
            modes[str(local_z)] = {
                "num_samples": int(len(mode_descs)),
                "fallback_to_category": bool(len(mode_descs) < min_mode_samples),
                "stats": aggregate_descriptor_stats(mode_descs, config),
            }
        categories[category] = {
            "category_stats": aggregate_descriptor_stats(descs, config),
            "modes": modes,
        }
    return {
        "schema_version": GRAMMAR_BANK_SCHEMA_VERSION,
        "enabled": True,
        "config": _merged_config(config),
        "categories": categories,
    }


def js_divergence(p: list[float], q: list[float]) -> float:
    p_norm = normalize_distribution(p)
    q_norm = normalize_distribution(q)
    m = [(a + b) * 0.5 for a, b in zip(p_norm, q_norm)]

    def kl(a: list[float], b: list[float]) -> float:
        total = 0.0
        for av, bv in zip(a, b):
            if av > 0.0 and bv > 0.0:
                total += av * math.log(av / bv, 2)
        return total

    return max(0.0, min(1.0, 0.5 * kl(p_norm, m) + 0.5 * kl(q_norm, m)))


def wasserstein_1d(p: list[float], q: list[float]) -> float:
    p_norm = normalize_distribution(p)
    q_norm = normalize_distribution(q)
    acc_p = 0.0
    acc_q = 0.0
    dist = 0.0
    for pv, qv in zip(p_norm, q_norm):
        acc_p += pv
        acc_q += qv
        dist += abs(acc_p - acc_q)
    return dist / float(max(1, len(p_norm)))


def sparse_js_for_motifs(candidate_counts: dict[str, int], prior_stats: dict[str, Any]) -> float:
    prior_probs = {str(item["hash"]): float(item["prob"]) for item in prior_stats.get("motif2_prob_top", [])}
    prior_other = float(prior_stats.get("motif2_other_count", 0)) / float(max(1, int(prior_stats.get("motif2_total_count", 0))))
    total = sum(int(v) for v in candidate_counts.values())
    cand_probs = {str(k): float(v) / float(max(1, total)) for k, v in candidate_counts.items()}
    keys = set(prior_probs) | set(cand_probs)
    p = [cand_probs.get(k, 0.0) for k in keys]
    q = [prior_probs.get(k, 0.0) for k in keys]
    p_seen = sum(p)
    q_seen = sum(q)
    p.append(max(0.0, 1.0 - p_seen))
    q.append(max(0.0, prior_other if prior_other > 0.0 else 1.0 - q_seen))
    return js_divergence(p, q)


def _flatten_matrix(matrix: list[list[float]]) -> list[float]:
    return [float(v) for row in matrix for v in row]


def _offdiag_distribution(matrix: list[list[float]]) -> tuple[list[float], float]:
    flat: list[float] = []
    same = 0.0
    total = 0.0
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            v = float(value)
            total += v
            if i == j:
                same += v
            else:
                flat.append(v)
    diff = max(0.0, total - same)
    if diff <= 0.0:
        return [0.0 for _ in flat], 0.0
    return [v / diff for v in flat], diff / max(1e-12, total)


def _transition_energy_direction(candidate: list[list[float]], prior: list[list[float]]) -> tuple[float, float, float, float, float]:
    cand_full = _flatten_matrix(candidate)
    prior_full = _flatten_matrix(prior)
    cand_off, cand_diff = _offdiag_distribution(candidate)
    prior_off, prior_diff = _offdiag_distribution(prior)
    cand_same = max(0.0, 1.0 - cand_diff)
    prior_same = max(0.0, 1.0 - prior_diff)
    missing_penalty = 0.0
    if prior_diff > 0.15 and cand_diff < 0.03:
        missing_penalty = 2.0
    energy = 0.25 * js_divergence(cand_full, prior_full) + 0.25 * ((cand_same - prior_same) ** 2) + 0.50 * js_divergence(cand_off, prior_off) + missing_penalty
    return float(energy), float(cand_same), float(cand_diff), float(prior_diff), float(missing_penalty)


def _q(stats: dict[str, Any] | None, key: str, qkey: str) -> float | None:
    if not isinstance(stats, dict):
        return None
    quantiles = stats.get(key)
    if not isinstance(quantiles, dict) or qkey not in quantiles:
        return None
    return float(quantiles[qkey])


def _strict_lower(category_value: float | None, mode_value: float | None) -> float:
    values = [v for v in (category_value, mode_value) if v is not None]
    return max(values) if values else 0.0


def _strict_upper(category_value: float | None, mode_value: float | None) -> float:
    values = [v for v in (category_value, mode_value) if v is not None]
    return min(values) if values else 1.0


def _hinge_quantile(value: float, quantiles: dict[str, float] | None, *, low_key: str = "q10", high_key: str = "q90", far_low_key: str = "q01", far_high_key: str = "q99", clip: float = 5.0) -> float:
    if not quantiles:
        return 0.0
    low = float(quantiles.get(low_key, value))
    high = float(quantiles.get(high_key, value))
    far_low = float(quantiles.get(far_low_key, low))
    far_high = float(quantiles.get(far_high_key, high))
    below = max(0.0, low - value) / max(1e-6, low - far_low)
    above = max(0.0, value - high) / max(1e-6, far_high - high)
    return min(clip, below * below + above * above)


class GrammarEnergy:
    def __init__(self, grammar_bank: dict[str, Any], weights: dict[str, float] | None = None, config: dict[str, Any] | None = None):
        self.grammar_bank = grammar_bank
        merged_config = _merged_config(None)
        bank_config = grammar_bank.get("config", {})
        if isinstance(bank_config, dict):
            merged_config.update(bank_config)
        if isinstance(config, dict):
            merged_config.update(config)
        self.config = merged_config
        merged_weights: dict[str, float] = {}
        merged_weights.update(DEFAULT_RERANK_WEIGHTS)
        if isinstance(weights, dict):
            merged_weights.update(weights)
        self.weights = merged_weights
        self.invalid_penalty = float(self.config.get("invalid_penalty", 999.0))

    def _stats(self, category: str, mode_z: int | None = None) -> dict[str, Any]:
        categories = self.grammar_bank.get("categories", {})
        if category not in categories:
            raise KeyError(f"grammar_bank missing category {category!r}")
        entry = categories[category]
        if mode_z is not None:
            mode = entry.get("modes", {}).get(str(int(mode_z)))
            if isinstance(mode, dict) and int(mode.get("num_samples", 0)) >= int(self.config.get("min_mode_samples", 3)) and not bool(mode.get("fallback_to_category", False)):
                return mode["stats"]
        return entry["category_stats"]

    def _category_and_mode_stats(self, category: str, mode_z: int | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
        categories = self.grammar_bank.get("categories", {})
        if category not in categories:
            raise KeyError(f"grammar_bank missing category {category!r}")
        entry = categories[category]
        category_stats = entry["category_stats"]
        mode_stats = None
        if mode_z is not None:
            mode = entry.get("modes", {}).get(str(int(mode_z)))
            if isinstance(mode, dict) and int(mode.get("num_samples", 0)) >= int(self.config.get("min_mode_samples", 3)) and not bool(mode.get("fallback_to_category", False)):
                mode_stats = mode["stats"]
        return category_stats, mode_stats

    def hard_valid(self, y20: list[list[int]], category: str, mode_z: int | None = None) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        try:
            desc = compute_candidate_descriptors(y20, self.config)
        except Exception as exc:
            return False, [f"invalid_y20:{exc}"]
        area = float(desc["foreground_area_ratio"])
        if int(desc["foreground_area"]) <= 0:
            reasons.append("empty_foreground")
        if area < 0.005:
            reasons.append("foreground_area_too_low")
        if area > 0.80:
            reasons.append("foreground_area_too_high")
        if area >= 0.99:
            reasons.append("full_foreground")
        try:
            q = self._stats(category, mode_z).get("foreground_area_ratio_quantiles", {})
            lower = float(q.get("q005", 0.0)) - 0.05
            upper = float(q.get("q995", 1.0)) + 0.05
            if area < max(0.0, lower):
                reasons.append("foreground_area_below_category_q005_margin")
            if area > min(1.0, upper):
                reasons.append("foreground_area_above_category_q995_margin")
        except Exception:
            pass
        return len(reasons) == 0, reasons

    def score(self, y20: list[list[int]], category: str, mode_z: int | None = None) -> dict[str, Any]:
        desc = compute_candidate_descriptors(y20, self.config)
        category_stats, mode_stats = self._category_and_mode_stats(category, mode_z)
        stats = mode_stats or category_stats
        valid, reasons = self.hard_valid(y20, category, mode_z)
        area_e = _hinge_quantile(float(desc["foreground_area_ratio"]), stats.get("foreground_area_ratio_quantiles"))
        hist_e = js_divergence(desc["label_hist_norm_16"], stats.get("label_hist_norm_16_mean", [0.0] * NUM_LABELS))
        trans_h_e, same_h, diff_h, prior_diff_h, missing_h = _transition_energy_direction(desc["transition_h_norm_16x16"], stats.get("transition_h_norm_16x16", [[0.0] * NUM_LABELS for _ in range(NUM_LABELS)]))
        trans_v_e, same_v, diff_v, prior_diff_v, missing_v = _transition_energy_direction(desc["transition_v_norm_16x16"], stats.get("transition_v_norm_16x16", [[0.0] * NUM_LABELS for _ in range(NUM_LABELS)]))
        trans_missing_penalty = missing_h + missing_v
        trans_e = 0.5 * trans_h_e + 0.5 * trans_v_e
        occ_e = 0.5 * js_divergence(_flatten_matrix(desc["occupancy_transition_h_norm_2x2"]), _flatten_matrix(stats.get("occupancy_transition_h_norm_2x2", [[0.0, 0.0], [0.0, 0.0]]))) + 0.5 * js_divergence(_flatten_matrix(desc["occupancy_transition_v_norm_2x2"]), _flatten_matrix(stats.get("occupancy_transition_v_norm_2x2", [[0.0, 0.0], [0.0, 0.0]])))
        motif_js = sparse_js_for_motifs(desc["motif2_counts"], stats)
        motif_cat_q10 = _q(category_stats, "motif2_entropy_quantiles", "q10")
        motif_mode_q10 = _q(mode_stats, "motif2_entropy_quantiles", "q10")
        motif_lower = _strict_lower(motif_cat_q10, motif_mode_q10)
        motif_entropy = float(desc["motif2_entropy"])
        motif_entropy_penalty = 0.0
        if motif_entropy < motif_lower:
            motif_entropy_penalty = ((motif_lower - motif_entropy) / max(1.0, motif_lower)) ** 2
        if motif_lower >= 1.0 and motif_entropy < 0.1:
            motif_entropy_penalty += 5.0
        motif_entropy_penalty = min(10.0, motif_entropy_penalty)
        motif_e = motif_js + motif_entropy_penalty
        rowcol_e = 0.5 * wasserstein_1d(desc["row_fg_projection"], stats.get("row_fg_projection_mean", [0.0] * CANONICAL_SIZE)) + 0.5 * wasserstein_1d(desc["col_fg_projection"], stats.get("col_fg_projection_mean", [0.0] * CANONICAL_SIZE))
        conn_e = (
            _hinge_quantile(float(desc["num_connected_components"]), stats.get("num_components_quantiles"))
            + _hinge_quantile(float(desc["largest_component_ratio"]), stats.get("largest_component_ratio_quantiles"))
            + _hinge_quantile(float(desc["tiny_island_count"]), stats.get("tiny_island_count_quantiles"))
        ) / 3.0
        div_cat_q10 = _q(category_stats, "label_diversity_quantiles", "q10")
        div_mode_q10 = _q(mode_stats, "label_diversity_quantiles", "q10")
        div_lower = _strict_lower(div_cat_q10, div_mode_q10)
        label_div = float(desc["label_diversity"])
        div_penalty_reason = ""
        div_e = 0.0
        if label_div < div_lower:
            div_e = ((div_lower - label_div) / max(1.0, div_lower)) ** 2
            div_penalty_reason = "below_strict_q10"
        if div_cat_q10 is not None and div_cat_q10 >= 3.0 and label_div <= 1.0:
            div_e += 5.0
            div_penalty_reason = "single_label_below_category_diversity"
        div_e = min(10.0, div_e)
        dom_cat_q90 = _q(category_stats, "dominant_label_ratio_quantiles", "q90")
        dom_mode_q90 = _q(mode_stats, "dominant_label_ratio_quantiles", "q90")
        dom_upper = _strict_upper(dom_cat_q90, dom_mode_q90)
        dom_ratio = float(desc["dominant_label_ratio"])
        dom_q995 = _q(category_stats, "dominant_label_ratio_quantiles", "q995")
        dom_scale = max(0.05, (dom_q995 - dom_cat_q90) if dom_q995 is not None and dom_cat_q90 is not None else 0.1)
        dom_penalty_reason = ""
        dom_e = 0.0
        margin = float(self.config.get("dominant_label_margin", 0.03))
        if dom_ratio > dom_upper + margin:
            dom_e = ((dom_ratio - dom_upper) / dom_scale) ** 2
            dom_penalty_reason = "above_strict_q90"
        if dom_cat_q90 is not None and dom_cat_q90 < 0.95 and dom_ratio >= 0.98:
            dom_e += 5.0
            dom_penalty_reason = "near_single_label_above_category_q90"
        dom_e = min(10.0, dom_e)
        parts = {
            "area": float(area_e),
            "conn": float(conn_e),
            "hist": float(hist_e),
            "rowcol": float(rowcol_e),
            "trans": float(trans_e),
            "occ_trans": float(occ_e),
            "motif": float(motif_e),
            "div": float(div_e),
            "dom": float(dom_e),
            "graph": 0.0,
            "nmf": 0.0,
        }
        total = sum(float(self.weights.get(key, 0.0)) * float(value) for key, value in parts.items())
        if not valid:
            total += self.invalid_penalty
        return {
            "total": float(total),
            **parts,
            "valid": bool(valid),
            "invalid_reasons": reasons,
            "diagnostics": {
                "foreground_area_ratio": float(desc["foreground_area_ratio"]),
                "label_diversity": int(desc["label_diversity"]),
                "label_diversity_category_q10": div_cat_q10,
                "label_diversity_mode_q10": div_mode_q10,
                "label_diversity_lower_used": div_lower,
                "div_penalty_reason": div_penalty_reason,
                "dominant_label_ratio": float(desc["dominant_label_ratio"]),
                "dominant_category_q90": dom_cat_q90,
                "dominant_mode_q90": dom_mode_q90,
                "dominant_upper_used": dom_upper,
                "dom_penalty_reason": dom_penalty_reason,
                "same_label_h_ratio": same_h,
                "same_label_v_ratio": same_v,
                "diff_label_h_ratio": diff_h,
                "diff_label_v_ratio": diff_v,
                "prior_diff_h_ratio": prior_diff_h,
                "prior_diff_v_ratio": prior_diff_v,
                "trans_diff_missing_penalty": trans_missing_penalty,
                "motif2_entropy": motif_entropy,
                "motif2_entropy_category_q10": motif_cat_q10,
                "motif2_entropy_mode_q10": motif_mode_q10,
                "motif2_entropy_lower_used": motif_lower,
                "num_components": int(desc["num_connected_components"]),
                "largest_component_ratio": float(desc["largest_component_ratio"]),
                "tiny_island_count": int(desc["tiny_island_count"]),
            },
        }


def save_grammar_bank_inspection(grammar_bank: dict[str, Any], category: str, output_dir: Path, *, top_motifs: int = 20) -> dict[str, str]:
    if category not in grammar_bank.get("categories", {}):
        raise ValueError(f"grammar_bank does not contain category {category!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = grammar_bank["categories"][category]["category_stats"]
    paths: dict[str, str] = {}

    def write_heatmap_png(name: str, matrix: list[list[float]]) -> None:
        try:
            import numpy as np
            from PIL import Image
        except Exception:
            save_json(output_dir / f"{name}.json", {"matrix": matrix})
            paths[f"{name}.json"] = str(output_dir / f"{name}.json")
            return
        arr = np.asarray(matrix, dtype=np.float32)
        if arr.size == 0:
            arr = np.zeros((1, 1), dtype=np.float32)
        mn = float(arr.min())
        mx = float(arr.max())
        norm = (arr - mn) / max(1e-8, mx - mn)
        rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
        rgb[..., 0] = np.clip((norm * 255.0).round(), 0, 255).astype(np.uint8)
        rgb[..., 1] = np.clip(((1.0 - abs(norm - 0.5) * 2.0) * 255.0).round(), 0, 255).astype(np.uint8)
        rgb[..., 2] = np.clip(((1.0 - norm) * 255.0).round(), 0, 255).astype(np.uint8)
        scale = 24 if arr.shape[0] <= 2 else 16
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        path = output_dir / name
        Image.fromarray(rgb, mode="RGB").save(path)
        paths[name] = str(path)

    def write_csv(name: str, matrix: list[list[float]]) -> None:
        path = output_dir / name
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(matrix)
        paths[name] = str(path)

    write_csv("transition_h_norm_16x16.csv", stats["transition_h_norm_16x16"])
    write_csv("transition_v_norm_16x16.csv", stats["transition_v_norm_16x16"])
    write_csv("occupancy_transition_h_norm_2x2.csv", stats["occupancy_transition_h_norm_2x2"])
    write_csv("occupancy_transition_v_norm_2x2.csv", stats["occupancy_transition_v_norm_2x2"])
    write_heatmap_png("transition_h_heatmap.png", stats["transition_h_norm_16x16"])
    write_heatmap_png("transition_v_heatmap.png", stats["transition_v_norm_16x16"])
    write_heatmap_png("occupancy_transition_h_heatmap.png", stats["occupancy_transition_h_norm_2x2"])
    write_heatmap_png("occupancy_transition_v_heatmap.png", stats["occupancy_transition_v_norm_2x2"])
    write_csv("row_fg_projection_mean.csv", [stats["row_fg_projection_mean"]])
    write_csv("col_fg_projection_mean.csv", [stats["col_fg_projection_mean"]])
    quantiles = {
        "foreground_area_ratio_quantiles": stats.get("foreground_area_ratio_quantiles", {}),
        "num_components_quantiles": stats.get("num_components_quantiles", {}),
        "largest_component_ratio_quantiles": stats.get("largest_component_ratio_quantiles", {}),
        "tiny_island_count_quantiles": stats.get("tiny_island_count_quantiles", {}),
        "label_diversity_quantiles": stats.get("label_diversity_quantiles", {}),
        "dominant_label_ratio_quantiles": stats.get("dominant_label_ratio_quantiles", {}),
    }
    save_json(output_dir / "category_quantiles.json", quantiles)
    save_json(output_dir / "top_motif2_patterns.json", {"top_motifs": stats.get("motif2_counts_top", [])[:top_motifs], "motif2_total_count": stats.get("motif2_total_count", 0), "motif2_other_count": stats.get("motif2_other_count", 0)})
    paths["category_quantiles.json"] = str(output_dir / "category_quantiles.json")
    paths["top_motif2_patterns.json"] = str(output_dir / "top_motif2_patterns.json")
    return paths
