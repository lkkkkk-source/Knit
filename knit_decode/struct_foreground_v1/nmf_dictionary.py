from __future__ import annotations

import importlib
import math
from collections import Counter
from typing import Any


DICTIONARY_BANK_SCHEMA_VERSION = "foreground_v1_nmf_dual_dictionary_matched_v1"
CANONICAL_SIZE = 20
NUM_LABELS = 16
IGNORE_INDEX = -100


DEFAULT_NMF_DICTIONARY_CONFIG: dict[str, Any] = {
    "enabled": False,
    "inspect_only": True,
    "feature_mode": "label_balanced_onehot",
    "rank_default": 8,
    "rank_by_category": {},
    "max_iter": 1000,
    "init": "nndsvda",
    "solver": "cd",
    "beta_loss": "frobenius",
    "alpha_W": 0.001,
    "alpha_H": 0.001,
    "l1_ratio": 0.5,
    "random_state": 0,
    "eps": 1.0e-8,
    "prune_percentile": 0.0,
    "gamma_fg": 1.0,
    "save_basis_float16": False,
    "save_sample_codes": True,
    "inspect_sparse_top_r": 3,
    "dual_dictionary": {
        "enabled": True,
        "schema_version": DICTIONARY_BANK_SCHEMA_VERSION,
    },
    "support_dictionary": {
        "feature_mode": "fg_mask",
        "rank_same_as_label": True,
        "normalize_sample_area": False,
        "smooth": {"enabled": True, "kernel_size": 3, "strength": 0.25},
        "alpha_W": 0.0005,
        "alpha_H": 0.0005,
        "l1_ratio": 0.2,
        "prune": {"enabled": False},
        "gamma_fg": 1.0,
        "min_active_pixels": 8,
    },
    "label_dictionary": {
        "feature_mode": "label_balanced_onehot",
        "use_existing_balanced_config": True,
        "basis_unweight_after_fit": False,
        "alpha_W": 0.001,
        "alpha_H": 0.001,
        "l1_ratio": 0.5,
        "prune": {"enabled": False},
    },
    "prior_fusion": {
        "fg_from": "support_dictionary",
        "label_prob_from": "label_dictionary",
        "label_mass_mask_from_support": True,
        "support_mass_threshold": 0.05,
        "label_confidence_threshold": 0.0,
        "normalize_label_prob": True,
    },
    "category_adaptive": {
        "enabled": True,
        "use_category_stats": True,
        "min_effective_rank": 2,
        "min_effective_rank_ratio": 0.5,
        "max_collapsed_fraction_default": 0.5,
        "max_support_empty_basis_fraction": 0.5,
        "allow_low_label_entropy_if_category_low_entropy": True,
        "allow_low_diversity_if_category_low_diversity": True,
    },
    "label_balance": {
        "enabled": True,
        "mode": "inverse_sqrt",
        "min_weight": 0.5,
        "max_weight": 12.0,
        "eps": 1.0e-6,
    },
    "sample_normalize": {
        "enabled": True,
        "mode": "area_sqrt",
        "eps": 1.0e-6,
    },
    "channel_normalize": {
        "enabled": True,
        "mode": "per_label_mass_sqrt",
        "eps": 1.0e-6,
    },
    "basis_unweight_after_fit": False,
    "basis_normalize_mass_max": True,
    "support_basis_mode": "weighted",
    "sparsity": {
        "alpha_W": 0.001,
        "alpha_H": 0.001,
        "l1_ratio": 0.5,
    },
    "prune": {
        "enabled": False,
        "percentile": 0,
        "min_value": 1.0e-8,
    },
    "anti_collapse": {
        "enabled": True,
        "max_basis_dominant_ratio_warn": 0.90,
        "max_collapsed_basis_fraction_warn": 0.50,
        "min_mean_label_entropy_warn": 0.40,
    },
    "code_mode_count_default": 8,
    "code_mode_count_by_category": {},
    "category_overrides": {},
}


def _merged_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_NMF_DICTIONARY_CONFIG)
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                nested = dict(merged[key])
                nested.update(value)
                merged[key] = nested
            else:
                merged[key] = value
    if not isinstance(merged.get("rank_by_category"), dict):
        merged["rank_by_category"] = {}
    if not isinstance(merged.get("code_mode_count_by_category"), dict):
        merged["code_mode_count_by_category"] = {}
    return merged


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _category_config(config: dict[str, Any], category: str) -> dict[str, Any]:
    overrides = config.get("category_overrides", {})
    if isinstance(overrides, dict) and isinstance(overrides.get(category), dict):
        return _deep_merge(config, overrides[category])
    return config


def _require_sklearn_decomposition() -> object:
    try:
        return importlib.import_module("sklearn.decomposition")
    except ImportError as error:
        raise ImportError("scikit-learn is required for NMF dictionary bank building. Install with `pip install scikit-learn`.") from error


def _require_sklearn_cluster() -> object:
    try:
        return importlib.import_module("sklearn.cluster")
    except ImportError as error:
        raise ImportError("scikit-learn is required for NMF code-mode clustering. Install with `pip install scikit-learn`.") from error


def _require_numpy() -> object:
    try:
        return importlib.import_module("numpy")
    except ImportError as error:
        raise ImportError("NumPy is required for NMF dictionary bank building.") from error


def _require_torch() -> object:
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required to serialize NMF dictionary bank tensors.") from error


def _rank_for_category(category: str, config: dict[str, Any], sample_count: int) -> int:
    by_category = config.get("rank_by_category", {})
    requested = int(by_category.get(category, config.get("rank_default", 8))) if isinstance(by_category, dict) else int(config.get("rank_default", 8))
    return min(max(1, requested), max(1, int(sample_count)))


def _code_mode_count_for_category(category: str, config: dict[str, Any], sample_count: int) -> int:
    by_category = config.get("code_mode_count_by_category", {})
    requested = int(by_category.get(category, config.get("code_mode_count_default", 8))) if isinstance(by_category, dict) else int(config.get("code_mode_count_default", 8))
    return min(max(1, requested), max(1, int(sample_count)))


def _fg_y20_to_feature(fg_y20: list[list[int]]) -> list[float]:
    feature = [0.0 for _ in range(NUM_LABELS * CANONICAL_SIZE * CANONICAL_SIZE)]
    if len(fg_y20) != CANONICAL_SIZE or any(len(row) != CANONICAL_SIZE for row in fg_y20):
        raise ValueError("NMF dictionary expects fg_y20 shape [20,20].")
    for y_pos in range(CANONICAL_SIZE):
        for x_pos in range(CANONICAL_SIZE):
            label_value = int(fg_y20[y_pos][x_pos])
            if 1 <= label_value <= NUM_LABELS:
                feature[((label_value - 1) * CANONICAL_SIZE * CANONICAL_SIZE) + (y_pos * CANONICAL_SIZE) + x_pos] = 1.0
            elif label_value not in (0, IGNORE_INDEX):
                raise ValueError(f"NMF dictionary fg_y20 has invalid label {label_value}; expected 1..16 or ignore/background.")
    return feature


def _fg_y20_to_onehot_array(fg_y20: list[list[int]], np: object) -> object:
    if len(fg_y20) != CANONICAL_SIZE or any(len(row) != CANONICAL_SIZE for row in fg_y20):
        raise ValueError("NMF dictionary expects fg_y20 shape [20,20].")
    onehot = np.zeros((NUM_LABELS, CANONICAL_SIZE, CANONICAL_SIZE), dtype=np.float32)
    for y_pos in range(CANONICAL_SIZE):
        for x_pos in range(CANONICAL_SIZE):
            label_value = int(fg_y20[y_pos][x_pos])
            if 1 <= label_value <= NUM_LABELS:
                onehot[label_value - 1, y_pos, x_pos] = 1.0
            elif label_value not in (0, IGNORE_INDEX):
                raise ValueError(f"NMF dictionary fg_y20 has invalid label {label_value}; expected 1..16 or ignore/background.")
    return onehot


def _label_frequency_and_weights(onehots: object, config: dict[str, Any], np: object) -> tuple[object, object]:
    label_counts = np.asarray(onehots, dtype=np.float32).sum(axis=(0, 2, 3))
    total = float(label_counts.sum())
    if total <= 0.0:
        freq = np.zeros((NUM_LABELS,), dtype=np.float32)
    else:
        freq = (label_counts / total).astype(np.float32)
    feature_mode = str(config.get("feature_mode", "label_balanced_onehot"))
    balance_cf = config.get("label_balance", {}) if isinstance(config.get("label_balance"), dict) else {}
    if feature_mode == "raw_onehot" or not bool(balance_cf.get("enabled", True)):
        weights = np.ones((NUM_LABELS,), dtype=np.float32)
        return freq, weights
    mode = str(balance_cf.get("mode", "inverse_sqrt"))
    eps = float(balance_cf.get("eps", 1.0e-6))
    if mode != "inverse_sqrt":
        raise ValueError(f"Unsupported nmf_dictionary.label_balance.mode={mode!r}; expected 'inverse_sqrt'.")
    weights = 1.0 / np.sqrt(freq + eps)
    min_weight = float(balance_cf.get("min_weight", 0.5))
    max_weight = float(balance_cf.get("max_weight", 8.0))
    weights = np.clip(weights, min_weight, max_weight).astype(np.float32)
    return freq, weights


def _weighted_feature_matrix(samples: list[dict[str, object]], config: dict[str, Any], np: object) -> tuple[object, object, object]:
    onehots = np.asarray([_fg_y20_to_onehot_array(sample["fg_y20"], np) for sample in samples], dtype=np.float32)
    label_freq, label_weights = _label_frequency_and_weights(onehots, config, np)
    feature_mode = str(config.get("feature_mode", "label_balanced_onehot"))
    if feature_mode not in ("raw_onehot", "label_balanced_onehot"):
        raise ValueError(f"Unsupported nmf_dictionary.feature_mode={feature_mode!r}; expected raw_onehot or label_balanced_onehot.")
    x = onehots.astype(np.float32, copy=True)
    if feature_mode == "label_balanced_onehot":
        sample_cf = config.get("sample_normalize", {}) if isinstance(config.get("sample_normalize"), dict) else {}
        if bool(sample_cf.get("enabled", True)):
            mode = str(sample_cf.get("mode", "area_sqrt"))
            if mode != "area_sqrt":
                raise ValueError(f"Unsupported nmf_dictionary.sample_normalize.mode={mode!r}; expected 'area_sqrt'.")
            eps = float(sample_cf.get("eps", 1.0e-6))
            area = x.sum(axis=(1, 2, 3))
            x = x / np.sqrt(area[:, None, None, None] + eps)
        channel_cf = config.get("channel_normalize", {}) if isinstance(config.get("channel_normalize"), dict) else {}
        if bool(channel_cf.get("enabled", True)):
            mode = str(channel_cf.get("mode", "per_label_mass_sqrt"))
            if mode != "per_label_mass_sqrt":
                raise ValueError(f"Unsupported nmf_dictionary.channel_normalize.mode={mode!r}; expected 'per_label_mass_sqrt'.")
            eps = float(channel_cf.get("eps", 1.0e-6))
            channel_mass = x.sum(axis=(2, 3))
            denom = np.sqrt(channel_mass[:, :, None, None] + eps)
            x = np.where(channel_mass[:, :, None, None] > 0.0, x / denom, x)
        x = x * label_weights[None, :, None, None]
    return x.reshape(len(samples), NUM_LABELS * CANONICAL_SIZE * CANONICAL_SIZE).astype(np.float32), label_freq.astype(np.float32), label_weights.astype(np.float32)


def _raw_mask_array(fg_y20: list[list[int]], np: object) -> object:
    mask = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE), dtype=np.float32)
    if len(fg_y20) != CANONICAL_SIZE or any(len(row) != CANONICAL_SIZE for row in fg_y20):
        raise ValueError("NMF support dictionary expects fg_y20 shape [20,20].")
    for y_pos in range(CANONICAL_SIZE):
        for x_pos in range(CANONICAL_SIZE):
            label_value = int(fg_y20[y_pos][x_pos])
            if 1 <= label_value <= NUM_LABELS:
                mask[y_pos, x_pos] = 1.0
            elif label_value not in (0, IGNORE_INDEX):
                raise ValueError(f"NMF support dictionary fg_y20 has invalid label {label_value}; expected 1..16 or ignore/background.")
    return mask


def _smooth_mask(mask: object, support_cf: dict[str, Any], np: object) -> object:
    smooth_cf = support_cf.get("smooth", {}) if isinstance(support_cf.get("smooth"), dict) else {}
    if not bool(smooth_cf.get("enabled", False)):
        return mask
    kernel_size = int(smooth_cf.get("kernel_size", 3))
    if kernel_size != 3:
        raise ValueError(f"Unsupported support_dictionary.smooth.kernel_size={kernel_size}; expected 3.")
    strength = float(smooth_cf.get("strength", 0.25))
    padded = np.pad(mask, ((1, 1), (1, 1)), mode="edge")
    acc = np.zeros_like(mask, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            acc += padded[dy : dy + CANONICAL_SIZE, dx : dx + CANONICAL_SIZE]
    mean = acc / 9.0
    return ((1.0 - strength) * mask + strength * mean).astype(np.float32)


def _support_feature_matrix(samples: list[dict[str, object]], support_cf: dict[str, Any], np: object) -> object:
    masks = []
    normalize_area = bool(support_cf.get("normalize_sample_area", False))
    eps = 1.0e-8
    for sample in samples:
        mask = _raw_mask_array(sample["fg_y20"], np)
        mask = _smooth_mask(mask, support_cf, np)
        if normalize_area:
            area = float(mask.sum())
            mask = mask / math.sqrt(area + eps) if area > 0.0 else mask
        masks.append(mask.reshape(CANONICAL_SIZE * CANONICAL_SIZE))
    return np.asarray(masks, dtype=np.float32)


def _fit_nmf_components(x_c: object, rank: int, config: dict[str, Any], np: object) -> tuple[object, object]:
    sklearn_decomposition = _require_sklearn_decomposition()
    nmf_cls = getattr(sklearn_decomposition, "NMF")
    model = nmf_cls(
        n_components=rank,
        init=str(config.get("init", "nndsvda")),
        solver=str(config.get("solver", "cd")),
        beta_loss=str(config.get("beta_loss", "frobenius")),
        alpha_W=float(config.get("alpha_W", 0.001)),
        alpha_H=float(config.get("alpha_H", 0.001)),
        l1_ratio=float(config.get("l1_ratio", 0.5)),
        max_iter=int(config.get("max_iter", 1000)),
        random_state=int(config.get("random_state", 0)),
    )
    codes = model.fit_transform(x_c).astype(np.float32)
    components = np.maximum(np.asarray(model.components_, dtype=np.float32), 0.0)
    return codes, components


def _normalize_component_mass(components: object, np: object, eps: float) -> object:
    components = np.maximum(np.asarray(components, dtype=np.float32), 0.0)
    flat = components.reshape(components.shape[0], -1)
    scale = flat.max(axis=1)
    return np.where(scale.reshape(-1, *([1] * (components.ndim - 1))) > 0.0, components / np.maximum(scale.reshape(-1, *([1] * (components.ndim - 1))), eps), components)


def _normalize_mass_for_matching(mass: object, np: object, eps: float) -> object:
    arr = np.maximum(np.asarray(mass, dtype=np.float32), 0.0)
    flat = arr.reshape(arr.shape[0], -1)
    total = flat.sum(axis=1)
    return np.where(total[:, None] > eps, flat / np.maximum(total[:, None], eps), flat)


def _support_label_overlap_matrix(support_mass: object, label_mass: object, np: object, eps: float) -> object:
    support = _normalize_mass_for_matching(support_mass, np, eps)
    labels = _normalize_mass_for_matching(label_mass, np, eps)
    support_norm = np.linalg.norm(support, axis=1)
    label_norm = np.linalg.norm(labels, axis=1)
    dot = support @ labels.T
    cosine = dot / np.maximum(support_norm[:, None] * label_norm[None, :], eps)
    minimum = np.minimum(support[:, None, :], labels[None, :, :]).sum(axis=2)
    maximum = np.maximum(support[:, None, :], labels[None, :, :]).sum(axis=2)
    soft_iou = minimum / np.maximum(maximum, eps)
    return (0.5 * cosine + 0.5 * soft_iou).astype(np.float32)


def _linear_sum_assignment_max(score_matrix: object, np: object) -> tuple[list[int], str]:
    matrix = np.asarray(score_matrix, dtype=np.float32)
    rank = int(matrix.shape[0])
    try:
        scipy_optimize = importlib.import_module("scipy.optimize")
        row_ind, col_ind = scipy_optimize.linear_sum_assignment(-matrix)
        assignment = [-1 for _ in range(rank)]
        for row, col in zip(row_ind.tolist(), col_ind.tolist()):
            assignment[int(row)] = int(col)
        if any(value < 0 for value in assignment):
            raise ValueError("linear_sum_assignment returned incomplete assignment")
        return assignment, "scipy_linear_sum_assignment"
    except Exception:
        remaining = set(range(int(matrix.shape[1])))
        assignment = [-1 for _ in range(rank)]
        for row in range(rank):
            if not remaining:
                break
            best_col = max(remaining, key=lambda col: float(matrix[row, col]))
            assignment[row] = int(best_col)
            remaining.remove(best_col)
        if any(value < 0 for value in assignment):
            for row in range(rank):
                if assignment[row] < 0:
                    assignment[row] = int(row % max(1, int(matrix.shape[1])))
        return assignment, "greedy_fallback_no_scipy"


def _match_support_label_bases(support_mass: object, label_component_mass: object, np: object, eps: float, *, low_overlap_threshold: float = 0.05) -> dict[str, object]:
    overlap_matrix = _support_label_overlap_matrix(support_mass, label_component_mass, np, eps)
    assignment, method = _linear_sum_assignment_max(overlap_matrix, np)
    matched_scores = [float(overlap_matrix[row, int(col)]) for row, col in enumerate(assignment)]
    low_indices = [int(index) for index, value in enumerate(matched_scores) if float(value) < float(low_overlap_threshold)]
    warnings: list[str] = []
    if method.startswith("greedy"):
        warnings.append("scipy.optimize.linear_sum_assignment unavailable; used greedy support-label matching")
    if low_indices:
        warnings.append("low matched support-label overlap detected")
    return {
        "assignment": [int(value) for value in assignment],
        "overlap_matrix": [[float(value) for value in row] for row in overlap_matrix.tolist()],
        "matched_overlap_scores": matched_scores,
        "mean_matched_overlap": float(sum(matched_scores) / max(1, len(matched_scores))),
        "min_matched_overlap": float(min(matched_scores) if matched_scores else 0.0),
        "low_overlap_basis_indices": low_indices,
        "low_overlap_basis_count": int(len(low_indices)),
        "matching_method": method,
        "warnings": warnings,
    }


def _component_count_and_largest(mask_grid: list[list[int]]) -> tuple[int, float]:
    visited = [[False for _ in range(CANONICAL_SIZE)] for _ in range(CANONICAL_SIZE)]
    sizes: list[int] = []
    for y_pos in range(CANONICAL_SIZE):
        for x_pos in range(CANONICAL_SIZE):
            if visited[y_pos][x_pos] or int(mask_grid[y_pos][x_pos]) <= 0:
                continue
            stack = [(y_pos, x_pos)]
            visited[y_pos][x_pos] = True
            size = 0
            while stack:
                cy, cx = stack.pop()
                size += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < CANONICAL_SIZE and 0 <= nx < CANONICAL_SIZE and not visited[ny][nx] and int(mask_grid[ny][nx]) > 0:
                        visited[ny][nx] = True
                        stack.append((ny, nx))
            sizes.append(size)
    total = sum(sizes)
    return len(sizes), float(max(sizes) / total) if total > 0 else 0.0


def _support_stats_for_component(component: object, *, threshold: float, min_active_pixels: int, np: object) -> dict[str, object]:
    grid = np.asarray(component, dtype=np.float32)
    active = grid > float(threshold)
    active_count = int(active.sum())
    mask_grid = [[1 if bool(active[y_pos, x_pos]) else 0 for x_pos in range(CANONICAL_SIZE)] for y_pos in range(CANONICAL_SIZE)]
    num_components, largest_ratio = _component_count_and_largest(mask_grid)
    area = float(active_count / float(CANONICAL_SIZE * CANONICAL_SIZE))
    return {
        "support_basis_mass_sum": float(grid.sum()),
        "support_active_pixel_count": active_count,
        "support_area_estimate": area,
        "support_fragmentation_score": float(num_components / max(1, active_count)),
        "support_num_components": int(num_components),
        "support_largest_component_ratio": float(largest_ratio),
        "support_empty": bool(active_count < int(min_active_pixels)),
    }


def _category_label_stats(samples: list[dict[str, object]]) -> dict[str, object]:
    entropies: list[float] = []
    diversities: list[int] = []
    for sample in samples:
        counts = [0.0 for _ in range(NUM_LABELS)]
        total = 0.0
        for row in sample["fg_y20"]:
            for value in row:
                label_value = int(value)
                if 1 <= label_value <= NUM_LABELS:
                    counts[label_value - 1] += 1.0
                    total += 1.0
        if total > 0.0:
            probs = [count / total for count in counts if count > 0.0]
            entropies.append(float(-sum(prob * math.log(prob + 1e-12) for prob in probs)))
            diversities.append(len(probs))
    mean_entropy = float(sum(entropies) / max(1, len(entropies)))
    mean_diversity = float(sum(diversities) / max(1, len(diversities)))
    return {
        "mean_label_entropy": mean_entropy,
        "mean_label_diversity": mean_diversity,
        "low_entropy_category": bool(mean_entropy < 0.40),
        "low_diversity_category": bool(mean_diversity <= 1.5),
    }


def _as_float_grid(value: object, *, context: str) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or len(value) != CANONICAL_SIZE:
        raise ValueError(f"{context} must have shape [20,20].")
    out: list[list[float]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != CANONICAL_SIZE:
            raise ValueError(f"{context} row {row_index} must have 20 columns.")
        out.append([float(part) for part in row])
    return out


def _masked_argmax_and_confidence(label_prob_16: object, label_mass: object, *, threshold: float) -> tuple[list[list[int]], list[list[float]]]:
    if hasattr(label_prob_16, "detach"):
        label_prob_16 = label_prob_16.detach().cpu()
    if hasattr(label_prob_16, "tolist"):
        label_prob_16 = label_prob_16.tolist()
    if not isinstance(label_prob_16, list) or len(label_prob_16) != NUM_LABELS:
        raise ValueError("basis_label_prob_16 must have shape [16,20,20].")
    mass_grid = _as_float_grid(label_mass, context="basis_label_mass")
    argmax_grid = [[0 for _ in range(CANONICAL_SIZE)] for _ in range(CANONICAL_SIZE)]
    confidence_grid = [[0.0 for _ in range(CANONICAL_SIZE)] for _ in range(CANONICAL_SIZE)]
    for y_pos in range(CANONICAL_SIZE):
        for x_pos in range(CANONICAL_SIZE):
            values = [float(label_prob_16[label_index][y_pos][x_pos]) for label_index in range(NUM_LABELS)]
            best_index = max(range(NUM_LABELS), key=lambda label_index: values[label_index])
            if float(mass_grid[y_pos][x_pos]) > float(threshold):
                confidence_grid[y_pos][x_pos] = float(values[best_index])
                argmax_grid[y_pos][x_pos] = int(best_index + 1)
    return argmax_grid, confidence_grid


def _basis_stats(
    *,
    basis_index: int,
    basis_mass: object,
    fg_mask_prob: object,
    label_prob_16: object,
    argmax_grid: list[list[int]],
    threshold: float,
    eps: float,
) -> dict[str, object]:
    mass_grid = _as_float_grid(basis_mass, context=f"basis[{basis_index}].basis_label_mass")
    if hasattr(fg_mask_prob, "detach"):
        fg_mask_prob = fg_mask_prob.detach().cpu()
    if hasattr(fg_mask_prob, "tolist"):
        fg_mask_prob = fg_mask_prob.tolist()
    if isinstance(fg_mask_prob, list) and len(fg_mask_prob) == 1:
        fg_mask_prob = fg_mask_prob[0]
    fg_grid = _as_float_grid(fg_mask_prob, context=f"basis[{basis_index}].basis_fg_mask_prob")
    if hasattr(label_prob_16, "detach"):
        label_prob_16 = label_prob_16.detach().cpu()
    if hasattr(label_prob_16, "tolist"):
        label_prob_16 = label_prob_16.tolist()
    active_positions = [(y_pos, x_pos) for y_pos in range(CANONICAL_SIZE) for x_pos in range(CANONICAL_SIZE) if float(mass_grid[y_pos][x_pos]) > float(threshold)]
    label_hist = [0.0 for _ in range(NUM_LABELS)]
    for y_pos, x_pos in active_positions:
        probs = [float(label_prob_16[label_index][y_pos][x_pos]) for label_index in range(NUM_LABELS)]
        total = sum(probs)
        if total <= 0.0:
            continue
        mass_value = float(mass_grid[y_pos][x_pos])
        normalized = [value / total for value in probs]
        for index, value in enumerate(normalized):
            label_hist[index] += mass_value * value
    label_mass_total = sum(label_hist)
    label_entropy = 0.0
    dominant_label_ratio = 0.0
    label_diversity = 0.0
    dominant_label = 0
    if label_mass_total > 0.0:
        label_hist = [value / label_mass_total for value in label_hist]
        dominant_label_ratio = max(label_hist)
        label_diversity = float(sum(1 for value in label_hist if value > 1.0e-4))
        label_entropy = -sum(value * math.log(value + 1e-12) for value in label_hist if value > 0.0)
        dominant_label = int(max(range(NUM_LABELS), key=lambda index: label_hist[index]) + 1)
    transition_h = 0
    transition_v = 0
    edge_h = 0
    edge_v = 0
    motif_counter: Counter[tuple[int, int, int, int]] = Counter()
    for y_pos in range(CANONICAL_SIZE):
        for x_pos in range(CANONICAL_SIZE - 1):
            left = int(argmax_grid[y_pos][x_pos])
            right = int(argmax_grid[y_pos][x_pos + 1])
            if left > 0 or right > 0:
                edge_h += 1
                if left != right:
                    transition_h += 1
    for y_pos in range(CANONICAL_SIZE - 1):
        for x_pos in range(CANONICAL_SIZE):
            top = int(argmax_grid[y_pos][x_pos])
            bottom = int(argmax_grid[y_pos + 1][x_pos])
            if top > 0 or bottom > 0:
                edge_v += 1
                if top != bottom:
                    transition_v += 1
    for y_pos in range(CANONICAL_SIZE - 1):
        for x_pos in range(CANONICAL_SIZE - 1):
            motif = (
                int(argmax_grid[y_pos][x_pos]),
                int(argmax_grid[y_pos][x_pos + 1]),
                int(argmax_grid[y_pos + 1][x_pos]),
                int(argmax_grid[y_pos + 1][x_pos + 1]),
            )
            if any(value > 0 for value in motif):
                motif_counter[motif] += 1
    motif_total = sum(motif_counter.values())
    motif2_entropy = 0.0
    if motif_total > 0:
        motif2_entropy = -sum((count / motif_total) * math.log((count / motif_total) + 1e-12) for count in motif_counter.values())
    mass_values = [float(value) for row in mass_grid for value in row]
    fg_values = [float(value) for row in fg_grid for value in row]
    basis_mass_sum = float(sum(mass_values))
    active_mass_pixel_count = int(len(active_positions))
    basis_empty = bool(basis_mass_sum <= float(eps) or active_mass_pixel_count == 0)
    return {
        "basis_index": int(basis_index),
        "basis_mass_sum": basis_mass_sum,
        "basis_area_estimate": float(sum(1 for value in fg_values if value > float(threshold)) / float(CANONICAL_SIZE * CANONICAL_SIZE)),
        "label_hist_16": [float(value) for value in label_hist],
        "basis_dominant_label": int(dominant_label),
        "basis_dominant_label_ratio_on_mass": float(dominant_label_ratio),
        "basis_label_entropy_on_mass": float(label_entropy),
        "label_diversity_on_mass": float(label_diversity),
        "dominant_label_ratio_on_mass": float(dominant_label_ratio),
        "label_entropy_on_mass": float(label_entropy),
        "transition_h": float(transition_h / max(1, edge_h)),
        "transition_v": float(transition_v / max(1, edge_v)),
        "motif2_entropy": float(motif2_entropy),
        "foreground_mass_mean": float(sum(fg_values) / float(max(1, len(fg_values)))),
        "foreground_mass_max": float(max(fg_values) if fg_values else 0.0),
        "active_mass_pixel_count": active_mass_pixel_count,
        "basis_empty": basis_empty,
        "empty_basis": basis_empty,
    }


def _build_code_modes(category: str, codes: object, config: dict[str, Any]) -> dict[str, object]:
    np = _require_numpy()
    sklearn_cluster = _require_sklearn_cluster()
    cluster_cls = getattr(sklearn_cluster, "MiniBatchKMeans")
    codes_np = np.asarray(codes, dtype=np.float32)
    sample_count, rank = codes_np.shape
    amp = codes_np.sum(axis=1)
    eps = float(config.get("eps", 1.0e-8))
    beta = codes_np / (amp[:, None] + eps)
    mode_count = _code_mode_count_for_category(category, config, sample_count)
    if sample_count <= 0:
        raise ValueError(f"NMF code-mode clustering for {category!r} received no samples.")
    if sample_count == 1 or mode_count == 1:
        assigned = np.zeros((sample_count,), dtype=np.int64)
    else:
        kmeans = cluster_cls(n_clusters=mode_count, batch_size=1024, random_state=int(config.get("random_state", 0)), n_init="auto")
        assigned = kmeans.fit_predict(beta).astype(np.int64)
    mode_weights: list[float] = []
    mode_code_mean: list[list[float]] = []
    mode_code_std: list[list[float]] = []
    mode_amp_mean: list[float] = []
    mode_amp_std: list[float] = []
    mode_num_samples: list[int] = []
    for mode_index in range(mode_count):
        mask = assigned == mode_index
        selected = codes_np[mask]
        selected_amp = amp[mask]
        mode_num = int(selected.shape[0])
        mode_num_samples.append(mode_num)
        mode_weights.append(float(mode_num / float(sample_count)))
        if mode_num > 0:
            mode_code_mean.append([float(value) for value in selected.mean(axis=0).tolist()])
            mode_code_std.append([float(value) for value in selected.std(axis=0).tolist()])
            mode_amp_mean.append(float(selected_amp.mean()))
            mode_amp_std.append(float(selected_amp.std()))
        else:
            mode_code_mean.append([0.0 for _ in range(rank)])
            mode_code_std.append([0.0 for _ in range(rank)])
            mode_amp_mean.append(0.0)
            mode_amp_std.append(0.0)
    return {
        "mode_count": int(mode_count),
        "sample_to_code_mode": {},
        "mode_weights": mode_weights,
        "mode_code_mean": mode_code_mean,
        "mode_code_std": mode_code_std,
        "mode_amp_mean": mode_amp_mean,
        "mode_amp_std": mode_amp_std,
        "mode_num_samples": mode_num_samples,
        "_assigned": [int(value) for value in assigned.tolist()],
    }


def _anti_collapse_summary(basis_stats: list[dict[str, object]], config: dict[str, Any]) -> dict[str, object]:
    anti_cf = config.get("anti_collapse", {}) if isinstance(config.get("anti_collapse"), dict) else {}
    enabled = bool(anti_cf.get("enabled", True))
    max_dominant = float(anti_cf.get("max_basis_dominant_ratio_warn", 0.90))
    max_fraction = float(anti_cf.get("max_collapsed_basis_fraction_warn", 0.50))
    min_entropy = float(anti_cf.get("min_mean_label_entropy_warn", 0.40))
    collapsed = [
        bool(row.get("basis_empty", row.get("empty_basis", False)))
        or bool(float(row.get("dominant_label_ratio_on_mass", 0.0)) > max_dominant)
        or bool(float(row.get("label_entropy_on_mass", 0.0)) < min_entropy)
        for row in basis_stats
    ]
    for row, is_collapsed in zip(basis_stats, collapsed):
        row["basis_collapsed"] = bool(is_collapsed)
    collapsed_count = int(sum(1 for value in collapsed if value))
    collapsed_fraction = float(collapsed_count / float(max(1, len(basis_stats))))
    mean_entropy = float(sum(float(row.get("label_entropy_on_mass", 0.0)) for row in basis_stats) / float(max(1, len(basis_stats))))
    mean_dominant = float(
        sum(1.0 if bool(row.get("empty_basis", False)) else float(row.get("dominant_label_ratio_on_mass", 0.0)) for row in basis_stats)
        / float(max(1, len(basis_stats)))
    )
    warnings: list[str] = []
    empty_count = int(sum(1 for row in basis_stats if bool(row.get("basis_empty", row.get("empty_basis", False)))))
    empty_fraction = float(empty_count / float(max(1, len(basis_stats))))
    label1_count = int(sum(1 for row in basis_stats if int(row.get("basis_dominant_label", 0)) == 1))
    label1_fraction = float(label1_count / float(max(1, len(basis_stats))))
    if enabled and collapsed_fraction > max_fraction:
        warnings.append("NMF dictionary label collapse detected.")
    if enabled and mean_entropy < min_entropy:
        warnings.append("NMF dictionary mean label entropy is below threshold.")
    if enabled and empty_count > 0:
        warnings.append("NMF dictionary has empty basis components.")
    return {
        "collapsed_basis_count": collapsed_count,
        "collapsed_basis_fraction": collapsed_fraction,
        "empty_basis_count": empty_count,
        "empty_basis_fraction": empty_fraction,
        "mean_basis_label_entropy": mean_entropy,
        "mean_basis_dominant_ratio": mean_dominant,
        "label1_dominant_basis_count": label1_count,
        "label1_dominant_basis_fraction": label1_fraction,
        "anti_collapse_warnings": warnings,
        "anti_collapse_thresholds": {
            "max_basis_dominant_ratio_warn": max_dominant,
            "max_collapsed_basis_fraction_warn": max_fraction,
            "min_mean_label_entropy_warn": min_entropy,
        },
    }


def _quality_summary(categories: dict[str, dict[str, object]]) -> dict[str, object]:
    unusable = [category for category, entry in categories.items() if isinstance(entry, dict) and not bool(entry.get("quality_report", {}).get("dictionary_usable", False))]
    usable = [category for category, entry in categories.items() if isinstance(entry, dict) and bool(entry.get("quality_report", {}).get("dictionary_usable", False))]
    warnings_by_category = {
        category: entry.get("quality_report", {}).get("warnings", [])
        for category, entry in categories.items()
        if isinstance(entry, dict) and entry.get("quality_report", {}).get("warnings")
    }
    unusable_reasons_by_category = {
        category: entry.get("quality_report", {}).get("unusable_reasons", [])
        for category, entry in categories.items()
        if isinstance(entry, dict) and entry.get("quality_report", {}).get("unusable_reasons")
    }
    mean_matched_overlap_by_category = {
        category: float(entry.get("quality_report", {}).get("mean_matched_overlap", 0.0))
        for category, entry in categories.items()
        if isinstance(entry, dict)
    }
    return {
        "categories_total": int(len(categories)),
        "categories_usable": int(len(usable)),
        "categories_unusable": int(len(unusable)),
        "usable_categories": sorted(usable),
        "unusable_categories": sorted(unusable),
        "warnings_by_category": warnings_by_category,
        "unusable_reasons_by_category": unusable_reasons_by_category,
        "mean_matched_overlap_by_category": mean_matched_overlap_by_category,
    }


def _build_dual_dictionary_bank(items: list[dict[str, object]], config: dict[str, Any]) -> dict[str, object]:
    np = _require_numpy()
    torch = _require_torch()
    categories = sorted({str(item.get("category")) for item in items if not bool(item.get("is_empty_foreground", False))})
    category_payloads: dict[str, dict[str, object]] = {}
    for category in categories:
        cfg = _category_config(config, category)
        samples = [item for item in items if str(item.get("category")) == category and not bool(item.get("is_empty_foreground", False))]
        if not samples:
            continue
        sample_ids = [str(sample["sample_id"]) for sample in samples]
        try:
            rank = _rank_for_category(category, cfg, len(samples))
            eps = float(cfg.get("eps", 1.0e-8))
            support_cf = cfg.get("support_dictionary", {}) if isinstance(cfg.get("support_dictionary"), dict) else {}
            label_cf = cfg.get("label_dictionary", {}) if isinstance(cfg.get("label_dictionary"), dict) else {}
            fusion_cf = cfg.get("prior_fusion", {}) if isinstance(cfg.get("prior_fusion"), dict) else {}
            adaptive_cf = cfg.get("category_adaptive", {}) if isinstance(cfg.get("category_adaptive"), dict) else {}
            anti_cf = cfg.get("anti_collapse", {}) if isinstance(cfg.get("anti_collapse"), dict) else {}
            support_fit_cf = {
                "init": cfg.get("init", "nndsvda"),
                "solver": cfg.get("solver", "cd"),
                "beta_loss": cfg.get("beta_loss", "frobenius"),
                "max_iter": cfg.get("max_iter", 1000),
                "random_state": cfg.get("random_state", 0),
                "alpha_W": support_cf.get("alpha_W", 0.0005),
                "alpha_H": support_cf.get("alpha_H", 0.0005),
                "l1_ratio": support_cf.get("l1_ratio", 0.2),
            }
            label_fit_cf = {
                "init": cfg.get("init", "nndsvda"),
                "solver": cfg.get("solver", "cd"),
                "beta_loss": cfg.get("beta_loss", "frobenius"),
                "max_iter": cfg.get("max_iter", 1000),
                "random_state": cfg.get("random_state", 0),
                "alpha_W": label_cf.get("alpha_W", cfg.get("sparsity", {}).get("alpha_W", 0.001) if isinstance(cfg.get("sparsity"), dict) else 0.001),
                "alpha_H": label_cf.get("alpha_H", cfg.get("sparsity", {}).get("alpha_H", 0.001) if isinstance(cfg.get("sparsity"), dict) else 0.001),
                "l1_ratio": label_cf.get("l1_ratio", cfg.get("sparsity", {}).get("l1_ratio", 0.5) if isinstance(cfg.get("sparsity"), dict) else 0.5),
            }
            support_x = _support_feature_matrix(samples, support_cf, np)
            support_codes, support_components_flat = _fit_nmf_components(support_x, rank, support_fit_cf, np)
            support_components = support_components_flat.reshape(rank, CANONICAL_SIZE, CANONICAL_SIZE).astype(np.float32)
            support_mass = _normalize_component_mass(support_components, np, eps)
            gamma_fg = float(support_cf.get("gamma_fg", cfg.get("gamma_fg", 1.0)))
            basis_fg_mask_prob = (1.0 - np.exp(-gamma_fg * support_mass)).astype(np.float32)[:, None, :, :]

            label_x, label_freq, label_weights = _weighted_feature_matrix(samples, cfg, np)
            label_codes, label_components_flat = _fit_nmf_components(label_x, rank, label_fit_cf, np)
            label_components = label_components_flat.reshape(rank, NUM_LABELS, CANONICAL_SIZE, CANONICAL_SIZE).astype(np.float32)
            label_components = _normalize_component_mass(label_components, np, eps)
            label_component_mass = label_components.sum(axis=1).astype(np.float32)
            label_prob_16 = ((label_components + eps) / (label_component_mass[:, None, :, :] + (NUM_LABELS * eps))).astype(np.float32)
            match_info = _match_support_label_bases(
                support_mass,
                label_component_mass,
                np,
                eps,
                low_overlap_threshold=float(fusion_cf.get("low_overlap_threshold", 0.05)),
            )
            support_label_assignment = [int(value) for value in match_info["assignment"]]
            assignment_np = np.asarray(support_label_assignment, dtype=np.int64)
            matched_label_components = label_components[assignment_np]
            matched_label_component_mass = label_component_mass[assignment_np]
            matched_label_prob_16 = label_prob_16[assignment_np]
            matched_label_codes = label_codes[:, assignment_np]

            support_threshold = float(fusion_cf.get("support_mass_threshold", 0.05))
            matched_label_confidence = matched_label_prob_16.max(axis=1).astype(np.float32)
            if bool(fusion_cf.get("label_mass_mask_from_support", True)):
                basis_label_mass = (support_mass * matched_label_confidence).astype(np.float32)
            else:
                basis_label_mass = (support_mass * matched_label_component_mass).astype(np.float32)
            argmax_raw = matched_label_prob_16.argmax(axis=1).astype(np.int64) + 1
            basis_label_argmax = np.where(support_mass > support_threshold, argmax_raw, 0).astype(np.int64)
            raw_label_basis_argmax = label_prob_16.argmax(axis=1).astype(np.int64)
            matched_label_basis_argmax = matched_label_prob_16.argmax(axis=1).astype(np.int64)

            min_active = int(support_cf.get("min_active_pixels", 8))
            support_stats = [_support_stats_for_component(support_mass[index], threshold=support_threshold, min_active_pixels=min_active, np=np) for index in range(rank)]
            raw_label_stats: list[dict[str, object]] = []
            matched_label_stats: list[dict[str, object]] = []
            basis_stats: list[dict[str, object]] = []
            category_stats = _category_label_stats(samples)
            allow_low_entropy = bool(cfg.get("allow_low_label_entropy", False)) or (bool(adaptive_cf.get("allow_low_label_entropy_if_category_low_entropy", True)) and bool(category_stats.get("low_entropy_category", False)))
            allow_low_diversity = bool(adaptive_cf.get("allow_low_diversity_if_category_low_diversity", True)) and bool(category_stats.get("low_diversity_category", False))
            allow_single_label = bool(anti_cf.get("allow_single_label", False)) or bool(cfg.get("allow_high_label1_fraction", False)) or allow_low_entropy or allow_low_diversity
            max_dominant = float(anti_cf.get("max_basis_dominant_ratio_warn", 0.90))
            min_entropy = float(anti_cf.get("min_mean_label_entropy_warn", 0.40))
            dropped_empty: list[int] = []
            dropped_collapsed: list[int] = []
            overlap_values = [float(value) for value in match_info["matched_overlap_scores"]]
            for label_index in range(rank):
                raw_argmax_grid = [[int(raw_label_basis_argmax[label_index, y_pos, x_pos] + 1) for x_pos in range(CANONICAL_SIZE)] for y_pos in range(CANONICAL_SIZE)]
                raw_stat = _basis_stats(
                    basis_index=label_index,
                    basis_mass=label_component_mass[label_index],
                    fg_mask_prob=label_component_mass[label_index],
                    label_prob_16=label_prob_16[label_index],
                    argmax_grid=raw_argmax_grid,
                    threshold=support_threshold,
                    eps=eps,
                )
                raw_label_stats.append(
                    {
                        "basis_index": label_index,
                        "label_basis_entropy": raw_stat["basis_label_entropy_on_mass"],
                        "label_basis_dominant_label": raw_stat["basis_dominant_label"],
                        "label_basis_dominant_ratio": raw_stat["basis_dominant_label_ratio_on_mass"],
                        "label1_dominant": bool(int(raw_stat["basis_dominant_label"]) == 1),
                        "label_empty": bool(float(label_component_mass[label_index].sum()) <= eps),
                    }
                )
            for basis_index in range(rank):
                label_argmax = [[int(basis_label_argmax[basis_index, y_pos, x_pos]) for x_pos in range(CANONICAL_SIZE)] for y_pos in range(CANONICAL_SIZE)]
                stat = _basis_stats(
                    basis_index=basis_index,
                    basis_mass=basis_label_mass[basis_index],
                    fg_mask_prob=basis_fg_mask_prob[basis_index],
                    label_prob_16=matched_label_prob_16[basis_index],
                    argmax_grid=label_argmax,
                    threshold=support_threshold,
                    eps=eps,
                )
                support_empty = bool(support_stats[basis_index]["support_empty"])
                matched_label_index = int(support_label_assignment[basis_index])
                label_empty = bool(float(matched_label_component_mass[basis_index].sum()) <= eps)
                label_collapsed = (not allow_single_label) and (
                    float(stat["basis_dominant_label_ratio_on_mass"]) > max_dominant or float(stat["basis_label_entropy_on_mass"]) < min_entropy
                )
                stat.update(
                    {
                        **support_stats[basis_index],
                        "matched_label_basis_index": matched_label_index,
                        "matched_label_overlap": float(overlap_values[basis_index]),
                        "label_basis_entropy": float(stat["basis_label_entropy_on_mass"]),
                        "label_basis_dominant_label": int(stat["basis_dominant_label"]),
                        "label_basis_dominant_ratio": float(stat["basis_dominant_label_ratio_on_mass"]),
                        "label1_dominant": bool(int(stat["basis_dominant_label"]) == 1),
                        "label_empty": bool(label_empty),
                        "support_empty": bool(support_empty),
                        "basis_empty": bool(support_empty or label_empty),
                        "basis_collapsed": bool(support_empty or label_empty or label_collapsed),
                    }
                )
                stat["support_label_overlap"] = float(overlap_values[basis_index])
                basis_stats.append(stat)
                matched_label_stats.append(
                    {
                        "basis_index": basis_index,
                        "matched_label_basis_index": matched_label_index,
                        "label_basis_entropy": stat["label_basis_entropy"],
                        "label_basis_dominant_label": stat["label_basis_dominant_label"],
                        "label_basis_dominant_ratio": stat["label_basis_dominant_ratio"],
                        "label1_dominant": stat["label1_dominant"],
                        "label_empty": stat["label_empty"],
                        "matched_label_overlap": float(overlap_values[basis_index]),
                    }
                )
                if stat["basis_empty"]:
                    dropped_empty.append(basis_index)
                elif stat["basis_collapsed"]:
                    dropped_collapsed.append(basis_index)
            effective_indices = [index for index, stat in enumerate(basis_stats) if not bool(stat["basis_empty"])]
            min_effective_rank = int(adaptive_cf.get("min_effective_rank", 2))
            min_effective_rank_ratio = float(cfg.get("min_effective_rank_ratio", adaptive_cf.get("min_effective_rank_ratio", 0.5)))
            max_collapsed_fraction = float(cfg.get("max_collapsed_fraction_default", adaptive_cf.get("max_collapsed_fraction_default", 0.5)))
            max_support_empty_fraction = float(cfg.get("max_support_empty_basis_fraction", adaptive_cf.get("max_support_empty_basis_fraction", 0.5)))
            allow_low_effective_rank = bool(cfg.get("allow_low_effective_rank", adaptive_cf.get("allow_low_effective_rank", False)))
            effective_rank = len(effective_indices)
            effective_rank_ratio = float(effective_rank / max(1, rank))
            support_empty_count = sum(1 for stat in basis_stats if bool(stat.get("support_empty")))
            label_empty_count = sum(1 for stat in basis_stats if bool(stat.get("label_empty")))
            collapsed_count = sum(1 for stat in basis_stats if bool(stat.get("basis_collapsed")))
            support_empty_fraction = float(support_empty_count / max(1, rank))
            label_empty_fraction = float(label_empty_count / max(1, rank))
            collapsed_fraction = float(collapsed_count / max(1, rank))
            unusable_reasons: list[str] = []
            if effective_rank < min_effective_rank:
                unusable_reasons.append(f"effective_rank_below_min:{effective_rank}<{min_effective_rank}")
            if not allow_low_effective_rank and effective_rank_ratio < min_effective_rank_ratio:
                unusable_reasons.append(f"effective_rank_ratio_below_min:{effective_rank_ratio:.4f}<{min_effective_rank_ratio:.4f}")
            if support_empty_fraction > max_support_empty_fraction:
                unusable_reasons.append(f"support_empty_basis_fraction_high:{support_empty_fraction:.4f}>{max_support_empty_fraction:.4f}")
            if (not allow_single_label) and collapsed_fraction > max_collapsed_fraction:
                unusable_reasons.append(f"collapsed_basis_fraction_high:{collapsed_fraction:.4f}>{max_collapsed_fraction:.4f}")
            dictionary_usable = bool(not unusable_reasons)
            warnings: list[str] = []
            if not dictionary_usable:
                warnings.append("dictionary unusable: " + "; ".join(unusable_reasons))
            if collapsed_count:
                warnings.append(f"collapsed_basis_count={collapsed_count}")
            warnings.extend(str(value) for value in match_info.get("warnings", []))
            selected_indices = effective_indices if effective_indices else list(range(rank))
            basis_stats_eff = [basis_stats[index] for index in selected_indices]
            quality_report = {
                "requested_rank": int(rank),
                "effective_rank": int(effective_rank),
                "effective_rank_ratio": float(effective_rank_ratio),
                "dictionary_usable": bool(dictionary_usable),
                "unusable_reasons": unusable_reasons,
                "support_empty_basis_count": int(support_empty_count),
                "support_empty_basis_fraction": float(support_empty_fraction),
                "label_empty_basis_count": int(label_empty_count),
                "label_empty_basis_fraction": float(label_empty_fraction),
                "collapsed_basis_count": int(collapsed_count),
                "collapsed_basis_fraction": float(collapsed_fraction),
                "support_label_assignment": [int(value) for value in support_label_assignment],
                "matched_label_basis_indices": [int(support_label_assignment[index]) for index in selected_indices],
                "matched_overlap_scores": [float(value) for value in overlap_values],
                "mean_matched_overlap": float(match_info["mean_matched_overlap"]),
                "min_matched_overlap": float(match_info["min_matched_overlap"]),
                "low_overlap_basis_indices": [int(value) for value in match_info["low_overlap_basis_indices"]],
                "low_overlap_basis_count": int(match_info["low_overlap_basis_count"]),
                "matching_method": str(match_info["matching_method"]),
                "mean_support_area": float(sum(float(stat["support_area_estimate"]) for stat in basis_stats) / max(1, rank)),
                "mean_support_active_pixels": float(sum(float(stat["support_active_pixel_count"]) for stat in basis_stats) / max(1, rank)),
                "mean_support_largest_component_ratio": float(sum(float(stat["support_largest_component_ratio"]) for stat in basis_stats) / max(1, rank)),
                "mean_label_entropy": float(sum(float(stat["basis_label_entropy_on_mass"]) for stat in basis_stats) / max(1, rank)),
                "mean_label_dominant_ratio": float(sum(float(stat["basis_dominant_label_ratio_on_mass"]) for stat in basis_stats) / max(1, rank)),
                "label1_dominant_basis_fraction": float(sum(1 for stat in basis_stats if int(stat["basis_dominant_label"]) == 1) / max(1, rank)),
                "category_label_stats": category_stats,
                "allow_single_label": bool(allow_single_label),
                "warnings": warnings,
            }
            eff = np.asarray(selected_indices, dtype=np.int64)
            support_codes_eff = support_codes[:, eff]
            label_codes_eff = matched_label_codes[:, eff]
            code_mode = _build_code_modes(category, label_codes_eff, cfg)
            assigned = code_mode.pop("_assigned")
            code_mode["sample_to_code_mode"] = {sample_id: int(mode_index) for sample_id, mode_index in zip(sample_ids, assigned)}
            tensor_dtype = getattr(torch, "float16") if bool(cfg.get("save_basis_float16", False)) else getattr(torch, "float32")
            category_payload: dict[str, object] = {
                "num_samples": int(len(samples)),
                "requested_rank": int(rank),
                "rank": int(len(selected_indices)),
                "effective_rank": int(effective_rank),
                "effective_basis_indices": [int(index) for index in effective_indices],
                "dropped_empty_basis_indices": [int(index) for index in dropped_empty],
                "dropped_collapsed_basis_indices": [int(index) for index in dropped_collapsed],
                "dictionary_usable": bool(dictionary_usable),
                "basis_label_prob_16": torch.tensor(matched_label_prob_16[eff], dtype=tensor_dtype),
                "basis_fg_mask_prob": torch.tensor(basis_fg_mask_prob[eff], dtype=tensor_dtype),
                "basis_label_argmax": basis_label_argmax[eff].tolist(),
                "basis_label_mass": torch.tensor(basis_label_mass[eff], dtype=tensor_dtype),
                "support_basis_mass": torch.tensor(support_mass[eff], dtype=tensor_dtype),
                "label_basis_argmax": raw_label_basis_argmax[eff].astype(np.int64).tolist(),
                "matched_label_basis_argmax": matched_label_basis_argmax[eff].astype(np.int64).tolist(),
                "support_label_assignment": [int(value) for value in support_label_assignment],
                "support_label_overlap_matrix": match_info["overlap_matrix"],
                "matched_overlap_scores": [float(value) for value in match_info["matched_overlap_scores"]],
                "mean_matched_overlap": float(match_info["mean_matched_overlap"]),
                "min_matched_overlap": float(match_info["min_matched_overlap"]),
                "low_overlap_basis_indices": [int(value) for value in match_info["low_overlap_basis_indices"]],
                "matching_method": str(match_info["matching_method"]),
                "basis_stats": basis_stats_eff,
                "support_basis_stats": [support_stats[index] for index in selected_indices],
                "label_basis_stats": [raw_label_stats[index] for index in selected_indices],
                "matched_label_basis_stats": [matched_label_stats[index] for index in selected_indices],
                "quality_report": quality_report,
                "label_weights_16": [float(value) for value in label_weights.tolist()],
                "label_freq_16": [float(value) for value in label_freq.tolist()],
                "feature_mode": "dual_dictionary",
                "basis_for_label_prob": "label_dictionary",
                "basis_for_support": "support_dictionary",
                "normalization_config": {
                    "dual_dictionary": cfg.get("dual_dictionary", {}),
                    "support_dictionary": support_cf,
                    "label_dictionary": label_cf,
                    "prior_fusion": fusion_cf,
                    "category_adaptive": adaptive_cf,
                    "category_overrides": cfg.get("category_overrides", {}),
                },
                "sample_ids": sample_ids,
                "sample_to_basis_argmax": {sample_id: int(np.argmax(label_codes_eff[index]).item()) for index, sample_id in enumerate(sample_ids)},
                "code_mode": code_mode,
                "code_mode_summary": {
                    "mode_count": code_mode.get("mode_count", 0),
                    "mode_weights": code_mode.get("mode_weights", []),
                    "mode_num_samples": code_mode.get("mode_num_samples", []),
                },
            }
            if bool(cfg.get("save_sample_codes", True)):
                category_payload["codes_H"] = torch.tensor(label_codes_eff, dtype=getattr(torch, "float32"))
                category_payload["support_codes_H"] = torch.tensor(support_codes_eff, dtype=getattr(torch, "float32"))
            category_payloads[category] = category_payload
        except Exception as error:
            raise RuntimeError(f"Failed to build dual NMF dictionary bank for category={category!r}: {error}") from error
    quality_summary = _quality_summary(category_payloads)
    return {
        "schema_version": DICTIONARY_BANK_SCHEMA_VERSION,
        "enabled": True,
        "inspect_only": bool(config.get("inspect_only", True)),
        "categories": category_payloads,
        "quality_summary": quality_summary,
        "config": config,
    }


def build_dictionary_bank(items: list[dict[str, object]], config: dict[str, Any] | None) -> dict[str, object]:
    cfg = _merged_config(config)
    if not bool(cfg.get("enabled", False)):
        return {
            "schema_version": DICTIONARY_BANK_SCHEMA_VERSION,
            "enabled": False,
            "inspect_only": bool(cfg.get("inspect_only", True)),
            "categories": {},
        }
    dual_cf = cfg.get("dual_dictionary", {}) if isinstance(cfg.get("dual_dictionary"), dict) else {}
    if bool(dual_cf.get("enabled", True)):
        return _build_dual_dictionary_bank(items, cfg)
    np = _require_numpy()
    torch = _require_torch()
    sklearn_decomposition = _require_sklearn_decomposition()
    nmf_cls = getattr(sklearn_decomposition, "NMF")
    categories = sorted({str(item.get("category")) for item in items if not bool(item.get("is_empty_foreground", False))})
    category_payloads: dict[str, dict[str, object]] = {}
    for category in categories:
        samples = [item for item in items if str(item.get("category")) == category and not bool(item.get("is_empty_foreground", False))]
        if not samples:
            continue
        sample_ids = [str(sample["sample_id"]) for sample in samples]
        try:
            x_c, label_freq, label_weights = _weighted_feature_matrix(samples, cfg, np)
            rank = _rank_for_category(category, cfg, int(x_c.shape[0]))
            sparsity_cf = cfg.get("sparsity", {}) if isinstance(cfg.get("sparsity"), dict) else {}
            model = nmf_cls(
                n_components=rank,
                init=str(cfg.get("init", "nndsvda")),
                solver=str(cfg.get("solver", "cd")),
                beta_loss=str(cfg.get("beta_loss", "frobenius")),
                alpha_W=float(sparsity_cf.get("alpha_W", cfg.get("alpha_W", 0.001))),
                alpha_H=float(sparsity_cf.get("alpha_H", cfg.get("alpha_H", 0.001))),
                l1_ratio=float(sparsity_cf.get("l1_ratio", cfg.get("l1_ratio", 0.7))),
                max_iter=int(cfg.get("max_iter", 1000)),
                random_state=int(cfg.get("random_state", 0)),
            )
            codes_h = model.fit_transform(x_c).astype(np.float32)
            basis_weighted = np.maximum(np.asarray(model.components_, dtype=np.float32).reshape(rank, NUM_LABELS, CANONICAL_SIZE, CANONICAL_SIZE), 0.0)
            basis_for_label_prob = basis_weighted.astype(np.float32, copy=True)
            if bool(cfg.get("basis_unweight_after_fit", True)):
                unweight_eps = float(cfg.get("eps", 1.0e-8))
                basis_support = basis_weighted / np.maximum(label_weights[None, :, None, None], unweight_eps)
                basis_for_support_name = "unweighted"
            else:
                support_mode = str(cfg.get("support_basis_mode", "weighted"))
                if support_mode != "weighted":
                    raise ValueError(f"Unsupported nmf_dictionary.support_basis_mode={support_mode!r}; expected 'weighted' when basis_unweight_after_fit=false.")
                basis_support = basis_weighted
                basis_for_support_name = "weighted"
            basis_for_label_prob = np.maximum(basis_for_label_prob.astype(np.float32), 0.0)
            basis_support = np.maximum(basis_support.astype(np.float32), 0.0)
            if bool(cfg.get("basis_normalize_mass_max", True)):
                label_mass_before_norm = basis_for_label_prob.sum(axis=1)
                label_scale = label_mass_before_norm.max(axis=(1, 2))
                support_mass_before_norm = basis_support.sum(axis=1)
                support_scale = support_mass_before_norm.max(axis=(1, 2))
                norm_eps = float(cfg.get("eps", 1.0e-8))
                basis_for_label_prob = np.where(label_scale[:, None, None, None] > 0.0, basis_for_label_prob / np.maximum(label_scale[:, None, None, None], norm_eps), basis_for_label_prob)
                basis_support = np.where(support_scale[:, None, None, None] > 0.0, basis_support / np.maximum(support_scale[:, None, None, None], norm_eps), basis_support)
            basis_label_before_prune = basis_for_label_prob.copy()
            basis_support_before_prune = basis_support.copy()
            prune_cf = cfg.get("prune", {}) if isinstance(cfg.get("prune"), dict) else {}
            prune_enabled = bool(prune_cf.get("enabled", False))
            prune_percentile = float(prune_cf.get("percentile", cfg.get("prune_percentile", 0.0)))
            prune_min_value = float(prune_cf.get("min_value", 0.0))
            if prune_enabled and prune_min_value > 0.0:
                basis_for_label_prob[basis_for_label_prob < prune_min_value] = 0.0
                basis_support[basis_support < prune_min_value] = 0.0
            if (prune_enabled or prune_percentile > 0.0) and prune_percentile > 0.0:
                for target_basis in (basis_for_label_prob, basis_support):
                    for basis_index in range(rank):
                        for label_index in range(NUM_LABELS):
                            channel = target_basis[basis_index, label_index]
                            positive = channel[channel > 0.0]
                            if positive.size:
                                threshold = float(np.percentile(positive, prune_percentile))
                                channel[channel < threshold] = 0.0
            if prune_enabled:
                label_empty_after_prune = [
                    basis_index
                    for basis_index in range(rank)
                    if float(basis_for_label_prob[basis_index].sum()) <= 0.0 and float(basis_label_before_prune[basis_index].sum()) > 0.0
                ]
                support_empty_after_prune = [
                    basis_index
                    for basis_index in range(rank)
                    if float(basis_support[basis_index].sum()) <= 0.0 and float(basis_support_before_prune[basis_index].sum()) > 0.0
                ]
                if label_empty_after_prune or support_empty_after_prune:
                    basis_for_label_prob = basis_label_before_prune.copy()
                    basis_support = basis_support_before_prune.copy()
                    prune_auto_disabled = True
                else:
                    prune_auto_disabled = False
            else:
                prune_auto_disabled = False
            prune_empty_restored_count = 0
            for basis_index in range(rank):
                if float(basis_for_label_prob[basis_index].sum()) <= 0.0 and float(basis_label_before_prune[basis_index].sum()) > 0.0:
                    basis_for_label_prob[basis_index] = basis_label_before_prune[basis_index]
                    prune_empty_restored_count += 1
                if float(basis_support[basis_index].sum()) <= 0.0 and float(basis_support_before_prune[basis_index].sum()) > 0.0:
                    basis_support[basis_index] = basis_support_before_prune[basis_index]
                    prune_empty_restored_count += 1
            basis_mass = basis_support.sum(axis=1).astype(np.float32)
            label_mass = basis_for_label_prob.sum(axis=1).astype(np.float32)
            eps = float(cfg.get("eps", 1.0e-8))
            gamma_fg = float(cfg.get("gamma_fg", 1.0))
            fg_mask_prob = (1.0 - np.exp(-gamma_fg * basis_mass)).astype(np.float32)[:, None, :, :]
            label_prob_16 = ((basis_for_label_prob + eps) / (label_mass[:, None, :, :] + (NUM_LABELS * eps))).astype(np.float32)
            if bool(cfg.get("save_basis_float16", False)):
                tensor_dtype = getattr(torch, "float16")
            else:
                tensor_dtype = getattr(torch, "float32")
            label_prob_tensor = torch.tensor(label_prob_16, dtype=tensor_dtype)
            fg_prob_tensor = torch.tensor(fg_mask_prob, dtype=tensor_dtype)
            mass_tensor = torch.tensor(basis_mass, dtype=tensor_dtype)
            basis_argmax: list[list[list[int]]] = []
            basis_confidence: list[list[list[float]]] = []
            basis_stats: list[dict[str, object]] = []
            stats_threshold = 0.05
            for basis_index in range(rank):
                argmax_grid, confidence_grid = _masked_argmax_and_confidence(
                    label_prob_tensor[basis_index],
                    mass_tensor[basis_index],
                    threshold=stats_threshold,
                )
                basis_argmax.append(argmax_grid)
                basis_confidence.append(confidence_grid)
                basis_stats.append(
                    _basis_stats(
                        basis_index=basis_index,
                        basis_mass=mass_tensor[basis_index],
                        fg_mask_prob=fg_prob_tensor[basis_index],
                        label_prob_16=label_prob_tensor[basis_index],
                        argmax_grid=argmax_grid,
                        threshold=stats_threshold,
                        eps=eps,
                    )
                )
            collapse_summary = _anti_collapse_summary(basis_stats, cfg)
            for warning in collapse_summary["anti_collapse_warnings"]:
                print(f"WARNING category={category}: {warning}", flush=True)
            code_mode = _build_code_modes(category, codes_h, cfg)
            assigned = code_mode.pop("_assigned")
            code_mode["sample_to_code_mode"] = {sample_id: int(mode_index) for sample_id, mode_index in zip(sample_ids, assigned)}
            category_payload: dict[str, object] = {
                "num_samples": int(len(samples)),
                "rank": int(rank),
                "basis_label_prob_16": label_prob_tensor,
                "basis_fg_mask_prob": fg_prob_tensor,
                "basis_label_argmax": basis_argmax,
                "basis_label_mass": mass_tensor,
                "basis_stats": basis_stats,
                "basis_for_label_prob": "weighted",
                "basis_for_support": basis_for_support_name,
                "label_weights_16": [float(value) for value in label_weights.tolist()],
                "label_freq_16": [float(value) for value in label_freq.tolist()],
                "feature_mode": str(cfg.get("feature_mode", "label_balanced_onehot")),
                "normalization_config": {
                    "label_balance": cfg.get("label_balance", {}),
                    "sample_normalize": cfg.get("sample_normalize", {}),
                    "channel_normalize": cfg.get("channel_normalize", {}),
                    "basis_unweight_after_fit": bool(cfg.get("basis_unweight_after_fit", True)),
                    "basis_normalize_mass_max": bool(cfg.get("basis_normalize_mass_max", True)),
                    "support_basis_mode": str(cfg.get("support_basis_mode", "weighted")),
                    "sparsity": cfg.get("sparsity", {}),
                    "prune": cfg.get("prune", {}),
                    "prune_empty_restored_count": int(prune_empty_restored_count),
                    "prune_auto_disabled": bool(prune_auto_disabled),
                },
                "collapse_summary": collapse_summary,
                "sample_ids": sample_ids,
                "sample_to_basis_argmax": {sample_id: int(np.argmax(codes_h[index]).item()) for index, sample_id in enumerate(sample_ids)},
                "code_mode": code_mode,
                "code_mode_summary": {
                    "mode_count": code_mode.get("mode_count", 0),
                    "mode_weights": code_mode.get("mode_weights", []),
                    "mode_num_samples": code_mode.get("mode_num_samples", []),
                },
            }
            if bool(cfg.get("save_sample_codes", True)):
                category_payload["codes_H"] = torch.tensor(codes_h, dtype=getattr(torch, "float32"))
            category_payloads[category] = category_payload
        except Exception as error:
            raise RuntimeError(f"Failed to build NMF dictionary bank for category={category!r}: {error}") from error
    return {
        "schema_version": DICTIONARY_BANK_SCHEMA_VERSION,
        "enabled": True,
        "inspect_only": bool(cfg.get("inspect_only", True)),
        "categories": category_payloads,
        "config": cfg,
    }


def dictionary_bank_summary(dictionary_bank: object) -> dict[str, object]:
    if not isinstance(dictionary_bank, dict):
        return {"enabled": False, "schema_version": None, "categories": [], "usable_categories": [], "unusable_categories": [], "total_nmf_basis": 0, "total_effective_basis": 0, "warnings_by_category": {}, "unusable_reasons_by_category": {}, "mean_matched_overlap_by_category": {}}
    categories = dictionary_bank.get("categories", {})
    if not isinstance(categories, dict):
        return {"enabled": bool(dictionary_bank.get("enabled", False)), "schema_version": dictionary_bank.get("schema_version"), "categories": [], "usable_categories": [], "unusable_categories": [], "total_nmf_basis": 0, "total_effective_basis": 0, "warnings_by_category": {}, "unusable_reasons_by_category": {}, "mean_matched_overlap_by_category": {}}
    unusable = []
    usable = []
    for category, entry in categories.items():
        if isinstance(entry, dict) and bool(entry.get("quality_report", {}).get("dictionary_usable", True)):
            usable.append(str(category))
        else:
            unusable.append(str(category))
    quality_summary = dictionary_bank.get("quality_summary", {}) if isinstance(dictionary_bank.get("quality_summary", {}), dict) else {}
    return {
        "enabled": bool(dictionary_bank.get("enabled", False)),
        "schema_version": dictionary_bank.get("schema_version"),
        "categories": sorted(str(category) for category in categories.keys()),
        "usable_categories": sorted(usable),
        "unusable_categories": sorted(unusable),
        "total_nmf_basis": int(sum(int(entry.get("rank", 0)) for entry in categories.values() if isinstance(entry, dict))),
        "total_effective_basis": int(sum(int(entry.get("effective_rank", entry.get("rank", 0))) for entry in categories.values() if isinstance(entry, dict))),
        "warnings_by_category": quality_summary.get("warnings_by_category", {}),
        "unusable_reasons_by_category": quality_summary.get("unusable_reasons_by_category", {}),
        "mean_matched_overlap_by_category": quality_summary.get("mean_matched_overlap_by_category", {}),
    }


def inspect_dictionary_bank_category(
    dictionary_bank: dict[str, object],
    category: str,
    output_dir: object,
    *,
    basis_mass_threshold: float = 0.05,
    num_basis_samples: int = 16,
    cols: int = 8,
    cell_size: int = 18,
    save_tiled_grid: Any,
    grid_to_rgb: Any,
    prob_grid_to_rgb: Any,
    save_json: Any,
) -> dict[str, object]:
    from pathlib import Path

    categories = dictionary_bank.get("categories")
    if not isinstance(categories, dict):
        raise ValueError("dictionary_bank.categories is missing or invalid. Rebuild cache with nmf_dictionary.enabled=true.")
    if category not in categories:
        raise ValueError(f"Category {category!r} not found in dictionary_bank.categories.")
    entry = categories[category]
    if not isinstance(entry, dict):
        raise ValueError(f"dictionary_bank.categories[{category!r}] must be a dict.")
    label_prob = entry.get("basis_label_prob_16")
    mass = entry.get("basis_label_mass")
    fg_prob = entry.get("basis_fg_mask_prob")
    if label_prob is None or mass is None or fg_prob is None:
        raise ValueError(f"dictionary_bank category {category!r} is missing basis tensors.")
    if hasattr(label_prob, "detach"):
        label_prob = label_prob.detach().cpu()
    if hasattr(mass, "detach"):
        mass = mass.detach().cpu()
    if hasattr(fg_prob, "detach"):
        fg_prob = fg_prob.detach().cpu()
    support_mass = entry.get("support_basis_mass")
    if hasattr(support_mass, "detach"):
        support_mass = support_mass.detach().cpu()
    label_argmax = entry.get("label_basis_argmax")
    if hasattr(label_argmax, "detach"):
        label_argmax = label_argmax.detach().cpu()
    matched_label_argmax = entry.get("matched_label_basis_argmax")
    if hasattr(matched_label_argmax, "detach"):
        matched_label_argmax = matched_label_argmax.detach().cpu()
    rank = int(entry.get("rank", len(label_prob)))
    schema_version = dictionary_bank.get("schema_version")
    if schema_version != DICTIONARY_BANK_SCHEMA_VERSION:
        print(f"WARNING dictionary_bank schema_version={schema_version!r}; expected {DICTIONARY_BANK_SCHEMA_VERSION!r}.", flush=True)
    max_tiles = min(rank, max(1, int(num_basis_samples)))
    argmax_tiles = []
    mass_tiles = []
    confidence_tiles = []
    support_tiles = []
    label_argmax_tiles = []
    matched_label_argmax_tiles = []
    labels = []
    basis_rows: list[dict[str, object]] = []
    for basis_index in range(max_tiles):
        argmax_grid, confidence_grid = _masked_argmax_and_confidence(label_prob[basis_index], mass[basis_index], threshold=float(basis_mass_threshold))
        mass_grid = _as_float_grid(mass[basis_index], context=f"{category}.basis_label_mass[{basis_index}]")
        support_grid = _as_float_grid(support_mass[basis_index], context=f"{category}.support_basis_mass[{basis_index}]") if support_mass is not None else mass_grid
        if label_argmax is not None:
            label_grid_value = label_argmax.tolist() if hasattr(label_argmax, "tolist") else label_argmax
            label_grid = [[int(label_grid_value[basis_index][y_pos][x_pos]) + 1 if int(label_grid_value[basis_index][y_pos][x_pos]) >= 0 else 0 for x_pos in range(CANONICAL_SIZE)] for y_pos in range(CANONICAL_SIZE)]
        else:
            label_grid = argmax_grid
        if matched_label_argmax is not None:
            matched_label_grid_value = matched_label_argmax.tolist() if hasattr(matched_label_argmax, "tolist") else matched_label_argmax
            matched_label_grid = [[int(matched_label_grid_value[basis_index][y_pos][x_pos]) + 1 if int(matched_label_grid_value[basis_index][y_pos][x_pos]) >= 0 else 0 for x_pos in range(CANONICAL_SIZE)] for y_pos in range(CANONICAL_SIZE)]
        else:
            matched_label_grid = argmax_grid
        argmax_tiles.append(grid_to_rgb(argmax_grid, mode="label"))
        mass_tiles.append(prob_grid_to_rgb(mass_grid))
        confidence_tiles.append(prob_grid_to_rgb(confidence_grid))
        support_tiles.append(prob_grid_to_rgb(support_grid))
        label_argmax_tiles.append(grid_to_rgb(label_grid, mode="label"))
        matched_label_argmax_tiles.append(grid_to_rgb(matched_label_grid, mode="label"))
        labels.append(f"k={basis_index}")
        basis_rows.append(
            {
                "basis_index": int(basis_index),
                "argmax_grid": argmax_grid,
                "label_mass_mean": float(sum(sum(row) for row in mass_grid) / 400.0),
                "label_mass_max": float(max(max(row) for row in mass_grid)),
                "confidence_mean_on_mass": float(
                    sum(confidence_grid[y_pos][x_pos] for y_pos in range(CANONICAL_SIZE) for x_pos in range(CANONICAL_SIZE) if mass_grid[y_pos][x_pos] > float(basis_mass_threshold))
                    / max(1, sum(1 for y_pos in range(CANONICAL_SIZE) for x_pos in range(CANONICAL_SIZE) if mass_grid[y_pos][x_pos] > float(basis_mass_threshold)))
                ),
            }
        )
    output_path = Path(output_dir)
    save_tiled_grid(support_tiles, labels, output_path / f"{category}_nmf_support_basis_mass_grid.png", cols=int(cols), cell_size=int(cell_size))
    save_tiled_grid(label_argmax_tiles, labels, output_path / f"{category}_nmf_label_basis_argmax_grid.png", cols=int(cols), cell_size=int(cell_size))
    save_tiled_grid(matched_label_argmax_tiles, labels, output_path / f"{category}_nmf_matched_label_basis_argmax_grid.png", cols=int(cols), cell_size=int(cell_size))
    save_tiled_grid(argmax_tiles, labels, output_path / f"{category}_nmf_fused_basis_argmax_grid.png", cols=int(cols), cell_size=int(cell_size))
    save_tiled_grid(mass_tiles, labels, output_path / f"{category}_nmf_fused_basis_mass_grid.png", cols=int(cols), cell_size=int(cell_size))
    save_tiled_grid(argmax_tiles, labels, output_path / f"{category}_nmf_basis_label_argmax_grid.png", cols=int(cols), cell_size=int(cell_size))
    save_tiled_grid(mass_tiles, labels, output_path / f"{category}_nmf_basis_mass_grid.png", cols=int(cols), cell_size=int(cell_size))
    save_tiled_grid(confidence_tiles, labels, output_path / f"{category}_nmf_basis_confidence_grid.png", cols=int(cols), cell_size=int(cell_size))
    code_mode = entry.get("code_mode", {})
    code_mode_summary = dict(code_mode) if isinstance(code_mode, dict) else {}
    collapse_summary = entry.get("collapse_summary", {})
    if isinstance(collapse_summary, dict):
        for warning in collapse_summary.get("anti_collapse_warnings", []):
            print(f"WARNING category={category}: {warning}", flush=True)
    save_json(
        output_path / f"{category}_nmf_basis_stats.json",
        {
            "schema_version": dictionary_bank.get("schema_version"),
            "category": category,
            "rank": int(entry.get("rank", 0)),
            "requested_rank": int(entry.get("requested_rank", entry.get("rank", 0))),
            "effective_rank": int(entry.get("effective_rank", entry.get("rank", 0))),
            "dictionary_usable": bool(entry.get("dictionary_usable", True)),
            "quality_report": entry.get("quality_report", {}),
            "effective_basis_indices": entry.get("effective_basis_indices", []),
            "dropped_empty_basis_indices": entry.get("dropped_empty_basis_indices", []),
            "dropped_collapsed_basis_indices": entry.get("dropped_collapsed_basis_indices", []),
            "num_samples": int(entry.get("num_samples", 0)),
            "feature_mode": entry.get("feature_mode"),
            "basis_for_label_prob": entry.get("basis_for_label_prob"),
            "basis_for_support": entry.get("basis_for_support"),
            "label_weights_16": entry.get("label_weights_16", []),
            "label_freq_16": entry.get("label_freq_16", []),
            "normalization_config": entry.get("normalization_config", {}),
            "support_label_assignment": entry.get("support_label_assignment", []),
            "support_label_overlap_matrix": entry.get("support_label_overlap_matrix", []),
            "matched_overlap_scores": entry.get("matched_overlap_scores", []),
            "mean_matched_overlap": entry.get("mean_matched_overlap"),
            "min_matched_overlap": entry.get("min_matched_overlap"),
            "low_overlap_basis_indices": entry.get("low_overlap_basis_indices", []),
            "matching_method": entry.get("matching_method"),
            "basis_mass_threshold": float(basis_mass_threshold),
            "basis_stats": entry.get("basis_stats", []),
            "support_basis_stats": entry.get("support_basis_stats", []),
            "label_basis_stats": entry.get("label_basis_stats", []),
            "matched_label_basis_stats": entry.get("matched_label_basis_stats", []),
            **(collapse_summary if isinstance(collapse_summary, dict) else {}),
            "visualized_basis": basis_rows,
            "code_mode_summary": {
                "mode_count": code_mode_summary.get("mode_count", 0),
                "mode_weights": code_mode_summary.get("mode_weights", []),
                "mode_num_samples": code_mode_summary.get("mode_num_samples", []),
            },
        },
    )
    save_json(
        output_path / f"{category}_nmf_code_modes.json",
        {
            "category": category,
            "rank": int(entry.get("rank", 0)),
            "num_samples": int(entry.get("num_samples", 0)),
            "code_mode": code_mode_summary,
        },
    )
    dense_tiles = []
    dense_labels = []
    sparse_tiles = []
    sparse_labels = []
    label_prob_list = label_prob.tolist() if hasattr(label_prob, "tolist") else label_prob
    mass_list = mass.tolist() if hasattr(mass, "tolist") else mass
    mode_means = code_mode_summary.get("mode_code_mean", [])
    bank_config = dictionary_bank.get("config", {})
    sparse_top_r = int(bank_config.get("inspect_sparse_top_r", 3)) if isinstance(bank_config, dict) else 3

    def alpha_to_grid(alpha_values: list[float]) -> list[list[int]]:
        positive_total = sum(max(0.0, float(value)) for value in alpha_values)
        normalized_alpha = [max(0.0, float(value)) / positive_total for value in alpha_values] if positive_total > 0.0 else [0.0 for _ in alpha_values]
        soft_basis = [
            [
                [
                    sum(
                        float(normalized_alpha[basis_index])
                        * float(label_prob_list[basis_index][label_index][y_pos][x_pos])
                        * float(mass_list[basis_index][y_pos][x_pos])
                        for basis_index in range(min(rank, len(normalized_alpha)))
                    )
                    for x_pos in range(CANONICAL_SIZE)
                ]
                for y_pos in range(CANONICAL_SIZE)
            ]
            for label_index in range(NUM_LABELS)
        ]
        soft_mass = [[sum(float(soft_basis[label_index][y_pos][x_pos]) for label_index in range(NUM_LABELS)) for x_pos in range(CANONICAL_SIZE)] for y_pos in range(CANONICAL_SIZE)]
        soft_argmax, _ = _masked_argmax_and_confidence(soft_basis, soft_mass, threshold=float(basis_mass_threshold))
        return soft_argmax

    if isinstance(mode_means, list):
        for mode_index, alpha in enumerate(mode_means[:max_tiles]):
            if not isinstance(alpha, list):
                continue
            dense_tiles.append(grid_to_rgb(alpha_to_grid([float(value) for value in alpha]), mode="label"))
            dense_labels.append(f"m={mode_index}")
            sparse_alpha = [0.0 for _ in alpha]
            top_indices = sorted(range(len(alpha)), key=lambda index: float(alpha[index]), reverse=True)[: max(1, sparse_top_r)]
            for index in top_indices:
                sparse_alpha[index] = float(alpha[index])
            sparse_tiles.append(grid_to_rgb(alpha_to_grid(sparse_alpha), mode="label"))
            sparse_labels.append(f"m={mode_index}")
    if dense_tiles:
        save_tiled_grid(dense_tiles, dense_labels, output_path / f"{category}_nmf_sampled_prior_dense_grid.png", cols=int(cols), cell_size=int(cell_size))
    if sparse_tiles:
        save_tiled_grid(sparse_tiles, sparse_labels, output_path / f"{category}_nmf_sampled_prior_sparse_grid.png", cols=int(cols), cell_size=int(cell_size))
        save_tiled_grid(sparse_tiles, sparse_labels, output_path / f"{category}_nmf_sampled_prior_grid.png", cols=int(cols), cell_size=int(cell_size))
    return {
        "basis_argmax": output_path / f"{category}_nmf_basis_label_argmax_grid.png",
        "support_basis_mass": output_path / f"{category}_nmf_support_basis_mass_grid.png",
        "label_basis_argmax": output_path / f"{category}_nmf_label_basis_argmax_grid.png",
        "matched_label_basis_argmax": output_path / f"{category}_nmf_matched_label_basis_argmax_grid.png",
        "fused_basis_argmax": output_path / f"{category}_nmf_fused_basis_argmax_grid.png",
        "fused_basis_mass": output_path / f"{category}_nmf_fused_basis_mass_grid.png",
        "basis_mass": output_path / f"{category}_nmf_basis_mass_grid.png",
        "basis_confidence": output_path / f"{category}_nmf_basis_confidence_grid.png",
        "basis_stats": output_path / f"{category}_nmf_basis_stats.json",
        "code_modes": output_path / f"{category}_nmf_code_modes.json",
        "sampled_prior_dense": output_path / f"{category}_nmf_sampled_prior_dense_grid.png" if dense_tiles else None,
        "sampled_prior_sparse": output_path / f"{category}_nmf_sampled_prior_sparse_grid.png" if sparse_tiles else None,
        "sampled_prior": output_path / f"{category}_nmf_sampled_prior_grid.png" if sparse_tiles else None,
    }


def inspect_all_dictionary_bank_categories(
    dictionary_bank: dict[str, object],
    output_dir: object,
    *,
    basis_mass_threshold: float = 0.05,
    num_basis_samples: int = 16,
    cols: int = 8,
    cell_size: int = 18,
    save_tiled_grid: Any,
    grid_to_rgb: Any,
    prob_grid_to_rgb: Any,
    save_json: Any,
) -> dict[str, object]:
    from pathlib import Path

    output_path = Path(output_dir)
    categories = dictionary_bank.get("categories", {})
    if not isinstance(categories, dict):
        raise ValueError("dictionary_bank.categories is missing or invalid.")
    category_paths: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for category in sorted(str(key) for key in categories.keys()):
        category_dir = output_path / category
        category_paths[category] = inspect_dictionary_bank_category(
            dictionary_bank,
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
        entry = categories[category]
        quality = entry.get("quality_report", {}) if isinstance(entry, dict) else {}
        rows.append(
            {
                "category": category,
                "requested_rank": quality.get("requested_rank", entry.get("requested_rank") if isinstance(entry, dict) else None),
                "effective_rank": quality.get("effective_rank", entry.get("effective_rank") if isinstance(entry, dict) else None),
                "effective_rank_ratio": quality.get("effective_rank_ratio"),
                "dictionary_usable": quality.get("dictionary_usable", entry.get("dictionary_usable") if isinstance(entry, dict) else None),
                "collapsed_basis_fraction": quality.get("collapsed_basis_fraction"),
                "mean_matched_overlap": quality.get("mean_matched_overlap"),
                "min_matched_overlap": quality.get("min_matched_overlap"),
                "mean_support_area": quality.get("mean_support_area"),
                "mean_label_entropy": quality.get("mean_label_entropy"),
                "label1_dominant_basis_fraction": quality.get("label1_dominant_basis_fraction"),
                "warnings": quality.get("warnings", []),
                "unusable_reasons": quality.get("unusable_reasons", []),
            }
        )
    summary = {
        "schema_version": dictionary_bank.get("schema_version"),
        "quality_summary": dictionary_bank.get("quality_summary", {}),
        "categories": rows,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "all_categories_dictionary_quality_summary.json"
    md_path = output_path / "all_categories_dictionary_quality_summary.md"
    save_json(json_path, summary)
    lines = [
        f"# Dictionary Quality Summary",
        "",
        f"schema_version: {dictionary_bank.get('schema_version')}",
        "",
        "| category | usable | requested_rank | effective_rank | effective_rank_ratio | collapsed_fraction | mean_matched_overlap | min_matched_overlap | label1_fraction | warnings | unusable_reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        warnings = ", ".join(str(value) for value in row.get("warnings", []))
        unusable_reasons = ", ".join(str(value) for value in row.get("unusable_reasons", []))
        lines.append(
            "| {category} | {usable} | {requested} | {effective} | {ratio} | {collapsed} | {mean_overlap} | {min_overlap} | {label1} | {warnings} | {reasons} |".format(
                category=row["category"],
                usable=row.get("dictionary_usable"),
                requested=row.get("requested_rank"),
                effective=row.get("effective_rank"),
                ratio=row.get("effective_rank_ratio"),
                collapsed=row.get("collapsed_basis_fraction"),
                mean_overlap=row.get("mean_matched_overlap"),
                min_overlap=row.get("min_matched_overlap"),
                label1=row.get("label1_dominant_basis_fraction"),
                warnings=warnings,
                reasons=unusable_reasons,
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "categories": category_paths,
        "summary": {
            "json": json_path,
            "markdown": md_path,
        },
    }
