from __future__ import annotations

import importlib
import math
from collections import Counter
from typing import Any


DICTIONARY_BANK_SCHEMA_VERSION = "foreground_v1_nmf_dictionary_bank_v1"
CANONICAL_SIZE = 20
NUM_LABELS = 16
IGNORE_INDEX = -100


DEFAULT_NMF_DICTIONARY_CONFIG: dict[str, Any] = {
    "enabled": False,
    "inspect_only": True,
    "rank_default": 8,
    "rank_by_category": {},
    "max_iter": 1000,
    "init": "nndsvda",
    "solver": "cd",
    "beta_loss": "frobenius",
    "alpha_W": 0.001,
    "alpha_H": 0.001,
    "l1_ratio": 0.7,
    "random_state": 0,
    "eps": 1.0e-8,
    "prune_percentile": 0.0,
    "gamma_fg": 1.0,
    "save_basis_float16": False,
    "save_sample_codes": True,
    "code_mode_count_default": 8,
    "code_mode_count_by_category": {},
}


def _merged_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_NMF_DICTIONARY_CONFIG)
    if isinstance(config, dict):
        merged.update(config)
    if not isinstance(merged.get("rank_by_category"), dict):
        merged["rank_by_category"] = {}
    if not isinstance(merged.get("code_mode_count_by_category"), dict):
        merged["code_mode_count_by_category"] = {}
    return merged


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
    if label_mass_total > 0.0:
        label_hist = [value / label_mass_total for value in label_hist]
        dominant_label_ratio = max(label_hist)
        label_diversity = float(sum(1 for value in label_hist if value > 1.0e-4))
        label_entropy = -sum(value * math.log(value + 1e-12) for value in label_hist if value > 0.0)
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
    return {
        "basis_index": int(basis_index),
        "basis_mass_sum": float(sum(mass_values)),
        "basis_area_estimate": float(sum(1 for value in fg_values if value > float(threshold)) / float(CANONICAL_SIZE * CANONICAL_SIZE)),
        "label_hist_16": [float(value) for value in label_hist],
        "label_diversity_on_mass": float(label_diversity),
        "dominant_label_ratio_on_mass": float(dominant_label_ratio),
        "label_entropy_on_mass": float(label_entropy),
        "transition_h": float(transition_h / max(1, edge_h)),
        "transition_v": float(transition_v / max(1, edge_v)),
        "motif2_entropy": float(motif2_entropy),
        "foreground_mass_mean": float(sum(fg_values) / float(max(1, len(fg_values)))),
        "foreground_mass_max": float(max(fg_values) if fg_values else 0.0),
        "active_mass_pixel_count": int(len(active_positions)),
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


def build_dictionary_bank(items: list[dict[str, object]], config: dict[str, Any] | None) -> dict[str, object]:
    cfg = _merged_config(config)
    if not bool(cfg.get("enabled", False)):
        return {
            "schema_version": DICTIONARY_BANK_SCHEMA_VERSION,
            "enabled": False,
            "inspect_only": bool(cfg.get("inspect_only", True)),
            "categories": {},
        }
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
            x_c = np.asarray([_fg_y20_to_feature(sample["fg_y20"]) for sample in samples], dtype=np.float32)
            rank = _rank_for_category(category, cfg, int(x_c.shape[0]))
            model = nmf_cls(
                n_components=rank,
                init=str(cfg.get("init", "nndsvda")),
                solver=str(cfg.get("solver", "cd")),
                beta_loss=str(cfg.get("beta_loss", "frobenius")),
                alpha_W=float(cfg.get("alpha_W", 0.001)),
                alpha_H=float(cfg.get("alpha_H", 0.001)),
                l1_ratio=float(cfg.get("l1_ratio", 0.7)),
                max_iter=int(cfg.get("max_iter", 1000)),
                random_state=int(cfg.get("random_state", 0)),
            )
            codes_h = model.fit_transform(x_c).astype(np.float32)
            basis = np.maximum(np.asarray(model.components_, dtype=np.float32).reshape(rank, NUM_LABELS, CANONICAL_SIZE, CANONICAL_SIZE), 0.0)
            prune_percentile = float(cfg.get("prune_percentile", 0.0))
            if prune_percentile > 0.0:
                for basis_index in range(rank):
                    positive = basis[basis_index][basis[basis_index] > 0.0]
                    if positive.size:
                        threshold = float(np.percentile(positive, prune_percentile))
                        basis[basis_index][basis[basis_index] < threshold] = 0.0
            basis_mass = basis.sum(axis=1).astype(np.float32)
            eps = float(cfg.get("eps", 1.0e-8))
            gamma_fg = float(cfg.get("gamma_fg", 1.0))
            fg_mask_prob = (1.0 - np.exp(-gamma_fg * basis_mass)).astype(np.float32)[:, None, :, :]
            label_prob_16 = ((basis + eps) / (basis_mass[:, None, :, :] + (NUM_LABELS * eps))).astype(np.float32)
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
                    )
                )
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
                "sample_ids": sample_ids,
                "sample_to_basis_argmax": {sample_id: int(np.argmax(codes_h[index]).item()) for index, sample_id in enumerate(sample_ids)},
                "code_mode": code_mode,
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
        return {"enabled": False, "categories": [], "total_nmf_basis": 0}
    categories = dictionary_bank.get("categories", {})
    if not isinstance(categories, dict):
        return {"enabled": bool(dictionary_bank.get("enabled", False)), "categories": [], "total_nmf_basis": 0}
    return {
        "enabled": bool(dictionary_bank.get("enabled", False)),
        "categories": sorted(str(category) for category in categories.keys()),
        "total_nmf_basis": int(sum(int(entry.get("rank", 0)) for entry in categories.values() if isinstance(entry, dict))),
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
    rank = int(entry.get("rank", len(label_prob)))
    max_tiles = min(rank, max(1, int(num_basis_samples)))
    argmax_tiles = []
    mass_tiles = []
    confidence_tiles = []
    labels = []
    basis_rows: list[dict[str, object]] = []
    for basis_index in range(max_tiles):
        argmax_grid, confidence_grid = _masked_argmax_and_confidence(label_prob[basis_index], mass[basis_index], threshold=float(basis_mass_threshold))
        mass_grid = _as_float_grid(mass[basis_index], context=f"{category}.basis_label_mass[{basis_index}]")
        argmax_tiles.append(grid_to_rgb(argmax_grid, mode="label"))
        mass_tiles.append(prob_grid_to_rgb(mass_grid))
        confidence_tiles.append(prob_grid_to_rgb(confidence_grid))
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
    save_tiled_grid(argmax_tiles, labels, output_path / f"{category}_nmf_basis_label_argmax_grid.png", cols=int(cols), cell_size=int(cell_size))
    save_tiled_grid(mass_tiles, labels, output_path / f"{category}_nmf_basis_mass_grid.png", cols=int(cols), cell_size=int(cell_size))
    save_tiled_grid(confidence_tiles, labels, output_path / f"{category}_nmf_basis_confidence_grid.png", cols=int(cols), cell_size=int(cell_size))
    code_mode = entry.get("code_mode", {})
    code_mode_summary = dict(code_mode) if isinstance(code_mode, dict) else {}
    save_json(
        output_path / f"{category}_nmf_basis_stats.json",
        {
            "schema_version": dictionary_bank.get("schema_version"),
            "category": category,
            "rank": int(entry.get("rank", 0)),
            "num_samples": int(entry.get("num_samples", 0)),
            "basis_mass_threshold": float(basis_mass_threshold),
            "basis_stats": entry.get("basis_stats", []),
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
    sampled_tiles = []
    sampled_labels = []
    label_prob_list = label_prob.tolist() if hasattr(label_prob, "tolist") else label_prob
    mass_list = mass.tolist() if hasattr(mass, "tolist") else mass
    mode_means = code_mode_summary.get("mode_code_mean", [])
    if isinstance(mode_means, list):
        for mode_index, alpha in enumerate(mode_means[:max_tiles]):
            if not isinstance(alpha, list):
                continue
            soft_basis = [
                [
                    [
                        sum(
                            float(alpha[basis_index])
                            * float(label_prob_list[basis_index][label_index][y_pos][x_pos])
                            * float(mass_list[basis_index][y_pos][x_pos])
                            for basis_index in range(min(rank, len(alpha)))
                        )
                        for x_pos in range(CANONICAL_SIZE)
                    ]
                    for y_pos in range(CANONICAL_SIZE)
                ]
                for label_index in range(NUM_LABELS)
            ]
            soft_mass = [[sum(float(soft_basis[label_index][y_pos][x_pos]) for label_index in range(NUM_LABELS)) for x_pos in range(CANONICAL_SIZE)] for y_pos in range(CANONICAL_SIZE)]
            soft_argmax, _ = _masked_argmax_and_confidence(soft_basis, soft_mass, threshold=float(basis_mass_threshold))
            sampled_tiles.append(grid_to_rgb(soft_argmax, mode="label"))
            sampled_labels.append(f"m={mode_index}")
    if sampled_tiles:
        save_tiled_grid(sampled_tiles, sampled_labels, output_path / f"{category}_nmf_sampled_prior_grid.png", cols=int(cols), cell_size=int(cell_size))
    return {
        "basis_argmax": output_path / f"{category}_nmf_basis_label_argmax_grid.png",
        "basis_mass": output_path / f"{category}_nmf_basis_mass_grid.png",
        "basis_confidence": output_path / f"{category}_nmf_basis_confidence_grid.png",
        "basis_stats": output_path / f"{category}_nmf_basis_stats.json",
        "code_modes": output_path / f"{category}_nmf_code_modes.json",
        "sampled_prior": output_path / f"{category}_nmf_sampled_prior_grid.png" if sampled_tiles else None,
    }
