from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import cast

from .grammar_energy import build_grammar_bank
from .nmf_dictionary import build_dictionary_bank, dictionary_bank_summary
from .utils import IGNORE_INDEX, REQUIRED_FOREGROUND_CACHE_SCHEMA_VERSION, assert_no_forbidden_cache_fields, bbox_vector, canonicalize_foreground, clustering_feature_from_parts, descriptor_global_stats, descriptor_stats_by_category, ensure_descriptor_dim, finish_progress, foreground_descriptor, format_metric_line, foreground_area, load_config, print_progress, require_centroid_sketch_fields, require_foreground_cache_fields, require_ignore_index, resolve_canonical_mode, resolve_manifest_path, validate_foreground_labels


def _require_sklearn() -> object:
    import importlib

    try:
        return importlib.import_module("sklearn.cluster")
    except ImportError as error:
        raise ImportError("scikit-learn is required for foreground cache building. Install with `pip install scikit-learn`.") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build foreground-canonical cache from instruction17 labels.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fit-kmeans", action="store_true")
    parser.add_argument("--kmeans-source-cache", type=Path, default=None)
    return parser


def _load_manifest(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _infer_manifest_root(manifest_path: Path, rows: list[dict[str, object]]) -> Path:
    search_roots = [manifest_path.parent, *manifest_path.parents]
    for candidate_root in search_roots:
        if all((candidate_root / str(row["target_path"])).exists() for row in rows[: min(32, len(rows))] if isinstance(row.get("target_path"), str)):
            return candidate_root
    return manifest_path.parent


def _default_output_path(output_dir: Path, split_name: str) -> Path:
    split_lower = split_name.lower()
    if "train" in split_lower:
        return output_dir / "foreground_cache_train.pt"
    if "val" in split_lower:
        return output_dir / "foreground_cache_val.pt"
    if "test" in split_lower:
        return output_dir / "foreground_cache_test.pt"
    return output_dir / f"{split_name}.pt"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data_cf = config["data"]
    planner_cf = config["planner"]
    clustering_cf = cast(dict[str, object], config.get("clustering", {})) if isinstance(config.get("clustering", {}), dict) else {}
    grammar_bank_cf = cast(dict[str, object], config.get("grammar_bank", {})) if isinstance(config.get("grammar_bank", {}), dict) else {}
    nmf_dictionary_cf = cast(dict[str, object], config.get("nmf_dictionary", {})) if isinstance(config.get("nmf_dictionary", {}), dict) else {}
    canonical_mode = resolve_canonical_mode(data_cf)
    ignore_index = require_ignore_index(data_cf)
    manifest_path = Path(args.manifest or data_cf["train_manifest"])
    output_dir = Path(data_cf["cache_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    split_name = manifest_path.stem.lower()
    output_path = Path(args.output or _default_output_path(output_dir, manifest_path.stem))
    rows = _load_manifest(manifest_path)
    manifest_root = _infer_manifest_root(manifest_path, rows)
    total = len(rows)
    items: list[dict[str, object]] = []
    descriptors_by_category: dict[str, list[list[float]]] = {}
    clustering_features_by_category: dict[str, list[list[float]]] = {}
    clustering_feature_by_sample_id: dict[str, list[float]] = {}
    nondegenerate_by_category: dict[str, list[dict[str, object]]] = {}
    descriptor_slices: dict[str, object] | None = None
    from .utils import load_label_grid

    for index, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        target_path = resolve_manifest_path(row["target_path"], manifest_root, sample_id=sample_id, field_name="target_path")
        _ = resolve_manifest_path(row["input_path"], manifest_root, sample_id=sample_id, field_name="input_path")
        _ = resolve_manifest_path(row["index_path"], manifest_root, sample_id=sample_id, field_name="index_path")
        y20 = load_label_grid(target_path, sample_id=sample_id)
        canonical = canonicalize_foreground(
            y20,
            background_class_id=int(data_cf["background_class_id"]),
            canonical_size=int(data_cf["canonical_size"]),
            canonical_mode=canonical_mode,
            ignore_index=ignore_index,
        )
        descriptor = foreground_descriptor(canonical["fg_y20"], canonical["fg_mask20"], canonical["bbox"])
        validate_foreground_labels(canonical["fg_y20"], canonical["fg_mask20"], canonical_size=int(data_cf["canonical_size"]), context=f"cache[{sample_id}]")
        ensure_descriptor_dim(descriptor["descriptor"], context=f"cache[{sample_id}]")
        fg_area = foreground_area(canonical["fg_mask20"])
        clustering_feature_payload = clustering_feature_from_parts(
            canonical["fg_y20"],
            canonical["fg_mask20"],
            bbox_vector(canonical["bbox"], canonical_size=int(data_cf["canonical_size"])),
            descriptor["row_projection"],
            descriptor["col_projection"],
            canonical_size=int(data_cf["canonical_size"]),
            label_spatial_feature_weight=float(clustering_cf.get("label_spatial_feature_weight", 0.5)),
            label_spatial_area_norm_weight=float(clustering_cf.get("label_spatial_area_norm_weight", 1.0)),
            label_spatial_channel_balanced_weight=float(clustering_cf.get("label_spatial_channel_balanced_weight", 1.0)),
            label_transition_feature_weight=float(clustering_cf.get("label_transition_feature_weight", 0.75)),
            mask_feature_weight=float(clustering_cf.get("mask_feature_weight", 0.05)),
            row_col_feature_weight=float(clustering_cf.get("row_col_feature_weight", 0.05)),
            bbox_feature_weight=float(clustering_cf.get("bbox_feature_weight", 0.0)),
        )
        item = {
            "sample_id": row["sample_id"],
            "category": row["category"],
            "input_path": row["input_path"],
            "target_path": row["target_path"],
            "index_path": row["index_path"],
            "original_y20": y20,
            "original_bbox": canonical["bbox"],
            "fg_y20": canonical["fg_y20"],
            "fg_mask20": canonical["fg_mask20"],
            "is_empty_foreground": bool(canonical["is_empty_foreground"]),
            "canonical_mode": canonical_mode,
            "fg_area": fg_area,
            "bbox_stats": bbox_vector(canonical["bbox"], canonical_size=int(data_cf["canonical_size"])),
            **descriptor,
        }
        clustering_feature = cast(list[float], clustering_feature_payload["clustering_feature"])
        clustering_feature_by_sample_id[sample_id] = clustering_feature
        items.append(item)
        descriptor_slices = descriptor["descriptor_slices"]
        if not item["is_empty_foreground"]:
            descriptors_by_category.setdefault(item["category"], []).append(item["descriptor"])
            clustering_features_by_category.setdefault(item["category"], []).append(clustering_feature)
            nondegenerate_by_category.setdefault(item["category"], []).append(item)
        print_progress("fg-cache", index, total, f"empty={int(item['is_empty_foreground'])}")
    finish_progress()

    is_train_like = "train" in split_name
    if is_train_like and not args.fit_kmeans:
        raise ValueError("Train foreground cache build requires --fit-kmeans.")
    if is_train_like and args.kmeans_source_cache is not None:
        raise ValueError("Train foreground cache build must not use --kmeans-source-cache; use --fit-kmeans only.")
    if (not is_train_like) and args.fit_kmeans:
        raise ValueError("Validation/test foreground cache must not refit KMeans; use --kmeans-source-cache.")
    if (not is_train_like) and args.kmeans_source_cache is None:
        raise ValueError("Validation/test foreground cache requires --kmeans-source-cache from the train foreground cache.")

    num_modes_per_category = int(planner_cf["num_modes_per_category"])
    min_samples_per_mode = int(planner_cf["min_samples_per_mode"])
    category_kmeans_centers: dict[str, list[list[float]]] = {}
    category_to_num_modes: dict[str, int] = {}
    descriptors_by_category_stats: dict[str, list[list[float]]] = {}
    descriptor_mean_by_category: dict[str, list[float]] = {}
    descriptor_std_by_category: dict[str, list[float]] = {}
    category_foreground_area_stats: dict[str, dict[str, float]] = {}
    centroid_sketch_by_category: dict[str, dict[int, dict[str, object]]] = {}

    descriptor_global_mean: list[float] = []
    descriptor_global_std: list[float] = []
    torch = __import__("torch")

    if args.fit_kmeans:
        sklearn_cluster = _require_sklearn()
        cluster_cls = getattr(sklearn_cluster, "MiniBatchKMeans")
        descriptors_by_category_stats, descriptor_mean_by_category, descriptor_std_by_category, category_foreground_area_stats = descriptor_stats_by_category(items, sorted({item["category"] for item in items}))
        descriptor_global_mean, descriptor_global_std = descriptor_global_stats(items)
        for category, descs in descriptors_by_category.items():
            samples_c = nondegenerate_by_category[category]
            clustering_descs = clustering_features_by_category[category]
            k_c = min(num_modes_per_category, max(2, math.floor(len(samples_c) / max(1, min_samples_per_mode))))
            k_c = min(k_c, max(1, len(samples_c)))
            kmeans = cluster_cls(n_clusters=k_c, batch_size=1024, random_state=42, n_init="auto")
            assigned = kmeans.fit_predict(clustering_descs).tolist()
            category_kmeans_centers[category] = [list(map(float, row)) for row in kmeans.cluster_centers_.tolist()]
            category_to_num_modes[category] = k_c
            for sample, z in zip(samples_c, assigned):
                sample["local_z"] = int(z)
                sample["num_modes_for_category"] = k_c
        for item in items:
            if item["is_empty_foreground"]:
                item["local_z"] = 0
                item["num_modes_for_category"] = max(1, category_to_num_modes.get(item["category"], 1))
                item["is_unseen_category"] = False
    else:
        source_payload = torch.load(Path(args.kmeans_source_cache), map_location="cpu")
        require_foreground_cache_fields(source_payload, context="Source train foreground cache")
        require_centroid_sketch_fields(source_payload, context="Source train foreground cache")
        category_kmeans_centers = source_payload["category_kmeans_centers"]
        category_to_num_modes = source_payload["category_to_num_modes"]
        descriptors_by_category_stats = source_payload["descriptors_by_category"]
        descriptor_mean_by_category = source_payload["descriptor_mean_by_category"]
        descriptor_std_by_category = source_payload["descriptor_std_by_category"]
        descriptor_global_mean = source_payload["descriptor_global_mean"]
        descriptor_global_std = source_payload["descriptor_global_std"]
        category_foreground_area_stats = source_payload["category_foreground_area_stats"]
        centroid_sketch_by_category = source_payload["centroid_sketch_by_category"]
        global_centers = []
        for centers in category_kmeans_centers.values():
            global_centers.extend(centers)
        global_tensor = torch.tensor(global_centers, dtype=torch.float32) if global_centers else None
        for item in items:
            category = item["category"]
            descriptor = clustering_feature_by_sample_id[str(item["sample_id"])]
            if item["is_empty_foreground"]:
                if category in category_kmeans_centers:
                    item["local_z"] = 0
                    item["num_modes_for_category"] = max(1, category_to_num_modes.get(category, 1))
                    item["is_unseen_category"] = False
                else:
                    item["local_z"] = -1
                    item["num_modes_for_category"] = 0
                    item["is_unseen_category"] = True
                continue
            if category in category_kmeans_centers:
                centers_tensor = torch.tensor(category_kmeans_centers[category], dtype=torch.float32)
                desc_tensor = torch.tensor(descriptor, dtype=torch.float32)
                dist = ((desc_tensor.unsqueeze(0) - centers_tensor) ** 2).sum(dim=-1)
                item["local_z"] = int(dist.argmin().item())
                item["num_modes_for_category"] = int(category_to_num_modes[category])
                item["is_unseen_category"] = False
            else:
                item["is_unseen_category"] = True
                item["local_z"] = -1
                item["num_modes_for_category"] = 0

    grammar_bank = build_grammar_bank(items, grammar_bank_cf) if bool(grammar_bank_cf.get("enabled", True)) else {"enabled": False, "categories": {}}
    dictionary_bank = build_dictionary_bank(items, nmf_dictionary_cf)
    if args.fit_kmeans:
        for category, samples in nondegenerate_by_category.items():
            num_modes_c = int(category_to_num_modes.get(category, 1))
            centroid_sketch_by_category[category] = {}
            for mode_index in range(num_modes_c):
                mode_samples = [sample for sample in samples if int(sample["local_z"]) == mode_index]
                if not mode_samples:
                    continue
                fg_mask_prob = [[sum(float(sample["fg_mask20"][y_pos][x_pos]) for sample in mode_samples) / float(len(mode_samples)) for x_pos in range(20)] for y_pos in range(20)]
                threshold = 0.5
                centroid_fg_mask = [[1 if float(fg_mask_prob[y_pos][x_pos]) >= threshold else 0 for x_pos in range(20)] for y_pos in range(20)]
                fallback_used = False
                if sum(sum(row) for row in centroid_fg_mask) <= 0:
                    threshold = max(0.1, sum(sum(row) for row in fg_mask_prob) / 400.0)
                    centroid_fg_mask = [[1 if float(fg_mask_prob[y_pos][x_pos]) >= threshold else 0 for x_pos in range(20)] for y_pos in range(20)]
                    fallback_used = True
                centroid_fg_mask_prob_tensor = getattr(torch, "tensor")(fg_mask_prob, dtype=getattr(torch, "float32")).unsqueeze(0)
                centroid_label_prob_16_tensor = getattr(torch, "tensor")(
                    [
                        [
                            [
                                sum(1.0 for sample in mode_samples if int(sample["fg_y20"][y_pos][x_pos]) == (label_index + 1)) / float(len(mode_samples))
                                for x_pos in range(20)
                            ]
                            for y_pos in range(20)
                        ]
                        for label_index in range(16)
                    ],
                    dtype=getattr(torch, "float32"),
                )
                if centroid_fg_mask_prob_tensor.dtype != getattr(torch, "float32") or tuple(centroid_fg_mask_prob_tensor.shape) != (1, 20, 20):
                    raise ValueError(f"centroid_fg_mask_prob_tensor for category={category!r} local_z={mode_index} must be float32 [1,20,20].")
                if centroid_label_prob_16_tensor.dtype != getattr(torch, "float32") or tuple(centroid_label_prob_16_tensor.shape) != (16, 20, 20):
                    raise ValueError(f"centroid_label_prob_16_tensor for category={category!r} local_z={mode_index} must be float32 [16,20,20].")
                if float(centroid_fg_mask_prob_tensor.min().item()) < 0.0 or float(centroid_fg_mask_prob_tensor.max().item()) > 1.0:
                    raise ValueError(f"centroid_fg_mask_prob_tensor for category={category!r} local_z={mode_index} must stay within [0,1].")
                if float(centroid_label_prob_16_tensor.min().item()) < 0.0 or float(centroid_label_prob_16_tensor.max().item()) > 1.0:
                    raise ValueError(f"centroid_label_prob_16_tensor for category={category!r} local_z={mode_index} must stay within [0,1].")
                centroid_sketch_by_category[category][mode_index] = {
                    "centroid_fg_mask_prob": centroid_fg_mask_prob_tensor,
                    "centroid_fg_mask": centroid_fg_mask,
                    "centroid_label_prob_16": centroid_label_prob_16_tensor,
                    "centroid_fg_mask_threshold": float(threshold),
                    "fallback_used": bool(fallback_used),
                    "num_samples": int(len(mode_samples)),
                    "centroid_label_hist": [sum(float(sample["label_hist_16"][index]) for sample in mode_samples) / float(len(mode_samples)) for index in range(16)],
                    "centroid_row_projection": [sum(float(sample["row_projection"][index]) for sample in mode_samples) / float(len(mode_samples)) for index in range(20)],
                    "centroid_col_projection": [sum(float(sample["col_projection"][index]) for sample in mode_samples) / float(len(mode_samples)) for index in range(20)],
                    "centroid_adjacency": [sum(float(sample["adjacency_signature"][index]) for sample in mode_samples) / float(len(mode_samples)) for index in range(256)],
                    "centroid_transition_stats": [sum(float(sample["transition_2x2_stats"][index]) for sample in mode_samples) / float(len(mode_samples)) for index in range(len(mode_samples[0]["transition_2x2_stats"]))],
                    "centroid_bbox_stats": [sum(float(sample["bbox_stats"][index]) for sample in mode_samples) / float(len(mode_samples)) for index in range(len(mode_samples[0]["bbox_stats"]))],
                    "nearest_train_sample_id": str(mode_samples[0]["sample_id"]),
                }
    payload = {
        "meta": {
            "manifest": str(manifest_path),
            "manifest_root": str(manifest_root),
            "canonical_size": int(data_cf["canonical_size"]),
            "background_class_id": int(data_cf["background_class_id"]),
            "ignore_index": ignore_index,
            "canonical_mode": canonical_mode,
            "schema_version": REQUIRED_FOREGROUND_CACHE_SCHEMA_VERSION,
            "split": "train" if is_train_like else ("val" if "val" in split_name else "test"),
            "source_kmeans_cache": str(args.kmeans_source_cache) if args.kmeans_source_cache is not None else None,
        },
        "items": items,
        "category_kmeans_centers": category_kmeans_centers,
        "category_to_num_modes": category_to_num_modes,
        "descriptors_by_category": descriptors_by_category_stats,
        "descriptor_mean_by_category": descriptor_mean_by_category,
        "descriptor_std_by_category": descriptor_std_by_category,
        "descriptor_global_mean": descriptor_global_mean,
        "descriptor_global_std": descriptor_global_std,
        "category_foreground_area_stats": category_foreground_area_stats,
        "centroid_sketch_by_category": centroid_sketch_by_category,
        "grammar_bank": grammar_bank,
        "dictionary_bank": dictionary_bank,
        "descriptor_slices": descriptor_slices or {},
        "config": config,
    }
    assert_no_forbidden_cache_fields(payload, context="Foreground cache payload")
    require_centroid_sketch_fields(payload, context="Foreground cache payload")
    torch.save(payload, output_path)
    file_size_mb = output_path.stat().st_size / (1024.0 * 1024.0)
    num_centroids = sum(len(centroids) for centroids in centroid_sketch_by_category.values())
    grammar_categories = sorted(grammar_bank.get("categories", {}).keys()) if isinstance(grammar_bank, dict) else []
    grammar_total_modes = sum(len(entry.get("modes", {})) for entry in grammar_bank.get("categories", {}).values()) if isinstance(grammar_bank, dict) else 0
    dictionary_summary = dictionary_bank_summary(dictionary_bank)
    print(
        format_metric_line(
            "saved foreground cache:",
            [
                ("output", str(output_path)),
                ("file_size_mb", file_size_mb),
                ("num_items", len(items)),
                ("num_centroids", num_centroids),
                ("schema_version", REQUIRED_FOREGROUND_CACHE_SCHEMA_VERSION),
                ("stores_clustering_feature", False),
                ("grammar_bank_enabled", bool(grammar_bank.get("enabled", False)) if isinstance(grammar_bank, dict) else False),
                ("grammar_categories", len(grammar_categories)),
                ("grammar_total_modes", grammar_total_modes),
                ("dictionary_bank_enabled", dictionary_summary["enabled"]),
                ("dictionary_schema", dictionary_summary["schema_version"]),
                ("dictionary_categories", ",".join(cast(list[str], dictionary_summary["categories"]))),
                ("dictionary_usable_categories", ",".join(cast(list[str], dictionary_summary["usable_categories"]))),
                ("dictionary_unusable_categories", ",".join(cast(list[str], dictionary_summary["unusable_categories"]))),
                ("total_nmf_basis", dictionary_summary["total_nmf_basis"]),
                ("total_effective_basis", dictionary_summary["total_effective_basis"]),
                ("warnings_by_category", dictionary_summary["warnings_by_category"]),
                ("stores_onehot", False),
                ("stores_X_matrix", False),
                ("stores_large_features", False),
            ],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
