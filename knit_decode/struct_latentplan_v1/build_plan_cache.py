from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .utils import compute_category_fg_stats, compute_category_occ_stats, compute_grammar_descriptor, compute_plan_statistics, ensure_palette_path, finish_progress, format_metric_line, load_config, print_progress


def _require_sklearn() -> object:
    import importlib

    try:
        return importlib.import_module("sklearn.cluster")
    except ImportError as error:
        raise ImportError("scikit-learn is required for plan cache building. Install with `pip install -e .[train]`.") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build latent global planning cache from instruction17 labels.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-modes", type=int, default=None)
    parser.add_argument("--fit-kmeans", action="store_true")
    parser.add_argument("--kmeans-source-cache", type=Path, default=None)
    parser.add_argument("--exclude-degenerate-from-kmeans", action="store_true", default=True)
    parser.add_argument("--write-debug-json", action="store_true")
    return parser


def _build_entries(manifest_path: Path, palette_path: Path, background_class_id: int, coarse_size: int, coarse_threshold: float, num_classes: int) -> tuple[list[dict[str, object]], list[list[float]], dict[str, list[list[float]]], dict[str, object]]:
    from knit_decode.parser_t_inverse.dataset import ParserInverseDataset

    dataset = ParserInverseDataset(manifest_path, palette_path=palette_path, image_size=(160, 160))
    entries: list[dict[str, object]] = []
    descriptors: list[list[float]] = []
    descriptors_by_category: dict[str, list[list[float]]] = {}
    descriptor_slices: dict[str, object] | None = None
    total = len(dataset.samples)
    for index, sample in enumerate(dataset.samples, start=1):
        item = dataset[index - 1]
        y20 = item["target"].tolist()
        stats = compute_plan_statistics(y20, background_class_id=background_class_id, coarse_size=coarse_size, coarse_threshold=coarse_threshold)
        grammar = compute_grammar_descriptor(y20, background_class_id=background_class_id, coarse_threshold=coarse_threshold, num_classes=num_classes)
        entry = {
            "sample_id": sample["sample_id"],
            "category": sample["category"],
            "input_path": sample["input_path"],
            "target_path": sample["target_path"],
            "index_path": sample["index_path"],
            "y20": y20,
            "fg20": [[1 if value else 0 for value in row] for row in stats["fg20"]],
            "o5": grammar["o5"],
            "c5": grammar["c5"],
            "o10": grammar["o10"],
            "c10": grammar["c10"],
            "r17": stats["r17"],
            "fg_ratio": stats["fg_ratio"],
            "bbox_stats": stats["bbox_stats"],
            "component_stats": stats["component_stats"],
            "row_projection": grammar["row_projection"],
            "col_projection": grammar["col_projection"],
            "row_run_stats": grammar["row_run_stats"],
            "col_run_stats": grammar["col_run_stats"],
            "adjacency_signature": grammar["adjacency_signature"],
            "transition_2x2_stats": grammar["transition_2x2_stats"],
            "vertical_continuity": grammar["vertical_continuity"],
            "horizontal_continuity": grammar["horizontal_continuity"],
            "symmetry_score": grammar["symmetry_score"],
            "center_band_score": grammar["center_band_score"],
            "stripe_score": grammar["stripe_score"],
            "grammar_signature": grammar["grammar_signature"],
            "descriptor": grammar["descriptor"],
            "descriptor_slices": grammar["descriptor_slices"],
            "is_all_background": 1 if stats["fg_ratio"] <= 0.0 else 0,
            "is_all_foreground": 1 if stats["fg_ratio"] >= 1.0 else 0,
            "is_degenerate": 1 if stats["fg_ratio"] <= 0.0 or stats["fg_ratio"] >= 1.0 else 0,
        }
        entries.append(entry)
        descriptors.append(grammar["descriptor"])
        descriptors_by_category.setdefault(sample["category"], []).append(grammar["descriptor"])
        descriptor_slices = grammar["descriptor_slices"]
        print_progress("plan-cache", index, total, f"fg_ratio={stats['fg_ratio']:.4f}")
    finish_progress()
    return entries, descriptors, descriptors_by_category, (descriptor_slices or {})


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data_cf = config["data"]
    planner_cf = config["planner"]
    manifest_path = Path(args.manifest or data_cf["train_manifest"])
    output_dir = Path(data_cf["plan_cache_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output or (output_dir / f"{manifest_path.stem}.pt"))
    palette_path = ensure_palette_path(manifest_path, data_cf.get("palette_path"))
    entries, descriptors, descriptors_by_category, descriptor_slices = _build_entries(
        manifest_path,
        palette_path=palette_path,
        background_class_id=int(data_cf["background_class_id"]),
        coarse_size=int(data_cf["coarse_size"]),
        coarse_threshold=float(data_cf.get("coarse_threshold", 0.25)),
        num_classes=int(data_cf["num_classes"]),
    )
    num_modes = int(args.num_modes or planner_cf["num_modes"])
    num_modes_per_category = int(planner_cf.get("num_modes_per_category", 16))
    max_num_modes_per_category = int(planner_cf.get("max_num_modes_per_category", num_modes_per_category))
    min_samples_per_mode = int(planner_cf.get("min_samples_per_mode", 20))
    use_category_local = bool(config.get("cache", {}).get("use_category_local_modes", False))
    split_name = manifest_path.stem.lower()
    is_train_like = "train" in split_name
    should_fit = bool(args.fit_kmeans)
    if args.kmeans_source_cache is None and not should_fit:
        if is_train_like:
            raise ValueError("Train cache build requires --fit-kmeans to make the z space explicit.")
        raise ValueError("Validation/test cache build requires --kmeans-source-cache and must not silently refit KMeans.")
    if should_fit and not is_train_like:
        raise ValueError(f"Refusing to fit KMeans on non-train manifest: {manifest_path}. Use --kmeans-source-cache instead.")
    if should_fit:
        sklearn_cluster = _require_sklearn()
        cluster_cls = getattr(sklearn_cluster, "MiniBatchKMeans")
        if use_category_local:
            category_kmeans_centers: dict[str, list[list[float]]] = {}
            category_to_num_modes: dict[str, int] = {}
            assignments = []
            for entry in entries:
                assignments.append(0)
            sample_id_to_assignment: dict[str, int] = {}
            for category, category_desc in descriptors_by_category.items():
                category_entries_all = [entry for entry in entries if entry["category"] == category]
                category_entries_fit = [entry for entry in category_entries_all if not entry["is_degenerate"]] if args.exclude_degenerate_from_kmeans else category_entries_all
                if len(category_entries_fit) < 2:
                    print(f"warning: category {category} has too few non-degenerate samples for KMeans; falling back to all samples")
                    category_entries_fit = category_entries_all
                category_desc_fit = [entry["descriptor"] for entry in category_entries_fit]
                k_c = min(num_modes_per_category, max(2, math.floor(len(category_desc_fit) / max(1, min_samples_per_mode))))
                k_c = min(k_c, max(1, len(category_desc)))
                kmeans = cluster_cls(n_clusters=k_c, batch_size=1024, random_state=42, n_init="auto")
                assigned_fit = kmeans.fit_predict(category_desc_fit).tolist()
                category_kmeans_centers[category] = [list(map(float, row)) for row in kmeans.cluster_centers_.tolist()]
                category_to_num_modes[category] = k_c
                for entry, assignment in zip(category_entries_fit, assigned_fit):
                    sample_id_to_assignment[entry["sample_id"]] = int(assignment)
                if len(category_entries_fit) != len(category_entries_all):
                    import torch
                    centers_tensor = getattr(torch, "tensor")(category_kmeans_centers[category], dtype=getattr(torch, "float32"))
                    for entry in category_entries_all:
                        if entry["sample_id"] in sample_id_to_assignment:
                            continue
                        descriptor_tensor = getattr(torch, "tensor")(entry["descriptor"], dtype=getattr(torch, "float32"))
                        dist = ((descriptor_tensor.unsqueeze(0) - centers_tensor) ** 2).sum(dim=-1)
                        sample_id_to_assignment[entry["sample_id"]] = int(dist.argmin().item())
            for index, entry in enumerate(entries):
                assignments[index] = int(sample_id_to_assignment[entry["sample_id"]])
            centers = []
        else:
            kmeans = cluster_cls(n_clusters=num_modes, batch_size=1024, random_state=42, n_init="auto")
            assignments = kmeans.fit_predict(descriptors).tolist()
            centers = [list(map(float, row)) for row in kmeans.cluster_centers_.tolist()]
            category_kmeans_centers = {}
            category_to_num_modes = {}
        source_cache = None
        fitted_on = str(manifest_path)
    else:
        import torch

        source_path = Path(args.kmeans_source_cache)
        source_payload = getattr(torch, "load")(source_path, map_location="cpu")
        source_meta = source_payload["meta"]
        centers = source_payload["kmeans_centers"]
        source_category_centers = source_payload.get("category_kmeans_centers", {})
        source_category_modes = source_payload.get("category_to_num_modes", {})
        if int(source_meta["num_modes"]) != num_modes:
            raise ValueError(
                f"KMeans num_modes mismatch: source={source_meta['num_modes']} current={num_modes}"
            )
        descriptor_dim = len(descriptors[0]) if descriptors else 0
        if int(source_meta["descriptor_dim"]) != descriptor_dim:
            raise ValueError(
                f"Descriptor dim mismatch: source={source_meta['descriptor_dim']} current={descriptor_dim}"
            )
        if use_category_local:
            assignments = []
            category_kmeans_centers = source_category_centers
            category_to_num_modes = source_category_modes
            global_centers = source_payload.get("kmeans_centers", [])
            global_tensor = getattr(torch, "tensor")(global_centers, dtype=getattr(torch, "float32")) if global_centers else None
            for entry, descriptor in zip(entries, descriptors):
                category = entry["category"]
                if category in category_kmeans_centers:
                    centers_tensor = getattr(torch, "tensor")(category_kmeans_centers[category], dtype=getattr(torch, "float32"))
                    descriptor_tensor = getattr(torch, "tensor")(descriptor, dtype=getattr(torch, "float32"))
                    dist = ((descriptor_tensor.unsqueeze(0) - centers_tensor) ** 2).sum(dim=-1)
                    assignments.append(int(dist.argmin().item()))
                    entry["is_unseen_category"] = False
                else:
                    entry["is_unseen_category"] = True
                    if global_tensor is not None:
                        descriptor_tensor = getattr(torch, "tensor")(descriptor, dtype=getattr(torch, "float32"))
                        dist = ((descriptor_tensor.unsqueeze(0) - global_tensor) ** 2).sum(dim=-1)
                        assignments.append(int(dist.argmin().item()))
                    else:
                        assignments.append(0)
        else:
            centers_tensor = getattr(torch, "tensor")(centers, dtype=getattr(torch, "float32"))
            descriptor_tensor = getattr(torch, "tensor")(descriptors, dtype=getattr(torch, "float32"))
            dist = ((descriptor_tensor.unsqueeze(1) - centers_tensor.unsqueeze(0)) ** 2).sum(dim=-1)
            assignments = dist.argmin(dim=-1).tolist()
            category_kmeans_centers = {}
            category_to_num_modes = {}
        source_cache = str(source_path)
        fitted_on = str(source_meta.get("kmeans_fitted_on", source_meta.get("manifest", source_path)))
    for entry, assignment in zip(entries, assignments):
        entry["z"] = int(assignment)
        entry["local_z"] = int(assignment)
        entry["num_modes_for_category"] = int(category_to_num_modes.get(entry["category"], len(category_kmeans_centers.get(entry["category"], [])) or num_modes))
    category_fg_stats = compute_category_fg_stats(entries, sorted({entry["category"] for entry in entries}))
    category_occ_stats = compute_category_occ_stats(entries, sorted({entry["category"] for entry in entries}))
    payload = {
        "meta": {
            "manifest": str(manifest_path),
            "palette_path": str(palette_path),
            "num_modes": num_modes,
            "num_modes_per_category": num_modes_per_category,
            "max_num_modes_per_category": max_num_modes_per_category,
            "coarse_size": int(data_cf["coarse_size"]),
            "background_class_id": int(data_cf["background_class_id"]),
            "descriptor_dim": len(descriptors[0]) if descriptors else 0,
            "kmeans_fitted_on": fitted_on,
            "split": "train" if should_fit else ("val" if "val" in split_name else "test"),
            "source_kmeans_cache": source_cache,
            "num_classes": int(data_cf["num_classes"]),
            "mode_type": planner_cf.get("mode_type", "global_kmeans"),
        },
        "items": entries,
        "descriptors": descriptors,
        "z": assignments,
        "kmeans_centers": centers,
        "category_kmeans_centers": category_kmeans_centers,
        "category_to_num_modes": category_to_num_modes,
        "category_fg_stats": category_fg_stats,
        "category_occ_stats": category_occ_stats,
        "descriptor_slices": descriptor_slices,
        "config": config,
        "category_to_id": {category: index for index, category in enumerate(sorted({entry["category"] for entry in entries}))},
        "id_to_category": {index: category for index, category in enumerate(sorted({entry["category"] for entry in entries}))},
    }
    import torch

    getattr(torch, "save")(payload, output_path)
    if args.write_debug_json:
        debug_path = output_path.with_suffix(".json")
        debug_path.write_text(json.dumps({"meta": payload["meta"], "items": entries}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        format_metric_line(
            "saved plan cache:",
            [
                ("output", str(output_path)),
                ("entries", len(entries)),
                ("num_modes", num_modes),
            ],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
