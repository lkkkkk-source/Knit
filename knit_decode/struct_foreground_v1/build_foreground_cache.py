from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .utils import IGNORE_INDEX, bbox_vector, canonicalize_foreground, descriptor_stats_by_category, finish_progress, foreground_descriptor, format_metric_line, load_config, print_progress


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
        if all((candidate_root / str(row["target_path"])).exists() for row in rows[: min(4, len(rows))]):
            return candidate_root
    return manifest_path.parent


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data_cf = config["data"]
    planner_cf = config["planner"]
    manifest_path = Path(args.manifest or data_cf["train_manifest"])
    output_dir = Path(data_cf["cache_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output or (output_dir / f"{manifest_path.stem}.pt"))
    rows = _load_manifest(manifest_path)
    manifest_root = _infer_manifest_root(manifest_path, rows)
    total = len(rows)
    items: list[dict[str, object]] = []
    descriptors_by_category: dict[str, list[list[float]]] = {}
    nondegenerate_by_category: dict[str, list[dict[str, object]]] = {}
    descriptor_slices: dict[str, object] | None = None
    from .utils import load_label_grid

    for index, row in enumerate(rows, start=1):
        target_path = Path(str(row["target_path"]))
        if not target_path.is_absolute():
            target_path = (manifest_root / target_path).resolve()
        y20 = load_label_grid(target_path)
        canonical = canonicalize_foreground(y20, background_class_id=int(data_cf["background_class_id"]), canonical_size=int(data_cf["canonical_size"]))
        descriptor = foreground_descriptor(canonical["fg_y20"], canonical["fg_mask20"], canonical["bbox"])
        fg_area = sum(int(value) for row_fg in canonical["fg_mask20"] for value in row_fg) / float(max(1, int(data_cf["canonical_size"]) * int(data_cf["canonical_size"])))
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
            "fg_area": fg_area,
            "bbox_stats": bbox_vector(canonical["bbox"], canonical_size=int(data_cf["canonical_size"])),
            **descriptor,
        }
        items.append(item)
        descriptor_slices = descriptor["descriptor_slices"]
        if not item["is_empty_foreground"]:
            descriptors_by_category.setdefault(item["category"], []).append(item["descriptor"])
            nondegenerate_by_category.setdefault(item["category"], []).append(item)
        print_progress("fg-cache", index, total, f"empty={int(item['is_empty_foreground'])}")
    finish_progress()

    split_name = manifest_path.stem.lower()
    is_train_like = "train" in split_name
    if is_train_like and not args.fit_kmeans:
        raise ValueError("Train foreground cache build requires --fit-kmeans.")
    if (not is_train_like) and args.fit_kmeans:
        raise ValueError("Validation/test foreground cache must not refit KMeans; use --kmeans-source-cache.")

    num_modes_per_category = int(planner_cf["num_modes_per_category"])
    min_samples_per_mode = int(planner_cf["min_samples_per_mode"])
    category_kmeans_centers: dict[str, list[list[float]]] = {}
    category_to_num_modes: dict[str, int] = {}
    descriptors_by_category_stats: dict[str, list[list[float]]] = {}
    descriptor_mean_by_category: dict[str, list[float]] = {}
    descriptor_std_by_category: dict[str, list[float]] = {}
    category_foreground_area_stats: dict[str, dict[str, float]] = {}

    if args.fit_kmeans:
        sklearn_cluster = _require_sklearn()
        cluster_cls = getattr(sklearn_cluster, "MiniBatchKMeans")
        descriptors_by_category_stats, descriptor_mean_by_category, descriptor_std_by_category, category_foreground_area_stats = descriptor_stats_by_category(items, sorted({item["category"] for item in items}))
        for category, descs in descriptors_by_category.items():
            samples_c = nondegenerate_by_category[category]
            k_c = min(num_modes_per_category, max(2, math.floor(len(samples_c) / max(1, min_samples_per_mode))))
            k_c = min(k_c, max(1, len(samples_c)))
            kmeans = cluster_cls(n_clusters=k_c, batch_size=1024, random_state=42, n_init="auto")
            assigned = kmeans.fit_predict(descs).tolist()
            category_kmeans_centers[category] = [list(map(float, row)) for row in kmeans.cluster_centers_.tolist()]
            category_to_num_modes[category] = k_c
            for sample, z in zip(samples_c, assigned):
                sample["local_z"] = int(z)
                sample["num_modes_for_category"] = k_c
        for item in items:
            if item["is_empty_foreground"]:
                item["local_z"] = 0
                item["num_modes_for_category"] = max(1, category_to_num_modes.get(item["category"], 1))
    else:
        torch = __import__("torch")
        source_payload = torch.load(Path(args.kmeans_source_cache), map_location="cpu")
        category_kmeans_centers = source_payload["category_kmeans_centers"]
        category_to_num_modes = source_payload["category_to_num_modes"]
        descriptors_by_category_stats = source_payload["descriptors_by_category"]
        descriptor_mean_by_category = source_payload["descriptor_mean_by_category"]
        descriptor_std_by_category = source_payload["descriptor_std_by_category"]
        category_foreground_area_stats = source_payload["category_foreground_area_stats"]
        global_centers = []
        for centers in category_kmeans_centers.values():
            global_centers.extend(centers)
        global_tensor = torch.tensor(global_centers, dtype=torch.float32) if global_centers else None
        for item in items:
            category = item["category"]
            descriptor = item["descriptor"]
            if item["is_empty_foreground"]:
                item["local_z"] = 0
                item["num_modes_for_category"] = max(1, category_to_num_modes.get(category, 1))
                item["is_unseen_category"] = category not in category_kmeans_centers
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
                if global_tensor is not None:
                    desc_tensor = torch.tensor(descriptor, dtype=torch.float32)
                    dist = ((desc_tensor.unsqueeze(0) - global_tensor) ** 2).sum(dim=-1)
                    item["local_z"] = int(dist.argmin().item())
                else:
                    item["local_z"] = 0
                item["num_modes_for_category"] = num_modes_per_category

    centroid_sketch_by_category: dict[str, dict[int, dict[str, object]]] = {}
    for category, samples in nondegenerate_by_category.items():
        num_modes_c = int(category_to_num_modes.get(category, 1))
        centroid_sketch_by_category[category] = {}
        for mode_index in range(num_modes_c):
            mode_samples = [sample for sample in samples if int(sample["local_z"]) == mode_index]
            if not mode_samples:
                continue
            fg_mask_mean = [[sum(int(sample["fg_mask20"][y_pos][x_pos]) for sample in mode_samples) / float(len(mode_samples)) for x_pos in range(20)] for y_pos in range(20)]
            centroid_sketch_by_category[category][mode_index] = {
                "centroid_fg_mask": fg_mask_mean,
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
            "canonical_size": int(data_cf["canonical_size"]),
            "background_class_id": int(data_cf["background_class_id"]),
            "ignore_index": IGNORE_INDEX,
            "split": "train" if is_train_like else ("val" if "val" in split_name else "test"),
            "source_kmeans_cache": str(args.kmeans_source_cache) if args.kmeans_source_cache is not None else None,
        },
        "items": items,
        "category_kmeans_centers": category_kmeans_centers,
        "category_to_num_modes": category_to_num_modes,
        "descriptors_by_category": descriptors_by_category_stats,
        "descriptor_mean_by_category": descriptor_mean_by_category,
        "descriptor_std_by_category": descriptor_std_by_category,
        "category_foreground_area_stats": category_foreground_area_stats,
        "centroid_sketch_by_category": centroid_sketch_by_category,
        "descriptor_slices": descriptor_slices or {},
        "config": config,
    }
    torch = __import__("torch")
    torch.save(payload, output_path)
    print(format_metric_line("saved foreground cache:", [("output", str(output_path)), ("items", len(items)), ("categories", len(category_kmeans_centers))]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
