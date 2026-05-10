from __future__ import annotations

import argparse
import json
from pathlib import Path

from .utils import compute_plan_statistics, ensure_palette_path, finish_progress, format_metric_line, load_config, print_progress


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
    parser.add_argument("--write-debug-json", action="store_true")
    return parser


def _build_entries(manifest_path: Path, palette_path: Path, background_class_id: int, coarse_size: int, coarse_threshold: float) -> tuple[list[dict[str, object]], list[list[float]]]:
    from knit_decode.parser_t_inverse.dataset import ParserInverseDataset

    dataset = ParserInverseDataset(manifest_path, palette_path=palette_path, image_size=(160, 160))
    entries: list[dict[str, object]] = []
    descriptors: list[list[float]] = []
    total = len(dataset.samples)
    for index, sample in enumerate(dataset.samples, start=1):
        item = dataset[index - 1]
        y20 = item["target"].tolist()
        stats = compute_plan_statistics(y20, background_class_id=background_class_id, coarse_size=coarse_size, coarse_threshold=coarse_threshold)
        entry = {
            "sample_id": sample["sample_id"],
            "category": sample["category"],
            "input_path": sample["input_path"],
            "target_path": sample["target_path"],
            "index_path": sample["index_path"],
            "y20": y20,
            "fg20": [[1 if value else 0 for value in row] for row in stats["fg20"]],
            "o5": stats["o5"],
            "c5": stats["c5"],
            "r17": stats["r17"],
            "fg_ratio": stats["fg_ratio"],
            "bbox_stats": stats["bbox_stats"],
            "component_stats": stats["component_stats"],
            "descriptor": stats["descriptor"],
        }
        entries.append(entry)
        descriptors.append(stats["descriptor"])
        print_progress("plan-cache", index, total, f"fg_ratio={stats['fg_ratio']:.4f}")
    finish_progress()
    return entries, descriptors


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
    entries, descriptors = _build_entries(
        manifest_path,
        palette_path=palette_path,
        background_class_id=int(data_cf["background_class_id"]),
        coarse_size=int(data_cf["coarse_size"]),
        coarse_threshold=float(data_cf.get("coarse_threshold", 0.25)),
    )
    sklearn_cluster = _require_sklearn()
    cluster_cls = getattr(sklearn_cluster, "MiniBatchKMeans")
    num_modes = int(args.num_modes or planner_cf["num_modes"])
    kmeans = cluster_cls(n_clusters=num_modes, batch_size=1024, random_state=42, n_init="auto")
    assignments = kmeans.fit_predict(descriptors).tolist()
    for entry, assignment in zip(entries, assignments):
        entry["z"] = int(assignment)
    payload = {
        "meta": {
            "manifest": str(manifest_path),
            "palette_path": str(palette_path),
            "num_modes": num_modes,
            "coarse_size": int(data_cf["coarse_size"]),
            "background_class_id": int(data_cf["background_class_id"]),
            "descriptor_dim": len(descriptors[0]) if descriptors else 0,
        },
        "items": entries,
        "descriptors": descriptors,
        "z": assignments,
        "kmeans_centers": [list(map(float, row)) for row in kmeans.cluster_centers_.tolist()],
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
