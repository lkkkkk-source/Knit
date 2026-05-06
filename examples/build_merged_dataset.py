from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _normalize_dataset2_id(stem: str) -> tuple[str, str | None]:
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1] in {"front", "back"}:
        return "_".join(parts[:-1]), parts[-1]
    return stem, None


def _dataset_records(root: Path) -> list[dict[str, object]]:
    dataset_root = root / "dataset"
    stitch_root = dataset_root / "stitch code patterns"
    sim_root = dataset_root / "simulation images"
    records: list[dict[str, object]] = []
    for category_dir in sorted(stitch_root.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        sim_dir = sim_root / category
        sim_index = {path.stem.lstrip("0"): path for path in sim_dir.glob("*.png")} if sim_dir.exists() else {}
        for image_path in sorted(category_dir.glob("*.png")):
            record_id = f"dataset/{category}/{image_path.stem}"
            pairing_key = image_path.stem.replace("_resized", "").split("_", 1)[0].lstrip("0")
            simulation_path = sim_index.get(pairing_key)
            records.append(
                {
                    "record_id": record_id,
                    "source_dataset": "dataset",
                    "category": category,
                    "raw_sample_id": image_path.stem,
                    "group_id": pairing_key or image_path.stem,
                    "view_variant": None,
                    "domain_membership": ["dataset_code"],
                    "split_membership": {},
                    "modalities": {
                        "stitch_code_pattern": str(image_path.relative_to(root)).replace("\\", "/"),
                        "simulation_image": str(simulation_path.relative_to(root)).replace("\\", "/") if simulation_path else None,
                    },
                    "label_capabilities": ["legend_backed_stitch_chart"],
                    "quality_flags": ["missing_simulation"] if simulation_path is None else [],
                    "provenance": {
                        "source_root": "dataset",
                        "pairing_key": pairing_key or image_path.stem,
                    },
                }
            )
    return records


def _dataset2_records(root: Path) -> tuple[list[dict[str, object]], list[str], list[dict[str, object]], list[dict[str, object]], dict[str, list[str]]]:
    dataset2_root = root / "dataset2"
    instruction_dir = dataset2_root / "instruction"
    pattern_viz_dir = dataset2_root / "pattern-viz" / "400x400"
    rendering_dir = dataset2_root / "rendering"
    real_best_rgb_dir = dataset2_root / "real" / "best" / "rgb"
    real_best_points_dir = dataset2_root / "real" / "best" / "points"
    real_best_npoints_dir = dataset2_root / "real" / "best" / "npoints"
    real_160_rgb_dir = dataset2_root / "real" / "160x160" / "rgb"
    real_160_gray_dir = dataset2_root / "real" / "160x160" / "gray"

    split_files = {
        "train_real": dataset2_root / "train_real.txt",
        "val_real": dataset2_root / "val_real.txt",
        "test_real": dataset2_root / "test_real.txt",
        "train_synt": dataset2_root / "train_synt.txt",
        "val_synt": dataset2_root / "val_synt.txt",
        "test_synt": dataset2_root / "test_synt.txt",
        "train_unsup": dataset2_root / "train_unsup.txt",
    }
    split_memberships: dict[str, list[str]] = {}
    for split_name, split_path in split_files.items():
        for item in _read_lines(split_path):
            split_memberships.setdefault(item, []).append(split_name)

    instruction_index = {path.stem: path for path in instruction_dir.glob("*.png")}
    pattern_viz_index = {path.stem: path for path in pattern_viz_dir.glob("*.png")}
    rendering_index = {path.stem: path for path in rendering_dir.glob("*.jpg")}
    best_rgb_index = {path.stem: path for path in real_best_rgb_dir.glob("*.png")}
    best_points_index = {path.stem: path for path in real_best_points_dir.glob("*.txt")}
    best_npoints_index = {path.stem: path for path in real_best_npoints_dir.glob("*.txt")}
    rgb160_index = {path.stem: path for path in real_160_rgb_dir.glob("*.png")}
    gray160_index = {path.stem: path for path in real_160_gray_dir.glob("*.png")}

    records: list[dict[str, object]] = []
    unassigned: list[str] = []
    exclusions: list[dict[str, object]] = []
    syntax_rows: list[dict[str, object]] = []
    transfer_rows: list[dict[str, object]] = []

    all_ids = sorted(instruction_index.keys())
    for sample_id in all_ids:
        base_id, view_variant = _normalize_dataset2_id(sample_id)
        split_tags = split_memberships.get(sample_id, [])
        if not split_tags:
            unassigned.append(sample_id)
        domains: list[str] = []
        if any(tag.endswith("_synt") for tag in split_tags):
            domains.append("synt")
        if any(tag.endswith("_real") for tag in split_tags):
            domains.append("real")
        if any(tag == "train_unsup" for tag in split_tags):
            domains.append("unsup")

        record = {
            "record_id": f"dataset2/{sample_id}",
            "source_dataset": "dataset2",
            "category": sample_id.split("_", 1)[0],
            "raw_sample_id": sample_id,
            "group_id": base_id,
            "view_variant": view_variant,
            "domain_membership": domains,
            "split_membership": {tag: True for tag in split_tags},
            "modalities": {
                "instruction": str(instruction_index[sample_id].relative_to(root)).replace("\\", "/"),
                "pattern_viz": str(pattern_viz_index[sample_id].relative_to(root)).replace("\\", "/") if sample_id in pattern_viz_index else None,
                "rendering": str(rendering_index[sample_id].relative_to(root)).replace("\\", "/") if sample_id in rendering_index else None,
                "real_best_rgb": str(best_rgb_index[sample_id].relative_to(root)).replace("\\", "/") if sample_id in best_rgb_index else None,
                "real_best_points": str(best_points_index[sample_id].relative_to(root)).replace("\\", "/") if sample_id in best_points_index else None,
                "real_best_npoints": str(best_npoints_index[sample_id].relative_to(root)).replace("\\", "/") if sample_id in best_npoints_index else None,
                "real_160_rgb": str(rgb160_index[sample_id].relative_to(root)).replace("\\", "/") if sample_id in rgb160_index else None,
                "real_160_gray": str(gray160_index[sample_id].relative_to(root)).replace("\\", "/") if sample_id in gray160_index else None,
            },
            "label_capabilities": ["manifest_driven_multimodal_record"],
            "quality_flags": [],
            "provenance": {
                "source_root": "dataset2",
                "group_id": base_id,
            },
        }
        if sample_id not in rendering_index:
            record["quality_flags"].append("missing_rendering")
            exclusions.append({"record_id": record["record_id"], "reason": "missing_rendering"})
        if sample_id not in pattern_viz_index:
            record["quality_flags"].append("missing_pattern_viz")
            exclusions.append({"record_id": record["record_id"], "reason": "missing_pattern_viz"})
        if "real" in domains and sample_id not in rgb160_index:
            record["quality_flags"].append("missing_real_160_rgb")
            exclusions.append({"record_id": record["record_id"], "reason": "missing_real_160_rgb"})
        records.append(record)

    for syntax_file in sorted((dataset2_root / "syntax").glob("*.txt")):
        syntax_rows.append(
            {
                "record_id": f"dataset2/syntax/{syntax_file.stem}",
                "source_dataset": "dataset2",
                "path": str(syntax_file.relative_to(root)).replace("\\", "/"),
                "quarantine_reason": "corpus_level_syntax_table_not_sample_level_label",
            }
        )

    transfer_dir = dataset2_root / "transfer"
    for path in sorted(transfer_dir.rglob("*")):
        if path.is_file():
            transfer_rows.append(
                {
                    "record_id": f"dataset2/transfer/{path.relative_to(transfer_dir).as_posix()}",
                    "source_dataset": "dataset2",
                    "path": str(path.relative_to(root)).replace("\\", "/"),
                    "quarantine_reason": "transfer_assets_not_promoted_to_core_manifest",
                }
            )

    return records, unassigned, exclusions, syntax_rows, transfer_rows, split_memberships


def build_merged_dataset(root: Path) -> dict[str, object]:
    merged_root = root / "merged_dataset"
    manifests_dir = merged_root / "manifests"
    splits_dir = merged_root / "splits"
    legends_dir = merged_root / "legends"
    assets_dir = merged_root / "assets"
    for directory in (manifests_dir, splits_dir, legends_dir, assets_dir / "dataset", assets_dir / "dataset2"):
        directory.mkdir(parents=True, exist_ok=True)

    dataset_records = _dataset_records(root)
    dataset2_records, unassigned, exclusions, syntax_rows, transfer_rows, split_memberships = _dataset2_records(root)
    merged_records = dataset_records + dataset2_records

    _write_json(merged_root / "schema_version.json", {"schema": "merged_dataset_v1", "style": "metadata_first_union"})
    (merged_root / "README.md").write_text(
        "Merged dataset built as a metadata-first union of dataset/ and dataset2/.\n"
        "dataset/ remains the canonical legend-backed topology source. dataset2/ is included as multimodal auxiliary data.\n",
        encoding="utf-8",
    )
    _write_jsonl(manifests_dir / "merged_samples.jsonl", merged_records)
    _write_jsonl(manifests_dir / "exclusions.jsonl", exclusions)
    _write_jsonl(manifests_dir / "auxiliary_syntax.jsonl", syntax_rows)
    _write_jsonl(manifests_dir / "auxiliary_transfer.jsonl", transfer_rows)
    _write_json(
        manifests_dir / "source_stats.json",
        {
            "dataset_records": len(dataset_records),
            "dataset2_records": len(dataset2_records),
            "merged_records": len(merged_records),
            "dataset2_unassigned": len(unassigned),
            "dataset2_exclusions": len(exclusions),
            "syntax_rows": len(syntax_rows),
            "transfer_rows": len(transfer_rows),
        },
    )

    split_names = [
        "train_real",
        "val_real",
        "test_real",
        "train_synt",
        "val_synt",
        "test_synt",
        "train_unsup",
    ]
    for split_name in split_names:
        split_record_ids = [f"dataset2/{sample_id}" for sample_id, tags in split_memberships.items() if split_name in tags]
        _write_lines(splits_dir / f"dataset2_{split_name}.txt", split_record_ids)
    _write_lines(splits_dir / "dataset2_unassigned.txt", [f"dataset2/{sample_id}" for sample_id in unassigned])
    _write_lines(splits_dir / "dataset_records_unsplit.txt", [record["record_id"] for record in dataset_records])

    legend_src = root / "dataset" / "all_info.json"
    if legend_src.exists():
        (legends_dir / "dataset_all_info.json").write_text(legend_src.read_text(encoding="utf-8"), encoding="utf-8")

    return {
        "merged_root": str(merged_root),
        "dataset_records": len(dataset_records),
        "dataset2_records": len(dataset2_records),
        "merged_records": len(merged_records),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    summary = build_merged_dataset(root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
