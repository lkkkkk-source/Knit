from __future__ import annotations

import json
from pathlib import Path
import shutil


def build_paired_dataset(root: Path) -> dict[str, object]:
    merged_manifest_path = root / "merged_dataset" / "manifests" / "merged_samples.jsonl"
    dataset_root = root / "dataset"
    output_root = root / "dataset_paired"

    if output_root.exists():
        shutil.rmtree(output_root)

    simulation_out = output_root / "simulation images"
    code_out = output_root / "stitch code patterns"
    simulation_out.mkdir(parents=True, exist_ok=True)
    code_out.mkdir(parents=True, exist_ok=True)

    paired_records: list[dict[str, object]] = []
    for line in merged_manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("source_dataset") != "dataset":
            continue
        modalities = row.get("modalities", {})
        stitch_code = modalities.get("stitch_code_pattern")
        simulation = modalities.get("simulation_image")
        if not stitch_code or not simulation:
            continue

        stitch_src = root / Path(str(stitch_code))
        simulation_src = root / Path(str(simulation))
        stitch_rel = stitch_src.relative_to(dataset_root / "stitch code patterns")
        simulation_rel = simulation_src.relative_to(dataset_root / "simulation images")

        stitch_dst = code_out / stitch_rel
        simulation_dst = simulation_out / simulation_rel
        stitch_dst.parent.mkdir(parents=True, exist_ok=True)
        simulation_dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(stitch_src, stitch_dst)
        shutil.copy2(simulation_src, simulation_dst)
        paired_records.append(
            {
                "record_id": row["record_id"],
                "category": row["category"],
                "stitch_code_pattern": str(stitch_dst.relative_to(output_root)).replace("\\", "/"),
                "simulation_image": str(simulation_dst.relative_to(output_root)).replace("\\", "/"),
            }
        )

    legend_src = dataset_root / "all_info.json"
    shutil.copy2(legend_src, output_root / "all_info.json")

    category_counts: dict[str, int] = {}
    for record in paired_records:
        category = str(record["category"])
        category_counts[category] = category_counts.get(category, 0) + 1

    summary = {
        "output_root": str(output_root),
        "paired_records": len(paired_records),
        "categories": dict(sorted(category_counts.items())),
    }
    (output_root / "paired_manifest.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in paired_records) + ("\n" if paired_records else ""),
        encoding="utf-8",
    )
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    summary = build_paired_dataset(root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
