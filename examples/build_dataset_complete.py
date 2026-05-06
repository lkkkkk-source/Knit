from __future__ import annotations

import json
from pathlib import Path
import re
import shutil


PRIMARY_CATEGORIES = [
    "Cable1",
    "Cable2",
    "Hem",
    "Links1",
    "Links2",
    "Mesh",
    "Miss",
    "Move1",
    "Move2",
    "Tuck",
]

STITCH_ONLY_CATEGORIES = ["Jacquard", "Jaquard2", "Three-dimensional"]
SIM_ONLY_CATEGORIES = ["Links"]
CATEGORY_ALIASES = {"Cable1": "Cable", "Cable2": "Cable"}
PAIRING_OVERRIDES = {("Miss", "BORDER40"): "BORDDER40"}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _normalize_stem(category: str, stem: str) -> str:
    base = stem[:-8] if stem.lower().endswith("_resized") else stem
    if category == "Hem" and "_" in base:
        base = base.split("_", 1)[0]

    m_numeric = re.match(r"^0*(\d+)([A-Za-z]*)$", base)
    if m_numeric:
        return f"{int(m_numeric.group(1))}{m_numeric.group(2).upper()}"

    m_prefixed = re.match(r"^([A-Za-z]+)0*(\d+)([A-Za-z]*)$", base)
    if m_prefixed:
        prefix, number, suffix = m_prefixed.groups()
        return f"{prefix.upper()}{int(number)}{suffix.upper()}"

    return base.upper().replace(" ", "")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_dataset_complete(root: Path) -> dict[str, object]:
    dataset_root = root / "dataset"
    stitch_root = dataset_root / "stitch code patterns"
    sim_root = dataset_root / "simulation images"
    out_root = root / "dataset_complete"

    if out_root.exists():
        shutil.rmtree(out_root)

    (out_root / "stitch code patterns").mkdir(parents=True, exist_ok=True)
    (out_root / "simulation images").mkdir(parents=True, exist_ok=True)
    (out_root / "manifests").mkdir(parents=True, exist_ok=True)
    (out_root / "quarantine" / "stitch code patterns").mkdir(parents=True, exist_ok=True)
    (out_root / "quarantine" / "simulation images").mkdir(parents=True, exist_ok=True)

    paired_rows: list[dict[str, object]] = []
    quarantined_rows: list[dict[str, object]] = []
    counts: dict[str, int] = {}

    for category in PRIMARY_CATEGORIES:
        stitch_dir = stitch_root / category
        sim_dir = sim_root / category
        sim_index: dict[str, Path] = {}
        for sim_path in sim_dir.glob("*.png"):
            key = _normalize_stem(category, sim_path.stem)
            sim_index[key] = sim_path

        category_count = 0
        for stitch_path in stitch_dir.glob("*.png"):
            key = _normalize_stem(category, stitch_path.stem)
            key = PAIRING_OVERRIDES.get((category, key), key)
            sim_path = sim_index.get(key)
            if sim_path is None:
                quarantined_rows.append(
                    {
                        "record_id": f"dataset/{category}/{stitch_path.stem}",
                        "category": category,
                        "stitch_code_pattern": str(stitch_path.relative_to(root)).replace("\\", "/"),
                        "reason": "missing_simulation_pair",
                    }
                )
                continue
            stitch_dst = out_root / "stitch code patterns" / category / stitch_path.name
            sim_dst = out_root / "simulation images" / category / sim_path.name
            _copy_file(stitch_path, stitch_dst)
            _copy_file(sim_path, sim_dst)
            paired_rows.append(
                {
                    "record_id": f"dataset/{category}/{stitch_path.stem}",
                    "category": category,
                    "category_original": category,
                    "super_category": CATEGORY_ALIASES.get(category),
                    "stitch_code_pattern": str(stitch_dst.relative_to(out_root)).replace("\\", "/"),
                    "simulation_image": str(sim_dst.relative_to(out_root)).replace("\\", "/"),
                    "pairing_key": key,
                    "pairing_method": "override" if (category, _normalize_stem(category, stitch_path.stem)) in PAIRING_OVERRIDES else "normalized",
                    "source_stitch_path": str(stitch_path.relative_to(root)).replace("\\", "/"),
                    "source_simulation_path": str(sim_path.relative_to(root)).replace("\\", "/"),
                }
            )
            category_count += 1
        counts[category] = category_count

    for category in STITCH_ONLY_CATEGORIES:
        src = stitch_root / category
        dst = out_root / "quarantine" / "stitch code patterns" / category
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            quarantined_rows.append({"category": category, "source": str(src.relative_to(root)).replace("\\", "/"), "reason": "stitch_only_category"})

    for category in SIM_ONLY_CATEGORIES:
        src = sim_root / category
        dst = out_root / "quarantine" / "simulation images" / category
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            quarantined_rows.append({"category": category, "source": str(src.relative_to(root)).replace("\\", "/"), "reason": "simulation_only_category"})

    _copy_file(dataset_root / "all_info.json", out_root / "all_info.json")

    _write_jsonl(out_root / "manifests" / "samples.jsonl", paired_rows)
    _write_jsonl(out_root / "manifests" / "quarantined_records.jsonl", quarantined_rows)
    _write_json(out_root / "manifests" / "pairing_overrides.json", {f"{k[0]}:{k[1]}": v for k, v in PAIRING_OVERRIDES.items()})
    _write_json(out_root / "manifests" / "category_aliases.json", CATEGORY_ALIASES)

    raw_stitch_counts = {category: len(list((stitch_root / category).glob("*.png"))) for category in PRIMARY_CATEGORIES + STITCH_ONLY_CATEGORIES}
    raw_sim_counts = {category: len(list((sim_root / category).glob("*.png"))) for category in PRIMARY_CATEGORIES + SIM_ONLY_CATEGORIES if (sim_root / category).exists()}
    summary = {
        "paired_records": len(paired_rows),
        "paired_by_category": counts,
        "raw_stitch_counts": raw_stitch_counts,
        "raw_simulation_counts": raw_sim_counts,
        "quarantined_records": len(quarantined_rows),
        "category_aliases": CATEGORY_ALIASES,
    }
    _write_json(out_root / "manifests" / "source_inventory.json", {"stitch": raw_stitch_counts, "simulation": raw_sim_counts})
    _write_json(out_root / "manifests" / "summary.json", summary)
    _write_json(out_root / "manifests" / "validation_report.json", {"expected_primary_categories": PRIMARY_CATEGORIES, "paired_records": len(paired_rows), "quarantine_categories": STITCH_ONLY_CATEGORIES + SIM_ONLY_CATEGORIES})

    (out_root / "README.md").write_text(
        "dataset_complete is a cleaned, dataset-only export of data1.\n"
        "It preserves paired simulation/code samples for the primary 10 categories and quarantines non-clean categories.\n"
        "Cable1 and Cable2 remain separate folders, but are grouped by super_category=Cable in metadata.\n",
        encoding="utf-8",
    )

    return summary


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    summary = build_dataset_complete(root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
