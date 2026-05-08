from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build dataset2 manifests for inverse-style rendering->instruction training.")
    parser.add_argument("--dataset2-root", type=Path, required=True, help="Path to dataset2")
    parser.add_argument("--input-modality", type=str, choices=("pattern-viz", "rendering", "real-gray"), default="rendering")
    parser.add_argument("--train-split", type=Path, default=None, help="Optional override split file")
    parser.add_argument("--val-split", type=Path, default=None, help="Optional override split file")
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    return parser


def _read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_input(dataset2_root: Path, sample_id: str, modality: str) -> Path:
    if modality == "pattern-viz":
        return dataset2_root / "pattern-viz" / "400x400" / f"{sample_id}.png"
    if modality == "rendering":
        return dataset2_root / "rendering" / f"{sample_id}.jpg"
    if modality == "real-gray":
        return dataset2_root / "real" / "160x160" / "gray" / f"{sample_id}.jpg"
    raise ValueError(f"Unsupported modality: {modality}")


def _build_rows(dataset2_root: Path, sample_ids: list[str], split_name: str, modality: str) -> tuple[list[dict[str, object]], list[str]]:
    manifest_root = dataset2_root.parent.resolve()
    instruction_root = dataset2_root / "instruction"
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for sample_id in sample_ids:
        input_path = _resolve_input(dataset2_root, sample_id, modality)
        target_path = instruction_root / f"{sample_id}.png"
        index_path = instruction_root / "index" / f"{sample_id}.txt"
        if not input_path.exists() or not target_path.exists() or not index_path.exists():
            missing.append(sample_id)
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "category": sample_id.split("_", 1)[0],
                "split": split_name,
                "input_path": str(input_path.resolve().relative_to(manifest_root)).replace("\\", "/"),
                "target_path": str(target_path.resolve().relative_to(manifest_root)).replace("\\", "/"),
                "index_path": str(index_path.resolve().relative_to(manifest_root)).replace("\\", "/"),
            }
        )
    return rows, missing


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dataset2_root = args.dataset2_root
    train_split = args.train_split or (dataset2_root / "train_synt.txt")
    val_split = args.val_split or (dataset2_root / "val_synt.txt")
    train_ids = _read_split(train_split)
    val_ids = _read_split(val_split)
    train_rows, train_missing = _build_rows(dataset2_root, train_ids, train_split.stem, args.input_modality)
    val_rows, val_missing = _build_rows(dataset2_root, val_ids, val_split.stem, args.input_modality)
    _write_jsonl(args.train_output, train_rows)
    _write_jsonl(args.val_output, val_rows)
    print(
        json.dumps(
            {
                "train_manifest": str(args.train_output),
                "val_manifest": str(args.val_output),
                "input_modality": args.input_modality,
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "train_missing": len(train_missing),
                "val_missing": len(val_missing),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
