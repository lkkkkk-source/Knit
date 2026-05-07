from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build dataset2 <input modality> -> instruction manifests.")
    parser.add_argument("--dataset2-root", type=Path, required=True, help="Path to dataset2")
    parser.add_argument(
        "--input-modality",
        type=str,
        choices=("pattern-viz", "rendering"),
        default="pattern-viz",
        help="Input modality to pair with instruction targets.",
    )
    parser.add_argument("--train-split", type=Path, default=None, help="Optional override for train_synt.txt")
    parser.add_argument("--val-split", type=Path, default=None, help="Optional override for val_synt.txt")
    parser.add_argument("--train-output", type=Path, required=True, help="Output JSONL manifest for train")
    parser.add_argument("--val-output", type=Path, required=True, help="Output JSONL manifest for val")
    return parser


def _read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _build_rows(dataset2_root: Path, sample_ids: list[str], split_name: str, input_modality: str) -> tuple[list[dict[str, object]], list[str]]:
    if input_modality == "pattern-viz":
        input_root = dataset2_root / "pattern-viz" / "400x400"
        suffix = ".png"
    elif input_modality == "rendering":
        input_root = dataset2_root / "rendering"
        suffix = ".jpg"
    else:
        raise ValueError(f"Unsupported input modality: {input_modality}")

    instruction_root = dataset2_root / "instruction"
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    manifest_root = dataset2_root.parent.resolve()

    for sample_id in sample_ids:
        input_path = input_root / f"{sample_id}{suffix}"
        target_path = instruction_root / f"{sample_id}.png"
        if not input_path.exists() or not target_path.exists():
            missing.append(sample_id)
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "split": split_name,
                "input_path": str(input_path.resolve().relative_to(manifest_root)).replace("\\", "/"),
                "target_path": str(target_path.resolve().relative_to(manifest_root)).replace("\\", "/"),
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

    train_rows, train_missing = _build_rows(dataset2_root, train_ids, "train_synt", args.input_modality)
    val_rows, val_missing = _build_rows(dataset2_root, val_ids, "val_synt", args.input_modality)

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
