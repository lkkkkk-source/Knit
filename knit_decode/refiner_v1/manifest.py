from __future__ import annotations

import argparse
import json
from pathlib import Path


_VIEW_SUFFIXES = ("_front", "_back", "_black_front", "_black_back")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build dataset2 real-to-rendering paired manifests.")
    parser.add_argument("--dataset2-root", type=Path, required=True, help="Path to dataset2")
    parser.add_argument("--real-subdir", type=str, default="160x160/rgb", help="Subdirectory under dataset2/real to use as input")
    parser.add_argument("--train-split", type=Path, default=None, help="Optional override for train_real.txt")
    parser.add_argument("--val-split", type=Path, default=None, help="Optional override for val_real.txt")
    parser.add_argument("--train-output", type=Path, required=True, help="Output JSONL manifest for train")
    parser.add_argument("--val-output", type=Path, required=True, help="Output JSONL manifest for val")
    return parser


def _read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _strip_view_suffix(sample_id: str) -> str:
    for suffix in _VIEW_SUFFIXES:
        if sample_id.endswith(suffix):
            return sample_id[: -len(suffix)]
    return sample_id


def _build_rows(dataset2_root: Path, sample_ids: list[str], split_name: str, real_subdir: str) -> tuple[list[dict[str, object]], list[str]]:
    real_root = dataset2_root / "real" / real_subdir
    rendering_root = dataset2_root / "rendering"
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    manifest_root = dataset2_root.parent.resolve()
    for sample_id in sample_ids:
        input_path = real_root / f"{sample_id}.jpg"
        rendering_id = _strip_view_suffix(sample_id)
        target_path = rendering_root / f"{rendering_id}.jpg"
        if not input_path.exists() or not target_path.exists():
            missing.append(sample_id)
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "paired_rendering_id": rendering_id,
                "category": rendering_id.split("_", 1)[0],
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
    train_split = args.train_split or (dataset2_root / "train_real.txt")
    val_split = args.val_split or (dataset2_root / "val_real.txt")
    train_ids = _read_split(train_split)
    val_ids = _read_split(val_split)
    train_rows, train_missing = _build_rows(dataset2_root, train_ids, "train_real", args.real_subdir)
    val_rows, val_missing = _build_rows(dataset2_root, val_ids, "val_real", args.real_subdir)
    _write_jsonl(args.train_output, train_rows)
    _write_jsonl(args.val_output, val_rows)
    print(
        json.dumps(
            {
                "train_manifest": str(args.train_output),
                "val_manifest": str(args.val_output),
                "real_subdir": args.real_subdir,
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "train_missing": len(train_missing),
                "val_missing": len(val_missing),
                "train_missing_examples": train_missing[:20],
                "val_missing_examples": val_missing[:20],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
