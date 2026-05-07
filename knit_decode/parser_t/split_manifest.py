from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create balanced parser train/val splits grouped by category.")
    parser.add_argument("--manifest", type=Path, required=True, help="Source manifest.jsonl")
    parser.add_argument("--train-output", type=Path, required=True, help="Output train manifest")
    parser.add_argument("--val-output", type=Path, required=True, help="Output val manifest")
    parser.add_argument("--train-per-class", type=int, default=10, help="Max train samples per class")
    parser.add_argument("--val-per-class", type=int, default=2, help="Max val samples per class")
    parser.add_argument("--category", type=str, default=None, help="Optional category filter, for example `Tuck`.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    grouped: dict[str, list[dict[str, object]]] = {}
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Invalid manifest row: {row!r}")
        category = row.get("category")
        if not isinstance(category, str):
            raise ValueError(f"Missing category in manifest row: {row!r}")
        if args.category is not None and category != args.category:
            continue
        grouped.setdefault(category, []).append(row)

    train_rows: list[dict[str, object]] = []
    val_rows: list[dict[str, object]] = []
    for category in sorted(grouped):
        rows = grouped[category]
        if len(rows) <= 1:
            train_rows.extend(rows)
            continue
        val_count = min(args.val_per_class, max(1, len(rows) // 5))
        train_count = min(args.train_per_class, max(1, len(rows) - val_count))
        train_rows.extend(rows[:train_count])
        val_rows.extend(rows[-val_count:])

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.val_output.parent.mkdir(parents=True, exist_ok=True)
    args.train_output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in train_rows) + ("\n" if train_rows else ""),
        encoding="utf-8",
    )
    args.val_output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in val_rows) + ("\n" if val_rows else ""),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "train_manifest": str(args.train_output),
                "val_manifest": str(args.val_output),
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
                "categories": len(grouped),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
