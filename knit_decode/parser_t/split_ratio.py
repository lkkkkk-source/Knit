from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split a parser manifest by ratio, optionally filtering one category.")
    parser.add_argument("--manifest", type=Path, required=True, help="Source manifest.jsonl")
    parser.add_argument("--train-output", type=Path, required=True, help="Output train manifest")
    parser.add_argument("--val-output", type=Path, required=True, help="Output val manifest")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio, for example 0.2")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--category", type=str, default=None, help="Optional category filter, for example `Tuck`.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    rows: list[dict[str, object]] = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Invalid manifest row: {row!r}")
        category = row.get("category")
        if args.category is not None and category != args.category:
            continue
        rows.append(row)

    if not rows:
        raise ValueError("No samples matched the requested filter.")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError(f"val-ratio must be between 0 and 1, received {args.val_ratio}")

    random.Random(args.seed).shuffle(rows)
    val_count = max(1, int(round(len(rows) * args.val_ratio)))
    val_count = min(val_count, len(rows) - 1)
    train_rows = rows[:-val_count]
    val_rows = rows[-val_count:]

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
                "category": args.category,
                "seed": args.seed,
                "val_ratio": args.val_ratio,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
