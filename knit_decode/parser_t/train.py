from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a first-pass simulation-image to topology parser.")
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL manifest mapping simulation images to stitch-code color-map targets.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-shift", type=int, default=1, help="Global shift tolerance used by the Inverse-Knitting-style CE.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"manifest: {args.manifest}")
    print(f"output_dir: {args.output_dir}")
    print(f"epochs: {args.epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"max_shift: {args.max_shift}")
    print("Training loop scaffold created for simulation-image to stitch-code color-map supervision.")
    print("Next step: wire palette indexing, tensor collation, and torch optimization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
