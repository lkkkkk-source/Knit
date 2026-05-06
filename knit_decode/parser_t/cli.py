from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import build_parser_manifest_from_dataset_complete


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utilities for the simulation-image to topology parser T.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_manifest = subparsers.add_parser("build-manifest", help="Build a starter manifest from dataset_complete.")
    build_manifest.add_argument("--dataset-root", type=Path, required=True)
    build_manifest.add_argument("--output-path", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build-manifest":
        output_path = build_parser_manifest_from_dataset_complete(args.dataset_root, args.output_path)
        print(json.dumps({"manifest_path": str(output_path)}, indent=2, ensure_ascii=False))
        return 0

    raise ValueError(f"Unsupported command: {args.command}")

