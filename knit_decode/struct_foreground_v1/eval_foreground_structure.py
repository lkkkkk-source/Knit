from __future__ import annotations

import argparse
from pathlib import Path

from .utils import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate foreground canonical candidates.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _ = load_config(args.config)
    raise ValueError(
        "eval_foreground_structure cannot compute supervised foreground metrics from category-only sampled candidates "
        "because candidates.json does not include paired ground-truth references. The previous placeholder 0.0 metrics were removed."
    )


if __name__ == "__main__":
    raise SystemExit(main())
