from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from .pipeline import DEFAULT_CATEGORIES, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode stitch-code images into QA artifacts")
    _ = parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Dataset root containing all_info.json and the stitch/simulation folders.",
    )
    _ = parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="Categories to process. Defaults to Tuck Hem.",
    )
    _ = parser.add_argument("--limit", type=int, default=None, help="Limit the number of discovered samples.")
    _ = parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for per-sample artifacts and run summaries.",
    )
    _ = parser.add_argument("--cell-width", type=int, default=None, help="Override inferred cell width.")
    _ = parser.add_argument("--cell-height", type=int, default=None, help="Override inferred cell height.")
    _ = parser.add_argument("--export-ar", action="store_true", help="Export AR training artifacts next to decoded grids.")
    _ = parser.add_argument("--ar-ambiguous-id", type=int, default=-1, help="Sentinel id for ambiguous cells in AR exports.")
    _ = parser.add_argument("--ar-row-sep-token", type=int, default=-2, help="Row separator token for flattened AR sequences.")
    _ = parser.add_argument("--ar-eos-token", type=int, default=-3, help="EOS token for flattened AR sequences.")
    _ = parser.add_argument("--ar-bos-token", type=int, default=None, help="Optional BOS token for flattened AR sequences.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dataset_root = cast(Path, args.dataset_root)
    output_dir = cast(Path, args.output_dir)
    categories = cast(list[str], args.categories)
    limit = cast(int | None, args.limit)
    cell_width = cast(int | None, args.cell_width)
    cell_height = cast(int | None, args.cell_height)
    export_ar = cast(bool, args.export_ar)
    ar_ambiguous_id = cast(int, args.ar_ambiguous_id)
    ar_row_sep_token = cast(int, args.ar_row_sep_token)
    ar_eos_token = cast(int, args.ar_eos_token)
    ar_bos_token = cast(int | None, args.ar_bos_token)
    summary = run_pipeline(
        dataset_root=dataset_root,
        output_root=output_dir,
        categories=categories,
        limit=limit,
        cell_width=cell_width,
        cell_height=cell_height,
        export_ar=export_ar,
        ar_ambiguous_id=ar_ambiguous_id,
        ar_row_sep_token=ar_row_sep_token,
        ar_eos_token=ar_eos_token,
        ar_bos_token=ar_bos_token,
    )

    print(f"status: {summary['status']}")
    print(f"processed samples: {summary['processed_samples']} / {summary['discovered_samples']}")
    print(f"run summary: {output_dir / 'run_summary.json'}")
    return 0 if summary["status"] in {"ok", "dataset_root_missing", "dataset_layout_incomplete"} else 1
