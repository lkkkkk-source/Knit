from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
from time import perf_counter
from typing import cast

from .ar_export import write_ar_manifest, write_ar_vocab, write_sample_ar_exports
from .discovery import StitchSample, discover_samples
from .image_ops import (
    build_diff_image,
    crop_active_region,
    DecodedGrid,
    decode_grid,
    format_action_grid,
    infer_grid_spec,
    load_rgb_image,
    quantize_to_legend,
    reconstruct_grid,
    render_grid_overlay,
)
from .legend import Legend, load_legend, rgb_to_hex


DEFAULT_CATEGORIES = ["Tuck", "Hem"]
Metrics = dict[str, object]


@dataclass(frozen=True)
class ProcessedSample:
    sample: StitchSample
    metrics: Metrics
    ar_export: Metrics | None = None


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _decoded_grid_to_jsonable(decoded_grid: DecodedGrid) -> list[list[dict[str, object]]]:
    rows: list[list[dict[str, object]]] = []
    for row in decoded_grid.cells:
        rows.append(
            [
                {
                    "row": cell.row,
                    "column": cell.column,
                    "color_hex": cell.color_hex,
                    "action_id": cell.action_id,
                    "candidate_ids": list(cell.candidate_ids),
                    "purity": round(cell.purity, 6),
                }
                for cell in row
            ]
        )
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def process_sample(
    sample: StitchSample,
    legend: Legend,
    output_root: str | Path,
    cell_width: int | None = None,
    cell_height: int | None = None,
    export_ar: bool = False,
    ar_ambiguous_id: int = -1,
    ar_row_sep_token: int = -2,
    ar_eos_token: int = -3,
    ar_bos_token: int | None = None,
) -> ProcessedSample:
    output_root = Path(output_root)
    sample_output_dir = output_root / sample.category / sample.source_path.stem
    _ensure_output_dir(sample_output_dir)

    source_image = load_rgb_image(sample.source_path)
    cropped = crop_active_region(source_image)
    quantized = quantize_to_legend(cropped.image, legend)
    grid_spec = infer_grid_spec(quantized.image, cell_width=cell_width, cell_height=cell_height)
    usable_quantized = quantized.image.crop((0, 0, grid_spec.usable_width, grid_spec.usable_height))
    usable_cropped = cropped.image.crop((0, 0, grid_spec.usable_width, grid_spec.usable_height))
    decoded_grid = decode_grid(usable_quantized, grid_spec, legend)
    reconstruction = reconstruct_grid(decoded_grid)
    overlay = render_grid_overlay(usable_cropped, grid_spec)
    diff_image = build_diff_image(usable_quantized, reconstruction)

    usable_pixels = usable_quantized.width * usable_quantized.height
    ambiguous_cells = [cell for row in decoded_grid.cells for cell in row if cell.action_id is None]
    trimmed_width = cropped.image.width - grid_spec.usable_width
    trimmed_height = cropped.image.height - grid_spec.usable_height
    trimmed_pixel_count = (cropped.image.width * cropped.image.height) - usable_pixels
    warnings: list[str] = []
    if trimmed_pixel_count > 0:
        warnings.append("trimmed_remainder_pixels")
    if decoded_grid.ambiguous_cell_count > 0:
        warnings.append("ambiguous_cells_present")

    usable_cropped.save(sample_output_dir / "cropped_source.png")
    usable_quantized.save(sample_output_dir / "quantized.png")
    overlay.save(sample_output_dir / "grid_overlay.png")
    reconstruction.save(sample_output_dir / "reconstructed.png")
    diff_image.save(sample_output_dir / "diff.png")
    (sample_output_dir / "decoded_grid.tsv").write_text(format_action_grid(decoded_grid) + "\n", encoding="utf-8")
    _write_json(sample_output_dir / "decoded_grid.json", _decoded_grid_to_jsonable(decoded_grid))

    exact_match_pixels = 0
    for y_pos in range(usable_quantized.height):
        for x_pos in range(usable_quantized.width):
            if usable_quantized.getpixel((x_pos, y_pos)) == reconstruction.getpixel((x_pos, y_pos)):
                exact_match_pixels += 1
    metrics: Metrics = {
        "category": sample.category,
        "source_path": str(sample.source_path),
        "source_stem": sample.source_stem,
        "pairing_key": sample.pairing_key,
        "simulation_path": str(sample.simulation_path) if sample.simulation_path else None,
        "source_size": list(source_image.size),
        "crop_box": list(cropped.crop_box),
        "background_color": rgb_to_hex(cropped.background_rgb),
        "grid_rows": grid_spec.rows,
        "grid_columns": grid_spec.columns,
        "cell_width": grid_spec.cell_width,
        "cell_height": grid_spec.cell_height,
        "usable_size": [grid_spec.usable_width, grid_spec.usable_height],
        "trimmed_width": trimmed_width,
        "trimmed_height": trimmed_height,
        "trimmed_pixel_count": trimmed_pixel_count,
        "warnings": warnings,
        "mean_quantization_distance": round(quantized.mean_distance, 6),
        "max_quantization_distance": round(quantized.max_distance, 6),
        "average_cell_purity": round(decoded_grid.average_cell_purity, 6),
        "ambiguous_cell_count": decoded_grid.ambiguous_cell_count,
        "ambiguous_cells": [
            {
                "row": cell.row,
                "column": cell.column,
                "candidate_ids": list(cell.candidate_ids),
                "color_hex": cell.color_hex,
            }
            for cell in ambiguous_cells
        ],
        "exact_pixel_match_ratio": round(exact_match_pixels / float(max(1, usable_pixels)), 6),
    }
    _write_json(sample_output_dir / "metrics.json", metrics)
    ar_export_record: Metrics | None = None
    if export_ar:
        ar_export_record = cast(
            Metrics,
            write_sample_ar_exports(
                sample_output_dir=sample_output_dir,
                decoded_grid=decoded_grid,
                sample_metadata=metrics,
                ambiguous_id=ar_ambiguous_id,
                row_sep_token=ar_row_sep_token,
                eos_token=ar_eos_token,
                bos_token=ar_bos_token,
            ),
        )
    return ProcessedSample(sample=sample, metrics=metrics, ar_export=ar_export_record)


def _empty_run_summary(dataset_root: Path, output_root: Path, categories: list[str], reason: str) -> Metrics:
    return {
        "status": reason,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "categories": categories,
        "legend_duplicate_colors": {},
        "discovered_samples": 0,
        "processed_samples": 0,
        "failed_samples": 0,
        "duration_seconds": 0.0,
        "warnings": [reason],
        "samples": [],
        "by_category": {},
        "ar_exported_samples": 0,
    }


def _write_run_csv(path: Path, sample_metrics: list[Metrics]) -> None:
    fieldnames = [
        "category",
        "source_stem",
        "simulation_path",
        "grid_rows",
        "grid_columns",
        "cell_width",
        "cell_height",
        "average_cell_purity",
        "ambiguous_cell_count",
        "exact_pixel_match_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in sample_metrics:
            writer.writerow({name: metrics.get(name) for name in fieldnames})


def _validate_ar_tokens(
    legend: Legend,
    ambiguous_id: int,
    row_sep_token: int,
    eos_token: int,
    bos_token: int | None,
) -> None:
    special_tokens = [ambiguous_id, row_sep_token, eos_token]
    if bos_token is not None:
        special_tokens.append(bos_token)
    if len(special_tokens) != len(set(special_tokens)):
        raise ValueError("AR special tokens must all be distinct")

    action_ids = {entry.action_id for entry in legend.entries}
    collisions = sorted(token for token in special_tokens if token in action_ids)
    if collisions:
        raise ValueError(f"AR special tokens collide with legend action ids: {collisions}")


def run_pipeline(
    dataset_root: str | Path = "dataset",
    output_root: str | Path = "outputs",
    categories: list[str] | None = None,
    limit: int | None = None,
    cell_width: int | None = None,
    cell_height: int | None = None,
    export_ar: bool = False,
    ar_ambiguous_id: int = -1,
    ar_row_sep_token: int = -2,
    ar_eos_token: int = -3,
    ar_bos_token: int | None = None,
) -> Metrics:
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    selected_categories = list(categories or DEFAULT_CATEGORIES)
    _ensure_output_dir(output_root)

    if not dataset_root.exists():
        summary = _empty_run_summary(dataset_root, output_root, selected_categories, "dataset_root_missing")
        _write_json(output_root / "run_summary.json", summary)
        return summary

    legend_path = dataset_root / "all_info.json"
    stitch_root = dataset_root / "stitch code patterns"
    if not legend_path.exists() or not stitch_root.exists():
        summary = _empty_run_summary(dataset_root, output_root, selected_categories, "dataset_layout_incomplete")
        _write_json(output_root / "run_summary.json", summary)
        return summary

    legend = load_legend(legend_path)
    if export_ar:
        _validate_ar_tokens(
            legend,
            ambiguous_id=ar_ambiguous_id,
            row_sep_token=ar_row_sep_token,
            eos_token=ar_eos_token,
            bos_token=ar_bos_token,
        )
    samples = discover_samples(dataset_root, selected_categories)
    if limit is not None:
        samples = samples[: max(0, limit)]

    start_time = perf_counter()
    processed: list[Metrics] = []
    failures: list[Metrics] = []
    category_metrics: dict[str, list[Metrics]] = defaultdict(list)
    ar_exports: list[Metrics] = []
    for sample in samples:
        try:
            record = process_sample(
                sample=sample,
                legend=legend,
                output_root=output_root,
                cell_width=cell_width,
                cell_height=cell_height,
                export_ar=export_ar,
                ar_ambiguous_id=ar_ambiguous_id,
                ar_row_sep_token=ar_row_sep_token,
                ar_eos_token=ar_eos_token,
                ar_bos_token=ar_bos_token,
            )
            processed.append(record.metrics)
            category_metrics[sample.category].append(record.metrics)
            if record.ar_export is not None:
                ar_exports.append(record.ar_export)
        except Exception as error:  # pragma: no cover - kept for resilient CLI runs.
            failures.append(
                {
                    "category": sample.category,
                    "source_stem": sample.source_stem,
                    "source_path": str(sample.source_path),
                    "error": str(error),
                }
            )

    duration_seconds = round(perf_counter() - start_time, 6)
    by_category: dict[str, Metrics] = {}
    for category, metrics_list in category_metrics.items():
        by_category[category] = {
            "samples": len(metrics_list),
            "ambiguous_cells": sum(cast(int, metrics["ambiguous_cell_count"]) for metrics in metrics_list),
            "mean_cell_purity": round(
                statistics.fmean(cast(float, metrics["average_cell_purity"]) for metrics in metrics_list),
                6,
            ),
            "mean_exact_pixel_match_ratio": round(
                statistics.fmean(cast(float, metrics["exact_pixel_match_ratio"]) for metrics in metrics_list),
                6,
            ),
        }

    summary: Metrics = {
        "status": "ok" if not failures else "completed_with_failures",
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "categories": selected_categories,
        "legend_duplicate_colors": legend.duplicate_colors,
        "discovered_samples": len(samples),
        "processed_samples": len(processed),
        "failed_samples": len(failures),
        "duration_seconds": duration_seconds,
        "warnings": sorted({warning for metrics in processed for warning in cast(list[str], metrics.get("warnings", []))}),
        "failures": failures,
        "samples": processed,
        "by_category": by_category,
        "ar_exported_samples": len(ar_exports),
    }
    _write_json(output_root / "run_summary.json", summary)
    _write_run_csv(output_root / "samples.csv", processed)
    if export_ar:
        write_ar_manifest(output_root / "ar_manifest.jsonl", [cast(dict[str, object], entry) for entry in ar_exports])
        write_ar_vocab(
            output_root / "ar_vocab.json",
            legend,
            ambiguous_id=ar_ambiguous_id,
            row_sep_token=ar_row_sep_token,
            eos_token=ar_eos_token,
            bos_token=ar_bos_token,
        )
    return summary
