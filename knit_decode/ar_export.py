from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from .image_ops import DecodedGrid
from .legend import Legend


def build_id_grid(decoded_grid: DecodedGrid, ambiguous_id: int = -1) -> list[list[int]]:
    grid: list[list[int]] = []
    for row in decoded_grid.cells:
        grid.append([
            cell.action_id if cell.action_id is not None else ambiguous_id
            for cell in row
        ])
    return grid


def build_token_sequence(
    id_grid: list[list[int]],
    row_sep_token: int = -2,
    eos_token: int = -3,
    bos_token: int | None = None,
) -> list[int]:
    tokens: list[int] = []
    if bos_token is not None:
        tokens.append(bos_token)
    for row_index, row in enumerate(id_grid):
        tokens.extend(row)
        if row_index < len(id_grid) - 1:
            tokens.append(row_sep_token)
    tokens.append(eos_token)
    return tokens


def write_sample_ar_exports(
    sample_output_dir: Path,
    decoded_grid: DecodedGrid,
    sample_metadata: Mapping[str, object],
    ambiguous_id: int = -1,
    row_sep_token: int = -2,
    eos_token: int = -3,
    bos_token: int | None = None,
) -> dict[str, object]:
    id_grid = build_id_grid(decoded_grid, ambiguous_id=ambiguous_id)
    token_sequence = build_token_sequence(
        id_grid,
        row_sep_token=row_sep_token,
        eos_token=eos_token,
        bos_token=bos_token,
    )

    id_grid_payload = {
        "rows": decoded_grid.grid_spec.rows,
        "columns": decoded_grid.grid_spec.columns,
        "ambiguous_id": ambiguous_id,
        "grid": id_grid,
    }
    token_payload = {
        "flatten_order": "row_major",
        "row_sep_token": row_sep_token,
        "eos_token": eos_token,
        "bos_token": bos_token,
        "sequence": token_sequence,
    }

    id_grid_path = sample_output_dir / "ar_id_grid.json"
    token_json_path = sample_output_dir / "ar_token_sequence.json"
    token_txt_path = sample_output_dir / "ar_token_ids.txt"

    id_grid_path.write_text(json.dumps(id_grid_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    token_json_path.write_text(json.dumps(token_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    token_txt_path.write_text(" ".join(str(token) for token in token_sequence) + "\n", encoding="utf-8")

    output_root = sample_output_dir.parents[1]
    return {
        "sample_id": sample_output_dir.relative_to(sample_output_dir.parents[1]).as_posix(),
        "sample_meta": dict(sample_metadata),
        "ar_id_grid_path": id_grid_path.relative_to(output_root).as_posix(),
        "ar_token_sequence_path": token_json_path.relative_to(output_root).as_posix(),
        "ar_token_ids_path": token_txt_path.relative_to(output_root).as_posix(),
        "rows": decoded_grid.grid_spec.rows,
        "columns": decoded_grid.grid_spec.columns,
        "ambiguous_id": ambiguous_id,
        "row_sep_token": row_sep_token,
        "eos_token": eos_token,
        "bos_token": bos_token,
        "sequence_length": len(token_sequence),
    }


def write_ar_manifest(path: Path, entries: list[dict[str, object]]) -> None:
    lines = [json.dumps(entry, ensure_ascii=False) for entry in entries]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_ar_vocab(
    path: Path,
    legend: Legend,
    ambiguous_id: int = -1,
    row_sep_token: int = -2,
    eos_token: int = -3,
    bos_token: int | None = None,
) -> None:
    payload = {
        "actions": [
            {
                "action_id": entry.action_id,
                "label": entry.label,
                "color_hex": entry.color_hex,
            }
            for entry in legend.entries
        ],
        "special_tokens": {
            "ambiguous_id": ambiguous_id,
            "row_sep_token": row_sep_token,
            "eos_token": eos_token,
            "bos_token": bos_token,
        },
        "duplicate_colors": legend.duplicate_colors,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
