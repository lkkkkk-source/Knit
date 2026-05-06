from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import TypedDict, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from .ar_dataset import ArSampleRecord, JsonObject, load_ar_manifest


class KnitGridItem(TypedDict):
    sample_id: str
    class_grid: list[list[int]]
    rows: int
    columns: int
    sample_meta: JsonObject


@dataclass(frozen=True)
class GridTokenMap:
    pad_class_id: int
    pad_raw_token: int | None
    ambiguous_class_id: int | None
    action_class_ids: tuple[int, ...]
    vocab_size: int
    raw_to_contiguous: dict[int, int]
    contiguous_to_raw: list[int | None]

    def encode_token(self, raw_token: int) -> int:
        if raw_token not in self.raw_to_contiguous:
            raise KeyError(f"Raw grid token {raw_token} is not present in the grid token map")
        return self.raw_to_contiguous[raw_token]

    def encode_grid(self, raw_grid: list[list[int]]) -> list[list[int]]:
        return [[self.encode_token(token) for token in row] for row in raw_grid]


@dataclass(frozen=True)
class KnitGridBatchCollator:
    pad_class_id: int
    grid_vocab_size: int
    ignore_index: int = -100

    def __call__(self, batch: Sequence[KnitGridItem]) -> JsonObject:
        return collate_knit_grid_batch(
            batch,
            pad_class_id=self.pad_class_id,
            grid_vocab_size=self.grid_vocab_size,
            ignore_index=self.ignore_index,
        )


def _read_json_object(path: Path) -> JsonObject:
    payload_raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_raw, dict):
        raise ValueError(f"Expected JSON object in {path}, received {type(payload_raw).__name__}")
    return cast(JsonObject, payload_raw)


def _require_int(mapping: JsonObject, key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Expected integer field {key!r}, received {value!r}")
    return value


def _require_grid(path: Path) -> tuple[int, int, list[list[int]]]:
    payload = _read_json_object(path)
    rows = _require_int(payload, "rows")
    columns = _require_int(payload, "columns")
    raw_grid = payload.get("grid")
    if not isinstance(raw_grid, list):
        raise ValueError(f"Expected list grid in {path}")

    grid: list[list[int]] = []
    for row in raw_grid:
        if not isinstance(row, list):
            raise ValueError(f"Expected list row in {path}")
        parsed_row: list[int] = []
        for token in row:
            if not isinstance(token, int):
                raise ValueError(f"Expected integer grid token in {path}, received {token!r}")
            parsed_row.append(token)
        grid.append(parsed_row)

    return rows, columns, grid


def _validate_grid_shape(grid: list[list[int]], rows: int, columns: int, sample_id: str) -> None:
    if len(grid) != rows:
        raise ValueError(f"Grid row count mismatch for {sample_id}: manifest/file rows={rows}, actual={len(grid)}")
    for row in grid:
        if len(row) != columns:
            raise ValueError(f"Grid column count mismatch for {sample_id}: manifest/file cols={columns}, actual_row={len(row)}")


def load_grid_token_map(export_root: str | Path) -> GridTokenMap:
    export_root = Path(export_root)
    payload = _read_json_object(export_root / "ar_vocab.json")
    actions = payload.get("actions")
    special_tokens = payload.get("special_tokens")
    if not isinstance(actions, list) or not isinstance(special_tokens, dict):
        raise ValueError("AR vocab payload is missing actions or special_tokens")

    raw_to_contiguous: dict[int, int] = {}
    contiguous_to_raw: list[int | None] = [None]
    next_class_id = 1

    ambiguous_raw = special_tokens.get("ambiguous_id")
    ambiguous_class_id: int | None = None
    if isinstance(ambiguous_raw, int):
        raw_to_contiguous[ambiguous_raw] = next_class_id
        contiguous_to_raw.append(ambiguous_raw)
        ambiguous_class_id = next_class_id
        next_class_id += 1

    action_ids: list[int] = []
    for entry in actions:
        if not isinstance(entry, dict):
            raise ValueError("Each action entry in ar_vocab.json must be an object")
        action_id = _require_int(cast(JsonObject, entry), "action_id")
        action_ids.append(action_id)
    action_class_ids: list[int] = []
    for action_id in sorted(action_ids):
        if action_id in raw_to_contiguous:
            continue
        raw_to_contiguous[action_id] = next_class_id
        contiguous_to_raw.append(action_id)
        action_class_ids.append(next_class_id)
        next_class_id += 1

    return GridTokenMap(
        pad_class_id=0,
        pad_raw_token=None,
        ambiguous_class_id=ambiguous_class_id,
        action_class_ids=tuple(action_class_ids),
        vocab_size=next_class_id,
        raw_to_contiguous=raw_to_contiguous,
        contiguous_to_raw=contiguous_to_raw,
    )


class KnitGridDataset(Dataset[KnitGridItem]):
    def __init__(self, export_root: str | Path) -> None:
        self.export_root = Path(export_root)
        self.records: list[ArSampleRecord] = load_ar_manifest(self.export_root)
        self.token_map = load_grid_token_map(self.export_root)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> KnitGridItem:
        record = self.records[index]
        grid_rows, grid_columns, raw_grid = _require_grid(self.export_root / record.ar_id_grid_path)
        _validate_grid_shape(raw_grid, grid_rows, grid_columns, record.sample_id)
        _validate_grid_shape(raw_grid, record.rows, record.columns, record.sample_id)
        return {
            "sample_id": record.sample_id,
            "class_grid": self.token_map.encode_grid(raw_grid),
            "rows": record.rows,
            "columns": record.columns,
            "sample_meta": record.sample_meta,
        }


def _normalize_class_grid(class_grid: list[list[int]], grid_vocab_size: int) -> list[list[float]]:
    if grid_vocab_size <= 1:
        return [[0.0 for _ in row] for row in class_grid]
    scale = float(grid_vocab_size - 1)
    return [[(value / scale) * 2.0 - 1.0 for value in row] for row in class_grid]


def _pad_normalized_value(grid_vocab_size: int) -> float:
    return 0.0 if grid_vocab_size <= 1 else -1.0


def collate_knit_grid_batch(
    batch: Sequence[KnitGridItem],
    pad_class_id: int,
    grid_vocab_size: int,
    ignore_index: int = -100,
) -> JsonObject:
    if not batch:
        raise ValueError("Cannot collate an empty knit grid batch")
    _ = pad_class_id

    max_rows = max(sample["rows"] for sample in batch)
    max_columns = max(sample["columns"] for sample in batch)

    padded_inputs: list[list[list[float]]] = []
    padded_targets: list[list[list[int]]] = []
    grid_mask: list[list[list[int]]] = []
    pad_input_value = _pad_normalized_value(grid_vocab_size)

    for sample in batch:
        class_grid = sample["class_grid"]
        rows = sample["rows"]
        columns = sample["columns"]
        normalized = _normalize_class_grid(class_grid, grid_vocab_size)

        padded_input = [row + [pad_input_value] * (max_columns - len(row)) for row in normalized]
        padded_target = [row + [ignore_index] * (max_columns - len(row)) for row in class_grid]
        padded_input.extend([[pad_input_value] * max_columns for _ in range(max_rows - rows)])
        padded_target.extend([[ignore_index] * max_columns for _ in range(max_rows - rows)])
        padded_inputs.append(padded_input)
        padded_targets.append(padded_target)

        mask_rows = [[1] * columns + [0] * (max_columns - columns) for _ in range(rows)]
        mask_rows.extend([[0] * max_columns for _ in range(max_rows - rows)])
        grid_mask.append(mask_rows)

    input_grid: Tensor = torch.tensor(padded_inputs, dtype=torch.float32).unsqueeze(1)
    target_grid: Tensor = torch.tensor(padded_targets, dtype=torch.long).unsqueeze(1)

    return {
        "sample_ids": [sample["sample_id"] for sample in batch],
        "input_grid": input_grid,
        "target_grid": target_grid,
        "grid_mask": torch.tensor(grid_mask, dtype=torch.long),
        "rows": torch.tensor([sample["rows"] for sample in batch], dtype=torch.long),
        "columns": torch.tensor([sample["columns"] for sample in batch], dtype=torch.long),
        "sample_meta": [sample["sample_meta"] for sample in batch],
    }


def build_knit_grid_dataloader(
    export_root: str | Path,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    ignore_index: int = -100,
) -> DataLoader[KnitGridItem]:
    dataset = KnitGridDataset(export_root)
    return build_knit_grid_dataloader_from_dataset(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        ignore_index=ignore_index,
    )


def _dataset_token_map(dataset: Dataset[KnitGridItem]) -> GridTokenMap:
    token_map = getattr(dataset, "token_map", None)
    if token_map is not None:
        return cast(GridTokenMap, token_map)
    parent_dataset = getattr(dataset, "dataset", None)
    if parent_dataset is not None:
        return _dataset_token_map(cast(Dataset[KnitGridItem], parent_dataset))
    raise AttributeError("Dataset does not expose a token_map")


def build_knit_grid_dataloader_from_dataset(
    dataset: Dataset[KnitGridItem],
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    ignore_index: int = -100,
) -> DataLoader[KnitGridItem]:
    token_map = _dataset_token_map(dataset)
    collator = KnitGridBatchCollator(
        pad_class_id=token_map.pad_class_id,
        grid_vocab_size=token_map.vocab_size,
        ignore_index=ignore_index,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collator)


def split_dataset_indices(dataset_size: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if dataset_size < 2:
        raise ValueError("Need at least 2 samples to create a train/validation split")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, received {val_fraction}")

    indices = list(range(dataset_size))
    random.Random(seed).shuffle(indices)
    val_size = max(1, int(round(dataset_size * val_fraction)))
    val_size = min(val_size, dataset_size - 1)
    train_size = dataset_size - val_size
    if train_size < 1 or val_size < 1:
        raise ValueError(
            f"Could not create a non-empty train/validation split: dataset_size={dataset_size}, val_fraction={val_fraction}"
        )
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    return train_indices, val_indices


def subset_knit_grid_dataset(dataset: KnitGridDataset, indices: Sequence[int]) -> Subset[KnitGridItem]:
    return Subset(dataset, list(indices))
