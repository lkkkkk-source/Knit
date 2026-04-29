from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import TypedDict, cast


JsonObject = dict[str, object]


class ArManifestEntry(TypedDict):
    sample_id: str
    sample_meta: JsonObject
    ar_id_grid_path: str
    ar_token_sequence_path: str
    ar_token_ids_path: str
    rows: int
    columns: int
    sequence_length: int


class ArDatasetItem(TypedDict):
    sample_id: str
    input_ids: list[int]
    target_ids: list[int]
    grid_ids: list[list[int]]
    length: int
    rows: int
    columns: int
    sample_meta: JsonObject


@dataclass(frozen=True)
class ArTokenMap:
    pad_token_id: int
    vocab_size: int
    raw_to_contiguous: dict[int, int]
    contiguous_to_raw: list[int | None]
    special_token_ids: dict[str, int | None]

    def encode_token(self, raw_token: int) -> int:
        if raw_token not in self.raw_to_contiguous:
            raise KeyError(f"Raw token {raw_token} is not present in the AR token map")
        return self.raw_to_contiguous[raw_token]

    def encode_sequence(self, raw_tokens: list[int]) -> list[int]:
        return [self.encode_token(token) for token in raw_tokens]

    def encode_grid(self, raw_grid: list[list[int]]) -> list[list[int]]:
        return [self.encode_sequence(row) for row in raw_grid]


@dataclass(frozen=True)
class ArSampleRecord:
    sample_id: str
    sample_meta: JsonObject
    ar_id_grid_path: str
    ar_token_sequence_path: str
    ar_token_ids_path: str
    rows: int
    columns: int
    sequence_length: int


@dataclass(frozen=True)
class ArBatchCollator:
    pad_token_id: int
    label_pad_id: int = -100

    def __call__(self, batch: Sequence[ArDatasetItem]) -> JsonObject:
        return collate_ar_batch(batch, pad_token_id=self.pad_token_id, label_pad_id=self.label_pad_id)


def _read_json(path: Path) -> object:
    payload_raw: object = json.loads(path.read_text(encoding="utf-8"))
    return payload_raw


def _read_json_object(path: Path) -> JsonObject:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, received {type(payload).__name__}")
    return cast(JsonObject, payload)


def _require_str(mapping: JsonObject, key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Expected string field {key!r}, received {value!r}")
    return value


def _require_int(mapping: JsonObject, key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Expected integer field {key!r}, received {value!r}")
    return value


def _parse_manifest_entry(raw_record: JsonObject) -> ArManifestEntry:
    sample_meta = raw_record.get("sample_meta", {})
    if not isinstance(sample_meta, dict):
        raise ValueError("Manifest sample_meta must be a JSON object")
    return {
        "sample_id": _require_str(raw_record, "sample_id"),
        "sample_meta": cast(JsonObject, sample_meta),
        "ar_id_grid_path": _require_str(raw_record, "ar_id_grid_path"),
        "ar_token_sequence_path": _require_str(raw_record, "ar_token_sequence_path"),
        "ar_token_ids_path": _require_str(raw_record, "ar_token_ids_path"),
        "rows": _require_int(raw_record, "rows"),
        "columns": _require_int(raw_record, "columns"),
        "sequence_length": _require_int(raw_record, "sequence_length"),
    }


def load_ar_manifest(export_root: str | Path) -> list[ArSampleRecord]:
    export_root = Path(export_root)
    manifest_path = export_root / "ar_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"AR manifest not found: {manifest_path}")

    records: list[ArSampleRecord] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw_record = json.loads(line)
        if not isinstance(raw_record, dict):
            raise ValueError("Each manifest line must be a JSON object")
        parsed = _parse_manifest_entry(cast(JsonObject, raw_record))
        records.append(
            ArSampleRecord(
                sample_id=parsed["sample_id"],
                sample_meta=parsed["sample_meta"],
                ar_id_grid_path=parsed["ar_id_grid_path"],
                ar_token_sequence_path=parsed["ar_token_sequence_path"],
                ar_token_ids_path=parsed["ar_token_ids_path"],
                rows=parsed["rows"],
                columns=parsed["columns"],
                sequence_length=parsed["sequence_length"],
            )
        )
    return records


def load_ar_token_map(export_root: str | Path) -> ArTokenMap:
    export_root = Path(export_root)
    vocab_path = export_root / "ar_vocab.json"
    payload = _read_json_object(vocab_path)

    actions = payload.get("actions")
    special_tokens = payload.get("special_tokens")
    if not isinstance(actions, list) or not isinstance(special_tokens, dict):
        raise ValueError("AR vocab payload is missing actions or special_tokens")
    special_tokens = cast(JsonObject, special_tokens)

    raw_to_contiguous: dict[int, int] = {}
    contiguous_to_raw: list[int | None] = [None]
    special_token_ids: dict[str, int | None] = {
        "pad_token": 0,
        "bos_token": None,
        "eos_token": None,
        "row_sep_token": None,
        "ambiguous_id": None,
    }

    next_token_id = 1
    for key in ("bos_token", "eos_token", "row_sep_token", "ambiguous_id"):
        raw_value = special_tokens.get(key)
        if raw_value is None:
            continue
        if not isinstance(raw_value, int):
            raise ValueError(f"Special token {key!r} must be an integer or null, received {raw_value!r}")
        raw_token = raw_value
        raw_to_contiguous[raw_token] = next_token_id
        contiguous_to_raw.append(raw_token)
        special_token_ids[key] = next_token_id
        next_token_id += 1

    action_ids: list[int] = []
    for entry in actions:
        if not isinstance(entry, dict):
            raise ValueError("Each action entry in ar_vocab.json must be an object")
        action_ids.append(_require_int(cast(JsonObject, entry), "action_id"))
    action_ids.sort()
    for action_id in action_ids:
        if action_id in raw_to_contiguous:
            continue
        raw_to_contiguous[action_id] = next_token_id
        contiguous_to_raw.append(action_id)
        next_token_id += 1

    return ArTokenMap(
        pad_token_id=0,
        vocab_size=next_token_id,
        raw_to_contiguous=raw_to_contiguous,
        contiguous_to_raw=contiguous_to_raw,
        special_token_ids=special_token_ids,
    )


def load_raw_token_sequence(path: str | Path) -> list[int]:
    payload = _read_json_object(Path(path))
    sequence = payload.get("sequence")
    if not isinstance(sequence, list):
        raise ValueError(f"AR token sequence is missing a list at {path}")
    parsed_tokens: list[int] = []
    for token in sequence:
        if not isinstance(token, int):
            raise ValueError(f"AR token sequence must contain integers at {path}, received {token!r}")
        parsed_tokens.append(token)
    return parsed_tokens


def load_raw_id_grid(path: str | Path) -> list[list[int]]:
    payload = _read_json_object(Path(path))
    grid = payload.get("grid")
    if not isinstance(grid, list):
        raise ValueError(f"AR id grid is missing a list at {path}")
    rows: list[list[int]] = []
    for row in grid:
        if not isinstance(row, list):
            raise ValueError(f"AR id grid rows must be lists at {path}")
        parsed_row: list[int] = []
        for token in row:
            if not isinstance(token, int):
                raise ValueError(f"AR id grid must contain integers at {path}, received {token!r}")
            parsed_row.append(token)
        rows.append(parsed_row)
    return rows


def _validate_grid_shape(grid: list[list[int]], rows: int, columns: int, sample_id: str) -> None:
    if len(grid) != rows:
        raise ValueError(f"Grid row count mismatch for {sample_id}: manifest={rows}, file={len(grid)}")
    for row in grid:
        if len(row) != columns:
            raise ValueError(f"Grid column count mismatch for {sample_id}: manifest={columns}, file row={len(row)}")


class ArExportDataset:
    def __init__(self, export_root: str | Path) -> None:
        self.export_root: Path = Path(export_root)
        self.records: list[ArSampleRecord] = load_ar_manifest(self.export_root)
        self.token_map: ArTokenMap = load_ar_token_map(self.export_root)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ArDatasetItem:
        record = self.records[index]
        raw_sequence = load_raw_token_sequence(self.export_root / record.ar_token_sequence_path)
        raw_grid = load_raw_id_grid(self.export_root / record.ar_id_grid_path)
        _validate_grid_shape(raw_grid, record.rows, record.columns, record.sample_id)
        remapped_sequence = self.token_map.encode_sequence(raw_sequence)
        remapped_grid = self.token_map.encode_grid(raw_grid)
        if len(remapped_sequence) < 2:
            raise ValueError(f"AR sequence for {record.sample_id} is too short for autoregressive training")

        return {
            "sample_id": record.sample_id,
            "input_ids": remapped_sequence[:-1],
            "target_ids": remapped_sequence[1:],
            "grid_ids": remapped_grid,
            "length": len(remapped_sequence) - 1,
            "rows": record.rows,
            "columns": record.columns,
            "sample_meta": record.sample_meta,
        }


def _require_torch() -> object:
    try:
        return importlib.import_module("torch")
    except ImportError as error:  # pragma: no cover - depends on external install.
        raise ImportError(
            "PyTorch is required for batching/loading. Install it with `pip install -e .[train]` or install torch manually."
        ) from error


def collate_ar_batch(
    batch: Sequence[ArDatasetItem],
    pad_token_id: int,
    label_pad_id: int = -100,
) -> JsonObject:
    if not batch:
        raise ValueError("Cannot collate an empty AR batch")

    torch = _require_torch()
    tensor = cast(Callable[..., object], getattr(torch, "tensor"))
    long_dtype = getattr(torch, "long")
    lengths = [sample["length"] for sample in batch]
    max_length = max(lengths)
    max_rows = max(sample["rows"] for sample in batch)
    max_columns = max(sample["columns"] for sample in batch)

    padded_inputs: list[list[int]] = []
    padded_targets: list[list[int]] = []
    attention_mask: list[list[int]] = []
    padded_grids: list[list[list[int]]] = []
    grid_mask: list[list[list[int]]] = []

    for sample in batch:
        input_ids = cast(list[int], sample["input_ids"])
        target_ids = cast(list[int], sample["target_ids"])
        grid_ids = cast(list[list[int]], sample["grid_ids"])
        sample_rows = sample["rows"]
        sample_columns = sample["columns"]
        sample_length = len(input_ids)

        padded_inputs.append(input_ids + [pad_token_id] * (max_length - sample_length))
        padded_targets.append(target_ids + [label_pad_id] * (max_length - sample_length))
        attention_mask.append([1] * sample_length + [0] * (max_length - sample_length))

        padded_grid = [row + [pad_token_id] * (max_columns - len(row)) for row in grid_ids]
        padded_grid.extend([[pad_token_id] * max_columns for _ in range(max_rows - sample_rows)])
        padded_grids.append(padded_grid)

        active_grid_mask = [[1] * sample_columns for _ in range(sample_rows)]
        active_grid_mask = [row + [0] * (max_columns - len(row)) for row in active_grid_mask]
        active_grid_mask.extend([[0] * max_columns for _ in range(max_rows - sample_rows)])
        grid_mask.append(active_grid_mask)

    return {
        "sample_ids": [str(sample["sample_id"]) for sample in batch],
        "input_ids": tensor(padded_inputs, dtype=long_dtype),
        "target_ids": tensor(padded_targets, dtype=long_dtype),
        "attention_mask": tensor(attention_mask, dtype=long_dtype),
        "grid_ids": tensor(padded_grids, dtype=long_dtype),
        "grid_mask": tensor(grid_mask, dtype=long_dtype),
        "lengths": tensor(lengths, dtype=long_dtype),
        "rows": tensor([sample["rows"] for sample in batch], dtype=long_dtype),
        "columns": tensor([sample["columns"] for sample in batch], dtype=long_dtype),
        "sample_meta": [cast(JsonObject, sample["sample_meta"]) for sample in batch],
    }


def build_ar_dataloader(
    export_root: str | Path,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    label_pad_id: int = -100,
) -> object:
    torch = _require_torch()
    utils = getattr(torch, "utils")
    data_module = getattr(utils, "data")
    data_loader = cast(Callable[..., object], getattr(data_module, "DataLoader"))
    dataset = ArExportDataset(export_root)
    collator = ArBatchCollator(pad_token_id=dataset.token_map.pad_token_id, label_pad_id=label_pad_id)

    return data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )
