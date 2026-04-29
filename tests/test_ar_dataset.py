from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from typing import cast

from knit_decode.ar_dataset import ArDatasetItem, ArExportDataset, build_ar_dataloader, collate_ar_batch, load_ar_token_map


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _create_export_root(root: Path) -> Path:
    export_root = root / "ar-root"
    _write_json(
        export_root / "ar_vocab.json",
        {
            "actions": [
                {"action_id": 5, "label": "A", "color_hex": "#111111"},
                {"action_id": 95, "label": "B", "color_hex": "#222222"},
                {"action_id": 249, "label": "C", "color_hex": "#333333"},
            ],
            "special_tokens": {
                "ambiguous_id": -1,
                "row_sep_token": -2,
                "eos_token": -3,
                "bos_token": None,
            },
            "duplicate_colors": {},
        },
    )
    _write_text(
        export_root / "ar_manifest.jsonl",
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "Tuck/001_resized",
                        "sample_meta": {"category": "Tuck", "exact_pixel_match_ratio": 1.0},
                        "ar_id_grid_path": "Tuck/001_resized/ar_id_grid.json",
                        "ar_token_sequence_path": "Tuck/001_resized/ar_token_sequence.json",
                        "ar_token_ids_path": "Tuck/001_resized/ar_token_ids.txt",
                        "rows": 2,
                        "columns": 2,
                        "sequence_length": 6,
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "Hem/001_Rib_resized",
                        "sample_meta": {"category": "Hem", "exact_pixel_match_ratio": 1.0},
                        "ar_id_grid_path": "Hem/001_Rib_resized/ar_id_grid.json",
                        "ar_token_sequence_path": "Hem/001_Rib_resized/ar_token_sequence.json",
                        "ar_token_ids_path": "Hem/001_Rib_resized/ar_token_ids.txt",
                        "rows": 1,
                        "columns": 3,
                        "sequence_length": 4,
                    }
                ),
            ]
        )
        + "\n",
    )

    _write_json(
        export_root / "Tuck" / "001_resized" / "ar_id_grid.json",
        {"rows": 2, "columns": 2, "ambiguous_id": -1, "grid": [[249, 95], [5, -1]]},
    )
    _write_json(
        export_root / "Tuck" / "001_resized" / "ar_token_sequence.json",
        {"flatten_order": "row_major", "row_sep_token": -2, "eos_token": -3, "bos_token": None, "sequence": [249, 95, -2, 5, -1, -3]},
    )
    _write_text(export_root / "Tuck" / "001_resized" / "ar_token_ids.txt", "249 95 -2 5 -1 -3\n")

    _write_json(
        export_root / "Hem" / "001_Rib_resized" / "ar_id_grid.json",
        {"rows": 1, "columns": 3, "ambiguous_id": -1, "grid": [[95, 249, 5]]},
    )
    _write_json(
        export_root / "Hem" / "001_Rib_resized" / "ar_token_sequence.json",
        {"flatten_order": "row_major", "row_sep_token": -2, "eos_token": -3, "bos_token": None, "sequence": [95, 249, 5, -3]},
    )
    _write_text(export_root / "Hem" / "001_Rib_resized" / "ar_token_ids.txt", "95 249 5 -3\n")
    return export_root


class FakeTensor:
    def __init__(self, data: object, dtype: object) -> None:
        self.data: object = data
        self.dtype: object = dtype

    @property
    def shape(self) -> tuple[int, ...]:
        def _shape(value: object) -> tuple[int, ...]:
            if isinstance(value, list) and value:
                return (len(value),) + _shape(value[0])
            if isinstance(value, list):
                return (0,)
            return ()

        return _shape(self.data)


class FakeDataLoader:
    def __init__(self, dataset: object, batch_size: int, shuffle: bool, num_workers: int, collate_fn: object) -> None:
        self.dataset: object = dataset
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.num_workers: int = num_workers
        self.collate_fn: object = collate_fn


class ArDatasetTests(unittest.TestCase):
    def test_load_ar_token_map_builds_dense_vocab_from_sparse_and_negative_raw_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            export_root = _create_export_root(Path(temp_dir))
            token_map = load_ar_token_map(export_root)

        self.assertEqual(token_map.pad_token_id, 0)
        self.assertEqual(token_map.vocab_size, 7)
        self.assertEqual(token_map.raw_to_contiguous[-3], 1)
        self.assertEqual(token_map.raw_to_contiguous[-2], 2)
        self.assertEqual(token_map.raw_to_contiguous[-1], 3)
        self.assertEqual(token_map.raw_to_contiguous[5], 4)
        self.assertEqual(token_map.raw_to_contiguous[95], 5)
        self.assertEqual(token_map.raw_to_contiguous[249], 6)

    def test_dataset_reads_existing_exports_and_returns_shifted_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            export_root = _create_export_root(Path(temp_dir))
            dataset = ArExportDataset(export_root)
            sample = dataset[0]

        self.assertEqual(len(dataset), 2)
        self.assertEqual(sample["sample_id"], "Tuck/001_resized")
        self.assertEqual(sample["input_ids"], [6, 5, 2, 4, 3])
        self.assertEqual(sample["target_ids"], [5, 2, 4, 3, 1])
        self.assertEqual(sample["grid_ids"], [[6, 5], [4, 3]])
        self.assertEqual(sample["rows"], 2)
        self.assertEqual(sample["columns"], 2)

    def test_collate_ar_batch_pads_inputs_and_masks_labels(self) -> None:
        fake_torch = mock.Mock()
        fake_torch.long = "long"
        fake_torch.tensor.side_effect = lambda data, dtype=None: FakeTensor(data, dtype)
        batch: list[ArDatasetItem] = [
            {
                "sample_id": "a",
                "input_ids": [6, 5, 2, 4, 3],
                "target_ids": [5, 2, 4, 3, 1],
                "grid_ids": [[6, 5], [4, 3]],
                "length": 5,
                "rows": 2,
                "columns": 2,
                "sample_meta": {"category": "Tuck"},
            },
            {
                "sample_id": "b",
                "input_ids": [5, 6, 4],
                "target_ids": [6, 4, 1],
                "grid_ids": [[5, 6, 4]],
                "length": 3,
                "rows": 1,
                "columns": 3,
                "sample_meta": {"category": "Hem"},
            },
        ]

        with mock.patch("knit_decode.ar_dataset._require_torch", return_value=fake_torch):
            collated = collate_ar_batch(batch, pad_token_id=0, label_pad_id=-100)

        self.assertEqual(cast(FakeTensor, collated["input_ids"]).data, [[6, 5, 2, 4, 3], [5, 6, 4, 0, 0]])
        self.assertEqual(cast(FakeTensor, collated["target_ids"]).data, [[5, 2, 4, 3, 1], [6, 4, 1, -100, -100]])
        self.assertEqual(cast(FakeTensor, collated["attention_mask"]).data, [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        self.assertEqual(cast(FakeTensor, collated["grid_ids"]).data, [[[6, 5, 0], [4, 3, 0]], [[5, 6, 4], [0, 0, 0]]])
        self.assertEqual(cast(FakeTensor, collated["grid_mask"]).data, [[[1, 1, 0], [1, 1, 0]], [[1, 1, 1], [0, 0, 0]]])

    def test_build_ar_dataloader_wraps_dataset_and_collate(self) -> None:
        fake_torch = mock.Mock()
        fake_torch.long = "long"
        fake_torch.tensor.side_effect = lambda data, dtype=None: FakeTensor(data, dtype)
        fake_torch.utils.data.DataLoader = FakeDataLoader

        with tempfile.TemporaryDirectory() as temp_dir:
            export_root = _create_export_root(Path(temp_dir))
            with mock.patch("knit_decode.ar_dataset._require_torch", return_value=fake_torch):
                dataloader = cast(FakeDataLoader, build_ar_dataloader(export_root, batch_size=2, shuffle=True, num_workers=1))

        self.assertIsInstance(dataloader, FakeDataLoader)
        self.assertEqual(dataloader.batch_size, 2)
        self.assertTrue(dataloader.shuffle)
        self.assertEqual(dataloader.num_workers, 1)
        self.assertEqual(len(cast(ArExportDataset, dataloader.dataset)), 2)

    def test_dataset_rejects_manifest_grid_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            export_root = _create_export_root(Path(temp_dir))
            broken_grid = {
                "rows": 2,
                "columns": 2,
                "ambiguous_id": -1,
                "grid": [[249, 95, 5], [5, -1, 249]],
            }
            _write_json(export_root / "Tuck" / "001_resized" / "ar_id_grid.json", broken_grid)
            dataset = ArExportDataset(export_root)

            with self.assertRaisesRegex(ValueError, "column count mismatch"):
                _ = dataset[0]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
