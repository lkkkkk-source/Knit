from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from typing import cast

import torch

from knit_decode.pixelcnn_dataset import KnitGridBatchCollator, KnitGridDataset, KnitGridItem, build_knit_grid_dataloader, collate_knit_grid_batch, load_grid_token_map


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _create_export_root(root: Path) -> Path:
    export_root = root / "pixelcnn-root"
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
        json.dumps(
            {
                "sample_id": "Tuck/001_resized",
                "sample_meta": {"category": "Tuck"},
                "ar_id_grid_path": "Tuck/001_resized/ar_id_grid.json",
                "ar_token_sequence_path": "Tuck/001_resized/ar_token_sequence.json",
                "ar_token_ids_path": "Tuck/001_resized/ar_token_ids.txt",
                "rows": 2,
                "columns": 2,
                "sequence_length": 6,
            }
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
    return export_root


class PixelCnnDatasetTests(unittest.TestCase):
    def test_load_grid_token_map_excludes_sequence_only_special_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            export_root = _create_export_root(Path(temp_dir))
            token_map = load_grid_token_map(export_root)

        self.assertEqual(token_map.pad_class_id, 0)
        self.assertEqual(token_map.ambiguous_class_id, 1)
        self.assertEqual(token_map.action_class_ids, (2, 3, 4))
        self.assertEqual(token_map.vocab_size, 5)
        self.assertEqual(token_map.raw_to_contiguous[-1], 1)
        self.assertEqual(token_map.raw_to_contiguous[5], 2)
        self.assertEqual(token_map.raw_to_contiguous[95], 3)
        self.assertEqual(token_map.raw_to_contiguous[249], 4)
        self.assertNotIn(-2, token_map.raw_to_contiguous)
        self.assertNotIn(-3, token_map.raw_to_contiguous)

    def test_dataset_reads_grid_exports_and_encodes_contiguous_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            export_root = _create_export_root(Path(temp_dir))
            dataset = KnitGridDataset(export_root)
            sample = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(sample["sample_id"], "Tuck/001_resized")
        self.assertEqual(sample["class_grid"], [[4, 3], [2, 1]])
        self.assertEqual(sample["rows"], 2)
        self.assertEqual(sample["columns"], 2)

    def test_collate_knit_grid_batch_pads_targets_and_masks(self) -> None:
        batch: list[KnitGridItem] = [
            {"sample_id": "a", "class_grid": [[4, 3], [2, 1]], "rows": 2, "columns": 2, "sample_meta": {"category": "Tuck"}},
            {"sample_id": "b", "class_grid": [[3, 4, 2]], "rows": 1, "columns": 3, "sample_meta": {"category": "Hem"}},
        ]
        collated = collate_knit_grid_batch(batch, pad_class_id=0, grid_vocab_size=5, ignore_index=-100)

        self.assertTrue(torch.equal(cast(torch.Tensor, collated["input_grid"]), torch.tensor([[[[1.0, 0.5, -1.0], [0.0, -0.5, -1.0]]], [[[0.5, 1.0, 0.0], [-1.0, -1.0, -1.0]]]], dtype=torch.float32)))
        self.assertTrue(torch.equal(cast(torch.Tensor, collated["target_grid"]), torch.tensor([[[[4, 3, -100], [2, 1, -100]]], [[[3, 4, 2], [-100, -100, -100]]]], dtype=torch.long)))
        self.assertTrue(torch.equal(cast(torch.Tensor, collated["grid_mask"]), torch.tensor([[[1, 1, 0], [1, 1, 0]], [[1, 1, 1], [0, 0, 0]]], dtype=torch.long)))

    def test_build_knit_grid_dataloader_wraps_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            export_root = _create_export_root(Path(temp_dir))
            dataloader = build_knit_grid_dataloader(export_root, batch_size=2, shuffle=True, num_workers=0)
            batch = next(iter(dataloader))

        self.assertIsInstance(dataloader.collate_fn, KnitGridBatchCollator)
        self.assertEqual(len(cast(KnitGridDataset, dataloader.dataset)), 1)
        self.assertEqual(cast(torch.Tensor, batch["input_grid"]).shape, torch.Size([1, 1, 2, 2]))
        self.assertEqual(cast(torch.Tensor, batch["target_grid"]).shape, torch.Size([1, 1, 2, 2]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
