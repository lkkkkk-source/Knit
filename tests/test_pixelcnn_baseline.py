from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from knit_decode.pixelcnn_baseline import PixelCnnBaselineConfig, grid_vocab_n_bits, run_pixelcnn_baseline


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _create_export_root(root: Path) -> Path:
    export_root = root / "pixelcnn-smoke-root"
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
    manifest = {
        "sample_id": "Tuck/001_resized",
        "sample_meta": {"category": "Tuck"},
        "ar_id_grid_path": "Tuck/001_resized/ar_id_grid.json",
        "ar_token_sequence_path": "Tuck/001_resized/ar_token_sequence.json",
        "ar_token_ids_path": "Tuck/001_resized/ar_token_ids.txt",
        "rows": 2,
        "columns": 2,
        "sequence_length": 6,
    }
    _write_text(export_root / "ar_manifest.jsonl", json.dumps(manifest) + "\n")
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


class PixelCnnBaselineTests(unittest.TestCase):
    def test_grid_vocab_n_bits(self) -> None:
        self.assertEqual(grid_vocab_n_bits(1), 1)
        self.assertEqual(grid_vocab_n_bits(5), 3)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_run_pixelcnn_baseline_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            export_root = _create_export_root(root)
            output_dir = root / "results"
            config = PixelCnnBaselineConfig(
                export_root=str(export_root),
                output_dir=str(output_dir),
                batch_size=1,
                n_epochs=1,
                n_channels=4,
                n_out_conv_channels=8,
                kernel_size=3,
                n_res_layers=1,
                num_workers=0,
                train=True,
                evaluate=True,
                generate=True,
                n_samples=1,
                seed=0,
            )

            summary = run_pixelcnn_baseline(config)
            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            generated = json.loads((output_dir / "generated_grids.json").read_text(encoding="utf-8"))
            self.assertIsNotNone(summary["train_loss"])
            self.assertIsNotNone(summary["eval_loss"])
            self.assertEqual(summary["grid_vocab_size"], 5)
            self.assertTrue((output_dir / "checkpoint.pt").exists())
            self.assertTrue((output_dir / "best_checkpoint.pt").exists())
            self.assertTrue((output_dir / "metrics_history.json").exists())
            self.assertEqual(metrics["grid_vocab_size"], 5)
            self.assertEqual(metrics["epochs_completed"], 1)
            self.assertEqual(metrics["best_epoch"], 1)
            self.assertEqual(metrics["best_metric_name"], "eval_loss")
            history = json.loads((output_dir / "metrics_history.json").read_text(encoding="utf-8"))
            self.assertEqual(len(history), 1)
            self.assertEqual(generated["rows"], 2)
            self.assertEqual(generated["columns"], 2)
            self.assertEqual(len(generated["contiguous_samples"]), 1)
            self.assertEqual(len(generated["raw_action_samples"]), 1)
            self.assertNotIn(None, generated["raw_action_samples"][0][0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
