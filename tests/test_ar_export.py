from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from knit_decode.ar_export import build_token_sequence
from knit_decode.pipeline import run_pipeline


def _write_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


class ArExportTests(unittest.TestCase):
    def test_build_token_sequence_inserts_row_separator_and_eos(self) -> None:
        tokens = build_token_sequence([[1, 2], [3, 4]], row_sep_token=-2, eos_token=-3)
        self.assertEqual(tokens, [1, 2, -2, 3, 4, -3])

    def test_pipeline_can_export_ar_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            output_root = root / "outputs"

            legend = {
                "A": [{"id": 1, "color": "#FF0000"}],
                "B": [{"id": 2, "color": "#00FF00"}],
            }
            (dataset_root / "all_info.json").parent.mkdir(parents=True, exist_ok=True)
            (dataset_root / "all_info.json").write_text(json.dumps(legend), encoding="utf-8")

            code_image = Image.new("RGB", (4, 4), (255, 255, 255))
            code_image.putpixel((1, 1), (255, 0, 0))
            code_image.putpixel((2, 1), (0, 255, 0))
            code_image.putpixel((1, 2), (0, 255, 0))
            code_image.putpixel((2, 2), (255, 0, 0))
            _write_png(dataset_root / "stitch code patterns" / "Tuck" / "001_resized.png", code_image)
            _write_png(dataset_root / "simulation images" / "Tuck" / "1.png", Image.new("RGB", (4, 4), (0, 0, 0)))

            summary = run_pipeline(
                dataset_root=dataset_root,
                output_root=output_root,
                categories=["Tuck"],
                export_ar=True,
                cell_width=1,
                cell_height=1,
            )

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["ar_exported_samples"], 1)

            sample_dir = output_root / "Tuck" / "001_resized"
            id_grid = json.loads((sample_dir / "ar_id_grid.json").read_text(encoding="utf-8"))
            token_payload = json.loads((sample_dir / "ar_token_sequence.json").read_text(encoding="utf-8"))
            token_text = (sample_dir / "ar_token_ids.txt").read_text(encoding="utf-8").strip()
            manifest_lines = (output_root / "ar_manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
            vocab = json.loads((output_root / "ar_vocab.json").read_text(encoding="utf-8"))

            self.assertEqual(id_grid["grid"], [[1, 2], [2, 1]])
            self.assertEqual(token_payload["sequence"], [1, 2, -2, 2, 1, -3])
            self.assertEqual(token_text, "1 2 -2 2 1 -3")
            self.assertEqual(len(manifest_lines), 1)
            manifest_entry = json.loads(manifest_lines[0])
            self.assertEqual(manifest_entry["rows"], 2)
            self.assertEqual(manifest_entry["ar_id_grid_path"], "Tuck/001_resized/ar_id_grid.json")
            self.assertEqual(vocab["special_tokens"]["ambiguous_id"], -1)

    def test_pipeline_rejects_colliding_special_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            output_root = root / "outputs"
            legend = {"A": [{"id": 1, "color": "#FF0000"}]}
            (dataset_root / "all_info.json").parent.mkdir(parents=True, exist_ok=True)
            (dataset_root / "all_info.json").write_text(json.dumps(legend), encoding="utf-8")
            _write_png(dataset_root / "stitch code patterns" / "Tuck" / "001_resized.png", Image.new("RGB", (2, 2), (255, 0, 0)))
            _write_png(dataset_root / "simulation images" / "Tuck" / "1.png", Image.new("RGB", (2, 2), (0, 0, 0)))

            with self.assertRaisesRegex(ValueError, "collide"):
                run_pipeline(
                    dataset_root=dataset_root,
                    output_root=output_root,
                    categories=["Tuck"],
                    export_ar=True,
                    ar_eos_token=1,
                    cell_width=1,
                    cell_height=1,
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
