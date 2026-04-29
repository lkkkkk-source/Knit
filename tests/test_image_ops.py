from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from knit_decode.image_ops import crop_active_region, decode_grid, infer_grid_spec, quantize_to_legend, reconstruct_grid
from knit_decode.legend import load_legend


class ImageOpsTests(unittest.TestCase):
    def test_synthetic_round_trip_decodes_and_reconstructs(self) -> None:
        payload = {
            "A": [{"id": 1, "color": "#FF0000"}],
            "B": [{"id": 2, "color": "#00FF00"}],
            "C": [{"id": 3, "color": "#0000FF"}],
            "C alias": [{"id": 4, "color": "#0000FF"}],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            legend_path = Path(temp_dir) / "all_info.json"
            legend_path.write_text(json.dumps(payload), encoding="utf-8")
            legend = load_legend(legend_path)

        image = Image.new("RGB", (14, 12), (250, 250, 250))
        pattern = [
            [(255, 0, 0), (0, 255, 0), (0, 0, 255)],
            [(0, 255, 0), (255, 0, 0), (0, 0, 255)],
        ]
        for row_index, row in enumerate(pattern):
            for column_index, rgb in enumerate(row):
                for y_pos in range(1 + row_index * 5, 1 + (row_index + 1) * 5):
                    for x_pos in range(1 + column_index * 4, 1 + (column_index + 1) * 4):
                        image.putpixel((x_pos, y_pos), rgb)

        cropped = crop_active_region(image)
        self.assertEqual(cropped.crop_box, (1, 1, 13, 11))

        quantized = quantize_to_legend(cropped.image, legend)
        grid_spec = infer_grid_spec(quantized.image, cell_width=4, cell_height=5)
        decoded = decode_grid(quantized.image, grid_spec, legend)
        reconstruction = reconstruct_grid(decoded)

        self.assertEqual((grid_spec.columns, grid_spec.rows), (3, 2))
        self.assertEqual(decoded.cells[0][0].action_id, 1)
        self.assertEqual(decoded.cells[0][1].action_id, 2)
        self.assertIsNone(decoded.cells[0][2].action_id)
        self.assertEqual(decoded.cells[0][2].candidate_ids, (3, 4))
        self.assertEqual(quantized.image.tobytes(), reconstruction.tobytes())

    def test_auto_grid_inference_uses_dominant_color_runs(self) -> None:
        image = Image.new("RGB", (8, 12), (255, 0, 0))
        for x_pos in range(0, 8, 2):
            for y_pos in range(12):
                image.putpixel((x_pos, y_pos), (0, 255, 0))

        grid_spec = infer_grid_spec(image)

        self.assertEqual(grid_spec.cell_width, 1)
        self.assertEqual(grid_spec.cell_height, 12)
        self.assertEqual(grid_spec.columns, 8)
        self.assertEqual(grid_spec.rows, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
