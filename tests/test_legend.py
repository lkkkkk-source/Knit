from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from knit_decode.legend import load_legend


class LegendTests(unittest.TestCase):
    def test_duplicate_colors_are_detected_without_collapsing_ids(self) -> None:
        payload = {
            "Front Knit": [{"id": 1, "color": "#112233"}],
            "Back Knit": [{"id": 2, "color": "#112233"}],
            "Miss": [{"id": 3, "color": "#445566"}],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            legend_path = Path(temp_dir) / "all_info.json"
            legend_path.write_text(json.dumps(payload), encoding="utf-8")
            legend = load_legend(legend_path)

        self.assertEqual(legend.candidate_ids_for_color((17, 34, 51)), [1, 2])
        self.assertEqual(legend.duplicate_colors, {"#112233": [1, 2]})
        self.assertEqual(len(legend.unique_colors), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
