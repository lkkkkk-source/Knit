from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from PIL import Image

from knit_decode.discovery import discover_samples, normalize_pairing_key


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (255, 255, 255)).save(path)


class DiscoveryTests(unittest.TestCase):
    def test_pairing_normalization_handles_resized_prefixes_and_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "dataset"
            _write_png(dataset_root / "stitch code patterns" / "Hem" / "001_Rib_resized.png")
            _write_png(dataset_root / "stitch code patterns" / "Hem" / "039A_resized.png")
            _write_png(dataset_root / "simulation images" / "Hem" / "1.png")
            _write_png(dataset_root / "simulation images" / "Hem" / "39A.png")

            samples = discover_samples(dataset_root, ["Hem"])

        self.assertEqual(normalize_pairing_key("001_Rib_resized"), "1")
        self.assertEqual(normalize_pairing_key("039A_resized"), "39A")
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].source_stem, "001_Rib")
        self.assertTrue(samples[0].simulation_path and samples[0].simulation_path.name == "1.png")
        self.assertTrue(samples[1].simulation_path and samples[1].simulation_path.name == "39A.png")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
