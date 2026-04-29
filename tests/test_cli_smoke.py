from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from knit_decode.cli import main


class CliSmokeTests(unittest.TestCase):
    def test_cli_is_safe_when_dataset_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "outputs"
            exit_code = main([
                "--dataset-root",
                str(Path(temp_dir) / "missing-dataset"),
                "--output-dir",
                str(output_dir),
            ])
            summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["status"], "dataset_root_missing")
        self.assertEqual(summary["processed_samples"], 0)
        self.assertEqual(summary["discovered_samples"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
