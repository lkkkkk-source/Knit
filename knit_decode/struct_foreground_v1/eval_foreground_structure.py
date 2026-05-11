from __future__ import annotations

import argparse
import json
from pathlib import Path

from .utils import fg_mask_iou, format_metric_line, load_config, load_label_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate foreground canonical candidates.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, required=True)
    return parser


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _ = load_config(args.config)
    payload = json.loads((args.samples_dir / "candidates.json").read_text(encoding="utf-8"))
    samples = payload["samples"]
    result = {
        "fg_mask_iou": 0.0,
        "foreground_label_ce": 0.0,
        "fg_label_acc_on_fg": 0.0,
        "bbox_error": 0.0,
        "foreground_adjacency_distance": 0.0,
        "transition_distance": 0.0,
        "category_descriptor_margin": 0.0,
        "composed_all_background_rate": _mean([1.0 if "empty_foreground" in sample.get("invalid_reasons", []) else 0.0 for sample in samples]),
        "composed_all_foreground_rate": _mean([1.0 if "full_foreground" in sample.get("invalid_reasons", []) else 0.0 for sample in samples]),
        "valid_foreground_area_rate": _mean([1.0 if sample.get("is_valid_foreground", False) else 0.0 for sample in samples]),
        "label_diversity_on_foreground": _mean([float(sample.get("label_diversity", 0.0)) for sample in samples]),
    }
    (args.samples_dir / "foreground_eval.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(format_metric_line("eval-foreground:", [("valid_fg_rate", result["valid_foreground_area_rate"]), ("all_bg", result["composed_all_background_rate"]), ("label_div", result["label_diversity_on_foreground"])]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
