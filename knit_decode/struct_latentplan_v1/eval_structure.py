from __future__ import annotations

import argparse
import json
from pathlib import Path

from .utils import format_metric_line, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate latent-plan structure candidates.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, required=True)
    return parser


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _ = load_config(args.config)
    samples_dir = args.samples_dir.resolve()
    ranked_path = samples_dir / "ranked_candidates.json"
    source_path = ranked_path if ranked_path.exists() else samples_dir / "candidates.json"
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    samples = payload["ranked"] if "ranked" in payload else payload["samples"]
    all_background_rate = _mean([float(sample.get("all_background_penalty", 0.0)) for sample in samples])
    all_foreground_rate = _mean([float(sample.get("all_foreground_penalty", sample.get("all_foreground", 0.0))) for sample in samples])
    valid_plan_rate = _mean([1.0 if bool(sample.get("is_valid_plan", False)) else 0.0 for sample in samples])
    fg_ratio_out_of_range_rate = _mean([float(sample.get("fg_ratio_out_of_range_penalty", 0.0)) for sample in samples])
    mean_fg_ratio = _mean([float(sample.get("fg_ratio", 0.0)) for sample in samples])
    mean_components = _mean([float(sample.get("connected_components", 0.0)) for sample in samples])
    mean_largest_component_ratio = _mean([float(sample.get("largest_component_ratio", 0.0)) for sample in samples])
    tiny_island_count = _mean([float(sample.get("tiny_island_penalty", 0.0)) * 20.0 for sample in samples])
    count_error = _mean([float(sample.get("class_count_l1", 0.0)) for sample in samples])
    coarse_occ = _mean([1.0 - float(sample.get("coarse_occupancy_violation", 0.0)) for sample in samples])
    coarse_cls = _mean([1.0 - float(sample.get("coarse_label_violation", 0.0)) for sample in samples])
    fg_iou = _mean([float(sample.get("foreground_iou", 0.0)) for sample in samples])
    rerank_scores = [float(sample.get("rerank_score", sample.get("refiner_score", 0.0))) for sample in samples]
    top1 = rerank_scores[0] if rerank_scores else 0.0
    topk = _mean(rerank_scores[: min(8, len(rerank_scores))]) if rerank_scores else 0.0
    result = {
        "all_background_rate": all_background_rate,
        "all_foreground_rate": all_foreground_rate,
        "valid_plan_rate": valid_plan_rate,
        "fg_ratio_out_of_range_rate": fg_ratio_out_of_range_rate,
        "mean_fg_ratio": mean_fg_ratio,
        "fg_ratio_distribution": [float(sample.get("fg_ratio", 0.0)) for sample in samples],
        "mean_fg_ratio_by_category": {
            category: _mean([float(sample.get("fg_ratio", 0.0)) for sample in samples if sample.get("category") == category])
            for category in sorted({str(sample.get("category")) for sample in samples})
        },
        "valid_plan_rate_by_category": {
            category: _mean([1.0 if bool(sample.get("is_valid_plan", False)) else 0.0 for sample in samples if sample.get("category") == category])
            for category in sorted({str(sample.get("category")) for sample in samples})
        },
        "mean_connected_components": mean_components,
        "mean_largest_component_ratio": mean_largest_component_ratio,
        "tiny_island_count": tiny_island_count,
        "class_count_distribution_error": count_error,
        "coarse_occupancy_agreement": coarse_occ,
        "coarse_label_agreement": coarse_cls,
        "foreground_iou": fg_iou,
        "intra_category_diversity": len({json.dumps(sample.get("c5", [])) for sample in samples}) / float(max(1, len(samples))),
        "top1_rerank_score": top1,
        "topk_rerank_score": topk,
    }
    (args.samples_dir / "structure_eval.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(format_metric_line("structure-eval:", [("all_bg", result["all_background_rate"]), ("fg_iou", result["foreground_iou"]), ("count_l1", result["class_count_distribution_error"]), ("top1", result["top1_rerank_score"])]))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
