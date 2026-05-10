from __future__ import annotations

import argparse
import json
from pathlib import Path

from .utils import count_tiny_islands, format_metric_line, load_config, upsample_nearest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rerank latent-plan candidates using structure-only metrics.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, required=True)
    return parser


def _coarse_violations(y20: list[list[int]], c5: list[list[int]], o5: list[list[int]], background_class_id: int) -> tuple[float, float]:
    size = len(y20)
    coarse_size = len(c5)
    block = size // coarse_size
    occ_violations = 0
    cls_violations = 0
    total = 0
    for y_pos in range(coarse_size):
        for x_pos in range(coarse_size):
            total += 1
            block_values = [
                y20[yy][xx]
                for yy in range(y_pos * block, min(size, (y_pos + 1) * block))
                for xx in range(x_pos * block, min(size, (x_pos + 1) * block))
            ]
            pred_fg = sum(1 for value in block_values if value != background_class_id) / float(max(1, len(block_values)))
            pred_occ = 1 if pred_fg >= 0.25 else 0
            occ_violations += float(pred_occ != int(o5[y_pos][x_pos]))
            if pred_occ:
                counts: dict[int, int] = {}
                for value in block_values:
                    if value != background_class_id:
                        counts[int(value)] = counts.get(int(value), 0) + 1
                dominant = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if counts else background_class_id
                cls_violations += float(dominant != int(c5[y_pos][x_pos]))
    return occ_violations / float(max(1, total)), cls_violations / float(max(1, total))


def _score_sample(sample: dict[str, object], lambdas: dict[str, float], background_class_id: int) -> dict[str, object]:
    y20 = sample["y20"]
    c5 = sample["c5"]
    o5 = sample["o5"]
    pred_fg = [[int(value) != background_class_id for value in row] for row in y20]
    all_background_penalty = 1.0 if not any(any(row) for row in pred_fg) else 0.0
    disconnected_penalty = max(0.0, float(sample.get("connected_components", 0.0)) - 1.0)
    tiny_island_penalty = count_tiny_islands(pred_fg, max_area=int(lambdas.get("tiny_island_threshold", 2))) / 20.0
    coarse_occ_violation, coarse_label_violation = _coarse_violations(y20, c5, o5, background_class_id=background_class_id)
    score = (
        float(sample["refiner_score"])
        - lambdas["lambda_count"] * float(sample["class_count_l1"])
        - lambdas["lambda_fg"] * float(sample["foreground_ratio_error"])
        - lambdas["lambda_conn"] * disconnected_penalty
        - lambdas["lambda_island"] * tiny_island_penalty
        - lambdas["lambda_plan"] * (coarse_occ_violation + coarse_label_violation)
        - 2.0 * all_background_penalty
    )
    sample["all_background_penalty"] = all_background_penalty
    sample["disconnected_penalty"] = disconnected_penalty
    sample["tiny_island_penalty"] = tiny_island_penalty
    sample["coarse_occupancy_violation"] = coarse_occ_violation
    sample["coarse_label_violation"] = coarse_label_violation
    sample["rerank_score"] = score
    return sample


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    rerank_cf = config["rerank"]
    background_class_id = int(config["data"]["background_class_id"])
    candidates_path = args.samples_dir / "candidates.json"
    payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    scored = [_score_sample(sample, rerank_cf, background_class_id=background_class_id) for sample in payload["samples"]]
    scored.sort(key=lambda row: row["rerank_score"], reverse=True)
    result = {
        "category": payload["category"],
        "num_candidates": payload["num_candidates"],
        "ranked": scored,
    }
    (args.samples_dir / "ranked_candidates.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    top = scored[0] if scored else {}
    print(format_metric_line("rerank:", [("category", payload["category"]), ("num_candidates", payload["num_candidates"]), ("top_score", float(top.get("rerank_score", 0.0))) ]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
