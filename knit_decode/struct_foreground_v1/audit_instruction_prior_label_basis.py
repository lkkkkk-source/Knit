from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


NUM_LABELS = 16
CANONICAL_SIZE = 20
IGNORE_INDEX = -100


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise ImportError("PyTorch is required to read the foreground cache .pt file.") from error
    return torch


def _to_tensor(value: Any, *, name: str) -> Any | None:
    if value is None:
        return None
    torch = _require_torch()
    if hasattr(value, "detach"):
        return value.detach().cpu()
    return torch.tensor(value)


def _as_label_grid(value: Any, *, context: str) -> list[list[int]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or len(value) != CANONICAL_SIZE:
        raise ValueError(f"{context} must have shape [20,20].")
    grid: list[list[int]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != CANONICAL_SIZE:
            raise ValueError(f"{context} row {row_index} must have 20 columns.")
        out_row: list[int] = []
        for col_index, cell in enumerate(row):
            label = int(cell)
            if label == IGNORE_INDEX:
                label = 0
            if not 0 <= label <= NUM_LABELS:
                raise ValueError(
                    f"{context} contains invalid label {label} at ({row_index},{col_index}); "
                    f"expected 0..16 or {IGNORE_INDEX}."
                )
            out_row.append(label)
        grid.append(out_row)
    return grid


def _entropy_from_probs(values: list[float]) -> float:
    return float(max(0.0, -sum(value * math.log(value) for value in values if value > 0.0)))


def _target_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts: Counter[int] = Counter()
    sample_rows: list[dict[str, Any]] = []
    for item in items:
        sample_id = str(item.get("sample_id"))
        raw = _as_label_grid(item.get("fg_y20"), context=f"item[{sample_id}].fg_y20")
        sample_counts: Counter[int] = Counter()
        for row in raw:
            for value in row:
                if 1 <= int(value) <= NUM_LABELS:
                    sample_counts[int(value)] += 1
                    label_counts[int(value)] += 1
        sample_total = int(sum(sample_counts.values()))
        sample_rows.append(
            {
                "sample_id": sample_id,
                "target_fg_area": sample_total,
                "target_label_unique_tokens": sorted(sample_counts),
                "target_label_diversity": int(len(sample_counts)),
                "target_dominant_label_ratio": float(max(sample_counts.values()) / sample_total) if sample_total > 0 else 0.0,
                "target_label_histogram": {str(key): int(sample_counts[key]) for key in sorted(sample_counts)},
            }
        )
    total = int(sum(label_counts.values()))
    probs = [float(label_counts.get(label, 0)) / float(total) if total > 0 else 0.0 for label in range(1, NUM_LABELS + 1)]
    return {
        "num_samples": int(len(items)),
        "target_fg_area": total,
        "target_label_unique_tokens": sorted(label_counts),
        "target_label_diversity": int(len(label_counts)),
        "target_dominant_label_ratio": float(max(label_counts.values()) / total) if total > 0 else 0.0,
        "target_label_entropy": _entropy_from_probs(probs),
        "target_label_histogram": {str(key): int(label_counts[key]) for key in sorted(label_counts)},
        "per_sample": sample_rows,
    }


def _basis_stats_for_tensor(
    *,
    name: str,
    label_prob: Any | None,
    fg_prob: Any,
    fg_threshold: float,
    top_k: int,
) -> list[dict[str, Any]]:
    torch = _require_torch()
    label_prob = _to_tensor(label_prob, name=name)
    fg_prob = _to_tensor(fg_prob, name="basis_fg_mask_prob")
    if label_prob is None:
        return [{"tensor": name, "missing": True}]
    if label_prob.ndim != 4 or tuple(int(dim) for dim in label_prob.shape[1:]) != (NUM_LABELS, CANONICAL_SIZE, CANONICAL_SIZE):
        raise ValueError(f"{name} must have shape [K,16,20,20], got {tuple(int(dim) for dim in label_prob.shape)}.")
    if fg_prob.ndim != 4 or tuple(int(dim) for dim in fg_prob.shape[1:]) != (1, CANONICAL_SIZE, CANONICAL_SIZE):
        raise ValueError(f"basis_fg_mask_prob must have shape [K,1,20,20], got {tuple(int(dim) for dim in fg_prob.shape)}.")
    if int(label_prob.shape[0]) != int(fg_prob.shape[0]):
        raise ValueError(f"{name} mode count does not match basis_fg_mask_prob.")

    rows: list[dict[str, Any]] = []
    for mode_index in range(int(label_prob.shape[0])):
        mask = fg_prob[mode_index, 0] >= float(fg_threshold)
        prior_fg_area = int(mask.sum().item())
        row: dict[str, Any] = {"tensor": name, "mode": int(mode_index), "prior_fg_area": prior_fg_area}
        if prior_fg_area <= 0:
            row.update(
                {
                    "basis_label_argmax_unique_tokens_over_prior_foreground": [],
                    "basis_label_prob_16_entropy_mean_over_prior_foreground": 0.0,
                    "dominant_label_ratio": 0.0,
                    "nonzero_label_channels": [],
                    "top_label_channels_and_probabilities": [],
                }
            )
            rows.append(row)
            continue

        values = label_prob[mode_index, :, mask].to(dtype=torch.float32)
        argmax_tokens = (values.argmax(dim=0) + 1).tolist()
        argmax_counts = Counter(int(value) for value in argmax_tokens)
        mean_prob = values.mean(dim=1)
        entropy_per_pixel = -(values.clamp_min(1.0e-12) * values.clamp_min(1.0e-12).log()).sum(dim=0)
        top_probs = sorted(
            [(label + 1, float(mean_prob[label].item())) for label in range(NUM_LABELS)],
            key=lambda item: item[1],
            reverse=True,
        )[: max(1, int(top_k))]
        nonzero_channels = [label + 1 for label in range(NUM_LABELS) if float(mean_prob[label].item()) > 1.0e-8]
        row.update(
            {
                "basis_label_argmax_unique_tokens_over_prior_foreground": sorted(argmax_counts),
                "basis_label_argmax_histogram_over_prior_foreground": {str(key): int(argmax_counts[key]) for key in sorted(argmax_counts)},
                "basis_label_prob_16_entropy_mean_over_prior_foreground": float(entropy_per_pixel.mean().item()),
                "dominant_label_ratio": float(max(argmax_counts.values()) / sum(argmax_counts.values())),
                "nonzero_label_channels": nonzero_channels,
                "top_label_channels_and_probabilities": [[int(label), float(prob)] for label, prob in top_probs],
            }
        )
        rows.append(row)
    return rows


def _category_hist_rows(category_hist: Any) -> list[list[float]]:
    if category_hist is None:
        return []
    if hasattr(category_hist, "detach"):
        category_hist = category_hist.detach().cpu().tolist()
    return [[label + 1, float(value)] for label, value in enumerate(category_hist) if float(value) > 0.0]


def _summarize_category(
    *,
    category: str,
    items: list[dict[str, Any]],
    prior_entry: dict[str, Any] | None,
    fg_threshold: float,
    top_k: int,
) -> dict[str, Any]:
    target = _target_stats(items)
    result: dict[str, Any] = {
        "category": category,
        "target": target,
        "prior_present": bool(prior_entry),
    }
    if not prior_entry:
        result["verdict"] = "RISK"
        result["reason"] = "missing instruction_matrix_grammar_prior category entry"
        return result

    fg_prob = prior_entry.get("basis_fg_mask_prob")
    raw_rows = _basis_stats_for_tensor(
        name="basis_label_prob_16_raw",
        label_prob=prior_entry.get("basis_label_prob_16_raw"),
        fg_prob=fg_prob,
        fg_threshold=fg_threshold,
        top_k=top_k,
    )
    calibrated_rows = _basis_stats_for_tensor(
        name="basis_label_prob_16_calibrated",
        label_prob=prior_entry.get("basis_label_prob_16_calibrated"),
        fg_prob=fg_prob,
        fg_threshold=fg_threshold,
        top_k=top_k,
    )
    active_rows = _basis_stats_for_tensor(
        name="basis_label_prob_16",
        label_prob=prior_entry.get("basis_label_prob_16"),
        fg_prob=fg_prob,
        fg_threshold=fg_threshold,
        top_k=top_k,
    )
    basis_rows = {
        "raw": raw_rows,
        "calibrated": calibrated_rows,
        "active": active_rows,
    }
    result.update(
        {
            "prior_meta": {
                "strategy": prior_entry.get("strategy"),
                "requested_modes": prior_entry.get("requested_modes"),
                "effective_modes": prior_entry.get("effective_modes"),
                "num_samples": prior_entry.get("num_samples"),
                "category_usable": prior_entry.get("category_usable"),
                "unusable_reasons": prior_entry.get("unusable_reasons"),
                "label_hist_interpolation_alpha": prior_entry.get("label_hist_interpolation_alpha"),
                "mode_prior_smoothing_beta": prior_entry.get("mode_prior_smoothing_beta"),
                "mode_num_samples": prior_entry.get("mode_num_samples"),
            },
            "category_label_hist_16": _category_hist_rows(prior_entry.get("category_label_hist_16")),
            "basis": basis_rows,
        }
    )

    target_multi = int(target["target_label_diversity"]) > 1
    active_single = all(
        len(row.get("basis_label_argmax_unique_tokens_over_prior_foreground", [])) <= 1
        for row in active_rows
        if not row.get("missing")
    )
    raw_single = all(
        len(row.get("basis_label_argmax_unique_tokens_over_prior_foreground", [])) <= 1
        for row in raw_rows
        if not row.get("missing")
    )
    calibrated_single = all(
        len(row.get("basis_label_argmax_unique_tokens_over_prior_foreground", [])) <= 1
        for row in calibrated_rows
        if not row.get("missing")
    )
    result["diagnosis"] = {
        "is_target_multi_token": bool(target_multi),
        "is_active_basis_argmax_single_token_per_mode": bool(active_single),
        "is_raw_basis_argmax_single_token_per_mode": bool(raw_single),
        "is_calibrated_basis_argmax_single_token_per_mode": bool(calibrated_single),
        "collapse_happens_in": [
            name
            for name, collapsed in [
                ("raw", raw_single),
                ("calibrated", calibrated_single),
                ("active", active_single),
            ]
            if collapsed
        ],
    }
    if target_multi and active_single:
        result["verdict"] = "FAIL"
        result["reason"] = "target is multi-token but active grammar prior basis argmax is single-token per mode"
    elif target_multi and (raw_single or calibrated_single):
        result["verdict"] = "RISK"
        result["reason"] = "target is multi-token and at least one prior tensor collapses to single-token per mode"
    else:
        result["verdict"] = "PASS"
        result["reason"] = "no target-vs-prior label collapse detected by this audit"
    return result


def _print_text_report(report: dict[str, Any]) -> None:
    print(f"cache: {report['cache']}")
    print(f"schema_version: {report.get('schema_version')}")
    print(f"prior_enabled: {report.get('prior_enabled')}")
    for category_report in report["categories"]:
        print("")
        print(f"category: {category_report['category']}")
        print(f"verdict: {category_report['verdict']}")
        print(f"reason: {category_report['reason']}")
        target = category_report["target"]
        print(
            "target: "
            f"samples={target['num_samples']} "
            f"fg_area={target['target_fg_area']} "
            f"unique={target['target_label_unique_tokens']} "
            f"diversity={target['target_label_diversity']} "
            f"dominant_ratio={target['target_dominant_label_ratio']:.6f} "
            f"entropy={target['target_label_entropy']:.6f}"
        )
        print(f"target_histogram: {json.dumps(target['target_label_histogram'], ensure_ascii=False, sort_keys=True)}")
        if not category_report.get("prior_present"):
            continue
        print(f"prior_meta: {json.dumps(category_report['prior_meta'], ensure_ascii=False, sort_keys=True)}")
        print(f"category_label_hist_16: {json.dumps(category_report['category_label_hist_16'], ensure_ascii=False)}")
        print(f"diagnosis: {json.dumps(category_report['diagnosis'], ensure_ascii=False, sort_keys=True)}")
        for group_name in ["raw", "calibrated", "active"]:
            for row in category_report["basis"][group_name]:
                print(f"{group_name}: {json.dumps(row, ensure_ascii=False, sort_keys=True)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit instruction_matrix_grammar_prior label-basis diversity from a foreground cache.")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("knit_decode/struct_foreground_v1/cache/foreground_v1/foreground_cache_train.pt"),
        help="Path to foreground_cache_train.pt.",
    )
    parser.add_argument("--category", default="Cable1", help="Category to inspect, or 'ALL'.")
    parser.add_argument("--fg-threshold", type=float, default=0.05, help="Foreground threshold for prior foreground pixels.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of top label channels to print per mode.")
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write the full report JSON.")
    args = parser.parse_args(argv)

    torch = _require_torch()
    if not args.cache.exists():
        raise FileNotFoundError(f"cache file not found: {args.cache}")
    payload = torch.load(args.cache, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"cache payload must be a dict, got {type(payload).__name__}.")

    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError("cache payload field 'items' must be a list.")
    prior = payload.get("instruction_matrix_grammar_prior", {})
    prior_categories = prior.get("categories", {}) if isinstance(prior, dict) and isinstance(prior.get("categories", {}), dict) else {}

    item_categories = sorted({str(item.get("category")) for item in items})
    if str(args.category).upper() == "ALL":
        categories = item_categories
    else:
        categories = [str(args.category)]

    category_reports: list[dict[str, Any]] = []
    for category in categories:
        category_items = [item for item in items if str(item.get("category")) == category]
        if not category_items:
            category_reports.append(
                {
                    "category": category,
                    "verdict": "RISK",
                    "reason": "no cache items for category",
                    "target": _target_stats([]),
                    "prior_present": bool(category in prior_categories),
                }
            )
            continue
        prior_entry = prior_categories.get(category)
        category_reports.append(
            _summarize_category(
                category=category,
                items=category_items,
                prior_entry=prior_entry if isinstance(prior_entry, dict) else None,
                fg_threshold=float(args.fg_threshold),
                top_k=int(args.top_k),
            )
        )

    verdict_order = {"FAIL": 3, "RISK": 2, "PASS": 1}
    overall = max((row["verdict"] for row in category_reports), key=lambda value: verdict_order.get(str(value), 0), default="RISK")
    report = {
        "cache": str(args.cache),
        "schema_version": (payload.get("meta", {}) or {}).get("schema_version") if isinstance(payload.get("meta", {}), dict) else None,
        "prior_enabled": prior.get("enabled") if isinstance(prior, dict) else False,
        "overall_verdict": overall,
        "categories": category_reports,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"overall_verdict: {overall}")
        _print_text_report(report)
        if args.output_json is not None:
            print(f"wrote_json: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
