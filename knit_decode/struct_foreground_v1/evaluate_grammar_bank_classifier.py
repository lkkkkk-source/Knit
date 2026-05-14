from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .grammar_energy import CANONICAL_SIZE, NUM_LABELS, GrammarEnergy, compute_candidate_descriptors
from .utils import IGNORE_INDEX, canonicalize_foreground, format_metric_line, load_config, load_label_grid, resolve_manifest_path, save_json, save_jsonl


LARGE_ENERGY = 1.0e12
ABLATION_GROUPS = {
    "area_only": {"area"},
    "hist_only": {"hist"},
    "transition_only": {"trans"},
    "motif_only": {"motif"},
    "component_only": {"conn"},
    "rowcol_only": {"rowcol"},
    "hist_transition_motif": {"hist", "trans", "motif"},
    "full": {"area", "conn", "hist", "rowcol", "trans", "occ_trans", "motif", "div", "dom", "single_label"},
}


def _str_to_bool(value: str | bool | None) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate grammar_bank as a nearest-grammar category classifier.")
    parser.add_argument("--cache", type=Path, default=Path("knit_decode/struct_foreground_v1/cache/foreground_v1/foreground_cache_train.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/manifests/inverse_rendering_val_frontonly.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("knit_decode/struct_foreground_v1/configs/foreground_v1.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/struct_foreground_v1/grammar_bank_classifier/val"))
    parser.add_argument("--split-name", type=str, default="val")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--categories", type=str, default=None, help="Optional comma-separated grammar categories to evaluate/classify against.")
    parser.add_argument("--skip-unknown-category", nargs="?", const=True, default=True, type=_str_to_bool)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--save-per-sample", nargs="?", const=True, default=True, type=_str_to_bool)
    parser.add_argument("--ablation", nargs="?", const=True, default=True, type=_str_to_bool)
    parser.add_argument("--weights-json", type=str, default=None, help="Optional JSON object overriding grammar_rerank.weights.")
    return parser


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Manifest row {line_no} is not an object: {path}")
        rows.append(row)
    return rows


def _infer_manifest_root(manifest_path: Path, rows: list[dict[str, Any]]) -> Path:
    search_roots = [manifest_path.parent, *manifest_path.parents]
    sample = rows[: min(32, len(rows))]
    for candidate_root in search_roots:
        ok = True
        checked = 0
        for row in sample:
            raw = row.get("target_path")
            if not isinstance(raw, str):
                continue
            checked += 1
            if not (candidate_root / raw).exists():
                ok = False
                break
        if ok and checked > 0:
            return candidate_root
    return manifest_path.parent


def _parse_category_filter(raw: str | None, available: list[str]) -> list[str]:
    if raw is None or not raw.strip():
        return available
    requested = [part.strip() for part in raw.split(",") if part.strip()]
    missing = [cat for cat in requested if cat not in set(available)]
    if missing:
        raise ValueError(f"--categories contains values absent from grammar_bank: {missing}")
    return requested


def _load_weights(config: dict[str, Any], override_json: str | None) -> dict[str, float]:
    rerank_cf = config.get("grammar_rerank", {}) if isinstance(config.get("grammar_rerank", {}), dict) else {}
    raw_weights = rerank_cf.get("weights", {}) if isinstance(rerank_cf.get("weights", {}), dict) else {}
    weights = {str(key): float(value) for key, value in raw_weights.items()}
    if override_json:
        override = json.loads(override_json)
        if not isinstance(override, dict):
            raise ValueError("--weights-json must decode to an object.")
        weights.update({str(key): float(value) for key, value in override.items()})
    return weights


def _ablation_weights(base_weights: dict[str, float], group_name: str) -> dict[str, float]:
    keep = ABLATION_GROUPS[group_name]
    keys = set(base_weights) | {"area", "conn", "hist", "rowcol", "trans", "occ_trans", "motif", "div", "dom", "single_label", "graph", "nmf"}
    return {key: (float(base_weights.get(key, 0.0)) if key in keep else 0.0) for key in keys}


def _matrix_shape(value: Any) -> tuple[int, int]:
    if not isinstance(value, list):
        return (0, 0)
    return (len(value), len(value[0]) if value and isinstance(value[0], list) else 0)


def _validate_descriptor_shapes(desc: dict[str, Any], sample_id: str) -> None:
    checks = [
        ("label_hist_norm_16", (len(desc.get("label_hist_norm_16", [])),), (NUM_LABELS,)),
        ("transition_h_norm_16x16", _matrix_shape(desc.get("transition_h_norm_16x16")), (NUM_LABELS, NUM_LABELS)),
        ("transition_v_norm_16x16", _matrix_shape(desc.get("transition_v_norm_16x16")), (NUM_LABELS, NUM_LABELS)),
        ("occupancy_transition_h_norm_2x2", _matrix_shape(desc.get("occupancy_transition_h_norm_2x2")), (2, 2)),
        ("occupancy_transition_v_norm_2x2", _matrix_shape(desc.get("occupancy_transition_v_norm_2x2")), (2, 2)),
        ("row_fg_projection", (len(desc.get("row_fg_projection", [])),), (CANONICAL_SIZE,)),
        ("col_fg_projection", (len(desc.get("col_fg_projection", [])),), (CANONICAL_SIZE,)),
    ]
    for name, got, expected in checks:
        if got != expected:
            raise ValueError(f"Descriptor shape check failed for sample_id={sample_id}: {name} got {got}, expected {expected}.")


def _raw_y20_from_manifest_row(row: dict[str, Any], manifest_root: Path, data_cf: dict[str, Any]) -> list[list[int]]:
    sample_id = str(row.get("sample_id", row.get("id", "unknown")))
    if not isinstance(row.get("target_path"), str):
        raise ValueError(f"Manifest row missing target_path for sample_id={sample_id}.")
    target_path = resolve_manifest_path(str(row["target_path"]), manifest_root, sample_id=sample_id, field_name="target_path")
    y20 = load_label_grid(target_path, sample_id=sample_id)
    canonical = canonicalize_foreground(
        y20,
        background_class_id=int(data_cf.get("background_class_id", 0)),
        canonical_size=int(data_cf.get("canonical_size", CANONICAL_SIZE)),
        canonical_mode="full_masked",
        ignore_index=int(data_cf.get("ignore_index", IGNORE_INDEX)),
    )
    fg_y20 = canonical["fg_y20"]
    fg_mask20 = canonical["fg_mask20"]
    return [
        [int(fg_y20[y][x]) if int(fg_mask20[y][x]) else 0 for x in range(CANONICAL_SIZE)]
        for y in range(CANONICAL_SIZE)
    ]


def _finite_or_large(value: Any) -> tuple[float, bool]:
    energy = float(value)
    if not math.isfinite(energy) or math.isnan(energy):
        return LARGE_ENERGY, False
    return energy, True


def _score_sample(
    y20: list[list[int]],
    categories: list[str],
    scorer: GrammarEnergy,
    sample_id: str,
) -> tuple[dict[str, float], dict[str, dict[str, Any]], list[str]]:
    energy_by_category: dict[str, float] = {}
    breakdown_by_category: dict[str, dict[str, Any]] = {}
    nan_categories: list[str] = []
    for category in categories:
        try:
            score = scorer.score_matrix_against_category(y20, category)
            total, is_finite = _finite_or_large(score.get("total", LARGE_ENERGY))
            if not is_finite:
                nan_categories.append(category)
            energy_by_category[category] = total
            breakdown_by_category[category] = {
                "total": total,
                "area": float(score.get("area", 0.0)),
                "conn": float(score.get("conn", 0.0)),
                "hist": float(score.get("hist", 0.0)),
                "rowcol": float(score.get("rowcol", 0.0)),
                "trans": float(score.get("trans", 0.0)),
                "occ_trans": float(score.get("occ_trans", 0.0)),
                "motif": float(score.get("motif", 0.0)),
                "div": float(score.get("div", 0.0)),
                "dom": float(score.get("dom", 0.0)),
                "single_label": float(score.get("single_label_collapse", 0.0)),
                "classifier_valid": bool(score.get("classifier_valid", True)),
                "classifier_invalid_reasons": score.get("classifier_invalid_reasons", []),
                "rerank_invalid_reasons_ignored_for_classifier": score.get("rerank_invalid_reasons_ignored_for_classifier", []),
                "diagnostics": score.get("diagnostics", {}),
            }
        except Exception as error:
            energy_by_category[category] = LARGE_ENERGY
            breakdown_by_category[category] = {"total": LARGE_ENERGY, "error": f"{type(error).__name__}: {error}"}
            nan_categories.append(category)
    if len(nan_categories) == len(categories):
        raise ValueError(f"All category energies failed for sample_id={sample_id}.")
    return energy_by_category, breakdown_by_category, nan_categories


def _prediction_from_energy(energy_by_category: dict[str, float], top_k: int) -> tuple[str, list[str], list[float], float]:
    ranked = sorted(energy_by_category.items(), key=lambda item: (float(item[1]), item[0]))
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else best
    top = ranked[: max(1, min(top_k, len(ranked)))]
    margin = float(second[1]) - float(best[1]) if len(ranked) > 1 else 0.0
    return best[0], [cat for cat, _ in top], [float(value) for _, value in top], float(margin)


def _confusion_matrix(rows: list[dict[str, Any]], categories: list[str]) -> list[list[int]]:
    index = {category: i for i, category in enumerate(categories)}
    matrix = [[0 for _ in categories] for _ in categories]
    for row in rows:
        true_category = str(row["true_category"])
        pred_category = str(row["pred_category"])
        if true_category in index and pred_category in index:
            matrix[index[true_category]][index[pred_category]] += 1
    return matrix


def _accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return float(sum(1 for row in rows if bool(row["correct"]))) / float(len(rows))


def _topk_accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return float(sum(1 for row in rows if str(row["true_category"]) in row.get("topk_categories", []))) / float(len(rows))


def _per_category_accuracy(rows: list[dict[str, Any]], categories: list[str]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for category in categories:
        cat_rows = [row for row in rows if str(row["true_category"]) == category]
        correct = sum(1 for row in cat_rows if bool(row["correct"]))
        out[category] = {
            "support": int(len(cat_rows)),
            "correct": int(correct),
            "accuracy": float(correct) / float(len(cat_rows)) if cat_rows else 0.0,
        }
    return out


def _balanced_accuracy(per_category: dict[str, dict[str, float | int]]) -> float:
    values = [float(row["accuracy"]) for row in per_category.values() if int(row["support"]) > 0]
    return sum(values) / float(len(values)) if values else 0.0


def _worst_confusions(matrix: list[list[int]], categories: list[str], top_n: int = 8) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for i, true_category in enumerate(categories):
        row_total = sum(matrix[i])
        if row_total <= 0:
            continue
        for j, pred_category in enumerate(categories):
            if i == j or matrix[i][j] <= 0:
                continue
            pairs.append({
                "true_category": true_category,
                "pred_category": pred_category,
                "count": int(matrix[i][j]),
                "rate_within_true": float(matrix[i][j]) / float(row_total),
            })
    return sorted(pairs, key=lambda item: (float(item["rate_within_true"]), int(item["count"])), reverse=True)[:top_n]


def _classification_metrics(rows: list[dict[str, Any]], categories: list[str]) -> dict[str, Any]:
    per_category = _per_category_accuracy(rows, categories)
    matrix = _confusion_matrix(rows, categories)
    correct_margins = [float(row["margin"]) for row in rows if bool(row["correct"])]
    incorrect_margins = [float(row["margin"]) for row in rows if not bool(row["correct"])]
    return {
        "num_evaluated": int(len(rows)),
        "top1_accuracy": _accuracy(rows),
        "topk_accuracy": _topk_accuracy(rows),
        "balanced_accuracy": _balanced_accuracy(per_category),
        "per_category_accuracy": per_category,
        "confusion_matrix": matrix,
        "mean_margin": sum(float(row["margin"]) for row in rows) / float(len(rows)) if rows else 0.0,
        "mean_correct_margin": sum(correct_margins) / float(len(correct_margins)) if correct_margins else 0.0,
        "mean_incorrect_margin": sum(incorrect_margins) / float(len(incorrect_margins)) if incorrect_margins else 0.0,
        "worst_confused_category_pairs": _worst_confusions(matrix, categories),
    }


def _save_confusion_csv(path: Path, matrix: list[list[int]], categories: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *categories])
        for category, row in zip(categories, matrix):
            writer.writerow([category, *row])


def _save_per_category_csv(path: Path, per_category: dict[str, dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "support", "correct", "accuracy"])
        for category, row in per_category.items():
            writer.writerow([category, int(row["support"]), int(row["correct"]), f"{float(row['accuracy']):.8f}"])


def _save_confusion_png(path: Path, matrix: list[list[int]], categories: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig_w = max(7.0, 0.55 * len(categories) + 2.0)
    fig_h = max(6.0, 0.45 * len(categories) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_xticks(range(len(categories)))
    ax.set_yticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_yticklabels(categories)
    max_value = max([value for row in matrix for value in row], default=0)
    threshold = max_value * 0.5
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            if value:
                ax.text(x, y, str(value), ha="center", va="center", color="white" if value > threshold else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_ablation_csv(path: Path, ablation_summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ablation", "top1_accuracy", "balanced_accuracy", "num_evaluated"])
        for name, metrics in ablation_summary.items():
            writer.writerow([name, f"{float(metrics['top1_accuracy']):.8f}", f"{float(metrics['balanced_accuracy']):.8f}", int(metrics["num_evaluated"])])


def _interpretation(summary: dict[str, Any], ablation_summary: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    categories = summary["categories"]
    random_baseline = 1.0 / float(max(1, len(categories)))
    top1 = float(summary["top1_accuracy"])
    lines.append(f"- Random baseline is about {random_baseline:.4f} for {len(categories)} categories; full top1 is {top1:.4f}.")
    area = ablation_summary.get("area_only", {})
    hist = ablation_summary.get("hist_only", {})
    full = ablation_summary.get("full", {})
    motif = ablation_summary.get("motif_only", {})
    trans = ablation_summary.get("transition_only", {})
    if full and area:
        delta = float(full["top1_accuracy"]) - float(area["top1_accuracy"])
        lines.append(f"- Full minus area_only top1 delta is {delta:.4f}; positive delta means grammar features add signal beyond mask area.")
    if full and hist:
        delta = float(full["top1_accuracy"]) - float(hist["top1_accuracy"])
        if delta > 0.02:
            lines.append(f"- Full beats hist_only by {delta:.4f}, so transitions, motifs, components, or layout add category-specific signal.")
        else:
            lines.append(f"- hist_only is close to full; token distribution may dominate category separation.")
    if trans and motif:
        lines.append(f"- transition_only top1={float(trans['top1_accuracy']):.4f}, motif_only top1={float(motif['top1_accuracy']):.4f}; use these to judge structural grammar signal.")
    if top1 <= random_baseline * 1.5:
        lines.append("- Full accuracy is near random baseline; grammar_bank category stats may not be discriminative.")
    if summary.get("mostly_one_predicted_category", False):
        lines.append("- Predictions are concentrated in one category; inspect confusion_matrix.csv before trusting the bank.")
    return lines


def _save_summary_md(path: Path, summary: dict[str, Any], ablation_summary: dict[str, Any]) -> None:
    best_ablation = max(ablation_summary.items(), key=lambda item: float(item[1]["top1_accuracy"]))[0] if ablation_summary else "n/a"
    worst_pairs = summary.get("worst_confused_category_pairs", [])
    lines = [
        f"# Grammar Bank Classifier Evaluation - {summary['split_name']}",
        "",
        "## Overall",
        f"- split name: {summary['split_name']}",
        f"- num rows read: {summary['num_rows_read']}",
        f"- num evaluated: {summary['num_evaluated']}",
        f"- num skipped: {summary['num_skipped']}",
        f"- skipped unknown category: {summary['skipped_unknown_category']}",
        f"- top1 accuracy: {float(summary['top1_accuracy']):.4f}",
        f"- top{summary['top_k']} accuracy: {float(summary['topk_accuracy']):.4f}",
        f"- balanced accuracy: {float(summary['balanced_accuracy']):.4f}",
        f"- mean margin: {float(summary['mean_margin']):.4f}",
        f"- best ablation: {best_ablation}",
        "",
        "## Worst Confused Category Pairs",
    ]
    if worst_pairs:
        for item in worst_pairs:
            lines.append(f"- true={item['true_category']} predicted={item['pred_category']} count={item['count']} rate={float(item['rate_within_true']):.4f}")
    else:
        lines.append("- none")
    lines.extend(["", "## Ablation"])
    for name, metrics in ablation_summary.items():
        lines.append(f"- {name}: top1={float(metrics['top1_accuracy']):.4f}, balanced={float(metrics['balanced_accuracy']):.4f}")
    lines.extend(["", "## Interpretation", *_interpretation(summary, ablation_summary)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_ablation_md(path: Path, ablation_summary: dict[str, Any]) -> None:
    lines = ["# Grammar Bank Classifier Ablation", ""]
    for name, metrics in ablation_summary.items():
        lines.append(f"- {name}: top1={float(metrics['top1_accuracy']):.4f}, balanced={float(metrics['balanced_accuracy']):.4f}, evaluated={int(metrics['num_evaluated'])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rows_for_ablation(base_rows: list[dict[str, Any]], predictions_by_sample: dict[str, dict[str, Any]], ablation_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in base_rows:
        sample_id = str(row["sample_id"])
        pred = predictions_by_sample[sample_id][ablation_name]
        rows.append({
            **row,
            "pred_category": pred["pred_category"],
            "correct": pred["correct"],
            "topk_categories": pred["topk_categories"],
            "margin": pred["margin"],
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data_cf = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    grammar_bank_cf = config.get("grammar_bank", {}) if isinstance(config.get("grammar_bank", {}), dict) else {}
    rerank_cf = config.get("grammar_rerank", {}) if isinstance(config.get("grammar_rerank", {}), dict) else {}
    weights = _load_weights(config, args.weights_json)

    import torch

    try:
        payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(args.cache, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("grammar_bank"), dict):
        raise ValueError(f"Cache does not contain grammar_bank: {args.cache}")
    grammar_bank = payload["grammar_bank"]
    grammar_categories = sorted(str(category) for category in grammar_bank.get("categories", {}).keys())
    categories = _parse_category_filter(args.categories, grammar_categories)
    if not categories:
        raise ValueError("No grammar_bank categories selected.")

    rows = _load_manifest(args.manifest)
    manifest_root = _infer_manifest_root(args.manifest, rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(format_metric_line("grammar-bank-classifier:", [("split", args.split_name), ("manifest", str(args.manifest)), ("rows_read", len(rows))]))
    print(format_metric_line("grammar-bank-categories:", [("count", len(grammar_categories)), ("categories", ",".join(grammar_categories))]))
    print(format_metric_line("grammar-bank-classifier-weights:", sorted(weights.items())))

    scorer_full = GrammarEnergy(grammar_bank, weights=weights, config={**grammar_bank_cf, **rerank_cf})
    ablation_scorers = {
        name: GrammarEnergy(grammar_bank, weights=_ablation_weights(weights, name), config={**grammar_bank_cf, **rerank_cf})
        for name in ABLATION_GROUPS
    } if bool(args.ablation) else {"full": scorer_full}
    if "full" not in ablation_scorers:
        ablation_scorers["full"] = scorer_full

    predictions: list[dict[str, Any]] = []
    predictions_by_sample: dict[str, dict[str, Any]] = {}
    skipped_unknown = 0
    skipped_filtered = 0
    skipped_errors = 0
    skipped_empty_or_illegal = 0
    nan_energy_sample_ids: list[str] = []

    for row in rows:
        if args.max_samples is not None and len(predictions) >= int(args.max_samples):
            break
        sample_id = str(row.get("sample_id", row.get("id", len(predictions))))
        true_category = str(row.get("category", ""))
        if true_category not in grammar_categories:
            skipped_unknown += 1
            if bool(args.skip_unknown_category):
                continue
            raise ValueError(f"Unknown category {true_category!r} for sample_id={sample_id}.")
        if true_category not in categories:
            skipped_filtered += 1
            continue
        try:
            y20 = _raw_y20_from_manifest_row(row, manifest_root, data_cf)
            desc = compute_candidate_descriptors(y20, grammar_bank_cf)
            _validate_descriptor_shapes(desc, sample_id)
            if int(desc["foreground_area"]) <= 0:
                skipped_empty_or_illegal += 1
                continue
        except Exception as error:
            skipped_errors += 1
            print(format_metric_line("grammar-bank-classifier-skip:", [("sample_id", sample_id), ("reason", f"{type(error).__name__}:{error}")]))
            continue

        energy_by_category, breakdown_by_category, nan_categories = _score_sample(y20, categories, scorer_full, sample_id)
        if nan_categories:
            nan_energy_sample_ids.append(sample_id)
        pred_category, topk_categories, topk_energies, margin = _prediction_from_energy(energy_by_category, int(args.top_k))
        correct = pred_category == true_category
        base_row = {
            "sample_id": sample_id,
            "true_category": true_category,
            "pred_category": pred_category,
            "correct": correct,
            "top3_categories": topk_categories[:3],
            "top3_energies": topk_energies[:3],
            "topk_categories": topk_categories,
            "topk_energies": topk_energies,
            "margin": margin,
            "energy_by_category": energy_by_category,
            "energy_breakdown_by_category": breakdown_by_category,
            "nan_energy_categories": nan_categories,
            "descriptor_summary": {
                "foreground_area_ratio": float(desc["foreground_area_ratio"]),
                "label_diversity": int(desc["label_diversity"]),
                "dominant_label_ratio": float(desc["dominant_label_ratio"]),
                "motif2_entropy": float(desc["motif2_entropy"]),
                "same_label_h_ratio": float(desc["same_label_h_ratio"]),
                "same_label_v_ratio": float(desc["same_label_v_ratio"]),
                "diff_label_h_ratio": float(desc["diff_label_h_ratio"]),
                "diff_label_v_ratio": float(desc["diff_label_v_ratio"]),
                "num_components": int(desc["num_connected_components"]),
                "largest_component_ratio": float(desc["largest_component_ratio"]),
                "tiny_island_count": int(desc["tiny_island_count"]),
            },
        }
        predictions.append(base_row)
        predictions_by_sample[sample_id] = {"full": {
            "pred_category": pred_category,
            "correct": correct,
            "topk_categories": topk_categories,
            "margin": margin,
        }}

        for ablation_name, scorer in ablation_scorers.items():
            if ablation_name == "full":
                continue
            ab_energy, _, ab_nan = _score_sample(y20, categories, scorer, sample_id)
            if ab_nan:
                nan_energy_sample_ids.append(sample_id)
            ab_pred, ab_topk, _, ab_margin = _prediction_from_energy(ab_energy, int(args.top_k))
            predictions_by_sample[sample_id][ablation_name] = {
                "pred_category": ab_pred,
                "correct": ab_pred == true_category,
                "topk_categories": ab_topk,
                "margin": ab_margin,
            }

    full_metrics = _classification_metrics(predictions, categories)
    pred_counts = Counter(str(row["pred_category"]) for row in predictions)
    mostly_one = bool(predictions) and (max(pred_counts.values()) / float(len(predictions)) >= 0.80)
    ablation_summary: dict[str, Any] = {}
    for ablation_name in ablation_scorers:
        ab_rows = predictions if ablation_name == "full" else _rows_for_ablation(predictions, predictions_by_sample, ablation_name)
        metrics = _classification_metrics(ab_rows, categories)
        ablation_summary[ablation_name] = {
            "num_evaluated": metrics["num_evaluated"],
            "top1_accuracy": metrics["top1_accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "per_category_accuracy": metrics["per_category_accuracy"],
            "confusion_matrix": metrics["confusion_matrix"],
        }

    summary = {
        "split_name": args.split_name,
        "cache": str(args.cache),
        "manifest": str(args.manifest),
        "config": str(args.config),
        "output_dir": str(output_dir),
        "num_rows_read": int(len(rows)),
        "num_evaluated": int(len(predictions)),
        "num_skipped": int(skipped_unknown + skipped_filtered + skipped_errors + skipped_empty_or_illegal),
        "skipped_unknown_category": int(skipped_unknown),
        "skipped_filtered_category": int(skipped_filtered),
        "skipped_errors": int(skipped_errors),
        "skipped_empty_or_illegal": int(skipped_empty_or_illegal),
        "grammar_bank_categories": grammar_categories,
        "categories": categories,
        "top_k": int(args.top_k),
        "top3_accuracy": full_metrics["topk_accuracy"] if int(args.top_k) == 3 else None,
        "weights_used": weights,
        "prediction_counts": dict(sorted(pred_counts.items())),
        "mostly_one_predicted_category": mostly_one,
        "nan_energy_count": int(len(set(nan_energy_sample_ids))),
        "nan_energy_sample_ids": sorted(set(nan_energy_sample_ids))[:50],
        **full_metrics,
    }

    save_json(output_dir / "summary.json", summary)
    _save_summary_md(output_dir / "summary.md", summary, ablation_summary)
    if bool(args.save_per_sample):
        save_jsonl(output_dir / "per_sample_predictions.jsonl", predictions)
    _save_confusion_csv(output_dir / "confusion_matrix.csv", full_metrics["confusion_matrix"], categories)
    _save_confusion_png(output_dir / "confusion_matrix.png", full_metrics["confusion_matrix"], categories)
    _save_per_category_csv(output_dir / "per_category_accuracy.csv", full_metrics["per_category_accuracy"])
    save_json(output_dir / "ablation_summary.json", ablation_summary)
    _save_ablation_csv(output_dir / "ablation_summary.csv", ablation_summary)
    _save_ablation_md(output_dir / "ablation_summary.md", ablation_summary)

    print(format_metric_line("grammar-bank-classifier-result:", [
        ("split", args.split_name),
        ("rows_read", len(rows)),
        ("evaluated", len(predictions)),
        ("skipped_unknown", skipped_unknown),
        ("top1", float(summary["top1_accuracy"])),
        ("topk", float(summary["topk_accuracy"])),
        ("balanced", float(summary["balanced_accuracy"])),
    ]))
    print(format_metric_line("grammar-bank-classifier-output:", [("dir", str(output_dir))]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
