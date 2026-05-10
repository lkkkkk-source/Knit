from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models.planner import LatentPlanner
from .models.refiner import PlanConditionedMaskRefiner
from .utils import finish_progress, format_metric_line, load_config, print_progress, save_binary_map, save_label_map, structure_metrics


def _require_torch() -> object:
    import importlib

    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError("PyTorch is required for candidate sampling. Install with `pip install -e .[train]`.") from error


def _resolve_device(torch: object, device_name: str) -> object:
    device_cls = getattr(torch, "device")
    if device_name == "cpu":
        return device_cls("cpu")
    if not getattr(torch, "cuda").is_available():
        raise RuntimeError(f"Requested device {device_name!r}, but CUDA is not available.")
    return device_cls(device_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample category-only structure candidates from latent planner + refiner.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--output-dir", "--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--allow-random-init-for-smoke-test", action="store_true")
    return parser


def _mask_schedule(step: int, total_steps: int) -> float:
    import math

    ratio = float(step + 1) / float(max(1, total_steps))
    return max(0.0, min(1.0, math.cos(math.pi * 0.5 * ratio)))


def _iterative_refine(refiner: object, category_ids: object, z_ids: object, c5: object, o5: object, r17: object, fg_ratio: object, steps: int, sample_temperature: float) -> tuple[object, object, object]:
    torch = _require_torch()
    functional = __import__("importlib").import_module("torch.nn.functional")
    batch_size = int(category_ids.shape[0])
    mask_token_id = int(refiner.mask_token_id)
    tokens = getattr(torch, "full")((batch_size, 20, 20), mask_token_id, dtype=getattr(torch, "long"), device=category_ids.device)
    score_accum = getattr(torch, "zeros")((batch_size,), dtype=getattr(torch, "float32"), device=category_ids.device)
    conf_accum = getattr(torch, "zeros")((batch_size,), dtype=getattr(torch, "float32"), device=category_ids.device)
    for _ in range(max(1, steps)):
        current_step = _
        outputs = refiner(tokens, category_ids, z_ids, c5, o5, r17, fg_ratio)
        logits = outputs["merged_logits"]
        scaled_logits = logits / max(float(sample_temperature), 1e-6)
        sampled = getattr(torch, "distributions").Categorical(logits=scaled_logits.permute(0, 2, 3, 1).reshape(-1, 17)).sample().reshape(batch_size, 20, 20)
        log_probs = functional.log_softmax(logits, dim=1)
        probs = functional.softmax(logits, dim=1)
        chosen_logprob = log_probs.gather(1, sampled.unsqueeze(1)).squeeze(1)
        chosen_conf = probs.gather(1, sampled.unsqueeze(1)).squeeze(1)
        masked_positions = tokens.eq(mask_token_id)
        updated = getattr(torch, "where")(masked_positions, sampled, tokens)
        score_accum = score_accum + (chosen_logprob * masked_positions.to(dtype=chosen_logprob.dtype)).sum(dim=(1, 2)) / masked_positions.to(dtype=chosen_logprob.dtype).sum(dim=(1, 2)).clamp_min(1.0)
        conf_accum = conf_accum + (chosen_conf * masked_positions.to(dtype=chosen_conf.dtype)).sum(dim=(1, 2)) / masked_positions.to(dtype=chosen_conf.dtype).sum(dim=(1, 2)).clamp_min(1.0)
        if current_step == max(1, steps) - 1:
            tokens = updated
            continue
        flat_conf = chosen_conf.reshape(batch_size, -1)
        flat_tokens = updated.reshape(batch_size, -1)
        unknown_count0 = tokens.eq(mask_token_id).reshape(batch_size, -1).sum(dim=-1)
        remask_ratio = _mask_schedule(current_step, steps)
        for batch_index in range(batch_size):
            remask_len = int(max(1, min(int(flat_tokens.shape[-1]) - 1, round(float(unknown_count0[batch_index].item()) * remask_ratio))))
            if remask_len <= 0:
                continue
            low_conf_idx = getattr(torch, "topk")(flat_conf[batch_index], k=remask_len, largest=False).indices
            flat_tokens[batch_index, low_conf_idx] = mask_token_id
        tokens = flat_tokens.reshape(batch_size, 20, 20)
    return tokens, score_accum / float(max(1, steps)), conf_accum / float(max(1, steps))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data_cf = config["data"]
    planner_cf = config["planner"]
    refiner_cf = config["refiner"]
    sampling_cf = config["sampling"]
    planner_ckpt = Path(config["train_planner"]["output_dir"]) / "checkpoint.pt"
    refiner_ckpt = Path(config["train_refiner"]["output_dir"]) / "checkpoint.pt"
    num_candidates = int(args.num_candidates or sampling_cf["num_candidates"])
    output_dir = Path(args.output_dir or (Path(config["train_refiner"]["output_dir"]) / "samples" / args.category))
    output_dir.mkdir(parents=True, exist_ok=True)

    torch = _require_torch()
    planner_payload = None
    refiner_payload = None
    planner_metrics = None
    if planner_ckpt.exists():
        planner_payload = getattr(torch, "load")(planner_ckpt, map_location="cpu")
        planner_metrics = planner_payload["metrics"]
        category_to_id = planner_metrics["category_to_id"]
    else:
        if not args.allow_random_init_for_smoke_test:
            raise FileNotFoundError(f"Missing planner checkpoint: {planner_ckpt}. Use --allow-random-init-for-smoke-test only for code-path validation.")
        category_to_id = {args.category: 0}
    if args.category not in category_to_id:
        raise KeyError(
            f"Unknown category {args.category!r}. "
            f"Available categories from checkpoint: {sorted(category_to_id)}"
        )
    device = _resolve_device(torch, args.device or config["train_refiner"]["device"])

    planner = LatentPlanner(
        num_categories=int(planner_metrics["num_categories"]) if planner_metrics is not None and "num_categories" in planner_metrics else len(category_to_id),
        num_modes=int(planner_cf["num_modes"]),
        coarse_size=int(data_cf["coarse_size"]),
        num_classes=int(data_cf["num_classes"]),
        category_embed_dim=int(planner_cf["category_embed_dim"]),
        mode_embed_dim=int(planner_cf["mode_embed_dim"]),
        hidden_dim=int(planner_cf["hidden_dim"]),
        num_layers=int(planner_cf["num_layers"]),
    )
    if planner_payload is not None:
        planner.load_state_dict(planner_payload["model_state_dict"])
    planner.to(device)
    planner.eval()

    refiner = PlanConditionedMaskRefiner(
        num_categories=int(planner_metrics["num_categories"]) if planner_metrics is not None and "num_categories" in planner_metrics else len(category_to_id),
        num_modes=int(planner_cf["num_modes"]),
        num_classes=int(data_cf["num_classes"]),
        grid_size=int(data_cf["label_size"]),
        hidden_dim=int(refiner_cf["hidden_dim"]),
        num_layers=int(refiner_cf["num_layers"]),
        num_heads=int(refiner_cf["num_heads"]),
        use_2d_rope=bool(refiner_cf["use_2d_rope"]),
    )
    if refiner_ckpt.exists():
        refiner_payload = getattr(torch, "load")(refiner_ckpt, map_location="cpu")
        refiner.load_state_dict(refiner_payload["model_state_dict"])
    elif not args.allow_random_init_for_smoke_test:
        raise FileNotFoundError(f"Missing refiner checkpoint: {refiner_ckpt}. Use --allow-random-init-for-smoke-test only for code-path validation.")
    refiner.to(device)
    refiner.eval()

    category_ids = getattr(torch, "full")((num_candidates,), int(category_to_id[args.category]), device=device, dtype=getattr(torch, "long"))
    with getattr(torch, "no_grad")():
        planner_outputs = planner(
            category_ids,
            z_ids=None,
            sample_mode="sample",
            z_temperature=float(sampling_cf["z_temperature"]),
            z_top_p=float(sampling_cf["z_top_p"]),
        )
        c5 = planner_outputs["c5_logits"].argmax(dim=1)
        o5 = getattr(torch, "sigmoid")(planner_outputs["o5_logits"].squeeze(1)) >= 0.5
        r17 = planner_outputs["r17_pred"]
        fg_ratio = planner_outputs["fg_ratio_pred"]
        y20, refiner_score, mean_confidence = _iterative_refine(
            refiner,
            category_ids=category_ids,
            z_ids=planner_outputs["z_ids"],
            c5=c5,
            o5=o5.to(dtype=getattr(torch, "float32")),
            r17=r17,
            fg_ratio=fg_ratio,
            steps=int(sampling_cf["refinement_steps"]),
            sample_temperature=float(config["train_refiner"]["sample_choice_temperature"]),
        )

    rows: list[dict[str, object]] = []
    for index in range(num_candidates):
        sample_dir = output_dir / f"candidate_{index:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        y20_list = y20[index].detach().cpu().tolist()
        c5_list = c5[index].detach().cpu().tolist()
        o5_list = [[int(value) for value in row] for row in o5[index].detach().cpu().tolist()]
        save_label_map(y20_list, sample_dir / "y20.png", scale=12)
        save_label_map(c5_list, sample_dir / "c5.png", scale=32)
        save_binary_map(o5_list, sample_dir / "o5.png", scale=32)
        metrics = structure_metrics(y20_list, y20_list, background_class_id=int(data_cf["background_class_id"]), num_classes=int(data_cf["num_classes"]))
        row = {
            "candidate_id": index,
            "category": args.category,
            "z_id": int(planner_outputs["z_ids"][index].item()),
            "planner_score": float(planner_outputs["z_logprob"][index].item()),
            "refiner_score": float(refiner_score[index].item()),
            "mean_confidence": float(mean_confidence[index].item()),
            "fg_ratio": float(fg_ratio[index].item()),
            "r17": [float(value) for value in r17[index].detach().cpu().tolist()],
            **metrics,
            "sample_dir": str(sample_dir),
            "y20": y20_list,
            "c5": c5_list,
            "o5": o5_list,
        }
        (sample_dir / "meta.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append(row)
        print_progress("sample", index + 1, num_candidates, f"z={row['z_id']} planner={row['planner_score']:.4f} refiner={row['refiner_score']:.4f} conf={row['mean_confidence']:.4f}")
    finish_progress()
    payload = {
        "category": args.category,
        "num_candidates": num_candidates,
        "samples": rows,
    }
    (output_dir / "candidates.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(format_metric_line("saved candidates:", [("output", str(output_dir)), ("num_candidates", num_candidates)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
