# struct_latentplan_v1

`struct_latentplan_v1` implements a category-only one-to-many structure prior:

`category -> sampled global structure mode -> coarse canvas -> instruction17`

This line is explicitly **not**:

- `category -> directly fill 20x20 instruction17`
- `category + prototype -> instruction17`
- deterministic `category -> unique structure`

## Task Definition

- Inference input remains `category` only.
- `z` and the coarse canvas are internally sampled/generated.
- No prototype is used at inference.
- `instruction17` is a training-time structure space and supervision target.
- `parser_t_inverse` is treated as a teacher/eval/rerank tool, not as a direct syntax loss for the main generator.

## Modules

- `build_plan_cache.py`
  - reads instruction17 labels
  - builds coarse occupancy `o5`, dominant canvas `c5`, class ratio `r17`, foreground ratio, bbox/component stats
  - clusters global descriptors with `MiniBatchKMeans`
  - assigns discrete global mode id `z`

- `train_planner.py`
  - trains `P(z | category)` and plan heads for `c5 / o5 / r17 / fg_ratio`

- `train_refiner.py`
  - trains a plan-conditioned masked refiner
  - refiner conditions on `category + z + c5 + o5 + r17 + fg_ratio`

- `sample_candidates.py`
  - category-only candidate generation
  - planner samples `z`
  - planner predicts `c5/o5/r17/fg_ratio`
  - refiner samples `instruction17` candidates

- `rerank_candidates.py`
  - structure-only reranking
  - uses refiner score, count error, foreground ratio error, connectivity, tiny islands, and coarse plan violations

- `eval_structure.py`
  - reports structure-focused metrics rather than token accuracy only

## Early Success Criteria

This first version is meant to test whether explicit global planning improves the main failure modes of `struct_maskgit_v1`.

Early success criteria:

1. all-background rate is clearly lower than the category-only MaskGIT baseline
2. results are no longer just red background plus a few isolated color points
3. foreground forms contiguous regions
4. different sampled `z` values lead to different coarse layouts within the same category
5. class composition is closer to the same-category training distribution
6. reranked top candidates look usable as downstream rendering controls

## Minimal Commands

```bash
python -m knit_decode.struct_latentplan_v1.build_plan_cache --config knit_decode/struct_latentplan_v1/configs/latentplan_v1_5x5_k128.yaml
python -m knit_decode.struct_latentplan_v1.build_plan_cache --config knit_decode/struct_latentplan_v1/configs/latentplan_v1_5x5_k128.yaml --manifest outputs/manifests/inverse_rendering_val.jsonl
python -m knit_decode.struct_latentplan_v1.train_planner --config knit_decode/struct_latentplan_v1/configs/latentplan_v1_5x5_k128.yaml
python -m knit_decode.struct_latentplan_v1.train_refiner --config knit_decode/struct_latentplan_v1/configs/latentplan_v1_5x5_k128.yaml
python -m knit_decode.struct_latentplan_v1.sample_candidates --config knit_decode/struct_latentplan_v1/configs/latentplan_v1_5x5_k128.yaml --category Cable1 --num-candidates 64
python -m knit_decode.struct_latentplan_v1.sample_candidates --config knit_decode/struct_latentplan_v1/configs/latentplan_v1_5x5_k128.yaml --category Cable1 --num-candidates 4 --allow-random-init-for-smoke-test --out-dir /tmp/latentplan_smoke/Cable1
python -m knit_decode.struct_latentplan_v1.rerank_candidates --config knit_decode/struct_latentplan_v1/configs/latentplan_v1_5x5_k128.yaml --samples-dir struct_latentplan_v1/runs/refiner_latentplan_v1_5x5_k128/samples/Cable1
python -m knit_decode.struct_latentplan_v1.eval_structure --config knit_decode/struct_latentplan_v1/configs/latentplan_v1_5x5_k128.yaml --samples-dir struct_latentplan_v1/runs/refiner_latentplan_v1_5x5_k128/samples/Cable1
```

## Notes

- `background_class_id = 0`
- official palette background is red `(255, 0, 16)`
- parser inverse checkpoint is expected at:

`knit_decode/parser_t_inverse/runs/rendering_instruction17_inverse_v1_lr5e4_sw005/checkpoint.pt`

- parser inverse is optional for eval/rerank only and is not wired into the main planner/refiner loss
