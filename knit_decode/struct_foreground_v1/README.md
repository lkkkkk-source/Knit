# struct_foreground_v1

Foreground-canonical category-only structure prior:

`category -> foreground canonical structure -> compose onto fixed red background -> 20x20 instruction17`

Motivation:

The full-canvas planner learned generic foreground blobs. Since background class `0` is a fixed red background, foreground grammar should be learned in foreground-canonical coordinates. Final `y20` is produced by composing generated foreground onto a fixed red background.

Key constraints:

- inference remains category-only
- no prototype
- instruction17 only for training/eval/cache
- background ignored in main foreground CE
- final output is still `20x20 instruction17`

Minimal commands:

```bash
python -m knit_decode.struct_foreground_v1.build_foreground_cache --config knit_decode/struct_foreground_v1/configs/foreground_v1.yaml --manifest outputs/manifests/inverse_rendering_train.jsonl --fit-kmeans
python -m knit_decode.struct_foreground_v1.build_foreground_cache --config knit_decode/struct_foreground_v1/configs/foreground_v1.yaml --manifest outputs/manifests/inverse_rendering_val.jsonl --kmeans-source-cache knit_decode/struct_foreground_v1/cache/foreground_v1/foreground_cache_train.pt
python -m knit_decode.struct_foreground_v1.train_foreground_planner --config knit_decode/struct_foreground_v1/configs/foreground_v1.yaml
python -m knit_decode.struct_foreground_v1.inspect_foreground_planner --config knit_decode/struct_foreground_v1/configs/foreground_v1.yaml --checkpoint knit_decode/struct_foreground_v1/runs/foreground_planner_v1/checkpoint.pt --category Cable1 --num-samples 64
python -m knit_decode.struct_foreground_v1.sample_foreground_candidates --config knit_decode/struct_foreground_v1/configs/foreground_v1.yaml --checkpoint knit_decode/struct_foreground_v1/runs/foreground_planner_v1/checkpoint.pt --category Cable1 --num-candidates 32
python -m knit_decode.struct_foreground_v1.eval_foreground_structure --config knit_decode/struct_foreground_v1/configs/foreground_v1.yaml --samples-dir knit_decode/struct_foreground_v1/runs/foreground_planner_v1/samples/Cable1
```
