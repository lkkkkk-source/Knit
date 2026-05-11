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
- default `canonical_mode=full_masked`

Minimal commands:

```bash
python -m knit_decode.struct_foreground_v1.build_foreground_cache --config knit_decode/struct_foreground_v1/configs/foreground_v1.yaml --manifest outputs/manifests/inverse_rendering_train.jsonl --fit-kmeans
python -m knit_decode.struct_foreground_v1.build_foreground_cache --config knit_decode/struct_foreground_v1/configs/foreground_v1.yaml --manifest outputs/manifests/inverse_rendering_val.jsonl --kmeans-source-cache knit_decode/struct_foreground_v1/cache/foreground_v1/foreground_cache_train.pt
python -m knit_decode.struct_foreground_v1.train_foreground_planner --config knit_decode/struct_foreground_v1/configs/foreground_v1.yaml
python -m knit_decode.struct_foreground_v1.inspect_foreground_planner --config knit_decode/struct_foreground_v1/configs/foreground_v1.yaml --checkpoint knit_decode/struct_foreground_v1/runs/foreground_planner_v1/checkpoint.pt --category Cable1 --num-samples 64
python -m knit_decode.struct_foreground_v1.sample_foreground_candidates --config knit_decode/struct_foreground_v1/configs/foreground_v1.yaml --checkpoint knit_decode/struct_foreground_v1/runs/foreground_planner_v1/checkpoint.pt --category Cable1 --num-candidates 32
python -m knit_decode.struct_foreground_v1.eval_foreground_structure --config knit_decode/struct_foreground_v1/configs/foreground_v1.yaml --samples-dir knit_decode/struct_foreground_v1/runs/foreground_planner_v1/samples/Cable1
```

Front-only manifest filtering

- Some back-side instruction images contain `back` in the filename, for example `dataset2/instruction/Tuck_042_0_6_back.png`.
- These samples may also use non-standard border/background colors such as green, which are not suitable for the current front-only foreground prior.
- If kept in training, they can pollute `fg_mask`, descriptor statistics, KMeans, centroids, and foreground CE.
- This stage does not attempt front/back joint modeling; it only builds a reproducible front-only manifest.
- Original data files and original manifests are preserved. The filter only creates new `*_frontonly.jsonl` manifests.

```bash
python -m knit_decode.struct_foreground_v1.filter_front_manifest \
  --manifest outputs/manifests/inverse_rendering_train.jsonl \
  --output outputs/manifests/inverse_rendering_train_frontonly.jsonl

python -m knit_decode.struct_foreground_v1.filter_front_manifest \
  --manifest outputs/manifests/inverse_rendering_val.jsonl \
  --output outputs/manifests/inverse_rendering_val_frontonly.jsonl
```

Then rebuild cache from the filtered manifests:

```bash
python -m knit_decode.struct_foreground_v1.build_foreground_cache \
  --config knit_decode/struct_foreground_v1/configs/foreground_v1.yaml \
  --manifest outputs/manifests/inverse_rendering_train_frontonly.jsonl \
  --fit-kmeans

python -m knit_decode.struct_foreground_v1.build_foreground_cache \
  --config knit_decode/struct_foreground_v1/configs/foreground_v1.yaml \
  --manifest outputs/manifests/inverse_rendering_val_frontonly.jsonl \
  --kmeans-source-cache knit_decode/struct_foreground_v1/cache/foreground_v1/foreground_cache_train.pt
```

Inspect foreground cache

```bash
python -m knit_decode.struct_foreground_v1.inspect_foreground_cache \
  --cache knit_decode/struct_foreground_v1/cache/foreground_v1/foreground_cache_train.pt \
  --category Cable1 \
  --num-samples 64

python -m knit_decode.struct_foreground_v1.inspect_foreground_cache \
  --cache knit_decode/struct_foreground_v1/cache/foreground_v1/foreground_cache_train.pt \
  --category Cable2 \
  --num-samples 64
```

Interpretation:

1. If `real_fg_y20` itself is a solid block, the instruction17 foreground labels do not contain enough internal Cable structure.
2. If `real_fg_y20` has structure but `centroid_masks` are still blobs, the descriptor/KMeans/centroid design is the likely problem.
3. If both `real_fg_y20` and centroids have structure but model-generated foreground is fragmented, the planner training/conditioning path is the likely problem.
4. If the foreground mask itself is fragmented, add connected-component or compactness diagnostics before changing the planner.
5. Do not judge only from `composed_y20`; inspect the canonical foreground crop first.
6. `centroid_fg_mask_prob` is the cluster foreground occupancy probability map, while `centroid_fg_mask` is the thresholded binary mask.
7. Planner conditioning continues to use internal centroid statistics; cache inspection should compare both probability and binary centroid masks.

Current spatial-centroid planner requirements:

- centroid cache now includes `centroid_label_prob_16` with shape `[16,20,20]` for each `category/local_z`
- planner consumes `centroid_fg_mask_prob [1,20,20]` and `centroid_label_prob_16 [16,20,20]` as direct spatial conditions
- training adds mask BCE + Dice + soft-IoU + foreground CE + low-weight centroid label MSE
- diagnostics now track empty/full, fg_area_low/high, low_label_diversity, num_components, largest_component_ratio, and tiny_component_count
- old `foreground_cache_train.pt` / `foreground_cache_val.pt` built before `centroid_label_prob_16` are incompatible and must be rebuilt
- old checkpoints built before the spatial-centroid planner update are incompatible and must not be reused

Notes:

- train cache is saved as `foreground_cache_train.pt`
- val cache is saved as `foreground_cache_val.pt`
- val/test cache assignment must use `--kmeans-source-cache` from train cache and never refit KMeans
- full_masked keeps the original `20x20` instruction17 coordinates and ignores background class `0` in foreground CE
- bbox_crop is retained only as a legacy non-default path
- `checkpoint_last.pt` is saved every epoch; best `val_valid_foreground_rate` is saved as `checkpoint.pt`
- `eval_foreground_structure.py` no longer emits placeholder metrics; it now fails fast until paired supervised references are wired in explicitly
