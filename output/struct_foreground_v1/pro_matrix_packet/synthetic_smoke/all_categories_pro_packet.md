# Pro Matrix Packet

## Problem statement
- Y in {0..16}^{20x20}
- 0 is background
- 1..16 are foreground knitting structure tokens
- The target task is category-only generation.
- no prototype / instruction / rendering / pattern-viz / real at inference
- At inference time, do not use prototype, instruction, rendering, pattern-viz, or real inputs.
- instruction17 may be used only for training, supervision, cache statistics, exported descriptors, and evaluation.
- The goal is to infer a generalizable mathematical structure prior across categories.

## Categories included
- CableLike: exported 2 / available 2
- MeshLike: exported 2 / available 2

## Per-category compact summary table
- CableLike: area=0.1862, div=8.0000, dom=0.1611, v=0.1461, h=0.1395, lcr=1.0000
- MeshLike: area=0.2625, div=11.0000, dom=0.1282, v=0.1526, h=0.1684, lcr=1.0000

## Representative matrices
### CableLike
- CableLike_00
```text
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  6  3  4  0 10  7  8  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  3  4  0  7  8  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  4  5  6  9 10  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  5  6  9 10  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  6  3  4  7  8  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  3  4  0  7  8  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  4  5  6  8  9 10  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  5  6  6  7  9 10  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  6  3  4  0 10  7  8  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  3  4  0  7  8  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  4  5  6  8  9 10  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  5  6  0  9 10  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  6  3  4  7  8  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  3  4  0  7  8  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  4  5  6  8  9 10  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
```
- CableLike_01
```text
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  3  4  7  8  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  4  5  6  8  9 10  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  5  6  0  0  9 10  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  6  3  4  6 10  7  8  0  0  0  0  0  0
 0  0  0  0  0  0  0  3  4  0  0  7  8  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  4  5  6  8  9 10  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  5  6  0  9 10  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  6  3  4  7  8  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  3  4  7  8  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  4  5  6  8  9 10  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  5  6  0  9 10  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  6  3  4 10  7  8  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  3  4  0  0  7  8  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  4  5  6  8  9 10  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  5  6  0  9 10  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
```
### MeshLike
- MeshLike_00
```text
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0 13  2  3 13  5  6 13  3  4 13  6  2 13  0  0  0
 0  0  0  0  9 15  0  9  0 15  9  0  0 15  0  0  9  0  0  0
 0  0  0  0 10  0  0 10  0  0 10  0  0 10  0  0 10  0  0  0
 0  0  0  0 11  2  3 11  5  6 11  3  4 11  6  2 11  0  0  0
 0  0  0  0 12  0  0 12  0  0 12  0  0 12  0  0 12  0  0  0
 0  0  0  0 13 15  0 13  0 15 13  0  0 15  0  0 13  0  0  0
 0  0  0  0  9  2  3  9  5  6  9  3  4  9  6  2  9  0  0  0
 0  0  0  0 10  0  0 10  0  0 10  0  0 10  0  0 10  0  0  0
 0  0  0  0 11  0  0 11  0  0 11  0  0 11  0  0 11  0  0  0
 0  0  0  0 12 15  3 12  5 15 12  3  4 15  6  2 12  0  0  0
 0  0  0  0 13  0  0 13  0  0 13  0  0 13  0  0 13  0  0  0
 0  0  0  0  9  0  0  9  0  0  9  0  0  9  0  0  9  0  0  0
 0  0  0  0 10  2  3 10  5  6 10  3  4 10  6  2 10  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
```
- MeshLike_01
```text
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  2  9  4  5  9  2  3  9  5  6  9  3  4  0  0  0
 0  0  0  0  0 15  0  0 10 15  0 10  0 15 10  0  0  0  0  0
 0  0  0  0  0 11  0  0 11  0  0 11  0  0 11  0  0  0  0  0
 0  0  0  0  2 12  4  5 12  2  3 12  5  6 12  3  4  0  0  0
 0  0  0  0  0 13  0  0 13  0  0 13  0  0 13  0  0  0  0  0
 0  0  0  0  0 15  0  0  9 15  0  9  0 15  9  0  0  0  0  0
 0  0  0  0  2 10  4  5 10  2  3 10  5  6 10  3  4  0  0  0
 0  0  0  0  0 11  0  0 11  0  0 11  0  0 11  0  0  0  0  0
 0  0  0  0  0 12  0  0 12  0  0 12  0  0 12  0  0  0  0  0
 0  0  0  0  2 15  4  5 13 15  3 13  5 15 13  3  4  0  0  0
 0  0  0  0  0  9  0  0  9  0  0  9  0  0  9  0  0  0  0  0
 0  0  0  0  0 10  0  0 10  0  0 10  0  0 10  0  0  0  0  0
 0  0  0  0  2 11  4  5 11  2  3 11  5  6 11  3  4  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
```

## Cross-category observations
- The packet intentionally reports objective descriptor summaries only.
- Use cross-category distances, label diversity, transition entropy, continuity, and component ratios to derive a general prior.

## Questions for GPT Pro
Given these cross-category 20x20 instruction17 matrices and descriptors, infer a mathematical structure prior that can guide category-only foreground generation without using prototype, instruction, rendering, pattern-viz, or real inputs at inference.

Please address:
- recommended general prior representation
- category-specific parameters
- exact mathematical definitions
- energy function E(Y | category,z)
- rerank score for generated candidates
- how to estimate the prior from training data
- how to inject or use it in current foreground planner
- what should replace or supplement KMeans local_z
- how to keep inference category-only
- implementation steps in struct_foreground_v1
- ablation plan across categories