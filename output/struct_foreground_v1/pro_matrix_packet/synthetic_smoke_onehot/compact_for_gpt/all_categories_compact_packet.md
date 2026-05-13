# All categories compact packet

## Problem statement
- Y in {0..16}^{20x20}
- 0 is background
- 1..16 are foreground knitting structure tokens
- The target task is category-only generation.
- At inference time, do not use prototype, instruction, rendering, pattern-viz, or real inputs.
- instruction17 may be used only for training, supervision, cache statistics, exported descriptors, and evaluation.
- The goal is to infer a generalizable mathematical structure prior across categories.

## Categories included
- CableLike: 2 samples
- MeshLike: 2 samples

## Per-category compact summary table
- CableLike: foreground_area_ratio_mean=0.1862, label_diversity_mean=8.0000, dominant_label_ratio_mean=0.1611, vertical_continuity_score_mean=0.1461, horizontal_continuity_score_mean=0.1395, largest_component_ratio_mean=1.0000, top_labels=[{'label': 6, 'mean_prob': 0.1610810810810811}, {'label': 4, 'mean_prob': 0.15432432432432433}, {'label': 8, 'mean_prob': 0.14765765765765765}, {'label': 10, 'mean_prob': 0.12756756756756757}, {'label': 7, 'mean_prob': 0.1072972972972973}], top_horizontal_transitions=[{'from': 9, 'to': 10, 'prob': 0.14150943396226415}, {'from': 7, 'to': 8, 'prob': 0.14150943396226415}, {'from': 5, 'to': 6, 'prob': 0.14150943396226415}, {'from': 3, 'to': 4, 'prob': 0.14150943396226415}, {'from': 4, 'to': 5, 'prob': 0.07547169811320754}], top_vertical_transitions=[{'from': 5, 'to': 6, 'prob': 0.11744639376218323}, {'from': 7, 'to': 8, 'prob': 0.11695906432748537}, {'from': 8, 'to': 9, 'prob': 0.09990253411306042}, {'from': 4, 'to': 5, 'prob': 0.09844054580896686}, {'from': 6, 'to': 3, 'prob': 0.08187134502923976}]
- MeshLike: foreground_area_ratio_mean=0.2625, label_diversity_mean=11.0000, dominant_label_ratio_mean=0.1282, vertical_continuity_score_mean=0.1526, horizontal_continuity_score_mean=0.1684, largest_component_ratio_mean=1.0000, top_labels=[{'label': 10, 'mean_prob': 0.12326278499409574}, {'label': 9, 'mean_prob': 0.11867562903079298}, {'label': 11, 'mean_prob': 0.10527750022708693}, {'label': 13, 'mean_prob': 0.09887364883277319}, {'label': 3, 'mean_prob': 0.09537651012807703}], top_horizontal_transitions=[{'from': 3, 'to': 4, 'prob': 0.078125}, {'from': 5, 'to': 6, 'prob': 0.0625}, {'from': 2, 'to': 3, 'prob': 0.0625}, {'from': 6, 'to': 2, 'prob': 0.0390625}, {'from': 4, 'to': 5, 'prob': 0.0390625}], top_vertical_transitions=[{'from': 9, 'to': 10, 'prob': 0.20552884615384615}, {'from': 10, 'to': 11, 'prob': 0.18389423076923078}, {'from': 13, 'to': 9, 'prob': 0.1592548076923077}, {'from': 11, 'to': 12, 'prob': 0.14723557692307693}, {'from': 12, 'to': 13, 'prob': 0.1298076923076923}]

## Representative matrices
### CableLike
#### CableLike_00
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
#### CableLike_01
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
#### MeshLike_00
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
#### MeshLike_01
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
- This section reports only objective exported descriptor summaries.
- Compare label histograms, horizontal and vertical transition probabilities, row and column projections, connected components, 2x2 motifs, continuity scores, and graph descriptors across categories.
- Treat differences as empirical evidence to be modeled; do not assume they are causal without validation.

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