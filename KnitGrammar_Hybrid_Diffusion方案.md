# KnitGrammar-Hybrid Diffusion 方案草案

> 版本：v0.1  
> 目标：暂存为后续实验路线文档，等当前 `struct_foreground_v1 / full_masked` 实验跑完后再决定是否进入实现。

---

## 1. 一句话总结

本方案的核心目标是：

```text
通过色码图 / instruction17 学会纬编花型中不同组织结构的工艺规则，
再把这种可解释的结构语法先验注入高保真生成模型，
最终实现 category-only 的多样化、高保真纬编花型图像生成。
```

完整推理链路为：

```text
category
  -> category-local grammar mode
  -> 20x20 instruction17 structure matrix candidates
  -> math grammar / syntax / parser rerank
  -> selected internal structure condition
  -> diffusion / DiT / SD-style renderer
  -> high-fidelity knitting rendering
```

外部推理输入仍然只有：

```text
category
```

不把以下内容作为推理输入：

```text
instruction
色码图
prototype
pattern-viz
rendering
real
```

它们只用于训练期监督、teacher、评估或候选重排。

---

## 2. 当前实验经验总结

已有实验说明：

1. 任务本质是 one-to-many generation，不是 deterministic `category -> unique structure`。
2. `struct_ar_v1` 的 teacher forcing / deterministic 结构预测定义不适合该问题。
3. `struct_prior_v2` 能改善随机撒点，但仍难以学到稳定局部 grammar。
4. `struct_maskgit_v1` 在 category-only 情况下容易背景塌缩、离散点、无可用 grammar。
5. `struct_latentplan_v1.1` 能显著改善 all-background / all-foreground collapse，但可视化仍偏 foreground blob，说明 validity 高不等于学到 Cable grammar。
6. `struct_foreground_v1` 的 bbox crop / canonicalize 可能破坏原始 20x20 的中心、比例和空间语义。
7. cache 可视化显示真实 Cable foreground 里确实存在结构，不是单纯纯色块。
8. centroid 不是空的，但当前 planner 若没有真正吃到 `centroid_fg_mask_prob` 的 20x20 空间条件，仍然可能生成碎片或无效结构。

因此下一阶段更合理的方向是：

```text
full 20x20 matrix
+ background ignore
+ category-local grammar codebook
+ centroid spatial condition
+ neural generator
+ mathematical grammar diagnostics / rerank
+ optional low-weight syntax regularization
+ high-fidelity diffusion renderer
```

---

## 3. 数据表示：色码图到工艺矩阵

色码图不应当作为普通 RGB 图片学习，而应映射为离散工艺矩阵：

```text
Y ∈ {0, 1, ..., 16}^{20×20}

0     = background / fixed red background
1..16 = knitting structure token / foreground organization class
```

训练 foreground structure 时：

```text
fg_mask = (Y != 0)
Y_fg = Y
Y_fg[fg_mask == False] = IGNORE_INDEX = -100
```

模型 label head 只预测前景组织 token：

```text
fg_label_logits: [B, 16, 20, 20]

真实 label 1 -> logit 0
真实 label 2 -> logit 1
...
真实 label 16 -> logit 15
background 0 -> ignore_index，不参与 foreground CE
```

最终 compose：

```text
canvas = zeros([20,20])  # class 0 background
canvas[generated_fg_mask] = generated_labels_1_to_16
```

输出仍然是完整的 `0..16` instruction17 matrix。

---

## 4. 关键建模原则

### 4.1 保留原始 20x20 坐标

下一版应优先使用：

```text
canonical_mode = full_masked
```

即：

```text
不 bbox crop
不 resize
不重新 canonicalize foreground bbox
保留原始 20x20 空间坐标
只 mask 背景 class 0
```

原因：

- 20x20 本身就是纬编结构图的空间坐标系。
- cable 的中心、比例、纵向连续性、边界关系可能都有语义。
- bbox crop 会改变结构比例，可能把中心结构变成窄条或碎片。

### 4.2 显式 one-to-many latent

不要做：

```text
category -> unique matrix
```

而应做：

```text
category -> category-local z -> matrix candidate
```

其中 `z` 表示同一 category 内不同合法结构模式。

### 4.3 背景固定，foreground grammar 主导

背景 class 0 不参与 foreground label CE。  
背景只在最终 compose 时恢复。

这样可以避免模型把主要能力浪费在学习大面积红色背景上。

---

## 5. Layer 1：Grammar Matrix Encoder

从 `instruction17` 矩阵中提取工艺语法 descriptor。

对每个 20x20 matrix 提取：

```text
foreground area
row projection
col projection
label_hist_16
adjacency_signature
transition_2x2_stats
vertical_continuity
horizontal_continuity
symmetry_score
center_band_score
stripe_score
connected_components
largest_component_ratio
tiny_island_count
bbox_stats
```

这些 descriptor 必须基于 foreground labels 1..16，而不是被背景 0 主导。

---

## 6. Layer 2：Category-Local Grammar Codebook

对每个 category 单独聚类 foreground descriptors：

```text
Cable1 -> z = 0..K_Cable1-1
Cable2 -> z = 0..K_Cable2-1
Mesh   -> z = 0..K_Mesh-1
...
```

每个 mode 保存一个内部 grammar centroid：

```text
centroid_fg_mask_prob      # [20,20], foreground occupancy probability
centroid_fg_mask_bin       # [20,20], thresholded binary mask
centroid_label_hist        # [16]
centroid_row_projection    # [20]
centroid_col_projection    # [20]
centroid_adjacency         # label adjacency stats
centroid_transition_stats  # local 2x2 transition grammar
centroid_bbox_stats
connectedness_stats
num_samples
fallback_used
```

注意：

- centroid 是训练集统计得到的内部结构先验。
- 它不是推理输入 prototype。
- 推理时只输入 category，系统内部根据 category 采样 local_z 并读取对应 centroid。

---

## 7. Layer 3：Neural-Math Hybrid Structure Prior

### 7.1 神经结构生成器

输入：

```text
category_id
local_z
centroid_fg_mask_prob  # 必须作为 20x20 spatial condition 输入
centroid grammar vector
```

输出：

```text
local_z_logits:  [B, Kmax]
fg_mask_logits:  [B, 1, 20, 20]
fg_label_logits: [B, 16, 20, 20]
```

关键要求：

```text
centroid_fg_mask_prob 不能只 flatten 成全局向量；
它必须以 20x20 空间图形式进入 spatial decoder。
```

可行实现：

```text
centroid_fg_mask_prob -> small conv stem -> hidden spatial feature
spatial feature + category/mode embedding + position embedding -> decoder
```

### 7.2 主训练目标

```text
L = CE_fg_labels
  + mask_BCE
  + local_z_CE
  + λ_centroid * BCE(mask_logits, centroid_fg_mask_prob)
  + λ_rowcol * row/col projection loss
  + λ_hist * label histogram loss
  + λ_adj * adjacency loss
  + λ_trans * transition loss
```

推荐起始权重：

```text
centroid_mask_weight = 0.05
rowcol_weight        = 0.05 ~ 0.1
hist_weight          = 0.02 ~ 0.05
adj_weight           = 0.02 ~ 0.05
transition_weight    = 0.02 ~ 0.05
```

注意：

- foreground CE 和 mask BCE 是主损失。
- centroid / rowcol / adjacency / transition 是辅助规则约束。
- syntax loss 初期不进入主训练，优先用于 eval / rerank。

---

## 8. Layer 4：Mathematical Grammar Energy

对任意生成矩阵 `Y_hat` 定义可解释 grammar energy：

```text
E(Y_hat | category, z) =
    w_area    * foreground_area_error
  + w_conn    * connected_component_penalty
  + w_island  * tiny_island_penalty
  + w_rowcol  * row_col_projection_error
  + w_hist    * label_hist_error
  + w_adj     * adjacency_error
  + w_trans   * transition_error
  + w_sym     * symmetry_error
  + w_center  * center_band_error
  + w_syntax  * syntax_violation
  + w_margin  * category_descriptor_margin
```

这些数学指标的作用：

1. 训练过程诊断。
2. 生成候选筛选。
3. 后处理修正。
4. 论文可解释性分析。

初期建议不要把所有 energy 项直接放进训练主 loss，而是先做 diagnostics 和 rerank。

---

## 9. Layer 5：多候选采样与 Rerank

推理时不要只生成一个矩阵。

流程：

```text
for i in 1..N:
    sample local_z
    generate fg_mask + fg_labels
    compose 20x20 matrix
    compute grammar energy
    compute syntax score
    compute descriptor margin
    compute parser_t_inverse consistency
rank candidates
select top-k diverse matrices
```

推荐：

```text
N = 64 或 128
top-k = 4 或 8
```

rerank score：

```text
score =
  - math_energy
  - syntax_violation
  - category_descriptor_margin
  + connectedness_score
  + label_diversity_score
  + parser_t_inverse_consistency
  + diversity_bonus
```

其中：

```text
category_descriptor_margin < 0
```

表示生成矩阵更接近自己的 category 分布。

---

## 10. parser_t_inverse / syntax 的使用位置

已有经验：

```text
parser_t_inverse 的 CE consistency 有价值；
syntax loss 不适合大权重直接压到生成器主损失。
```

推荐三阶段使用：

### Stage 1：Eval / Rerank

```text
syntax_score 只用于候选排序和可解释评估。
```

### Stage 2：低权重辅助

如果确认 syntax 与视觉/结构质量正相关，再加入：

```text
syntax_weight = 0.005 ~ 0.02
```

### Stage 3：局部修正

对 syntax violation 区域做 matrix local repair，而不是让 syntax loss 主导全部训练。

---

## 11. Layer 6：Grammar-Controlled High-Fidelity Renderer

等结构矩阵质量稳定后，再进入 high-fidelity generation。

内部结构矩阵作为条件：

```text
instruction17 one-hot map: 17 channels
foreground mask
category embedding
local_z embedding
grammar descriptor / syntax score optional
```

可选生成器：

```text
latent diffusion + ControlNet-style adapter
DiT / diffusion transformer + structure adapter
SD3 Medium + lightweight condition adapter
```

训练原则：

```text
主 loss = diffusion denoising loss
辅助 = low-weight parser_t_inverse CE consistency
syntax 不直接大权重进入主生成器 loss
```

目标：

```text
structure matrix controls global grammar;
strong generator learns yarn texture, stitch detail, lighting, high-frequency knitting appearance.
```

---

## 12. 最小实验路线

### Experiment A：full_masked matrix cache

目标：验证保留原始 20x20 坐标后，cache / centroid 是否正常。

命令顺序：

```bash
rm -f knit_decode/struct_foreground_v1/cache/foreground_v1/*.pt

python -m knit_decode.struct_foreground_v1.build_foreground_cache \
  --config knit_decode/struct_foreground_v1/configs/foreground_v1.yaml \
  --manifest outputs/manifests/inverse_rendering_train.jsonl \
  --fit-kmeans

python -m knit_decode.struct_foreground_v1.build_foreground_cache \
  --config knit_decode/struct_foreground_v1/configs/foreground_v1.yaml \
  --manifest outputs/manifests/inverse_rendering_val.jsonl \
  --kmeans-source-cache knit_decode/struct_foreground_v1/cache/foreground_v1/foreground_cache_train.pt
```

验收：

```text
fg_y20 shape [20,20]
背景为 -100
前景为 1..16
没有 bbox crop / resize
centroid_fg_mask_prob 保留 20x20 原始坐标结构
```

### Experiment B：centroid spatial condition planner

目标：验证 planner 真正利用 `centroid_fg_mask_prob`。

验收：

```text
Cable1 generated matrix 不再是 blob / 碎片
valid foreground rate 提升
label diversity 合理
largest_component_ratio 合理
category_descriptor_margin < 0
```

### Experiment C：math rerank

目标：生成多个 candidate，用 grammar energy 选择更合理结构。

验收：

```text
top-ranked candidates 比 raw samples 更连贯、更像 category
syntax violation 更低
descriptor margin 更低
connectedness 更好
```

### Experiment D：renderer condition

目标：用筛选后的 matrix 控制高保真图像生成。

验收：

```text
保持 category-specific structure
生成纹理更真实
同一 category 下有多样性
不同 candidates 显著不同但都合法
```

---

## 13. 关键评估指标

结构 prior 指标：

```text
foreground_label_ce
fg_mask_iou
fg_label_acc_on_fg
valid_foreground_rate
empty_foreground_rate
full_foreground_rate
fg_area_low / high rate
label_diversity_on_fg
effective_local_modes
unique_local_z_count
category_descriptor_margin
syntax_violation
num_components
largest_component_ratio
tiny_island_count
center_band_score
vertical_continuity
row/col projection distance
adjacency distance
transition distance
```

渲染指标：

```text
parser_t_inverse CE consistency
perceptual quality
texture realism
structure preservation
candidate diversity
human visual preference
```

---

## 14. 方案创新点

1. **色码图工艺语言化**  
   把纬编色码图转换为 0..16 离散结构矩阵，学习工艺 token 的二维语法。

2. **Category-local grammar codebook**  
   每个 category 有自己的 local grammar modes，而不是全局 latent 混用。

3. **Neural-Math Hybrid prior**  
   神经网络负责 one-to-many 生成，数学规则负责可解释约束、诊断和重排。

4. **Background-ignored foreground grammar learning**  
   背景不参与 foreground CE，避免模型被红色背景主导。

5. **Internal structure control**  
   推理输入仍然只有 category，但系统内部生成 structure matrix 并控制高保真渲染。

6. **Syntax-aware reranking**  
   syntax 不强行主导训练，而是先作为可解释 rerank / repair 工具。

---

## 15. 风险与应对

### 风险 1：生成仍然碎片化

应对：

```text
加入 connected component diagnostics
加入 largest_component_ratio rerank
增加 tiny island penalty
加强 centroid_fg_mask_prob spatial condition
```

### 风险 2：候选合法但不像 category

应对：

```text
使用 category_descriptor_margin
加强 category-local codebook
增加 adjacency / transition energy
```

### 风险 3：syntax loss 导致单调结构

应对：

```text
先用于 rerank
低权重进入训练
保留 diversity bonus
```

### 风险 4：renderer 忽略结构条件

应对：

```text
ControlNet-style adapter
structure consistency loss
parser_t_inverse low-noise CE consistency
```

---

## 16. 当前下一步建议

等当前 `full_masked` 实验完成后，优先检查：

1. `full_masked` cache 是否保留原始空间语义。
2. `centroid_fg_mask_prob` 是否有清晰结构。
3. planner 是否真正吃到 spatial centroid condition。
4. Cable1 / Cable2 matrix samples 是否不再是 blob 或碎片。
5. `category_descriptor_margin` 是否为负。
6. connectedness / label diversity 是否合理。

若这些通过，再进入 math rerank；若 rerank 后结构质量稳定，再接 high-fidelity renderer。

---

## 17. 最终目标形式

最终系统应达到：

```text
输入：category
输出：多个高保真、多样化纬编花型 rendering
内部：每个 rendering 都有可解释 20x20 instruction17 matrix
可解释性：能说明该图的 grammar mode、syntax score、descriptor margin、connectedness、label composition
```

这不是普通图像生成，而是：

```text
category-conditioned craft grammar generation + controllable high-fidelity rendering
```
