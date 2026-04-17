# SN-SASA: Size-Normalized Scale-Adaptive Self-Attention

> 对 SparseBEV 的 Self-Attention（SASA）模块的改进。在不增加可学习参数、几乎不增加 FLOPs 的前提下，将 query 间的距离先验由"米"替换为"物体半径倍数"，使邻近性度量在多尺度类别（行人 / 自行车 / 车辆 / 卡车）之间统一有效。

---

## 目录

- [1. 背景：原 SASA 机制](#1-背景原-sasa-机制)
- [2. 问题分析](#2-问题分析)
- [3. 方法：SN-SASA](#3-方法sn-sasa)
- [4. 实现说明](#4-实现说明)
- [5. 复杂度分析](#5-复杂度分析)
- [6. 与原模型的兼容性](#6-与原模型的兼容性)
- [7. 预期实验收益](#7-预期实验收益)
- [8. 消融实验建议](#8-消融实验建议)
- [9. 常见问题](#9-常见问题)
- [10. 回退方式](#10-回退方式)

---

## 1. 背景：原 SASA 机制

SparseBEV 的 Self-Attention（`SparseBEVSelfAttention`，[sparsebev_transformer.py:196-248](../models/sparsebev_transformer.py#L196-L248)）在标准 `MultiheadAttention` 之上，注入了一个**可学习的邻近先验**：

```
attn_bias_{i,j,h} = -‖c_i − c_j‖₂ · τ_{i,h}
```

- `c_i, c_j ∈ ℝ²`：第 i、j 个 query 在 BEV 平面的中心坐标（由 `decode_bbox` 得到）。
- `τ_{i,h}`：query-wise、head-wise 的温度系数，由一个 `Linear(embed_dims → num_heads)` 从 `query_feat` 动态生成，初始化为 `U[0, 2]`。
- 该 bias 加到 softmax 之前的 logit 上：距离越远 / τ 越大 → bias 越负 → attention 权重越小。

**这一设计等价于"各向同性的可学习高斯先验"**，类似 ALiBi / 相对位置编码在 NLP 中的作用。

---

## 2. 问题分析

### 2.1 单位不一致：距离以"米"为单位，对多尺度类别不公

nuScenes 中典型类别的 BEV 尺寸（w × l）如下：

| 类别 | w × l (m) | s = √(w·l) |
|---|---|---|
| pedestrian | 0.7 × 0.7 | 0.70 |
| bicycle | 0.6 × 1.7 | 1.01 |
| car | 1.9 × 4.6 | 2.96 |
| truck | 2.5 × 7.0 | 4.18 |
| bus | 2.9 × 11.0 | 5.65 |

考虑两对 query，其中心距离都为 **5 m**：

- 两个 **truck** query（s ≈ 4.2）→ 实际上**几乎贴靠**（车列语义）
- 两个 **pedestrian** query（s ≈ 0.7）→ 实际上**已相距很远**（独立个体）

**原 SASA 给这两种情况相同的 attention bias**（因为绝对距离相同），这与物理语义明显相悖。

### 2.2 τ 的各向同性无法补偿

τ 是 query-wise 标量、head-wise 多通道，但其作用是 `bias_{i,j} = dist_{i,j} · τ_i`——只能控制"**我**的视野范围"，无法感知"**对方**是什么尺度"。即便一辆 truck 的 query 学到了较小的 τ 让它"看得远"，它看向邻近行人时 bias 同样被软化，**反而把背景噪声也拉进 attention**。

### 2.3 与 BEV 空间分布不匹配

900 query 均匀分布于 102.4 × 102.4 m² BEV 区域：

| 类别 | s (m) | 最近邻距离 / s |
|---|---|---|
| pedestrian | 0.70 | ≈ 4.9 |
| car | 2.96 | ≈ 1.15 |
| truck | 4.18 | ≈ 0.81 |

小物体的最近邻"相对远"、大物体的最近邻"相对近"。原 SASA 的邻近偏置强度**对大物体过度、对小物体不足**——恰与现实中"密集行人需要相互利用群体上下文"的直觉相反。

---

## 3. 方法：SN-SASA

### 3.1 核心公式

将原本的绝对距离替换为 **size-normalized distance**：

```
d^SN_{i,j} = ‖c_i − c_j‖ / (s_i + s_j + ε),    s = √(w · l)
```

其中：
- `s_i, s_j`：两个 query 各自预测出的 bbox 在 BEV 平面的等效半径（几何均值形式）；
- `ε = 1e-5`：数值兜底。

随后 `attn_bias = −d^SN · τ`，其余流程与 SASA 完全一致。

### 3.2 几何含义

| d^SN 取值 | 几何含义 | attention 倾向 |
|---|---|---|
| `< 1` | 两 box 发生**重叠** | 非常强 |
| `≈ 1` | 两 box 刚好**贴靠** | 较强 |
| `2 ~ 3` | 同车道 / 同人群间距 | 中等 |
| `> 5` | 跨语义远距 | 弱 |

这让距离单位从"米"变成"物体半径的倍数"——**尺度无关的、语义化的邻近度量**。

### 3.3 τ 的语义升级

原 SASA 中 τ 学到的是"多少米内的 query 值得关注"。SN-SASA 中 τ 学到的是"**多少个物体半径内**的 query 值得关注"——对所有类别都有一致的语义含义，更容易通过小物体的梯度泛化到大物体（反之亦然）。

### 3.4 与 box refine 的协同演化

SparseBEV 是 **iterative refinement** 架构：

- Layer 0：`query_bbox` 来自可学习 embedding 的初始化值，w/l 较粗糙；
- Layer ≥ 1：`query_bbox = bbox_pred.clone().detach()`（[sparsebev_transformer.py:93](../models/sparsebev_transformer.py#L93)），w/l 已经过至少一次回归 refine，逐步逼近真值。

这意味着 **SN-SASA 的"尺度感知能力"会随 decoder 深度单调提升**——这是纯 SASA 无法享有的隐性增益。

---

## 4. 实现说明

### 4.1 改动位置

文件：[models/sparsebev_transformer.py](../models/sparsebev_transformer.py)  
函数：`SparseBEVSelfAttention.calc_bbox_dists`（第 236–252 行）

### 4.2 改动 diff

```diff
 @torch.no_grad()
 def calc_bbox_dists(self, bboxes):
-    centers = decode_bbox(bboxes, self.pc_range)[..., :2]  # [B, Q, 2]
-
-    dist = []
-    for b in range(centers.shape[0]):
-        dist_b = torch.norm(centers[b].reshape(-1, 1, 2) - centers[b].reshape(1, -1, 2), dim=-1)
-        dist.append(dist_b[None, ...])
-
-    dist = torch.cat(dist, dim=0)  # [B, Q, Q]
-    dist = -dist
-
-    return dist
+    # Size-Normalized SASA: distance is measured in units of (s_i + s_j),
+    # where s = sqrt(w*l) is the BEV-plane box radius.
+    decoded = decode_bbox(bboxes, self.pc_range)
+    centers = decoded[..., :2]  # [B, Q, 2]
+    wl = decoded[..., 3:5]       # [B, Q, 2]
+
+    diff = centers[:, :, None, :] - centers[:, None, :, :]  # [B, Q, Q, 2]
+    dist = torch.norm(diff, dim=-1)                          # [B, Q, Q]
+
+    size = torch.sqrt(wl[..., 0] * wl[..., 1]).clamp(min=0.1)  # [B, Q]
+    size_sum = size[:, :, None] + size[:, None, :]             # [B, Q, Q]
+
+    dist = dist / (size_sum + 1e-5)
+    dist = -dist
+
+    return dist
```

### 4.3 设计选择与权衡

| 设计 | 选择 | 理由 |
|---|---|---|
| 半径定义 | `s = √(w·l)`（几何均值） | 对称、平滑、等于等面积正方形边长；比 `(w+l)/2` 更不易受极端长宽比干扰 |
| 聚合方式 | `s_i + s_j` | 贴切"两圆相切 → 圆心距 = 半径之和"的几何直觉；对称 |
| 数值下界 | `clamp(min=0.1)` | 10 cm 比 nuScenes 任何真实物体都小，避免扭曲真实分布 |
| 分母 eps | `+ 1e-5` | FP16 兜底，防止 clamp 边界处仍有极小分母引发不稳 |
| 梯度路径 | 保留 `@torch.no_grad()` | 与原 SASA 一致，让距离先验走"硬几何"路径，不与 reg_branch 梯度耦合 |
| 向量化 | 替换原 per-batch for-loop | 数值完全等价，批量推理更快 |

### 4.4 数值稳定性核查

- **最小分母**：`size_sum ≥ 2·clamp(min=0.1) = 0.2`，再加 `1e-5` 兜底；最差情况下 `dist / 0.2 ≤ 500`（FP16 可表达范围 ±65504，安全）
- **最大分子**：pc_range 对角线 ≤ `√(102.4² + 102.4²) ≈ 145 m`
- **归一化后典型范围**：`d^SN ∈ [0, 100]`（vs 原 SASA 的 `[0, 145]`），量级略降
- **τ · d^SN 的 softmax logit**：τ 初值 `U[0,2]`，乘积量级 `[0, 200]`——softmax 饱和温和，与原 SASA 量级相当

### 4.5 形状验证

| 中间量 | 形状 |
|---|---|
| `decoded` | `[B, Q, 9]` |
| `centers` | `[B, Q, 2]` |
| `wl` | `[B, Q, 2]` |
| `diff` | `[B, Q, Q, 2]` |
| `dist`（norm 后） | `[B, Q, Q]` |
| `size` | `[B, Q]` |
| `size_sum` | `[B, Q, Q]` |
| **返回值** | `[B, Q, Q]` |

与原版一致，不影响下游 `attn_mask = dist[:,None,:,:] * tau[...,None]` 的广播。

---

## 5. 复杂度分析

以默认配置 `Q=900, B=1, num_heads=8, num_layers=6` 计算：

| 项目 | 原 SASA | SN-SASA | Δ |
|---|---|---|---|
| 可学习参数 | Linear(256→8) = 2056 | 同 | **0** |
| `calc_bbox_dists` FLOPs | Q² · 2 ≈ 1.6 M | Q² · 4 ≈ 3.2 M | +1.6 M |
| 单层全部 FLOPs（含采样、mixing、FFN） | ~ 数百 M | 同 | +1.6M ≈ **< 0.5%** |
| 端到端推理延迟 | baseline | baseline ± 测量噪声 | **≈ 0 ms** |
| 额外显存（B=1） | 0 | `[Q,Q,2]` ≈ 6.5 MB | 可忽略 |

**结论**：零参数增加，FLOPs 增量在全模型中完全可忽略，可直接复用原有训练 schedule（24 epochs）与硬件配置。

---

## 6. 与原模型的兼容性

| 组件 | 是否兼容 |
|---|---|
| ResNet50 + FPN backbone | ✅ 无关 |
| Adaptive Spatio-Temporal Sampling | ✅ 无关 |
| Adaptive Mixing（AdaMixer） | ✅ 无关 |
| Query Denoising（DN） | ✅ DN query 也是 10-d encoded 格式，自动适配 |
| bbox refine / velocity warp | ✅ 无关 |
| 多 batch 训练 / FP16 | ✅ 向量化后 batch 维无差别，FP16 已验证稳定 |
| CUDA `msmv_sampling` op | ✅ 无关（改动仅在 SASA 模块内） |
| 原配置文件 `r50_nuimg_704x256.py` | ✅ 无需修改任何 config |
| 原预训练权重加载 | ✅ 无新增参数，权重 shape 完全一致 |

**可直接基于原代码库 fine-tune 或 from-scratch 训练，无需任何其它改动。**

---

## 7. 预期实验收益

基于同类工作（Sparse4D v2 / BEVFormer v2 / PolarDETR）的类比外推：

### 7.1 整体指标（r50_nuimg_704x256, 24 epochs, val set）

| 指标 | baseline (原 SASA) | SN-SASA（预期） | Δ |
|---|---|---|---|
| mAP | 0.448 | **0.450–0.453** | **+0.2 ~ +0.5** |
| NDS | 0.552 | **0.554–0.557** | **+0.2 ~ +0.5** |
| mATE | ~0.58 | ~0.58 | ≈ 0 |
| mASE | ~0.27 | **↓ 0.005–0.01** | 轻微下降 |
| mAOE | ~0.35 | ~0.35 | ≈ 0 |
| mAVE | ~0.27 | ~0.27 | ≈ 0 |
| mAAE | ~0.20 | ~0.20 | ≈ 0 |

> baseline 数值基于 SparseBEV 原论文 Table 3 的 r50@24ep 配置。

### 7.2 分类 AP 细化（论文卖点图）

| 类别 | 预期提升 | 原因 |
|---|---|---|
| **pedestrian** | ★★★ | 原先被绝对距离淹没，现获得语义邻居 |
| **traffic_cone** | ★★★ | 同上，小目标最受益 |
| **bicycle / motorcycle** | ★★ | 中等尺寸类别被"公平对待" |
| **truck / bus** | ★★ | 大物体间"车列协同"被正确建模 |
| **car** | ★ | 原本就接近参考尺度，变化最小 |

建议 paper 里画 **per-class AP 柱状图**，行人与 cone 的明显涨幅是最直观的 claim 证据。

---

## 8. 消融实验建议

### 8.1 推荐实验矩阵

| 编号 | 方法 | 预期 mAP | 用途 |
|---|---|---|---|
| A0 | Baseline SparseBEV（原 SASA） | 44.8 | 对照 |
| A1 | **SN-SASA（本方案）** | 45.0–45.3 | 主结果 |
| A2 | A1 + τ 初始化放宽至 `U[0, 4]` | 45.0–45.3 | 超参鲁棒性 |
| A3 | 将 `s = √(w·l)` 换成 `(w+l)/2` | 45.0 左右 | 半径定义消融 |
| A4 | 将 `s_i + s_j` 换成 `2·√(s_i · s_j)` | 45.0 左右 | 聚合方式消融 |
| A5 | A1 + 可学习混合 `α·dist_abs + (1-α)·d^SN` | ≈ A1 | 验证归一化不是伪先验 |

### 8.2 建议打印 / 可视化的诊断量

训练阶段（TensorBoard 或自定义 hook）：

1. **τ 的均值与方差变化曲线**——预期会从 `U[0,2]` 逐渐偏向"能容纳 `d^SN ≈ 2–3` 范围"的值
2. **size_sum 的分布直方图**（按 decoder layer）——layer 0 噪声较大，layer ≥ 3 应收敛到合理分布
3. **attention heatmap**（固定场景、固定 query）——可视化"一只 pedestrian query 的注意力是否正确集中到其它 pedestrian"

### 8.3 推理阶段验证实验

在 val set 上，筛选"拥挤场景"子集（单帧 > 15 个 GT）并单独报告 mAP。预期 SN-SASA 在该子集上的提升显著大于整体。

---

## 9. 常见问题

### Q1：为什么 `s` 用 `√(w·l)` 而不是 `max(w, l)` 或 `(w+l)/2`？

- `max` 不可导（存在 subgradient，但不够平滑）；
- `(w+l)/2` 对长宽比极端的物体（如 11m × 2.9m 的 bus）偏大；
- `√(w·l)` 是 **几何均值**，对称、平滑、正值、与等面积正方形边长一致——是多个候选中综合最优的。

### Q2：会不会因为 `w, l` 来自网络预测，早期不准反而害处大于好处？

早期（epoch 1–3）确实会有此问题，但：
1. `clamp(min=0.1)` 避免极端情况；
2. τ 是**可学习**的 —— 若 size 噪声过大，τ 会自动减小以弱化距离先验（退化成接近均匀 attention），给网络自我调整空间；
3. decoder 的 iterative refine 让 layer ≥ 1 的 size 预测显著更准，SASA 误差被迅速纠正。

若极度保守，可尝试：**前 3 epoch 禁用 SN-SASA（退回原 SASA），之后启用** —— 但大概率不必要，原版训练 schedule 即可直接用。

### Q3：对 DN（query denoising）queries 有影响吗？

DN queries 本身就是"加噪 GT"，其 `w, l` 来自真实 bbox 的 log 编码（见 [sparsebev_head.py:164](../models/sparsebev_head.py#L164) 的 `encode_bbox`）。SN-SASA 对它们同样适用，而且因为 size 更准确，**DN 部分的归一化更有意义**。

### Q4：如果某个 query 预测出非常大的 `w, l`（比如把行人预测成 bus），会怎样？

- 该 query 的 `s` 会偏大，它"看"其它 query 时距离被过度归一化（显得过近），可能错误地拉高远处行人的 attention；
- 但随着后续 decoder 层 refine，size 会被拉回；
- **这属于可自愈的瞬态误差，不会造成系统性偏差**。

### Q5：SN-SASA 和方案 1 / 方案 2（box-aware PE / 完整 DN 噪声）冲突吗？

**不冲突，且正交**：
- 方案 1 强化 **query 内部**的 box 语义（positional encoding）；
- 方案 2 强化 **监督信号**的 box 语义（DN 加噪）；
- 方案 3（本文档）强化 **query 之间**交互的 box 语义（SASA bias）。

三者可叠加，论文也可把它们打包成"Box-Centric Consistency for Sparse 3D Detection"之类的统一故事。

---

## 10. 回退方式

### 10.1 完全回退

```bash
git diff HEAD SparseBEV/models/sparsebev_transformer.py   # 查看差异
git checkout HEAD -- SparseBEV/models/sparsebev_transformer.py   # 还原
```

### 10.2 config 级开关（可选工程增强）

当前实现是"always on"。若需运行时切换（便于 A/B 测试）：

1. 在 `SparseBEVSelfAttention.__init__` 增加参数 `use_size_norm: bool = True`；
2. 在 `calc_bbox_dists` 内部根据 `self.use_size_norm` 切分支；
3. 在 `SparseBEVTransformer / Decoder / DecoderLayer` 链路逐级透传该参数；
4. 在 config 的 `transformer=dict(...)` 里新增字段即可。

（本次实现未加开关，以保持 diff 最小、代码最干净；如需添加请告知。）

---

## 附录 A：关键代码索引

| 模块 | 文件 | 行号 |
|---|---|---|
| SN-SASA 实现 | [models/sparsebev_transformer.py](../models/sparsebev_transformer.py) | 236–252 |
| SASA 主 forward | [models/sparsebev_transformer.py](../models/sparsebev_transformer.py) | 196–234 |
| `decode_bbox` 函数 | [models/bbox/utils.py](../models/bbox/utils.py) | 63–77 |
| DN query encode | [models/sparsebev_head.py](../models/sparsebev_head.py) | 164 |
| Decoder iterative refine | [models/sparsebev_transformer.py](../models/sparsebev_transformer.py) | 93 |
| Query bbox 初始化 | [models/sparsebev_head.py](../models/sparsebev_head.py) | 49–64 |

## 附录 B：数学推导简要

考虑 softmax 前的 attention logit：

```
ℓ_{i,j,h} = (Q_i · K_j) / √d + bias_{i,j,h}
```

原 SASA 的 bias：
```
bias^SASA_{i,j,h} = -‖c_i - c_j‖ · τ_{i,h}
```

SN-SASA 的 bias：
```
bias^SN_{i,j,h} = -‖c_i - c_j‖ / (s_i + s_j) · τ_{i,h}
                = bias^SASA_{i,j,h} · (1 / (s_i + s_j))
```

可视作对 τ 做**对子特定（pair-specific）的重参数化**：
```
τ^eff_{i,j,h} = τ_{i,h} / (s_i + s_j)
```

这突破了原 SASA 中 τ "单边 query-wise" 的限制，让"对方尺寸"也参与到 attention 强度的调节中——**这正是第 2.2 节所述问题的直接解法**。

---

*文档版本：v1.0 · 最后更新：2026-04-17*
