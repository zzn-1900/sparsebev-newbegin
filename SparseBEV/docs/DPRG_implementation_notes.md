# Dual-Path Reliability Gating (DPRG) 实现说明

> **文件**：`models/dual_path_gating.py`（新增）、`models/sparsebev_transformer.py`（修改）、`configs/r50_nuimg_704x256.py`（修改）
> **基准模型**：SparseBEV (ICCV 2023)，nuScenes 3D 目标检测
> **版本**：v2 双路并行 gating（非 v1 线性插值 anchor）

---

## 目录

1. [背景：原 AdaptiveMixing 的问题](#1-背景原-adaptivemixing-的问题)
2. [改进思路](#2-改进思路)
3. [模型整体数据流与插入位置](#3-模型整体数据流与插入位置)
4. [AdaptiveMixing 内部数据流（含 DPRG）](#4-adaptivemixing-内部数据流含-dprg)
5. [DPRG 模块详细计算过程](#5-dprg-模块详细计算过程)
   - [5.1 分组重塑：进入 DPRG 前](#51-分组重塑进入-dprg-前)
   - [5.2 Step 1：Query-Guided Pooling](#52-step-1query-guided-pooling)
   - [5.3 Step 2：当前帧质量估计 → β](#53-step-2当前帧质量估计--β)
   - [5.4 Step 3：双路并行 Gating](#54-step-3双路并行-gating)
   - [5.5 分组重塑：退出 DPRG 后](#55-分组重塑退出-dprg-后)
6. [参数量统计](#6-参数量统计)
7. [初始化策略](#7-初始化策略)
8. [超参配置](#8-超参配置)
9. [训练 Warmup（可选）](#9-训练-warmup可选)
10. [符号速查表](#10-符号速查表)

---

## 1. 背景：原 AdaptiveMixing 的问题

SparseBEV 的 Adaptive Mixing 模块分两步：

1. **Channel Mixing**：用 query 生成权重矩阵 $M \in \mathbb{R}^{C_\text{eff} \times C_\text{eff}}$，对每个采样点做通道变换
2. **Point Mixing**：用 query 生成权重矩阵 $S \in \mathbb{R}^{P_\text{out} \times P_\text{in}}$，对所有采样点做空间聚合

**核心缺陷**：两组权重矩阵 $M$ 和 $S$ **完全由 query feature 决定**，与采样点本身的实际特征内容无关。

这导致：
- 即使采样点命中遮挡、背景或图像边缘，其混合权重不受影响
- 时序融合中远帧（误差累积大）和近帧（可靠）被一视同仁
- 当前帧（最可靠的时序锚点）的信息完全被 query 的全局语义淹没

---

## 2. 改进思路

在 Channel Mixing 和 Point Mixing **之间**插入 **Dual-Path Reliability Gating（DPRG）**，让采样点特征本身参与到"哪些点应该被信任"的决策中。

**核心设计**：

- 以当前帧（时间差为 0，最可靠帧）的采样点聚合出 $\mathbf{f}_\text{cur}$，作为 "现实锚点"
- 估计当前帧的可信度 $\beta \in [0,1]$（遮挡时 $\beta \to 0$，场景清晰时 $\beta \to 1$）
- **双路并行 gating**：
  - Path Q：让 query 对所有 128 个点打可靠性分数 $g^{(q)}$（全局语义路）
  - Path C：让 $\mathbf{f}_\text{cur}$ 对所有 128 个点打可靠性分数 $g^{(c)}$（当前帧外观路）
  - **在标量空间**用 $\beta$ 融合：$g = (1-\beta) g^{(q)} + \beta g^{(c)}$

**为什么不做特征层面的线性插值（v1 方案）**：若直接计算 $a = (1-\beta)q + \beta f_\text{cur}$，当 $q$ 和 $f_\text{cur}$ 在某些维度上符号相反，两个特征会互相抵消（高维对冲），导致不可恢复的信息损失。双路设计在标量空间融合两个 $[0,1]$ 之间的 gate 分数，无对冲问题。

---

## 3. 模型整体数据流与插入位置

以下是一个 Decoder Layer 的完整前向流程（`SparseBEVTransformerDecoderLayer.forward`），标注了 DPRG 的插入位置：

```
输入：
  query_bbox:  [B, Q, 10]    — 目标框坐标（编码后）
  query_feat:  [B, Q, 256]   — Query 特征向量
  mlvl_feats:  4个 FPN 层    — 多帧多视角图像特征

                         ┌──────────────────────────────────┐
                         │  Position Encoding               │
                         │  Linear(3→256) × 2 + LN + ReLU  │
                         │  query_feat += pos_embed         │
                         └──────────────┬───────────────────┘
                                        ▼ [B, Q, 256]
                         ┌──────────────────────────────────┐
                         │  Self-Attention (SASA)           │
                         │  Multi-head Attn, 8 heads        │
                         │  + LN (norm1)                    │
                         └──────────────┬───────────────────┘
                                        ▼ [B, Q, 256]
                         ┌──────────────────────────────────┐
                         │  SparseBEVSampling               │
                         │  3D 点预测 → 图像采样             │
                         │  msmv_sampling(多尺度多视角)      │
                         └──────────────┬───────────────────┘
                                        ▼ [B, Q, G=4, T×P=32, 64]
                         ┌══════════════════════════════════╗
                         ║  AdaptiveMixing                  ║
                         ║  ┌────────────────────────────┐  ║
                         ║  │ Channel Mixing             │  ║
                         ║  │ matmul(feat, M)            │  ║
                         ║  └────────────┬───────────────┘  ║
                         ║               ▼ [B*Q, G, 32, 64] ║
                         ║  ┌────────────────────────────┐  ║
                         ║  │ ★ DPRG (新增)             │  ║   ← 本改进的插入位置
                         ║  │ reliability gating         │  ║
                         ║  └────────────┬───────────────┘  ║
                         ║               ▼ [B*Q, G, 32, 64] ║
                         ║  ┌────────────────────────────┐  ║
                         ║  │ Point Mixing               │  ║
                         ║  │ matmul(S, feat)            │  ║
                         ║  └────────────┬───────────────┘  ║
                         ╚═══════════════╪══════════════════╝
                                        ▼ [B, Q, 256]
                         ┌──────────────────────────────────┐
                         │  FFN + LN (norm2, norm3)         │
                         └──────────────┬───────────────────┘
                                        ▼ [B, Q, 256]
                         cls_score [B, Q, 10]，bbox_pred [B, Q, 10]
```

**插入位置的具体代码行**（`sparsebev_transformer.py`，`AdaptiveMixing.inner_forward`）：

```python
'''adaptive channel mixing'''
out = torch.matmul(out, M)          # ← channel mixing 结束
out = F.layer_norm(out, [...])
out = self.act(out)

'''dual-path reliability gating'''  # ← DPRG 插在此处
if self.use_dprg:
    ...

'''adaptive point mixing'''         # ← point mixing 开始
out = torch.matmul(S, out)
```

**为什么插在这里**：Channel mixing 已将原始视觉特征"翻译"到 query 对齐的语义空间，在这个空间里计算特征间的相似度（cosine similarity、dot product）语义更准确，DPRG 的 gating 决策质量更高。

---

## 4. AdaptiveMixing 内部数据流（含 DPRG）

### 超参与尺寸（默认配置，nuScenes r50_nuimg_704x256）

| 符号 | 含义 | 值 |
|------|------|----|
| B | batch size | 1（训练时）或 1（推理时）|
| Q | num_query | 900 |
| G | n_groups | 4 |
| T | num_frames | 8 |
| P_g | num_points（每组每帧） | 4 |
| P | in_points = T × P_g | 32 |
| C | embed_dims（全通道） | 256 |
| C_eff | eff_in_dim = C / G | 64 |
| P_out | out_points | 128 |
| r | gate_rank（DPRG 低秩维度） | 16 |
| S | current-frame 点数 = P_g | 4 |
| B' | B × Q × G（DPRG 展开 batch） | B×900×4 |

### 逐步张量变换

```
x (采样特征，来自 SparseBEVSampling)
  形状: [B, Q, G, P, C_eff] = [B, 900, 4, 32, 64]

──── reshape ────────────────────────────────────────────────────
out = x.reshape(B*Q, G, P, C)
  形状: [B*Q, 4, 32, 64]   （展开 batch × query 维）

──── 生成混合参数 ─────────────────────────────────────────────────
params = parameter_generator(query)
  Linear(256 → 4 × (64×64 + 32×128)) = Linear(256 → 32768)
  query 形状: [B, Q, 256]
  params 形状: [B*Q, 4, 8192]  (reshape 后)

  分裂:
    M (channel weights): [B*Q, 4, 64, 64]   ← 64×64=4096 per group
    S (point weights):   [B*Q, 4, 128, 32]  ← 128×32=4096 per group

──── Channel Mixing ──────────────────────────────────────────────
out = torch.matmul(out, M)
  [B*Q, 4, 32, 64] × [B*Q, 4, 64, 64] → [B*Q, 4, 32, 64]
  （对每个采样点做 query-conditioned 线性变换，作用在通道维）

out = F.layer_norm(out, [32, 64])
  对最后两维 (P, C_eff) 做 LayerNorm
  形状不变: [B*Q, 4, 32, 64]

out = ReLU(out)
  形状不变: [B*Q, 4, 32, 64]

══════════════════════════════════════════════════════
★ DPRG 插入点（详见第 5 节）
  输入:  out    形状 [B*Q, 4, 32, 64]
         query  形状 [B, Q, 256]
  输出:  out    形状 [B*Q, 4, 32, 64]（形状不变，值被 gate 调制）
══════════════════════════════════════════════════════

──── Point Mixing ────────────────────────────────────────────────
out = torch.matmul(S, out)
  [B*Q, 4, 128, 32] × [B*Q, 4, 32, 64] → [B*Q, 4, 128, 64]
  （将 32 个输入点混合聚合为 128 个输出点）

out = F.layer_norm(out, [128, 64])
  形状不变: [B*Q, 4, 128, 64]

out = ReLU(out)
  形状不变: [B*Q, 4, 128, 64]

──── 输出投影 ─────────────────────────────────────────────────────
out = out.reshape(B, Q, 4*128*64)
  形状: [B, Q, 32768]

out = out_proj(out)
  Linear(32768 → 256)
  形状: [B, Q, 256]

out = query + out   (残差连接)
  形状: [B, Q, 256]
```

---

## 5. DPRG 模块详细计算过程

DPRG（`DualPathReliabilityGating`）以**每组独立**的方式运行，即把 G=4 个 group 和 B×Q 的 batch 全部展开到一个"超 batch" $B' = B \times Q \times G$ 上并行处理。

### 5.1 分组重塑：进入 DPRG 前

```python
# 在 AdaptiveMixing.inner_forward 中：
# out 此时形状: [B*Q, G=4, P=32, C_eff=64]
# query 此时形状: [B, Q, 256]

query_grouped = query.reshape(B * Q * G, self.eff_in_dim)
#   [B, Q, 256] → [B, Q, 4, 64] → [B*Q*4, 64]
#   原理: query_dim(256) = G(4) × eff_in_dim(64)，按组等分

out_flat = out.reshape(B * Q * G, P, C)
#   [B*Q, 4, 32, 64] → [B*Q*4, 32, 64]

# 进入 DPRG.forward(query=query_grouped, mixed_feats=out_flat)
# B' = B*Q*G，后续所有计算在此展开 batch 上独立进行
```

### 5.2 Step 1：Query-Guided Pooling

**目的**：从当前帧（frame 0）的 S=4 个采样点中，以 query 为 attention 权重，聚合出代表当前帧外观的向量 $\mathbf{f}_\text{cur}$。

```
输入张量:
  query:       [B', 64]       B' = B×Q×G
  mixed_feats: [B', 32, 64]

① 切取当前帧点（current_frame_idx=0，前 S=4 个点）
   cur_feats = mixed_feats[:, 0:4, :]
   形状: [B', 4, 64]

   [当前帧是 frame index 0 的原因：
    time_diff = timestamps[:, 0:1, :] - timestamps
    → time_diff[:, 0] = 0，即第 0 帧无时间偏移，确认为当前帧]

② 投影（Key）
   cur_proj = W_a(cur_feats)
   算子: Linear(64→64, bias=False)  权重 W_a ∈ [64, 64]
   形状: [B', 4, 64]

③ Attention logits（点积相似度）
   attn_logits = (query.unsqueeze(1) * cur_proj).sum(-1) * scale
   ─ query.unsqueeze(1): [B', 1, 64] → broadcast 到 [B', 4, 64]
   ─ element-wise 乘积后沿 C 维求和: [B', 4]
   ─ scale = 1/√64 = 0.125（防止 softmax 梯度消失）
   形状: [B', 4]

④ Softmax → 注意力权重
   alpha = softmax(attn_logits, dim=-1).unsqueeze(-1)
   形状: [B', 4, 1]

⑤ 加权聚合（用原始特征，非投影特征，类比 Attention 中的 V）
   f_cur = (alpha * cur_feats).sum(dim=1)
   ─ [B', 4, 1] * [B', 4, 64] → [B', 4, 64]
   ─ .sum(dim=1) → [B', 64]
   形状: [B', 64]
```

### 5.3 Step 2：当前帧质量估计 → β

**目的**：输出标量 $\beta \in [0,1]$，编码"当前帧特征有多可靠"。

**信号 1：内部一致性 $c_1$**

衡量当前帧 4 个采样点的语义凝聚程度（一致性高说明没采到乱七八糟的背景）：

```
⑥ L2 归一化
   cur_normed = F.normalize(cur_feats, dim=-1, eps=1e-6)
   算子: 向量除以其 L2 范数
   形状: [B', 4, 64]

⑦ 计算各点与均值的余弦相似度
   cur_mean = cur_normed.mean(dim=1, keepdim=True)
   形状: [B', 1, 64]

   c1 = (cur_normed * cur_mean).sum(-1).mean(dim=-1, keepdim=True)
   ─ element-wise 乘积: [B', 4, 64]
   ─ .sum(-1): [B', 4]         ← 每个点与均值的 dot product（即余弦相似度，因均值未归一化）
   ─ .mean(dim=-1, keepdim=True): [B', 1]
   形状: c1 ∈ [B', 1]，值域 ≈ [-1, 1]
```

**信号 2：Query-Current 对齐度 $c_2$**

衡量聚合出的 $\mathbf{f}_\text{cur}$ 与 query 在语义空间的吻合程度：

```
⑧ 余弦相似度
   c2 = F.cosine_similarity(query, f_cur, dim=-1, eps=1e-6).unsqueeze(-1)
   算子: F.cosine_similarity，沿 C 维计算，数值归一化到 [-1,1]
   形状: c2 ∈ [B', 1]
```

**β 预测**：

```
⑨ 拼接两个信号
   beta_input = torch.cat([c1, c2], dim=-1)
   形状: [B', 2]

⑩ 通过 MLP 预测 β
   beta = sigmoid(beta_mlp(beta_input))

   beta_mlp 结构:
     Linear(2 → 16) + ReLU + Linear(16 → 1)
     ─ Linear(2→16): 权重 [16,2]，偏置 [16]，共 48 个参数
     ─ ReLU
     ─ Linear(16→1): 权重 [1,16]，偏置 [1]，共 17 个参数（零初始化）
   
   中间: [B', 2] → [B', 16] → [B', 1]
   sigmoid 后: beta ∈ [B', 1]，值域 (0, 1)

   [初始化说明：最后一层 Linear(16→1) 权重和 bias 均零初始化，
    因此训练初期 beta_mlp 输出 = 0，β = sigmoid(0) = 0.5，
    两路 gate 各占 50% 比重，让模型从均衡状态自由学习。]
```

### 5.4 Step 3：双路并行 Gating

**目的**：让 query 和 $\mathbf{f}_\text{cur}$ 各自独立地对所有 P=32 个采样点打可靠性分数（gate），然后在**标量空间**用 β 加权融合，最终调制采样点特征。

**低秩投影**（W1、W2 共享，仅计算一次 W2）：

```
⑪ W2 投影所有采样点（shared，两路复用）
   v = W2(mixed_feats)
   算子: Linear(64 → 16, bias=False)  权重 W2 ∈ [16, 64]
   形状: [B', 32, 16]

⑫ Path Q：query-driven gate
   u_q = W1(query).unsqueeze(1)
   算子: Linear(64 → 16, bias=False)  权重 W1 ∈ [16, 64]（与 Path C 共享）
   形状: [B', 1, 16]

   gate_logits_q = (u_q * v).sum(-1) * scale + gate_bias
   ─ [B', 1, 16] * [B', 32, 16] → broadcast → [B', 32, 16]
   ─ .sum(-1): [B', 32]           ← 每个点的 dot product
   ─ * scale(=1/√64=0.125): [B', 32]
   ─ + gate_bias(标量，初始化为 1.0): [B', 32]
   
   gate_q = sigmoid(gate_logits_q)
   形状: [B', 32]，值域 (0, 1)
   [初期：输入约为 1.0，sigmoid(1.0) ≈ 0.73，gate 初始大部分开启]

⑬ Path C：current-frame-driven gate
   u_c = W1(f_cur).unsqueeze(1)
   算子: 与 ⑫ 共享同一个 W1
   形状: [B', 1, 16]

   gate_logits_c = (u_c * v).sum(-1) * scale + gate_bias
   ─ 计算方式同 ⑫，v 复用不重算
   形状: [B', 32]

   gate_c = sigmoid(gate_logits_c)
   形状: [B', 32]，值域 (0, 1)

⑭ 标量层面加权融合
   gate = (1 - beta) * gate_q + beta * gate_c
   ─ beta: [B', 1]，broadcast 到 [B', 32]
   ─ gate_q, gate_c: [B', 32]
   ─ 加权结果仍在 [0,1] 之间（无对冲风险）
   形状: [B', 32]

⑮ 应用 gate（Hadamard 乘积）
   gated_feats = mixed_feats * gate.unsqueeze(-1)
   ─ gate.unsqueeze(-1): [B', 32, 1]，broadcast 到 [B', 32, 64]
   ─ element-wise 乘法：等价于对每个点的 64 维特征乘以同一个标量 gate 值
   形状: [B', 32, 64]
```

### 5.5 分组重塑：退出 DPRG 后

```python
# 退出 DPRG 后，在 AdaptiveMixing.inner_forward 中：
out = out_flat.reshape(B * Q, G, P, C)
#   [B*Q*4, 32, 64] → [B*Q, 4, 32, 64]
# 恢复原始 group 结构，继续送入 Point Mixing
```

---

## 6. 参数量统计

### DPRG 模块参数

| 子模块 | 形状 | 参数量 |
|--------|------|--------|
| `W_a` | Linear(64→64, no bias) | 64 × 64 = **4,096** |
| `beta_mlp[0]` | Linear(2→16, bias) | 2×16 + 16 = **48** |
| `beta_mlp[2]` | Linear(16→1, bias) | 16×1 + 1 = **17** |
| `W1` | Linear(64→16, no bias) | 64 × 16 = **1,024** |
| `W2` | Linear(64→16, no bias) | 64 × 16 = **1,024** |
| `gate_bias` | 标量 Parameter | **1** |
| **DPRG 总计** | | **6,210** |

> DPRG 权重被所有 B×Q×G 组**共享**（不是每组独立一套权重），因此总参数量就是 6,210，仅约 **0.006M**。

### 对比原 AdaptiveMixing 参数量

| 模块 | 参数量 |
|------|--------|
| `parameter_generator` Linear(256→32768) | 256 × 32768 = **8,388,608** |
| `out_proj` Linear(32768→256) | 32768 × 256 = **8,388,608** |
| 原 AdaptiveMixing 总计 | ≈ **16.8M** |
| **新增 DPRG** | **+6,210（+0.04%）** |

### 整体模型参数量影响

DPRG 在所有 `num_layers=6` 个 Decoder Layer 中共享**同一个** `mixing` 实例（参数共享设计），因此全模型仅新增 **6,210 个参数**。

---

## 7. 初始化策略

安全初始化确保**训练初期不破坏 baseline 行为**：

| 参数 | 初始化方式 | 训练初期效果 |
|------|-----------|-------------|
| `W_a.weight` | `normal_(std=0.02)` | 初期 softmax 接近均匀，$\alpha_j \approx 1/S$ |
| `beta_mlp[0].weight` | `xavier_uniform_` | 正常梯度传递 |
| `beta_mlp[0].bias` | `zeros_` | 无偏置 |
| `beta_mlp[2].weight` | `zeros_` | **关键：使初期 $\beta = \sigma(0) = 0.5$** |
| `beta_mlp[2].bias` | `zeros_` | **关键：同上** |
| `W1.weight` | `xavier_uniform_` | 正常尺度 |
| `W2.weight` | `xavier_uniform_` | 正常尺度 |
| `gate_bias` | `ones(1)` → 值为 1.0 | $\sigma(1.0) \approx 0.73$，gate 初期大部分开启 |

**初始化逻辑**：
- `gate_bias=1.0` 确保初期几乎所有点都通过 gate（不会截断信息）
- `beta_mlp` 末层零初始化确保初期两路各占 50%，无偏向
- `W_a` 小初始化使 query-guided pooling 初期近似均值池化，稳定训练

---

## 8. 超参配置

### 配置文件修改（`configs/r50_nuimg_704x256.py`）

```python
transformer=dict(
    type='SparseBEVTransformer',
    embed_dims=256,
    num_frames=8,
    num_points=4,
    num_layers=6,
    num_levels=4,
    num_classes=10,
    code_size=10,
    pc_range=point_cloud_range,
    use_dprg=True,    # ← 新增：开启 DPRG
    gate_rank=16,     # ← 新增：低秩维度（可选 8/16/32）
)
```

### 参数传递链

```
SparseBEVTransformer(use_dprg, gate_rank)
  └─ SparseBEVTransformerDecoder(use_dprg, gate_rank)
       └─ SparseBEVTransformerDecoderLayer(use_dprg, gate_rank)
            └─ AdaptiveMixing(use_dprg=True,
                              num_frames=8,
                              num_points_per_frame=4,
                              gate_rank=16)
                 └─ DualPathReliabilityGating(
                        embed_dim=64,      # = embed_dims / n_groups = 256/4
                        num_frames=8,
                        num_points_per_frame=4,
                        gate_rank=16,
                        current_frame_idx=0,
                    )
```

### `gate_rank` 的选择

| `gate_rank` | DPRG 参数量 | 低秩比例 | 建议场景 |
|-------------|------------|----------|----------|
| 8 | ~4.3K | 8/64=12.5% | 显存极限，快速验证 |
| **16（默认）** | **~6.2K** | **16/64=25%** | **推荐，平衡效果与开销** |
| 32 | ~8.3K | 32/64=50% | 消融实验，探索上界 |

---

## 9. 训练 Warmup（可选）

前 2 个 epoch 强制 $\beta=0$（纯 query-driven gate），让模型先学好基础采样，再逐步引入当前帧分支：

```python
# 训练循环中（如 train.py）
for epoch in range(num_epochs):
    if epoch < 2:
        model.pts_bbox_head.transformer.decoder.decoder_layer.mixing.dprg.beta_force_zero = True
    else:
        model.pts_bbox_head.transformer.decoder.decoder_layer.mixing.dprg.beta_force_zero = False
    
    train_one_epoch(...)
```

`beta_force_zero=True` 时，DPRG 退化为纯 query-driven gate，等价于仅做了一个基于 query 的轻量可靠性过滤，不引入当前帧分支的额外复杂性。

---

## 10. 符号速查表

| 符号 | 含义 | 具体值 |
|------|------|--------|
| $B$ | Batch size | 1 |
| $Q$ | Num queries | 900 |
| $G$ | Num groups (AdaptiveMixing) | 4 |
| $T$ | Num temporal frames | 8 |
| $P_g$ | Points per group per frame | 4 |
| $P$ | in_points = $T \times P_g$ | 32 |
| $P_\text{out}$ | out_points | 128 |
| $C$ | embed_dims | 256 |
| $C_\text{eff}$ | eff_in_dim = $C / G$ | 64 |
| $S$ | Current-frame points per group | 4（= $P_g$） |
| $B'$ | Flattened batch for DPRG = $B \times Q \times G$ | $B \times 900 \times 4$ |
| $r$ | gate_rank | 16 |
| $\beta$ | Current-frame trust weight | scalar in $(0,1)$ |
| $c_1$ | Intra-frame consistency | scalar in $(-1,1)$ |
| $c_2$ | Query-current cosine similarity | scalar in $(-1,1)$ |
| $g^{(q)}$ | Query-driven gate per point | vector in $(0,1)^P$ |
| $g^{(c)}$ | Current-frame-driven gate per point | vector in $(0,1)^P$ |
| $g$ | Final fused gate | vector in $(0,1)^P$ |
| $\mathbf{f}_\text{cur}$ | Query-guided current-frame pooling | $\mathbb{R}^{C_\text{eff}}$ |
| $\mathbf{W}_a$ | Pooling projection (Key) | $\mathbb{R}^{64 \times 64}$ |
| $\mathbf{W}_1$ | Shared anchor projection (low-rank) | $\mathbb{R}^{16 \times 64}$ |
| $\mathbf{W}_2$ | Point projection (low-rank) | $\mathbb{R}^{16 \times 64}$ |
| $b_g$ | Gate bias | scalar, init=1.0 |
