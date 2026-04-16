# DualPathReliabilityGating 模块详解

> 对应文件：`models/dual_path_gating.py`

---

## 符号约定

| 符号 | 含义 | 默认值 |
|------|------|--------|
| $B'$ | 展开后的 batch 大小 $= B \times Q \times G$ | $1 \times 900 \times 4 = 3600$ |
| $C$ | 每组特征维度 `embed_dim` $= 256 / G$ | 64 |
| $D$ | 完整 query 维度 `query_dim` | 256 |
| $S$ | 当前帧每组采样点数 `num_points_per_frame` | 4 |
| $T$ | 时序帧数 `num_frames` | 8 |
| $P$ | 每组总采样点数 $= T \times S$ | 32 |
| $r$ | 低秩投影维度 `gate_rank` | 16 |

---

## 输入与输出

```
输入：
  query       (B', C)     组内 query（完整 query 按 G 等分后的 1/4）
  mixed_feats (B', P, C)  channel mixing 之后的采样点特征
  query_full  (B', D)     完整 query（每组都收到相同的 256 维 query）

输出：
  gated_feats (B', P, C)  逐点乘以可靠性 gate 后的特征，形状不变
```

---

## 可学习参数

| 参数名 | 类型 | 形状 | 参数量 | 用途 |
|--------|------|------|--------|------|
| `W_a` | `Linear(C, C, bias=False)` | $[64, 64]$ | 4,096 | Step 1：当前帧点投影 |
| `beta_mlp[0]` | `Linear(2, 16, bias=True)` | $[16,2]+[16]$ | 48 | Step 2：β 预测第一层 |
| `beta_mlp[2]` | `Linear(16, 1, bias=True)` | $[1,16]+[1]$ | 17 | Step 2：β 预测第二层 |
| `W1_q` | `Linear(D, r, bias=False)` | $[16, 256]$ | 4,096 | Step 3：完整 query 投影 |
| `W1_c` | `Linear(C, r, bias=False)` | $[16, 64]$ | 1,024 | Step 3：f_cur 投影 |
| `W2` | `Linear(C, r, bias=False)` | $[16, 64]$ | 1,024 | Step 3：采样点投影 |
| `gate_bias` | `nn.Parameter` | 标量 | 1 | Step 3：gate 偏置 |
| **合计** | | | **10,306** | |

---

## 前向计算

### 准备：切取当前帧点

```python
cur_feats = mixed_feats[:, 0 : S, :]
```

`current_frame_idx=0`，即每组第 0 帧（时间差为 0 的帧）对应前 $S=4$ 个位置。

```
mixed_feats  (B', P=32, C=64)
                 └── [:, 0:4, :]
cur_feats    (B',  S=4, C=64)
```

---

### Step 1：Query-Guided Pooling → $\mathbf{f}_\text{cur}$

**目的**：从当前帧 $S$ 个点中，以 query 为注意力权重，聚合出单个向量 $\mathbf{f}_\text{cur}$，代表"当前帧与当前 query 最相关的外观"。

```
                cur_feats  (B', 4, 64)
                    │
                    │  W_a: Linear(64→64, no bias)
                    ▼
                cur_proj   (B', 4, 64)
                    │
                    │  element-wise 乘 query.unsqueeze(1)  (B', 1, 64) broadcast
                    │  → .sum(-1)                          沿 C 维求和
                    │  → × scale_attn (= 1/√64 = 0.125)
                    ▼
               attn_logits  (B', 4)
                    │
                    │  F.softmax(dim=-1)
                    ▼
                  alpha    (B', 4, 1)      ← 4 个点的注意力权重，和为 1
                    │
                    │  × cur_feats  (B', 4, 64)  element-wise broadcast
                    │  → .sum(dim=1)             对 S 维加权求和
                    ▼
                  f_cur    (B', 64)
```

**注意**：`W_a` 只投影用于计算 attention logit 的 Key（`cur_proj`），聚合时用的是原始 `cur_feats` 作为 Value。这仿照 Attention 中 K/V 分离的设计，使 $\mathbf{f}_\text{cur}$ 保留原始特征的幅值信息。

---

### Step 2：估计 β

β 是一个标量，取值 $(0, 1)$，含义是"当前帧多可靠"。由两个信号 $c_1$、$c_2$ 经 MLP 预测而来。

#### 信号 $c_1$：帧内一致性

**计算过程**：

```
cur_feats      (B', 4, 64)
    │
    │  F.normalize(dim=-1, eps=1e-6)   对每个点做 L2 归一化
    ▼
cur_normed     (B', 4, 64)    ← 每行是单位向量

    │  .mean(dim=1, keepdim=True)      对 S=4 个点取均值
    ▼
cur_mean       (B', 1, 64)    ← 4 个单位向量的均值（本身不是单位向量）

    │  cur_normed × cur_mean           element-wise，broadcast (B',4,64)
    │  → .sum(-1)                      沿 C 维点积    → (B', 4)
    │  → .mean(dim=-1, keepdim=True)   对 S=4 个点取均值
    ▼
c1             (B', 1)        ← 值域 ≈ (-1, 1)
```

**数学等价**：设 4 个单位向量为 $\hat{f}_1, \ldots, \hat{f}_4$，均值为 $\bar{f}$，则

$$c_1 = \frac{1}{S}\sum_j \hat{f}_j \cdot \bar{f} = \bar{f} \cdot \bar{f} = \|\bar{f}\|^2$$

即 $c_1$ 等于单位向量均值的 **L2 范数的平方**。所有点方向一致时 $c_1 \to 1$；方向分散时 $c_1 \to 0$。

---

#### 信号 $c_2$：query 与 $\mathbf{f}_\text{cur}$ 的对齐度

```
F.cosine_similarity(query, f_cur, dim=-1, eps=1e-6)
    ├── query  (B', 64)    ← 组内 query（256 维按组等分的 1/4）
    └── f_cur  (B', 64)    ← Step 1 聚合的当前帧特征
    ▼
c2  (B', 1)    ← 值域 (-1, 1)
```

$c_2$ 衡量"当前帧聚合特征与 query 的语义方向是否一致"。

---

#### β 预测

```
c1  (B', 1)  ┐
             ├─ torch.cat(dim=-1)
c2  (B', 1)  ┘
             ▼
beta_input   (B', 2)
             │
             │  beta_mlp[0]: Linear(2→16, bias)  → ReLU
             │  beta_mlp[2]: Linear(16→1, bias)
             ▼
             (B', 1)
             │
             │  torch.sigmoid
             ▼
beta         (B', 1)    ← 值域 (0, 1)
```

**初始化保证**：`beta_mlp[2]` 的权重和偏置均**零初始化**，因此训练初期无论 $c_1, c_2$ 取何值，MLP 输出恒为 0，$\beta = \sigma(0) = 0.5$。两路 gate 各占一半，不偏向任何一路。

---

### Step 3：双路并行 Gating

两路分别从不同的"参考向量"出发，对全部 $P=32$ 个采样点打分。

#### 共享的点投影（只算一次）

```
mixed_feats  (B', 32, 64)
    │
    │  W2: Linear(64→16, no bias)    对每个点独立做线性变换
    ▼
v            (B', 32, 16)
```

$v$ 被两路共用，节省一次 $(B' \times P \times C \times r)$ 的矩阵乘法。

---

#### Path Q：完整 query 驱动

```
query_full   (B', 256)
    │
    │  W1_q: Linear(256→16, no bias)
    ▼
             (B', 16)
    │
    │  .unsqueeze(1)
    ▼
u_q          (B',  1, 16)
    │
    │  × v  (B', 32, 16)    element-wise broadcast
    │  → .sum(-1)            沿 r 维内积
    │  → × scale_gate (= 1/√16 = 0.25)
    │  → + gate_bias  (标量, 初始化=1.0)
    ▼
             (B', 32)
    │
    │  torch.sigmoid
    ▼
gate_q       (B', 32)    ← 值域 (0,1)，初期约 0.73
```

每个元素 $g_k^{(q)} = \sigma\!\left(\dfrac{\mathbf{W}_{1q}\,\mathbf{q}_\text{full} \;\cdot\; \mathbf{W}_2\,\mathbf{f}_k}{\sqrt{r}} + b\right)$ 衡量"第 $k$ 个采样点与完整 query 的语义有多接近"。

---

#### Path C：当前帧驱动

```
f_cur        (B', 64)
    │
    │  W1_c: Linear(64→16, no bias)
    ▼
             (B', 16)
    │
    │  .unsqueeze(1)
    ▼
u_c          (B',  1, 16)
    │
    │  × v  (B', 32, 16)    （v 复用，不重算）
    │  → .sum(-1)
    │  → × scale_gate
    │  → + gate_bias
    ▼
             (B', 32)
    │
    │  torch.sigmoid
    ▼
gate_c       (B', 32)    ← 值域 (0,1)
```

每个元素 $g_k^{(c)} = \sigma\!\left(\dfrac{\mathbf{W}_{1c}\,\mathbf{f}_\text{cur} \;\cdot\; \mathbf{W}_2\,\mathbf{f}_k}{\sqrt{r}} + b\right)$ 衡量"第 $k$ 个采样点与当前帧外观有多接近"。

**两路投影维度不同是关键**：`W1_q` 输入 256 维，`W1_c` 输入 64 维，两个矩阵学习目标不同，投影到同一个 $r=16$ 维度量空间后才做 β 融合，保证可比性的同时给 β 真正需要调和的两个互补信号。

---

#### 融合与应用

```
beta    (B',  1)
gate_q  (B', 32)
gate_c  (B', 32)

gate = (1 - beta) * gate_q + beta * gate_c     (B', 32)
    ↑
    标量层面加权：beta broadcast 到 (B', 32)
    两个 [0,1] 值的凸组合，结果仍在 [0,1]

gated_feats = mixed_feats × gate.unsqueeze(-1)  (B', 32, 64)
    ↑
    gate.unsqueeze(-1): (B', 32, 1) broadcast 到 (B', 32, 64)
    每个点的 64 个通道乘以同一个标量 gate 值
```

---

## 完整数据流一览

```
query_full  (B', 256) ──┐
                        │ W1_q(256→16)         ┐
query       (B',  64) ──┤                      │
                        │ Step1: attention      │  Path Q
                        ▼                      │
mixed_feats (B', 32, 64)──────── W_a ──────────┤  gate_q (B', 32)
    │                   ▼                      │
    │               f_cur (B', 64)             │
    │                   │                      │
    │               ┌───┴────┐                 │
    │               c1 (B',1) c2 (B',1)        │
    │               └───┬────┘                 │
    │               beta_mlp                   │
    │                   ▼                      │
    │               beta (B', 1)               │  Path C
    │                   │                      │
    │               W1_c(64→16)               ┘  gate_c (B', 32)
    │                   │
    │               gate = (1-β)·gate_q + β·gate_c    (B', 32)
    │                   │
    └──────── × gate.unsqueeze(-1) ────────────────────────────▶
                                            gated_feats (B', 32, 64)
         ↑
    W2(64→16) 只算一次，两路共用
```

---

## 初始化策略

| 参数 | 初始化 | 训练初期效果 |
|------|--------|------------|
| `W_a` | `normal_(std=0.02)` | attention logit 幅值小，softmax 接近均匀分布 |
| `beta_mlp[0]` weight | `xavier_uniform_` | 正常梯度传递 |
| `beta_mlp[0]` bias | `zeros_` | 无偏置 |
| `beta_mlp[2]` weight | **`zeros_`** | MLP 输出恒 0 → $\beta = 0.5$，两路各半 |
| `beta_mlp[2]` bias | **`zeros_`** | 同上 |
| `W1_q` | `xavier_uniform_` | 正常尺度 |
| `W1_c` | `xavier_uniform_` | 正常尺度 |
| `W2` | `xavier_uniform_` | 正常尺度 |
| `gate_bias` | `ones(1)` → 值 1.0 | $\sigma(1.0) \approx 0.73$，gate 初期大部分开启，不截断信息 |
