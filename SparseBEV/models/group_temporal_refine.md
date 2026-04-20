# GroupTemporalRefine 模块说明（v3：SA + FFN + 逐 (group, frame) 不确定度门）

## 0. 版本演进

| 版本 | 结构 | 主要问题 |
|---|---|---|
| v1 `HistoricalFrameGating` | 乘法 gate，对历史帧打分 | 乘法单边压制 → mAOE 变差 |
| v2 `GroupTemporalRefine` (self-attn, 所有 Q/K/V 都被 query 偏置，无 FFN) | SA over G·T tokens，加法残差 | query 同时塑造 K/V 的投影，路径过强；refine 是 per-(g,t) 的 C_g 向量、无显式 FFN |
| **v3（当前）** | **SA over G·T tokens（仅 Q 条件化）+ FFN + 逐 (g,t) 2 标量不确定度门** | 以下展开 |

v3 核心改动：
- **只让 query 塑造"该关注谁"**（Q），不塑造 token 自身内容（K/V）
- **显式 FFN**：给交互后的特征一层非线性
- **每个 (g, t) 出 2 个标量**（μ, logσ²），所有 C_g 通道共享这对标量；残差只在 P 维广播
- **削弱 > 放大**：uncertain (g, t) 自动被压小，保留 v1 "weakening" 的直觉，同时避免 v1 的结构伤害

---

## 1. 目的

在 SparseBEV decoder layer 的 `SparseBEVSampling` 之后、`AdaptiveMixing` 之前，用一个轻量的 self-attention + FFN 去做 **per-(group, frame) refine**：

- Q 是 G·T=32 个 (g, t) tokens（带 query 偏置）
- K/V 也是同样 32 个 tokens（**不带 query 条件**）
- 输出一个 C_g 维 refinement `h_{g,t}`，再由 `scale_{g,t} = μ_{g,t} · exp(-0.5·logσ²_{g,t})` 标量门缩放
- 残差只在 P 维广播后加回原 sampled_feat（每个 (g, t) 的 P 个采样点共享该 (g, t) 的 refine）

---

## 2. 在 decoder 中的位置

```
query_bbox, query_feat
        │
        ▼
  SparseBEVSelfAttention       (query 间 SA)
        │
        ▼
  SparseBEVSampling            (按 query_bbox 从 FPN 采样，vel*time_diff warp 历史帧)
        │
        ▼
  GroupTemporalRefine  ◄── 本模块
        │
        ▼
  AdaptiveMixing
        │
        ▼
  FFN → cls_branch / reg_branch
```

调用点：`SparseBEVTransformerDecoderLayer.forward`（[`sparsebev_transformer.py:173`](sparsebev_transformer.py#L173)）

---

## 3. 输入 / 输出

| 名称 | 形状 | 说明 |
|---|---|---|
| 输入 `sampled_feat` | `[B, Q, G, T·P, C_g]` | 采样特征，G=4, T=8, P=4, C_g=64 |
| 输入 `query_feat`   | `[B, Q, embed_dims]` | embed_dims=256 |
| 输出 | `[B, Q, G, T·P, C_g]` | 形状与输入相同，逐 (g, t) 残差 refine |

---

## 4. 整体数据流

```
                          sampled_feat [B, Q, G, T·P, C_g]
                                     │
                                     │ view(B,Q,G,T,P,C_g)
                                     ▼
                             x [B,Q,G,T,P,C_g] ──────────────────┐
                                     │                              │ (保留，用于残差)
                                     │ mean over P                  │
                                     ▼                              │
                             gt [B,Q,G,T,C_g]                       │
                                     │                              │
                                     │ + pe_group + pe_frame        │
                                     ▼                              │
                             gt + PE [B,Q,G,T,C_g]                  │
                                     │                              │
                                     │ reshape to [BQ, G*T, C_g]    │
                                     ▼                              │
                             gt_flat [BQ, G·T=32, C_g]              │
                                     │                              │
        ┌────────────────────────────┼─────────┐                    │
        │                            │         │                    │
        │ Q 源（+ query 偏置）          │ K/V 源（无 query 条件）  │
        │                            │         │                    │
   q_proj_g(gt_flat)                 │  kv_proj(gt_flat)            │
     + q_proj_query(query_feat)      │         │                    │
   [BQ, 1, C_g] broadcast            │         │                    │
        ▼                            │         ▼                    │
     q [BQ, 32, C_g]                 │    k,v [BQ, 32, C_g]         │
        │                            │         │                    │
        │    multi-head split        │         │                    │
        ▼                            ▼         ▼                    │
     q [BQ, H, 32, D]           k,v [BQ, H, 32, D]                  │
        │                                      │                    │
        └──────► SDPA (Flash) ──► out [BQ, H, 32, D]                 │
                                     │                              │
                                     │ reshape + attn_out            │
                                     ▼                              │
                             attn_out [BQ, 32, C_g]                 │
                                     │                              │
                                     │ + gt_flat → norm1            │
                                     ▼                              │
                             h [BQ, 32, C_g]                        │
                                     │                              │
                                     │ + ffn(h) → norm2             │
                                     ▼                              │
                             h [BQ, 32, C_g]   (refinement 特征)     │
                                     │                              │
                             ┌───────┴──────────────┐               │
                             │                      │               │
                             │ gate_head            │               │
                             │ Linear(C_g, 2)       │               │
                             ▼                      │               │
                        (μ, logσ²) [B,Q,G,T,2]      │               │
                             │                      │               │
                scale = μ·exp(-0.5·logσ²)           │               │
                        [B,Q,G,T]                   │               │
                             │                      │               │
                             └──────► *  ◄──────────┘               │
                                         │                          │
                                 refine [B, Q, G, T, C_g]           │
                                         │                          │
                                         │ unsqueeze(4)             │
                                         │ broadcast over P         │
                                         ▼                          │
                                 refine [B,Q,G,T,1,C_g]             │
                                         │                          │
                                         └──────►  +  ◄─────────────┘
                                                    │
                                                    ▼
                                        x_new [B,Q,G,T,P,C_g]
                                                    │
                                                    │ reshape
                                                    ▼
                                        [B, Q, G, T·P, C_g] → AdaptiveMixing
```

---

## 5. 前向 step-by-step

### Step 1：pool P → 得到 (group, frame) tokens

```python
x = sampled_feat.view(B, Q, G, T, P, C_g)
gt = x.mean(dim=4)                                           # [B, Q, G, T, C_g]
pe = self.pe_group[:, None, :] + self.pe_frame[None, :, :]   # [G, T, C_g]
gt = gt + pe
```

- 聚合每个 (g, t) 位置的 P 个采样点，得到 64 维 token — "第 g 组在第 t 帧的整体语义"
- 加 learnable (group, frame) PE；组和帧身份分开编码（参数少 + 泛化好）

**为什么 pool P**：P 内部是同一 box 的邻近采样点，差异不大；pool 掉之后 attention 的 seq_len 是 32 而不是 128，4× 省计算。P 维的细节通过残差广播回去，不丢信息。

### Step 2：Flatten + 分离的 Q / K / V 投影

```python
N = G * T                                                    # 32
gt_flat = gt.reshape(BQ, N, C_g)                             # [BQ, 32, 64]

# Q：token 投影 + query 偏置（在 G·T 维广播）
q = self.q_proj_g(gt_flat) + self.q_proj_query(query_feat).reshape(BQ, 1, C_g)

# K, V：token 独立投影，query 不参与
kv = self.kv_proj(gt_flat)                                   # [BQ, 32, 2·C_g]
k, v = kv.split(C_g, dim=-1)
```

**核心**：query_feat 只偏置 Q，不偏置 K/V。语义上 query 改变"该 token 去看谁"，不污染 token 自身的内容和 value。

- `q_proj_query(query_feat) [BQ, 1, C_g]` 在 G·T=32 维广播 → 同一 query 的所有 32 个 Q 共享这个偏置
- 数学上等价于 concat 后投影：`W_q · [g_{gt}; q] = W_q^g · g_{gt} + W_q^q · q`，但省显存（无 `expand` materialise）

**与 v2 的区别**：v2 的 `q_bias` 加到 QKV 三份上（`qkv = qkv_g + qkv_q`），v3 只加到 Q 上。

### Step 3：Multi-head SDPA（self-attention）

```python
H, D = self.num_heads, self.head_dim                         # H=4, D=16
q = q.view(BQ, N, H, D).transpose(1, 2)                      # [BQ, H, 32, D]
k = k.view(BQ, N, H, D).transpose(1, 2)
v = v.view(BQ, N, H, D).transpose(1, 2)

out = F.scaled_dot_product_attention(q, k, v)                # [BQ, H, 32, D]
out = out.transpose(1, 2).reshape(BQ, N, C_g)                # [BQ, 32, 64]
out = self.attn_out(out)                                     # Linear(C_g, C_g)
```

- 每个 (g, t) token 对所有 32 个 (g', t') tokens 做 attention，拿跨组 + 跨帧的上下文
- FA2 命中友好：H=4, D=16, seq_len=32, BQ=B·900 足大

### Step 4：Post-norm + FFN

```python
h = self.norm1(gt_flat + out)                                # 残差 + LN
h = self.norm2(h + self.ffn(h))                              # FFN 残差 + LN
```

标准 post-norm transformer：
- `ffn = Linear(C_g, 2·C_g) → GELU → Linear(2·C_g, C_g)`
- ffn_ratio=2（ffn_dim=128）

### Step 5：逐 (g, t) 2 标量门（不确定度）

```python
gate = self.gate_head(h).view(B, Q, G, T, 2)                 # Linear(C_g, 2)
mean_s, log_var_s = gate.unbind(dim=-1)                      # [B, Q, G, T] each
log_var_s = log_var_s.clamp(-10, 10)
scale = mean_s * torch.exp(-0.5 * log_var_s)                 # [B, Q, G, T]
```

**每个 (g, t) 位置预测 2 个标量**：`μ_{g,t}` 和 `log σ²_{g,t}`，该 (g, t) 的所有 C_g 通道共享这一对。精度加权合成：

$$
\text{scale}_{g,t} = \mu_{g,t} \cdot \exp\!\left(-\tfrac{1}{2}\log\sigma_{g,t}^2\right) = \mu_{g,t} / \sigma_{g,t}
$$

- μ_{g,t}：期望权重（这个 (g, t) 应该 refine 多少）
- `exp(-0.5·log σ²_{g,t}) = 1/σ_{g,t}`：precision（置信度）
- 组合：σ 大（不确定）→ precision 趋于 0 → 该 (g, t) 的整组贡献被削弱
- **不同 (g, t) 的 scale 互相独立**，某些 (g, t) 可以被压很低，另一些保持不变

### Step 6：应用标量门 + 在 P 上广播

```python
h = h.view(B, Q, G, T, C_g)
refine = h * scale.unsqueeze(-1)                             # [B, Q, G, T, C_g]
x = x + refine.unsqueeze(4)                                  # [B,Q,G,T,1,C_g] → 广播 P
return x.reshape(B, Q, G, T·P, C_g)
```

- refine 在 **P 维广播**：同一 (g, t) 的 P 个采样点接收同一个 refine 向量（(g, t) 级共识修正所有 P 个点）
- 加法残差，scale=0 就退化成 identity
- **T 维 NOT 广播**：每个 t 都有自己独立的 refine，保留帧级差异

---

## 6. 关键设计决策

### 6.1 为什么只 Q 被 query 条件化，K/V 不

| 分量 | 含义 | query 条件化 |
|---|---|---|
| Q  | "该 token 要去看谁"（attention pattern） | **是** |
| K  | "每个 (g,t) token 自己是什么内容" | 否 |
| V  | "每个 (g,t) token 能提供什么信息" | 否 |

**语义**：query 塑造注意力方向，不污染 token 自身的表达。避免 v2 "K/V 被 query 偏置 → token 内容变成 query 的函数" 造成的路径污染。

**数学**：attention score
$$
q_i \cdot k_j = (W_q^g \cdot g_i + W_q^q \cdot q) \cdot (W_k^g \cdot g_j)
$$
pairwise score 里包含 `q · g_j` 交叉项 → query 影响"第 j 个 token 对这一条 query 该多重要" ✓。但 V 是纯 token 内容，query 不参与被聚合的值。

### 6.2 为什么残差 gate 是逐 (g, t) 而不是逐 g

之前尝试过 "pool T → per-group attention query → per-group scalar gate"，但这样：
- attention 输出只有 G=4 个向量，T 维的差异被丢失
- refine 在 T 上广播，同一 group 所有帧共享同一个 refine
- **不符合"每个 (g, t) 有独立可靠度"的直觉**——历史帧 t=0 和当前帧 t=7 的可靠度显然不同

v3 改成逐 (g, t)：
- attention 输出 G·T=32 个 token，每个 (g, t) 位置有自己的 64 维 refine
- gate 也是逐 (g, t) 的 2 标量
- T 维在残差里不广播，每帧独立缩放

### 6.3 为什么 pool P 而不是把 P 作为 token 维

- G·T·P=128 个 token 的 SA 比 G·T=32 个慢 16×（O(N²)）
- P 内 4 个采样点差异有限（同 box 邻近点），pool 掉损失少
- P 维细节通过残差 `x + refine.unsqueeze(4)` 原样保留，SA 只是加 refinement

### 6.4 为什么要 FFN（v2 没有）

标准 transformer 块：Attn 提供 **token 交互**，FFN 提供 **逐 token 的非线性变换**。

v2 缺 FFN 的后果：attention 后直接 out_proj 线性映射到残差，token 表达能力弱。v3 每个 (g, t) token 都经过 64→128→64 的 GELU 非线性通道，表达空间变大。

### 6.5 为什么每 (g, t) 只出 2 标量（不是 C_g 或 2·C_g 向量）

- "per-channel 不确定度" 没有监督信号，易过拟合
- 2 标量足以表达 "该 (g, t) 整体该 refine 多少、多确定"
- 更省参（G·T·2 vs G·T·2·C_g）
- 语义干净：整组标量门 = "这个 (g, t) 整体靠不靠谱"

### 6.6 不确定度门的形式：`μ · exp(-0.5·log σ²)`

等价于 `μ / σ`（precision-weighted mean）。解读：
- μ_{g,t}：模型认为该 (g, t) 应该 refine 多少
- σ_{g,t}：估计的不确定度
- 合成：σ 大 → μ 被压缩 → 不自信就保守不动

**对比 sigmoid gate**：
- `sigmoid(x) ∈ (0, 1)` 只能衰减（更像 v1 乘法 gate）
- `μ · exp(-0.5·log σ²)` 可正可负、零对称，学习更自由；默认行为偏削弱（σ²>0 时 1/σ<1），但不禁止放大
- 没有显式 NLL loss 监督 log σ²，它作为一个 soft gate，但保留 heteroscedastic 的几何结构

### 6.7 Identity-at-init

初始化：
- `pe_group`, `pe_frame`: `trunc_normal_(std=0.02)`
- `q_proj_g`, `q_proj_query`, `kv_proj`, `attn_out`, `ffn`, `norm*`: PyTorch 默认
- **`gate_head.weight = 0`, `gate_head.bias = 0`** → μ = log σ² = 0 → `scale = 0·exp(0) = 0`

效果：`refine = h · 0 = 0`，`x + 0 = x`，初始严格等价 baseline。

**梯度流**：虽然 forward=0，但 `d(gate_head)/d(loss)` 非零（h ≠ 0），step 1 后 gate_head 解锁；之后 `d(h)/d(loss)` 通过 scale 非零，attn + FFN 全部解锁学习。

---

## 7. 参数与计算

### 参数（每个 decoder layer，embed_dims=256, G=4, T=8, C_g=64, H=4, ffn_ratio=2）

| 模块 | shape | 参数 |
|---|---|---|
| `q_proj_g`     | Linear(64, 64)           | 4,160 |
| `q_proj_query` | Linear(256, 64), no bias | 16,384 |
| `kv_proj`      | Linear(64, 128)          | 8,320 |
| `attn_out`     | Linear(64, 64)           | 4,160 |
| `norm1`        | LN(64)                   | 128 |
| `ffn[0]`       | Linear(64, 128)          | 8,320 |
| `ffn[2]`       | Linear(128, 64)          | 8,256 |
| `norm2`        | LN(64)                   | 128 |
| `gate_head`    | Linear(64, 2)            | 130 |
| `pe_group`     | (4, 64)                  | 256 |
| `pe_frame`     | (8, 64)                  | 512 |
| **合计**       |                          | **~51k** |

6 层 decoder 共 ~300k，相对全网可忽略。

### 计算（per sample，Q=900）

| 操作 | FLOPs (≈) |
|---|---|
| pool P + PE                       | < 10 M |
| `q_proj_g` × 32 × 900              | 12 M |
| `q_proj_query` × 900               | 30 M |
| `kv_proj` × 32 × 900               | 240 M |
| SDPA (FA2, H=4, N=32, D=16) × 900  | ~170 M |
| `attn_out` × 32 × 900              | 115 M |
| `ffn` × 32 × 900                   | 480 M |
| `gate_head` × 32 × 900             | 4 M |
| 标量乘 + 广播残差                    | < 10 M |
| **单层合计**                        | **~1.0 G** |

6 层 ~6 GFLOPs，相对整网 ~200 GFLOPs 占 ~3%。

训练时长预估 +3~5%（A100/H100 + AMP 命中 FA2）。

---

## 8. FlashAttention 命中检查

满足以下条件时 `F.scaled_dot_product_attention` dispatch 到 Flash Attention 2：

- ✓ CUDA tensor
- ✓ 无 attention mask
- ✓ head_dim = 16（FA2 支持 ≤ 256）
- ✓ seq_len = 32（2 的幂）
- ✓ fp16 / bf16（AMP 命中）
- ✓ 有效 batch = B · Q ≈ 900+

验证：

```python
from torch.backends.cuda import sdp_kernel
with sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
    out = F.scaled_dot_product_attention(q, k, v)
```

---

## 9. 调参建议

### 9.1 `num_heads`

默认 4，head_dim=16。`num_heads=2` 或 `num_heads=8` 都能跑，一般差异 < 0.5% mAP。

### 9.2 `ffn_ratio`

默认 2（ffn_dim=128）。如需更大表达力可设 4（ffn_dim=256），参数 ~+16k/层。

### 9.3 `log_var_clamp`

默认 `(-10, 10)`，precision `∈ [e^-5, e^5] ≈ [0.007, 148]`。若训练不稳收紧到 `(-4, 4)`。

### 9.4 何时禁用本模块

`num_frames = 1` 时自动跳过（`if T <= 1: return sampled_feat`）。

### 9.5 是否挂所有 decoder layer

默认所有 6 层都挂。省计算方案：
- 只挂后 2 层：砍 2/3 开销
- 只挂第一层：让基础采样特征先交互

---

## 10. v2 vs v3 对比总结

| 维度 | v2（之前） | v3（当前） |
|---|---|---|
| Attention 类型 | SA over G·T | **SA over G·T** |
| Query 条件化 | 同时偏置 Q, K, V | **仅偏置 Q** |
| FFN | ✗ | ✓（ratio=2, GELU） |
| 归一化 | ✗ | LN post-norm |
| 输出粒度 | per-(g, t)，C_g 向量 | per-(g, t)，C_g 向量 × 标量门 |
| 不确定度 | ✗ | ✓（μ, log σ²，每 (g,t) 2 标量） |
| Residual 广播 | 广播到 P | 广播到 P（T 维保留独立门控） |
| 总参数/层 | ~66k | ~51k |

---

## 11. 关键代码位置

- 模块定义：[`sparsebev_transformer.py:323-436`](sparsebev_transformer.py#L323-L436)
- 调用点：[`sparsebev_transformer.py:173`](sparsebev_transformer.py#L173)（`sampled_feat = self.refine(sampled_feat, query_feat)`）
- decoder layer 构造：[`sparsebev_transformer.py:124`](sparsebev_transformer.py#L124)（`self.refine = GroupTemporalRefine(...)`）
