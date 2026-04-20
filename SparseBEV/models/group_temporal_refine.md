# GroupTemporalRefine 模块说明

## 1. 目的

在 SparseBEV decoder layer 的 `SparseBEVSampling` 之后、`AdaptiveMixing` 之前插入一个轻量级的 self-attention 模块，用 **query 条件化的跨 (group, frame) 自注意力** 对采样特征做残差 refinement。

替换了初版的 `HistoricalFrameGating`，解决后者"单边压制历史帧"结构上伤 mAOE 的问题。

## 2. 在 decoder 中的位置

```
query_bbox, query_feat
        │
        ▼
  SparseBEVSelfAttention   (query 之间的 SA)
        │
        ▼
  SparseBEVSampling        (按 query_bbox 从 FPN 采样，vel*time_diff warp 历史帧)
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

调用位置：`SparseBEVTransformerDecoderLayer.forward`（`sparsebev_transformer.py` 内）

## 3. 输入 / 输出

| 名称 | 形状 | 说明 |
|---|---|---|
| 输入 `sampled_feat` | `[B, Q, G, T*P, C_g]` | 采样特征，G=组数=4，T=帧数=8，P=每帧每组采样点=4，C_g=embed_dims/G=64 |
| 输入 `query_feat`  | `[B, Q, embed_dims]` | 当前层的 query 特征，embed_dims=256 |
| 输出 | `[B, Q, G, T*P, C_g]` | 残差 refine 后的采样特征，形状与输入相同 |

## 4. 结构流程

```
                          sampled_feat [B,Q,G,T*P,C_g]
                                     │
                                     │ view(B,Q,G,T,P,C_g)
                                     ▼
                             x [B,Q,G,T,P,C_g] ──────────┐
                                     │                     │ (保留，用于残差)
                                     │ mean(dim=P)         │
                                     ▼                     │
                             g_tok [B,Q,G,T,C_g]           │
                                     │                     │
                                     │ + pe_group + pe_frame (learnable)
                                     ▼                     │
                             g_tok+PE [B,Q,G,T,C_g]        │
                                     │                     │
                                     │ reshape -> [BQ, N=G*T, C_g]
                                     ▼                     │
           query_feat [B,Q,256]      tokens [BQ, 32, 64]   │
                  │                        │               │
        qkv_q(·)  │                        │ qkv_g(·)     │
        Linear    │                        │ Linear       │
        256→192   │                        │ 64→192       │
                  ▼                        ▼               │
           q_bias [BQ,1,192] ────► + ◄──── qkv [BQ,32,192] │
                                     │                     │
                                     │ (广播加法，等价 concat+大 Linear)
                                     ▼                     │
                             qkv [BQ,32,192]              │
                                     │                     │
                                     │ split 3 × [BQ,H=4,32,D=16]
                                     ▼                     │
                                  q, k, v                  │
                                     │                     │
                                     │ SDPA (FlashAttn backend)
                                     ▼                     │
                             out [BQ, H, 32, 16]           │
                                     │                     │
                                     │ transpose + reshape │
                                     ▼                     │
                             out [BQ, 32, 64]              │
                                     │                     │
                                     │ out_proj (Linear 64→64)
                                     ▼                     │
                             refine [B,Q,G,T,C_g]          │
                                     │                     │
                                     │ unsqueeze(P)  广播到 P 维
                                     ▼                     │
                             refine [B,Q,G,T,1,C_g]        │
                                     │                     │
                                     └─────►  +  ◄─────────┘
                                             │
                                             ▼
                                   x_new [B,Q,G,T,P,C_g]
                                             │
                                             │ reshape
                                             ▼
                                   [B,Q,G,T*P,C_g]  (送 AdaptiveMixing)
```

## 5. 前向 step-by-step

### Step 1：Pool P 得到 (group, frame) token

```python
x = sampled_feat.view(B, Q, G, T, P, C_g)
g_tokens = x.mean(dim=4)                       # [B, Q, G, T, C_g]
```

每个 (g, t) 位置聚合该组该帧的 P 个采样点，得到一个 64 维 token，代表"第 g 组在第 t 帧的整体语义"。共 G×T = 32 个 token。

**设计依据**：pool 掉 P 把 token 数从 128 降到 32，4× 省 SA 计算；P 内的空间细节靠残差加回去（见 Step 7）不丢失。

### Step 2：加 learnable PE

```python
pe = self.pe_group[:, None, :] + self.pe_frame[None, :, :]   # [G, T, C_g]
g_tokens = g_tokens + pe
```

- `pe_group [G=4, C_g=64]`：组身份编码
- `pe_frame [T=8, C_g=64]`：帧身份编码
- 相加广播 → 每个 (g, t) 有唯一的位置身份

**分开而非联合编码**：参数少（`4*64 + 8*64` vs `32*64`），泛化性好——组身份和帧身份是可分离的两种信号。

**Query 的 token（如果存在）不加 PE**：当前版本已去掉 query token，PE 只作用于 g_tokens。

### Step 3：Flatten 到 SA 友好的形状

```python
BQ = B * Q
N = G * T                                      # 32
tokens = g_tokens.reshape(BQ, N, C_g)          # [BQ, 32, 64]
```

把 (B, Q) 合并成有效 batch 维。B=1、Q=900 时 BQ=900 条独立序列并行送 SDPA，GPU 吃得满。

### Step 4：Query-conditioned QKV 投影

```python
qkv_g = self.qkv_g(tokens)                     # [BQ, N, 3*C_g]  - 64→192
q_bias = self.qkv_q(query_feat).reshape(BQ, 1, 3 * C_g)   # [BQ, 1, 3*C_g]  - 256→192
qkv = qkv_g + q_bias                           # 广播加法
```

**核心设计点**：qkv 由两条路径相加而成：
- `qkv_g`：g_token 自己的内容生成的 Q/K/V
- `qkv_q`：query_feat 生成的 Q/K/V 偏置，在 G*T 维上广播（同一 query 的所有 g_token 共享这个偏置）

这等价于"把 query_feat 投影后 concat 到每个 g_token 再做 Linear"：

$$W_\text{qkv} \cdot [g\ ;\ q] = W_\text{qkv}^{(g)} \cdot g + W_\text{qkv}^{(q)} \cdot q$$

但避免了显式 concat 时的 `expand` materialize 和 2× 输入维度 GEMM。

**这样做的语义**：两个 g_token 之间的 attention score 展开为

$$q_i \cdot k_j = g_i \cdot g_j + g_i \cdot W_q \cdot q + (W_k \cdot q) \cdot g_j + (\text{const})$$

出现了 `g_i · q`、`q · g_j` 这种**依赖 token 内容 × query** 的交叉项——query 真正地塑造了 g↔g 的 pairwise attention pattern，而不只是全局偏置。

### Step 5：多头 reshape + SDPA

```python
qkv = qkv.view(BQ, N, 3, H, D).permute(2, 0, 3, 1, 4)   # [3, BQ, H, N, D]
q, k, v = qkv[0], qkv[1], qkv[2]                         # 各 [BQ, H=4, N=32, D=16]

out = F.scaled_dot_product_attention(q, k, v)            # [BQ, H, N, D]
```

`F.scaled_dot_product_attention` 是 PyTorch 2.x 原生接口，自动 dispatch 到三种后端：
- **Flash Attention 2**（推荐，训练时命中）
- Memory-efficient attention（FP32 / 特殊 shape 回退）
- Math backend（兜底）

当前 shape：H=4, N=32, D=16，所有参数都在 FA2 友好区间内。

### Step 6：Out projection

```python
out = out.transpose(1, 2).reshape(BQ, N, C_g)            # [BQ, 32, 64]
refine = self.out_proj(out).view(B, Q, G, T, C_g)        # [B, Q, G, T, 64]
```

`out_proj`：`Linear(64, 64)`，**权重和 bias 都零初始化**——这是 identity-at-init 的关键。

### Step 7：残差加回 P 维

```python
x = x + refine.unsqueeze(4)                              # broadcast over P
return x.reshape(B, Q, G, TP, C_g)
```

`refine [B,Q,G,T,1,C_g]` 广播到 `[B,Q,G,T,P,C_g]`，加到原 x 上。同一 (g, t) 的 P 个采样点接收同一个 refine 向量——这是 mean-pool over P 的对偶：token 级共识去修正所有点。

**形式**：加法残差，**不是** 乘法 gate。允许任意方向的修正（增强/抑制），没有"历史必须 ≤ 当前"的结构约束。

## 6. 关键设计决策

### 6.1 为什么用 SA 而不是 gating

初版 `HistoricalFrameGating` 的问题：
- 乘法 gate `x * w, w ∈ (0, 1)` **单边压制** 历史帧
- 特征如 cos 相似度、帧间 diff 会让网络学到"视角差异大 = 不可靠"的错误先验
- 实测 mAOE（朝向误差）+0.05 rad，NDS -0.5%

SA 的优势：
- **加法残差**，允许增强或抑制，无方向性约束
- 每个 (g, t) 的修正是 **C_g=64 维向量**，表达力 vs 标量 gate 多 64×
- query 通过条件化 QKV 告诉每个 g_token "该怎么看其他 g_token"，而不是全局压制

### 6.2 为什么 pool P 而不是把 P 作为 token 维

- G×T×P = 128 个 token 的 SA 比 32 个慢 16×（attention 复杂度 O(N²)）
- P 内部的 4 个采样点本身差异不大（同一 box 的邻近点），pool 掉损失有限
- P 维细节通过残差 `x + refine.unsqueeze(4)` 原封不动保留，SA 只是加了 refinement

### 6.3 为什么 query condition QKV 而不是 FiLM 或 prepend

| 方案 | query 角色 | 是否塑造 g↔g attention |
|---|---|---|
| Prepend query token | value 源 | ✗ 只能注入信息，不改 pairwise score |
| FiLM (uniform scale/shift) | 特征偏置 | 弱（均匀偏置，无法告诉"pair(i,j) 该强该弱"） |
| **Query-conditioned QKV** | **Q 和 K 的投影参数** | **✓ pairwise score 本身是 query 的函数** |

### 6.4 Concat 为什么实现为两条 Linear 相加

数学上等价：`W_qkv @ [g ; q] = W_qkv^g @ g + W_qkv^q @ q`

工程上更优：
- 避免 `expand()` 和 `torch.cat()` 的显存 materialize（~7MB/sample）
- qkv 主线的 GEMM 输入维度不变（`C_g` 而非 `2*C_g`），GEMM 大小对半
- `qkv_q` 一次 Linear + 一次广播加，在 G*T 维免费复制

### 6.5 Identity-at-init

初始化策略：
- `pe_group`, `pe_frame`：`trunc_normal_(std=0.02)`，标准 transformer PE 初始化
- `qkv_g`, `qkv_q`：PyTorch 默认（kaiming uniform）
- **`out_proj.weight = 0`, `out_proj.bias = 0`**

效果：`refine = out_proj(out) = 0`，整个模块输出恒为 0，残差 `x + 0 = x` → 初始严格等价 baseline。

**梯度流**：尽管 forward 是 0，`d(out_proj.W)/d(loss)` 非零，step 1 后 `out_proj` 解锁，后续所有参数开始学习。

## 7. 参数与计算量

### 参数（每个 decoder layer）

| 模块 | shape | 参数 |
|---|---|---|
| `qkv_g` | Linear(64, 192) | 12,480 |
| `qkv_q` | Linear(256, 192), 无 bias | 49,152 |
| `out_proj` | Linear(64, 64) | 4,160 |
| `pe_group` | (4, 64) | 256 |
| `pe_frame` | (8, 64) | 512 |
| **合计** | | **~66k** |

6 层 decoder 总参数 ~400k，相对全网可忽略。

### 计算（每层，per sample）

| 操作 | FLOPs |
|---|---|
| Mean pool P | 7 M |
| PE + reshape | < 1 M |
| `qkv_g`: Linear(64, 192) × 32 × 900 | 350 M |
| `qkv_q`: Linear(256, 192) × 900 | 44 M |
| 广播加 | < 5 M |
| SDPA (flash, N=32, H=4, D=16) × 900 | 170 M |
| `out_proj`: Linear(64, 64) × 32 × 900 | 115 M |
| 残差加 | 7 M |
| **单层合计** | **~700 M** |

6 层 ~4.2 G FLOPs，相对整网 ~200 G FLOPs 占 ~2%。

### 训练时长

预估单次迭代 +3~5%（A100/H100 + AMP 命中 FA2）。包含 gradient checkpointing 的 recompute 开销。

## 8. FlashAttention 命中检查

以下条件都满足时 `F.scaled_dot_product_attention` dispatch 到 Flash Attention 2 backend：

- ✓ CUDA tensor
- ✓ 无 attention mask
- ✓ head_dim = 16（FA2 支持 ≤ 256）
- ✓ seq_len = 32（2 的幂，block tiling 友好）
- ✓ fp16 / bf16（AMP 训练时自动）
- ✓ 有效 batch = B × Q（900+），GPU 吃满

Debug 验证 FA2 是否命中：

```python
from torch.backends.cuda import sdp_kernel
with sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
    out = F.scaled_dot_product_attention(q, k, v)
# 正常返回：FA2 命中
# 抛 error：shape/dtype 不支持，查看报错信息
```

## 9. 调参建议

### 9.1 `num_heads`

当前默认 4，head_dim = 16。可选：
- `num_heads=2`, head_dim=32：更少的 attention 并行但更大的 head，FA2 更好命中
- `num_heads=8`, head_dim=8：head_dim 较小，FA2 支持但效率略低

实测差异通常 < 0.5% mAP，默认即可。

### 9.2 何时禁用本模块

`num_frames = 1` 时模块自动跳过（`if T <= 1: return sampled_feat`）。单帧模型没有时序交互需求。

### 9.3 是否挂所有 decoder layer

当前挂在所有 6 层。如需减计算：
- 只挂后 2 层：砍 2/3 开销，可能损失部分收益
- 只挂第一层：让最基础的采样特征先交互，后续层走轻量
```

## 10. 和 `AdaptiveMixing` 的关系

`GroupTemporalRefine` 产出的是 **refine 过的 sampled_feat**（形状不变），直接喂给 `AdaptiveMixing`：

- `AdaptiveMixing` 做 **point-level mixing**（跨 P 点、跨通道），生成 query_feat 更新
- `GroupTemporalRefine` 做 **(g, t)-level refinement**，补充跨时序/跨组的上下文

两者互补：
- AdaptiveMixing 关注"采样点组合"
- Refine 关注"组和帧的语义交互"

顺序上 refine 在前，确保 AdaptiveMixing 看到的是 query-aware 的 sampled_feat。
