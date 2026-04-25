# FrameSemanticGate 模块说明

在时序融合（`AdaptiveMixing`）之前插入的语义筛选模块。

利用 query_feat 生成 4 组语义参考向量，对采样得到的所有时空点（7 个历史帧 × 每帧 P 个点 × G=4 组）进行**点级可信度打分**，并通过 sigmoid 软门控衰减低可信度的历史帧采样点。**当前帧不被门控**。

---

## 1. 在原模型中的插入位置

`SparseBEVTransformerDecoderLayer.forward` 内部，每层 decoder 都执行一次：

```
self_attn  →  sampling  →  [FrameSemanticGate]  →  mixing  →  ffn
```

输入是 `sampling` 的输出 `sampled_feat`，输出是被门控后的同形状张量，传给 `mixing`。

---

## 2. 输入 / 输出形状

| 张量 | 形状 | 说明 |
|---|---|---|
| `query_feat` (输入) | `[B, Q, C]` | C=256，每条 query 的 embedding |
| `sampled_feat` (输入) | `[B, Q, G, F·P, C/G]` | G=4 组、F=8 帧、P=4 点、每组 64 维 |
| 输出 | `[B, Q, G, F·P, C/G]` | 同上，被逐点标量门调制后的特征 |

记号：
- `B` = batch size
- `Q` = query 数（如 900）
- `G` = 4（组数 / 探针数）
- `F` = 8（帧数：1 当前 + 7 历史）
- `P` = 4（每帧采样点数）
- `C_g` = 64（每组通道数 = 256 / 4）

---

## 3. 数据流图

```
                         query_feat                         sampled_feat
                         [B, Q, 256]                        [B, Q, 4, 32, 64]
                              │                                   │
                              ▼                                   │
                     ┌────────────────┐                           │
                     │   ref_proj     │                           │
                     │ Linear 256→256 │                           │
                     └────────────────┘                           │
                              │                                   │
                          view(B,Q,4,64)                          │
                              ▼                                   │
                       ref [B,Q,4,64]                             │
                              │                                   │
                              │              broadcast on F·P     │
                              └─────────────×─────────────────────┤
                                            │                     │
                                       Hadamard 积                │
                                            │                     │
                                            ▼                     │
                                inter [B, Q, 4, 32, 64]           │
                                "逐通道共激活证据"                │
                                            │                     │
                                            ▼                     │
                              ┌──────────────────────────┐        │
                              │       matmul             │        │
                              │  W: [G=4, 64, 1]         │        │
                              │  per-group  64 → 1        │        │
                              └──────────────────────────┘        │
                                            │                     │
                                       × scale (1/√64)            │
                                            ▼                     │
                                   logits [B, Q, 4, 32, 1]        │
                                            │                     │
                                       view(B,Q,4,F=8,P=4)        │
                                            ▼                     │
                                   logits [B, Q, 4, 8, 4]         │
                                            │                     │
                                  + bias [G]  (per-group 标量)    │
                                            ▼                     │
                                       sigmoid                    │
                                            ▼                     │
                                     gate [B, Q, 4, 8, 4]         │
                                            │                     │
                              ┌─────────────┴─────────────┐       │
                              │   当前帧 (frame 0) 强制 1  │       │
                              │   历史帧 (frame 1..7) 用 gate│     │
                              └─────────────┬─────────────┘       │
                                            ▼                     │
                                     gate [B, Q, 4, 8, 4]         │
                                            │                     │
                                       view(B,Q,4,32,1)           │
                                            │                     │
                                            └────×────────────────┘
                                                 │
                                       逐元素相乘
                                       (gate 标量广播到 64 通道)
                                                 ▼
                                  gated_feat [B, Q, 4, 32, 64]
                                       → 进入 mixing
```

---

## 4. 各步骤的作用

### Step 1：生成 4 组语义参考向量（探针）

```python
ref = self.ref_proj(query_feat).view(B, Q, G, C_g)   # [B, Q, 4, 64]
```

`ref_proj` 是 `Linear(256 → 256)`，输出按通道切分成 4 组：
- `ref[:,:,0,:]` = group 0 的 64 维探针（前 64 通道）
- `ref[:,:,1,:]` = group 1 的探针（中间 64 通道）
- ...

每组探针是从完整的 256 维 query_feat 学出来的**独立**语义查询子空间，对应 sampling 阶段已经按通道分好的 4 组特征。

### Step 2：Hadamard 积构造"包含关系证据"

```python
inter = ref[:, :, :, None, :] * sampled_feat   # [B, Q, 4, 32, 64]
```

逐通道相乘是连续版的 AND——**只有 ref 和 feat 在同一通道都激活时**结果才大。每个 `(group, frame, point)` 位置得到一个 64 维的"共激活证据向量"。

> 关键：这一步建模的是 **ref 与 feat 的关联**（包含 / 共激活），而非单方对另一方的调制。

### Step 3：可信度打分（每点一个标量）

```python
gate = torch.matmul(inter, self.gate_weight) * self.scale  # [B,Q,4,32,1]
```

`gate_weight` 形状 `[G=4, 64, 1]`，**每组一个独立的 64→1 线性映射**。它学的是：**怎样把 64 维证据加权融合成一个标量**——可以理解为可学习的、通道加权的 cosine 相似度。

`scale = 1/√64` 防止 sigmoid 饱和（与 attention 同思路）。

输出是每个点针对该组语义的"可信度 logit"。

### Step 4：sigmoid + 当前帧豁免

```python
gate = torch.sigmoid(logits + bias[G])
gate[:, :, :, 0] = 1.0   # 当前帧不门控
```

- `bias` 初始化为 4.0，sigmoid(4) ≈ 0.98 → **训练初期近似恒等映射**，不会破坏原模型。
- 当前帧（frame 0）的 gate 强制为 1，永远不衰减——只筛选 7 个历史帧。

### Step 5：标量门广播应用

```python
out = sampled_feat * gate.unsqueeze(-1)   # gate 的 1 维通道广播到 64 通道
```

每个点的 64 个通道被同一个 0~1 标量缩放，等价于"按可信度衰减这个点的整体贡献"。

---

## 5. 设计选择与权衡

| 设计点 | 选择 | 原因 |
|---|---|---|
| 关联建模方式 | **Hadamard + Linear** | 加法门 `σ(W_h(ref)+W_x(feat))` 等价于 OR，不能表达"包含"关系；Hadamard 是 AND 的连续版 |
| 门粒度 | **每点一个标量门** | 64→1 fusion，几乎零 FLOPs；下游 mixing 会做通道级变换，不需要这一步再做 |
| 4 组探针 | **G=4，与 sampling 的 group 严格对齐** | 每组独立 ref + 独立 64→1 gate weight，跨组不串扰 |
| 当前帧 | **gate=1 不筛选** | 当前帧总是最可靠，且语义是"筛选历史帧" |
| 初始化 | `gate_weight=0`, `bias=4.0` | sigmoid(4)≈0.98，训练初期保持恒等，模型从原行为出发学习 |
| 缩放 | `1/√C_g` | 防 sigmoid 饱和 |

---

## 6. 参数量

| 部件 | 形状 | 参数量 |
|---|---|---|
| `ref_proj.weight` | `[256, 256]` | 65,536 |
| `ref_proj.bias` | `[256]` | 256 |
| `gate_weight` | `[4, 64, 1]` | 256 |
| `gate_bias` | `[4]` | 4 |
| **合计** | | **~66 K** |

每个 decoder layer 一份，6 层共 ~400K，相对于整网络可忽略。

---

## 7. 计算量（FLOPs，单层，B=1, Q=900）

| 步骤 | FLOPs |
|---|---|
| `ref_proj`（Linear 256→256） | ~59 M |
| Hadamard `ref * feat` | ~0.7 M |
| matmul（每位置 64×1） | ~7.4 M |
| sigmoid + 乘法 + 杂项 | < 1 M |
| **合计 / 层** | **~68 M** |

6 层共 ~410 M，**远低于** `mixing` 单层 ~1.4 G 的开销，整体训练耗时增加可控。

---

## 8. 关键代码（最终版）

```python
class FrameSemanticGate(BaseModule):
    def __init__(self, embed_dims=256, num_groups=4, num_frames=8, init_cfg=None):
        super().__init__(init_cfg)
        self.num_groups = num_groups
        self.num_frames = num_frames
        self.eff_dim = embed_dims // num_groups
        self.scale = self.eff_dim ** -0.5

        self.ref_proj = nn.Linear(embed_dims, embed_dims)
        self.gate_weight = nn.Parameter(torch.zeros(num_groups, self.eff_dim, 1))
        self.gate_bias = nn.Parameter(torch.full((num_groups,), 4.0))

    def inner_forward(self, sampled_feat, query_feat):
        B, Q, G, FP, C_g = sampled_feat.shape
        F_ = self.num_frames
        if F_ <= 1:
            return sampled_feat
        P = FP // F_

        ref = self.ref_proj(query_feat).view(B, Q, G, C_g)
        inter = ref[:, :, :, None, :] * sampled_feat                # [B,Q,G,FP,C_g]
        gate = torch.matmul(inter, self.gate_weight) * self.scale   # [B,Q,G,FP,1]
        gate = gate.view(B, Q, G, F_, P)
        gate = torch.sigmoid(gate + self.gate_bias[None, None, :, None, None])

        gate = torch.cat([
            torch.ones_like(gate[:, :, :, :1]),
            gate[:, :, :, 1:]
        ], dim=3)
        gate = gate.view(B, Q, G, FP, 1)

        return sampled_feat * gate
```

---

## 9. 接入点（在 `SparseBEVTransformerDecoderLayer` 中）

```python
# __init__
self.frame_gate = FrameSemanticGate(embed_dims, num_groups=4, num_frames=num_frames)

# init_weights
self.frame_gate.init_weights()

# forward（在 sampling 之后、mixing 之前）
sampled_feat = self.sampling(query_bbox, query_feat, mlvl_feats, img_metas)
sampled_feat = self.frame_gate(sampled_feat, query_feat)
query_feat   = self.norm2(self.mixing(sampled_feat, query_feat))
```
