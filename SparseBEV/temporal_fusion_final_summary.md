# SparseBEV 时序融合改进方案 —— 完整技术细节总结

> 本文档精确记录所有改动的每一个细节，包括张量维度、算子、初始化策略和数值范围。
>
> **符号约定**（以 `r50_nuimg_704x256` 配置为例）：
>
> | 符号 | 含义 | 默认值 |
> |------|------|--------|
> | B | batch size | 8 |
> | Q | query 数量 | 900 |
> | F/T | 时间帧数 | 8 |
> | G | 采样分组数 | 4 |
> | P | 每组每帧采样点数 | 4 |
> | N | 相机视角数 | 6 |
> | L | FPN 层数 | 4 |
> | C | 特征通道数 (embed_dims) | 256 |
> | C_g | 分组通道数 (C / G) | 64 |

---

## 一、改动文件清单

| 文件 | 改动类型 |
|------|---------|
| `models/sparsebev_transformer.py` | 修改 4 个类 |
| `models/sparsebev_sampling.py` | 修改 `sampling_4d` 函数返回值 |

---

## 二、改动总览

```
原始流程:
  sampling_offset(query_feat) → make_sample_points → 速度补偿(v×Δt) → sampling_4d
  → AdaptiveMixing(channel_mix + point_mix + 残差) → FFN → cls/reg

改进后流程:
  sampling_offset(query_feat) → make_sample_points → 速度补偿(v×Δt + 非线性修正)
  → sampling_4d(返回特征 + 有效率)
  → 查询自适应时序权重 + 有效性softmax加权 → 特征加权
  → AdaptiveMixing(连续时序编码 + channel_mix + point_mix，无残差)
  → 运动自适应门控残差 → FFN → cls/reg
```

---

## 三、逐模块详细改动

### 3.1 SparseBEVSampling —— 采样阶段

#### 3.1.1 新增参数

```python
# ① 时序感知尺度偏置 [F, L] = [8, 4]
self.temporal_scale_bias = nn.Parameter(torch.zeros(8, 4))
# 参数量: 32, 初始化: zeros

# ② 查询自适应时序权重 Linear(C, G*F) = Linear(256, 32)
self.temporal_refine = nn.Linear(256, 32)
# 参数量: 256×32 + 32 = 8,224, 初始化: zeros(weight), zeros(bias)

# ③ 自适应速度补偿修正 Linear(C, F*2) = Linear(256, 16)
self.velocity_correction = nn.Linear(256, 16)
# 参数量: 256×16 + 16 = 4,112, 初始化: zeros(weight), zeros(bias)
```

#### 3.1.2 自适应速度补偿残差修正

**位置**：`inner_forward`，在原始线性速度补偿之后

**原始代码**：
```python
dist = vel * time_diff                    # [B, Q, F, 2]
dist = dist[:, :, :, None, None, :]       # [B, Q, F, 1, 1, 2]
```

**改进后**：
```python
# ---- 原始线性补偿 ----
time_diff = img_metas[0]['time_diff']          # [B, F]          值域: 秒, 当前帧=0
time_diff = time_diff[:, None, :, None]        # [B, 1, F, 1]
vel = query_bbox[..., 8:].detach()             # [B, Q, 2]      单位: m/s
vel = vel[:, :, None, :]                       # [B, Q, 1, 2]
dist = vel * time_diff                         # [B, Q, F, 2]   单位: m (世界坐标)

# ---- 新增: 非线性修正 ----
vel_correction = self.velocity_correction(query_feat)  # Linear(256→16)
                                                        # 输入: [B, Q, 256]
                                                        # 输出: [B, Q, 16]
vel_correction = vel_correction.view(B, Q, F, 2)       # [B, Q, 8, 2]
vel_correction = torch.tanh(vel_correction) * 2.0       # [B, Q, 8, 2]  值域: [-2.0, +2.0] 米

# 屏蔽当前帧 (time_diff=0的帧不需要修正)
td_mask = img_metas[0]['time_diff'].abs()               # [B, F]
td_mask = (td_mask > 1e-5).float()[:, None, :, None]   # [B, 1, F, 1]  二值: 0或1
vel_correction = vel_correction * td_mask                # [B, Q, F, 2]  当前帧修正=0

dist = dist + vel_correction                             # [B, Q, F, 2]

dist = dist[:, :, :, None, None, :]                     # [B, Q, F, 1, 1, 2]
sampling_points[..., 0:2] -= dist                        # [B, Q, F, G, P, 2] -= [B, Q, F, 1, 1, 2]
```

**张量流**：
```
query_feat [B,Q,256] → Linear(256,16) → [B,Q,16] → view → [B,Q,8,2]
  → tanh → ×2.0 → [B,Q,8,2] (值域 ±2m)
  → × td_mask [B,1,8,1] → [B,Q,8,2] (当前帧=0)
  → + dist [B,Q,8,2] → 最终位移
```

#### 3.1.3 时序感知的多尺度权重

**位置**：`inner_forward`，scale_weights 计算处

**原始代码**：
```python
scale_weights = self.scale_weights(query_feat).view(B, Q, G, 1, P, L)  # [B,Q,4,1,4,4]
scale_weights = torch.softmax(scale_weights, dim=-1)                     # 在L维softmax
scale_weights = scale_weights.expand(B, Q, G, F, P, L)                  # [B,Q,4,8,4,4]
```

**改进后**：
```python
scale_weights = self.scale_weights(query_feat).view(B, Q, G, 1, P, L)   # [B,Q,4,1,4,4]

tsb = self.temporal_scale_bias.view(1, 1, 1, F, 1, L)                    # [1,1,1,8,1,4]
# temporal_scale_bias: [8, 4], 每帧对每个FPN层的偏好偏置

scale_weights = scale_weights.expand(B, Q, G, F, P, L)                   # [B,Q,4,8,4,4]
scale_weights = torch.softmax(scale_weights + tsb, dim=-1)                # [B,Q,4,8,4,4]
# 先expand再加偏置再softmax，不同帧有不同的尺度偏好
```

**张量流**：
```
scale_weights [B,Q,4,1,4,4] → expand → [B,Q,4,8,4,4]
  + temporal_scale_bias [1,1,1,8,1,4] (broadcast)
  → softmax(dim=-1) → [B,Q,4,8,4,4]
```

#### 3.1.4 查询自适应时序权重 + 有效性加权

**位置**：`inner_forward`，sampling_4d 返回之后

**完整计算流程**：
```python
# ---- 步骤1: 查询自适应时序权重 (logits) ----
query_temporal_w = self.temporal_refine(query_feat)             # Linear(256→32)
                                                                 # 输入: [B, Q, 256]
                                                                 # 输出: [B, Q, 32]
query_temporal_w = query_temporal_w.view(B, Q, G, F)            # [B, Q, 4, 8]
# 每个 query 对每个 group 的每帧产生一个标量 logit
# 初始: 零初始化 → 全零 → softmax 后均匀分布

# ---- 步骤2: 采样点有效性置信度 ----
fv = frame_validity.detach().clone()                             # [B, Q, G, F] = [B, Q, 4, 8]
fv[:, :, :, 0] = fv[:, :, :, 0].clamp(min=0.1)                # 当前帧最小权重保护

# 将有效率转化为 log-space mask（无效帧→大负值→softmax后权重≈0）
validity_mask = torch.log(fv + 1e-6)                            # [B, Q, 4, 8]
# fv=1.0 → log(1.0)=0.0 (不影响)
# fv=0.0 → log(1e-6)=-13.8 (softmax后≈0)
# fv=0.5 → log(0.5)=-0.69 (适度降权)

# ---- 步骤3: softmax 归一化 ----
temporal_weights = torch.softmax(
    query_temporal_w + validity_mask,                            # [B, Q, 4, 8]
    dim=-1                                                       # 在 F 维 softmax
) * F                                                            # [B, Q, 4, 8]
# softmax 输出 sum=1，乘 F=8 后 sum=F=8
# 初始(全零logits + 全valid): softmax([0,...,0]) = [1/8,...,1/8] × 8 = [1,...,1]
# 与原始等权融合完全一致

# ---- 步骤4: 展开到逐点级别 ----
temporal_weights = temporal_weights.unsqueeze(-1)                # [B, Q, 4, 8, 1]
temporal_weights = temporal_weights.expand(B, Q, G, F, P)        # [B, Q, 4, 8, 4]
temporal_weights = temporal_weights.reshape(B, Q, G, F*P, 1)     # [B, Q, 4, 32, 1]

# ---- 步骤5: 应用到采样特征 ----
sampled_feats = sampled_feats * temporal_weights                  # [B,Q,4,32,64] × [B,Q,4,32,1]
                                                                   # broadcast → [B, Q, 4, 32, 64]
```

**张量流总结**：
```
query_feat [B,Q,C] → Linear(256,32) → [B,Q,32]
  → view → [B,Q,G,F] (logits)
                           │
frame_validity [B,Q,G,F]  │
  → clone → clamp(t=0,min=0.1)
  → log(·+1e-6) → validity_mask [B,Q,G,F]
                           │
                logits + validity_mask
                           │
                    softmax(dim=F) × F
                           │
                    expand → [B,Q,G,F*P,1]
                           │
sampled_feats [B,Q,G,F*P,C] ──→ × ──→ weighted_feats [B,Q,G,F*P,C]
```

---

### 3.2 sampling_4d —— 有效性信息返回

#### 3.2.1 改动位置

`models/sparsebev_sampling.py`，函数末尾

#### 3.2.2 新增代码

```python
# ---- 原始: 仅返回 final ----
# return final  # [B, Q, G, FP, C]

# ---- 改进: 额外返回 frame_validity ----

# valid_mask 在 argmax 选择最佳视角后: [B, T, Q, G*P, 1]
valid_mask = valid_mask.squeeze(-1)                       # [B, T, Q, G*P]
valid_mask = valid_mask.reshape(B, T, Q, G, P)            # [B, 8, Q, 4, 4]
valid_mask = valid_mask.permute(0, 2, 3, 1, 4)           # [B, Q, 4, 8, 4]
frame_validity = valid_mask.mean(dim=-1)                  # [B, Q, 4, 8]
# 含义: 每帧中有效采样点(落在图像内)的比例
# 值域: [0.0, 1.0]
# 0.0 = 该帧所有4个采样点都在图像外
# 1.0 = 该帧所有4个采样点都在图像内

return final, frame_validity
# final:          [B, Q, G, F*P, C_g] = [B, Q, 4, 32, 64]
# frame_validity: [B, Q, G, F]        = [B, Q, 4, 8]
```

**张量维度变换**：
```
valid_mask [B, T, Q, GP, 1]
  → squeeze(-1) → [B, T, Q, GP]       = [B, 8, 900, 16]
  → reshape     → [B, T, Q, G, P]     = [B, 8, 900, 4, 4]
  → permute     → [B, Q, G, T, P]     = [B, 900, 4, 8, 4]
  → mean(dim=-1)→ [B, Q, G, T]        = [B, 900, 4, 8]
```

---

### 3.3 AdaptiveMixing —— 连续时序位置编码

#### 3.3.1 新增参数

```python
# 替代原始的 nn.Embedding(num_frames, eff_in_dim)
# 使用连续 MLP 编码器: 1 → 16 → C_g
self.temporal_pos_encoder = nn.Sequential(
    nn.Linear(1, 16),           # 参数: 1×16 + 16 = 32
    nn.ReLU(inplace=True),
    nn.Linear(16, 64),          # 参数: 16×64 + 64 = 1,088
)                                # 总参数: 1,120
# 初始化: 第二层(nn.Linear(16,64)) weight=zeros, bias=zeros
# 效果: 初始时编码器输出全零向量，不干扰原始特征
```

#### 3.3.2 新增参数：num_frames

```python
def __init__(self, in_dim, in_points, n_groups=1, ..., num_frames=8):
    self.num_frames = num_frames                      # 8
    self.points_per_frame = in_points // num_frames   # 32 // 8 = 4
```

#### 3.3.3 接口变更

```python
# 原始: def forward(self, x, query)
# 改进: def forward(self, x, query, time_diff=None)
```

#### 3.3.4 inner_forward 中的计算

```python
def inner_forward(self, x, query, time_diff=None):
    B, Q, G, P, C = x.shape
    # x: [B, Q, 4, 32, 64]  (G=4, F*P=32, C_g=64)

    if time_diff is not None:
        # time_diff: [B, F] = [B, 8]
        td = time_diff.mean(dim=0, keepdim=False)    # [F] = [8]     取batch均值
        td = td.unsqueeze(-1)                         # [F, 1] = [8, 1]

        temp_pos = self.temporal_pos_encoder(td)       # [8, 1] → Linear(1,16) → ReLU
                                                        # → Linear(16,64) → [8, 64]

        temp_pos = temp_pos.unsqueeze(1)               # [8, 1, 64]
        temp_pos = temp_pos.expand(F, P_per_frame, C)  # [8, 4, 64]
        temp_pos = temp_pos.reshape(1, 1, 1, P, C)     # [1, 1, 1, 32, 64]

        x = x + temp_pos                               # [B,Q,4,32,64] + [1,1,1,32,64]
                                                         # broadcast → [B, Q, 4, 32, 64]
    # ... 后续 channel mixing + point mixing 不变
```

**张量流**：
```
time_diff [B, 8]
  → mean(dim=0) → [8]
  → unsqueeze(-1) → [8, 1]
  → Linear(1→16) → ReLU → Linear(16→64) → [8, 64]
  → unsqueeze(1) → [8, 1, 64]
  → expand → [8, 4, 64]
  → reshape → [1, 1, 1, 32, 64]
  → + x [B, Q, 4, 32, 64]  (broadcast)
```

#### 3.3.5 残差连接移除

```python
# 原始:
out = self.out_proj(out)    # [B, Q, 256]
out = query + out            # 残差连接

# 改进:
out = self.out_proj(out)    # [B, Q, 256]
# 不做残差，返回纯增量
return out                   # [B, Q, 256]  (mixing_delta)
```

---

### 3.4 SparseBEVTransformerDecoderLayer —— 运动自适应门控

#### 3.4.1 新增参数

```python
self.motion_gate = nn.Sequential(
    nn.Linear(2, 32),           # 参数: 2×32 + 32 = 96
    nn.ReLU(inplace=True),
    nn.Linear(32, 256),         # 参数: 32×256 + 256 = 8,448
    nn.Sigmoid()                # 输出值域: (0, 1)
)                                # 总参数: 8,544
```

#### 3.4.2 初始化策略

```python
# 目标: 初始时 gate ≈ 0.88，接近原始完整残差连接
nn.init.zeros_(self.motion_gate[0].weight)    # Linear(2,32).weight = 0
nn.init.zeros_(self.motion_gate[0].bias)      # Linear(2,32).bias = 0
nn.init.zeros_(self.motion_gate[2].weight)    # Linear(32,256).weight = 0
nn.init.constant_(self.motion_gate[2].bias, -2.0)  # Linear(32,256).bias = -2.0

# 计算过程 (任意输入 vel):
# layer1: 0*vel + 0 = [0,...,0] (32维)
# ReLU: [0,...,0]
# layer2: 0*[0,...,0] + (-2.0) = [-2.0,...,-2.0] (256维)
# Sigmoid: sigmoid(-2.0) = 0.119
# gate = 1.0 - 0.119 = 0.881 ≈ 0.88
```

#### 3.4.3 forward 中的计算

```python
# ---- 原始 ----
query_feat = self.norm2(self.mixing(sampled_feat, query_feat))

# ---- 改进 ----
# 1. mixing 返回纯增量 (无残差)
mixing_delta = self.mixing(sampled_feat, query_feat, img_metas[0]['time_diff'])
# mixing_delta: [B, Q, 256]

# 2. 运动自适应门控
vel = query_bbox[..., 8:10].detach()          # [B, Q, 2]      单位: m/s, 无梯度
gate_raw = self.motion_gate(vel)               # [B, Q, 2] → [B, Q, 256]  值域: (0,1)
gate = 1.0 - gate_raw                          # [B, Q, 256]
# 速度大 → gate_raw 大 → gate 小 → 减弱多帧融合增量
# 速度小 → gate_raw 小 → gate 大 → 充分利用多帧融合

# 3. 门控残差连接
query_feat = self.norm2(
    query_feat + gate * mixing_delta            # [B,Q,256] + [B,Q,256]*[B,Q,256]
)                                                # LayerNorm → [B, Q, 256]
```

**张量流**：
```
vel [B,Q,2] → Linear(2,32) → ReLU → Linear(32,256) → Sigmoid → [B,Q,256]
  → 1.0 - (·) → gate [B,Q,256]

mixing_delta [B,Q,256]
  → × gate [B,Q,256] → [B,Q,256]
  → + query_feat [B,Q,256]
  → LayerNorm → [B,Q,256]
```

---

## 四、参数量统计

### 4.1 逐项明细

| 模块 | 参数名 | 形状 | 参数量 | 初始化 |
|------|--------|------|--------|--------|
| SparseBEVSampling | temporal_scale_bias | [8, 4] | 32 | zeros |
| SparseBEVSampling | temporal_refine.weight | [32, 256] | 8,192 | zeros |
| SparseBEVSampling | temporal_refine.bias | [32] | 32 | zeros |
| SparseBEVSampling | velocity_correction.weight | [16, 256] | 4,096 | zeros |
| SparseBEVSampling | velocity_correction.bias | [16] | 16 | zeros |
| AdaptiveMixing | temporal_pos_encoder[0].weight | [16, 1] | 16 | default (kaiming) |
| AdaptiveMixing | temporal_pos_encoder[0].bias | [16] | 16 | default (zeros) |
| AdaptiveMixing | temporal_pos_encoder[2].weight | [64, 16] | 1,024 | zeros |
| AdaptiveMixing | temporal_pos_encoder[2].bias | [64] | 64 | zeros |
| DecoderLayer | motion_gate[0].weight | [32, 2] | 64 | zeros |
| DecoderLayer | motion_gate[0].bias | [32] | 32 | zeros |
| DecoderLayer | motion_gate[2].weight | [256, 32] | 8,192 | zeros |
| DecoderLayer | motion_gate[2].bias | [256] | 256 | constant(-2.0) |
| **总计** | | | **22,032** | |

### 4.2 占比分析

- 原始 Decoder Layer 参数量: ~2.8M
- 新增参数量: 22,036
- **占比: ~0.79%**

> 注: Decoder Layer 参数在 6 层间共享，新增参数同样共享。

---

## 五、计算量 (FLOPs) 估算

以单次 decoder layer forward、单 query 为单位：

| 操作 | 算子 | 维度 | FLOPs |
|------|------|------|-------|
| velocity_correction | Linear(256→16) | [1,256]×[256,16] | 8,192 |
| tanh | 逐元素 | [1,16] | 16 |
| td_mask 乘法 | 逐元素 | [1,8,2] | 16 |
| temporal_scale_bias 加法 | 逐元素 | [4,8,4,4] | 512 |
| softmax (scale_weights) | 已存在，不算增量 | — | 0 |
| exp(-rate*t) | — | — | ~~已删除~~ |
| temporal_refine | Linear(256→32) + softmax | [1,256]×[256,32] | 16,384 |
| sampled_feats 加权 | 逐元素 | [4,32,64] | 8,192 |
| temporal_pos_encoder | Linear(1→16→64) | [8,1]→[8,16]→[8,64] | ~1,280 |
| 加法 (x + temp_pos) | 逐元素 | [4,32,64] | 8,192 |
| motion_gate | Linear(2→32→256) | [1,2]→[1,32]→[1,256] | ~8,320 |
| sigmoid | 逐元素 | [1,256] | 256 |
| gate 乘法 + 加法 | 逐元素 | [1,256]×2 | 512 |
| **总计** | | | **~52,096** |

原始 decoder layer 单 query FLOPs 估算:
- Self-Attention: ~Q×C = 900×256 ≈ 230K (简化)
- AdaptiveMixing parameter_generator: 256 × (4×(64×64+32×128)) ≈ 86K → FLOPs 同量级
- channel/point mixing: 矩阵乘法为主 ≈ 数百K
- 保守估计原始 per-query FLOPs ≈ 500K~1M

**新增 ~52K FLOPs/query ≈ 5~10%，考虑全局操作摊薄后实际 < 5%。**

---

## 六、数值范围与稳定性分析

| 变量 | 值域 | 保护措施 |
|------|------|---------|
| query_temporal_w (logits) | (-∞, +∞) | 零初始化→初始全零→softmax均匀 |
| validity_mask | (-13.8, 0] | log(fv+1e-6)，无效帧→大负值 |
| temporal_weights (softmax后×F) | 每帧 ≥ 0, sum=F | softmax 天然归一化 |
| frame_validity | [0, 1] | valid_mask 是 float，mean 不超 1 |
| fv[:,:,:,0] (当前帧) | [0.1, 1] | clamp(min=0.1) 保护 |
| vel_correction | [-2.0, 2.0] 米 | tanh 硬约束 |
| vel_correction (当前帧) | 0.0 | td_mask 屏蔽 |
| motion_gate 输出 | (0, 1) | Sigmoid |
| gate (1-motion_gate) | (0, 1) | 1-Sigmoid |
| gate (初始) | ≈ 0.88 | bias=-2.0 初始化 |
| temp_pos (初始) | [0,...,0] | 输出层零初始化 |

---

## 七、初始化行为分析

### 7.1 训练开始时的等效行为

由于所有新增模块均采用零初始化（或使初始输出接近原始行为），**训练开始时改进模型等效于原始模型**：

| 模块 | 初始行为 |
|------|---------|
| velocity_correction | 输出全零 → 不修正线性补偿 |
| temporal_refine | 输出全零 → softmax均匀 → 每帧权重=1.0（原始等权行为） |
| temporal_pos_encoder | 输出全零 → 不注入位置信息 |
| motion_gate | gate≈0.88 → 接近完整残差连接 |
| temporal_scale_bias | 全零 → 所有帧共享相同尺度权重（原始行为） |

### 7.2 训练过程中的学习方向

| 模块 | 学习内容 |
|------|---------|
| velocity_correction | 学习对非匀速运动的采样点位移修正 |
| temporal_refine | 学习每个 query 对各帧的关注程度（取代全局衰减） |
| temporal_pos_encoder | 学习将真实时间差映射为有区分性的位置编码 |
| motion_gate | 学习高速物体减弱时序融合，低速物体增强 |
| temporal_scale_bias | 学习远帧偏好低分辨率层、近帧偏好高分辨率层 |
| temporal_decay_rate | 学习最优的全局时序衰减速率 |

---

## 八、与原始模型的 diff 摘要

### 8.1 `models/sparsebev_sampling.py`

```diff
  # 函数末尾
- return final
+ # 计算 frame_validity
+ valid_mask = valid_mask.squeeze(-1)
+ valid_mask = valid_mask.reshape(B, T, Q, G, P)
+ valid_mask = valid_mask.permute(0, 2, 3, 1, 4)
+ frame_validity = valid_mask.mean(dim=-1)
+ return final, frame_validity
```

### 8.2 `models/sparsebev_transformer.py`

**SparseBEVSampling.__init__**: +4 个参数/层
**SparseBEVSampling.init_weights**: +2 个零初始化
**SparseBEVSampling.inner_forward**: +速度修正、+尺度偏置、+时序权重计算
**AdaptiveMixing.__init__**: +temporal_pos_encoder, +num_frames, +points_per_frame
**AdaptiveMixing.init_weights**: +编码器零初始化
**AdaptiveMixing.inner_forward**: +连续时序编码, -残差连接, +time_diff 参数
**AdaptiveMixing.forward**: +time_diff 参数
**DecoderLayer.__init__**: +motion_gate
**DecoderLayer.init_weights**: +motion_gate 初始化
**DecoderLayer.forward**: +门控残差逻辑, mixing 调用传 time_diff

---

## 九、配置兼容性

| 配置项 | 是否需要修改 | 说明 |
|--------|-------------|------|
| `r50_nuimg_704x256.py` | 不需要 | 所有新参数通过已有的 num_frames/num_points/etc 自动推导 |
| `vit_eva02_1600x640_trainval_future.py` | 不需要 | num_frames=15 时所有新参数自动适配 |
| `vov99_dd3d_1600x640_trainval_future.py` | 不需要 | 同上 |
| 预训练权重加载 | 需 strict=False | 忽略新增参数的缺失 |
