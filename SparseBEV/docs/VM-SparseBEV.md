# VM-SparseBEV 设计与实现文档

> **VM-SparseBEV**：Visual-Prior and Motion-aware Sparse 4D Detection
> 基于 SparseBEV 的两个轻量增强 ——
> **VPS**（视觉先验条件化采样点生成）+ **SOMCTS**（二阶运动补偿采样）

---

## 0. 一句话概述

VM-SparseBEV 在 SparseBEV 的 `SparseBEVSampling` 模块内部插入两个互补的轻量子模块：

- **VPS** 让每个 query 的采样点 offset *看到* 当前帧图像里它落在什么视觉区域（视觉条件化）
- **SOMCTS** 让每个 query 在长时序帧上 *预测* 自己的二阶运动（加速度 + 角速度），修正常速假设的偏差

所有改动都在 `SparseBEVSampling` 内完成，**不动** `sampling_4d` 的多视角聚合、不动 SASA、不动 bbox 维度（仍是 10D）。

---

## 1. 整体架构改动一览

```
                    ┌─────────────── ORIGINAL SparseBEV ────────────────┐
                    │                                                   │
[6 cam × 8 frame] ─►│  Backbone(R50) ─► FPN ─► mlvl_feats               │
                    │                                                   │
                    │  query (900×10) ─► [Decoder × 6] ─► cls + bbox    │
                    │       └─► self-attn ─► sampling ─► mixing ─► ffn  │
                    └─────────────────────┬─────────────────────────────┘
                                          │
                          ┌───────────────┴────────────────┐
                          │  WHAT CHANGED IN VM-SparseBEV  │
                          └───────────────┬────────────────┘
                                          │
        ┌─────────────────────────────────┼──────────────────────────────────┐
        │                                 │                                  │
   ▼ VPS branch                    ▼ SOMCTS branch                    ▼ Plumbing
   (NEW)                           (NEW)                              (modified)
                                                                      
   1. VisualPriorHead              1. SparseBEVSampling                1. SparseBEV.compute_visual_prior
      (1×1 conv on FPN[-1]            +motion_branch (256→3)              (calls VisualPriorHead on t=0)
       current frame, 6 cams)         predicts (a_x, a_y, ω)            2. Forward chain plumbing of
   2. SparseBEVSampling             2. inner_forward                       prior_map kwarg through
      +vps_fuse Linear(K+1→64)        2nd-order warp gated by              Head → Transformer →
      +_sample_visual_prior           |Δt| > somcts_long_dt                Decoder → Layer → Sampling
      (project + argmax view +
       3×3 patch + Linear)
```

---

## 2. 完整 ASCII 模型结构图

```
                                 INPUT
                                 ─────
                       img: [B, T·N, 3, H, W]
                       img_metas (lidar2img, timestamps, ...)
                       T = 8 frames, N = 6 cameras
                                  │
                                  ▼
              ┌───────────────────────────────────────┐
              │  Backbone (ResNet-50)                 │
              │  Input  : [B·T·N, 3, 256, 704]        │
              │  Output : 4 stages C2..C5             │
              └───────────────────────────────────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────────┐
              │  FPN  (in=[256,512,1024,2048] → 256)  │
              │  Output: 4-level mlvl_feats           │
              │    each: [B, T·N, 256, H_l, W_l]      │
              │    deepest L=-1 ≈ 1/32 (8×22)         │
              └───────────────────────────────────────┘
                                  │
        ┌─────────────────────────┴──────────────────────────────┐
        │                                                        │
        │ ┌────────────────────────────────────────────────┐     │
        │ │  ★ VPS (NEW): VisualPriorHead — current-frame  │     │
        │ │     only, on deepest FPN level                 │     │
        │ │                                                │     │
        │ │  feat_t0 = mlvl_feats[-1][:, :6]               │     │
        │ │           shape: [B, 6, 256, 8, 22]            │     │
        │ │     │                                          │     │
        │ │     ▼  Conv 1×1 (256 → K=32)                   │     │
        │ │     ▼  GroupNorm + ReLU                        │     │
        │ │     ▼  AvgPool 2× (extra_pool=2)               │     │
        │ │  prior_map: [B, 6, K=32, 4, 11]                │     │
        │ └────────────────────────────────────────────────┘     │
        │                          │                             │
        │                          │                             │
        │  query_bbox [B, Q=900, 10]                             │
        │  query_feat [B, Q, 256]                                │
        │                          │                             │
        ▼                          ▼                             ▼
  ┌────────────────────────────────────────────────────────────────────┐
  │  SparseBEVTransformer (6 weight-shared decoder layers)             │
  │                                                                    │
  │  for layer in 0..5:                                                │
  │     ┌───────────────────────────────────────────────────────┐      │
  │     │  SparseBEVTransformerDecoderLayer                     │      │
  │     │                                                       │      │
  │     │  query_feat ← +position_encoder(query_bbox.xyz)       │      │
  │     │  query_feat ← norm1(SASA(query_bbox, query_feat))     │      │
  │     │                          │                            │      │
  │     │                          ▼                            │      │
  │     │     ┌─────────── SparseBEVSampling ──────────────┐    │      │
  │     │     │                                            │    │      │
  │     │     │  ┌─── ★ VPS (in-sampling part) ───┐        │    │      │
  │     │     │  │  _sample_visual_prior(            │     │    │      │
  │     │     │  │     query_bbox, prior_map,        │     │    │      │
  │     │     │  │     lidar2img[t=0],               │     │    │      │
  │     │     │  │     image_h, image_w)             │     │    │      │
  │     │     │  │       │                           │     │    │      │
  │     │     │  │  step1: project Q centers         │     │    │      │
  │     │     │  │         to 6 cams → (u,v) per cam │     │    │      │
  │     │     │  │  step2: valid mask, argmax →      │     │    │      │
  │     │     │  │         pick 1 view per query     │     │    │      │
  │     │     │  │  step3: 3×3 patch grid_sample     │     │    │      │
  │     │     │  │         on selected view of       │     │    │      │
  │     │     │  │         prior_map                 │     │    │      │
  │     │     │  │  step4: avg-pool patch → c ∈ R^K  │     │    │      │
  │     │     │  │  step5: Linear K+1 → 64           │     │    │      │
  │     │     │  │       │                           │     │    │      │
  │     │     │  │  visual_prior: [B, Q, 64]         │     │    │      │
  │     │     │  └──────────────┬─────────────────────┘    │    │      │
  │     │     │                 │                          │    │      │
  │     │     │                 ▼                          │    │      │
  │     │     │  offset_input = cat([query_feat, vp])      │    │      │
  │     │     │                  ↑          ↑              │    │      │
  │     │     │              [B,Q,256]  [B,Q,64] (zero     │    │      │
  │     │     │                          init early)       │    │      │
  │     │     │                 │                          │    │      │
  │     │     │                 ▼                          │    │      │
  │     │     │  Linear  sampling_offset (320 → G·P·3)     │    │      │
  │     │     │     ↑ ★ widened from 256 → 320             │    │      │
  │     │     │                 │                          │    │      │
  │     │     │                 ▼                          │    │      │
  │     │     │  make_sample_points → expand × T frames    │    │      │
  │     │     │  sampling_points: [B,Q,T=8,G=4,P=4,3]      │    │      │
  │     │     │                 │                          │    │      │
  │     │     │  ┌──── ★ SOMCTS (warp section) ────┐       │    │      │
  │     │     │  │  motion = motion_branch(qf)     │       │    │      │
  │     │     │  │     [B,Q,3] = (a_x, a_y, ω)     │       │    │      │
  │     │     │  │                                 │       │    │      │
  │     │     │  │  long_mask = |Δt| > 1s (gated)  │       │    │      │
  │     │     │  │                                 │       │    │      │
  │     │     │  │  ► translate                    │       │    │      │
  │     │     │  │    dist = v·Δt + 0.5·a·Δt² ·m   │       │    │      │
  │     │     │  │  ► rotate                       │       │    │      │
  │     │     │  │    yaw_delta = ω·Δt · long_mask │       │    │      │
  │     │     │  │    pts.xy = R(yaw)·(pts-c)+c    │       │    │      │
  │     │     │  └─────────────────────────────────┘       │    │      │
  │     │     │                                            │    │      │
  │     │     │  pts.xy ← pts.xy − dist (apply warp)       │    │      │
  │     │     │                                            │    │      │
  │     │     │  scale_weights (Linear 256 → G·P·L)        │    │      │
  │     │     │       softmax over L=4 levels              │    │      │
  │     │     │                                            │    │      │
  │     │     │  ┌── sampling_4d (UNCHANGED) ──┐           │    │      │
  │     │     │  │ project pts to 6 cams       │           │    │      │
  │     │     │  │ argmax pick 1 view          │           │    │      │
  │     │     │  │ msmv_sampling (grid_sample) │           │    │      │
  │     │     │  └─────────────┬───────────────┘           │    │      │
  │     │     └────────────────┼───────────────────────────┘    │      │
  │     │                      │                                │      │
  │     │       sampled_feat: [B, Q, G=4, F·P=32, C=64]         │      │
  │     │                      ▼                                │      │
  │     │           AdaptiveMixing (UNCHANGED)                  │      │
  │     │                      ▼                                │      │
  │     │           query_feat = norm2(...)                     │      │
  │     │                      ▼                                │      │
  │     │           query_feat = norm3(FFN(query_feat))         │      │
  │     │                      ▼                                │      │
  │     │      cls_branch & reg_branch & refine_bbox            │      │
  │     │                                                       │      │
  │     │      → (cls_score, bbox_pred[B,Q,10])                 │      │
  │     │      bbox_pred is detached → next layer's query_bbox  │      │
  │     └───────────────────────────────────────────────────────┘      │
  └────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                       cls_scores: [L=6, B, Q, 10]
                       bbox_preds: [L=6, B, Q, 10]
                                  │
                                  ▼
                    SparseBEVHead loss / NMS-free decode
                                  │
                                  ▼
                              outputs

  Legend:
    ★ = NEW or MODIFIED for VM-SparseBEV
    Q = 900 queries, T = 8 frames, N = 6 cameras
    G = 4 sampling groups, P = 4 sampling points/group/frame
    K = vps_channels = 32, vps_fuse_dim = 64
```

---

## 3. 创新 1 — VPS 详细设计

### 3.1 模块组成（两段式）

| 阶段 | 位置 | 模块 | 输入 → 输出 |
|------|------|------|-------------|
| **生成 prior_map** | `SparseBEV.compute_visual_prior` | `VisualPriorHead`（在 `models/visual_prior_head.py`） | FPN[-1] 当前帧 6 视角 [B, 6, 256, 8, 22] → [B, 6, 32, 4, 11] |
| **逐 query 采样** | `SparseBEVSampling._sample_visual_prior`（每层 decoder 调用一次） | 投影 + argmax view + 3×3 patch + Linear(K+1→64) | [B, Q, 10] + prior_map → [B, Q, 64] |

`VisualPriorHead` 在整个网络里**只调用 1 次**，生成的 `prior_map` 通过 `prior_map` kwarg 一路传到 6 个 decoder layer 的 sampling 模块复用。

### 3.2 VPS 内部数据流（详细 ASCII）

```
INPUT  ─────────────────────────────────────────────────────────────
  query_bbox   : [B, Q=900, 10]   ─ (cx, cy, cz, w, l, h, sin, cos, vx, vy)
  prior_map    : [B, 6, K=32, 4, 11]   ─ from VisualPriorHead
  lidar2img_t0 : [B, 6, 4, 4]     ─ projection of current frame
  image_h, image_w (scalars)
─────────────────────────────────────────────────────────────────────

STEP 1 — Decode + project query centers to all 6 cameras
─────────────────────────────────────────────────────────────────────
  xyz   = decode_bbox(query_bbox)[:, :, :3]            [B, Q, 3]
  pts   = cat([xyz, 1])                                [B, Q, 4]
  pts   = expand([B, N=6, Q, 4, 1])
  proj  = lidar2img_t0[:, :, None, :, :]               [B, N, Q, 4, 4]
  cam   = matmul(proj, pts).squeeze(-1)                [B, N, Q, 4]
  z     = cam[..., 2]                                  [B, N, Q]
  uv    = cam[..., :2] / max(z, eps)                   [B, N, Q, 2]
  u_norm, v_norm = uv[...,0]/W, uv[...,1]/H            [B, N, Q]

STEP 2 — Build per-view valid mask & pick ONE view via argmax
─────────────────────────────────────────────────────────────────────
  valid     = (z>0) & (0<u_norm<1) & (0<v_norm<1)      [B, N, Q]  float
  valid_qn  = valid.permute(B, Q, N)
  sel_view  = argmax(valid_qn, dim=-1, keepdim=True)   [B, Q, 1]
  sel_valid = gather valid at sel_view                 [B, Q]
              (= 1 if any view is valid for this query)

STEP 3 — Build a P×P grid in NORMALIZED feature-map coords ([-1,1])
─────────────────────────────────────────────────────────────────────
  P = vps_patch = 3
  offsets = linspace(-1, 1, 3)
  dx = offsets / (Wp-1)
  dy = offsets / (Hp-1)
  grid_y, grid_x = meshgrid(dy, dx)                    [3, 3]

  For each (B, N, Q), centered at (u_norm*2-1, v_norm*2-1):
  grid_all : [B, N, Q, 3, 3, 2]

STEP 4 — grid_sample on prior_map (all views) then GATHER selected
─────────────────────────────────────────────────────────────────────
  feat_in  = prior_map.reshape(B*N, K, Hp, Wp)
  grid_in  = grid_all.reshape(B*N, Q*P, P, 2)
  sampled  = F.grid_sample(feat_in, grid_in, bilinear, zero-pad)
                                                [B*N, K, Q*P, P]
  sampled  = sampled.reshape(B,N,K,Q,P,P).mean((-1,-2))
                                                [B, N, K, Q]
  sampled  = sampled.permute(B, Q, N, K)

  sel_idx     = sel_view.unsqueeze(-1).expand(B, Q, 1, K)
  sampled_sel = sampled.gather(2, sel_idx).squeeze(2)  [B, Q, K]
  sampled_sel = sampled_sel * sel_valid.unsqueeze(-1)
                ↑ zero out queries with no valid view

STEP 5 — Append valid bit + project to fuse_dim
─────────────────────────────────────────────────────────────────────
  feat = cat([sampled_sel, sel_valid.unsqueeze(-1)], dim=-1)   [B, Q, K+1=33]
  visual_prior = vps_fuse(feat)                                [B, Q, 64]

OUTPUT ─────────────────────────────────────────────────────────────
  visual_prior : [B, Q, vps_fuse_dim=64]
  → concat with query_feat → feeds the widened sampling_offset Linear
─────────────────────────────────────────────────────────────────────
```

### 3.3 关键设计选择

- **单视角 argmax 选择**与原 [`sampling_4d:102`](../models/sparsebev_sampling.py#L102) 完全相同的语义，保持架构一致性
- **3×3 patch + avg-pool** 比单点 grid_sample 更鲁棒，对半像素级投影误差有容忍
- **valid bit 显式输入** 给 `vps_fuse`，让 MLP 自己学会"该 query 投到相机外时 visual_prior 不可信"
- **完全无监督**：visual_prior_head 只靠下游检测 loss 端到端学习
- **软启动**：`vps_fuse.weight = vps_fuse.bias = 0` → 训练初期 `visual_prior = 0`，等价于 baseline

---

## 4. 创新 2 — SOMCTS 详细设计

### 4.1 模块组成（自包含于 sampling）

| 子模块 | 位置 | 作用 |
|--------|------|------|
| `motion_branch` | `SparseBEVSampling.__init__` | `Linear(256, 3)` 预测 (a_x, a_y, ω) |
| 二阶 warp 公式 | `SparseBEVSampling.inner_forward` | 把采样点按 motion 状态在长时序帧上做修正 |

**bbox 维度保持 10D 不变**——bbox encode/decode、Hungarian matcher、NMS-free coder、DN loss 全部一行不改，对 baseline checkpoint 完全可加载。

### 4.2 SOMCTS warp 公式

```
─────────────────────────────────────────────────────────────────────
  per query, per frame t (t = 0 is current frame):
    Δt = current_timestamp - frame_t_timestamp     (scalar, ≥ 0)
    long_mask = (|Δt| > somcts_long_dt) ? 1 : 0    (default long_dt = 1s)

  TRANSLATION  (replaces sparsebev_transformer.py:290-295)
    dist_xy = v·Δt          + 0.5·a·Δt² · long_mask
              └──first─┘    └────second-order────┘
    pts_xy ← pts_xy − dist_xy

  ROTATION  (NEW, only when long_mask=1)
    yaw_delta = ω · Δt · long_mask
    For each sampling point pts:
      rel = pts_xy − query_center_xy
      pts_xy = query_center_xy + R(yaw_delta) · rel
    where R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]
─────────────────────────────────────────────────────────────────────
```

ASCII 说明：

```
        BEV (top-down) view, current frame at t=0
                       ▲ y
                       │
                       │           ★ original sample point (offset from query c)
                       │          ╱
                       │  rel    ╱
                  c ───────────►●  query center
                  │   (vehicle)  ╲
                  │               ╲
                  │   long-horizon frame Δt = 2s, motion=(v=10m/s, a=2m/s², ω=0.05rad/s)
                  │
        Δt ──────►│       1) translate by  v·Δt + 0.5·a·Δt² = 22m
                  │       2) rotate around c by ω·Δt = 5.7°
                  │
                  │
        SOMCTS-warped sample point:  ★'  (lands on the actual object's
                                          predicted past position)
```

**关键点**：

1. **`motion_branch` 输出 (a, ω) 是 query 的属性**，每层独立预测（不跨层共享）；初始化为零，等价一阶 warp。
2. **门控 (long_mask)** 避免短时序帧上的二阶噪声放大。
3. **(a, ω) 暂无显式监督**——纯靠 sampling-warp 让下游检测 loss 反向传导驱动；如果训完一两个 epoch 后 motion 仍接近 0，可加 GT 有限差分弱监督（备选方案）。

---

## 5. 完整文件改动清单

### 5.1 新建文件

| 路径 | 行数 | 内容 |
|------|------|------|
| [`models/visual_prior_head.py`](../models/visual_prior_head.py) | ~50 | `VisualPriorHead` 类（1×1 Conv + GN + ReLU + 可选 avg-pool） |
| [`docs/VM-SparseBEV.md`](VM-SparseBEV.md) | （本文件） | 设计文档 |
| [`verify_vm.py`](../verify_vm.py) | ~100 | 一键运行时诊断脚本 |

### 5.2 修改文件

#### `models/sparsebev_transformer.py`

| 类 / 函数 | 改动 |
|-----------|------|
| `SparseBEVTransformer.__init__` | 新增 6 个 kwargs：`use_vps, vps_channels, vps_patch, vps_fuse_dim, use_somcts, somcts_long_dt`，全部转发给 decoder |
| `SparseBEVTransformer.forward` | 新增 `prior_map=None` kwarg |
| `SparseBEVTransformerDecoder.__init__` | 同上转发 |
| `SparseBEVTransformerDecoder.forward` | 接 `prior_map`，循环传给每个 decoder layer |
| `SparseBEVTransformerDecoderLayer.__init__` | 同上转发到 `SparseBEVSampling` |
| `SparseBEVTransformerDecoderLayer.forward` | 接 `prior_map`，传给 sampling |
| `SparseBEVSampling.__init__` | **核心改动**：新增 `vps_fuse`、`motion_branch`；`sampling_offset` 输入维度 256 → 256+vps_fuse_dim |
| `SparseBEVSampling.init_weights` | 对 `vps_fuse` / `motion_branch` 做 zero-init（软启动） |
| `SparseBEVSampling._sample_visual_prior` | **NEW** 方法（VPS 单 query 视觉先验采样，~50 行） |
| `SparseBEVSampling.inner_forward` | 接 `prior_map`；新增 VPS 路径；新增 SOMCTS 二阶 warp 路径 |
| `SparseBEVSampling.forward` | 把 `prior_map` 传给 inner_forward / checkpoint 包装 |

#### `models/sparsebev.py`

| 类 / 函数 | 改动 |
|-----------|------|
| `SparseBEV.__init__` | 新增 2 个 kwargs：`visual_prior_head=None, vps_feat_level=-1`；如果 `visual_prior_head` 不为 None，实例化 `VisualPriorHead` |
| `SparseBEV.compute_visual_prior` | **NEW** 方法：从 `pts_feats[vps_feat_level]` 取 t=0 的 6 视角喂给 `VisualPriorHead` |
| `SparseBEV.forward_pts_train` | 计算 `prior_map` 并通过 `pts_bbox_head` kwarg 传下去 |
| `SparseBEV.simple_test_pts` | 同上 |

#### `models/sparsebev_head.py`

| 类 / 函数 | 改动 |
|-----------|------|
| `SparseBEVHead.forward` | 新增 `prior_map=None` kwarg，转发给 `self.transformer` |

#### `configs/r50_nuimg_704x256.py`

| 区块 | 改动 |
|------|------|
| 文件顶部 | 新增 7 个开关：`use_vps, vps_channels, vps_patch, vps_fuse_dim, vps_extra_pool, use_somcts, somcts_long_dt` |
| `model.visual_prior_head` | 新增子配置（在 `use_vps=True` 时为 dict，否则 None） |
| `model.vps_feat_level` | 新增（默认 -1，即 FPN 最深层） |
| `model.pts_bbox_head.transformer` | 新增 6 个透传 kwargs |

---

## 6. 关键超参速查

| 超参 | 默认 | 含义 |
|------|------|------|
| `use_vps` | True | 是否启用 VPS |
| `vps_channels` (K) | 32 | prior_map 描述子通道数 |
| `vps_patch` (P) | 3 | grid_sample patch 边长（3×3） |
| `vps_fuse_dim` | 64 | 注入 sampling_offset 的 visual_prior 维度 |
| `vps_extra_pool` | 2 | VisualPriorHead 之后的额外 avg-pool 步长 |
| `vps_feat_level` | -1 | 取哪个 FPN 层，-1 = 最深（最小空间分辨率） |
| `use_somcts` | True | 是否启用 SOMCTS |
| `somcts_long_dt` | 1.0 | 仅 \|Δt\| > 该值（秒）的帧启用二阶项；越大越保守 |

---

## 7. 维度速查表（r50_704x256 默认配置，B=8）

| 张量 | shape |
|------|-------|
| 输入 img | [8, 48, 3, 256, 704] |
| FPN 最深层 mlvl_feats[-1] | [8, 48, 256, 8, 22] |
| 当前帧 6 视角特征 (feat_t0) | [8, 6, 256, 8, 22] |
| `prior_map` | [8, 6, 32, 4, 11] (extra_pool=2) |
| query_bbox | [8, 900, 10] |
| query_feat | [8, 900, 256] |
| `visual_prior` | [8, 900, 64] |
| `sampling_offset` 输入 | [8, 900, 320] |
| `sampling_offset` 输出 | [8, 900, 16, 3] (G·P=16) |
| `sampling_points` (after expand × T) | [8, 900, 8, 4, 4, 3] |
| `motion = motion_branch(qf)` | [8, 900, 3] |
| `sampled_feat` (sampling_4d 输出) | [8, 900, 4, 32, 64] |

---

## 8. 软启动机制

VPS 与 SOMCTS 都遵循 SparseBEV 风格的"零初始化软启动"：

| 模块 | 初始化 | 早期行为 |
|------|--------|---------|
| `vps_fuse.weight` / `bias` | zeros | `visual_prior = 0` → 等价于 baseline 的 sampling_offset |
| `motion_branch.weight` / `bias` | zeros | `(a, ω) = 0` → 二阶项消失，等价于一阶 warp |
| `VisualPriorHead.proj.weight` | Kaiming-normal | 即使 `vps_fuse=0`，prior_map 本身仍开始学习有意义的特征 |
| `sampling_offset.weight` | zeros (原版保持) | `bias = uniform(-0.5, 0.5)`，输出固定 offset，等同 baseline |

**含义**：训练**前几百 iter**的 loss 曲线和 baseline 几乎重合是预期的；从第 1-2 个 epoch 开始 `vps_fuse` 与 `motion_branch` 才会逐步学到非零权重，新模块开始真正起作用。

如果想立刻打破软启动（例如做 quick-debug），可以把 `init_weights` 里的 zero-init 换成小幅随机初始化（但损失训练稳定性）。

---

## 9. 计算量与参数量分析

新增参数（r50_704x256，K=32, vps_fuse_dim=64）：

| 模块 | 参数量 |
|------|--------|
| `VisualPriorHead.proj` (Conv 1×1, 256→32) | 256 × 32 + 32 = 8,224 |
| `VisualPriorHead.norm` (GroupNorm) | 64 |
| `vps_fuse` (Linear 33→64) | 33 × 64 + 64 = 2,176 |
| `sampling_offset` 多出的 64 输入维 → 输出 48 | 64 × 48 = 3,072 |
| `motion_branch` (Linear 256→3) | 256 × 3 + 3 = 771 |
| **合计** | **~14K (×6 layers for sampling内部)** ≈ 36K |

相对 ResNet-50 backbone 的 23M + 整网约 35M 参数，新增 **<0.1%**。

新增 FLOPs 估算（每 sample，B=1）：

| 模块 | FLOPs 量级 | 占比 |
|------|-----------|------|
| `VisualPriorHead` (1×1 Conv) | ~8M | <0.01% |
| `_sample_visual_prior` × 6 layers | ~18M | ~0.05% |
| `sampling_offset` 多出的 64 输入维 × 6 layers | ~16M | ~0.05% |
| `motion_branch` × 6 layers | ~4M | <0.01% |
| **合计** | **~46M** | **<0.1%** |

baseline SparseBEV r50_704x256 总 FLOPs ≈ 80G，所以**速度变化在测量噪声内**——这是设计意图，**不是 bug**。

---

## 10. 验证已接入

跑根目录下的 `verify_vm.py`：

```bash
cd SparseBEV
python verify_vm.py
```

关键检查项：
1. `model.visual_prior_head` 是 `VisualPriorHead` 实例
2. `sampling.vps_fuse` 是 `Linear(in_features=33, out_features=64)`
3. `sampling.motion_branch` 是 `Linear(in_features=256, out_features=3)`
4. `sampling.sampling_offset.in_features == 320`（256 + 64）
5. 前向 + 反向后 3 个新模块的 `grad-norm > 0`

5 条全部通过即说明 VPS / SOMCTS 完全接入并参与梯度回传。

---

## 11. 后续工作

按 plan ([`abundant-greeting-fern.md`](file:///C:/Users/zzn/.claude/plans/abundant-greeting-fern.md)) 的实施顺序：

- [ ] **Week 1** baseline 复现 + CR-AuxCent 辅助技巧
- [x] **Week 2** VPS 实现（已完成）
- [ ] **Week 3** VPS 子消融（K, P, extra_pool, feat_level）
- [x] **Week 4** SOMCTS 实现（已完成；如学不动则补 (a, ω) GT 弱监督）
- [ ] **Week 5** 全消融 + R101 主表 + 可视化
