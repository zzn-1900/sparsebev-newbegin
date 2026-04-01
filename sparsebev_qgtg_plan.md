# SparseBEV Query-Guided Temporal Gate (QGTG) Implementation Plan

## 1. Goal

Implement a lightweight **Query-Guided Temporal Gate (QGTG)** for SparseBEV.

The purpose is to make **query features act as the controller of temporal fusion**, instead of treating the query as only auxiliary information.

The first version should:

- keep the existing `AdaptiveMixing` unchanged
- insert a temporal gating module **after `sampling_4d` and before `AdaptiveMixing`**
- explicitly use:
  - current-frame sampled features
  - historical sampled features
  - time difference
  - motion priors
  - optional warp residuals

This first version is designed to validate the core hypothesis with minimal modification.

---

## 2. Design Principle

Temporal fusion should not rely only on similarity between sampled features.

Instead, the weight of each temporal feature should be decided by:

- what the **current query** is looking for
- whether the **historical evidence** is consistent with the **current-frame evidence**
- whether the evidence is reasonable under **temporal and motion priors**

In short:

> temporal fusion should be implemented as **query-guided evidence selection**.

---

## 3. First-Version Scope

### Keep unchanged

- existing `AdaptiveMixing`
- existing dynamic parameter generation in `AdaptiveMixing`
- existing sampling strategy
- existing velocity-based temporal warp

### Add

- a new module: `QueryGuidedTemporalGate`
- optional temporal-format return from `sampling_4d`

### Do not do in v1

- do not replace `AdaptiveMixing`
- do not add full attention over all sampled tokens
- do not add extra auxiliary losses unless needed for stability

---

## 4. Insertion Point

Insert QGTG in the pipeline:

```text
sampling_4d -> QueryGuidedTemporalGate -> flatten(T,S) -> AdaptiveMixing
```

Current SparseBEV behavior:

```text
sampling_4d -> flatten(T,S) -> AdaptiveMixing
```

The only structural change is to preserve temporal structure before flattening.

---

## 5. Required File Changes

### 5.1 `models/sparsebev_sampling.py`

Modify `sampling_4d()` to optionally return temporal layout.

Current output:

- `[B, Q, G, FP, C]` where `FP = T * S`

New optional output:

- `[B, Q, G, T, S, C]`

Suggested interface:

```python
sampling_4d(..., return_temporal=False)
```

Behavior:

- `return_temporal=False`: keep original behavior
- `return_temporal=True`: return temporal layout before flattening

---

### 5.2 `models/sparsebev_transformer.py`

Add a new module:

```python
class QueryGuidedTemporalGate(nn.Module):
    ...
```

And call it after sampling, before `AdaptiveMixing`.

---

## 6. Tensor Shapes

Assume typical SparseBEV settings:

- `T = 8` frames
- `S = 4` sampled points per frame
- `G = 4` groups
- `Cg = embed_dims / G`

### Inputs to QGTG

#### 6.1 Query feature

```text
query_feat: [B, Q, Cq]
```

This is the decoder query feature.

---

#### 6.2 Query bbox

```text
query_bbox: [B, Q, 10]
```

Expected layout:

```text
[cx, cy, cz, w, h, d, sin_yaw, cos_yaw, vx, vy]
```

Used to extract motion prior.

---

#### 6.3 Sampled temporal features

```text
sampled_feats_ts: [B, Q, G, T, S, Cg]
```

This must be preserved before flattening.

---

#### 6.4 Time difference

```text
time_diff: [B, T]
```

Time difference of each frame relative to current frame.

---

#### 6.5 Optional sampling points

```text
sampling_points: [B, Q, T, G, S, 3]
```

Used to compute warp residuals.

If exact shape differs in implementation, adapt accordingly.

---

## 7. Features Used by the Gate

For each `(b, q, g, t, s)`, use the following information.

### 7.1 Current-frame sampled feature

```text
x_cur = sampled_feats_ts[:, :, :, 0, :, :]   # [B, Q, G, S, Cg]
```

This is the anchor evidence at the current time step.

---

### 7.2 Historical sampled feature

```text
x_t = sampled_feats_ts[:, :, :, t, s, :]
```

---

### 7.3 Time encoding

For each frame:

```text
dt = time_diff[:, t]
```

Recommended embedding input:

```text
[dt, |dt|, dt^2]
```

Then project with a small MLP.

---

### 7.4 Motion prior

From `query_bbox`:

```text
vel = query_bbox[..., 8:10]    # [B, Q, 2]
speed = ||vel||
```

Optional extra fields:

- current-frame indicator
- velocity magnitude
- normalized velocity direction

---

### 7.5 Warp residual (recommended)

Compute residual between current-frame sampling point and motion-warped historical sampling point.

Example:

```text
residual = p_t_warped_xy - p_cur_xy
res_norm = ||residual||
```

This helps the gate estimate whether temporal alignment is trustworthy.

---

## 8. Scoring Function

For each `(b, q, g, t, s)`, build a fused feature vector:

```text
z = [
    q,
    x_t,
    x_cur,
    |x_t - x_cur|,
    dt_emb,
    vel,
    speed,
    residual,
    res_norm,
    is_current
]
```

Then compute a scalar score:

```text
a = MLP(z)
```

Normalization should be applied **only over the time dimension**:

```text
alpha = softmax(a, dim=time)
```

---

## 9. Gating Rule

Use **residual-style gating** for stability.

Recommended form:

```text
x_tilde = (1 + lambda * alpha) * x_t
```

Where:

- `alpha` is normalized over time
- `lambda` is a small scalar, e.g. `0.5`

Why residual gating:

- more stable than direct overwrite
- preserves original sampled evidence
- avoids overly aggressive suppression in early training

---

## 10. Module Definition Suggestion

Suggested implementation:

```python
class QueryGuidedTemporalGate(nn.Module):
    def __init__(self, query_dim, group_dim, hidden_dim=64, use_motion_prior=True):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.hist_proj = nn.Linear(group_dim, hidden_dim)
        self.cur_proj = nn.Linear(group_dim, hidden_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        motion_in_dim = 2 + 1 + 2 + 1 + 1   # vel(2), speed(1), residual(2), res_norm(1), is_current(1)
        self.motion_mlp = nn.Sequential(
            nn.Linear(motion_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.lambda_gate = 0.5
```

Note:

- dimensions can be adjusted
- if motion prior is disabled, reduce the scorer input accordingly

---

## 11. Forward Pass Pseudocode

```python
# Inputs:
# query_feat:       [B, Q, Cq]
# query_bbox:       [B, Q, 10]
# sampled_feats_ts: [B, Q, G, T, S, Cg]
# time_diff:        [B, T]
# sampling_points:  optional

B, Q, G, T, S, Cg = sampled_feats_ts.shape

x_cur = sampled_feats_ts[:, :, :, 0, :, :]                 # [B, Q, G, S, Cg]
x_cur = x_cur.unsqueeze(3).expand(B, Q, G, T, S, Cg)       # [B, Q, G, T, S, Cg]
x_hist = sampled_feats_ts                                   # [B, Q, G, T, S, Cg]

q = query_feat[:, :, None, None, None, :]                  # [B, Q, 1, 1, 1, Cq]
q = q.expand(B, Q, G, T, S, query_feat.size(-1))

# time embedding
# build [dt, |dt|, dt^2]
dt = time_diff[:, None, None, :, None]                     # [B, 1, 1, T, 1]
dt = dt.expand(B, Q, G, T, S)                              # [B, Q, G, T, S]
dt_feat = torch.stack([dt, dt.abs(), dt * dt], dim=-1)     # [B, Q, G, T, S, 3]

# motion prior
vel = query_bbox[..., 8:10]                                # [B, Q, 2]
speed = torch.norm(vel, dim=-1, keepdim=True)              # [B, Q, 1]

vel = vel[:, :, None, None, None, :].expand(B, Q, G, T, S, 2)
speed = speed[:, :, None, None, None, :].expand(B, Q, G, T, S, 1)

# optional residual
# residual: [B, Q, G, T, S, 2]
# res_norm: [B, Q, G, T, S, 1]
# if unavailable, fill zeros

# current frame indicator
is_current = torch.zeros(B, Q, G, T, S, 1, device=x_hist.device)
is_current[:, :, :, 0, :, :] = 1.0

# projections
q_feat = self.query_proj(q)
h_feat = self.hist_proj(x_hist)
c_feat = self.cur_proj(x_cur)
d_feat = self.hist_proj(torch.abs(x_hist - x_cur))
t_feat = self.time_mlp(dt_feat)
m_input = torch.cat([vel, speed, residual, res_norm, is_current], dim=-1)
m_feat = self.motion_mlp(m_input)

score_input = torch.cat([q_feat, h_feat, c_feat, d_feat, t_feat + m_feat], dim=-1)
scores = self.score_mlp(score_input)                       # [B, Q, G, T, S, 1]

alpha = torch.softmax(scores, dim=3)

gated = x_hist * (1.0 + self.lambda_gate * alpha)
```

After gating:

```python
gated = gated.reshape(B, Q, G, T * S, Cg)
```

Then feed into the original `AdaptiveMixing`.

---

## 12. Integration into Existing Pipeline

### Original

```python
sampled_feats = sampling_4d(...)
out = adaptive_mixing(sampled_feats, query_feat)
```

### New

```python
sampled_feats_ts = sampling_4d(..., return_temporal=True)
gated_feats = temporal_gate(
    query_feat=query_feat,
    query_bbox=query_bbox,
    sampled_feats_ts=sampled_feats_ts,
    time_diff=time_diff,
    sampling_points=sampling_points,
)
gated_feats = gated_feats.reshape(B, Q, G, T * S, Cg)
out = adaptive_mixing(gated_feats, query_feat)
```

---

## 13. Initialization Recommendations

### 13.1 Score head

Initialize the last linear layer of `score_mlp` with small weights.

Goal:

- avoid unstable or overly sharp temporal weights at the beginning of training

### 13.2 Gate behavior

Initial gate should be close to identity.

Recommended settings:

- `lambda_gate = 0.5`
- small final-layer initialization in scorer

### 13.3 Current-frame bias (optional)

Can add a small bias to the current frame score for stability:

```text
scores[t=0] += 0.1
```

Do not make this too strong.

---

## 14. Optional Regularization

Only if needed for stability.

### Temporal entropy regularization

To prevent collapse to a single frame too early:

```text
L_ent = -mu * sum(alpha * log(alpha))
```

Keep `mu` very small.

This is optional for v1.

---

## 15. Acceptance Criteria

### Functional

- original behavior remains unchanged when QGTG is disabled
- temporal return path from `sampling_4d` works correctly
- QGTG output shape is `[B, Q, G, T, S, Cg]`
- flattened gated output is accepted by original `AdaptiveMixing`

### Training stability

- training loss does not explode in early iterations
- no NaN in temporal weights
- weights are not trivially collapsed to a single frame for all queries

### Qualitative checks

Visualize average temporal weights:

- per time step
- per sampled point
- per category
- dynamic vs static objects

Expected trends:

- current frame gets slightly higher weight on average
- near-history frames remain useful
- large warp residuals should reduce temporal confidence

---

## 16. Recommended Ablation Order

### A0
Baseline SparseBEV

### A1
QGTG with:

- query feature
- historical sampled feature
- current-frame sampled feature
- time difference

### A2
A1 + motion prior

### A3
A2 + warp residual

Use this order to isolate the contribution of each prior.

---

## 17. Summary

This task introduces a lightweight **Query-Guided Temporal Gate** into SparseBEV with minimal structural change.

Key properties:

- query becomes the controller of temporal fusion
- current-frame feature is explicitly used as temporal reference
- time difference is explicitly encoded
- motion prior is explicitly included
- existing `AdaptiveMixing` is preserved for clean validation

This is the recommended first-stage implementation before exploring more advanced temporal mixers or semantic-motion decoupling.
