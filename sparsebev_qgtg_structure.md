# SparseBEV QGTG 结构说明

本文档详细说明当前 `QueryGuidedTemporalGate` 的实现结构，包含每一个算子以及对应的张量维度。

## 代码位置

- 模块位置：`SparseBEV/models/sparsebev_transformer.py`
- 类名：`QueryGuidedTemporalGate`
- 插入位置：`sampling -> QGTG -> flatten(T, P) -> AdaptiveMixing`

## 符号说明

- `B`：batch size
- `Q`：decoder query 数量
- `G`：group 数量
- `T`：时间帧数
- `P`：每帧采样点数
- `Cq`：query 特征维度
- `Cg`：每个 group 的采样特征维度
- `H`：QGTG 的隐藏维度

## 当前仓库中的默认维度

来自 `SparseBEV/configs/r50_nuimg_704x256.py`：

- `embed_dims = 256`
- `num_groups = 4`
- `num_frames = 8`
- `num_points = 4`
- `hidden_dim = 64`

因此：

- `Cq = 256`
- `G = 4`
- `Cg = 256 / 4 = 64`
- `H = 64`

默认输入输出形状为：

- `query_feat`：`[B, Q, 256]`
- `query_bbox`：`[B, Q, 10]`
- `sampled_feats_ts`：`[B, Q, 4, 8, 4, 64]`
- `time_diff`：`[B, 8]`
- `local_offset`：`[B, Q, 4, 4, 3]`
- `gated`：`[B, Q, 4, 8, 4, 64]`

其中 `Q` 是动态的。推理时通常为 `900`，训练时因为可能会拼接 denoising queries，实际值可能更大。

## 模块定义

当前构造函数为：

```python
QueryGuidedTemporalGate(
    query_dim=256,
    num_groups=4,
    group_dim=64,
    hidden_dim=64,
    score_dropout=0.1,
    lambda_gate=0.5,
    use_motion_prior=True,
    current_frame_bias=0.1,
    dump_alpha=False,
)
```

子模块如下：

1. `query_proj = Linear(256, 4 * 64 = 256)`
2. `hist_proj = Linear(64, 64)`
3. `cur_proj = Linear(64, 64)`
4. `diff_proj = Linear(64, 64)`
5. `offset_mlp = Linear(3, 64) -> SiLU -> Linear(64, 64)`
6. `time_mlp = Linear(3, 64) -> SiLU -> Linear(64, 64)`
7. `motion_mlp = Linear(4, 64) -> SiLU -> Linear(64, 64)`，当 `use_motion_prior=True` 时启用
8. `score_mlp = Linear(6 * 64 = 384, 64) -> SiLU -> Dropout(0.1) -> Linear(64, 1)`

这里最关键的一点是：

- `query_proj` 现在已经是 group-aware 的。
- 它不是把原始 256 维 query 按通道硬切成连续的 4 块。
- 它是先把完整 query 通过一个可学习投影映射到 `G * H` 维，再 reshape 成 `G` 个独立的 64 维 group-query 表示。

## Forward 输入

forward 接口为：

```python
forward(query_feat, query_bbox, sampled_feats_ts, time_diff, local_offset)
```

各输入含义如下：

1. `query_feat: [B, Q, Cq]`
   decoder 当前的 query 特征。

2. `query_bbox: [B, Q, 10]`
   布局为：
   `[cx, cy, cz, w, h, d, sin_yaw, cos_yaw, vx, vy]`

3. `sampled_feats_ts: [B, Q, G, T, P, Cg]`
   在 `sampling_4d(..., return_temporal=True)` 后保留时间结构的采样特征。

4. `time_diff: [B, T]`
   每个时间帧相对当前帧的时间差。

5. `local_offset: [B, Q, G, P, 3]`
   query 局部坐标系下的采样偏移。它来自：

```python
self.sampling_offset(query_feat).view(B, Q, G, P, 3)
```

   还没有做时间 warp，因此对所有历史帧共享同一套 point 几何位置。

## 整体计算图

完整计算流程如下：

```text
query_feat -------------------> query_proj ----------------------+
sampled_feats_ts ------------> hist_proj ------------------------+
sampled_feats_ts[:, :, :, 0] -> expand -> cur_proj -------------+
|sampled_feats_ts - x_cur| --> diff_proj -----------------------+
local_offset ----------------> offset_mlp -----------------------+
time_diff -------------------> stack(dt, |dt|, dt^2) -> time_mlp +
query_bbox(vx, vy) ---------> speed + is_current -> motion_mlp --+
                                                          add ----> temporal_feat

[q_feat, hist_feat, cur_feat, diff_feat, offset_feat, temporal_feat]
    -> concat
    -> score_mlp
    -> add current-frame bias
    -> softmax over time
    -> alpha
    -> residual gating: x_hist * (1 + lambda_gate * alpha)
    -> gated
```

## 逐步张量维度说明

### 1. 读取输入维度

```python
B, Q, G, T, P, _ = sampled_feats_ts.shape
```

默认配置下：

- `sampled_feats_ts`：`[B, Q, 4, 8, 4, 64]`
- `local_offset`：`[B, Q, 4, 4, 3]`

### 2. 历史特征分支

```python
x_hist = sampled_feats_ts
```

- 算子：恒等赋值
- 形状：`[B, Q, G, T, P, Cg]`
- 默认形状：`[B, Q, 4, 8, 4, 64]`

### 3. 当前帧参考特征分支

```python
x_cur = x_hist[:, :, :, :1, :, :].expand(B, Q, G, T, P, -1)
```

拆开后包含两步：

1. 先切出当前帧：

```python
x_hist[:, :, :, :1, :, :]
```

- 形状：`[B, Q, G, 1, P, Cg]`
- 默认形状：`[B, Q, 4, 1, 4, 64]`

2. 再沿时间维广播到所有帧：

```python
expand(B, Q, G, T, P, -1)
```

- 形状：`[B, Q, G, T, P, Cg]`
- 默认形状：`[B, Q, 4, 8, 4, 64]`

含义：

- `x_cur` 是当前帧采样特征复制到所有时间步之后的结果。
- 它作为所有历史帧的参考特征。

### 4. group-aware 的 query 分支

```python
q_feat = self.query_proj(query_feat)
q_feat = q_feat.reshape(B, Q, self.num_groups, -1)
q_feat = q_feat[:, :, :, None, None, :].expand(B, Q, G, T, P, -1)
```

拆开后如下：

1. 对完整 query 做投影：

```python
self.query_proj(query_feat)
```

- 算子：`Linear(256, 256)`
- 输入形状：`[B, Q, 256]`
- 输出形状：`[B, Q, 256]`

这个 256 维输出会被解释为 `4` 个 group，每个 group 对应 `64` 维。

2. reshape 成显式的 group 轴：

```python
reshape(B, Q, 4, 64)
```

- 形状：`[B, Q, G, H]`
- 默认形状：`[B, Q, 4, 64]`

3. 插入时间轴和点轴：

```python
q_feat[:, :, :, None, None, :]
```

- 形状：`[B, Q, G, 1, 1, H]`
- 默认形状：`[B, Q, 4, 1, 1, 64]`

4. 沿时间和采样点广播：

```python
expand(B, Q, G, T, P, -1)
```

- 形状：`[B, Q, G, T, P, H]`
- 默认形状：`[B, Q, 4, 8, 4, 64]`

含义：

- 每个 group 都有自己独立学习出来的 query 表示。
- 对同一个 `(b, q, g)` 而言，这个 query 表示在时间维和采样点维上保持不变。

### 5. 历史采样特征投影

```python
hist_feat = self.hist_proj(x_hist)
```

- 算子：`Linear(64, 64)`
- 输入形状：`[B, Q, G, T, P, 64]`
- 输出形状：`[B, Q, G, T, P, 64]`

### 6. 当前帧采样特征投影

```python
cur_feat = self.cur_proj(x_cur)
```

- 算子：`Linear(64, 64)`
- 输入形状：`[B, Q, G, T, P, 64]`
- 输出形状：`[B, Q, G, T, P, 64]`

### 7. 差分特征投影

```python
diff_feat = self.diff_proj(torch.abs(x_hist - x_cur))
```

拆开后如下：

1. 历史与当前参考特征做差：

```python
x_hist - x_cur
```

- 形状：`[B, Q, G, T, P, 64]`

2. 取绝对值：

```python
torch.abs(...)
```

- 形状：`[B, Q, G, T, P, 64]`

3. 再做线性投影：

```python
diff_proj(...)
```

- 算子：`Linear(64, 64)`
- 输出形状：`[B, Q, G, T, P, 64]`

含义：

- 这一路显式编码了每个历史采样特征与当前帧参考特征的差异程度。

### 8. 局部偏移分支

```python
offset_feat = self.offset_mlp(local_offset[:, :, :, None, :, :])
offset_feat = offset_feat.expand(B, Q, G, T, P, -1)
```

拆开后如下：

1. 给 `local_offset` 插入时间维：

```python
local_offset[:, :, :, None, :, :]
```

- 输入形状：`[B, Q, G, P, 3]`
- 输出形状：`[B, Q, G, 1, P, 3]`

2. 通过偏移编码 MLP：

```python
Linear(3, 64) -> SiLU -> Linear(64, 64)
```

- 输出形状：`[B, Q, G, 1, P, 64]`

3. 沿时间维广播：

```python
expand(B, Q, G, T, P, -1)
```

- 输出形状：`[B, Q, G, T, P, 64]`

含义：

- `local_offset` 表示每个点在 query 局部坐标系中的位置。
- 因为历史帧采样点是通过速度 warp 得到的，所以局部偏移对所有时间帧共享。
- 这一路让 QGTG 在 point-wise 打分时，不仅看到“采到了什么特征”，还显式知道“这个点在局部几何上处于什么位置”。

### 9. 时间编码分支

```python
dt = time_diff[:, None, None, :, None].expand(B, Q, G, T, P)
dt_feat = torch.stack([dt, dt.abs(), dt * dt], dim=-1)
temporal_feat = self.time_mlp(dt_feat)
```

拆开后如下：

1. 给 `time_diff` 插入维度并广播：

```python
time_diff[:, None, None, :, None]
```

- 形状：`[B, 1, 1, T, 1]`

```python
expand(B, Q, G, T, P)
```

- 形状：`[B, Q, G, T, P]`
- 默认形状：`[B, Q, 4, 8, 4]`

2. 构造显式时间特征：

```python
torch.stack([dt, dt.abs(), dt * dt], dim=-1)
```

- 形状：`[B, Q, G, T, P, 3]`
- 三个通道分别是：
  - `dt`
  - `|dt|`
  - `dt^2`

3. 时间 MLP：

```python
Linear(3, 64) -> SiLU -> Linear(64, 64)
```

- 输出形状：`[B, Q, G, T, P, 64]`

### 10. 当前帧指示量

```python
current_mask = torch.arange(T, device=x_hist.device)
current_mask = (current_mask.view(1, 1, 1, T, 1, 1) == 0).to(x_hist.dtype)
current_mask = current_mask.expand(B, Q, G, T, P, 1)
```

拆开后如下：

1. 生成 `0 ... T-1`：

- `torch.arange(T)` 形状为 `[T]`

2. reshape：

```python
view(1, 1, 1, T, 1, 1)
```

- 形状：`[1, 1, 1, T, 1, 1]`

3. 与 0 比较：

```python
== 0
```

- 形状：`[1, 1, 1, T, 1, 1]`
- 取值为：
  - 当前帧 `t = 0` 时为 `1`
  - 历史帧 `t > 0` 时为 `0`

4. 广播：

```python
expand(B, Q, G, T, P, 1)
```

- 形状：`[B, Q, G, T, P, 1]`

### 11. 运动先验分支

当前默认启用这一路。

```python
vel = query_bbox[..., 8:10][:, :, None, None, None, :]
vel = vel.expand(B, Q, G, T, P, -1)
speed = torch.norm(query_bbox[..., 8:10], dim=-1, keepdim=True)
speed = speed[:, :, None, None, None, :].expand(B, Q, G, T, P, -1)
motion_input = torch.cat([vel, speed, current_mask], dim=-1)
temporal_feat = temporal_feat + self.motion_mlp(motion_input)
```

拆开后如下：

1. 取速度：

```python
query_bbox[..., 8:10]
```

- 形状：`[B, Q, 2]`
- 两个通道分别是：
  - `vx`
  - `vy`

2. 扩展后得到：

- `vel`：`[B, Q, G, T, P, 2]`

3. 计算速度模长：

```python
torch.norm(query_bbox[..., 8:10], dim=-1, keepdim=True)
```

- 形状：`[B, Q, 1]`

4. 广播后得到：

- `speed`：`[B, Q, G, T, P, 1]`

5. 拼接运动输入：

```python
torch.cat([vel, speed, current_mask], dim=-1)
```

- 形状：`[B, Q, G, T, P, 4]`
- 四个通道分别是：
  - `vx`
  - `vy`
  - `speed`
  - `is_current`

6. 通过运动先验 MLP：

```python
Linear(4, 64) -> SiLU -> Linear(64, 64)
```

- 输出形状：`[B, Q, G, T, P, 64]`

7. 加到时间分支上：

```python
temporal_feat = temporal_feat + motion_mlp(motion_input)
```

- 最终 `temporal_feat` 形状仍是 `[B, Q, G, T, P, 64]`

含义：

- `temporal_feat` 最终同时包含时间编码和运动先验编码。

### 12. 拼接打分输入

```python
score_input = torch.cat(
    [q_feat, hist_feat, cur_feat, diff_feat, offset_feat, temporal_feat], dim=-1
)
```

- 每一路张量形状都是 `[B, Q, G, T, P, 64]`
- 在最后一个通道维上拼接
- 输出形状：`[B, Q, G, T, P, 384]`

这就是每个 `(b, q, g, t, p)` 位置用于打分的完整特征。

### 13. 打分网络

```python
scores = self.score_mlp(score_input)
```

`score_mlp` 结构为：

```python
Linear(384, 64) -> SiLU -> Dropout(0.1) -> Linear(64, 1)
```

因此：

- 输入形状：`[B, Q, G, T, P, 384]`
- 输出形状：`[B, Q, G, T, P, 1]`

解释：

- `scores[b, q, g, t, p, 0]` 是该时间帧 `t` 的未归一化时序分数。
- 它是对每个 query、每个 group、每个采样点分别独立计算的。
- 网络参数在不同位置之间是共享的，但输入不同，所以输出会不同。

### 14. 当前帧偏置

```python
scores = scores + current_mask * current_frame_bias
```

- `current_mask`：`[B, Q, G, T, P, 1]`
- `current_frame_bias`：标量，默认值 `0.1`
- `scores` 形状仍然保持为 `[B, Q, G, T, P, 1]`

效果：

- 只有当前帧 `t = 0` 会额外加上这个偏置。

### 15. 时间维归一化

```python
alpha = torch.softmax(scores, dim=3)
```

- 输入形状：`[B, Q, G, T, P, 1]`
- 输出形状：`[B, Q, G, T, P, 1]`

这里最重要的是：

- softmax 只在时间维 `T` 上做
- 所以对每个固定的 `(b, q, g, p)`，都有

```text
sum_t alpha[b, q, g, t, p, 0] = 1
```

也就是说，这个模块预测的是：

- 每个 `(batch, query, group, point)` 对应一组独立的时间分布

它不是：

- 整个 query 只共享一组时间分布
- 所有 group 共享一组时间分布
- 所有 point 共享一组时间分布

### 16. 残差式时序门控

```python
gated = x_hist * (1.0 + lambda_gate * alpha)
```

各张量形状为：

- `x_hist`：`[B, Q, G, T, P, 64]`
- `alpha`：`[B, Q, G, T, P, 1]`
- 缩放因子 `1.0 + lambda_gate * alpha`：`[B, Q, G, T, P, 1]`
- `gated`：`[B, Q, G, T, P, 64]`

默认超参数：

- `lambda_gate = 0.5`

含义：

- 每个时间帧的特征都会被乘上一个系数重新加权
- 该系数总落在 `[1, 1 + lambda_gate] = [1, 1.5]`
- 因此当前实现只会放大特征，不会把特征直接压到原始值以下

## QGTG 真正预测的是什么

QGTG 不直接预测：

- 类别
- 边界框

它真正预测的是：

- `scores[b, q, g, t, p]`
- 以及在时间维 softmax 之后得到的 `alpha[b, q, g, t, p]`

它真正返回的是：

- `gated[b, q, g, t, p, c]`

所以从学习目标上看，QGTG 学的是：

- 对每个 `(query, group, point)` 在时间维上的加权分布

## 输出给下游模块的形状

QGTG 返回：

- `gated`：`[B, Q, G, T, P, Cg]`

随后在外部会执行：

```python
gated = gated.flatten(3, 4)
```

得到：

- `[B, Q, G, T * P, Cg]`
- 默认形状：`[B, Q, 4, 32, 64]`

这样就能和原来的 `AdaptiveMixing` 接口完全对齐。

## 这次改动的核心总结

相对于更早的实现，这次有两处关键结构变化：

- query 分支：
  - 之前：`Linear(256, 64)`，同一个 `q_feat` 广播给所有 group
  - 现在：`Linear(256, 4 * 64)`，reshape 成 `[B, Q, 4, 64]`
  - 结果：每个 group 都有自己独立学习出的 query 表示

- point 几何分支：
  - 现在额外引入 `local_offset`
  - 通过 `offset_mlp` 编码后参与时间打分
  - 结果：QGTG 在 point-wise 打分时，显式知道每个采样点在 query 局部坐标中的位置

这样改之后，时序打分器同时具备了：

- query 侧的 group-aware 表达能力
- point 侧的局部几何感知能力

同时整体参数量和结构复杂度仍然保持得比较轻量。
