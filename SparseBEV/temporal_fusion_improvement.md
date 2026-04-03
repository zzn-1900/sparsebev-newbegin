# SparseBEV 时序融合改动索引

这个文件只保留当前版本的结构索引，详细张量维度和 ASCII 图请看：

- `SparseBEV/temporal_fusion_final_summary.md`

## 当前版本保留的改动

### 1. Scale 分支

目标：

- 让多尺度采样权重依赖真实时间距离、传播后距离和物体尺寸

输入：

- `query_feat`
- `|dt|`
- `range_t`
- `size_log`

结构：

```text
scale_ctx = ReLU(scale_query_proj(query_feat) + scale_motion_encoder([|dt|, range_t, size_log]))
scale_weights = softmax(scale_weights_head(scale_ctx), dim=L)
```

### 2. Temporal 分支

目标：

- 让逐帧时序权重依赖真实时间距离和速度，而不是固定 frame slot

输入：

- `query_feat`
- `|dt|`
- `speed`

结构：

```text
temporal_ctx = ReLU(temporal_query_proj(query_feat) + temporal_motion_encoder([|dt|, speed]))
temporal_weights = softmax(temporal_refine(temporal_ctx), dim=F) * F
```

### 3. Mixing 分支

目标：

- 保持原始 query-driven adaptive mixing

结构：

```text
params = parameter_generator(query)
out = AdaptiveMixing(x, params)
```

## 本轮额外调整

- 去掉了 `mixing` 里的点级时空位置编码
- `temporal_context_dim` 从 `16` 提到 `32`
- 在 `scale_motion_encoder` 和 `temporal_motion_encoder` 中加入 `Dropout(0.1)`

## 当前设计口径

```text
scale    看: 时间 + 距离 + 尺寸
temporal 看: 时间 + 速度
mixing   保持原始 query-driven mixing
```
