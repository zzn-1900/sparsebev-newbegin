# SparseBEV 时序融合改进索引

这个文件只保留当前版本的改动索引，避免和历史迭代方案混淆。  
详细结构、ASCII 图、张量维度、初始化和审查结论请看：

- `SparseBEV/temporal_fusion_final_summary.md`

## 当前版本只保留 3 个核心改动

### 1. Scale 分支

目标：

- 让 FPN 尺度选择真正依赖每个 frame 的真实条件

输入：

- `query_feat`
- `|dt|`
- `range_t`
- `size_log`

输出：

- `scale_weights: [B,Q,G,F,Ppf,L]`

结构：

```text
scale_ctx = ReLU(scale_query_proj(query_feat) + scale_motion_encoder([|dt|, range_t, size]))
scale_weights = softmax(scale_weights_head(scale_ctx), dim=L)
```

### 2. Temporal 分支

目标：

- 用真实时间距离而不是 frame slot 给各帧分配可信度

输入：

- `query_feat`
- `|dt|`
- `speed`

输出：

- `temporal_weights: [B,Q,G,F]`

结构：

```text
temporal_ctx = ReLU(temporal_query_proj(query_feat) + temporal_motion_encoder([|dt|, speed]))
temporal_weights = softmax(temporal_refine(temporal_ctx), dim=F) * F
```

### 3. Mixing 分支

目标：

- 在点级别显式建模时间距离和采样点局部几何关系

输入：

- `|dt|`
- `point_offset = sampling_point - query_center`

输出：

- `temp_pos: [B,Q,G,FP,C_g]`

结构：

```text
pos_desc = [|dt|, dx, dy, dz]
temp_pos = temporal_pos_encoder(pos_desc)
x = x + temp_pos
```

## 当前版本的设计口径

```text
scale    看: 时间 + 距离 + 尺寸
temporal 看: 时间 + 速度
mixing   看: 时间 + 点偏移
```

## 当前版本的审查结论

- 没有发现明显的 shape 或 softmax 维度错误
- 三个分支都已经统一到真实 `|dt|` 语义
- 初始化是平滑退化的：
  - `temporal_refine` 零初始化 -> 初始各帧等权
  - `scale_weights_head` 零初始化 -> 初始各层等权
  - `temporal_pos_encoder` 最后一层零初始化 -> 初始不扰动 mixing

## 还需要记住的边界

- 当前不区分过去/未来，只看离当前多远
- `range_t` 是 BEV 距离，不是 3D 距离
- `speed` 是原始 m/s，没有做压缩
