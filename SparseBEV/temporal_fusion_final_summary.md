# SparseBEV 当前时序融合结构总览

本文档以当前代码实现为准，聚焦 3 个已经落地的改动：

1. `scale` 分支：用 `query_feat + [|dt|, range_t, size]` 直接生成多尺度采样权重
2. `temporal` 分支：用 `query_feat + [|dt|, speed]` 生成逐帧可信度权重
3. `mixing` 分支：用 `[|dt|, point_offset]` 给采样点注入时序几何编码

对应代码文件：

- `SparseBEV/models/sparsebev_transformer.py`

## 1. 审查结论

基于当前实现做了一次代码级审查，结论是：

- 这 3 个改动之间的语义是自洽的，没有发现明显的 shape 错误、softmax 维度错误或张量顺序错误。
- 三个分支都已经从“按 frame slot 编号建模”切到“按真实时间距离 `|dt|` 建模”，这一点和数据加载的随机时间间隔设定是匹配的。
- `scale`、`temporal`、`mixing` 的条件变量划分也基本合理：谁负责尺度、谁负责帧可信度、谁负责点间几何关系，边界比较清楚。

目前没有看到必须马上修改的硬错误，但有 3 个设计边界需要明确：

- 当前所有学习分支都只看 `|dt|`，不区分过去和未来。如果以后你希望模型显式地区分“历史帧”和“未来帧”，需要把有符号 `dt` 再引回来。
- `scale` 分支里的 `range_t` 目前是 BEV 平面距离 `sqrt(x^2 + y^2)`，不是 3D 欧式距离。这对尺度选择通常是合理的，因为 FPN 尺度更主要受横向距离影响。
- `temporal` 分支里的 `speed` 现在直接使用原始米每秒，没有做 `log1p` 或 clipping。当前不算 bug，但如果后面训练波动明显，可以优先检查这个输入量级。

已完成的本地检查：

- `python3 -m py_compile SparseBEV/models/sparsebev_transformer.py`

说明：

- 当前 shell 里的 `python3` 没有 `torch`，所以这次没有做随机张量前向单测；下面的结论来自代码级 shape 审查和初始化分析。

## 2. 符号约定

| 符号 | 含义 |
|---|---|
| `B` | batch size |
| `Q` | query 数量 |
| `F` | 时间帧数 `num_frames` |
| `G` | sampling group 数 |
| `Ppf` | 每帧每 group 的采样点数 `num_points` |
| `FP` | 总采样点数，`FP = F * Ppf` |
| `L` | FPN level 数 |
| `C` | query/embed 通道数 |
| `H` | 条件分支隐藏维度，当前为 `16` |
| `C_g` | group 内通道数，`C_g = C / G` |

在默认配置里通常是：

```text
C   = 256
F   = 8
G   = 4
Ppf = 4
FP  = 32
L   = 4
H   = 16
C_g = 64
```

## 3. 总体流程

```text
query_bbox [B,Q,10]          query_feat [B,Q,C]
        |                           |
        | position_encoder          |
        +------------ add ----------+
                    |
             self_attn + norm
                    |
                    v
        +-------------------------------+
        |      SparseBEVSampling        |
        |                               |
        |  1) sampling_offset           |
        |  2) make_sample_points        |
        |  3) dist = vel * dt           |
        |  4) temporal_ctx              |
        |  5) scale_ctx                 |
        |  6) scale_weights             |
        |  7) sampling_4d               |
        |  8) temporal_weights          |
        +-------------------------------+
                    |
     sampled_feat [B,Q,G,FP,C_g]
     point_offset [B,Q,G,FP,3]
                    |
                    v
        +-------------------------------+
        |         AdaptiveMixing        |
        |                               |
        |  pos_desc = [|dt|, dx,dy,dz]  |
        |  temp_pos -> add to x         |
        |  channel mixing               |
        |  point mixing                 |
        |  out_proj + residual          |
        +-------------------------------+
                    |
              norm + ffn
                    |
               cls / reg
```

## 4. 三个改动的结构细节

### 4.1 Scale 分支

目标：

- 在每个 frame 上，根据真实时间距离、该时刻物体离原点的距离、物体尺寸，决定该 frame 的采样点更适合从哪个 FPN level 取特征。

核心设计：

- 不再做“原始 `scale_weights(query_feat)` + 时序偏置”的加法修正。
- 改成直接用条件化后的 `scale_ctx` 生成完整的 `[G, Ppf, L]` logits。

#### 4.1.1 输入量

```text
query_feat : [B, Q, C]
time_diff  : [B, F]
range_t    : [B, Q, F, 1]
size_log   : [B, Q, 3]
```

其中：

- `time_diff` 是当前帧与各个时序帧的真实时间差，单位秒
- `range_t` 是传播到该 frame 后，query center 在 BEV 平面到原点的距离
- `size_log` 是 box 的对数尺寸，直接来自 `query_bbox[..., 3:6]`

#### 4.1.2 张量流

```text
query_feat [B,Q,C]
  -> scale_query_proj
  -> [B,Q,H]
  -> unsqueeze(2)
  -> query_ctx [B,Q,1,H]

time_diff [B,F]
  -> abs
  -> [B,1,F,1]
  -> expand
  -> td_abs [B,Q,F,1]

range_t [B,Q,F,1]
  -> log1p
  -> range_feat [B,Q,F,1]

size_log [B,Q,3]
  -> unsqueeze(2) + expand
  -> size_feat [B,Q,F,3]

cat([td_abs, range_feat, size_feat], dim=-1)
  -> scale_desc [B,Q,F,5]
  -> scale_motion_encoder
  -> [B,Q,F,H]

query_ctx [B,Q,1,H]
  + motion_ctx [B,Q,F,H]
  -> broadcast add
  -> [B,Q,F,H]
  -> ReLU
  -> scale_ctx [B,Q,F,H]

scale_ctx [B,Q,F,H]
  -> scale_weights_head
  -> [B,Q,F,G*Ppf*L]
  -> view
  -> [B,Q,F,G,Ppf,L]
  -> permute
  -> [B,Q,G,F,Ppf,L]
  -> softmax(dim=L)
  -> scale_weights [B,Q,G,F,Ppf,L]
```

#### 4.1.3 ASCII 结构图

```text
                        +------------------------+
query_feat [B,Q,C] ---->| scale_query_proj       |----+
                        +------------------------+    |
                                                      v
                                                query_ctx
                                                [B,Q,1,H]

time_diff [B,F] -- abs -- expand ----------------------+
range_t   [B,Q,F,1] -- log1p --------------------------+--> cat --> scale_desc [B,Q,F,5]
size_log  [B,Q,3] -- expand ---------------------------+

scale_desc [B,Q,F,5]
  -> Linear(5,H)
  -> ReLU
  -> Linear(H,H)
  -> motion_ctx [B,Q,F,H]

query_ctx [B,Q,1,H] + motion_ctx [B,Q,F,H]
  -> broadcast add
  -> ReLU
  -> scale_ctx [B,Q,F,H]
  -> Linear(H, G*Ppf*L)
  -> reshape / permute / softmax(L)
  -> scale_weights [B,Q,G,F,Ppf,L]
```

#### 4.1.4 为什么这里是“直接替代”而不是“原 logits 上加 bias”

当前你的目标已经从“给原始 scale 逻辑做一点修正”变成了：

- 让尺度选择同时由 `query_feat`、时间、距离、尺寸共同决定

这时如果还保留旧的 `scale_weights(query_feat)` 当主干，再额外加一个 bias，模型会天然更依赖旧路径；而现在直接用 `scale_ctx -> scale_weights_head`，就等于把“尺度选择”这件事完整交给条件化后的上下文去做，更符合你的设计目标。

### 4.2 Temporal 分支

目标：

- 根据 query 自身语义、真实时间距离、物体速度，给每个 frame 生成可信度权重。

核心设计：

- 不再只根据 `query_feat` 输出“固定 frame slot 权重”。
- 改成先构造 `temporal_ctx`，再输出每个 group 在每个 frame 上的 logits。

#### 4.2.1 输入量

```text
query_feat : [B, Q, C]
time_diff  : [B, F]
vel        : [B, Q, 2]
```

其中：

- `vel` 是 `query_bbox[..., 8:10]`，单位 m/s
- 分支里实际用的是 `speed = ||vel||_2`

#### 4.2.2 张量流

```text
query_feat [B,Q,C]
  -> temporal_query_proj
  -> [B,Q,H]
  -> unsqueeze(2)
  -> query_ctx [B,Q,1,H]

time_diff [B,F]
  -> abs
  -> [B,1,F,1]
  -> expand
  -> td_abs [B,Q,F,1]

vel [B,Q,2]
  -> norm(dim=-1)
  -> speed [B,Q,1]
  -> unsqueeze(2) + expand
  -> [B,Q,F,1]

cat([td_abs, speed], dim=-1)
  -> temporal_desc [B,Q,F,2]
  -> temporal_motion_encoder
  -> [B,Q,F,H]

query_ctx [B,Q,1,H]
  + motion_ctx [B,Q,F,H]
  -> broadcast add
  -> [B,Q,F,H]
  -> ReLU
  -> temporal_ctx [B,Q,F,H]

temporal_ctx [B,Q,F,H]
  -> temporal_refine
  -> [B,Q,F,G]
  -> permute
  -> [B,Q,G,F]
  -> softmax(dim=F) * F
  -> temporal_weights [B,Q,G,F]
  -> expand to points
  -> [B,Q,G,FP,1]
  -> sampled_feats *= temporal_weights
```

#### 4.2.3 ASCII 结构图

```text
                        +------------------------+
query_feat [B,Q,C] ---->| temporal_query_proj    |----+
                        +------------------------+    |
                                                      v
                                                query_ctx
                                                [B,Q,1,H]

time_diff [B,F] -- abs -- expand ----------------------+
vel [B,Q,2] -- norm -- expand -------------------------+--> cat --> temporal_desc [B,Q,F,2]

temporal_desc [B,Q,F,2]
  -> Linear(2,H)
  -> ReLU
  -> Linear(H,H)
  -> motion_ctx [B,Q,F,H]

query_ctx [B,Q,1,H] + motion_ctx [B,Q,F,H]
  -> broadcast add
  -> ReLU
  -> temporal_ctx [B,Q,F,H]
  -> Linear(H,G)
  -> logits [B,Q,F,G]
  -> permute -> [B,Q,G,F]
  -> softmax(F) * F
  -> temporal_weights [B,Q,G,F]
  -> expand to [B,Q,G,FP,1]
  -> reweight sampled_feats
```

#### 4.2.4 `softmax(F) * F` 的含义

原始等权融合下，每帧等效权重是 `1`。  
如果直接 softmax，所有帧权重和为 `1`，总幅值会缩小为原来的 `1/F`。

因此现在做的是：

```text
softmax(F 维) 后再乘 F
```

这样当 logits 全为 0 时：

```text
softmax([0, ..., 0]) = [1/F, ..., 1/F]
乘 F 后变成 [1, ..., 1]
```

所以零初始化时可以退化回原始“各帧等权”的幅值。

### 4.3 Mixing 分支

目标：

- 在点级别告诉 `AdaptiveMixing`：这个点离当前有多远、这个点本身相对 query 中心偏了多少。
- 让 point mixing 不再把 `F * Ppf` 个点当成纯平铺序列，而是显式知道时间和空间关系。

核心设计：

- 不再输入 `vx*t, vy*t`
- 也不再用 batch 内均值时间差
- 直接对每个样本使用真实 `|dt|`，并拼接每个采样点相对 query center 的 3D 偏移

#### 4.3.1 `point_offset` 是什么

在 `SparseBEVSampling` 里先算：

```text
sampling_points_current = make_sample_points(query_bbox, sampling_offset)
query_center            = decode_bbox(query_bbox)[..., :3]
point_offset            = sampling_points_current - query_center
```

shape:

```text
point_offset : [B,Q,G*Ppf,3]
```

然后扩展到所有 frame，并整理成：

```text
point_offset : [B,Q,G,FP,3]
```

这里虽然每个 frame 都复用了同一个 `point_offset`，但这是合理的，因为 warp 时采样点和 box center 都做了同样的平移 `dist = vel * dt`，所以“点相对中心的局部偏移”在各个 frame 上本来就不变。

#### 4.3.2 张量流

`AdaptiveMixing` 的输入：

```text
x           : [B,Q,G,FP,C_g]
query       : [B,Q,C]
time_diff   : [B,F]
point_offset: [B,Q,G,FP,3]
```

位置编码分支：

```text
time_diff [B,F]
  -> abs
  -> td_abs [B,F]
  -> reshape / expand
  -> td_per_point [B,Q,G,FP,1]

point_offset [B,Q,G,FP,3]

cat([td_per_point, point_offset], dim=-1)
  -> pos_desc [B,Q,G,FP,4]
  -> temporal_pos_encoder
  -> temp_pos [B,Q,G,FP,C_g]

x [B,Q,G,FP,C_g] + temp_pos [B,Q,G,FP,C_g]
  -> x_cond [B,Q,G,FP,C_g]
```

后续 mixing 主体：

```text
query [B,Q,C]
  -> parameter_generator
  -> [B,Q,G*(C_g*C_g + FP*out_points)]
  -> reshape
  -> params [B*Q,G,*]
  -> split
     M: [B*Q,G,C_g,C_g]
     S: [B*Q,G,out_points,FP]

x_cond [B,Q,G,FP,C_g]
  -> reshape -> [B*Q,G,FP,C_g]
  -> matmul with M
  -> [B*Q,G,FP,C_g]
  -> layer_norm over [FP,C_g]
  -> ReLU
  -> matmul with S
  -> [B*Q,G,out_points,C_g]
  -> layer_norm over [out_points,C_g]
  -> ReLU
  -> reshape [B,Q,*]
  -> out_proj
  -> [B,Q,C]
  -> residual add with query
```

#### 4.3.3 ASCII 结构图

```text
x [B,Q,G,FP,C_g] ----------------------------------------------+
                                                                |
time_diff [B,F] -- abs -- expand --> td_per_point [B,Q,G,FP,1] |
point_offset [B,Q,G,FP,3] -------------------------------------+--> cat
                                                                   |
                                                                   v
                                                            pos_desc [B,Q,G,FP,4]
                                                                   |
                                                            temporal_pos_encoder
                                                                   |
                                                            temp_pos [B,Q,G,FP,C_g]
                                                                   |
                                           x + temp_pos -----------+
                                                                   v
                                                            x_cond [B,Q,G,FP,C_g]

query [B,Q,C]
  -> parameter_generator
  -> params
  -> split into:
     M [B*Q,G,C_g,C_g]
     S [B*Q,G,out_points,FP]

x_cond [B*Q,G,FP,C_g]
  -> matmul(M) -> channel mixing
  -> LayerNorm + ReLU
  -> matmul(S) -> point mixing
  -> LayerNorm + ReLU
  -> out_proj
  -> + query
  -> [B,Q,C]
```

## 5. 三个分支怎样和原模型对齐

### 5.1 原始 scale 逻辑

原始版本更接近：

```text
scale_weights = Linear(query_feat)
```

也就是：

- 同一个 query 在所有 frame 上共享一套 level 偏好
- frame 维只是在后面被 expand 出来

当前版本改成：

```text
scale_weights = Head(scale_ctx(query_feat, |dt|, range_t, size))
```

所以现在的尺度选择真正变成了“逐 frame 决策”。

### 5.2 原始 temporal 逻辑

原始版本更接近：

```text
temporal_weights = Linear(query_feat)
```

所以它容易学成：

- 第 1 帧应该怎样
- 第 2 帧应该怎样
- 第 3 帧应该怎样

但不一定对应真实时间间隔。

当前版本改成：

```text
temporal_weights = Head(temporal_ctx(query_feat, |dt|, speed))
```

所以同一个 frame slot 在不同样本里只要真实 `|dt|` 不一样，权重也可以不一样。

### 5.3 原始 mixing 逻辑

你当前这一轮之前的版本更接近：

```text
pos_desc = [t, vx*t, vy*t]
```

而且时间差还取了 batch 均值。

现在改成：

```text
pos_desc = [|dt|, dx, dy, dz]
```

这样变化是：

- 时间用每个样本自己的真实值
- 运动方向不再由可能不准的 velocity correction 来承担
- 点的局部几何关系由采样点偏移直接表达

## 6. 为什么 `temporal_ctx` 和 `scale_ctx` 用“加法融合”

两者现在的骨架都是：

```text
ctx = ReLU(query_proj(query_feat) + desc_encoder(desc))
```

这里的“加法”不是把原始输出硬加到最终 logits 上，而是：

- 先把 `query_feat` 投到一个 `H=16` 的隐藏空间
- 再把条件描述符 `desc` 编到同一个隐藏空间
- 让两者在隐藏空间相加
- 最后再由输出头把 `ctx` 读成 logits

也就是说，真正的 logits 仍然是后面那层线性头产生的：

```text
logit = W_o * ctx + b
```

这类设计的优点是：

- 计算量小
- 初始化稳定
- 条件变量和 query 语义能在同一个低维空间里做交互

对于你现在这两个任务：

- 帧可信度分配
- FPN 层选择

这种轻量条件化通常已经够用。

## 7. 初始化与训练起点

当前初始化里最关键的是 3 件事：

### 7.1 `temporal_refine` 零初始化

```text
temporal_refine.weight = 0
temporal_refine.bias   = 0
```

因此初始时：

```text
query_temporal_w = 0
softmax = uniform
softmax * F = 1
```

即：

- 初始时各 frame 等权
- 不会在训练一开始把原模型的时序幅值打乱

### 7.2 `scale_weights_head` 零初始化

```text
scale_weights_head.weight = 0
scale_weights_head.bias   = 0
```

因此初始时：

```text
scale logits = 0
softmax(level) = uniform
```

即：

- 初始时所有 FPN level 等权
- 与原始模型“没有明确偏置”的中性起点一致

### 7.3 `temporal_pos_encoder` 最后一层零初始化

```text
temporal_pos_encoder[-1].weight = 0
temporal_pos_encoder[-1].bias   = 0
```

因此初始时：

```text
temp_pos = 0
x + temp_pos = x
```

即：

- mixing 分支新增的位置编码一开始不影响原特征
- 模型可以从原行为平滑过渡到新行为

## 8. 目前我认为“合理”的地方

### 8.1 三个分支各自负责不同问题

- `scale`：负责“去哪一层采样”
- `temporal`：负责“哪一帧更可信”
- `mixing`：负责“点和点之间的时空几何关系”

这个拆分比把所有条件全塞进一个大头里更清楚，也更轻量。

### 8.2 物理量和学习量的边界更清楚了

当前流程里：

- 物理平移补偿：`dist = vel * dt`
- 学习时序权重：`temporal_ctx`
- 学习尺度选择：`scale_ctx`
- 学习点间位置编码：`pos_desc`

这样做的好处是：

- 不把“几何传播”完全交给网络猜
- 也不把“可信度分配”和“尺度偏好”硬编码成固定规则

### 8.3 对随机时间间隔训练更友好

因为现在：

- temporal 分支看 `|dt|`
- scale 分支看 `|dt|`
- mixing 分支也看 `|dt|`

所以三段时序建模的归因是一致的，不会出现前面按 frame slot、后面按真实时间的冲突。

## 9. 目前我认为值得持续观察的点

这些不算已经确定的错误，但建议后面训练时重点看：

### 9.1 `speed` 是否需要轻度压缩

现在 `speed = ||v||_2` 直接喂入 `temporal_motion_encoder`。  
如果数据里速度分布跨度比较大，后面可以尝试：

```text
speed_feat = log1p(speed)
```

或者做简单 clipping。

### 9.2 `|dt|` 去掉了方向信息

这是你当前明确想要的行为，也和“只看离当前多远”一致。  
但如果将来你做未来帧预测，或者发现在历史帧和未来帧上行为应当不同，这里会是第一个需要恢复 signed `dt` 的地方。

### 9.3 `range_t` 用的是 BEV 距离

当前定义：

```text
range_t = sqrt(x_t^2 + y_t^2)
```

对尺度选择我认为是合理的，但如果你后面希望更贴近透视投影，也可以把：

- 相机深度
- 多相机可见性
- z 高度

加入 scale 分支。不过那会让结构更重。

## 10. 一句话总结当前结构

```text
scale    = query 语义 + 时间距离 + 传播后距离 + 物体尺寸
temporal = query 语义 + 时间距离 + 物体速度
mixing   = 时间距离 + 采样点局部偏移
```

这三者当前的职责划分、张量流和初始化策略，整体上是合理的。
