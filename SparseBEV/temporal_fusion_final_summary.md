# SparseBEV 当前时序融合结构总结

本文档以当前代码实现为准。

当前版本保留两条条件化分支：

1. `scale`：`query_feat + [|dt|, range_t, size]`
2. `temporal`：`query_feat + [|dt|, speed]`

`mixing` 已回退到原始 query-driven adaptive mixing，不再额外注入点级时空位置编码。

## 1. 审查结论

当前实现的整体判断是：

- `scale` 和 `temporal` 都已经基于真实 `|dt|` 建模，和随机时间间隔训练是一致的。
- `mixing` 去掉点级位置编码后，结构更保守，也避免了“位置编码只作用在 sampled feature 端、不作用在参数生成端”的弱作用路径。
- 这版相比上一版更像“只改时序权重与尺度选择，不碰 mixing 主体”的稳妥方案。

需要明确的边界：

- 当前仍然只看 `|dt|`，不区分过去和未来。
- `range_t` 仍然是 BEV 平面距离，不是 3D 欧氏距离。
- `speed` 仍然直接使用原始 m/s。

已完成检查：

- `python3 -m py_compile SparseBEV/models/sparsebev_sampling.py SparseBEV/models/sparsebev_transformer.py`

## 2. 符号约定

| 符号 | 含义 |
|---|---|
| `B` | batch size |
| `Q` | query 数量 |
| `F` | 时间帧数 |
| `G` | group 数 |
| `Ppf` | 每帧每 group 采样点数 |
| `FP` | 总点数，`FP = F * Ppf` |
| `L` | FPN level 数 |
| `C` | query 通道数 |
| `H` | 条件隐藏维度，当前为 `32` |
| `C_g` | group 内通道数，`C / G` |

默认配置里常见取值：

```text
C   = 256
F   = 8
G   = 4
Ppf = 4
FP  = 32
L   = 4
H   = 32
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
                    |
                    v
        +-------------------------------+
        |         AdaptiveMixing        |
        |                               |
        |  query -> parameter_generator |
        |  channel mixing               |
        |  point mixing                 |
        |  out_proj + residual          |
        +-------------------------------+
                    |
              norm + ffn
                    |
               cls / reg
```

## 4. Scale 分支

目标：

- 在每个 frame 上，根据真实时间距离、传播后的距离和物体尺寸，生成该 frame 的多尺度采样权重。

输入：

```text
query_feat : [B, Q, C]
time_diff  : [B, F]
range_t    : [B, Q, F, 1]
size_log   : [B, Q, 3]
```

张量流：

```text
query_feat [B,Q,C]
  -> scale_query_proj
  -> [B,Q,H]
  -> unsqueeze
  -> query_ctx [B,Q,1,H]

time_diff [B,F]
  -> abs + expand
  -> td_abs [B,Q,F,1]

range_t [B,Q,F,1]
  -> log1p
  -> range_feat [B,Q,F,1]

size_log [B,Q,3]
  -> expand
  -> size_feat [B,Q,F,3]

cat([td_abs, range_feat, size_feat], dim=-1)
  -> scale_desc [B,Q,F,5]
  -> Linear(5,H)
  -> ReLU
  -> Dropout(0.1)
  -> Linear(H,H)
  -> motion_ctx [B,Q,F,H]

query_ctx + motion_ctx
  -> ReLU
  -> scale_ctx [B,Q,F,H]

scale_ctx
  -> scale_weights_head
  -> [B,Q,F,G*Ppf*L]
  -> reshape + permute
  -> [B,Q,G,F,Ppf,L]
  -> softmax(dim=L)
  -> scale_weights [B,Q,G,F,Ppf,L]
```

ASCII 结构图：

```text
query_feat [B,Q,C] ----> scale_query_proj ----+
                                              |
time_diff/range_t/size_log --> scale_desc ----+--> scale_motion_encoder
                                              |
                                              v
                                   query_ctx + motion_ctx
                                              |
                                            ReLU
                                              |
                                          scale_ctx
                                              |
                                   scale_weights_head
                                              |
                              reshape / permute / softmax(L)
                                              |
                               scale_weights [B,Q,G,F,Ppf,L]
```

## 5. Temporal 分支

目标：

- 给每个 query、每个 group 的每个 frame 生成真实时间感知的权重。

输入：

```text
query_feat : [B, Q, C]
time_diff  : [B, F]
vel        : [B, Q, 2]
```

其中：

- 分支里实际使用的是 `speed = ||vel||_2`

张量流：

```text
query_feat [B,Q,C]
  -> temporal_query_proj
  -> [B,Q,H]
  -> unsqueeze
  -> query_ctx [B,Q,1,H]

time_diff [B,F]
  -> abs + expand
  -> td_abs [B,Q,F,1]

vel [B,Q,2]
  -> norm
  -> speed [B,Q,1]
  -> expand
  -> [B,Q,F,1]

cat([td_abs, speed], dim=-1)
  -> temporal_desc [B,Q,F,2]
  -> Linear(2,H)
  -> ReLU
  -> Dropout(0.1)
  -> Linear(H,H)
  -> motion_ctx [B,Q,F,H]

query_ctx + motion_ctx
  -> ReLU
  -> temporal_ctx [B,Q,F,H]

temporal_ctx
  -> temporal_refine
  -> [B,Q,F,G]
  -> permute
  -> [B,Q,G,F]
  -> softmax(dim=F) * F
  -> temporal_weights [B,Q,G,F]
  -> expand to [B,Q,G,FP,1]
  -> sampled_feats *= temporal_weights
```

ASCII 结构图：

```text
query_feat [B,Q,C] ----> temporal_query_proj ----+
                                                 |
time_diff/speed ---------> temporal_desc --------+--> temporal_motion_encoder
                                                 |
                                                 v
                                      query_ctx + motion_ctx
                                                 |
                                               ReLU
                                                 |
                                            temporal_ctx
                                                 |
                                           temporal_refine
                                                 |
                                    permute / softmax(F) * F
                                                 |
                                       temporal_weights [B,Q,G,F]
```

## 6. Mixing 分支

当前 `mixing` 已回退到原始 query-driven 形式，不再输入 `time_diff` 或 `point_offset`。

输入：

```text
x     : [B,Q,G,FP,C_g]
query : [B,Q,C]
```

张量流：

```text
query [B,Q,C]
  -> parameter_generator
  -> params
  -> split into:
     M [B*Q,G,C_g,C_g]
     S [B*Q,G,out_points,FP]

x [B,Q,G,FP,C_g]
  -> reshape [B*Q,G,FP,C_g]
  -> matmul(M)
  -> LayerNorm + ReLU
  -> matmul(S)
  -> LayerNorm + ReLU
  -> out_proj
  -> + query
  -> [B,Q,C]
```

ASCII 结构图：

```text
query [B,Q,C]
  -> parameter_generator
  -> params -> split(M, S)

x [B*Q,G,FP,C_g]
  -> matmul(M) -> channel mixing
  -> LayerNorm + ReLU
  -> matmul(S) -> point mixing
  -> LayerNorm + ReLU
  -> out_proj
  -> + query
  -> [B,Q,C]
```

## 7. 初始化与正则

当前最关键的初始化与正则有三点：

### 7.1 `temporal_refine` 零初始化

```text
temporal_refine.weight = 0
temporal_refine.bias   = 0
```

因此初始时各 frame 等权，幅值与原始实现一致。

### 7.2 `scale_weights_head` 零初始化

```text
scale_weights_head.weight = 0
scale_weights_head.bias   = 0
```

因此初始时各 level 等权。

### 7.3 条件编码器中的轻量 dropout

当前 `scale_motion_encoder` 和 `temporal_motion_encoder` 采用：

```text
Linear -> ReLU -> Dropout(0.1) -> Linear
```

这样做的考虑是：

- hidden 维从 `16` 增到 `32`
- 用轻量 dropout 稍微压一下新增分支的过拟合风险
- 不直接对最终 logits 或时序权重做 dropout，避免权重分配过抖

## 8. 一句话总结

```text
scale    = query 语义 + 时间距离 + 传播后距离 + 物体尺寸
temporal = query 语义 + 时间距离 + 物体速度
mixing   = 原始 query-driven adaptive mixing
```

这版结构比上一版更保守，重点只放在“尺度选择”和“时序权重”两件事上。
