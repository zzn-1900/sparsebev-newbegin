# ProtoQuery SparseBEV 设计方案

## 1. 目标

本文档给出一个适配当前 `SparseBEV` 的改进方案，核心约束只有一条:

- 不做任何视角采样相关改动。

这里的“不做”包括:

- 不修改 `SparseBEV/models/sparsebev_sampling.py`
- 不修改 `sampling_4d` / `msmv_sampling`
- 不新增视角选择、视角打分、多视角融合权重
- 不新增 frame reweighting、valid mask 统计、view-level diagnostics

因此，这份方案只保留三条线:

1. `query-space prototype bank`
2. `prototype-guided query refinement`
3. `prototype alignment loss`

## 2. 设计动机

`CorrBEV` 的真正价值，不是它的 2D correlation 形式本身，而是下面这个思想:

- 当目标被遮挡、模糊或证据不足时，当前 query 的表征会变弱。
- 这时如果能给 query 一个类别相关的先验，它更容易恢复到正确的语义方向。

对当前这份 `SparseBEV`，我认为最自然的做法不是去动视角采样，而是:

- 保持现有的采样链路完全不变
- 仅在 decoder 的 query feature 空间里做“先验补全”

这样做的好处是:

1. 工程侵入小
2. 不影响当前 CUDA 采样实现
3. 不会把问题引到视角选择和采样策略上
4. 更容易单独验证“prototype 先验”是否有效

## 3. 边界约束

本方案明确只允许改这些位置:

- `SparseBEV/models/sparsebev_head.py`
- `SparseBEV/models/sparsebev_transformer.py`
- 新增一个 prototype 相关模块文件

本方案明确不改这些位置:

- `SparseBEV/models/sparsebev_sampling.py`
- `SparseBEV/models/csrc/*`
- 数据集与数据管线
- 相机视角选择逻辑

换句话说，`SparseBEV` 原有的:

- query bbox 初始化
- self attention
- 中间特征聚合
- adaptive mixing
- bbox refine

主干流程保持不变。我们的工作只发生在 decoder layer 输出之后、下一层输入之前，以及 head 的 loss 侧。

## 4. 当前 SparseBEV 可利用的插点

### 4.1 Query 初始化是静态的

在 `SparseBEVHead` 中，query bbox 是规则网格初始化，初始 query feature 基本是统一的背景 token 变体。

这意味着:

- query 初值缺少显式类别先验
- 困难 query 和普通 query 在起点上没有区分

这正适合后续加入 query-space prototype 补全。

### 4.2 Decoder 层间是最合适的补全位置

当前 `SparseBEVTransformerDecoder` 的每一层都会输出:

- `query_feat`
- `cls_score`
- `bbox_pred`

这给了我们一个非常自然的插点:

- 用当前层 `cls_score` 估计 query 属于哪些类别
- 从 prototype bank 里检索对应的类原型
- 用一个门控模块决定是否对该 query 做补全
- refined query feature 再送入下一层

这个插点不需要碰现有主干实现。

### 4.3 当前 loss 缺少显式的类原型约束

`SparseBEVHead.loss` 目前只有:

- 分类损失
- bbox 损失
- DN loss

还没有一个约束去显式要求:

- 同类正样本 query 靠近该类 prototype
- 不同类 prototype 彼此可分

因此可以在 head 端增加一个很轻量的 prototype alignment loss。

## 5. 总体方案

方案只包含三个模块:

1. `Query Prototype Bank`
2. `Query Difficulty Estimator`
3. `Prototype-guided Query Refinement`

配套一个训练约束:

4. `Prototype Alignment Loss`

一句话概括:

- 用正样本 query feature 构建类别原型库
- 用 decoder 自己的预测判断哪些 query 更困难
- 只对困难 query 注入 prototype 先验
- 再用 alignment loss 稳定 prototype 空间

## 6. 模块设计

### 6.1 模块 A: Query Prototype Bank

位置:

- 新增 `SparseBEV/models/proto_query.py`
- 由 `SparseBEVHead` 持有

目标:

- 在 query feature 空间中维护每个类别的稳定原型

核心思想:

- 不使用 2D crop
- 不使用外部图像编码器
- 不使用语言原型
- 直接使用“和 GT 成功匹配的正样本 query feature”作为 prototype 来源

推荐结构:

- `prototype_bank`: `[L, num_classes, C]`
- `prototype_count`: `[L, num_classes]`

其中:

- `L` 是 decoder 层数
- `C` 是 query feature 维度

为什么按层维护:

- 不同 decoder 层的语义成熟度不同
- 早层更粗，后层更稳定
- 混成一个 bank 容易相互污染

更新方式:

```text
proto[l, c] = m * proto[l, c] + (1 - m) * mean(norm(pos_query_feat[l, c]))
```

推荐超参:

- `m = 0.99`

实现原则:

- 更新时使用 `detach()`
- bank 作为 buffer 保存
- DDP 下先 gather 再更新

冷启动策略:

- 当前实现采用 `prototype_count[l, c] < min_count` 的计数门控，未单独引入 epoch 级 warmup hook
- 当 `prototype_count[l, c] < min_count` 时跳过该类 prototype

### 6.2 模块 B: Query Difficulty Estimator

位置:

- `SparseBEV/models/proto_query.py`
- 在 `SparseBEVTransformerDecoder` 或 decoder layer 间调用

目标:

- 在不依赖额外几何统计的前提下，判断一个 query 是否属于“难样本”

可用信号全部来自 decoder 自己的输出:

1. `cls_entropy`
2. `cls_margin`
3. `bbox_delta_norm`
4. `query_drift`

定义建议:

- `cls_entropy`: 当前层分类分布的熵
- `cls_margin`: top1 和 top2 logit 的差
- `bbox_delta_norm`: 当前层 bbox 相对上一层 bbox 的更新幅度
- `query_drift`: 当前层 query feature 与上一层 query feature 的差异

直觉:

- 熵高、margin 低，说明分类不确定
- bbox 更新幅度大，说明定位还没稳定
- query drift 大，说明该 query 还处于震荡状态

difficulty score 公式:

```text
diff_feat = [
    cls_entropy,
    1 - cls_margin,
    bbox_delta_norm,
    query_drift
]

diff_score = sigmoid(MLP(diff_feat))
```

输出:

- `diff_score`: `[B, Q, 1]`

说明:

- `diff_score` 不是监督标签，只是 refinement 的 gating 信号
- 第一版不建议单独对它加 loss

### 6.3 模块 C: Prototype-guided Query Refinement

位置:

- `SparseBEVTransformerDecoder.forward`
- 或新增 `PrototypeRefiner`

目标:

- 不改变现有主干，仅在层间对 query feature 做补全

推荐插入位置:

- 第 `l` 层 decoder 输出之后
- 第 `l+1` 层输入之前

推荐流程:

1. 得到当前层 `query_feat_l`
2. 得到当前层 `cls_score_l`
3. 从 prototype bank 中检索类别原型
4. 计算 `diff_score_l`
5. 用 `diff_score_l` 控制 prototype 注入强度

prototype 检索方式:

```text
cls_prob = sigmoid(cls_score_l)
cls_prob = cls_prob / cls_prob.sum(-1, keepdim=True)
proto_mix = cls_prob @ prototype_bank[l]
```

第一版推荐 soft mixture，不用硬 top-1，原因是:

- 更稳定
- 对误分类更不敏感

refinement 公式:

```text
refine_in = concat(query_feat_l, proto_mix, diff_score_l)
delta = MLP(refine_in)
gate = sigmoid(MLP_gate(refine_in))
query_feat_l_refined = query_feat_l + gate * delta
```

解释:

- `delta` 表示 prototype 给 query 的补偿项
- `gate` 控制补偿强度
- 难 query 的 `gate` 应更大，易 query 的 `gate` 应更小

第一版建议:

- 只在最后两层之间开启 refinement
- 不修改第一层前的 query 初始化

### 6.4 模块 D: Prototype Alignment Loss

位置:

- `SparseBEVHead.loss`

目标:

- 让正样本 query 显式靠近其类别 prototype

输入:

- `all_query_feats`: `[L, B, Q, C]`
- 每层正样本 query 的 matched label
- `prototype_bank`

损失形式建议:

```text
sim = cosine(norm(q_pos), norm(proto_bank[l])) / tau
L_proto = cross_entropy(sim, gt_label)
```

说明:

- 只对正样本 query 计算
- 第一版只在最后一层计算
- DN query 不参与

推荐总损失:

```text
L = L_cls + L_bbox + L_dn + lambda_proto * L_proto
```

推荐初值:

- `lambda_proto = 0.1`
- `tau = 0.07`

## 7. 张量流与伪代码

### 7.1 Decoder 主流程

```text
query_bbox, query_feat = init_queries()
prev_query_feat = None
prev_bbox_pred = None

for l in range(num_layers):
    query_feat = run_original_decoder_block(
        query_bbox, query_feat, mlvl_feats, img_metas
    )

    cls_score = cls_branch(query_feat)
    bbox_pred = reg_branch(query_feat)

    save intermediate:
        all_query_feats[l] = query_feat

    if l < num_layers - 1 and refine_enabled:
        diff_score = estimate_difficulty(
            query_feat,
            prev_query_feat,
            bbox_pred,
            prev_bbox_pred,
            cls_score,
        )
        proto_mix = retrieve_prototype(cls_score, prototype_bank[l])
        query_feat = prototype_refine(query_feat, proto_mix, diff_score)

    prev_query_feat = query_feat.detach()
    prev_bbox_pred = bbox_pred.detach()
    query_bbox = bbox_pred.detach()
```

### 7.2 Loss 与 bank 更新流程

```text
outs = {
    all_cls_scores,
    all_bbox_preds,
    all_query_feats,
    dn_mask_dict,
}

loss_dict = cls_bbox_dn_loss(...)

loss_dict += prototype_alignment_loss(
    query_feats[last_layer][matched_pos],
    gt_labels[matched_pos],
    prototype_bank[last_layer]
)

update_prototype_bank_with_ema(
    query_feats[last_layer][matched_pos].detach(),
    gt_labels[matched_pos]
)
```

## 8. 代码改动建议

### 8.1 新增文件

建议新增:

- `SparseBEV/models/proto_query.py`

可包含:

- `QueryPrototypeBank`
- `QueryDifficultyEstimator`
- `PrototypeRefiner`

### 8.2 修改文件

#### `SparseBEV/models/sparsebev_transformer.py`

改动建议:

1. decoder 主循环收集 `all_query_feats`
2. 在 layer 间加入 difficulty estimation
3. 在 layer 间加入 prototype refinement

注意:

- 不修改 `SparseBEVSampling`
- 不修改原有 decoder block 的中间接口
- 不修改任何视角与融合逻辑

#### `SparseBEV/models/sparsebev_head.py`

改动建议:

1. `forward` 将 `all_query_feats` 放入 `outs`
2. `loss` 中新增 `prototype_alignment_loss`
3. 在训练阶段更新 prototype bank

建议:

- bank 更新函数使用 `@torch.no_grad()`
- bank 更新与 loss 计算分开

#### `SparseBEV/models/__init__.py`

注册新增模块导出

### 8.3 明确不改的文件

以下文件本方案不改:

- `SparseBEV/models/sparsebev_sampling.py`
- `SparseBEV/models/csrc/wrapper.py`
- `SparseBEV/models/csrc/msmv_sampling/*`

### 8.4 配置项建议

在 config 中新增:

```python
proto_query=dict(
    enabled=True,
    bank_momentum=0.99,
    min_proto_count=32,
    use_layers=[4, 5],
    lambda_proto=0.1,
    temperature=0.07,
    difficulty_hidden_dim=64,
    proto_hidden_dim=256,
    prototype_refine=True,
)
```

第一版推荐:

- 只在最后两层启用 refinement
- 只在最后一层做 bank update
- 只在最后一层做 prototype loss

## 9. 训练策略

### 9.1 当前实现的启用策略

- 当前代码路径没有额外接 runner hook，因此没有显式的 epoch 级 warmup 开关
- prototype bank 的冷启动保护由 `min_proto_count` 完成
- 当某类 prototype 尚未积累到足够样本时，该类不会参与 refinement，也不会参与 prototype loss

### 9.2 DDP 同步

如果使用多卡:

- 每轮 bank 更新前汇聚各卡上的正样本 query feature 与 label
- 统一执行 EMA 更新

### 9.3 精度建议

- bank update 用 `fp32`
- prototype 与 query 都先 `F.normalize`
- 未初始化类别默认跳过 refinement 和 prototype loss

## 10. 风险点

### 10.1 Prototype bank 早期不稳定

问题:

- 训练初期 query feature 很噪，容易污染 bank

应对:

- warmup
- EMA
- `min_proto_count`
- 只用最后一层更新

### 10.2 错类 prototype 误导 query

问题:

- 如果 `cls_score` 错误，prototype retrieval 可能把 query 拉向错误类别

应对:

- soft mixture 替代 top-1
- 用 `diff_score` 控制 gate
- 只在后两层启用 refinement

### 10.3 稀有类原型质量差

问题:

- 稀有类样本数少，prototype 质量可能偏低

应对:

- 增大 `min_proto_count`
- 只对成熟类别启用 refine
- 或者先只在 prototype loss 中使用成熟类别

### 10.4 额外显存开销

问题:

- 保存 `all_query_feats` 会增加少量显存

应对:

- 第一版只保留必要层
- prototype loss 默认只算最后一层

## 11. 推荐实验顺序

### 实验 A: Prototype Bank Only

目的:

- 验证 query-space prototype 是否稳定可用

改动:

- bank 收集
- EMA 更新
- 不开 refinement
- 不开 prototype loss

### 实验 B: A + Prototype Alignment Loss

目的:

- 验证显式类原型约束是否有效

### 实验 C: B + Prototype Refinement

目的:

- 验证 prototype 先验补全是否带来额外收益

推荐重点观察:

- 整体 `mAP/NDS`
- `Vis1/Vis2 recall`
- 易混类别的分类质量

## 12. MVP 落地版本

如果只做一个最小可跑版本，建议范围如下:

1. `SparseBEVHead` 增加单层 `prototype_bank`
2. `SparseBEVTransformerDecoder` 输出 `all_query_feats`
3. `SparseBEVHead.loss` 新增最后一层 `prototype_alignment_loss`
4. 训练阶段更新最后一层 prototype bank
5. 等 bank 稳定后，再在最后两层之间加入 refinement

这个版本的优点:

- 完全不碰视角采样逻辑
- 改动面小
- 容易定位收益来源

## 13. 结论

这份设计文档刻意删除了所有视角采样相关工作，只保留 query 和 prototype 这条线。

最终方案可以概括为:

- 不改采样
- 不改视角
- 不改数据流
- 只在 query feature 空间中引入类别先验

对应三个核心模块:

1. `Query Prototype Bank`
2. `Query Difficulty Estimator`
3. `Prototype-guided Query Refinement`

再配合一个:

4. `Prototype Alignment Loss`

这是一条更符合你当前要求的实现路径，也更适合做清晰、可控的增量实验。
