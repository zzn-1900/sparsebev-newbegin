# 当前工作区相对 `5910ed2` 的完整差异说明

## 1. 文档目的

本文档的目标是把当前工作区中与 prototype 相关的真实代码状态完整梳理清楚，作为后续继续修改时的基线说明。

本文档同时回答三件事：

1. 初始提交 `5910ed2f47e77eccf765af6e39106f33c7f190f8` 中模型是什么样子。
2. 当前工作区相对初始提交具体改了什么。
3. 当前工作区真实代码，与仓库中已有文档 `proto_temporal_sparsebev_design.md` 和 `changes_vs_initial_branch.md` 有哪些不一致。

本文档的比较对象是：

- 基线：`5910ed2f47e77eccf765af6e39106f33c7f190f8`
- 当前状态：当前工作区，包括尚未提交的本地修改

注意：

- 当前工作区不是单次改动，而是在已提交版本 `eee8f981d9c26fcc2e4fc8f903174b8ca46b34d9` 的基础上继续演化。
- 因此，旧文档中描述的方案并不等于当前代码真实实现。

## 2. 总体结论

相对初始提交，当前工作区的核心变化集中在 `query / prototype / loss` 这条线上，采样链路、CUDA 算子、数据管线没有改。

当前工作区已经从“原始 SparseBEV”变成“带 query-space prototype 机制的 SparseBEV”，并且这套 prototype 机制已经经历了两次方案变化：

1. 第一阶段是“layer-wise prototype + difficulty-aware refinement”的方案。
2. 当前工作区已经进一步演化成“class-level multi-slot prototype bank + online update + local maintenance + prototype cross-attention”的方案。

这意味着：

- 当前代码不是最开始设计文档里的 `QueryDifficultyEstimator + PrototypeRefiner` 版本。
- 当前代码也不是最开始变更文档里写的“按 decoder layer 维护单 prototype”版本。
- 当前代码的真实结构，应以 `SparseBEV/models/proto_query.py`、`SparseBEV/models/sparsebev_transformer.py`、`SparseBEV/models/sparsebev_head.py` 为准。

## 3. 与初始提交相比的文件变化

### 3.1 新增文件

与 `5910ed2` 相比，当前工作区新增了下列相关文件：

| 文件 | 作用 |
| --- | --- |
| `SparseBEV/models/proto_query.py` | 新增 prototype bank、prototype retrieval、prototype cross-attention 的实现 |
| `SparseBEV/configs/r50_nuimg_704x256-quicktest.py` | 便于快速实验的 mini 数据集配置 |
| `SparseBEV/configs/r50_nuimg_704x256-quick.py` | 对 quicktest 配置的简单继承入口 |
| `SparseBEV/docs/proto_temporal_sparsebev_design.md` | 早期设计文档，描述的是旧方案 |
| `SparseBEV/docs/changes_vs_initial_branch.md` | 早期变更说明，描述的是旧阶段实现 |

另有一个 PDF 文件：

- `Xue_CorrBEV_Multi-View_3D_Object_Detection_by_Correlation_Learning_with_Multi-modal_CVPR_2025_paper.pdf`

它不是模型代码的一部分。

### 3.2 修改文件

与 `5910ed2` 相比，当前工作区修改了下列核心文件：

| 文件 | 改动类型 |
| --- | --- |
| `SparseBEV/models/__init__.py` | 导出新增模块 |
| `SparseBEV/models/sparsebev_transformer.py` | 接入 prototype refinement |
| `SparseBEV/models/sparsebev_head.py` | 接入 bank、prototype loss、bank update |
| `SparseBEV/configs/r50_nuimg_704x256.py` | 接入 `proto_query_cfg` 并调整日志与评估频率 |

## 4. 初始提交 `5910ed2` 的模型状态

初始提交中的 `SparseBEV` 可以概括为“纯原始 SparseBEV”，没有任何 prototype 机制。

### 4.1 `SparseBEV/models/sparsebev_transformer.py`

初始状态下：

- `SparseBEVTransformer.__init__` 没有 `proto_query` 参数。
- `SparseBEVTransformer.forward` 只接收：
  - `query_bbox`
  - `query_feat`
  - `mlvl_feats`
  - `attn_mask`
  - `img_metas`
- `SparseBEVTransformer.forward` 只返回：
  - `cls_scores`
  - `bbox_preds`
- `SparseBEVTransformerDecoder` 只做原有 decoder 层堆叠，没有 prototype refinement。
- decoder 每一层执行顺序是：
  - 位置编码
  - self attention
  - image feature sampling
  - adaptive mixing
  - FFN
  - 分类分支
  - 回归分支

### 4.2 `SparseBEV/models/sparsebev_head.py`

初始状态下：

- `SparseBEVHead.__init__` 没有 `proto_query` 参数。
- head 不持有任何 prototype bank。
- `forward` 只调用 transformer，拿到 `cls_scores` 和 `bbox_preds`。
- `outs` 中没有 `final_query_feats` 或 `all_query_feats`。
- `get_targets` 只返回：
  - `labels_list`
  - `label_weights_list`
  - `bbox_targets_list`
  - `bbox_weights_list`
  - `num_total_pos`
  - `num_total_neg`
- `loss` 只包含：
  - 分类损失
  - bbox L1 损失
  - DN loss
- 没有正样本 query 质量建模。
- 没有 `loss_proto`。
- 没有训练时 bank 更新。

### 4.3 `SparseBEV/configs/r50_nuimg_704x256.py`

初始状态下：

- 没有 `proto_query_cfg`。
- `SparseBEVHead` 和 `SparseBEVTransformer` 都没有 prototype 相关配置入口。
- `log_config` 中：
  - `MyTextLoggerHook.interval = 1`
  - `MyTensorboardLoggerHook.interval = 500`
- `eval_config.interval = total_epochs`

## 5. 当前工作区的真实模型结构

当前工作区的真实实现不是旧文档描述的难度门控版本，而是下面这条链路：

1. 配置文件中定义 `proto_query_cfg`
2. `SparseBEVHead` 根据配置构造 `QueryPrototypeBank`
3. `SparseBEVHead.forward` 将 `prototype_bank`、`prototype_count` 传给 `SparseBEVTransformer`
4. `SparseBEVTransformerDecoder` 在指定层之间，对正常 query 执行 `PrototypeCrossAttention`
5. `SparseBEVHead.loss` 从最后一层 decoder query 中抽取正样本
6. `SparseBEVHead` 根据 GT 匹配结果计算 query quality
7. `SparseBEVHead` 计算 `loss_proto`
8. `SparseBEVHead` 用正样本 query 在线更新 bank
9. bank 内部按类进行空 slot 初始化、在线 slot 更新、recent buffer 缓存、局部重聚类维护

## 6. 逐文件详细差异

## 6.1 `SparseBEV/models/__init__.py`

### 初始提交

初始提交中只导出：

- `SparseBEV`
- `SparseBEVHead`
- `SparseBEVTransformer`

### 当前工作区

当前工作区新增导出：

- `QueryPrototypeBank`
- `PrototypeCrossAttention`

这说明当前仓库公开暴露的 prototype 相关实现只有这两个模块，没有导出 `QueryDifficultyEstimator` 或 `PrototypeRefiner`，因为当前代码中已经没有这两个类。

## 6.2 `SparseBEV/models/proto_query.py`

这是当前工作区新增的核心文件，也是当前真实实现与旧文档差异最大的地方。

### 6.2.1 文件中包含的对象

当前文件中实际存在：

- `normalize_query_logits`
- `_all_gather_tensor`
- `_select_weighted_diverse_indices`
- `_select_weighted_medoid`
- `QueryPrototypeBank`
- `PrototypeCrossAttention`

当前文件中实际不存在：

- `QueryDifficultyEstimator`
- `PrototypeRefiner`
- `mix_query_prototypes`

### 6.2.2 `normalize_query_logits`

功能：

- 对分类分支输出 `cls_score` 先做 `sigmoid`
- 再沿类别维度归一化

这意味着当前 retrieval 使用的是“归一化后的 sigmoid 分类概率”，不是 softmax 分类概率。

### 6.2.3 `_all_gather_tensor`

功能：

- 在 DDP 环境下，把不同进程的变长 tensor 先 pad 再 all-gather
- 最后按真实长度裁回并拼接

用途：

- bank 更新时收集所有进程上的正样本 query
- 保证 prototype bank 在多卡训练下的更新来源一致

### 6.2.4 `_select_weighted_diverse_indices`

功能：

- 从一组特征里选择若干个“高权重且相互多样”的种子

实现逻辑：

- 先按 `weights` 归一化成 `weight_score`
- 第一个种子取 `weight_score` 最大值
- 后续种子按：
  - 与已选种子最大相似度越低越好
  - 同时保留少量质量权重偏置

这个函数的目的不是求均值中心，而是为局部重聚类选 seed。

### 6.2.5 `_select_weighted_medoid`

功能：

- 在一个簇内部，从真实样本中挑一个最具有代表性的样本作为 medoid

实现逻辑：

- 计算簇内所有 token 两两 cosine similarity
- 再用 `weights` 加权求和
- 得分最高者作为代表原型

注意：

- 当前原型在局部重聚类后，是 medoid 式代表点，不是均值中心
- 这是当前实现与旧文档最关键的差异之一

### 6.2.6 `QueryPrototypeBank` 的状态结构

当前 `QueryPrototypeBank` 是按“类别 x slot”组织，而不是“层 x 类别”组织。

当前 buffer 包括：

| 名称 | 形状 | 作用 |
| --- | --- | --- |
| `prototype_bank` | `[num_classes, num_prototypes, C]` | 每类多个 prototype slot |
| `prototype_count` | `[num_classes, num_prototypes]` | 每个 slot 的支持度 |
| `prototype_quality` | `[num_classes, num_prototypes]` | 每个 slot 的质量 EMA |
| `prototype_radius` | `[num_classes, num_prototypes]` | 每个 slot 的离散度 |
| `prototype_age` | `[num_classes, num_prototypes]` | 每个 slot 距离上次更新的时间 |
| `prototype_updates` | `[1]` | 全局更新次数 |
| `recent_feats` | `[num_classes, recent_buffer_size, C]` | 最近保留的候选 query |
| `recent_quality` | `[num_classes, recent_buffer_size]` | recent query 的质量权重 |
| `recent_score` | `[num_classes, recent_buffer_size]` | recent query 的综合分数 |
| `recent_valid` | `[num_classes, recent_buffer_size]` | recent buffer 有效位 |

这里有几个必须说明的细节：

- 当前 bank 不再按 decoder layer 单独维护。
- 当前 bank 是类别级共享 bank。
- 当前 `memory_size_per_class` 不再表示旧实现中的完整 class memory 池大小。
- 当前代码中 `memory_size_per_class` 主要作为 `recent_buffer_size` 的默认值来源。

### 6.2.7 `get_valid_mask`

功能：

- 用 `prototype_count >= min_count` 判断某个 slot 是否可以参与检索和匹配

当前语义：

- `min_count` 是 slot 支持度门槛
- 不是旧文档里的 layer-class prototype 是否成熟的门槛

### 6.2.8 `get_normalized_bank`

功能：

- 对整个 `prototype_bank` 做 `F.normalize`

用途：

- prototype 匹配
- prototype retrieval
- prototype loss

### 6.2.9 `match_slots`

功能：

- 在计算 `loss_proto` 时，根据 GT 类别标签，给每个正样本 query 找到“同类中最相似的有效 slot”

输入：

- `feats`
- `labels`
- `min_count`

输出：

- `matched_bank`
- `matched_mask`
- `matched_slots`

注意：

- 这里不依赖预测类别分布
- 它使用的是 GT label 指定类别，再在该类内部找最近 slot
- 这是 loss 端的监督匹配，不是 refinement 端的 retrieval

### 6.2.10 `update`

功能：

- 训练时用正样本 query 更新 bank

步骤：

1. 对 `feats`、`labels`、`qualities` 做 `detach`
2. 用 `_all_gather_tensor` 聚合多卡数据
3. 对 `feats` 做归一化
4. 对每个出现的类别调用 `_update_class_online`
5. 对被触达的类别调用 `_maybe_maintain_class`
6. 增加 `prototype_updates`

额外细节：

- 每次更新前，所有已有 slot 的 `prototype_age` 会加一
- 只有被当前 batch 触达的类别，才会进入在线更新和维护流程

### 6.2.11 `_quality_to_weight`

功能：

- 把 head 侧传进来的 `quality` 映射到 `[0.1, 1.0]` 左右的范围

公式：

- `0.1 + 0.9 * sigmoid(quality)`

目的：

- 避免质量过低时完全失去作用
- 避免质量过高时过度放大

### 6.2.12 `_update_class_online`

功能：

- 对某个类别的一批正样本 query 逐个做在线更新

细节：

- 样本先按 `qualities` 从高到低排序
- 质量高的 query 先更新 bank

这意味着：

- 当前实现优先让高质量样本决定 prototype 的演化方向

### 6.2.13 `_update_single_query`

功能：

- 用单个高质量 query 更新对应类别的 bank

分支逻辑如下：

1. 如果该类当前一个 slot 都没有，直接 `_init_slot`
2. 否则，计算 query 与该类所有有效 slot 的相似度
3. 找到最佳 slot 和最佳相似度
4. 如果当前 slot 数还没填满，并且最佳相似度低于 `init_match_threshold`，则新开一个 slot
5. 否则，对最佳 slot 做在线更新
6. 同时把该 query 作为候选放进 recent buffer

这就是当前代码的冷启动和增量扩容策略。

### 6.2.14 `_init_slot`

功能：

- 在该类还有空 slot 的情况下，用当前 query 初始化一个新 slot

初始化内容：

- `prototype_bank = normalize(feat)`
- `prototype_count = 1.0`
- `prototype_quality = quality_weight`
- `prototype_radius = 0.0`
- `prototype_age = 0.0`

这部分是当前训练能正常启动的重要原因：

- 即使 bank 最开始全空，也可以随着高质量正样本逐步填充 slot
- 不需要额外 warmup hook 才能让训练跑起来

### 6.2.15 `_online_update_slot`

功能：

- 对已有 slot 做在线增量更新

核心变量：

- `old_center`
- `old_count`
- `old_quality`
- `old_radius`

步长 `alpha` 的来源：

1. 先根据 `quality_weight` 在线性区间 `[alpha_min, alpha_max]` 内生成基础步长
2. 再按 `1 / sqrt(count + 1)` 衰减
3. 如果 query 与 slot 的相似度低于 `online_match_threshold`，再额外减半
4. 如果 slot 很新，则给一个下界：
   - `count <= 1` 时，下界为 `alpha_min`
   - `1 < count <= 4` 时，下界为 `max(1 - momentum, 0.01)`

需要特别说明：

- 当前 `momentum` 不再直接作为旧实现那种 prototype EMA 系数使用
- 当前 `momentum` 只间接影响“新 slot 在早期阶段的最小更新下界”

更新内容：

- 更新中心向量
- `prototype_count += 1`
- `prototype_quality` 做 EMA
- `prototype_radius` 做 EMA
- `prototype_age = 0`

### 6.2.16 `_add_recent_candidate`

功能：

- 把“高质量且有新颖性”的 query 放入 recent buffer

细节：

- `score = quality_weight * (1 + novelty)`
- 如果 `score < recent_score_threshold`，则不进入 buffer
- 如果和已有 recent token 太像，且相似度高于 `recent_dedup_threshold`，则只在更优时替换
- 如果 buffer 没满，直接插入空位
- 如果 buffer 已满，则替换当前最低分样本，但前提是新样本更好

这相当于一个按质量和新颖性维护的小型候选池。

### 6.2.17 `_maybe_maintain_class`

功能：

- 决定某个类别是否要触发局部维护

触发条件包括：

- `need_fill`：该类当前 slot 数还没填满
- `interval_hit`：达到 `maintenance_interval`
- `recent_trigger`：recent buffer 中候选积累到一定数量
- `drift_trigger`：当前该类某个 slot 的 `prototype_radius` 超过阈值

这意味着当前实现不是每次更新都整类重刷，而是“在线更新为主，局部维护为辅”。

### 6.2.18 `_recluster_class`

功能：

- 对某个类别执行一次局部重聚类维护

输入 token 来源：

- 当前已有 slot
- recent buffer 中的候选 query

每个 token 的权重构成：

- `sqrt(support) * (0.5 + quality)`

流程：

1. 合并现有 slot 和 recent token
2. 计算权重
3. 用 `_select_weighted_diverse_indices` 选 seed
4. 把所有 token 分配到最近 seed
5. 每个簇内部用 `_select_weighted_medoid` 选代表原型
6. 统计每个簇的：
   - `new_bank`
   - `new_count`
   - `new_quality`
   - `new_radius`
7. 清空该类旧状态并用新簇结果替换
8. 清空该类 recent buffer

这里还需要说明几个关键点：

- 当前维护是“按类局部维护”，不是全局重聚类
- 当前重聚类后 `prototype_count` 的语义是该簇的支持度总和
- 这比旧版“每个 slot 写整类总样本数”更有表达力

### 6.2.19 `PrototypeCrossAttention`

这是当前 refinement 真正使用的模块。

当前没有 `QueryDifficultyEstimator`，也没有 `PrototypeRefiner`，而是直接用 cross-attention 把 prototype token 注入 query。

#### 输入

- `query_feat`
- `cls_score`
- `prototype_bank`
- `prototype_count`
- `min_count`

#### 过滤逻辑

- 只有 `prototype_count >= min_count` 的 slot 才参与
- 如果没有任何有效 slot，直接返回原始 `query_feat`

#### prototype 选择逻辑

当前 retrieval 采用的是“先选类别，再使用所选类别下的全部 slot”的方案。

具体逻辑是：

1. 对 `cls_score` 做归一化 sigmoid，得到 `cls_prob`
2. 用 `valid_slots.any(dim=-1)` 得到哪些类别至少有一个成熟 slot
3. 把没有成熟 slot 的类别在 `cls_prob` 中直接 mask 掉
4. 按类别概率选 top `topk_classes` 个类别
5. 对这几个类别，不再做 slot 级打分
6. 直接把这些类别下的全部 slot 展开成 prototype tokens

因此当前 retrieval 的关键特点是：

- 类别级选择使用 `cls_prob`
- slot 级不再做额外排序
- 所选类别下的所有 slot 都会进入 cross-attention
- `prototype_count` 只用于判断 slot 是否成熟，不再参与 slot 分数计算

#### attention 结构

- `query_norm`
- `memory_norm`
- `q_proj`
- `k_proj`
- `v_proj`
- `scaled_dot_product_attention`
- `out_proj`
- `ffn_norm`
- `ffn`

#### 初始化方式

- `out_proj` 全零初始化
- `ffn` 最后一层全零初始化

这意味着：

- 训练初期该模块更接近 identity
- 有助于减少一开始 prototype 注入过强导致的训练不稳定

#### 输出方式

- 只对 `query_has_proto` 的 query 做 refinement
- 更新方式是 residual：
  - `query + attn_out`
  - 再加一层 FFN residual

需要注意：

- 当前 retrieval 使用的是 `prototype_count`
- 当前 retrieval 没有直接使用 `prototype_quality`
- 当前 retrieval 没有直接使用 `prototype_radius`
- 当前 retrieval 没有直接使用 `prototype_age`

这些状态当前只服务于 bank 维护，不直接参与 attention 打分。

## 6.3 `SparseBEV/models/sparsebev_transformer.py`

### 6.3.1 `SparseBEVTransformer.__init__`

相对初始提交新增：

- `proto_query` 参数

作用：

- 把 prototype 配置传递给 decoder

### 6.3.2 `SparseBEVTransformer.forward`

相对初始提交新增输入：

- `prototype_bank`
- `prototype_count`
- `prototype_min_count`
- `dn_pad_size`
- `return_query_feats`

相对初始提交新增输出：

- `final_query_feats`

注意：

- 当前只返回最后一层的 query features
- 不再返回所有 decoder 层的 query features

### 6.3.3 `SparseBEVTransformerDecoder.configure_proto_query`

当前逻辑：

- 解析 `proto_query`
- 读取 `enabled`
- 读取 `refine_layers`
- 如果没有 `refine_layers`，退回到 `use_layers`
- 如果还没有，则默认取 `num_layers - 2`

并对层号做约束：

- 只有 `0 <= layer_idx < num_layers - 1` 的层会被保留

这意味着：

- refinement 只发生在“某层输出后、下一层输入前”
- 最后一层 decoder 之后没有后续层，因此不会作为 refinement 插点

### 6.3.4 当前默认 refinement 层

主配置和 quicktest 配置里都写的是：

- `refine_layers = [0, 1, 2, 3, 4]`

在 `num_layers = 6` 的情况下，全部有效。

也就是说：

- 当前默认会在前五层 decoder 输出后都做 refinement
- 第六层只是输出最终预测，不会再做后处理式注入

### 6.3.5 `prototype_cross_attention` 的构造参数

来自配置或默认值的参数包括：

- `num_prototypes`
- `prototype_topk_classes`
- `prototype_attn_heads`
- `prototype_attn_drop`
- `prototype_ffn_hidden_dim`

其中当前配置里显式写出的只有：

- `num_prototypes = 8`
- `prototype_topk_classes = 2`
- `prototype_attn_heads = 8`

当前默认值但未在配置里显式写出的是：

- `prototype_attn_drop = 0.1`
- `prototype_ffn_hidden_dim = 512`

需要注意：

- 当前 retrieval 不再按 slot 打分截断
- 当前真正生效的是 `prototype_topk_classes`
- 每个被选中的类别会直接使用该类别下的全部 `num_prototypes` 个 slots

### 6.3.6 decoder 主循环

与初始提交相比，当前每层多了这些行为：

1. 保留 `layer_query_feat`
2. 如果 `return_query_feats=True`，则把最后一层 feature 存入 `final_query_feats`
3. 如果当前层属于 `proto_refine_layers`：
   - 去掉 DN 前缀，只保留正常 query
   - 用 `PrototypeCrossAttention` 做 refinement
   - 再把 DN query 和正常 query 拼回去
4. 更新 `query_bbox = bbox_pred.detach()`

注意：

- DN query 不参与 prototype refinement
- 这是通过 `dn_pad_size` 显式切掉前缀实现的

### 6.3.7 未改动部分

当前文件中下列部分与 prototype 无关，保持原始 SparseBEV 逻辑：

- `sampling_4d`
- `make_sample_points`
- `MSMV_CUDA`
- time difference 计算
- `lidar2img` 组织方式
- image features 的预重排
- decoder layer 本体内部结构

## 6.4 `SparseBEV/models/sparsebev_head.py`

### 6.4.1 `__init__`

相对初始提交新增：

- `proto_query` 参数
- `self.proto_enabled`
- `self.proto_refine_layers`
- `self.proto_num_prototypes`
- `self.proto_memory_size`
- `self.proto_min_count`
- `self.proto_loss_weight`
- `self.prototype_bank`
- `self.proto_supervision_layer`

其中：

- `self.proto_supervision_layer = decoder.num_layers - 1`

这意味着当前 prototype loss 只用最后一层 supervision。

### 6.4.2 `QueryPrototypeBank` 的构造参数

当前 head 端显式支持下列 bank 超参：

| 参数 | 当前默认值 | 说明 |
| --- | --- | --- |
| `num_prototypes` | 8 | 每类 slot 数 |
| `memory_size_per_class` | 100 | 当前主要作为 `recent_buffer_size` 默认来源 |
| `bank_momentum` | 0.99 | 当前只间接影响早期更新下界 |
| `recent_buffer_size` | `memory_size_per_class` | recent buffer 长度 |
| `online_match_threshold` | 0.75 | 在线更新是否算“足够接近” |
| `init_match_threshold` | 0.55 | bank 未满时是否开新 slot |
| `maintenance_interval` | 64 | 局部维护周期 |
| `proto_alpha_min` | 0.05 | 在线更新最小步长基值 |
| `proto_alpha_max` | 0.20 | 在线更新最大步长基值 |
| `proto_quality_gamma` | 0.10 | slot 质量 EMA 系数 |
| `proto_radius_gamma` | 0.10 | slot 半径 EMA 系数 |
| `recent_score_threshold` | 0.20 | recent buffer 进入门槛 |
| `recent_dedup_threshold` | 0.95 | recent buffer 去重阈值 |
| `radius_refresh_threshold` | 0.30 | 根据簇离散度触发维护的阈值 |

当前主配置并没有把这些扩展超参全部写出来，很多值仍然依赖默认值。

### 6.4.3 `forward`

相对初始提交，当前新增流程：

1. 从 `self.prototype_bank` 中取出：
   - `prototype_bank`
   - `prototype_count`
2. 计算 `dn_pad_size`
3. 计算 `return_query_feats = self.training and self.proto_enabled`
4. 调用 transformer 时，把 prototype 信息传进去
5. 如果 transformer 返回了最后一层 `final_query_feats`，则把它写入 `outs`

注意：

- 推理阶段默认不回传 `final_query_feats`
- 只有训练阶段并且启用 prototype 时才会返回

### 6.4.4 DN 相关处理

当前 `forward` 中：

- DN query 仍然保留在 transformer 内部参与原始 DETR/DN 机制
- 但在输出 `outs` 时，`final_query_feats` 会和分类、回归结果一起切掉 DN 前缀

因此：

- 后续 `loss_proto` 只基于正常 matching query 计算
- 不会把 DN query 混入 prototype supervision

### 6.4.5 `get_targets`

相对初始提交新增返回：

- `pos_inds_list`
- `neg_inds_list`

原因：

- 后续需要根据正样本索引，从最后一层 query features 中抽取正样本 query

### 6.4.6 `_loss_single_impl`

相对初始提交的变化：

- 把原始 `loss_single` 拆成 `_loss_single_impl`
- 增加 `return_targets` 开关

当 `return_targets=True` 时，会额外返回：

- `labels_list`
- `bbox_targets_list`
- `pos_inds_list`

这给 prototype loss 复用了同一套匹配结果。

### 6.4.7 `compute_query_quality`

当前实现新增了 query quality 估计，用于 bank 更新。

质量来源由两部分误差构成：

1. 分类误差
2. bbox 误差

分类误差：

- 取 GT label 对应的 `gt_logits`
- 用 `binary_cross_entropy_with_logits(gt_logits, 1)` 计算

bbox 误差：

- 对 GT bbox 做 `normalize_bbox`
- 计算 `abs(pred - target)`
- 再乘 `code_weights` 求均值

最终质量：

- `-(cls_error + bbox_error)`

这意味着：

- 分类越准、框越准，quality 越高
- 这是当前在线 bank 更新的核心驱动信号

### 6.4.8 `collect_positive_queries`

功能：

- 根据 `pos_inds_list` 从每张图中收集：
  - `pos_feats`
  - `pos_labels`
  - `pos_scores`
  - `pos_bbox_preds`
  - `pos_bbox_targets`

然后调用 `compute_query_quality` 得到：

- `pos_quality`

### 6.4.9 `calc_prototype_loss`

当前实现中的 prototype loss 不是旧文档描述的“temperature CE 到类 prototype”，而是更简单的“同类最近 slot cosine 对齐损失”。

流程：

1. 从最后一层 `final_query_feats` 中抽取正样本 query
2. 生成 `bank_update = (feats, labels, qualities)`
3. 如果 `proto_loss_weight <= 0` 或没有正样本，则直接返回零损失
4. 调用 `prototype_bank.match_slots`
5. 只保留匹配到有效 slot 的正样本
6. 对 query 和 matched prototype 做归一化
7. 计算：

```text
loss_proto = 1 - cosine(query_feat, matched_slot_proto)
```

因此当前 `loss_proto` 的性质是：

- 监督对象是“同类中最相似且已成熟的 slot”
- 不是“所有类 prototype 上做分类式对比学习”
- 不依赖 temperature 参数
- 不需要显式负类 prototype

### 6.4.10 `update_prototype_bank`

功能：

- 把 `bank_update` 中的：
  - `feats`
  - `labels`
  - `qualities`
 传给 `self.prototype_bank.update`

也就是说：

- 当前 bank 更新发生在 `loss` 内部
- 是训练时在线更新
- 更新使用的是正样本 query 的 GT label 和 quality

### 6.4.11 `loss`

相对初始提交新增内容：

1. 从 `preds_dicts` 中读取 `final_query_feats`
2. 对所有 decoder 层调用 `loss_single_with_targets`
3. 额外保存 `all_target_infos`
4. 在最后一层上计算 `loss_proto`
5. 如果启用 prototype，则把 `loss_proto` 写入 `loss_dict`
6. 调用 `update_prototype_bank`

注意：

- prototype loss 当前只对最后一层 decoder 生效
- bank 更新也只使用最后一层正样本 query

## 6.5 `SparseBEV/configs/r50_nuimg_704x256.py`

### 6.5.1 新增 `proto_query_cfg`

当前主配置显式写出的参数是：

| 参数 | 当前值 |
| --- | --- |
| `enabled` | `True` |
| `num_prototypes` | `8` |
| `memory_size_per_class` | `100` |
| `bank_momentum` | `0.99` |
| `min_memory_count` | `8` |
| `refine_layers` | `[0, 1, 2, 3, 4]` |
| `lambda_proto` | `0.1` |
| `prototype_cross_attn` | `True` |
| `prototype_topk_classes` | `2` |
| `prototype_attn_heads` | `8` |

### 6.5.2 接入位置

当前 `proto_query_cfg` 同时接入：

- `SparseBEVHead`
- `SparseBEVTransformer`

### 6.5.3 与初始提交相比的其他变化

主配置还额外调整了：

- `MyTextLoggerHook.interval`：从 `1` 变成 `50`
- `MyTensorboardLoggerHook.interval`：从 `500` 变成 `50`
- `eval_config.interval`：从 `total_epochs` 变成 `1`

其他训练和数据配置没有本质变化。

## 6.6 `SparseBEV/configs/r50_nuimg_704x256-quicktest.py`

这个文件在初始提交中不存在，是当前工作区新增的快速实验配置。

与主配置相比，它的主要区别是：

- 数据文件改成 mini 版本：
  - `nuscenes_infos_train_mini_sweep.pkl`
  - `nuscenes_infos_val_mini_sweep.pkl`
  - `nuscenes_infos_test_mini_sweep.pkl`
- `proto_query_cfg` 目前与主配置保持一致
- 日志和评估频率更接近初始主配置：
  - `MyTextLoggerHook.interval = 1`
  - `MyTensorboardLoggerHook.interval = 500`
  - `eval_config.interval = total_epochs`

因此 quicktest 更适合快速确认功能是否能跑通，而不是频繁验证完整 val。

## 6.7 `SparseBEV/configs/r50_nuimg_704x256-quick.py`

这个文件只有一行：

```python
_base_ = ['./r50_nuimg_704x256-quicktest.py']
```

它本身没有独立配置项，只是给 quicktest 配置提供一个单独入口。

## 7. 当前代码与旧文档的偏差

当前仓库中有两份旧文档：

- `SparseBEV/docs/proto_temporal_sparsebev_design.md`
- `SparseBEV/docs/changes_vs_initial_branch.md`

这两份文档都不能再被当作“当前工作区真实实现”的准确描述。

### 7.1 与 `proto_temporal_sparsebev_design.md` 的偏差

| 文档中的描述 | 当前代码真实情况 | 影响 |
| --- | --- | --- |
| 按 decoder layer 维护 bank，形状是 `[L, num_classes, C]` | 当前 bank 是 `[num_classes, num_prototypes, C]` | 不再按层分离 prototype 空间 |
| 存在 `QueryDifficultyEstimator` | 当前代码没有这个类 | 当前没有显式难度分数 |
| 存在 `PrototypeRefiner` | 当前代码没有这个类 | 当前 refinement 由 cross-attention 完成 |
| refinement 依赖 `cls_entropy / cls_margin / bbox_delta_norm / query_drift` | 当前代码没有使用这些信号 | 当前不再做 difficulty-aware gating |
| 使用 `mix_query_prototypes` 生成 prototype 混合向量 | 当前代码没有这个函数 | 当前改为 slot-aware token retrieval |
| prototype loss 是 temperature CE | 当前代码是 cosine alignment 到 matched slot | loss 形式发生变化 |
| 收集所有 decoder 层 `all_query_feats` | 当前代码只用最后一层 `final_query_feats` | supervision 范围缩小 |
| `min_proto_count` 保护类 prototype | 当前代码用 `min_memory_count` 保护 slot | 门槛语义变化 |

### 7.2 与 `changes_vs_initial_branch.md` 的偏差

这份文档的问题更明显，因为它描述的是更早阶段的一版本地实现。

主要偏差包括：

- 文档把比较基线写成了“`HEAD -> 当前工作区`”，而不是“`5910ed2 -> 当前工作区`”
- 文档声称存在：
  - `QueryDifficultyEstimator`
  - `PrototypeRefiner`
  - `mix_query_prototypes`
- 文档声称当前实现是：
  - 按 layer 维护 prototype
  - 收集所有 decoder 层 query features
  - prototype loss 使用 temperature CE
- 这些都已经不是当前工作区代码的真实情况

因此：

- 旧变更文档只能作为“历史阶段说明”
- 不能作为当前实现的行为说明书

## 8. 当前真实流程的端到端描述

为了便于后续继续改动，这里把当前 prototype 相关流程按执行顺序完整描述一次。

### 8.1 模型构建阶段

1. 配置文件定义 `proto_query_cfg`
2. `SparseBEVHead` 读取配置
3. `SparseBEVHead` 调用 `decoder.configure_proto_query`
4. `SparseBEVHead` 创建 `QueryPrototypeBank`
5. `SparseBEVTransformerDecoder` 创建 `PrototypeCrossAttention`

### 8.2 训练前向阶段

1. head 初始化 query bbox 与 query feat
2. head 根据 GT 构建 DN query
3. head 从 bank 中读取：
   - `prototype_bank`
   - `prototype_count`
4. head 调用 transformer
5. decoder 每层先执行原始 SparseBEV block
6. 指定层把正常 query 拿出来
7. `PrototypeCrossAttention` 先按类别选 top-k 类，再取这些类别下的全部 slot token
8. refinement 后的 query 送入下一层
9. 最后一层 query feature 作为 `final_query_feats` 返回给 head

### 8.3 loss 计算阶段

1. head 先算原始分类与回归损失
2. head 保留最后一层的 target 信息
3. head 从最后一层 `final_query_feats` 中抽取正样本
4. head 计算正样本 quality
5. head 用 GT label 在同类 slot 中找最近 prototype
6. head 计算 `loss_proto`
7. head 把正样本 `(feat, label, quality)` 传给 bank

### 8.4 bank 更新阶段

1. bank 收到多卡聚合后的正样本
2. 每个类别按 quality 从高到低处理
3. 对每个 query：
   - 若类内没有 slot，则新建
   - 若 bank 未满且与已有 slot 差异足够大，则新建
   - 否则在线更新最近 slot
4. 同时把高质量高新颖性的 query 放进 recent buffer
5. 满足条件时，对该类做一次局部重聚类维护

## 9. 明确未改动的部分

尽管当前工作区增加了 prototype 机制，但以下部分与初始提交保持同一路线，没有被 prototype 改造：

- `SparseBEV/models/sparsebev_sampling.py`
- `SparseBEV/models/csrc/*`
- `sampling_4d`
- `make_sample_points`
- 数据集实现
- 数据增强管线
- 相机视角选择逻辑
- frame-level 或 view-level 采样权重

换句话说，当前所有 prototype 改动都发生在：

- decoder 层间的 query feature 空间
- head 的 supervision 与 bank update 侧

## 10. 需要特别注意的实现细节

下面这些点在后续继续修改时最容易误判，单独列出来。

### 10.1 `memory_size_per_class` 的语义已经变化

旧版本和旧文档里，它对应“class memory”的容量。

当前代码里：

- 它本身只保存为字段
- 真正被使用的是 `recent_buffer_size`
- 默认情况下 `recent_buffer_size = memory_size_per_class`

因此如果后面要继续加 memory 机制，不要误以为当前仍然存在旧版完整 memory 池。

### 10.2 `momentum` 的语义已经变化

旧版本和旧文档里，它是 prototype EMA 的核心系数。

当前代码里：

- 它不再直接参与“`new = m * old + (1 - m) * current_mean`”这种更新
- 它只在新 slot 早期阶段，用来生成一个最小更新下界

如果后面想恢复显式 EMA 语义，需要重新设计这部分。

### 10.3 当前 retrieval 没有直接使用 quality、radius、age

虽然 bank 中维护了：

- `prototype_quality`
- `prototype_radius`
- `prototype_age`

但当前 `PrototypeCrossAttention` 只使用：

- `prototype_bank`
- `prototype_count`

因此：

- 这些额外状态目前只服务于 bank 维护
- 还没有进入 retrieval scoring

另外还要注意：

- 当前 retrieval 也不再做 slot 级打分
- 它只按类别概率选 top-k 类，再直接展开这些类别下的全部成熟 slots

### 10.4 当前 `loss_proto` 只使用最后一层

当前不是多层 prototype supervision。

当前只有：

- 最后一层 query features
- 最后一层分类与回归输出
- 最后一层匹配结果

参与 `loss_proto` 和 bank update。

### 10.5 当前 refinement 层是 `[0, 1, 2, 3, 4]`

因为 decoder 一共有 6 层，所以：

- 第 0 到第 4 层输出后都做 refinement
- 第 5 层只输出最终结果

如果后面修改 `refine_layers`，需要记住最后一层不会被保留为 refinement 插点。

## 11. 验证情况

本次梳理对应的当前工作区代码，已经做过以下基础检查：

- `python3 -m py_compile SparseBEV/models/proto_query.py`
- `python3 -m py_compile SparseBEV/models/sparsebev_transformer.py`
- `python3 -m py_compile SparseBEV/models/sparsebev_head.py`
- `python3 -m py_compile SparseBEV/configs/r50_nuimg_704x256.py`
- `python3 -m py_compile SparseBEV/configs/r50_nuimg_704x256-quicktest.py`
- `python3 -m py_compile SparseBEV/configs/r50_nuimg_704x256-quick.py`

这些语法检查通过，说明当前修改至少在 Python 语法层面是可解析的。

但是：

- 当前没有在这个终端里完成完整训练或评测
- 当前没有在这份文档中给出 NDS、mAP 或 loss 曲线结论

因此本文档描述的是“当前真实代码结构”，不是“已经被实验完全验证的结论”。

## 12. 最终结论

如果只用一句话概括当前工作区相对初始提交的真实变化，可以写成：

当前工作区把原始 SparseBEV 扩展成了一套“类别级多 prototype slot 的在线 bank + 局部维护 + query-conditioned prototype cross-attention + 最后一层 prototype cosine alignment loss”的系统，而仓库中的旧文档仍然停留在更早期的“按层单 prototype + difficulty-aware refinement”描述上。

后续如果继续修改 prototype 机制，应优先以当前代码为准，而不是以旧文档为准。
