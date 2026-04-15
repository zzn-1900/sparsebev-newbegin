# 当前工作区相对 `5910ed2` 的累计改动清单

## 1. 比较口径

- 基线提交: `5910ed2f47e77eccf765af6e39106f33c7f190f8` (`Initial commit`)
- 当前代码状态: 本地工作区在撰写本文档前的真实代码状态
- 当前分支状态: `prototype`，`HEAD = 9cacb16`，相对 `origin/prototype` 超前 1 个提交

说明:

- 下面“累计改动”统计的是 `git diff 5910ed2f47e77eccf765af6e39106f33c7f190f8` 的结果。
- 为了避免自引用，本文档本身不计入下面的模型改动统计。
- 当前工作区里还存在少量相对 `HEAD` 的未提交删除项，这部分单独列在文末“附录”中。

## 2. 总体结论

相对初始提交 `5910ed2`，当前工作区的核心变化非常集中，全部落在以下 6 个文件上:

| 变更类型 | 文件 | 行数变化 |
| --- | --- | --- |
| A | `SparseBEV/configs/r50_nuimg_704x256-quicktest.py` | `+257 / -0` |
| M | `SparseBEV/configs/r50_nuimg_704x256.py` | `+24 / -3` |
| M | `SparseBEV/models/__init__.py` | `+11 / -9` |
| A | `SparseBEV/models/proto_query.py` | `+554 / -0` |
| M | `SparseBEV/models/sparsebev_head.py` | `+270 / -13` |
| M | `SparseBEV/models/sparsebev_transformer.py` | `+120 / -10` |

累计统计:

- `6 files changed, 1236 insertions(+), 35 deletions(-)`

一句话概括当前代码状态:

- 当前仓库已经从“原始 SparseBEV”演化成“带 query prototype bank、prototype cross-attention refinement、prototype alignment loss、quicktest 配置”的版本。

同样重要的是，当前改动没有碰下面这些模块:

- `SparseBEV/models/sparsebev_sampling.py`
- `SparseBEV/models/csrc/*`
- `SparseBEV/loaders/*`
- `SparseBEV/train.py`
- `SparseBEV/val.py`
- `SparseBEV/utils.py`

也就是说，这次累计改动没有改视角采样 CUDA 链路、数据集管线和训练入口，主要只改了 `config / head / transformer / prototype module` 这条线。

## 3. 提交演进轨迹

从 `5910ed2` 到当前 `HEAD`，中间共有 7 个后续提交:

| 提交 | message | 备注 |
| --- | --- | --- |
| `dff681c` | `v1.0` | 首次落地 prototype 相关代码、文档、quicktest 配置 |
| `eee8f98` | `v1.0` | 对主配置做小幅调整 |
| `fe64248` | `初代版本` | prototype 模块、head、transformer 出现较大重写 |
| `050d09c` | `更新` | 补充了对比说明文档 |
| `f6b11d4` | `quick配置` | quicktest 配置与 prototype 细节继续调整 |
| `a7fd826` | `优化簇更新机制` | prototype bank 的簇刷新/更新机制继续演化 |
| `9cacb16` | `修bug` | 对 prototype 与 transformer 交互做 bugfix |

从提交轨迹上看，prototype 相关能力不是一次性加完的，而是经历了“先接入、再重写、再优化簇更新、最后修 bug”的持续演化。

## 4. 当前累计改动的真实功能面貌

如果只看当前工作区真实代码，当前能力可以概括成下面这条链路:

1. 主配置里新增 `proto_query_cfg`
2. `SparseBEVHead` 根据配置构建 `QueryPrototypeBank`
3. `SparseBEVHead.forward` 把 `prototype_bank / prototype_count / prototype_min_count / dn_pad_size` 传给 transformer
4. `SparseBEVTransformerDecoder` 在指定 decoder 层后，对非 DN query 做 prototype cross-attention refinement
5. `SparseBEVHead.loss` 用最后一层 decoder query 的正样本计算 `loss_proto`
6. 同一批正样本 query 还会用于在线更新 prototype bank

需要特别注意的事实:

- 当前实现是“按类别维护多 slot prototype bank”，不是“按 decoder layer 维护一套 bank”。
- 当前 refinement 使用的是 `PrototypeCrossAttention`，不是旧设计里常见的 `QueryDifficultyEstimator + PrototypeRefiner` 组合。
- 当前 `loss_proto` 和 bank update 只使用最后一层 decoder 的 query feature，不是每层都监督。
- DN query 被显式排除在 prototype refinement 之外。

## 5. 逐文件详细改动

### 5.1 `SparseBEV/configs/r50_nuimg_704x256.py`

这是当前主训练配置，相对 `5910ed2` 的主要改动有 3 类。

第一类是新增 `proto_query_cfg`，参数大致分为四组:

- bank 规模:
  - `num_prototypes=8`
  - `memory_size_per_class=100`
  - `query_bank_size=100`
  - `bank_momentum=0.99`
  - `min_memory_count=8`
- refinement 行为:
  - `refine_layers=[0, 1, 2, 3, 4]`
  - `prototype_cross_attn=True`
  - `prototype_topk_classes=2`
  - `prototype_attn_heads=8`
- loss 权重:
  - `lambda_proto=0.1`
- bank 更新阈值:
  - `query_merge_threshold=0.75`
  - `query_new_threshold=0.55`
  - `query_score_threshold=0.20`
  - `query_dedup_threshold=0.95`
  - `query_replace_threshold=0.30`

第二类是把 `proto_query_cfg` 同时接到了两个入口:

- `pts_bbox_head.proto_query=proto_query_cfg`
- `transformer.proto_query=proto_query_cfg`

第三类是训练可观测性调整:

- `MyTextLoggerHook.interval` 从 `1` 改成 `50`
- `MyTensorboardLoggerHook.interval` 从 `500` 改成 `50`
- `eval_config.interval` 从 `total_epochs` 改成 `1`

这说明当前主配置已经默认启用 prototype 机制，而且训练时会更频繁做评估，同时降低日志刷屏频率。

### 5.2 `SparseBEV/configs/r50_nuimg_704x256-quicktest.py`

这是相对 `5910ed2` 新增的一份快速实验配置，本质上是主配置的一份轻量化实验副本。

它的主要特点是:

- 保留了和主配置同一套 `proto_query_cfg`
- 仍然使用相同的 `SparseBEVHead` 和 `SparseBEVTransformer` 接线方式
- 把数据注释文件换成了 `nuscenes_infos_*_mini_sweep.pkl`
- `total_epochs` 改成 `5`

具体看数据部分:

- 训练集使用 `nuscenes_infos_train_mini_sweep.pkl`
- 验证集使用 `nuscenes_infos_val_mini_sweep.pkl`
- 测试集使用 `nuscenes_infos_test_mini_sweep.pkl`

因此它的用途很明确:

- 不是新算法逻辑
- 而是为了快速验证 prototype 相关改动是否能正常跑通

### 5.3 `SparseBEV/models/__init__.py`

这个文件的功能改动很小，但意义很明确。

相对初始提交，它新增导出:

- `QueryPrototypeBank`
- `PrototypeCrossAttention`

这意味着 prototype 机制被正式纳入 `SparseBEV.models` 对外暴露的模块集合中。

从这里也可以反向确认一件事:

- 当前仓库真正稳定落地的 prototype 相关核心对象就是这两个
- 不是旧文档里提到的 `QueryDifficultyEstimator` 或 `PrototypeRefiner`

### 5.4 `SparseBEV/models/proto_query.py`

这是相对 `5910ed2` 新增的核心文件，也是当前所有改动里最关键的一部分。

当前文件包含的主要对象如下:

- `normalize_query_logits`
- `_all_gather_tensor`
- `_select_weighted_diverse_indices`
- `_select_weighted_medoid`
- `QueryPrototypeBank`
- `PrototypeCrossAttention`

#### 5.4.1 `normalize_query_logits`

当前 retrieval 前的类别分数不是直接 softmax，而是:

1. 先对 `cls_score` 做 `sigmoid`
2. 再沿类别维做归一化

这决定了 prototype cross-attention 在选 top-k 类别时使用的是“归一化后的 sigmoid 分类概率”。

#### 5.4.2 `_all_gather_tensor`

这个辅助函数解决的是 DDP 环境下的变长 tensor 汇总问题。

当前逻辑是:

- 先收集各卡样本数
- 按最大长度做 padding
- `all_gather`
- 再按真实长度裁回

用途:

- bank update 时收集不同 rank 上的正样本 query
- 让 prototype bank 的更新来源跨卡一致

#### 5.4.3 `QueryPrototypeBank`

当前 bank 的结构不是 layer-wise，而是 class-wise multi-slot。

核心 buffer 包括:

- `prototype_bank`: `[num_classes, num_prototypes, embed_dims]`
- `prototype_count`
- `prototype_quality`
- `prototype_radius`
- `prototype_age`
- `prototype_updates`

另外还有一套“每类当前 exemplar 池”:

- `bank_query_feats`
- `bank_query_support`
- `bank_query_quality`
- `bank_query_score`
- `bank_query_valid`

当前更新路径可以概括为:

1. 收集正样本 query 的 `feat / label / quality`
2. DDP 下先 `_all_gather_tensor`
3. 对每个类别依次执行 exemplar 更新
4. exemplar 更新内部按相似度和阈值决定 `add / merge / replace`
5. 类内 exemplar 池更新完后，再刷新固定数量的 prototype slots

其中 exemplar 维护的关键策略如下:

- `query_score_threshold` 决定候选 query 是否值得进入池子
- `query_dedup_threshold` 决定是否应视作重复样本并合并
- `query_merge_threshold` 决定是否并入现有 exemplar
- `query_new_threshold` 决定空位存在时是否应该新开 exemplar
- `query_replace_threshold` 决定池子满时是否值得替换旧 exemplar

slot 刷新逻辑不是简单均值，而是:

- 先用 `_select_weighted_diverse_indices` 选出多样化种子
- 再把当前 exemplar 分配到最近的种子
- 每个簇里用 `_select_weighted_medoid` 选 medoid 作为 slot prototype
- 同时统计 `count / quality / radius`

因此当前实现更接近:

- “在线 exemplar 池 + 多样化簇刷新”

而不是:

- “每次直接用正样本均值做 EMA”

#### 5.4.4 `match_slots`

这个方法负责训练时把正样本 query 和同类 prototype 对齐。

当前行为:

- 先按类别筛样本
- 只在该类别内的有效 slot 里匹配
- 用 cosine similarity 选最相近的 slot

这为 `loss_proto` 提供了“同类 query 对应哪一个 prototype slot”的映射。

#### 5.4.5 `PrototypeCrossAttention`

这是当前 transformer 中真正执行 refinement 的模块。

它的工作流程是:

1. 根据 `cls_score` 选每个 query 的 top-k 类别
2. 从这些类别中取出所有 prototype tokens
3. 用 `prototype_count >= min_count` 作为有效性掩码
4. 以 query 为 `q`，prototype tokens 为 `k/v` 做 `scaled_dot_product_attention`
5. 经过 `out_proj`
6. 再接一层残差 FFN

几个关键实现细节:

- 只有“有可用 prototype token”的 query 才会被 refinement
- 没有可用 prototype 的 query 会直接保留原特征
- 训练冷启动阶段额外有 `_attach_ddp_zero_residual`，用来避免 DDP 报 unused parameter

从当前代码看，真正落地的是“prototype cross-attention 注入语义先验”，而不是“显式难度估计器驱动的门控 refinement”。

### 5.5 `SparseBEV/models/sparsebev_transformer.py`

这个文件的累计改动，核心就是把 `proto_query.py` 中的能力接进 decoder 主循环，同时尽量不碰原始 sampling 主干。

#### 5.5.1 构造函数改动

`SparseBEVTransformer.__init__` 和 `SparseBEVTransformerDecoder.__init__` 都新增了:

- `proto_query=None`

decoder 初始化后会立即执行:

- `self.configure_proto_query(proto_query)`

#### 5.5.2 `configure_proto_query`

这个新方法负责把配置转成 decoder 内部行为，主要做了几件事:

- 解析 `enabled`
- 解析 `refine_layers`
- 生成 `self.proto_refine_layers`
- 判断 `self.prototype_refine_enabled`
- 在启用时构建 `PrototypeCrossAttention`

当前主配置 `refine_layers=[0, 1, 2, 3, 4]`，而 decoder 一共 6 层，所以现在的行为是:

- 第 0 到第 4 层输出后允许做 prototype refinement
- 最后一层只输出最终结果，不再把 refinement 送到后继层

#### 5.5.3 `forward` 输入输出扩展

相对初始提交，transformer `forward` 多了下面这些输入:

- `prototype_bank`
- `prototype_count`
- `prototype_min_count`
- `dn_pad_size`
- `return_query_feats`

输出从原来的:

- `cls_scores`
- `bbox_preds`

变成了:

- `cls_scores`
- `bbox_preds`
- `final_query_feats`

这里返回的是“最后一层 query feat”，不是“每一层 query feat 列表”。

#### 5.5.4 decoder 主循环中的新逻辑

当前每层 decoder 的处理顺序变成:

1. 先执行原有 `decoder_layer`
2. 拿到 `layer_query_feat / cls_score / bbox_pred`
3. 如果当前层在 `proto_refine_layers` 中，则只对非 DN query 做 `PrototypeCrossAttention`
4. refinement 后的 query_feat 再送去下一层
5. `bbox_pred` 仍然照常 `detach` 后作为下一层参考框

其中 DN query 的处理方式很明确:

- 用 `dn_pad_size` 切分
- 只对 `[:, dn_pad_size:]` 的正常 query 做 refinement
- DN 前缀 `[:, :dn_pad_size]` 原样拼回

#### 5.5.5 明确未动的内容

当前 transformer 虽然接入了 prototype，但下面这些函数和链路没有被改:

- `sampling_4d`
- `make_sample_points`
- 多视角采样的 CUDA 封装调用

所以这次改动并不是在 feature sampling 层面动刀，而是在 query feature 层面追加 prototype 先验。

### 5.6 `SparseBEV/models/sparsebev_head.py`

这是当前累计改动里业务逻辑最完整的一处，它承担了配置解析、bank 持有、loss 计算、bank 更新四项职责。

#### 5.6.1 初始化阶段

`__init__` 新增了 `proto_query=None` 参数，并解析出一整套内部字段:

- `self.proto_enabled`
- `self.proto_refine_layers`
- `self.proto_num_prototypes`
- `self.proto_memory_size`
- `self.proto_min_count`
- `self.proto_loss_weight`
- `self.prototype_bank`
- `self.proto_supervision_layer`

当 `proto_enabled=True` 时，还会:

1. 调用 `self.transformer.decoder.configure_proto_query(...)`
2. 读取 decoder 的 `proto_refine_layers`
3. 把 `proto_supervision_layer` 设为 `decoder.num_layers - 1`
4. 构建 `QueryPrototypeBank`

这里有两个很关键的结论:

- bank 实例由 head 持有，不是 transformer 持有
- prototype loss 的监督层固定为最后一层

#### 5.6.2 `forward`

当前 `forward` 新增了与 prototype 相关的准备逻辑:

- 从 `self.prototype_bank` 取 `prototype_bank`
- 从 `self.prototype_bank` 取 `prototype_count`
- 从 DN mask 里计算 `dn_pad_size`
- 仅在 `self.training and self.proto_enabled` 时要求 transformer 返回 `final_query_feats`

随后调用 transformer 时，显式传入:

- `prototype_bank`
- `prototype_count`
- `prototype_min_count`
- `dn_pad_size`
- `return_query_feats`

如果存在 DN query，输出里还会把:

- `all_cls_scores`
- `all_bbox_preds`
- `final_query_feats`

一起裁掉 DN 前缀，只把真实 query 部分放进 `outs`。

#### 5.6.3 target 信息增强

`get_targets` 的返回值相对初始提交新增了:

- `pos_inds_list`
- `neg_inds_list`

这一步是为了后面按正样本索引，从 `final_query_feats` 中抽取和 GT 匹配成功的 query。

#### 5.6.4 损失函数重构

原来的 `loss_single` 被拆成:

- `_loss_single_impl(...)`
- `loss_single(...)`
- `loss_single_with_targets(...)`

这样做的目的不是改分类/回归损失本身，而是让 loss 计算时顺手把 target 对齐信息返回出来，供 prototype 分支复用。

#### 5.6.5 query quality 计算

当前新增了 `compute_query_quality`。

quality 的定义来自两部分误差:

- GT 类别 logit 的 BCE 误差
- bbox 预测相对 GT 的加权绝对误差

最终返回的是:

- `-(cls_error + bbox_error)`

也就是说:

- 分类越准、框越准，quality 越高
- quality 之后会作为 bank 更新时的样本质量权重

#### 5.6.6 正样本 query 收集

`collect_positive_queries` 做的事情很直接:

- 根据 `pos_inds_list`
- 从 `query_feats / cls_scores / bbox_preds / bbox_targets` 中抽正样本
- 拼出 `pos_feats / pos_labels / pos_quality`

这一步明确把 prototype 训练信号限定在“和 GT 成功匹配的正样本 query”上。

#### 5.6.7 `loss_proto`

当前 `calc_prototype_loss` 的逻辑如下:

1. 只在 `proto_enabled` 且 `prototype_bank` 存在时继续
2. 从最后一层 query 中收集正样本
3. 先生成一份 `bank_update=(pos_feats, pos_labels, pos_quality)`
4. 若 `lambda_proto <= 0` 或没有正样本，则 loss 为 0，但 bank 仍然可以更新
5. 调用 `self.prototype_bank.match_slots(...)` 找到同类可用 slot
6. 对有匹配 slot 的样本，计算 cosine alignment loss:
   - `loss_proto = 1 - cos(pos_feat, matched_bank_slot)`

当前 `loss_proto` 的本质不是分类损失或对比学习大框架，而是一个非常直接的同类 prototype 对齐项。

#### 5.6.8 在线 bank 更新

`update_prototype_bank` 在 `loss()` 里被调用，输入就是上一步得到的 `bank_update`。

实际效果是:

- 每个训练 batch 结束时
- 最后一层正样本 query 会在线刷新 class-wise prototype bank

这让当前系统形成了一个闭环:

- forward 时使用历史 bank 做 refinement
- loss 时用当前 batch 的正样本对齐 bank
- 同时再用当前 batch 正样本回写 bank

#### 5.6.9 `loss()` 总体变化

当前 `loss()` 相比初始提交多了三件事:

1. 从 `preds_dicts` 读取 `final_query_feats`
2. `multi_apply` 调用 `loss_single_with_targets`，拿到 `all_target_infos`
3. 在最后一层上追加 `loss_proto` 和 `update_prototype_bank`

但原始检测主损失仍然保留:

- `loss_cls`
- `loss_bbox`
- `d{i}.loss_cls`
- `d{i}.loss_bbox`
- DN loss

因此当前 head 不是“改写原损失”，而是在原损失旁边追加了一条 prototype 分支。

## 6. 当前改动真正改变了什么

从运行行为上看，相对 `5910ed2`，当前代码新增了下面这些能力:

1. 训练时可以维护一个按类别组织、每类多个 slot 的 prototype bank
2. decoder 中间层可以读取 prototype，并对正常 query 做 cross-attention refinement
3. 最后一层正样本 query 可以通过 `loss_proto` 显式向同类 prototype 对齐
4. 正样本 query 还能在线反哺 prototype bank，使 bank 随训练持续更新
5. 提供了一份基于 mini sweep 标注的 quicktest 配置，方便快速验证逻辑

同时，当前代码没有新增下面这些方向的改动:

1. 没有改任何采样点生成逻辑
2. 没有改 CUDA `msmv_sampling` 实现
3. 没有改 view / frame 选择策略
4. 没有改数据增强或数据加载流程
5. 没有改主训练脚本入口

## 7. 附录: 当前工作区相对 `HEAD` 的额外未提交状态

虽然相对 `5910ed2` 的累计代码改动只有上面 6 个文件，但在本次检查开始时，当前工作区相对 `HEAD` 还存在下面这些未提交删除项:

- `SparseBEV/docs/changes_vs_initial_branch.md`
- `SparseBEV/docs/proto_temporal_sparsebev_design.md`
- `Xue_CorrBEV_Multi-View_3D_Object_Detection_by_Correlation_Learning_with_Multi-modal_CVPR_2025_paper.pdf`

这些删除项不会出现在 `git diff 5910ed2...当前工作区` 中，原因是:

- 它们都是在 `5910ed2` 之后才被加入仓库的
- 但当前工作区又把它们删掉了
- 所以相对 `5910ed2` 来看，它们是“加入后又删掉”，最终净变化为 0

如果后续还要继续维护这套对比文档，建议把“相对基线提交的累计差异”和“相对当前 `HEAD` 的未提交工作区状态”继续分开记录，这样最不容易混淆。
