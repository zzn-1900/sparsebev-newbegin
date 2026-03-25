# 与最开始分支状态的改动对比

## 1. 比较基线

本文件中的“最开始的分支”按当前仓库可确认的状态解释为:

- 当前所在分支: `main`
- 比较基线: `main` 分支当前 `HEAD`
- 对比对象: 本次本地未提交改动

说明:

- 由于当前没有额外的“修改前快照分支”或单独 commit，本文件按 `HEAD -> 当前工作区` 来描述变化。
- 其中 `git diff HEAD` 能看到已跟踪文件的改动。
- 新增但尚未被 git 跟踪的文件，会单独列出。

## 2. 总体结论

这次改动只沿着 `query / prototype / loss` 这条线展开，没有修改任何视角采样实现。

核心变化可以概括为:

1. 新增 query-space prototype 模块
2. 在 decoder 层间加入 difficulty-aware prototype refinement
3. 在 detection head 中加入 prototype bank 管理与 `loss_proto`
4. 在默认配置中接入 `proto_query_cfg`
5. 补充设计与变更说明文档

明确未改动的部分:

- `SparseBEV/models/sparsebev_sampling.py`
- `SparseBEV/models/csrc/*`
- 数据集与数据管线
- 视角选择、多视角融合、采样点生成逻辑

## 3. 已跟踪文件改动

根据 `git diff --name-status HEAD`，已跟踪文件改动如下:

- `M SparseBEV/configs/r50_nuimg_704x256.py`
- `M SparseBEV/models/__init__.py`
- `M SparseBEV/models/sparsebev_head.py`
- `M SparseBEV/models/sparsebev_transformer.py`

`git diff --stat HEAD` 结果:

```text
 SparseBEV/configs/r50_nuimg_704x256.py    |  14 +++
 SparseBEV/models/__init__.py              |  20 ++--
 SparseBEV/models/sparsebev_head.py        | 161 +++++++++++++++++++++++++++++-
 SparseBEV/models/sparsebev_transformer.py | 142 ++++++++++++++++++++++++--
 4 files changed, 314 insertions(+), 23 deletions(-)
```

注意:

- 上面的统计不包含新增但未跟踪文件。

## 4. 新增但未跟踪文件

当前工作区里，本次相关新增文件还有:

- `SparseBEV/models/proto_query.py`
- `SparseBEV/docs/proto_temporal_sparsebev_design.md`
- `SparseBEV/docs/changes_vs_initial_branch.md`

另外还有一个未跟踪的论文 PDF:

- `Xue_CorrBEV_Multi-View_3D_Object_Detection_by_Correlation_Learning_with_Multi-modal_CVPR_2025_paper.pdf`

其中 PDF 不是代码改动的一部分，只是工作目录中的论文文件。

## 5. 逐文件改动说明

### 5.1 `SparseBEV/models/proto_query.py`

这是本次新增的核心模块文件，提供了三类能力:

1. `QueryPrototypeBank`
2. `QueryDifficultyEstimator`
3. `PrototypeRefiner`

以及两个辅助函数:

- `normalize_query_logits`
- `mix_query_prototypes`

具体新增内容:

- `QueryPrototypeBank`
  - 按 `decoder layer x class` 维护 prototype
  - 使用 EMA 更新
  - 支持 DDP 下的 `all_gather`
- `QueryDifficultyEstimator`
  - 仅使用 decoder 输出构建难度分数
  - 输入信号是 `cls_entropy / cls_margin / bbox_delta_norm / query_drift`
- `PrototypeRefiner`
  - 用 `query_feat + proto_feat + diff_score` 做门控式 residual refinement

这部分是整个新方案的基础实现。

### 5.2 `SparseBEV/models/sparsebev_transformer.py`

这里的改动是把 prototype 机制接到 decoder 层间，但没有改 sampling。

主要变化:

1. `SparseBEVTransformer.__init__`
  - 新增 `proto_query` 配置入口

2. `SparseBEVTransformer.forward`
  - 新增输入:
    - `prototype_bank`
    - `prototype_count`
    - `prototype_min_count`
    - `dn_pad_size`
  - 输出从原来的
    - `cls_scores, bbox_preds`
    变成
    - `cls_scores, bbox_preds, query_feats`

3. `SparseBEVTransformerDecoder`
  - 新增 `configure_proto_query`
  - 在 decoder 内部持有:
    - `difficulty_estimator`
    - `prototype_refiner`
  - 记录 `proto_use_layers`

4. decoder 主循环新增层间 refinement
  - 每层仍然先走原有 decoder block
  - 保存 `layer_query_feat`
  - 仅在设定层之间做:
    - difficulty estimation
    - prototype retrieval
    - prototype refinement

5. DN query 处理
  - refinement 只作用于 `dn_pad_size` 之后的正常 query
  - DN 部分被显式跳过

6. 新增 `all_query_feats`
  - 收集每一层 decoder 的 query feature
  - 供 head 侧做 prototype loss 和 bank update

明确未改:

- `SparseBEVSampling`
- `sampling_4d`
- `make_sample_points`
- 任何 view / frame / valid mask 逻辑

### 5.3 `SparseBEV/models/sparsebev_head.py`

这是本次改动最多的文件，主要负责:

1. 持有 prototype bank
2. 将 bank 传入 transformer
3. 收集 `all_query_feats`
4. 计算 `loss_proto`
5. 更新 prototype bank

具体变化如下。

#### A. 初始化阶段

新增:

- `proto_query` 参数
- `self.proto_enabled`
- `self.proto_use_layers`
- `self.proto_min_count`
- `self.proto_temperature`
- `self.proto_loss_weight`
- `self.prototype_bank`

同时在 `proto_enabled` 时:

- 调用 `self.transformer.decoder.configure_proto_query(...)`
- 创建 `QueryPrototypeBank`

#### B. `forward`

新增逻辑:

- 从 `self.prototype_bank` 取出:
  - `prototype_bank`
  - `prototype_count`
- 计算 `dn_pad_size`
- 调用 transformer 时把 prototype 相关信息传进去

输出变化:

- 新增 `all_query_feats`

DN 场景下:

- `all_query_feats` 会和 `all_cls_scores / all_bbox_preds` 一样，把 DN 前缀裁掉后再放进 `outs`

#### C. target 信息增强

`get_targets` 的返回值从原来只返回:

- labels / weights / targets / num_pos / num_neg

改成还返回:

- `pos_inds_list`
- `neg_inds_list`

这是为了后续从 `all_query_feats` 中抽取正样本 query。

#### D. prototype 相关辅助方法

新增方法:

- `collect_positive_queries`
- `get_layer_target_info`
- `calc_prototype_loss`
- `update_prototype_bank`

用途分别是:

- 从 query feature 中收集正样本
- 单层重新做 matching target 提取
- 计算 prototype alignment loss
- 训练时更新 bank

#### E. `loss`

新增逻辑:

- 从 `preds_dicts` 里读取 `all_query_feats`
- 调用 `calc_prototype_loss`
- 将结果写入 `loss_dict['loss_proto']`
- 调用 `update_prototype_bank`

prototype loss 的形式是:

- 取正样本 query feature
- 与当前 layer prototype 做 cosine / temperature 分类
- 对有效 prototype 类别做交叉熵

这个 loss 只在 prototype 有足够计数时生效。

### 5.4 `SparseBEV/models/__init__.py`

这里的改动很简单:

- 导出新增模块
  - `QueryPrototypeBank`
  - `QueryDifficultyEstimator`
  - `PrototypeRefiner`

没有功能性逻辑变化，只是补注册导出。

### 5.5 `SparseBEV/configs/r50_nuimg_704x256.py`

配置文件新增了:

- `proto_query_cfg`

包含参数:

- `enabled`
- `bank_momentum`
- `min_proto_count`
- `use_layers`
- `lambda_proto`
- `temperature`
- `difficulty_hidden_dim`
- `proto_hidden_dim`
- `prototype_refine`

并且把它同时接到了:

- `SparseBEVHead`
- `SparseBEVTransformer`

说明:

- 当前只改了基础配置 `r50_nuimg_704x256.py`
- 其他配置文件通过 `_base_` 继承时，会默认带上这套开关
- 本次没有单独为大模型配置写额外覆盖项

## 6. 文档改动

### 6.1 `SparseBEV/docs/proto_temporal_sparsebev_design.md`

这份文档在本次工作中被整理成了“无采样改动版”的设计文档。

文档当前描述与代码保持一致的地方:

- 方案只走 `query / prototype / loss` 路线
- 不改 `sparsebev_sampling.py`
- 使用:
  - `Query Prototype Bank`
  - `Query Difficulty Estimator`
  - `Prototype-guided Query Refinement`
  - `Prototype Alignment Loss`

文档中也明确写了当前实现没有额外的 epoch 级 warmup hook，而是用 `min_proto_count` 作为冷启动保护。

### 6.2 `SparseBEV/docs/changes_vs_initial_branch.md`

这是本文件本身，用来记录与最开始分支状态的差异。

## 7. 刻意没有做的改动

为了满足“不要做任何视角采样相关工作”的要求，这次刻意没有做下面这些事:

1. 没有改 `SparseBEV/models/sparsebev_sampling.py`
2. 没有改 `sampling_4d`
3. 没有增加 `valid_mask` / `valid_point_ratio` / `frame_weight`
4. 没有增加 view-level 或 frame-level reweighting
5. 没有改任何 CUDA sampling wrapper
6. 没有引入 2D crop prototype、视觉模板库或语言原型

也就是说，这次实现并不是 `CorrBEV` 的 correlation 版本，而是一个纯 query-space 的 prototype 版本。

## 8. 验证情况

本地已经完成的检查:

- `python -m compileall .\\SparseBEV\\models .\\SparseBEV\\configs`
- `python -m py_compile .\\SparseBEV\\models\\proto_query.py .\\SparseBEV\\models\\sparsebev_transformer.py .\\SparseBEV\\models\\sparsebev_head.py .\\SparseBEV\\configs\\r50_nuimg_704x256.py`

结果:

- 语法级检查通过

当前没法完成的检查:

- 真正的前向 / loss 运行检查
- 单 batch 训练 smoke test

原因:

- 当前 shell 里的 `python` 环境没有安装 `torch`

因此，当前能确认的是:

- 代码结构已接好
- 语法通过
- 没有改动采样文件

但还不能替代一次真实训练环境下的 forward 检查。

## 9. 一句话总结

相对最开始的 `main` 分支状态，这次改动本质上是:

- 在不碰视角采样实现的前提下，
- 给 `SparseBEV` 加上了一套 query-space prototype 机制，
- 包括 bank、difficulty-aware refinement 和 `loss_proto`，
- 并把这套机制接入了默认配置与设计文档。
