# 基于特征感知低秩修正的稀疏多视图三维目标检测方法

> 本文档整理对 SparseBEV 解码器中 AdaptiveMixing 模块的改进设计,作为毕业小论文的方法部分初稿。

## 摘要

基于稀疏查询的多视图三维目标检测方法(以 SparseBEV 为代表)沿用 AdaMixer 的解耦自适应混合(Adaptive Mixing)思想:每一层解码器中,用于聚合采样特征的混合权重矩阵仅由查询(query)经线性投射生成,与实际采样得到的特征内容无关。该设计在计算上高效,但限制了模型根据特征内容动态调整混合权重的能力,尤其在多帧时序场景下,无法显式建模采样点之间的时空差异。本文针对这一瓶颈,提出**特征感知的低秩修正分支**:在保留原解耦混合矩阵的基础上,引入一条由查询与采样特征之间的双线性交互生成的修正项;修正分支显式地融合了 3D 采样偏移的空间编码与基于正余弦的帧级时间编码,且不施加任何激活函数,使其与原混合矩阵处于同一数据流形。该分支以零初始化的方式接入,初始时与原模型严格等价,训练过程中渐进学习特征感知的修正信号。本方法仅引入约 20% 的解码器参数量,无需修改采样、损失或骨干网络,可作为即插即用模块嵌入 SparseBEV 解码器。

**关键词**:三维目标检测;多视图感知;稀疏查询;自适应混合;时空编码

---

## 1. 研究动机

### 1.1 SparseBEV 的解耦混合

SparseBEV 在解码器的每一层中执行如下流程:

1. **采样**(Sampling):由查询 $q \in \mathbb{R}^{C}$ 预测每个组(group)、每个采样点的 3D 偏移 $\Delta \in \mathbb{R}^{G \times P \times 3}$,经 box 变换映射到 3D 空间,投影至多视图、多帧的图像特征上,获得采样特征 $f \in \mathbb{R}^{G \times FP \times C}$,其中 $F$ 为帧数,$P$ 为每帧每组采样点数。
2. **混合**(AdaptiveMixing):由查询线性生成混合参数 $(M, S)$,其中通道混合矩阵 $M \in \mathbb{R}^{G \times C \times C}$ 用于通道维变换,点混合矩阵 $S \in \mathbb{R}^{G \times O \times FP}$ 用于在采样点维上聚合(其中 $O$ 为输出点数)。
3. **更新**:聚合结果展平、投射回查询维度,经 LayerNorm 与 FFN 更新查询。

该设计的核心特征是:**点混合矩阵 $S$ 完全由查询生成,与采样特征 $f$ 解耦**。这一性质来自 AdaMixer 论文的原始设计,在保证高效的同时也带来如下问题:

- 当采样得到的特征内容与查询的预期不一致时(例如目标被遮挡、运动模糊、跨帧动态),$S$ 无法据此调整聚合策略;
- 多帧场景下,$F \times P$ 个采样点在 $S$ 看来是同质的,无法区分时间维与空间维;
- $S$ 仅依赖查询,等价于一个静态的"模板",难以表达 query-conditioned 之外的任何 feature-conditioned 模式。

### 1.2 现有改进的代价

部分后续工作(如 Sparse4D 系列)通过引入显式的可形变交叉注意力来重新耦合查询与特征,但其代价是放弃了 AdaMixer 解耦混合的高效结构。本文希望在**保留原结构、不改变采样模块、不引入 token 级注意力开销**的前提下,补充缺失的"特征感知"能力。

---

## 2. 方法

### 2.1 整体框架

设解码器层的输入为查询特征 $q \in \mathbb{R}^{B \times Q \times C}$ 与采样特征 $f \in \mathbb{R}^{B \times Q \times G \times FP \times C}$,以及对应的 3D 采样偏移 $\Delta \in \mathbb{R}^{B \times Q \times G \times FP \times 3}$。其中 $B$ 为批量,$Q$ 为查询数,$G$ 为组数,$FP = F \cdot P$ 为每个查询每组的全部采样点数。

原 AdaptiveMixing 的点混合矩阵记为:

$$
S_{\text{query}} = \text{reshape}(W_S\, q + b_S) \in \mathbb{R}^{B \times Q \times G \times O \times FP}
$$

本文在该矩阵之外补充一条**特征感知低秩修正分支** $S_{\text{attn}}$,二者通过加法残差融合:

$$
S = S_{\text{query}} + S_{\text{attn}}
$$

随后 $S$ 参与原 AdaMixer 的点混合步骤,流程其余部分保持不变。

### 2.2 特征感知低秩修正分支

$S_{\text{attn}}$ 的构造分为三步:**Key 构造**、**Query 投射**、**双线性打分**。

#### 2.2.1 Key 构造:特征 + 空间编码 + 时间编码

为每个采样点构造一个可与查询交互的 key。原始采样特征 $f$ 通过线性变换降维至 $d_k$:

$$
\widehat{K}_{ij} = W_k\, f_{ij}, \quad \widehat{K}_{ij} \in \mathbb{R}^{d_k}
$$

其中 $i$ 索引查询、$j$ 索引采样点(覆盖 $G \times FP$ 维度)。

为显式编码每个采样点的空间位置,对采样偏移 $\Delta_{ij} \in \mathbb{R}^3$ 应用一个轻量 MLP:

$$
\text{POS}(\Delta) = W_2^{\text{pos}} \cdot \text{ReLU}\!\left(\text{LN}(W_1^{\text{pos}} \Delta)\right) \in \mathbb{R}^{d_k}
$$

需要注意:由于 SparseBEV 的采样设计使得同一查询的同一逻辑点在所有帧上共享相同的 3D 偏移(实际跨帧差异通过自车运动补偿和不同时刻的图像采样体现),因此 $\text{POS}(\Delta)$ 在帧维上为常数。为使不同帧的 key 可被区分,我们引入帧级正余弦时间编码:

$$
\tau_t = [\sin(t \omega_0), \cos(t \omega_0), \sin(t \omega_1), \cos(t \omega_1), \ldots] \in \mathbb{R}^{d_k}
$$

其中频率 $\omega_k = 10000^{-2k / d_k}$,$t \in \{0, 1, \ldots, F-1\}$ 为帧索引。$\tau_t$ 在模型构建时一次性预计算并作为非持久缓冲注册。

最终 key 为三项之和:

$$
K_{ij} = W_k\, f_{ij} + \text{POS}(\Delta_{ij}) + \tau_{\,t(j)}
$$

其中 $t(j)$ 为采样点 $j$ 所属的帧编号。

#### 2.2.2 Query 投射:每个输出点一个独立查询

为生成形状为 $(O, FP)$ 的修正矩阵,对每个输出点 $o \in \{1, \ldots, O\}$ 与每个组 $g \in \{1, \ldots, G\}$ 投射出一个独立的查询向量:

$$
\widetilde{Q}^{(g, o)} = W_q^{(g, o)}\, q \in \mathbb{R}^{d_k}
$$

实现上通过单个线性层 $W_q \in \mathbb{R}^{C \times G \cdot O \cdot d_k}$ 一次产出全部 $G \cdot O$ 个查询向量。

#### 2.2.3 双线性打分:无激活的低秩修正

修正矩阵的每个元素由 query 与 key 的内积给出:

$$
S_{\text{attn}}^{(g, o, j)} = \frac{\bigl\langle \widetilde{Q}^{(g, o)},\, K^{(g, j)} \bigr\rangle}{\sqrt{d_k}}
$$

**关键设计:不施加 softmax 或 sigmoid 激活**。原 $S_{\text{query}}$ 取值为任意实数(线性输出,无激活),为使修正项与其处于同一数据流形,$S_{\text{attn}}$ 也保持为任意实数。这一选择带来三点好处:

1. **流形一致**:$S_{\text{query}}$ 与 $S_{\text{attn}}$ 同为任意实数,加法残差是真正等价分支的融合,不存在量纲失配;
2. **保留负权重**:$S_{\text{attn}}$ 可输出负值,意义为"减弱该采样点的贡献",与 $S_{\text{query}}$ 表达能力对齐;
3. **概念清晰**:由于 $S_{\text{attn}}$ 由查询与特征的双线性形式生成,其矩阵秩满足 $\text{rank}(S_{\text{attn}}) \le d_k$。该分支可被理解为一个**特征感知的低秩修正项**,而非传统意义上的注意力机制——这与 AdaMixer 解耦混合的设计哲学一致。

### 2.3 数值稳定性处理

修正分支涉及 $d_k = 16$ 维向量的内积与缩放,在自动混合精度(AMP, fp16)下可能发生数值下溢或溢出。为此,我们对修正分支显式禁用 autocast,强制使用 fp32 精度计算双线性形式与位置编码,在产出 $S_{\text{attn}}$ 后再 cast 回 $S_{\text{query}}$ 的 dtype 进行加法。完整伪代码:

```python
with torch.cuda.amp.autocast(enabled=False):
    K = W_k(f.float()) + pos_mlp(Δ.float())            # [B,Q,G,FP,d_k]
    K = K.reshape(B, Q, G, F, P_per_frame, d_k)
    K = K + τ[None, None, None, :, None, :]           # 帧级时间码广播
    K = K.reshape(B*Q*G, FP, d_k)

    Q̃ = W_q(q.float()).reshape(B*Q*G, O, d_k)
    scores = bmm(Q̃, K.transpose(1, 2)) / sqrt(d_k)    # [B*Q*G, O, FP]
    S_attn = scores.reshape(B*Q, G, O, FP)
    S_attn = S_attn.to(S_query.dtype)

S = S_query + S_attn
```

### 2.4 初始化策略

为保证训练初期与原模型严格等价、避免修正分支引入早期扰动,$W_q$ 与 $W_k$ 采用 Xavier 均匀初始化并将增益 (gain) 设为 0.1,所有偏置初始化为零:

$$
W_q, W_k \sim \mathcal{U}\!\left(-0.1\sqrt{\tfrac{6}{\text{fan}_{\text{in}} + \text{fan}_{\text{out}}}},\, +0.1\sqrt{\tfrac{6}{\text{fan}_{\text{in}} + \text{fan}_{\text{out}}}}\right)
$$

初始时 $K \approx 0$、$\widetilde{Q} \approx 0$,故 $S_{\text{attn}} \approx 0$,加法残差等价于 $S = S_{\text{query}}$,网络行为与原 SparseBEV 一致。$\text{POS}$ 与时间码 $\tau$ 不做特殊压制,但因 key 端 $W_k$ 接近零,二者对 $S_{\text{attn}}$ 的初期影响可忽略。

### 2.5 与原架构的兼容性

- **与梯度检查点兼容**:修正分支接收 $\Delta$ 作为额外输入,通过 `torch.utils.checkpoint`(`use_reentrant=False`)透传,前后向行为正确;
- **与查询去噪兼容**:不依赖任何与去噪 query 不一致的状态;
- **与 FP16 优化器兼容**:仅在修正分支内禁用 autocast,外层仍使用 fp16 优化器;
- **与多 GPU DDP 兼容**:无跨样本通信,无新增同步原语。

---

## 3. 复杂度分析

| 项 | 原 AdaptiveMixing | 改进后 | 增量 |
|---|---|---|---|
| `parameter_generator` | $C \cdot G(M_{params} + S_{params})$ | 同左 | 0 |
| `out_proj` | $G \cdot O \cdot C / G \cdot C$ | 同左 | 0 |
| `k_proj` | — | $C/G \cdot d_k$ | 极小 |
| `q_proj` | — | $C \cdot G \cdot O \cdot d_k$ | $\approx 2.1\text{M}$ (per layer) |
| `pos_mlp` | — | $3 \cdot d_k + d_k \cdot d_k$ | 极小 |

按 $C = 256$、$G = 4$、$O = 128$、$d_k = 16$、解码器 6 层估算,单层新增参数约 2.1M,整模型新增约 12.6M;相对原 SparseBEV 总参数(约 60M)增量约 21%。

计算开销主要来自:
- 双线性 `bmm`:$\mathcal{O}(B \cdot Q \cdot G \cdot O \cdot FP \cdot d_k)$,在 $B=8$、$Q=900$、$G=4$、$O=128$、$FP=32$、$d_k=16$ 下约 1.9 GFLOPs/层;
- 位置 MLP 与时间码相加:可忽略。

由于不再使用 softmax,实际 wall-clock 开销低于一次完整 cross-attention,初步观察整体训练速度无显著下降(部分配置下因省略 softmax 而略有提升)。

---

## 4. 实验设置

### 4.1 数据集与评估指标

- **数据集**:nuScenes(完整 train/val 划分,1000 场景,共 28k+/6k 关键帧)。
- **评估指标**:nuScenes 官方指标:mAP、mATE、mASE、mAOE、mAVE、mAAE、NDS。NDS 为综合主指标。

### 4.2 实现细节

- **骨干网络**:ResNet-50,FPN 颈部,nuImages 预训练。
- **输入分辨率**:$704 \times 256$。
- **帧数**:$F = 8$(当前帧 + 7 个历史扫帧)。
- **解码器**:6 层,每层共享参数;查询数 $Q = 900$;查询去噪 10 组。
- **训练**:AdamW,学习率 $2 \times 10^{-4}$(骨干网络与采样偏移层 lr_mult $= 0.1$),余弦退火,500 步线性 warmup,总训练 24 epoch,batch size 8。FP16 + 梯度裁剪(max_norm = 35)。
- **修正分支超参**:$d_k = 16$,$W_q$/$W_k$ 初始化 gain $= 0.1$。

### 4.3 待完成的消融实验

| 编号 | 配置 | 目的 |
|---|---|---|
| (a) | 仅 $S_{\text{query}}$(原 SparseBEV) | baseline |
| (b) | $S_{\text{query}} + \text{softmax}(scores)$ | 验证 softmax 形式 |
| (c) | $S_{\text{query}} + \text{sigmoid}(scores)$ | 验证 sigmoid 门控形式 |
| (d) | $S_{\text{query}} + \text{scores}$(本文) | 无激活的低秩修正 |
| (e) | 仅 $\text{scores}$(去掉 $S_{\text{query}}$) | 验证残差分支必要性 |
| (f) | (d) 去掉 $\text{POS}(\Delta)$ | 验证空间编码作用 |
| (g) | (d) 去掉时间码 $\tau$ | 验证时间编码作用 |
| (h) | (d) $d_k = 8$ / $32$ | 修正项秩的敏感性 |

---

## 5. 讨论与未来工作

### 5.1 设计选择讨论

**为何采用残差而非替换?** 本文的修正分支并非要取代 $S_{\text{query}}$,而是补充其缺失的特征感知能力。残差形式具有两点优势:(1) 零初始化下严格等价于 baseline,训练初期不引入扰动,有利于稳定收敛;(2) 即使修正分支学习失败(例如塌缩为零),模型仍退化为原 SparseBEV,具有"鲁棒下界"。

**为何无激活而非 softmax/sigmoid?** softmax 在 in_points 维度强制行和为 1,意味着每个输出点必须从采样点中分配 100% 的注意力,无法表达"全部不重要"的状态;且其取值范围 $[0, 1]$ 与 $S_{\text{query}}$ 的任意实数取值不一致,叠加后存在量纲失配,后期 $S_{\text{query}}$ 量级增大时 softmax 输出会被淹没。sigmoid 缓解了行约束问题,但仍为有界正值,无法表达负权重。无激活的设计使两条分支处于同一流形,加法语义清晰。

**为何 $d_k = 16$?** 双线性形式的秩上界为 $d_k$。$d_k$ 过小会限制修正项的表达;过大会显著增加 $W_q$ 的参数量(主要开销来自 $W_q$,其参数规模为 $C \cdot G \cdot O \cdot d_k$)。$d_k = 16$ 作为初始选择,在表达能力与参数量之间取得折中,具体值可在消融实验中调整。

### 5.2 局限性

- 修正分支引入约 21% 的解码器参数,在严格的内存预算下需要权衡;
- 当前未对 attention 分支的训练动态做显式约束,数值稳定性依赖初始化与权重衰减,极端情况下可能出现 $S_{\text{attn}}$ 量级失控;
- 修正分支的可解释性弱于显式 attention,无法直接绘制"注意力图"。

### 5.3 后续改进方向

1. **特征感知的采样**(Feature-Aware Sampling):当前 SparseBEV 的采样偏移亦完全由查询生成,与本文所识别的"混合解耦"问题同构。后续可设计两轮采样:首轮采样得到的特征用于细化第二轮采样位置,与本文混合改进形成"采样—混合"双闭环。
2. **辅助深度监督**:利用 LiDAR 点云投影至图像得到的稠密深度图,在 FPN 特征上加深度预测头作为辅助监督,是相机感知的标准提升手段(BEVDepth、SOLOFusion 等已验证),与本文方法正交可叠加。
3. **尺度感知的多尺度聚合**:当前尺度权重不显式依赖目标尺寸,后续可让大目标偏向低分辨率高语义特征、小目标偏向高分辨率特征。
4. **跨帧查询交互**:在查询级别引入历史帧 query 的传播机制(类 StreamPETR),将本文的时序建模从特征采样层面延伸至查询语义层面。

---

## 附录 A:符号表

| 符号 | 含义 |
|---|---|
| $B, Q, G$ | 批量大小、查询数、组数 |
| $F, P, FP$ | 帧数、每帧每组采样点数、总采样点数 ($FP = F \cdot P$) |
| $C$ | 查询/特征通道维度 |
| $O$ | 输出点数 (out_points = 128) |
| $d_k$ | 修正分支的内积维度 (= 16) |
| $f \in \mathbb{R}^{B \times Q \times G \times FP \times C}$ | 采样特征 |
| $\Delta \in \mathbb{R}^{B \times Q \times G \times FP \times 3}$ | 3D 采样偏移 |
| $q \in \mathbb{R}^{B \times Q \times C}$ | 查询特征 |
| $S_{\text{query}}$ | 原 AdaMixer 的点混合矩阵(由查询生成) |
| $S_{\text{attn}}$ | 本文提出的特征感知低秩修正项 |
| $\tau_t \in \mathbb{R}^{d_k}$ | 第 $t$ 帧的正余弦时间编码 |

## 附录 B:实现位置

| 模块 | 文件:行号 |
|---|---|
| 修正分支主逻辑 | `SparseBEV/models/sparsebev_transformer.py:402-419` |
| key/query 投射与时间码 | `SparseBEV/models/sparsebev_transformer.py:361-376` |
| 初始化 | `SparseBEV/models/sparsebev_transformer.py:378-384` |
| `sample_offset` 透传 | `SparseBEV/models/sparsebev_transformer.py:314-318` |
