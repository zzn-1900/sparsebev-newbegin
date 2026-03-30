# SparseBEV-Proto: 在线多模态原型增强的稀疏3D目标检测

## 1. 研究动机

### 1.1 CorrBEV的核心思想与局限

CorrBEV (CVPR 2025) 提出用多模态原型（视觉+语言）与BEV特征做相关学习，显著提升了遮挡场景下的检测性能。但其原型构建方式存在根本性限制：

- **视觉原型离线构建**：需要预先从训练集裁剪2D目标模板，用DeViT编码，流程割裂
- **语言原型冻结**：用预训练BERT编码类别名称，不参与端到端优化
- **2D空间操作**：correlation在BEV/图像特征图上做，引入大量中间张量，显存开销高

### 1.2 我们的改进目标

将CorrBEV的原型相关学习思想迁移到SparseBEV的稀疏query架构中，实现：
- 原型完全在线生成，无需离线预处理
- 在3D query空间而非2D特征图上做原型增强，零额外显存
- 端到端训练，所有组件联合优化

---

## 2. 方法概述

### 2.1 整体架构

在SparseBEV的每个Decoder Layer中，在Self-Attention之后、Spatio-temporal Sampling之前，插入一个**Prototype Enhancement**模块。query先从原型中获取类别先验，再带着这个先验去图像上采样。

```
原始 SparseBEV Decoder Layer:
  Position Encoding → Self-Attention → Sampling → Mixing → FFN → Heads

改进后:
  Position Encoding → Self-Attention → ★Prototype Enhancement → Sampling → Mixing → FFN → Heads
                                        ↑
                              OnlinePrototypeGenerator
                              (视觉原型 + 语言原型 → 融合原型)
```

### 2.2 与CorrBEV的对比

| 设计维度 | CorrBEV | 本方法 (SparseBEV-Proto) |
|---------|---------|------------------------|
| 视觉原型来源 | 离线裁剪2D模板 + DeViT编码 | 在线EMA积累匹配到GT的query特征 |
| 语言原型来源 | 预训练BERT编码类别名 (冻结) | 可学习nn.Embedding (端到端) |
| 原型作用方式 | 2D特征图上depth-wise correlation | 3D query空间cross-attention |
| 显存开销 | 大 (需存储correlation特征图) | 极小 (仅40×256的原型矩阵) |
| 子原型分化 | 按visibility标签分组 (需标注) | 自动聚类 (FPS初始化 + 频率感知分配) |
| 训练流程 | 两阶段 (先构建原型再训练) | 单阶段端到端 |

---

## 3. 核心模块详解

### 3.1 OnlinePrototypeGenerator

#### 3.1.1 双模态原型结构

```
语言原型 Pl: nn.Embedding(N=10, D=256)     ← 可学习，梯度直接回传
  │
  │ expand: (10, 256) → (40, 256)           ← 每类4个子原型共享同一语言原型
  │
  ├──cat──┐
  │       ▼
  │   [vis; lang]  (40, 512)
  │       │
  │   fusion_proj:
  │     Linear(512 → 256)
  │     LayerNorm(256)
  │     ReLU
  │       │
  │       ▼
  │   融合原型 P  (40, 256)                  ← 同时编码"长什么样"和"是什么类"
  │
视觉原型 Pv: register_buffer(NK=40, D=256)  ← 不可学习，EMA维护
```

**设计意图**：
- 语言原型提供类别语义锚点（"这是car的概念"），通过梯度学习
- 视觉原型提供外观统计（"car通常长这样"），通过EMA从训练数据中积累
- 融合投影让两种信息互补：即使视觉原型在训练早期还是零向量，语言原型也能提供有意义的类别先验

#### 3.1.2 Prototype Enhancement (Cross-Attention)

```
输入: query_feat (B, Q=900, D=256)

                    ┌─────────────────────────────────┐
                    │  nn.MultiheadAttention           │
                    │  embed_dim=256, num_heads=8      │
                    │  d_k = 256/8 = 32 per head       │
                    │                                   │
  query_feat ──Q──→ │                                   │ ──→ attn_out (B, 900, 256)
                    │  attn_weight (B, 8, 900, 40)     │
  prototypes ──K──→ │  = softmax(Q·K^T / √32)          │
  prototypes ──V──→ │                                   │
  (B, 40, 256)      └─────────────────────────────────┘
                                    │
                                    ▼
                    LayerNorm(query_feat + attn_out)
                                    │
                                    ▼
                    enhanced query_feat (B, 900, 256)
```

**计算量分析**：
- attention矩阵: (B, 8, 900, 40) — 仅900×40=36K个元素/head，远小于self-attention的900×900=810K
- 总FLOPs: ~2×B×900×40×256 ≈ 18.4M×B，相比sampling_4d的数十亿FLOPs可忽略
- 额外显存: 仅原型expand的 (B, 40, 256) ≈ 40KB/sample

**插入位置的选择**：
放在Self-Attention之后、Sampling之前。这样query先通过self-attention建立query间的空间关系，再从原型获取类别先验，最后带着"我可能是car"的先验去图像上采样——采样偏移量会被类别先验引导，更精准地采到目标区域。

#### 3.1.3 对比语义对齐损失

```
输入: pos_feat (Npos, 256) — 匹配到GT的正样本query特征 (detached)
      pos_label (Npos,)    — 对应GT类别

                pos_feat                    lang_proto.weight
               (Npos, 256)                    (10, 256)
                    │                             │
              L2 normalize                  L2 normalize
                    │                             │
                    └──── matmul ────┘
                           │
                    logits (Npos, 10)
                    × exp(logit_scale)      ← 可学习温度, 初始化=ln(100)
                           │
                    sigmoid_focal_loss
                    (α=0.25, γ=2.0)
                           │
                    loss × contrastive_weight(0.5)
```

**作用**：拉近同类query与语言原型的距离，推远异类。这迫使query特征空间与语言原型对齐，间接提升cross-attention中的注意力质量——query会更准确地attend到自己类别对应的原型。

---

### 3.2 视觉原型在线更新机制

这是本方法最核心的创新。CorrBEV的视觉原型是离线构建的静态向量，我们设计了一套完整的在线生命周期管理。

#### 3.2.1 三阶段生命周期

```
阶段1: 冷启动                    阶段2: 日常更新                 阶段3: 周期重校准
(ring_total < 128)              (cls_initialized = True)        (每500步)

攒样本到环形缓冲区               频率感知最近邻EMA               用最新缓冲区样本
     │                               │                         重新FPS
     ▼                               ▼                              │
攒够128个后                     自适应momentum                      ▼
FPS选4个最分散的                 频率惩罚防坍缩                 算出理想位置
初始化子原型                          │                         EMA软靠拢
     │                               ▼                         (m=0.8)
     ▼                          vis_proto更新
cls_initialized = True
```

#### 3.2.2 创新点1: FPS初始化 (替代顺序赋值)

**问题**：如果按样本到达顺序依次填入4个子原型，前4个样本可能都是同一种外观（如正面视角的car），导致子原型缺乏多样性。

**方案**：先攒128个样本到环形缓冲区，然后用最远点采样(FPS)选出4个在特征空间中最分散的点。

```
特征空间示意 (128个car样本):

    ●●●●  正面视角簇          ○○○  远距离簇
    ●●●                        ○○

              ▲▲▲▲  侧面视角簇
              ▲▲▲

    ■■■  遮挡簇
     ■■

FPS选出的4个初始子原型:  ● ○ ▲ ■  (每个簇各一个代表)
顺序赋值可能选出的:       ● ● ● ●  (全来自正面视角)
```

FPS算法复杂度 O(M×K) = O(128×4) = O(512)，可忽略。

#### 3.2.3 创新点2: 频率感知的最近邻分配 (防止子原型坍缩)

**问题**：纯最近邻分配天然"富者愈富"——位于特征空间中心的子原型会吸引大部分样本，其他子原型饿死，最终4个坍缩成1-2个有效的。

**方案**：在距离上加频率惩罚，更新越多的子原型距离被放大。

```
dist_adjusted[j] = dist[j] × (1 + log(1 + count[j]) / log(1 + Σcount))
                              ├─────────── freq_penalty ───────────┤
                              ∈ [0, 1]
                              count最多的 → 接近1 → 距离放大2倍
                              count最少的 → 接近0 → 距离不变

示例: 4个子原型更新次数 = [5000, 100, 100, 50]
  新样本离4个子原型的原始距离 = [0.3, 0.5, 0.6, 0.8]
  频率惩罚后调整距离           ≈ [0.51, 0.63, 0.75, 0.93]
  → 原本分配给子原型0，现在分配给子原型1
```

#### 3.2.4 创新点3: 自适应Momentum (解决类别不均衡)

**问题**：nuScenes类别分布极度不均衡。固定momentum=0.999意味着每次更新只吸收0.1%新信息——car更新几万次没问题，但construction_vehicle可能总共才更新几百次。

**方案**：momentum根据每个子原型的累计更新次数自适应。

```
m(count) = 0.999 - 0.499 × exp(-count / 200)

count=1    → m=0.50  → 吸收50%新信息 (快速初始化)
count=50   → m=0.89  → 吸收11%
count=200  → m=0.95  → 吸收5%
count=500  → m=0.98  → 吸收2%
count=1000 → m=0.997 → 吸收0.3% (趋近上限)

效果:
- car子原型: 更新了20000次 → m≈0.999 → 高度稳定，不被噪声干扰
- construction_vehicle子原型: 更新了200次 → m≈0.95 → 每次更新吸收5%，学习效率高20倍
```

每个子原型独立计数、独立momentum，同一类别的4个子原型也可以有不同的momentum。

#### 3.2.5 创新点4: 周期重校准 (持续优化子原型分布)

**问题**：训练早期decoder还没学好，FPS初始化用的是低质量特征。随着训练推进特征质量提升，但子原型分布已经被早期噪声固化。

**方案**：每500步用环形缓冲区中的最新128个样本重新做FPS，算出"理想子原型位置"，然后用EMA软靠拢。

```
每500步:
  1. 取ring_buffer中最新128个样本
  2. FPS选4个理想位置
  3. 贪心匹配: 当前子原型 ↔ 理想位置 (最小距离配对)
  4. 软靠拢: proto[j] = 0.8 × proto[j] + 0.2 × ideal[matched_k]

为什么不硬重置:
  - 硬重置会导致correlation特征突变 → 训练loss跳变
  - 软靠拢(m=0.8)每次移动20%，约5次重校准(2500步)后基本收敛到新位置
  - 渐进式调整，训练曲线平滑
```

环形缓冲区的关键作用：它始终保存最近128个样本，所以重校准用的是当前训练阶段的高质量特征，而非训练初期的噪声特征。

---

### 3.3 环形缓冲区设计

```
ring_buffer: (10, 128, 256)    ← 10类 × 128样本 × 256维
ring_ptr:    (10,)              ← 每类的写入指针
ring_total:  (10,)              ← 每类总共收到过多少样本

写入逻辑 (每个正样本都写):
  ring_buffer[cls_id, ptr] = feat
  ring_ptr[cls_id] = (ptr + 1) % 128    ← 循环覆盖最旧的
  ring_total[cls_id] += 1

读取逻辑:
  n_valid = min(ring_total[cls_id], 128)
  valid_feats = ring_buffer[cls_id, :n_valid]
```

**为什么用环形缓冲区而非固定缓冲区**：
- 固定缓冲区攒满就停，后续样本被丢弃，缓冲区内容逐渐过时
- 环形缓冲区永远保存最新的128个样本，重校准时用的是当前训练阶段的特征
- 显存开销固定: 10×128×256×4bytes = 1.25MB，可忽略

---

## 4. 损失函数

```
L_total = L_det + L_dn + L_contrastive

L_det = Σ_{l=1}^{6} (FocalLoss_cls + L1Loss_bbox)        ← 6层decoder每层都算
L_dn  = Σ_{l=1}^{6} (FocalLoss_cls_dn + L1Loss_bbox_dn)  ← query denoising
L_contrastive = 0.5 × sigmoid_focal_loss(pos_query · lang_proto)  ← 新增
```

对比损失只在正样本上计算（匈牙利匹配后），不增加负样本的计算量。

---

## 5. 推理流程

推理时：
- 视觉原型固定（来自训练积累的EMA值），不更新
- 语言原型固定（训练好的Embedding权重）
- 环形缓冲区不写入
- 对比损失不计算
- enhance_query正常执行（cross-attention到融合原型）

推理额外开销：仅一次cross-attention (Q=900, KV=40)，约0.1ms。

---

## 6. 创新点总结

| # | 创新点 | 解决的问题 | 技术手段 |
|---|--------|-----------|---------|
| 1 | 在线视觉原型生成 | CorrBEV需要离线构建原型 | EMA从匹配正样本积累 + 环形缓冲区 |
| 2 | 3D query空间原型增强 | 2D correlation显存开销大 | Cross-attention替代depth-wise correlation |
| 3 | 可学习语言原型 | BERT编码冻结不可优化 | nn.Embedding端到端训练 |
| 4 | FPS子原型初始化 | 顺序赋值缺乏多样性 | 最远点采样保证初始分散性 |
| 5 | 频率感知分配 | 最近邻导致子原型坍缩 | 距离加频率惩罚，均衡分配 |
| 6 | 自适应Momentum | 类别不均衡下稀有类学不好 | momentum随更新次数自适应 |
| 7 | 周期重校准 | 早期噪声初始化固化 | 定期FPS+EMA软靠拢，持续优化 |
| 8 | 对比语义对齐 | query特征与类别语义脱节 | query-语言原型focal loss |

---

## 7. 文件结构

```
SparseBEV/
├── models/
│   ├── corrbev_prototype.py          ← 新增: OnlinePrototypeGenerator
│   │   ├── 双模态原型 (视觉EMA + 语言Embedding)
│   │   ├── 融合投影 (Linear + LN + ReLU)
│   │   ├── Cross-Attention增强 (nn.MultiheadAttention)
│   │   ├── 对比损失 (sigmoid focal loss)
│   │   ├── FPS初始化 + 频率感知EMA + 周期重校准
│   │   └── 环形缓冲区管理
│   │
│   ├── sparsebev_head.py             ← 修改: 集成入口
│   │   ├── __init__: 创建prototype_gen
│   │   ├── forward: 传prototype_gen给transformer
│   │   └── loss: 对比损失 + EMA更新触发
│   │
│   └── sparsebev_transformer.py      ← 修改: 传递prototype_gen
│       ├── Transformer.forward: 透传prototype_gen
│       ├── Decoder.forward: 透传 + 缓存last_query_feat
│       └── DecoderLayer.forward: 调用enhance_query
│
└── configs/
    └── r50_nuimg_704x256.py          ← 修改: 添加corrbev配置
