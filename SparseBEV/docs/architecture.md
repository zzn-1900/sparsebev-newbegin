# SparseBEV + Online Prototype 架构文档

## 整体数据流

```
Image [B, 6, 3, H_img, W_img]
  │
  ▼
ResNet50 + FPN
  │
  ▼
mlvl_feats: list of [B, TN, 256, H_l, W_l]    ← 4层FPN, TN=T×6(多帧多视角)
  │
  ▼
SparseBEVHead.forward()
  ├── init_query_bbox [Q, 10]                    ← Q=900, 10=(cx,cy,cz,w,l,h,sin,cos,vx,vy)
  ├── prepare_for_dn_input()                     ← query denoising
  │     → query_bbox  [B, Q+dn_pad, 10]
  │     → query_feat  [B, Q+dn_pad, 256]
  │     → attn_mask   [Q+dn_pad, Q+dn_pad]
  │
  ▼
SparseBEVTransformer.forward(query_bbox, query_feat, mlvl_feats, prototype_gen)
  │
  ▼
SparseBEVTransformerDecoder.forward()
  ├── 预处理: 计算time_diff [B, F], lidar2img [B, N, 4, 4]
  ├── 重组FPN特征: [B, TN, 256, H, W] → [B*T*G, C, N, H, W]  (G=4组, C=64/组)
  │
  ├── ×6 DecoderLayer (共享权重):
  │     │
  │     ├── 1. Position Encoding
  │     │     query_bbox[..., :3] → Linear(3→256) → LN → ReLU → Linear(256→256) → LN → ReLU
  │     │     query_feat += query_pos                                    [B, Q, 256]
  │     │
  │     ├── 2. Scale-adaptive Self-Attention (SASA)
  │     │     ├── calc_bbox_dists: query_bbox → 解码中心坐标 → 两两L2距离 → 取负
  │     │     │     dist [B, Q, Q]
  │     │     ├── gen_tau: Linear(256→8)
  │     │     │     tau [B, Q, 8] → permute → [B, 8, Q]
  │     │     ├── attn_mask = dist[:, None] * tau[..., None]             [B, 8, Q, Q]
  │     │     ├── 合并dn_mask后 flatten → [B×8, Q, Q]
  │     │     └── MultiheadAttention(Q=K=V=query_feat, attn_mask)        [B, Q, 256]
  │     │     query_feat = LayerNorm(attn_out)
  │     │
  │     ├── 3. ★ Prototype Enhancement (新增)
  │     │     │
  │     │     │  OnlinePrototypeGenerator.enhance_query(query_feat)
  │     │     │
  │     │     ├── get_fused_prototypes():
  │     │     │     vis_proto                [NK=40, 256]   ← buffer, EMA维护
  │     │     │     lang_proto.weight         [N=10, 256]   ← nn.Embedding, 梯度更新
  │     │     │     lang_expanded             [40, 256]     ← 每类4个子原型共享语言原型
  │     │     │     cat([vis, lang])           [40, 512]
  │     │     │     fusion_proj:
  │     │     │       Linear(512→256) → LN(256) → ReLU
  │     │     │     → fused_prototypes        [40, 256]
  │     │     │
  │     │     ├── expand to batch:            [B, 40, 256]
  │     │     │
  │     │     ├── nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1):
  │     │     │     Q = query_feat            [B, Q, 256]
  │     │     │     K = V = fused_prototypes  [B, 40, 256]
  │     │     │     ────────────────────────────────────
  │     │     │     per head: d_k = 256/8 = 32
  │     │     │     Q_h [B, 8, Q, 32] × K_h^T [B, 8, 32, 40] → attn [B, 8, Q, 40]
  │     │     │     softmax(attn / √32) × V_h [B, 8, 40, 32] → [B, 8, Q, 32]
  │     │     │     concat heads → [B, Q, 256] → out_proj Linear(256→256)
  │     │     │     → attn_out                [B, Q, 256]
  │     │     │
  │     │     └── LayerNorm(query_feat + attn_out)  → enhanced query_feat [B, Q, 256]
  │     │
  │     ├── 4. Adaptive Spatio-temporal Sampling
  │     │     ├── sampling_offset: Linear(256 → G×P×3 = 4×4×3 = 48)
  │     │     │     → [B, Q, 16, 3] → make_sample_points → [B, Q, G, F, P, 3]
  │     │     │       (G=4组, F=8帧, P=4点)
  │     │     ├── 速度补偿: sampling_points[..., :2] -= vel × time_diff
  │     │     ├── scale_weights: Linear(256 → G×P×L = 4×4×4 = 64)
  │     │     │     → softmax → [B, Q, 4, 8, 4, 4]
  │     │     └── sampling_4d(points, mlvl_feats, weights, lidar2img, H, W)
  │     │           3D点 → lidar2img投影到2D → grid_sample双线性插值
  │     │           → sampled_feats [B, Q, G=4, F×P=32, C=64]
  │     │
  │     ├── 5. Adaptive Mixing
  │     │     ├── parameter_generator: Linear(256 → G×(m_params+s_params))
  │     │     │     m_params = 64×64 = 4096 (channel mixing矩阵)
  │     │     │     s_params = 32×128 = 4096 (point mixing矩阵)
  │     │     │     → [B×Q, G=4, 8192]
  │     │     ├── Channel Mixing:
  │     │     │     [B×Q, 4, 32, 64] × M[B×Q, 4, 64, 64] → [B×Q, 4, 32, 64]
  │     │     │     → LayerNorm → ReLU
  │     │     ├── Point Mixing:
  │     │     │     S[B×Q, 4, 128, 32] × [B×Q, 4, 32, 64] → [B×Q, 4, 128, 64]
  │     │     │     → LayerNorm → ReLU
  │     │     ├── reshape → [B, Q, 4×128×64 = 32768]
  │     │     ├── out_proj: Linear(32768 → 256)
  │     │     └── residual: query + out → [B, Q, 256]
  │     │     query_feat = LayerNorm(mixing_out)
  │     │
  │     ├── 6. FFN
  │     │     Linear(256→512) → ReLU → Dropout(0.1) → Linear(512→256)
  │     │     query_feat = LayerNorm(ffn_out)                            [B, Q, 256]
  │     │
  │     ├── 7. Prediction Heads
  │     │     cls_branch: Linear(256→256)→LN→ReLU → Linear(256→256)→LN→ReLU → Linear(256→10)
  │     │       → cls_score [B, Q, 10]
  │     │     reg_branch: Linear(256→256)→ReLU → Linear(256→256)→ReLU → Linear(256→10)
  │     │       → bbox_delta [B, Q, 10]
  │     │     refine_bbox: sigmoid(inverse_sigmoid(bbox[:3]) + delta[:3]), delta[3:]
  │     │       → bbox_pred [B, Q, 10]
  │     │
  │     └── query_bbox = bbox_pred.detach()  (下一层输入)
  │
  ├── 缓存 last_query_feat = query_feat.detach()  [B, Q, 256]
  │
  └── 输出: cls_scores [6, B, Q, 10], bbox_preds [6, B, Q, 10]
```

## 损失计算

```
SparseBEVHead.loss()
  │
  ├── 检测损失 (每层decoder都算, 共6层):
  │     ├── 匈牙利匹配: assigner.assign(bbox_pred, cls_score, gt_bboxes, gt_labels)
  │     ├── loss_cls:  FocalLoss(cls_scores, labels)
  │     └── loss_bbox: L1Loss(bbox_preds, normalized_gt_bboxes) × code_weights
  │
  ├── Query Denoising损失:
  │     ├── loss_cls_dn:  FocalLoss
  │     └── loss_bbox_dn: L1Loss
  │
  └── ★ CorrBEV对比损失 + EMA更新:
        │
        ├── _corrbev_loss_and_update():
        │     ├── 取缓存的 last_query_feat [B, Q, 256]
        │     ├── 对每张图重新做匈牙利匹配, 提取正样本:
        │     │     pos_feat  [Npos, 256]  (.detach())
        │     │     pos_label [Npos]
        │     │
        │     ├── 对比损失: contrastive_loss(pos_feat, pos_label)
        │     │     ├── L2归一化: query_norm, lang_norm
        │     │     ├── logits = exp(logit_scale) × query_norm @ lang_norm^T   [Npos, 10]
        │     │     │     logit_scale: 可学习标量, 初始化=ln(100)≈4.6
        │     │     └── sigmoid_focal_loss(logits, labels, alpha=0.25, gamma=2.0)
        │     │           target = one_hot(labels, 10)                          [Npos, 10]
        │     │           ce = BCE(logits, target)
        │     │           focal_weight = alpha_t × (1 - p_t)^gamma
        │     │           loss = sum(focal_weight × ce) / Npos × contrastive_weight(0.5)
        │     │
        │     └── EMA更新: update_visual_prototypes(pos_feat, pos_label)
        │           (详见下方"视觉原型更新机制")
        │
        └── loss_contrastive 加入 loss_dict

总loss = Σ(loss_cls + loss_bbox) × 6层
       + Σ(loss_cls_dn + loss_bbox_dn) × 6层
       + loss_contrastive
```

## 视觉原型更新机制

```
update_visual_prototypes(pos_feats [Npos, 256], pos_labels [Npos])
  │
  │  vis_proto:       [NK=40, 256]   ← 10类 × 4子原型
  │  vis_proto_count: [40]           ← 每个子原型的累计更新次数
  │  ring_buffer:     [10, 128, 256] ← 每类128个最新样本的环形缓冲区
  │  ring_ptr:        [10]           ← 每类的写入指针
  │  ring_total:      [10]           ← 每类总共收到过多少样本
  │
  ├── 对每个正样本 feat [256], cls_id:
  │     │
  │     ├── Step 1: 写入环形缓冲区 (永远执行)
  │     │     ring_buffer[cls_id, ptr] = feat
  │     │     ring_ptr[cls_id] = (ptr + 1) % 128
  │     │     ring_total[cls_id] += 1
  │     │
  │     ├── Step 2a: 冷启动 (cls_initialized[cls_id] == False)
  │     │     if ring_total[cls_id] >= 128:
  │     │       FPS(ring_buffer[cls_id], K=4) → 选4个最分散的
  │     │       vis_proto[cls_id*4 : cls_id*4+4] = selected
  │     │       cls_initialized[cls_id] = True
  │     │
  │     └── Step 2b: 日常EMA (cls_initialized[cls_id] == True)
  │           sub_protos = vis_proto[cls_id*4 : cls_id*4+4]   [4, 256]
  │           sub_counts = vis_proto_count[同上]                [4]
  │           │
  │           ├── 基础距离: dists = ||sub_protos - feat||       [4]
  │           ├── 频率惩罚: penalty = log(1+count_j) / log(1+Σcount)  [4]
  │           ├── 调整距离: dists_adj = dists × (1 + penalty)  [4]
  │           ├── 选最近: j = argmin(dists_adj)
  │           ├── 自适应momentum:
  │           │     m = 0.999 - 0.499 × exp(-count_j / 200)
  │           │     count=1→m≈0.50, count=200→m≈0.95, count=1000→m≈0.997
  │           └── 更新: vis_proto[j] = m × vis_proto[j] + (1-m) × feat
  │
  └── Step 3: 周期重校准 (每500步)
        _recalibrate_prototypes():
        对每个已初始化的类别:
          ├── 取ring_buffer中有效样本 [n_valid, 256]
          ├── FPS选4个理想位置 [4, 256]
          ├── 贪心匹配: 当前4个子原型 ↔ 4个理想位置 (最小距离配对)
          └── 软靠拢: vis_proto[j] = 0.8 × vis_proto[j] + 0.2 × ideal[matched_k]
```

## 模块参数量统计

| 模块 | 参数 | 数量 | 可学习 |
|------|------|------|--------|
| lang_proto | nn.Embedding(10, 256) | 2,560 | 是 |
| vis_proto | buffer(40, 256) | 10,240 | 否(EMA) |
| fusion_proj | Linear(512→256) + LN(256) | 131,584 | 是 |
| cross_attn | MHA(256, 8heads) | 263,168 | 是 |
| cross_attn_norm | LN(256) | 512 | 是 |
| logit_scale | scalar | 1 | 是 |
| ring_buffer | buffer(10, 128, 256) | 327,680 | 否 |
| **OnlinePrototypeGenerator 总计** | | **~398K可学习 + ~338K buffer** | |

## 配置参数

```python
corrbev=dict(
    enable=True,
    num_sub_protos=4,        # K: 每类子原型数, NK=10×4=40
    num_heads=8,             # cross-attention头数, d_k=256/8=32
    ema_momentum=0.999,      # 自适应momentum上限
    contrastive_weight=0.5,  # 对比损失权重
    buffer_size=128,         # 每类环形缓冲区大小
    recalib_interval=500,    # 重校准间隔(global steps)
)
```

## 文件清单

| 文件 | 角色 |
|------|------|
| `models/corrbev_prototype.py` | OnlinePrototypeGenerator: 原型维护 + enhance_query + 对比损失 + EMA更新 |
| `models/sparsebev_head.py` | 集成入口: 创建prototype_gen, 传给transformer, loss中触发对比损失和EMA |
| `models/sparsebev_transformer.py` | DecoderLayer中调用prototype_gen.enhance_query() |
| `configs/r50_nuimg_704x256.py` | corrbev配置参数 |
