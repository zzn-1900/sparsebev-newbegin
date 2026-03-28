import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlinePrototypeGenerator(nn.Module):
    """在线多模态原型生成器（3D空间版）

    视觉原型: 通过EMA从训练中匹配到GT的正样本特征逐步积累，按类别×子原型维护
    语言原型: 纯可学习Embedding，端到端训练
    融合原型: 视觉+语言拼接后线性投影

    原型增强方式: query_feat通过cross-attention attend到融合原型
    - 不在2D特征图上做correlation，零额外显存
    - query直接从原型中查询类别先验，在3D空间完成

    子原型生命周期:
    1. 冷启动: 每个类别攒样本到环形缓冲区，攒够后FPS初始化K个子原型
    2. 日常更新: 频率感知的最近邻EMA（自适应momentum + 频率惩罚防坍缩）
    3. 周期重校准: 每recalib_interval步，用缓冲区最新样本重新FPS，
                   算出"理想位置"后用EMA软靠拢（不硬重置，避免突变）
    """

    def __init__(self,
                 num_classes=10,
                 num_sub_protos=4,
                 embed_dims=256,
                 num_heads=8,
                 ema_momentum=0.999,
                 contrastive_weight=0.5,
                 buffer_size=128,
                 recalib_interval=500):
        """
        Args:
            num_heads: cross-attention的头数
            buffer_size: 每个类别的环形缓冲区大小
            recalib_interval: 每隔多少个global_step做一次FPS重校准
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_sub_protos = num_sub_protos
        self.embed_dims = embed_dims
        self.ema_momentum = ema_momentum
        self.contrastive_weight = contrastive_weight
        self.buffer_size = buffer_size
        self.recalib_interval = recalib_interval

        self.NK = num_classes * num_sub_protos  # 总原型数

        # 语言原型: 可学习，梯度直接回传
        self.lang_proto = nn.Embedding(num_classes, embed_dims)

        # 视觉原型: 不通过梯度更新，EMA维护
        self.register_buffer('vis_proto', torch.zeros(self.NK, embed_dims))
        self.register_buffer('vis_proto_count', torch.zeros(self.NK, dtype=torch.long))
        self.register_buffer('global_step', torch.tensor(0, dtype=torch.long))

        # 每个类别是否已完成冷启动
        self.register_buffer('cls_initialized', torch.zeros(num_classes, dtype=torch.bool))

        # 环形缓冲区
        self.register_buffer('ring_buffer', torch.zeros(num_classes, buffer_size, embed_dims))
        self.register_buffer('ring_ptr', torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer('ring_total', torch.zeros(num_classes, dtype=torch.long))

        # 融合投影: cat(vis, lang) → fused
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )

        # Cross-attention: query_feat attend到融合原型
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(embed_dims)

        # 对比损失的温度参数
        self.logit_scale = nn.Parameter(torch.tensor(4.6052))  # ln(100)

    def get_fused_prototypes(self):
        """获取融合后的原型 P ∈ (NK, D)"""
        lang = self.lang_proto.weight  # (N, D)
        lang_expanded = lang.unsqueeze(1).expand(
            -1, self.num_sub_protos, -1
        ).reshape(self.NK, self.embed_dims)  # (NK, D)

        vis = self.vis_proto  # (NK, D)

        fused = self.fusion_proj(torch.cat([vis, lang_expanded], dim=-1))  # (NK, D)
        return fused

    def enhance_query(self, query_feat):
        """用融合原型增强query_feat

        query_feat通过cross-attention attend到所有融合原型，
        从中提取类别先验信息，增强自身表征。

        Args:
            query_feat: (B, Q, D) decoder中的query特征

        Returns:
            enhanced: (B, Q, D) 增强后的query特征
        """
        B, Q, D = query_feat.shape

        # 融合原型作为KV: (NK, D) → (B, NK, D)
        prototypes = self.get_fused_prototypes()  # (NK, D)
        prototypes = prototypes.unsqueeze(0).expand(B, -1, -1)  # (B, NK, D)

        # cross-attention: Q=query_feat, K=V=prototypes
        attn_out, _ = self.cross_attn(
            query=query_feat,
            key=prototypes,
            value=prototypes,
        )  # (B, Q, D)

        # 残差 + LayerNorm
        enhanced = self.cross_attn_norm(query_feat + attn_out)
        return enhanced

    def contrastive_loss(self, query_feat, labels):
        """对比语义对齐损失

        Args:
            query_feat: decoder输出的query特征 (Npos, D)
            labels: 对应的GT类别标签 (Npos,)

        Returns:
            loss: focal loss标量
        """
        if query_feat.shape[0] == 0:
            return query_feat.sum() * 0.0

        lang = self.lang_proto.weight  # (N, D)
        scale = self.logit_scale.exp().clamp(max=100.0)

        query_norm = F.normalize(query_feat, dim=-1)
        lang_norm = F.normalize(lang, dim=-1)
        logits = scale * torch.matmul(query_norm, lang_norm.t())  # (Npos, N)

        loss = sigmoid_focal_loss(logits, labels, self.num_classes)
        return loss * self.contrastive_weight

    # ==================== 视觉原型在线更新 ====================

    def _farthest_point_sampling(self, feats, K):
        """最远点采样: 从feats中选K个最分散的点"""
        M = feats.shape[0]
        if M <= K:
            if M == 0:
                return torch.zeros(K, feats.shape[1], device=feats.device)
            pad = feats.mean(dim=0, keepdim=True).expand(K - M, -1)
            return torch.cat([feats, pad], dim=0)

        selected_idx = [0]
        min_dists = torch.full((M,), float('inf'), device=feats.device)

        for _ in range(K - 1):
            last = feats[selected_idx[-1]]
            dists = torch.norm(feats - last.unsqueeze(0), dim=-1)
            min_dists = torch.min(min_dists, dists)
            next_idx = min_dists.argmax().item()
            selected_idx.append(next_idx)

        return feats[selected_idx]

    def _adaptive_momentum(self, count):
        """自适应momentum: 更新少的原型吸收更多新信息"""
        m_min = 0.5
        m_max = self.ema_momentum
        tau = 200.0
        m = m_max - (m_max - m_min) * torch.exp(-count.float() / tau)
        return m.item()

    @torch.no_grad()
    def update_visual_prototypes(self, pos_feats, pos_labels):
        """EMA更新视觉原型

        1. 环形缓冲区写入（永不停止）
        2. 冷启动(FPS初始化) / 日常EMA(频率感知最近邻)
        3. 周期重校准(FPS软靠拢)
        """
        if pos_feats.shape[0] == 0:
            return

        self.global_step += 1
        K = self.num_sub_protos

        for i in range(pos_feats.shape[0]):
            feat = pos_feats[i]
            cls_id = pos_labels[i].item()
            start = cls_id * K

            # 写入环形缓冲区
            ptr = self.ring_ptr[cls_id].item()
            self.ring_buffer[cls_id, ptr] = feat
            self.ring_ptr[cls_id] = (ptr + 1) % self.buffer_size
            self.ring_total[cls_id] += 1

            if not self.cls_initialized[cls_id]:
                if self.ring_total[cls_id] >= self.buffer_size:
                    valid = self.ring_buffer[cls_id]
                    selected = self._farthest_point_sampling(valid, K)
                    self.vis_proto[start:start + K] = selected
                    self.vis_proto_count[start:start + K] = 1
                    self.cls_initialized[cls_id] = True
            else:
                sub_protos = self.vis_proto[start:start + K]
                sub_counts = self.vis_proto_count[start:start + K]

                dists = torch.norm(sub_protos - feat.unsqueeze(0), dim=-1)

                total_count = sub_counts.sum().float().clamp(min=1.0)
                freq_penalty = torch.log1p(sub_counts.float()) / torch.log1p(total_count)
                dists_adj = dists * (1.0 + freq_penalty)

                j = dists_adj.argmin().item()
                momentum = self._adaptive_momentum(self.vis_proto_count[start + j])
                self.vis_proto[start + j] = momentum * self.vis_proto[start + j] + (1 - momentum) * feat
                self.vis_proto_count[start + j] += 1

        if self.global_step % self.recalib_interval == 0:
            self._recalibrate_prototypes()

    @torch.no_grad()
    def _recalibrate_prototypes(self):
        """周期重校准: FPS算理想位置，EMA软靠拢"""
        K = self.num_sub_protos
        recalib_momentum = 0.8

        for cls_id in range(self.num_classes):
            if not self.cls_initialized[cls_id]:
                continue

            n_valid = min(self.ring_total[cls_id].item(), self.buffer_size)
            if n_valid < K:
                continue

            valid_feats = self.ring_buffer[cls_id, :n_valid]
            ideal = self._farthest_point_sampling(valid_feats, K)

            start = cls_id * K
            current = self.vis_proto[start:start + K]

            # 贪心匹配
            used_ideal = set()
            for j in range(K):
                best_dist = float('inf')
                best_k = 0
                for k in range(K):
                    if k in used_ideal:
                        continue
                    d = torch.norm(current[j] - ideal[k]).item()
                    if d < best_dist:
                        best_dist = d
                        best_k = k
                used_ideal.add(best_k)
                self.vis_proto[start + j] = (
                    recalib_momentum * self.vis_proto[start + j]
                    + (1 - recalib_momentum) * ideal[best_k]
                )


def sigmoid_focal_loss(logits, labels, num_classes, alpha=0.25, gamma=2.0):
    """Sigmoid focal loss for contrastive alignment"""
    target = F.one_hot(labels, num_classes).float()

    prob = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')

    p_t = prob * target + (1 - prob) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_weight = alpha_t * (1 - p_t) ** gamma

    loss = (focal_weight * ce_loss).sum() / max(labels.shape[0], 1)
    return loss