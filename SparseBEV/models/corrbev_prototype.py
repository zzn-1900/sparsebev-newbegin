import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlinePrototypeGenerator(nn.Module):
    """在线多模态原型生成器

    视觉原型: 通过EMA从训练中匹配到GT的正样本特征逐步积累，按类别×子原型维护
    语言原型: 纯可学习Embedding，端到端训练
    融合原型: 视觉+语言拼接后线性投影

    子原型生命周期:
    1. 冷启动: 每个类别攒样本到环形缓冲区，攒够后FPS初始化K个子原型
    2. 日常更新: 频率感知的最近邻EMA（自适应momentum + 频率惩罚防坍缩）
    3. 周期重校准: 每recalib_interval步，用缓冲区最新样本重新FPS，
                   算出"理想位置"后用EMA软靠拢（不硬重置，避免突变）

    支持:
    - Correlation: 用融合原型与backbone特征做相关
    - 对比语义对齐损失: query特征与语言原型的focal loss
    """

    def __init__(self,
                 num_classes=10,
                 num_sub_protos=4,
                 embed_dims=256,
                 corr_dims=64,
                 ema_momentum=0.999,
                 contrastive_weight=0.5,
                 buffer_size=128,
                 recalib_interval=500):
        """
        Args:
            buffer_size: 每个类别的环形缓冲区大小，持续收集最新样本
            recalib_interval: 每隔多少个global_step做一次FPS重校准
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_sub_protos = num_sub_protos
        self.embed_dims = embed_dims
        self.corr_dims = corr_dims
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

        # 环形缓冲区: 持续收集每个类别的最新样本，覆盖最旧的
        self.register_buffer('ring_buffer', torch.zeros(num_classes, buffer_size, embed_dims))
        self.register_buffer('ring_ptr', torch.zeros(num_classes, dtype=torch.long))
        # 每个类别总共收到过多少样本（用于判断缓冲区是否已满过一圈）
        self.register_buffer('ring_total', torch.zeros(num_classes, dtype=torch.long))

        # 融合投影: cat(vis, lang) → fused
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )

        # Correlation降维: NK*1 通道 → corr_dims
        self.corr_reduce = nn.Sequential(
            nn.Linear(self.NK, corr_dims),
            nn.LayerNorm(corr_dims),
            nn.ReLU(inplace=True),
        )

        # 对比损失的温度参数
        self.logit_scale = nn.Parameter(torch.tensor(4.6052))  # ln(100)

        # 特征投影: 当输入通道数 != embed_dims时使用（如per-group通道=64）
        self.feat_proj = nn.Linear(embed_dims // 4, embed_dims)

    def get_fused_prototypes(self):
        """获取融合后的原型 P ∈ (NK, D)"""
        # 语言原型broadcast: (N, D) → (NK, D)
        lang = self.lang_proto.weight  # (N, D)
        lang_expanded = lang.unsqueeze(1).expand(
            -1, self.num_sub_protos, -1
        ).reshape(self.NK, self.embed_dims)  # (NK, D)

        # 视觉原型
        vis = self.vis_proto  # (NK, D)

        # 融合
        fused = self.fusion_proj(torch.cat([vis, lang_expanded], dim=-1))  # (NK, D)
        return fused

    def compute_correlation(self, feat):
        """用融合原型对backbone特征做correlation

        Args:
            feat: backbone特征 (B_total, C, H, W)，C可以是任意通道数（如64 per group）

        Returns:
            corr_feat: correlation特征 (B_total, corr_dims, H, W)
        """
        B_total, C, H, W = feat.shape

        # 获取融合原型并投影到feat的通道维度
        P = self.get_fused_prototypes()  # (NK, D)

        # 如果feat通道数与embed_dims不同，需要投影
        if C != self.embed_dims:
            feat_flat = feat.reshape(B_total, C, H * W).permute(0, 2, 1)  # (B_total, HW, C)
            feat_proj = self.feat_proj(feat_flat)  # (B_total, HW, D)
        else:
            feat_proj = feat.reshape(B_total, C, H * W).permute(0, 2, 1)  # (B_total, HW, D)

        # correlation: (B_total, HW, D) @ (D, NK) → (B_total, HW, NK)
        corr = torch.matmul(feat_proj, P.t())

        # 降维: NK → corr_dims
        corr = self.corr_reduce(corr)  # (B_total, HW, corr_dims)
        corr = corr.permute(0, 2, 1).reshape(B_total, self.corr_dims, H, W)

        return corr

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

        # query与语言原型的相似度
        lang = self.lang_proto.weight  # (N, D)
        scale = self.logit_scale.exp().clamp(max=100.0)

        # L2归一化后点积
        query_norm = F.normalize(query_feat, dim=-1)
        lang_norm = F.normalize(lang, dim=-1)
        logits = scale * torch.matmul(query_norm, lang_norm.t())  # (Npos, N)

        # Focal loss
        loss = sigmoid_focal_loss(logits, labels, self.num_classes)
        return loss * self.contrastive_weight

    def _farthest_point_sampling(self, feats, K):
        """最远点采样: 从feats中选K个最有代表性（最分散）的点

        Args:
            feats: (M, D) 候选特征
            K: 要选的点数

        Returns:
            selected: (K, D) 选中的特征
        """
        M = feats.shape[0]
        if M <= K:
            # 不够K个，全部使用，不足的用均值填充
            if M == 0:
                return torch.zeros(K, feats.shape[1], device=feats.device)
            pad = feats.mean(dim=0, keepdim=True).expand(K - M, -1)
            return torch.cat([feats, pad], dim=0)

        # 标准FPS算法
        selected_idx = [0]  # 从第一个点开始
        min_dists = torch.full((M,), float('inf'), device=feats.device)

        for _ in range(K - 1):
            last = feats[selected_idx[-1]]  # (D,)
            dists = torch.norm(feats - last.unsqueeze(0), dim=-1)  # (M,)
            min_dists = torch.min(min_dists, dists)
            # 选离已选集合最远的点
            next_idx = min_dists.argmax().item()
            selected_idx.append(next_idx)

        return feats[selected_idx]  # (K, D)

    def _adaptive_momentum(self, count):
        """根据原型的累计更新次数自适应计算momentum

        核心思想: 更新次数越少的原型，momentum越低（吸收更多新信息）
        - count=1 时 momentum≈0.5（几乎直接替换，快速初始化）
        - count=100 时 momentum≈0.95（中等平滑）
        - count=1000+ 时 momentum→0.999（高度稳定）

        公式: m = m_max - (m_max - m_min) * exp(-count / tau)
        """
        m_min = 0.5
        m_max = self.ema_momentum  # 0.999
        tau = 200.0  # 控制过渡速度，约200次更新后接近目标momentum
        m = m_max - (m_max - m_min) * torch.exp(-count.float() / tau)
        return m.item()

    @torch.no_grad()
    def update_visual_prototypes(self, pos_feats, pos_labels):
        """EMA更新视觉原型

        三个阶段持续运行:

        1. 环形缓冲区写入（每个样本都写，永不停止）
           - 新样本写入 ring_buffer[cls_id, ptr]，ptr循环递增
           - 缓冲区始终保存该类别最近 buffer_size 个样本的特征

        2. 冷启动 / 日常EMA更新
           - 未初始化: 攒够 buffer_size 个样本后FPS初始化
           - 已初始化: 频率感知最近邻 + 自适应momentum EMA

        3. 周期重校准（每 recalib_interval 步）
           - 对每个已初始化的类别，用缓冲区最新样本重新FPS
           - 算出"理想子原型位置"，用EMA软靠拢（不硬重置）
           - 随着训练推进，特征质量提升，子原型分布持续优化

        Args:
            pos_feats: 正样本query特征 (Npos, D)，已detach
            pos_labels: 对应的GT类别标签 (Npos,)
        """
        if pos_feats.shape[0] == 0:
            return

        self.global_step += 1
        K = self.num_sub_protos

        # === Step 1 & 2: 逐样本写入缓冲区 + EMA更新 ===
        for i in range(pos_feats.shape[0]):
            feat = pos_feats[i]  # (D,)
            cls_id = pos_labels[i].item()
            start = cls_id * K

            # 写入环形缓冲区（永远写，覆盖最旧的）
            ptr = self.ring_ptr[cls_id].item()
            self.ring_buffer[cls_id, ptr] = feat
            self.ring_ptr[cls_id] = (ptr + 1) % self.buffer_size
            self.ring_total[cls_id] += 1

            if not self.cls_initialized[cls_id]:
                # 冷启动: 攒够 buffer_size 个样本后FPS初始化
                if self.ring_total[cls_id] >= self.buffer_size:
                    valid = self.ring_buffer[cls_id]  # (buffer_size, D)
                    selected = self._farthest_point_sampling(valid, K)
                    self.vis_proto[start:start + K] = selected
                    self.vis_proto_count[start:start + K] = 1
                    self.cls_initialized[cls_id] = True
            else:
                # 日常EMA: 频率感知最近邻
                sub_protos = self.vis_proto[start:start + K]
                sub_counts = self.vis_proto_count[start:start + K]

                dists = torch.norm(sub_protos - feat.unsqueeze(0), dim=-1)

                # 频率惩罚
                total_count = sub_counts.sum().float().clamp(min=1.0)
                freq_penalty = torch.log1p(sub_counts.float()) / torch.log1p(total_count)
                dists_adj = dists * (1.0 + freq_penalty)

                j = dists_adj.argmin().item()
                momentum = self._adaptive_momentum(self.vis_proto_count[start + j])
                self.vis_proto[start + j] = momentum * self.vis_proto[start + j] + (1 - momentum) * feat
                self.vis_proto_count[start + j] += 1

        # === Step 3: 周期重校准 ===
        if self.global_step % self.recalib_interval == 0:
            self._recalibrate_prototypes()

    @torch.no_grad()
    def _recalibrate_prototypes(self):
        """用缓冲区最新样本重新FPS，EMA软靠拢到理想位置

        不硬重置子原型，而是用较低的momentum（0.8）向理想位置靠拢。
        这样既能修正早期噪声初始化的偏差，又不会造成训练突变。
        """
        K = self.num_sub_protos
        recalib_momentum = 0.8  # 重校准时的靠拢强度

        for cls_id in range(self.num_classes):
            if not self.cls_initialized[cls_id]:
                continue

            # 获取缓冲区中的有效样本
            n_valid = min(self.ring_total[cls_id].item(), self.buffer_size)
            if n_valid < K:
                continue

            valid_feats = self.ring_buffer[cls_id, :n_valid]  # (n_valid, D)
            ideal = self._farthest_point_sampling(valid_feats, K)  # (K, D)

            # 用匈牙利匹配找当前子原型和理想位置的最优对应
            # （简化版：贪心匹配，对K=4足够）
            start = cls_id * K
            current = self.vis_proto[start:start + K]  # (K, D)

            # 贪心匹配: 每次找距离最近的未匹配对
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
                # EMA软靠拢
                self.vis_proto[start + j] = (
                    recalib_momentum * self.vis_proto[start + j]
                    + (1 - recalib_momentum) * ideal[best_k]
                )


def sigmoid_focal_loss(logits, labels, num_classes, alpha=0.25, gamma=2.0):
    """Sigmoid focal loss for contrastive alignment

    Args:
        logits: (Npos, N) 预测logits
        labels: (Npos,) GT类别索引
        num_classes: 类别数
    """
    # 构建one-hot target
    target = F.one_hot(labels, num_classes).float()  # (Npos, N)

    prob = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')

    p_t = prob * target + (1 - prob) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_weight = alpha_t * (1 - p_t) ** gamma

    loss = (focal_weight * ce_loss).sum() / max(labels.shape[0], 1)
    return loss
