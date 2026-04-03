import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.cnn import bias_init_with_prob
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN
from mmdet.models.utils.builder import TRANSFORMER
from .bbox.utils import decode_bbox
from .utils import inverse_sigmoid, DUMP
from .sparsebev_sampling import sampling_4d, make_sample_points
from .checkpoint import checkpoint as cp
from .csrc.wrapper import MSMV_CUDA


@TRANSFORMER.register_module()
class SparseBEVTransformer(BaseModule):
    def __init__(self, embed_dims, num_frames=8, num_points=4, num_layers=6, num_levels=4, num_classes=10, code_size=10, pc_range=[], init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                            'behavior, init_cfg is not allowed to be set'
        super(SparseBEVTransformer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.pc_range = pc_range

        self.decoder = SparseBEVTransformerDecoder(embed_dims, num_frames, num_points, num_layers, num_levels, num_classes, code_size, pc_range=pc_range)

    @torch.no_grad()
    def init_weights(self):
        self.decoder.init_weights()

    def forward(self, query_bbox, query_feat, mlvl_feats, attn_mask, img_metas):
        cls_scores, bbox_preds = self.decoder(query_bbox, query_feat, mlvl_feats, attn_mask, img_metas)

        cls_scores = torch.nan_to_num(cls_scores)
        bbox_preds = torch.nan_to_num(bbox_preds)

        return cls_scores, bbox_preds


class SparseBEVTransformerDecoder(BaseModule):
    def __init__(self, embed_dims, num_frames=8, num_points=4, num_layers=6, num_levels=4, num_classes=10, code_size=10, pc_range=[], init_cfg=None):
        super(SparseBEVTransformerDecoder, self).__init__(init_cfg)
        self.num_layers = num_layers
        self.pc_range = pc_range

        # params are shared across all decoder layers
        self.decoder_layer = SparseBEVTransformerDecoderLayer(
            embed_dims, num_frames, num_points, num_levels, num_classes, code_size, pc_range=pc_range
        )

    @torch.no_grad()
    def init_weights(self):
        self.decoder_layer.init_weights()

    def forward(self, query_bbox, query_feat, mlvl_feats, attn_mask, img_metas):
        cls_scores, bbox_preds = [], []

        # calculate time difference according to timestamps
        timestamps = np.array([m['img_timestamp'] for m in img_metas], dtype=np.float64)
        timestamps = np.reshape(timestamps, [query_bbox.shape[0], -1, 6])
        time_diff = timestamps[:, :1, :] - timestamps
        time_diff = np.mean(time_diff, axis=-1).astype(np.float32)  # [B, F]
        time_diff = torch.from_numpy(time_diff).to(query_bbox.device)  # [B, F]
        img_metas[0]['time_diff'] = time_diff

        # organize projections matrix and copy to CUDA
        lidar2img = np.asarray([m['lidar2img'] for m in img_metas]).astype(np.float32)
        lidar2img = torch.from_numpy(lidar2img).to(query_bbox.device)  # [B, N, 4, 4]
        img_metas[0]['lidar2img'] = lidar2img

        # group image features in advance for sampling, see `sampling_4d` for more details
        for lvl, feat in enumerate(mlvl_feats):
            B, TN, GC, H, W = feat.shape  # [B, TN, GC, H, W]
            N, T, G, C = 6, TN // 6, 4, GC // 4
            feat = feat.reshape(B, T, N, G, C, H, W)

            if MSMV_CUDA:  # Our CUDA operator requires channel_last
                feat = feat.permute(0, 1, 3, 2, 5, 6, 4)  # [B, T, G, N, H, W, C]
                feat = feat.reshape(B*T*G, N, H, W, C)
            else:  # Torch's grid_sample requires channel_first
                feat = feat.permute(0, 1, 3, 4, 2, 5, 6)  # [B, T, G, C, N, H, W]
                feat = feat.reshape(B*T*G, C, N, H, W)

            mlvl_feats[lvl] = feat.contiguous()

        for i in range(self.num_layers):
            DUMP.stage_count = i

            query_feat, cls_score, bbox_pred = self.decoder_layer(
                query_bbox, query_feat, mlvl_feats, attn_mask, img_metas
            )
            query_bbox = bbox_pred.clone().detach()

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)

        cls_scores = torch.stack(cls_scores)
        bbox_preds = torch.stack(bbox_preds)

        return cls_scores, bbox_preds


class SparseBEVTransformerDecoderLayer(BaseModule):
    def __init__(self, embed_dims, num_frames=8, num_points=4, num_levels=4, num_classes=10, code_size=10, num_cls_fcs=2, num_reg_fcs=2, pc_range=[], init_cfg=None):
        super(SparseBEVTransformerDecoderLayer, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.code_size = code_size
        self.pc_range = pc_range

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.self_attn = SparseBEVSelfAttention(embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range)
        self.sampling = SparseBEVSampling(embed_dims, num_frames=num_frames, num_groups=4, num_points=num_points, num_levels=num_levels, pc_range=pc_range)
        self.mixing = AdaptiveMixing(in_dim=embed_dims, in_points=num_points * num_frames, n_groups=4, out_points=128, num_frames=num_frames)
        self.ffn = FFN(embed_dims, feedforward_channels=512, ffn_drop=0.1)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

        cls_branch = []
        for _ in range(num_cls_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.num_classes))
        self.cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        self.reg_branch = nn.Sequential(*reg_branch)

    @torch.no_grad()
    def init_weights(self):
        self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)

    def refine_bbox(self, bbox_proposal, bbox_delta):
        xyz = inverse_sigmoid(bbox_proposal[..., 0:3])
        xyz_delta = bbox_delta[..., 0:3]
        xyz_new = torch.sigmoid(xyz_delta + xyz)

        return torch.cat([xyz_new, bbox_delta[..., 3:]], dim=-1)

    def forward(self, query_bbox, query_feat, mlvl_feats, attn_mask, img_metas):
        """
        query_bbox: [B, Q, 10] [cx, cy, cz, w, h, d, rot.sin, rot.cos, vx, vy]
        """
        query_pos = self.position_encoder(query_bbox[..., :3])
        query_feat = query_feat + query_pos

        query_feat = self.norm1(self.self_attn(query_bbox, query_feat, attn_mask))
        sampled_feat = self.sampling(query_bbox, query_feat, mlvl_feats, img_metas)

        # 将速度和时间差传给 mixing，用于运动感知的时序位置编码
        vel = query_bbox[..., 8:10].detach()       # [B, Q, 2]  m/s
        time_diff = img_metas[0]['time_diff']       # [B, F]     秒
        query_feat = self.norm2(self.mixing(sampled_feat, query_feat, time_diff, vel))

        query_feat = self.norm3(self.ffn(query_feat))

        cls_score = self.cls_branch(query_feat)  # [B, Q, num_classes]
        bbox_pred = self.reg_branch(query_feat)  # [B, Q, code_size]
        bbox_pred = self.refine_bbox(query_bbox, bbox_pred)

        # calculate absolute velocity according to time difference
        time_diff = img_metas[0]['time_diff']  # [B, F]
        if time_diff.shape[1] > 1:
            time_diff = time_diff.clone()
            time_diff[time_diff < 1e-5] = 1.0
            bbox_pred[..., 8:] = bbox_pred[..., 8:] / time_diff[:, 1:2, None]

        if DUMP.enabled:
            query_bbox_dec = decode_bbox(query_bbox, self.pc_range)
            bbox_pred_dec = decode_bbox(bbox_pred, self.pc_range)
            cls_score_sig = torch.sigmoid(cls_score)
            torch.save(query_bbox_dec.cpu(), '{}/query_bbox_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))
            torch.save(bbox_pred_dec.cpu(), '{}/bbox_pred_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))
            torch.save(cls_score_sig.cpu(), '{}/cls_score_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

        return query_feat, cls_score, bbox_pred


class SparseBEVSelfAttention(BaseModule):
    """Scale-adaptive Self Attention"""
    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1, pc_range=[], init_cfg=None):
        super().__init__(init_cfg)
        self.pc_range = pc_range

        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def inner_forward(self, query_bbox, query_feat, pre_attn_mask):
        """
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, C]
        """
        dist = self.calc_bbox_dists(query_bbox)
        tau = self.gen_tau(query_feat)  # [B, Q, 8]

        if DUMP.enabled:
            torch.save(tau.cpu(), '{}/sasa_tau_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

        tau = tau.permute(0, 2, 1)  # [B, 8, Q]
        attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, 8, Q, Q]

        if pre_attn_mask is not None:  # for query denoising
            attn_mask[:, :, pre_attn_mask] = float('-inf')

        attn_mask = attn_mask.flatten(0, 1)  # [Bx8, Q, Q]
        return self.attention(query_feat, attn_mask=attn_mask)

    def forward(self, query_bbox, query_feat, pre_attn_mask):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_bbox, query_feat, pre_attn_mask, use_reentrant=False)
        else:
            return self.inner_forward(query_bbox, query_feat, pre_attn_mask)

    @torch.no_grad()
    def calc_bbox_dists(self, bboxes):
        centers = decode_bbox(bboxes, self.pc_range)[..., :2]  # [B, Q, 2]

        dist = []
        for b in range(centers.shape[0]):
            dist_b = torch.norm(centers[b].reshape(-1, 1, 2) - centers[b].reshape(1, -1, 2), dim=-1)
            dist.append(dist_b[None, ...])

        dist = torch.cat(dist, dim=0)  # [B, Q, Q]
        dist = -dist

        return dist


class SparseBEVSampling(BaseModule):
    """Adaptive Spatio-temporal Sampling"""
    def __init__(self, embed_dims=256, num_frames=4, num_groups=4, num_points=8, num_levels=4, pc_range=[], init_cfg=None):
        super().__init__(init_cfg)

        self.num_frames = num_frames
        self.num_points = num_points
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.pc_range = pc_range

        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)

        # 查询驱动的时序尺度偏置
        # 每个query根据自身特征为不同帧生成不同的FPN层偏好
        # 输入: query_feat [B,Q,C] → 输出: [B,Q,F*L] → view [B,Q,F,L]
        self.temporal_scale_bias = nn.Linear(embed_dims, num_frames * num_levels)

        # [Round 4] Query-Adaptive Temporal Refinement
        # 让每个query根据自身特征微调时序权重（轻量：仅一层线性+sigmoid）
        self.temporal_refine = nn.Linear(embed_dims, num_groups * num_frames)

        # [Round 9] 自适应速度补偿残差修正
        # 学习对线性 v*Δt 补偿的非线性修正（加减速、转弯场景）
        # 输入：query特征 → 输出：每帧的 2D 位移修正量
        self.velocity_correction = nn.Linear(embed_dims, num_frames * 2)

    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)
        nn.init.zeros_(self.temporal_refine.weight)
        nn.init.zeros_(self.temporal_refine.bias)
        nn.init.zeros_(self.velocity_correction.weight)
        nn.init.zeros_(self.velocity_correction.bias)
        nn.init.zeros_(self.temporal_scale_bias.weight)
        nn.init.zeros_(self.temporal_scale_bias.bias)

    def inner_forward(self, query_bbox, query_feat, mlvl_feats, img_metas):
        '''
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, C]
        '''
        B, Q = query_bbox.shape[:2]
        image_h, image_w, _ = img_metas[0]['img_shape'][0]

        # sampling offset of all frames
        sampling_offset = self.sampling_offset(query_feat)
        sampling_offset = sampling_offset.view(B, Q, self.num_groups * self.num_points, 3)
        sampling_points = make_sample_points(query_bbox, sampling_offset, self.pc_range)  # [B, Q, GP, 3]
        sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.expand(B, Q, self.num_frames, self.num_groups, self.num_points, 3)

        # warp sample points based on velocity
        time_diff = img_metas[0]['time_diff']  # [B, F]
        time_diff = time_diff[:, None, :, None]  # [B, 1, F, 1]
        vel = query_bbox[..., 8:].detach()  # [B, Q, 2]
        vel = vel[:, :, None, :]  # [B, Q, 1, 2]
        dist = vel * time_diff  # [B, Q, F, 2]

        # [Round 9] 自适应速度补偿残差修正
        # 基于query特征学习对线性补偿的修正（捕获加减速/转弯）
        vel_correction = self.velocity_correction(query_feat)  # [B, Q, F*2]
        vel_correction = vel_correction.view(B, Q, self.num_frames, 2)  # [B, Q, F, 2]
        # 修正量用tanh约束范围，世界坐标系（米）下
        # 最大±2米，覆盖典型加减速/转弯的线性补偿误差
        vel_correction = torch.tanh(vel_correction) * 2.0
        # [Round 12] 当前帧(time_diff=0)不需要修正，用time_diff绝对值作为mask
        # time_diff_abs越大表示越远的帧，修正量越有意义
        td_mask = img_metas[0]['time_diff'].abs()  # [B, F]
        td_mask = (td_mask > 1e-5).float()[:, None, :, None]  # [B, 1, F, 1]
        vel_correction = vel_correction * td_mask
        dist = dist + vel_correction

        dist = dist[:, :, :, None, None, :]  # [B, Q, F, 1, 1, 2]
        sampling_points = torch.cat([
            sampling_points[..., 0:2] - dist,
            sampling_points[..., 2:3]
        ], dim=-1)

        # scale weights
        scale_weights = self.scale_weights(query_feat).view(B, Q, self.num_groups, 1, self.num_points, self.num_levels)
        # 查询驱动的时序尺度偏置：每个query为不同帧生成不同FPN层偏好
        tsb = self.temporal_scale_bias(query_feat)                            # [B, Q, F*L]
        tsb = tsb.view(B, Q, 1, self.num_frames, 1, self.num_levels)         # [B, Q, 1, F, 1, L]
        scale_weights = scale_weights.expand(B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)
        scale_weights = torch.softmax(scale_weights + tsb, dim=-1)            # [B, Q, G, F, P, L]

        # sampling
        sampled_feats, frame_validity = sampling_4d(
            sampling_points,
            mlvl_feats,
            scale_weights,
            img_metas[0]['lidar2img'],
            image_h, image_w
        )  # [B, Q, G, FP, C]

        # 查询自适应时序权重：每个query根据自身特征决定各帧权重
        query_temporal_w = self.temporal_refine(query_feat)                   # [B, Q, G*F]
        query_temporal_w = query_temporal_w.view(B, Q, self.num_groups, self.num_frames)  # [B, Q, G, F]
        temporal_weights = torch.softmax(query_temporal_w, dim=-1) * self.num_frames
        # softmax 归一化 sum=1，乘 F 保持与原始等权融合一致的总幅值

        # 展开到 [B, Q, G, F*P, 1]：每帧P个点共享同一权重
        temporal_weights = temporal_weights.unsqueeze(-1).expand(B, Q, self.num_groups, self.num_frames, self.num_points)
        temporal_weights = temporal_weights.reshape(B, Q, self.num_groups, self.num_frames * self.num_points, 1)
        # 应用时序权重
        sampled_feats = sampled_feats * temporal_weights

        return sampled_feats

    def forward(self, query_bbox, query_feat, mlvl_feats, img_metas):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_bbox, query_feat, mlvl_feats, img_metas, use_reentrant=False)
        else:
            return self.inner_forward(query_bbox, query_feat, mlvl_feats, img_metas)


class AdaptiveMixing(nn.Module):
    """Adaptive Mixing"""
    def __init__(self, in_dim, in_points, n_groups=1, query_dim=None, out_dim=None, out_points=None, num_frames=8):
        super(AdaptiveMixing, self).__init__()

        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points
        self.num_frames = num_frames
        self.points_per_frame = in_points // num_frames

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points
        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Linear(self.query_dim, self.n_groups * self.total_parameters)
        self.out_proj = nn.Linear(self.eff_out_dim * self.out_points * self.n_groups, self.query_dim)
        self.act = nn.ReLU(inplace=True)

        # 运动感知的连续时序位置编码
        # 输入: [t, vx·t, vy·t] (3维) —— 同时编码时间距离和运动位移
        # 高速物体: vx·t 大 → 编码距离远 → point mixing 自然学会降权
        # 静态物体: vx·t ≈ 0 → 编码仅由 t 决定 → 充分利用多帧
        self.temporal_pos_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, self.eff_in_dim),
        )

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)
        # [Round 6] 时序位置编码器零初始化，初始时不影响原始特征
        nn.init.zeros_(self.temporal_pos_encoder[-1].weight)
        nn.init.zeros_(self.temporal_pos_encoder[-1].bias)

    def inner_forward(self, x, query, time_diff=None, vel=None):
        B, Q, G, P, C = x.shape
        assert G == self.n_groups
        assert P == self.in_points
        assert C == self.eff_in_dim

        # 运动感知的连续时序位置编码
        if time_diff is not None:
            # time_diff: [B, F] → 取batch均值 [F]
            td = time_diff.mean(dim=0)                            # [F]

            if vel is not None:
                # vel: [B, Q, 2], td: [F]
                # 构造 per-query 的 [t, vx*t, vy*t]: [B, Q, F, 3]
                td_exp = td[None, None, :]                        # [1, 1, F]
                vx_t = vel[..., 0:1] * td_exp                    # [B, Q, F]
                vy_t = vel[..., 1:2] * td_exp                    # [B, Q, F]
                td_3d = torch.stack([
                    td_exp.expand(B, Q, self.num_frames),         # [B, Q, F]
                    vx_t,                                          # [B, Q, F]
                    vy_t,                                          # [B, Q, F]
                ], dim=-1)                                         # [B, Q, F, 3]
            else:
                td_3d = td[None, None, :, None].expand(B, Q, self.num_frames, 1)
                td_3d = torch.cat([td_3d, torch.zeros(B, Q, self.num_frames, 2, device=x.device)], dim=-1)

            temp_pos = self.temporal_pos_encoder(td_3d)            # [B, Q, F, C_g]
            # 每帧内P个点共享同一编码
            temp_pos = temp_pos.unsqueeze(3).expand(B, Q, self.num_frames, self.points_per_frame, C)
            temp_pos = temp_pos.reshape(B, Q, 1, P, C)            # [B, Q, 1, F*Pp, C_g]
            x = x + temp_pos                                       # broadcast G维

        '''generate mixing parameters'''
        params = self.parameter_generator(query)
        params = params.reshape(B*Q, G, -1)
        out = x.reshape(B*Q, G, P, C)

        M, S = params.split([self.m_parameters, self.s_parameters], 2)
        M = M.reshape(B*Q, G, self.eff_in_dim, self.eff_out_dim)
        S = S.reshape(B*Q, G, self.out_points, self.in_points)

        '''adaptive channel mixing'''
        out = torch.matmul(out, M)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''adaptive point mixing'''
        out = torch.matmul(S, out)  # implicitly transpose and matmul
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''linear transfomation to query dim'''
        out = out.reshape(B, Q, -1)
        out = self.out_proj(out)
        out = query + out  # 残差连接

        return out

    def forward(self, x, query, time_diff=None, vel=None):
        if self.training and x.requires_grad:
            return cp(self.inner_forward, x, query, time_diff, vel, use_reentrant=False)
        else:
            return self.inner_forward(x, query, time_diff, vel)
