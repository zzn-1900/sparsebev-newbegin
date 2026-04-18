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
        self.gating = HistoricalFrameGating(embed_dims=embed_dims, num_groups=4, num_frames=num_frames)
        self.mixing = AdaptiveMixing(in_dim=embed_dims, in_points=num_points * num_frames, n_groups=4, out_points=128)
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
        self.gating.init_weights()
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
        sampled_feat = self.gating(sampled_feat, query_feat, img_metas)
        query_feat = self.norm2(self.mixing(sampled_feat, query_feat))
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

    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

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
        dist = dist[:, :, :, None, None, :]  # [B, Q, F, 1, 1, 2]
        sampling_points = torch.cat([
            sampling_points[..., 0:2] - dist,
            sampling_points[..., 2:3]
        ], dim=-1)

        # scale weights
        scale_weights = self.scale_weights(query_feat).view(B, Q, self.num_groups, 1, self.num_points, self.num_levels)
        scale_weights = torch.softmax(scale_weights, dim=-1)
        scale_weights = scale_weights.expand(B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)

        # sampling
        sampled_feats = sampling_4d(
            sampling_points,
            mlvl_feats,
            scale_weights,
            img_metas[0]['lidar2img'],
            image_h, image_w
        )  # [B, Q, G, FP, C]

        return sampled_feats

    def forward(self, query_bbox, query_feat, mlvl_feats, img_metas):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_bbox, query_feat, mlvl_feats, img_metas, use_reentrant=False)
        else:
            return self.inner_forward(query_bbox, query_feat, mlvl_feats, img_metas)


class HistoricalFrameGating(BaseModule):
    """Reliability gating for historical frames.

    Scores each historical frame (t >= 1) against the current frame (t = 0, anchor)
    and the query feature, and produces a per-group per-frame weight in [0, 1] that
    scales the sampled features before AdaptiveMixing. The current frame keeps
    weight 1, so historical frames can never exceed it. Uncertainty is modeled via
    a per-(group, frame) log-variance head that subtracts from the reliability logit
    (precision weighting in a sigmoid form).

    Input : sampled_feat [B, Q, G, T*P, C_g], query_feat [B, Q, embed_dims]
    Output: gated sampled_feat with the same shape.
    """
    def __init__(self, embed_dims=256, num_groups=4, num_frames=8, d_q=64, init_bias=3.0, init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_groups = num_groups
        self.num_frames = num_frames
        self.c_g = embed_dims // num_groups
        self.d_q = d_q
        self.init_bias = init_bias

        self.q_proj = nn.Linear(embed_dims, d_q)
        self.q_ln = nn.LayerNorm(d_q)
        self.h_proj = nn.Linear(self.c_g, d_q)
        self.h_ln = nn.LayerNorm(d_q)

        # MLP input per historical frame: [h'_t, h'_t - a'_0, g_t, cos_t, dt]
        # dims:                            D_q + D_q + D_q + 1 + 1 = 3*D_q + 2
        self.score_mlp = nn.Sequential(
            nn.Linear(3 * d_q + 2, d_q),
            nn.ReLU(inplace=True),
            nn.Linear(d_q, 2),  # [logit, log_var]
        )

    @torch.no_grad()
    def init_weights(self):
        # identity-at-init: last layer zero => logit=log_var=0 => w ~= sigmoid(init_bias)
        nn.init.zeros_(self.score_mlp[-1].weight)
        nn.init.zeros_(self.score_mlp[-1].bias)

    def forward(self, sampled_feat, query_feat, img_metas):
        B, Q, G, TP, C_g = sampled_feat.shape
        T = self.num_frames
        if T <= 1:
            return sampled_feat
        P = TP // T

        # [B, Q, G, T, P, C_g]
        x = sampled_feat.view(B, Q, G, T, P, C_g)

        # per-frame per-group pooled summary, projected to D_q with LN
        h = x.mean(dim=4)                    # [B, Q, G, T, C_g]
        h = self.h_ln(self.h_proj(h))        # [B, Q, G, T, D_q]

        # query reference in the same D_q space (shared across G)
        q_ref = self.q_ln(self.q_proj(query_feat))     # [B, Q, D_q]
        q_ref_bcast = q_ref[:, :, None, None, :]       # [B, Q, 1, 1, D_q]

        a0 = h[:, :, :, 0:1, :]              # [B, Q, G, 1, D_q] current-frame anchor
        h_hist = h[:, :, :, 1:, :]           # [B, Q, G, T-1, D_q] historical frames

        # intra-source subtraction (same projection) = frame drift signal
        diff = h_hist - a0                                      # [B, Q, G, T-1, D_q]
        # cross-source interactions (distribution-robust)
        gate = h_hist * q_ref_bcast                             # [B, Q, G, T-1, D_q]
        h_norm = F.normalize(h_hist, dim=-1, eps=1e-6)
        q_norm = F.normalize(q_ref, dim=-1, eps=1e-6)
        cos_t = (h_norm * q_norm[:, :, None, None, :]).sum(dim=-1, keepdim=True)  # [B, Q, G, T-1, 1]

        # time-diff prior, [B, T] -> [B, Q, G, T-1, 1]
        time_diff = img_metas[0]['time_diff']
        dt = time_diff[:, None, None, 1:, None].expand(B, Q, G, T - 1, 1)

        feat = torch.cat([h_hist, diff, gate, cos_t, dt], dim=-1)  # [B, Q, G, T-1, 3D_q+2]
        score = self.score_mlp(feat)                                # [B, Q, G, T-1, 2]
        logit = score[..., 0]
        log_var = score[..., 1]

        # bounded reliability: sigmoid ensures w_hist in (0, 1), strictly <= w_cur = 1
        w_hist = torch.sigmoid(logit - log_var + self.init_bias)    # [B, Q, G, T-1]
        w_cur = torch.ones_like(w_hist[..., :1])                    # [B, Q, G, 1]
        w = torch.cat([w_cur, w_hist], dim=-1)                      # [B, Q, G, T]

        x = x * w[..., None, None]                                  # broadcast to (P, C_g)
        return x.reshape(B, Q, G, TP, C_g)


class AdaptiveMixing(nn.Module):
    """Adaptive Mixing"""
    def __init__(self, in_dim, in_points, n_groups=1, query_dim=None, out_dim=None, out_points=None):
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

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points
        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Linear(self.query_dim, self.n_groups * self.total_parameters)
        self.out_proj = nn.Linear(self.eff_out_dim * self.out_points * self.n_groups, self.query_dim)
        self.act = nn.ReLU(inplace=True)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def inner_forward(self, x, query):
        B, Q, G, P, C = x.shape
        assert G == self.n_groups
        assert P == self.in_points
        assert C == self.eff_in_dim

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
        out = query + out

        return out

    def forward(self, x, query):
        if self.training and x.requires_grad:
            return cp(self.inner_forward, x, query, use_reentrant=False)
        else:
            return self.inner_forward(x, query)
