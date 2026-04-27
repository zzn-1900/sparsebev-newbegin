import math
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
    def __init__(self, embed_dims, num_frames=8, num_points=4, num_layers=6, num_levels=4,
                 num_classes=10, code_size=10, pc_range=[],
                 temporal_mixer='linear', mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 mamba_impl='auto',
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                            'behavior, init_cfg is not allowed to be set'
        super(SparseBEVTransformer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.pc_range = pc_range

        self.decoder = SparseBEVTransformerDecoder(
            embed_dims, num_frames, num_points, num_layers, num_levels, num_classes, code_size,
            pc_range=pc_range,
            temporal_mixer=temporal_mixer,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_impl=mamba_impl,
        )

    @torch.no_grad()
    def init_weights(self):
        self.decoder.init_weights()

    def forward(self, query_bbox, query_feat, mlvl_feats, attn_mask, img_metas):
        cls_scores, bbox_preds = self.decoder(query_bbox, query_feat, mlvl_feats, attn_mask, img_metas)

        cls_scores = torch.nan_to_num(cls_scores)
        bbox_preds = torch.nan_to_num(bbox_preds)

        return cls_scores, bbox_preds


class SparseBEVTransformerDecoder(BaseModule):
    def __init__(self, embed_dims, num_frames=8, num_points=4, num_layers=6, num_levels=4,
                 num_classes=10, code_size=10, pc_range=[],
                 temporal_mixer='linear', mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 mamba_impl='auto',
                 init_cfg=None):
        super(SparseBEVTransformerDecoder, self).__init__(init_cfg)
        self.num_layers = num_layers
        self.pc_range = pc_range

        # params are shared across all decoder layers
        self.decoder_layer = SparseBEVTransformerDecoderLayer(
            embed_dims, num_frames, num_points, num_levels, num_classes, code_size,
            pc_range=pc_range,
            temporal_mixer=temporal_mixer,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_impl=mamba_impl,
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
    def __init__(self, embed_dims, num_frames=8, num_points=4, num_levels=4, num_classes=10, code_size=10,
                 num_cls_fcs=2, num_reg_fcs=2, pc_range=[],
                 temporal_mixer='linear', mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 mamba_impl='auto',
                 init_cfg=None):
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
        self.mixing = AdaptiveMixing(
            in_dim=embed_dims,
            in_points=num_points * num_frames,
            n_groups=4,
            out_points=128,
            num_frames=num_frames,
            num_points=num_points,
            temporal_mixer=temporal_mixer,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_impl=mamba_impl,
        )
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


class MambaBlock(nn.Module):
    """Pure-PyTorch causal Mamba block for short sequences.

    Designed to be a drop-in replacement of mamba_ssm.Mamba for the temporal
    mixer here, where L is small (e.g. F+1=9). Uses standard PyTorch ops only,
    so backward goes through the regular autograd graph and is fully compatible
    with non-reentrant gradient checkpointing.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = max(1, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # dt_proj init (Mamba paper): bias such that softplus(bias) ~ U(dt_min, dt_max)
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp_(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplus^{-1}
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A: real, negative; A_log stored so A = -exp(A_log) is always negative
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: [B, L, d_model]
        return: [B, L, d_model]
        """
        B, L, _ = x.shape

        xz = self.in_proj(x)                                            # [B, L, 2*d_inner]
        u, z = xz.chunk(2, dim=-1)                                      # each [B, L, d_inner]

        # causal depthwise conv1d (left-causal: pad on both sides, slice first L)
        u = u.transpose(1, 2)                                           # [B, d_inner, L]
        u = self.conv1d(u)[..., :L]
        u = u.transpose(1, 2)                                           # [B, L, d_inner]
        u = F.silu(u)

        # selective parameters
        x_dbl = self.x_proj(u)                                          # [B, L, dt_rank+2*d_state]
        dt_in, B_ssm, C_ssm = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_in))                            # [B, L, d_inner]

        # compute A in fp32 for numerical stability of exp, then cast to scan dtype
        A = (-torch.exp(self.A_log.float())).to(u.dtype)                # [d_inner, d_state]

        # sequential selective scan over L (compute discretization on-the-fly)
        h = u.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(L):
            dt_t = dt[:, t].unsqueeze(-1)                               # [B, d_inner, 1]
            dA_t = torch.exp(dt_t * A)                                  # [B, d_inner, d_state]
            dB_x_t = (dt_t * B_ssm[:, t].unsqueeze(1)) * u[:, t].unsqueeze(-1)  # [B, d_inner, d_state]
            h = dA_t * h + dB_x_t
            y_t = (h * C_ssm[:, t].unsqueeze(1)).sum(-1)                # [B, d_inner]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                                      # [B, L, d_inner]

        y = y + u * self.D                                              # skip
        y = y * F.silu(z)                                               # gate
        y = self.out_proj(y)                                            # [B, L, d_model]
        return y


class AdaptiveMixing(nn.Module):
    """Adaptive Mixing with optional Mamba temporal mixer.

    temporal_mixer:
        'linear' — original AdaMixer-style point mix over flattened F*P axis.
        'mamba'  — within-frame adaptive point mix + adaptive channel mix, then a causal
                   Mamba scan along the F axis with query_feat prepended as a context token.
    """
    def __init__(self, in_dim, in_points, n_groups=1, query_dim=None, out_dim=None, out_points=None,
                 num_frames=None, num_points=None, temporal_mixer='linear',
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 mamba_impl='auto'):
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
        self.temporal_mixer = temporal_mixer

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups

        self.act = nn.ReLU(inplace=True)

        if temporal_mixer == 'linear':
            self.m_parameters = self.eff_in_dim * self.eff_out_dim
            self.s_parameters = self.in_points * self.out_points
            self.total_parameters = self.m_parameters + self.s_parameters

            self.parameter_generator = nn.Linear(self.query_dim, self.n_groups * self.total_parameters)
            self.out_proj = nn.Linear(self.eff_out_dim * self.out_points * self.n_groups, self.query_dim)

        elif temporal_mixer == 'mamba':
            assert num_frames is not None and num_points is not None, \
                'Mamba temporal mixer requires num_frames and num_points'
            assert in_points == num_frames * num_points

            self.num_frames = num_frames
            self.num_points = num_points
            # no within-frame spatial expansion; keep out_p == P to control compute
            self.out_p_per_frame = num_points

            self.m_parameters = self.eff_in_dim * self.eff_out_dim
            self.s_parameters = self.out_p_per_frame * self.num_points
            self.total_parameters = self.m_parameters + self.s_parameters

            self.parameter_generator = nn.Linear(self.query_dim, self.n_groups * self.total_parameters)
            # per-group context token derived from query_feat
            self.query_proj = nn.Linear(self.query_dim, self.n_groups * self.eff_out_dim)

            assert mamba_impl in ['auto', 'ssm', 'torch']
            self.mamba_impl = mamba_impl
            self.mamba_uses_custom_autograd = False
            Mamba = None
            if mamba_impl in ['auto', 'ssm']:
                try:
                    from mamba_ssm import Mamba
                except ImportError:
                    if mamba_impl == 'ssm':
                        raise
                    Mamba = None

            if Mamba is not None:
                self.mamba = Mamba(
                    d_model=self.eff_out_dim,
                    d_state=mamba_d_state,
                    d_conv=mamba_d_conv,
                    expand=mamba_expand,
                )
                self.mamba_impl = 'ssm'
                self.mamba_uses_custom_autograd = True
            else:
                self.mamba = MambaBlock(
                    d_model=self.eff_out_dim,
                    d_state=mamba_d_state,
                    d_conv=mamba_d_conv,
                    expand=mamba_expand,
                )
                self.mamba_impl = 'torch'

            self.out_proj = nn.Linear(
                self.eff_out_dim * self.out_p_per_frame * self.n_groups, self.query_dim
            )
        else:
            raise ValueError('Unknown temporal_mixer: {}'.format(temporal_mixer))

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)
        if self.temporal_mixer == 'mamba':
            # zero-init residual path so the new module starts as identity w.r.t. query_feat
            nn.init.zeros_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)

    def inner_forward_linear(self, x, query):
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

    def inner_forward_mamba_before_mamba(self, x, query):
        B, Q, G, FP, C = x.shape
        F_ = self.num_frames
        P = self.num_points
        out_p = self.out_p_per_frame
        eff_in = self.eff_in_dim
        eff_out = self.eff_out_dim
        assert G == self.n_groups
        assert FP == F_ * P
        assert C == eff_in

        # frame 0 is current, F-1 is oldest in the sampled layout;
        # flip so Mamba scans past -> present and the last token is the current frame
        x_fp = x.reshape(B, Q, G, F_, P, eff_in)
        x_fp = torch.flip(x_fp, dims=[3])

        '''generate adaptive params (channel mix M and within-frame point mix S_p)'''
        params = self.parameter_generator(query)
        params = params.reshape(B*Q, G, -1)
        M, S_p = params.split([self.m_parameters, self.s_parameters], 2)
        M = M.reshape(B*Q, G, eff_in, eff_out)
        S_p = S_p.reshape(B*Q, G, out_p, P)

        '''within-frame adaptive point mix'''
        out = x_fp.reshape(B*Q, G, F_, P, eff_in)
        out = torch.matmul(S_p[:, :, None, :, :], out)  # [B*Q, G, F, out_p, eff_in]
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''adaptive channel mix'''
        out = torch.matmul(out, M[:, :, None, :, :])  # [B*Q, G, F, out_p, eff_out]
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''prepend query token along F as Mamba context'''
        q_tok = self.query_proj(query)                                  # [B, Q, G*eff_out]
        q_tok = q_tok.reshape(B*Q, G, 1, 1, eff_out)
        q_tok = q_tok.expand(B*Q, G, 1, out_p, eff_out)
        seq = torch.cat([q_tok, out], dim=2)                            # [B*Q, G, F+1, out_p, eff_out]

        '''causal Mamba scan over the temporal axis (per group, per spatial point)'''
        seq = seq.permute(0, 1, 3, 2, 4).contiguous()                   # [B*Q, G, out_p, F+1, eff_out]
        seq = seq.reshape(B*Q*G*out_p, F_ + 1, eff_out)
        return seq

    def inner_forward_mamba_after_mamba(self, seq, query):
        B, Q = query.shape[:2]
        G = self.n_groups
        out_p = self.out_p_per_frame
        eff_out = self.eff_out_dim

        '''take the last token (current frame, after history accumulation)'''
        last = seq[:, -1, :].reshape(B, Q, G * out_p * eff_out)

        out = self.out_proj(last)
        out = query + out

        return out

    def inner_forward_mamba(self, x, query):
        seq = self.inner_forward_mamba_before_mamba(x, query)
        seq = self.mamba(seq)
        return self.inner_forward_mamba_after_mamba(seq, query)

    def checkpointed_forward_mamba_ssm(self, x, query):
        seq = cp(
            self.inner_forward_mamba_before_mamba,
            x,
            query,
            use_reentrant=False,
        )
        seq = self.mamba(seq)
        return cp(
            self.inner_forward_mamba_after_mamba,
            seq,
            query,
            use_reentrant=False,
        )

    def inner_forward(self, x, query):
        if self.temporal_mixer == 'mamba':
            return self.inner_forward_mamba(x, query)
        return self.inner_forward_linear(x, query)

    def forward(self, x, query):
        if self.training and x.requires_grad:
            if self.temporal_mixer == 'mamba' and self.mamba_uses_custom_autograd:
                # Keep mamba_ssm itself outside checkpoint because its custom
                # backward conflicts with non-reentrant saved_tensors_hooks.
                # The surrounding pure PyTorch work is still checkpointed, and
                # no reentrant checkpoint touches the shared decoder params.
                return self.checkpointed_forward_mamba_ssm(x, query)
            return cp(self.inner_forward, x, query, use_reentrant=False)
        else:
            return self.inner_forward(x, query)
