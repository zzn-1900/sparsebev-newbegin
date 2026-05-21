import os
import torch
import torch.nn as nn
import numpy as np
from mmcv.runner import auto_fp16, force_fp32, get_dist_info
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from .csrc.wrapper import MSMV_CUDA
from .sparsebev_sampling import sampling_4d
from .utils import GridMask, GpuPhotoMetricDistortion, pad_multiple


@DETECTORS.register_module()
class QuerySelectorPretrain(MVXTwoStageDetector):
    def __init__(self,
                 data_aug=None,
                 freeze_img_backbone=True,
                 grid_size=30,
                 num_groups=4,
                 topk=500,
                 gaussian_radius=1,
                 pc_range=None,
                 vis_interval=50,
                 vis_dir='outputs/query_selector_pretrain/vis',
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(QuerySelectorPretrain, self).__init__(
            pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder,
            pts_fusion_layer, img_backbone, pts_backbone, img_neck, pts_neck,
            pts_bbox_head, img_roi_head, img_rpn_head, train_cfg, test_cfg,
            pretrained)

        self.data_aug = data_aug
        self.freeze_img_backbone = freeze_img_backbone
        self.grid_size = grid_size
        self.num_query = grid_size * grid_size
        self.num_groups = num_groups
        self.topk = topk
        self.gaussian_radius = gaussian_radius
        self.pc_range = pc_range
        self.vis_interval = vis_interval
        self.vis_dir = vis_dir

        self.fp16_enabled = False
        self.color_aug = GpuPhotoMetricDistortion()
        self.grid_mask = GridMask(ratio=0.5, prob=0.7)
        self.use_grid_mask = True

        self.selector = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self.register_buffer('step_count', torch.zeros((), dtype=torch.long), persistent=False)

        if self.freeze_img_backbone:
            self._freeze_img_backbone()

    def _freeze_img_backbone(self):
        if self.img_backbone is None:
            return
        self.img_backbone.eval()
        for param in self.img_backbone.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super(QuerySelectorPretrain, self).train(mode)
        if self.freeze_img_backbone:
            self._freeze_img_backbone()
        return self

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img):
        if self.use_grid_mask:
            img = self.grid_mask(img)

        if self.freeze_img_backbone:
            self.img_backbone.eval()
            with torch.no_grad():
                img_feats = self.img_backbone(img)
        else:
            img_feats = self.img_backbone(img)

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats

    def extract_feat(self, img, img_metas):
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        assert img.dim() == 5

        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W).float()

        if self.data_aug is not None:
            if self.data_aug.get('img_color_aug', False) and self.training:
                img = self.color_aug(img)

            if 'img_norm_cfg' in self.data_aug:
                img_norm_cfg = self.data_aug['img_norm_cfg']
                norm_mean = torch.tensor(img_norm_cfg['mean'], device=img.device)
                norm_std = torch.tensor(img_norm_cfg['std'], device=img.device)

                if img_norm_cfg['to_rgb']:
                    img = img[:, [2, 1, 0], :, :]

                img = img - norm_mean.reshape(1, 3, 1, 1)
                img = img / norm_std.reshape(1, 3, 1, 1)

            for b in range(B):
                img_shape = (img.shape[2], img.shape[3], img.shape[1])
                img_metas[b]['img_shape'] = [img_shape for _ in range(N)]
                img_metas[b]['ori_shape'] = [img_shape for _ in range(N)]

            if 'img_pad_cfg' in self.data_aug:
                img_pad_cfg = self.data_aug['img_pad_cfg']
                img = pad_multiple(img, img_metas, size_divisor=img_pad_cfg['size_divisor'])

        input_shape = img.shape[-2:]
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        img_feats = self.extract_img_feat(img)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    def _make_grid_points(self, batch_size, device):
        grid = torch.arange(self.grid_size, device=device)
        xx, yy = torch.meshgrid(grid, grid, indexing='ij')
        xy = (torch.stack([xx, yy], dim=-1).float() + 0.5) / self.grid_size

        x = xy[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        y = xy[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        z = torch.full_like(x, self.pc_range[2])
        points = torch.stack([x, y, z], dim=-1).reshape(1, self.num_query, 3)
        return points.repeat(batch_size, 1, 1)

    def _prepare_sampling_feats(self, mlvl_feats):
        sampling_feats = []
        for feat in mlvl_feats:
            B, TN, GC, H, W = feat.shape
            N, T, G, C = 6, TN // 6, self.num_groups, GC // self.num_groups
            assert T == 1, 'Query selector pretraining only supports current-frame inputs.'
            feat = feat.reshape(B, T, N, G, C, H, W)

            if MSMV_CUDA:
                feat = feat.permute(0, 1, 3, 2, 5, 6, 4)
                feat = feat.reshape(B * T * G, N, H, W, C)
            else:
                feat = feat.permute(0, 1, 3, 4, 2, 5, 6)
                feat = feat.reshape(B * T * G, C, N, H, W)

            sampling_feats.append(feat.contiguous())
        return sampling_feats

    def sample_bev_features(self, mlvl_feats, img_metas):
        B = mlvl_feats[0].shape[0]
        device = mlvl_feats[0].device
        image_h, image_w, _ = img_metas[0]['img_shape'][0]

        sample_points = self._make_grid_points(B, device)
        sample_points = sample_points.reshape(B, self.num_query, 1, 1, 1, 3)
        sample_points = sample_points.expand(B, self.num_query, 1, self.num_groups, 1, 3)

        num_levels = len(mlvl_feats)
        scale_weights = mlvl_feats[0].new_full(
            (B, self.num_query, self.num_groups, 1, 1, num_levels),
            1.0 / num_levels)

        lidar2img = np.asarray([m['lidar2img'][:6] for m in img_metas]).astype(np.float32)
        lidar2img = torch.from_numpy(lidar2img).to(device)

        sampled_feats = sampling_4d(
            sample_points,
            self._prepare_sampling_feats(mlvl_feats),
            scale_weights,
            lidar2img,
            image_h,
            image_w
        )
        return sampled_feats.reshape(B, self.num_query, -1)

    def make_heatmap_targets(self, gt_bboxes_3d, device):
        targets = torch.zeros(
            (len(gt_bboxes_3d), 1, self.grid_size, self.grid_size),
            dtype=torch.float32,
            device=device)

        radius = self.gaussian_radius
        diameter = 2 * radius + 1
        sigma = diameter / 6.0

        offsets = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                value = np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
                offsets.append((dx, dy, float(value)))

        for b, gt_bboxes in enumerate(gt_bboxes_3d):
            centers = gt_bboxes.gravity_center.to(device)
            if centers.numel() == 0:
                continue

            gx = ((centers[:, 0] - self.pc_range[0]) /
                  (self.pc_range[3] - self.pc_range[0]) * self.grid_size).long()
            gy = ((centers[:, 1] - self.pc_range[1]) /
                  (self.pc_range[4] - self.pc_range[1]) * self.grid_size).long()

            valid = (gx >= 0) & (gx < self.grid_size) & (gy >= 0) & (gy < self.grid_size)
            gx, gy = gx[valid], gy[valid]

            for x, y in zip(gx, gy):
                for dx, dy, value in offsets:
                    px = int(x) + dx
                    py = int(y) + dy
                    if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
                        value = targets.new_tensor(value)
                        targets[b, 0, px, py] = torch.maximum(targets[b, 0, px, py], value)

        return targets

    def gaussian_focal_loss(self, pred_logits, targets, alpha=2.0, beta=4.0, eps=1e-6):
        pred = pred_logits.sigmoid().clamp(min=eps, max=1.0 - eps)
        pos_inds = targets.eq(1.0).float()
        neg_inds = targets.lt(1.0).float()
        neg_weights = torch.pow(1.0 - targets, beta)

        pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos_inds
        neg_loss = -torch.log(1.0 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

        num_pos = pos_inds.sum()
        loss = (pos_loss.sum() + neg_loss.sum()) / torch.clamp(num_pos, min=1.0)
        return loss

    @torch.no_grad()
    def calc_recall(self, pred_logits, targets, k):
        pred = pred_logits.sigmoid().flatten(1)
        pos = targets.flatten(1).eq(1.0)
        k = min(k, pred.shape[1])
        topk_inds = pred.topk(k, dim=1).indices
        selected = torch.zeros_like(pos)
        selected.scatter_(1, topk_inds, True)
        num_pos = pos.sum()
        if num_pos == 0:
            return pred.new_tensor(0.0)
        return (selected & pos).sum().float() / num_pos.float()

    @torch.no_grad()
    def save_visualization(self, pred_logits, targets, step):
        rank, _ = get_dist_info()
        if rank != 0:
            return

        os.makedirs(self.vis_dir, exist_ok=True)

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        gt = targets[0, 0].detach().float().cpu().numpy()
        pred = pred_logits[0, 0].sigmoid().detach().float().cpu().numpy()
        topk = pred.reshape(-1).argsort()[-self.topk:]
        topk_x = topk // self.grid_size
        topk_y = topk % self.grid_size
        gt_x, gt_y = np.where(gt == 1.0)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(gt.T, origin='lower', vmin=0, vmax=1, cmap='hot')
        axes[0].set_title('GT heatmap')

        axes[1].imshow(pred.T, origin='lower', vmin=0, vmax=1, cmap='viridis')
        axes[1].set_title('Pred heatmap')

        axes[2].imshow(pred.T, origin='lower', vmin=0, vmax=1, cmap='viridis')
        axes[2].scatter(topk_x, topk_y, s=4, c='white', alpha=0.55, linewidths=0)
        if len(gt_x) > 0:
            axes[2].scatter(gt_x, gt_y, s=28, facecolors='none', edgecolors='red', linewidths=1.2)
        axes[2].set_title('Top500 queries')

        for ax in axes:
            ax.set_xlim(-0.5, self.grid_size - 0.5)
            ax.set_ylim(-0.5, self.grid_size - 0.5)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        fig.savefig(os.path.join(self.vis_dir, 'step_%06d.png' % step), dpi=150)
        plt.close(fig)

    @force_fp32(apply_to=('img',))
    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        return self.forward_test(**kwargs)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None):
        mlvl_feats = self.extract_feat(img, img_metas)
        bev_feats = self.sample_bev_features(mlvl_feats, img_metas)
        logits = self.selector(bev_feats).reshape(-1, 1, self.grid_size, self.grid_size)
        targets = self.make_heatmap_targets(gt_bboxes_3d, logits.device)

        loss_selector = self.gaussian_focal_loss(logits, targets)

        self.step_count += 1
        step = int(self.step_count.item())
        if self.vis_interval > 0 and step % self.vis_interval == 0:
            self.save_visualization(logits, targets, step)

        return {
            'loss_selector': loss_selector,
            'recall_top100': self.calc_recall(logits, targets, 100),
            'recall_top300': self.calc_recall(logits, targets, 300),
            'recall_top500': self.calc_recall(logits, targets, self.topk),
        }

    def forward_test(self, img_metas, img=None, **kwargs):
        img = img[0] if isinstance(img, list) else img
        img_metas = img_metas[0] if len(img_metas) > 0 and isinstance(img_metas[0], list) else img_metas
        mlvl_feats = self.extract_feat(img, img_metas)
        bev_feats = self.sample_bev_features(mlvl_feats, img_metas)
        logits = self.selector(bev_feats).reshape(-1, 1, self.grid_size, self.grid_size)
        scores = logits.sigmoid().flatten(1)
        topk_scores, topk_inds = scores.topk(min(self.topk, scores.shape[1]), dim=1)
        return [dict(scores=s, indices=i) for s, i in zip(topk_scores, topk_inds)]
