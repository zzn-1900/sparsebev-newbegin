import os
import torch
import torch.nn as nn
import numpy as np
from mmcv.runner import get_dist_info
from .csrc.wrapper import MSMV_CUDA
from .sparsebev_sampling import sampling_4d


class QuerySelector(nn.Module):
    def __init__(self,
                 grid_size=30,
                 num_groups=4,
                 topk=500,
                 gaussian_radius=1,
                 pc_range=None,
                 detach_feats=True,
                 loss_weight=1.0,
                 vis_interval=0,
                 vis_dir='outputs/query_selector/vis'):
        super(QuerySelector, self).__init__()
        self.grid_size = grid_size
        self.num_query = grid_size * grid_size
        self.num_groups = num_groups
        self.topk = topk
        self.gaussian_radius = gaussian_radius
        self.pc_range = pc_range
        self.detach_feats = detach_feats
        self.loss_weight = loss_weight
        self.vis_interval = vis_interval
        self.vis_dir = vis_dir

        self.selector = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.register_buffer('step_count', torch.zeros((), dtype=torch.long), persistent=False)

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
            feat = feat.detach() if self.detach_feats else feat
            B, TN, GC, H, W = feat.shape
            N, T, G, C = 6, TN // 6, self.num_groups, GC // self.num_groups
            feat = feat[:, :6]
            T = 1
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

    def predict_logits(self, mlvl_feats, img_metas):
        bev_feats = self.sample_bev_features(mlvl_feats, img_metas)
        return self.selector(bev_feats).reshape(-1, 1, self.grid_size, self.grid_size)

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
        return loss * self.loss_weight

    @torch.no_grad()
    def select_topk(self, pred_logits):
        scores = pred_logits.sigmoid().flatten(1)
        _, topk_inds = scores.topk(min(self.topk, scores.shape[1]), dim=1)
        return topk_inds.detach()

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
        axes[2].set_title('Top%d queries' % self.topk)

        for ax in axes:
            ax.set_xlim(-0.5, self.grid_size - 0.5)
            ax.set_ylim(-0.5, self.grid_size - 0.5)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        fig.savefig(os.path.join(self.vis_dir, 'step_%06d.png' % step), dpi=150)
        plt.close(fig)

    def forward_train(self, mlvl_feats, img_metas, gt_bboxes_3d):
        logits = self.predict_logits(mlvl_feats, img_metas)
        selected_query_indices = self.select_topk(logits)
        targets = self.make_heatmap_targets(gt_bboxes_3d, logits.device)
        loss_selector = self.gaussian_focal_loss(logits, targets)

        self.step_count += 1
        step = int(self.step_count.item())
        if self.vis_interval > 0 and step % self.vis_interval == 0:
            self.save_visualization(logits, targets, step)

        losses = {
            'loss_query_selector': loss_selector,
            'query_selector_recall_top100': self.calc_recall(logits, targets, 100),
            'query_selector_recall_top300': self.calc_recall(logits, targets, 300),
            'query_selector_recall_top500': self.calc_recall(logits, targets, self.topk),
        }
        return selected_query_indices, losses

    def forward_test(self, mlvl_feats, img_metas):
        logits = self.predict_logits(mlvl_feats, img_metas)
        return self.select_topk(logits)
