import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule


class VisualPriorHead(BaseModule):
    """
    Lightweight head producing a low-resolution visual-prior descriptor map
    from the **current frame** of a single FPN level.

    Input:  feat  [B, N=6, C, H, W]   (current frame, 6 cameras)
    Output: prior [B, N=6, K, H', W'] (prior descriptor map, K << C)
    """
    def __init__(self,
                 in_channels=256,
                 prior_channels=32,
                 extra_pool=2,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.prior_channels = prior_channels
        self.extra_pool = extra_pool

        self.proj = nn.Conv2d(in_channels, prior_channels, kernel_size=1, bias=True)
        self.norm = nn.GroupNorm(num_groups=min(8, prior_channels), num_channels=prior_channels)
        self.act = nn.ReLU(inplace=True)

    @torch.no_grad()
    def init_weights(self):
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, feat):
        """
        feat: [B, N, C, H, W]  (current frame only, N=6 cameras)
        """
        B, N, C, H, W = feat.shape
        x = feat.reshape(B * N, C, H, W)
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        if self.extra_pool > 1:
            x = F.avg_pool2d(x, kernel_size=self.extra_pool, stride=self.extra_pool)
        K, H2, W2 = x.shape[1], x.shape[2], x.shape[3]
        return x.reshape(B, N, K, H2, W2)
