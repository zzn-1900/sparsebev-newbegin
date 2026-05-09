"""
Standalone diagnostic to confirm VPS / SOMCTS are wired and active.
Run from the SparseBEV directory in your training env (the one that has mmcv/mmdet/mmdet3d):

    python verify_vm.py

What it checks:
  1. The visual_prior_head sub-module exists on SparseBEV
  2. SparseBEVSampling has vps_fuse + motion_branch
  3. sampling_offset.in_features grew from 256 to 256+vps_fuse_dim
  4. Counts the extra parameters (should be < 1M)
  5. Runs a synthetic forward and checks gradients flow through the new modules
"""
import sys
sys.path.insert(0, '.')

import torch
from mmcv import Config
from mmdet3d.models import build_detector

# also import the local plugin to register modules
import models  # noqa: F401  (your repo's models package)
import loaders  # noqa: F401

cfg = Config.fromfile('configs/r50_nuimg_704x256.py')

# build model on CPU; we only check structure & a tiny forward path
model = build_detector(cfg.model, train_cfg=cfg.model.get('train_cfg'),
                      test_cfg=cfg.model.get('test_cfg'))
model.eval()

print('=' * 60)
print('1) Module presence on SparseBEV')
print('=' * 60)
vph = getattr(model, 'visual_prior_head', None)
print(f'  model.visual_prior_head           = {type(vph).__name__ if vph else None}')
print(f'  model.vps_feat_level              = {getattr(model, "vps_feat_level", None)}')

sampling = model.pts_bbox_head.transformer.decoder.decoder_layer.sampling
print()
print('=' * 60)
print('2) SparseBEVSampling sub-modules')
print('=' * 60)
print(f'  sampling.use_vps                  = {sampling.use_vps}')
print(f'  sampling.vps_fuse                 = {sampling.vps_fuse}')
print(f'  sampling.use_somcts               = {sampling.use_somcts}')
print(f'  sampling.motion_branch            = {sampling.motion_branch}')

print()
print('=' * 60)
print('3) Sampling-offset input dim (should be 256+vps_fuse_dim with VPS)')
print('=' * 60)
print(f'  sampling.sampling_offset.in_features  = {sampling.sampling_offset.in_features}')
print(f'  sampling.sampling_offset.out_features = {sampling.sampling_offset.out_features}')
expected = 256 + (cfg.vps_fuse_dim if cfg.use_vps else 0)
print(f'  expected (256 + {cfg.vps_fuse_dim if cfg.use_vps else 0})            = {expected}')
print(f'  match                                  = {sampling.sampling_offset.in_features == expected}')

print()
print('=' * 60)
print('4) Extra parameter count (should be < 1M)')
print('=' * 60)
extra = 0
if vph is not None:
    p_vph = sum(p.numel() for p in vph.parameters())
    extra += p_vph
    print(f'  visual_prior_head             : {p_vph:>9,} params')
if sampling.vps_fuse is not None:
    p_vf = sum(p.numel() for p in sampling.vps_fuse.parameters())
    extra += p_vf
    print(f'  sampling.vps_fuse             : {p_vf:>9,} params')
if sampling.motion_branch is not None:
    p_mb = sum(p.numel() for p in sampling.motion_branch.parameters())
    extra += p_mb
    print(f'  sampling.motion_branch        : {p_mb:>9,} params')
# extra dim of sampling_offset: only the new portion
old_in = 256
new_in = sampling.sampling_offset.in_features
delta_so = (new_in - old_in) * sampling.sampling_offset.out_features
extra += delta_so
print(f'  sampling_offset (new dim only): {delta_so:>9,} params')
total = sum(p.numel() for p in model.parameters())
print(f'  ----')
print(f'  extra total                   : {extra:>9,} params  ({100.0*extra/total:.3f}% of total)')
print(f'  model total                   : {total:>9,} params')

print()
print('=' * 60)
print('5) Forward pass + gradient sanity (synthetic mini-batch)')
print('=' * 60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'  device = {device}')
model = model.to(device).train()

B, T, N, C, H, W = 1, 8, 6, 3, 256, 704
img = torch.randn(B, T * N, C, H, W, device=device)

# build minimal img_metas
import numpy as np
img_metas = [{
    'img_shape': [(H, W, 3)] * (T * N),
    'ori_shape': [(H, W, 3)] * (T * N),
    'pad_shape': [(H, W, 3)] * (T * N),
    'img_timestamp': [0.0 + 0.5 * (T - 1 - t) for t in range(T) for _ in range(N)],
    'lidar2img': [np.eye(4, dtype=np.float32) for _ in range(T * N)],
    'box_type_3d': None,
}]

with torch.no_grad():
    feats = model.extract_feat(img, img_metas)
prior_map = model.compute_visual_prior(feats)
print(f'  feats[-1] shape                = {tuple(feats[-1].shape)}')
print(f'  prior_map shape                = {tuple(prior_map.shape) if prior_map is not None else None}')
assert prior_map is not None and prior_map.shape[1] == 6, 'prior_map should be [B,6,K,H,W]'

# do a real forward through the head, then a dummy loss & backward
model.zero_grad()
outs = model.pts_bbox_head(feats, img_metas, prior_map=prior_map)
loss = outs['all_cls_scores'].abs().mean() + outs['all_bbox_preds'].abs().mean()
loss.backward()
print(f'  forward/backward OK, loss      = {loss.item():.4f}')

# check grad on new parameters
def gn(p):
    return None if p.grad is None else p.grad.norm().item()

print()
print('  gradient norms on new modules (should be > 0):')
print(f'    visual_prior_head.proj.weight       grad-norm = {gn(vph.proj.weight)}')
print(f'    sampling.vps_fuse.weight            grad-norm = {gn(sampling.vps_fuse.weight)}')
print(f'    sampling.motion_branch.weight       grad-norm = {gn(sampling.motion_branch.weight)}')

print()
print('=' * 60)
print('Done. If all 5 sections are sane, VM-SparseBEV is fully wired.')
