"""
Standalone diagnostic to confirm VPS / SOMCTS are wired and gradients flow.
Run from the SparseBEV directory in your training env (the one that has mmcv/mmdet/mmdet3d):

    python verify_vm.py

What it checks:
  1. The visual_prior_head sub-module exists on SparseBEV
  2. SparseBEVSampling has vps_fuse + vps_offset_delta + motion_branch
  3. sampling_offset stays at 256 input dim (baseline path); vps_offset_delta is the residual path
  4. Counts the extra parameters (should be < 1M)
  5. After model.init_weights() (the same call train.py makes), runs a synthetic
     forward + backward and checks gradients flow through the new modules.
     This is the regression test for the R1 zero-init deadlock.
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

# CRITICAL: trigger init_weights so the zero-init paths actually take effect,
# matching what train.py does. Without this, weights stay at PyTorch defaults
# and the deadlock test below would silently pass even when broken.
model.init_weights()
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
print(f'  sampling.vps_offset_delta         = {sampling.vps_offset_delta}')
print(f'  sampling.use_somcts               = {sampling.use_somcts}')
print(f'  sampling.motion_branch            = {sampling.motion_branch}')
print(f'  sampling.somcts_a_max             = {sampling.somcts_a_max}')
print(f'  sampling.somcts_omega_max         = {sampling.somcts_omega_max}')

print()
print('=' * 60)
print('3) Sampling-offset input dim — baseline path stays at embed_dims=256')
print('=' * 60)
print(f'  sampling.sampling_offset.in_features  = {sampling.sampling_offset.in_features}')
print(f'  sampling.sampling_offset.out_features = {sampling.sampling_offset.out_features}')
print(f'  expected (256)                         = 256')
print(f'  match                                  = {sampling.sampling_offset.in_features == 256}')
if sampling.vps_offset_delta is not None:
    print(f'  sampling.vps_offset_delta.in_features = {sampling.vps_offset_delta.in_features} (== vps_fuse_dim)')
print(f'  sampling.scale_weights.in_features    = {sampling.scale_weights.in_features}'
      f' (== embed_dims + vps_fuse_dim when VPS on)')

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
if sampling.vps_offset_delta is not None:
    p_vo = sum(p.numel() for p in sampling.vps_offset_delta.parameters())
    extra += p_vo
    print(f'  sampling.vps_offset_delta     : {p_vo:>9,} params')
if sampling.motion_branch is not None:
    p_mb = sum(p.numel() for p in sampling.motion_branch.parameters())
    extra += p_mb
    print(f'  sampling.motion_branch        : {p_mb:>9,} params')
# extra dim of scale_weights when VPS injects visual_prior into it
old_sw_in = 256
delta_sw = (sampling.scale_weights.in_features - old_sw_in) * sampling.scale_weights.out_features
if delta_sw > 0:
    extra += delta_sw
    print(f'  scale_weights (new dim only)  : {delta_sw:>9,} params')
total = sum(p.numel() for p in model.parameters())
print(f'  ----')
print(f'  extra total                   : {extra:>9,} params  ({100.0*extra/total:.3f}% of total)')
print(f'  model total                   : {total:>9,} params')

print()
print('=' * 60)
print('5) Forward pass + gradient sanity (synthetic mini-batch)')
print('    KEY REGRESSION TEST: vps_offset_delta + vps_fuse must have non-zero')
print('    grad after the FIRST iter, even with init_weights() applied.')
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
print('  gradient norms on new modules (must be > 0 — confirms R1 deadlock is broken):')
g_vph  = gn(vph.proj.weight) if vph is not None else None
g_fuse = gn(sampling.vps_fuse.weight) if sampling.vps_fuse is not None else None
g_off  = gn(sampling.vps_offset_delta.weight) if sampling.vps_offset_delta is not None else None
g_mot  = gn(sampling.motion_branch.weight) if sampling.motion_branch is not None else None
print(f'    visual_prior_head.proj.weight       grad-norm = {g_vph}')
print(f'    sampling.vps_fuse.weight            grad-norm = {g_fuse}')
print(f'    sampling.vps_offset_delta.weight    grad-norm = {g_off}')
print(f'    sampling.motion_branch.weight       grad-norm = {g_mot}')

# regression assertions
ok = True
for name, g in [('visual_prior_head', g_vph), ('vps_fuse', g_fuse), ('vps_offset_delta', g_off)]:
    if g is None or g <= 0.0:
        print(f'  [FAIL] {name} grad-norm = {g} — deadlock not broken!')
        ok = False
if ok:
    print('  [PASS] All new modules receive non-zero gradient at iter 0.')

print()
print('=' * 60)
print('Done.')
