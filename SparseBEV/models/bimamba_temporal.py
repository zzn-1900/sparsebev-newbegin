import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(time_diff, dim):
    """time_diff: [B, T] seconds (sign preserved). Returns [B, T, dim]."""
    B, T = time_diff.shape
    half = dim // 2
    device = time_diff.device
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = time_diff.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


def _load_mamba_cls():
    # Prefer Mamba v1: simpler internal layout, no nheads/headdim split,
    # so causal_conv1d's channel-last stride-alignment check always passes.
    # For T=8 the SSD speedup of Mamba2 is irrelevant anyway.
    try:
        from mamba_ssm import Mamba
        return Mamba, 'mamba1'
    except Exception:
        pass
    try:
        from mamba_ssm import Mamba2
        return Mamba2, 'mamba2'
    except Exception as e:
        raise RuntimeError(
            "mamba-ssm is required for use_bimamba_temporal=True. On 4090 run:\n"
            "  pip install causal-conv1d mamba-ssm"
        ) from e


def _build_mamba(d_model, d_state=16, d_conv=4, expand=2):
    cls, kind = _load_mamba_cls()
    if kind == 'mamba2':
        # Mamba2 builds zxbcdt of width (2*d_inner + 2*ngroups*d_state + nheads).
        # xBC is a channel-slice of zxbcdt, so its seq/batch stride equals the
        # full width — causal_conv1d requires that stride to be multiple of 8.
        # Enforce nheads % 8 == 0 by choosing headdim = d_inner // 8.
        d_inner = d_model * expand
        assert d_inner % 8 == 0, f"d_inner={d_inner} must be divisible by 8"
        headdim = max(d_inner // 8, 8)
        while d_inner % headdim != 0:
            headdim -= 1
        return cls(d_model=d_model, d_state=d_state, d_conv=d_conv,
                   expand=expand, headdim=headdim)
    return cls(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


class TemporalBiMamba(nn.Module):
    """Bidirectional Mamba over the temporal axis, inserted before AdaptiveMixing's
    channel mix. The residual gate is zero-initialized so the module is an identity
    at the start of training (lets the optimizer bring it in gradually)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.fwd = _build_mamba(d_model, d_state, d_conv, expand)
        self.bwd = _build_mamba(d_model, d_state, d_conv, expand)
        self.dt_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.zeros(1))
        self.d_model = d_model

    def forward(self, x, time_diff, qgp_shape):
        """
        x:         [N, T, C] with N = B * Q * G * P
        time_diff: [B, T]
        qgp_shape: (B, Q, G, P)
        """
        N, T, C = x.shape
        B, Q, G, P = qgp_shape
        assert N == B * Q * G * P, f"{N} != {B}*{Q}*{G}*{P}"

        dt = sinusoidal_time_embedding(time_diff, C)
        dt = self.dt_mlp(dt)
        dt = dt.view(B, 1, 1, 1, T, C).expand(B, Q, G, P, T, C).reshape(N, T, C)

        residual = x
        y = self.norm(x + dt.to(x.dtype)).contiguous()

        # Run Mamba in the ambient AMP dtype (fp16 under mmcv's Fp16OptimizerHook).
        # Mamba-ssm's CUDA kernels handle fp16/bf16 natively on sm_80+; forcing
        # fp32 here was doubling activation memory for no real benefit.
        out_fwd = self.fwd(y)
        out_bwd = self.bwd(torch.flip(y, dims=[1]).contiguous())
        out_bwd = torch.flip(out_bwd, dims=[1])
        out = out_fwd + out_bwd

        return residual + torch.tanh(self.gate) * out
