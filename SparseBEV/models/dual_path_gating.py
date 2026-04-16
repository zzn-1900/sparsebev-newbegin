import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPathReliabilityGating(nn.Module):
    """
    Dual-Path Reliability Gating (DPRG) v2.2

    Inserted between Channel Mixing and Point Mixing in AdaptiveMixing.
    Runs on flattened batch B' = B*Q*G.

    Steps:
      1. Project full query (D=256) → group subspace (C=64) via learned W_q_proj
      2. Query-guided pooling of current-frame points → f_cur
      3. Estimate β from (c1, c2)
      4. Dual-path gate:
           Path Q:  gate_q[k] = σ(W1_q(q_full)ᵀ W2(fₖ) / √r + b)   full 256-dim query
           Path C:  gate_c[k] = σ(W1_c(f_cur)ᵀ  W2(fₖ) / √r + b)   64-dim current frame
           gate[k]  = (1-β)·gate_q[k] + β·gate_c[k]
      5. Apply: f̃ₖ = gate[k] · fₖ

    Args:
        embed_dim: Feature dim per group (= in_dim // n_groups, default 64)
        query_dim: Full query dim       (= in_dim,              default 256)
        num_frames: Temporal frames T   (default 8)
        num_points_per_frame: Points per group per frame S (default 4)
        gate_rank: Low-rank projection r (default 16)
        current_frame_idx: Current frame index in T-axis (default 0)
    """

    def __init__(
        self,
        embed_dim=64,
        query_dim=256,
        num_frames=8,
        num_points_per_frame=4,
        gate_rank=16,
        current_frame_idx=0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_dim = query_dim
        self.T = num_frames
        self.S = num_points_per_frame
        self.P = num_frames * num_points_per_frame
        self.r = gate_rank
        self.current_frame_idx = current_frame_idx
        self.scale_attn = embed_dim ** -0.5   # Step 2: attention in C-dim space
        self.scale_gate = gate_rank ** -0.5   # Step 4: gate logits in r-dim space
        self.beta_force_zero = False

        # Step 1: learned projection, full query → group subspace
        # Replaces the arbitrary reshape(B*Q*G, 64) that split query by position
        self.W_q_proj = nn.Linear(query_dim, embed_dim, bias=False)

        # Step 2: query-guided pooling
        self.W_a = nn.Linear(embed_dim, embed_dim, bias=False)

        # Step 3: β predictor
        self.beta_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

        # Step 4: separate projections for each path
        self.W1_q = nn.Linear(query_dim, gate_rank, bias=False)   # full query → r
        self.W1_c = nn.Linear(embed_dim, gate_rank, bias=False)   # f_cur      → r
        self.W2   = nn.Linear(embed_dim, gate_rank, bias=False)   # each point → r (shared)
        self.gate_bias = nn.Parameter(torch.ones(1))              # init=1.0 → σ≈0.73

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_q_proj.weight)
        nn.init.normal_(self.W_a.weight, std=0.02)
        nn.init.xavier_uniform_(self.beta_mlp[0].weight)
        nn.init.zeros_(self.beta_mlp[0].bias)
        nn.init.zeros_(self.beta_mlp[2].weight)   # zero-init → β=0.5 at start
        nn.init.zeros_(self.beta_mlp[2].bias)
        nn.init.xavier_uniform_(self.W1_q.weight)
        nn.init.xavier_uniform_(self.W1_c.weight)
        nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, mixed_feats, query_full):
        """
        Args:
            mixed_feats: (B', P, C)  channel-mixed features,  C = embed_dim = 64
            query_full:  (B', D)     full query,               D = query_dim = 256

        Returns:
            gated_feats: (B', P, C)
        """
        Bp, P, C = mixed_feats.shape
        S = self.S
        assert P == self.P, f"DPRG expected P={self.P}, got {P}"

        # ── Step 1: project full query into group subspace ────────────────
        query_group = self.W_q_proj(query_full)                   # (B', C)

        # ── current-frame points ──────────────────────────────────────────
        cur_start = self.current_frame_idx * S
        cur_feats = mixed_feats[:, cur_start:cur_start + S, :]    # (B', S, C)

        # ── Step 2: Query-Guided Pooling → f_cur ─────────────────────────
        cur_proj    = self.W_a(cur_feats)                          # (B', S, C)
        attn_logits = (query_group.unsqueeze(1) * cur_proj).sum(-1) * self.scale_attn
        #                                                            (B', S)
        alpha  = F.softmax(attn_logits, dim=-1).unsqueeze(-1)      # (B', S, 1)
        f_cur  = (alpha * cur_feats).sum(dim=1)                    # (B', C)

        # ── Step 3: Estimate β ────────────────────────────────────────────
        cur_normed = F.normalize(cur_feats, dim=-1, eps=1e-6)      # (B', S, C)
        cur_mean   = cur_normed.mean(dim=1, keepdim=True)           # (B', 1, C)
        c1 = (cur_normed * cur_mean).sum(-1).mean(dim=-1, keepdim=True)   # (B', 1)

        c2 = F.cosine_similarity(query_group, f_cur, dim=-1, eps=1e-6).unsqueeze(-1)
        #                                                                   (B', 1)

        beta = torch.sigmoid(self.beta_mlp(torch.cat([c1, c2], dim=-1)))  # (B', 1)
        if self.beta_force_zero:
            beta = torch.zeros_like(beta)

        # ── Step 4: Dual-path gating ──────────────────────────────────────
        v = self.W2(mixed_feats)                                   # (B', P, r)

        u_q    = self.W1_q(query_full).unsqueeze(1)                # (B', 1, r)
        gate_q = torch.sigmoid(
            (u_q * v).sum(-1) * self.scale_gate + self.gate_bias
        )                                                          # (B', P)

        u_c    = self.W1_c(f_cur).unsqueeze(1)                     # (B', 1, r)
        gate_c = torch.sigmoid(
            (u_c * v).sum(-1) * self.scale_gate + self.gate_bias
        )                                                          # (B', P)

        gate = (1 - beta) * gate_q + beta * gate_c                # (B', P)

        return mixed_feats * gate.unsqueeze(-1)                    # (B', P, C)
