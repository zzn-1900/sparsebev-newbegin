import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def normalize_query_logits(cls_score):
    cls_prob = torch.sigmoid(cls_score)
    cls_prob = cls_prob / cls_prob.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return cls_prob


def _all_gather_tensor(tensor):
    if tensor is None or tensor.numel() == 0:
        return tensor
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    local_size = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
    size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    sizes = [int(size.item()) for size in size_list]
    max_size = max(sizes)

    padded_shape = [max_size] + list(tensor.shape[1:])
    padded = tensor.new_zeros(padded_shape)
    if tensor.shape[0] > 0:
        padded[:tensor.shape[0]] = tensor

    gathered = [tensor.new_zeros(padded_shape) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    outputs = [chunk[:size] for chunk, size in zip(gathered, sizes) if size > 0]
    if len(outputs) == 0:
        return tensor.new_zeros((0,) + tuple(tensor.shape[1:]))
    return torch.cat(outputs, dim=0)


def _select_weighted_diverse_indices(feats, weights, num_select):
    if num_select <= 0 or feats.numel() == 0:
        return torch.zeros(0, device=feats.device, dtype=torch.long)
    if feats.shape[0] <= num_select:
        return torch.arange(feats.shape[0], device=feats.device, dtype=torch.long)

    weight_score = weights - weights.min()
    weight_score = weight_score / weight_score.max().clamp(min=1e-6)

    first_idx = int(weight_score.argmax().item())
    selected = [first_idx]
    max_similarity = torch.matmul(feats, feats[first_idx])

    while len(selected) < num_select:
        score = (1.0 - max_similarity) + 0.1 * weight_score
        score[selected] = -1e6
        next_idx = int(score.argmax().item())
        selected.append(next_idx)
        max_similarity = torch.maximum(max_similarity, torch.matmul(feats, feats[next_idx]))

    return torch.tensor(selected, device=feats.device, dtype=torch.long)


def _select_weighted_medoid(feats, weights):
    if feats.shape[0] <= 1:
        return 0
    similarity = torch.matmul(feats, feats.transpose(0, 1))
    medoid_score = torch.matmul(similarity, weights)
    return int(medoid_score.argmax().item())


class QueryPrototypeBank(nn.Module):
    def __init__(self,
                 num_classes,
                 embed_dims,
                 num_prototypes=8,
                 memory_size_per_class=100,
                 momentum=0.99,
                 recent_buffer_size=None,
                 query_bank_size=None,
                 online_match_threshold=0.75,
                 query_merge_threshold=None,
                 init_match_threshold=0.55,
                 query_new_threshold=None,
                 maintenance_interval=64,
                 alpha_min=0.05,
                 alpha_max=0.20,
                 quality_gamma=0.10,
                 radius_gamma=0.10,
                 recent_score_threshold=0.20,
                 query_score_threshold=None,
                 recent_dedup_threshold=0.95,
                 query_dedup_threshold=None,
                 radius_refresh_threshold=0.30,
                 query_replace_threshold=None):
        super(QueryPrototypeBank, self).__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_prototypes = num_prototypes
        self.memory_size_per_class = memory_size_per_class
        if query_bank_size is None:
            query_bank_size = recent_buffer_size
        if query_merge_threshold is None:
            query_merge_threshold = online_match_threshold
        if query_new_threshold is None:
            query_new_threshold = init_match_threshold
        if query_score_threshold is None:
            query_score_threshold = recent_score_threshold
        if query_dedup_threshold is None:
            query_dedup_threshold = recent_dedup_threshold
        if query_replace_threshold is None:
            query_replace_threshold = radius_refresh_threshold

        self.query_bank_size = memory_size_per_class if query_bank_size is None else max(1, int(query_bank_size))
        self.momentum = momentum
        self.query_merge_threshold = float(query_merge_threshold)
        self.query_new_threshold = float(query_new_threshold)
        self.maintenance_interval = max(1, int(maintenance_interval))
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.quality_gamma = float(quality_gamma)
        self.radius_gamma = float(radius_gamma)
        self.query_score_threshold = float(query_score_threshold)
        self.query_dedup_threshold = float(query_dedup_threshold)
        self.query_replace_threshold = float(query_replace_threshold)

        self.register_buffer('prototype_bank', torch.zeros(num_classes, num_prototypes, embed_dims))
        self.register_buffer('prototype_count', torch.zeros(num_classes, num_prototypes))
        self.register_buffer('prototype_quality', torch.zeros(num_classes, num_prototypes))
        self.register_buffer('prototype_radius', torch.zeros(num_classes, num_prototypes))
        self.register_buffer('prototype_age', torch.zeros(num_classes, num_prototypes))
        self.register_buffer('prototype_updates', torch.zeros(1, dtype=torch.long))

        # Per-class current query pool. Slots are refreshed from these tokens instead
        # of from a historical recent buffer.
        self.register_buffer('bank_query_feats', torch.zeros(num_classes, self.query_bank_size, embed_dims))
        self.register_buffer('bank_query_support', torch.zeros(num_classes, self.query_bank_size))
        self.register_buffer('bank_query_quality', torch.zeros(num_classes, self.query_bank_size))
        self.register_buffer('bank_query_score', torch.full((num_classes, self.query_bank_size), -1e6))
        self.register_buffer('bank_query_valid', torch.zeros(num_classes, self.query_bank_size, dtype=torch.bool))

    def get_valid_mask(self, min_count=0):
        return self.prototype_count >= float(min_count)

    def get_normalized_bank(self):
        return F.normalize(self.prototype_bank, dim=-1, eps=1e-6)

    def match_slots(self, feats, labels, min_count=0):
        if feats is None or labels is None or feats.numel() == 0 or labels.numel() == 0:
            device = self.prototype_bank.device
            return (
                self.prototype_bank.new_zeros((0, self.embed_dims)),
                torch.zeros(0, device=device, dtype=torch.bool),
                torch.zeros(0, device=device, dtype=torch.long),
            )

        feats = F.normalize(feats.float(), dim=-1, eps=1e-6)
        bank = self.get_normalized_bank().float()
        valid_mask = self.get_valid_mask(min_count)

        matched_bank = feats.new_zeros(feats.shape)
        matched_mask = labels.new_zeros(labels.shape, dtype=torch.bool)
        matched_slots = labels.new_full(labels.shape, -1)

        for cls_idx in labels.unique().tolist():
            if cls_idx < 0 or cls_idx >= self.num_classes:
                continue

            cls_sample_inds = torch.nonzero(labels == cls_idx, as_tuple=False).flatten()
            cls_valid_slots = torch.nonzero(valid_mask[cls_idx], as_tuple=False).flatten()
            if cls_sample_inds.numel() == 0 or cls_valid_slots.numel() == 0:
                continue

            cls_feats = feats[cls_sample_inds]
            cls_bank = bank[cls_idx, cls_valid_slots]
            cls_sim = torch.matmul(cls_feats, cls_bank.transpose(0, 1))
            cls_slot_ids = cls_sim.argmax(dim=-1)

            matched_bank[cls_sample_inds] = cls_bank[cls_slot_ids]
            matched_mask[cls_sample_inds] = True
            matched_slots[cls_sample_inds] = cls_valid_slots[cls_slot_ids]

        return matched_bank, matched_mask, matched_slots

    @torch.no_grad()
    def update(self, feats, labels, qualities):
        if feats is None or labels is None or qualities is None:
            return
        if feats.numel() == 0 or labels.numel() == 0 or qualities.numel() == 0:
            return

        feats = _all_gather_tensor(feats.detach().float())
        labels = _all_gather_tensor(labels.detach().long())
        qualities = _all_gather_tensor(qualities.detach().float())
        if feats is None or labels is None or qualities is None:
            return
        if feats.numel() == 0 or labels.numel() == 0 or qualities.numel() == 0:
            return

        feats = F.normalize(feats, dim=-1, eps=1e-6)
        touched_classes = []

        valid_slots = self.prototype_count > 0
        self.prototype_age[valid_slots] += 1.0

        for cls_idx in labels.unique().tolist():
            if cls_idx < 0 or cls_idx >= self.num_classes:
                continue

            cls_mask = labels == cls_idx
            if not torch.any(cls_mask):
                continue

            touched_classes.append(cls_idx)
            self._update_class_query_bank(
                cls_idx,
                feats[cls_mask],
                qualities[cls_mask],
            )

        for cls_idx in touched_classes:
            self._refresh_class_slots(cls_idx)

        self.prototype_updates += 1

    def _quality_to_weight(self, quality):
        quality = quality.float()
        return 0.1 + 0.9 * torch.sigmoid(quality)

    @torch.no_grad()
    def _update_class_query_bank(self, cls_idx, feats, qualities):
        order = qualities.argsort(descending=True)
        for sample_idx in order.tolist():
            feat = feats[sample_idx]
            quality_weight = self._quality_to_weight(qualities[sample_idx]).to(feat.dtype)
            self._update_query_bank_single(cls_idx, feat, quality_weight)

    @torch.no_grad()
    def _update_query_bank_single(self, cls_idx, feat, quality_weight):
        valid_mask = self.bank_query_valid[cls_idx]
        valid_inds = torch.nonzero(valid_mask, as_tuple=False).flatten()
        candidate_score = self._compute_candidate_score(cls_idx, feat, quality_weight)
        if candidate_score.item() < self.query_score_threshold:
            return

        if valid_inds.numel() == 0:
            self._add_query_exemplar(cls_idx, feat, quality_weight, candidate_score)
            return

        exemplars = F.normalize(self.bank_query_feats[cls_idx, valid_inds], dim=-1, eps=1e-6)
        similarity = torch.matmul(exemplars, feat)
        best_local = int(similarity.argmax().item())
        best_idx = int(valid_inds[best_local].item())
        best_sim = similarity[best_local].clamp(min=-1.0, max=1.0).item()

        if best_sim >= self.query_dedup_threshold:
            self._merge_query_exemplar(cls_idx, best_idx, feat, quality_weight)
            return

        if best_sim >= self.query_merge_threshold:
            self._merge_query_exemplar(cls_idx, best_idx, feat, quality_weight)
            return

        if valid_inds.numel() < self.query_bank_size and best_sim < self.query_new_threshold:
            self._add_query_exemplar(cls_idx, feat, quality_weight, candidate_score)
            return

        if valid_inds.numel() < self.query_bank_size:
            self._merge_query_exemplar(cls_idx, best_idx, feat, quality_weight)
            return

        self._replace_query_exemplar(cls_idx, feat, quality_weight, candidate_score)

    def _compute_candidate_score(self, cls_idx, feat, quality_weight):
        valid_mask = self.bank_query_valid[cls_idx]
        if not torch.any(valid_mask):
            novelty = feat.new_tensor(1.0)
        else:
            exemplars = F.normalize(self.bank_query_feats[cls_idx][valid_mask], dim=-1, eps=1e-6)
            best_sim = torch.matmul(exemplars, feat).max().clamp(min=-1.0, max=1.0)
            novelty = 1.0 - best_sim
        return 0.5 * quality_weight + 0.5 * novelty

    def _compute_bank_query_score(self, support, quality):
        return torch.sqrt(support.clamp(min=1.0)) * (0.5 + quality)

    @torch.no_grad()
    def _add_query_exemplar(self, cls_idx, feat, quality_weight, candidate_score):
        empty_inds = torch.nonzero(~self.bank_query_valid[cls_idx], as_tuple=False).flatten()
        if empty_inds.numel() == 0:
            return False

        slot_idx = int(empty_inds[0].item())
        self.bank_query_feats[cls_idx, slot_idx] = F.normalize(feat, dim=0, eps=1e-6)
        self.bank_query_support[cls_idx, slot_idx] = 1.0
        self.bank_query_quality[cls_idx, slot_idx] = quality_weight
        self.bank_query_score[cls_idx, slot_idx] = candidate_score
        self.bank_query_valid[cls_idx, slot_idx] = True
        return True

    @torch.no_grad()
    def _merge_query_exemplar(self, cls_idx, exemplar_idx, feat, quality_weight):
        old_feat = self.bank_query_feats[cls_idx, exemplar_idx]
        old_support = self.bank_query_support[cls_idx, exemplar_idx]
        old_quality = self.bank_query_quality[cls_idx, exemplar_idx]

        base_alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * quality_weight
        base_alpha = base_alpha / torch.sqrt(old_support + 1.0)
        alpha_floor = 0.0
        if old_support.item() <= 1.0:
            alpha_floor = self.alpha_min
        elif old_support.item() <= 4.0:
            alpha_floor = max(1.0 - self.momentum, 0.01)
        alpha = base_alpha.clamp(min=alpha_floor, max=self.alpha_max)

        new_feat = old_feat * (1.0 - alpha) + feat * alpha
        new_feat = F.normalize(new_feat, dim=0, eps=1e-6)
        new_support = old_support + 1.0
        new_quality = (
            (1.0 - self.quality_gamma) * old_quality + self.quality_gamma * quality_weight
        )
        self.bank_query_feats[cls_idx, exemplar_idx] = new_feat
        self.bank_query_support[cls_idx, exemplar_idx] = new_support
        self.bank_query_quality[cls_idx, exemplar_idx] = new_quality
        self.bank_query_score[cls_idx, exemplar_idx] = self._compute_bank_query_score(
            new_support,
            new_quality,
        )

    @torch.no_grad()
    def _replace_query_exemplar(self, cls_idx, feat, quality_weight, candidate_score):
        valid_mask = self.bank_query_valid[cls_idx]
        valid_inds = torch.nonzero(valid_mask, as_tuple=False).flatten()
        if valid_inds.numel() == 0:
            self._add_query_exemplar(cls_idx, feat, quality_weight, candidate_score)
            return

        existing_feats = F.normalize(self.bank_query_feats[cls_idx, valid_inds], dim=-1, eps=1e-6)
        existing_support = self.bank_query_support[cls_idx, valid_inds]
        existing_quality = self.bank_query_quality[cls_idx, valid_inds]

        if existing_feats.shape[0] > 1:
            pairwise = torch.matmul(existing_feats, existing_feats.transpose(0, 1))
            pairwise.fill_diagonal_(-1.0)
            redundancy = pairwise.max(dim=-1).values.clamp(min=0.0, max=1.0)
        else:
            redundancy = existing_quality.new_zeros(existing_quality.shape)

        support_score = torch.sqrt(existing_support.clamp(min=1.0))
        support_score = support_score / support_score.max().clamp(min=1e-6)
        utility = 0.45 * support_score + 0.35 * existing_quality + 0.20 * (1.0 - redundancy)

        replace_local = int(utility.argmin().item())
        best_sim = torch.matmul(existing_feats, feat).max().clamp(min=-1.0, max=1.0)
        candidate_utility = 0.5 * quality_weight + 0.5 * (1.0 - best_sim)
        if (candidate_utility.item() < utility[replace_local].item()
                and candidate_utility.item() < self.query_replace_threshold):
            return

        replace_idx = int(valid_inds[replace_local].item())
        self.bank_query_feats[cls_idx, replace_idx] = F.normalize(feat, dim=0, eps=1e-6)
        self.bank_query_support[cls_idx, replace_idx] = 1.0
        self.bank_query_quality[cls_idx, replace_idx] = quality_weight
        self.bank_query_score[cls_idx, replace_idx] = candidate_score
        self.bank_query_valid[cls_idx, replace_idx] = True

    @torch.no_grad()
    def _refresh_class_slots(self, cls_idx):
        query_mask = self.bank_query_valid[cls_idx]
        query_feats = self.bank_query_feats[cls_idx][query_mask]
        query_support = self.bank_query_support[cls_idx][query_mask]
        query_quality = self.bank_query_quality[cls_idx][query_mask].clamp(min=0.05)

        if query_feats.numel() == 0:
            self.prototype_bank[cls_idx].zero_()
            self.prototype_count[cls_idx].zero_()
            self.prototype_quality[cls_idx].zero_()
            self.prototype_radius[cls_idx].zero_()
            self.prototype_age[cls_idx].zero_()
            return

        tokens = query_feats
        support = query_support
        quality = query_quality
        weights = torch.sqrt(support.clamp(min=1.0)) * (0.5 + quality)

        num_slots = min(self.num_prototypes, tokens.shape[0])
        seed_inds = _select_weighted_diverse_indices(tokens, weights, num_slots)
        seed_tokens = tokens[seed_inds]
        assign_ids = torch.matmul(tokens, seed_tokens.transpose(0, 1)).argmax(dim=-1)

        new_bank = tokens.new_zeros((self.num_prototypes, self.embed_dims))
        new_count = support.new_zeros((self.num_prototypes,))
        new_quality = quality.new_zeros((self.num_prototypes,))
        new_radius = quality.new_zeros((self.num_prototypes,))

        for slot_idx in range(num_slots):
            slot_mask = assign_ids == slot_idx
            if not torch.any(slot_mask):
                continue

            slot_tokens = tokens[slot_mask]
            slot_support = support[slot_mask]
            slot_quality_token = quality[slot_mask]
            slot_weights = weights[slot_mask]

            medoid_idx = _select_weighted_medoid(slot_tokens, slot_weights)
            slot_proto = F.normalize(slot_tokens[medoid_idx], dim=0, eps=1e-6)
            slot_radius = 1.0 - torch.matmul(slot_tokens, slot_proto).clamp(min=-1.0, max=1.0)
            slot_radius = (slot_radius * slot_weights).sum() / slot_weights.sum().clamp(min=1e-6)

            new_bank[slot_idx] = slot_proto
            new_count[slot_idx] = slot_support.sum()
            new_quality[slot_idx] = (
                (slot_quality_token * slot_weights).sum() / slot_weights.sum().clamp(min=1e-6)
            )
            new_radius[slot_idx] = slot_radius

        self.prototype_bank[cls_idx].zero_()
        self.prototype_count[cls_idx].zero_()
        self.prototype_quality[cls_idx].zero_()
        self.prototype_radius[cls_idx].zero_()
        self.prototype_age[cls_idx].zero_()

        self.prototype_bank[cls_idx] = new_bank
        self.prototype_count[cls_idx] = new_count
        self.prototype_quality[cls_idx] = new_quality
        self.prototype_radius[cls_idx] = new_radius
        self.prototype_age[cls_idx].zero_()


class PrototypeCrossAttention(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_classes,
                 num_prototypes,
                 topk_classes=2,
                 num_heads=8,
                 attn_drop=0.1,
                 ffn_hidden_dim=512):
        super(PrototypeCrossAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.topk_classes = max(1, min(num_classes, topk_classes))
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.attn_drop = attn_drop
        assert self.head_dim * num_heads == embed_dims

        self.query_norm = nn.LayerNorm(embed_dims)
        self.memory_norm = nn.LayerNorm(embed_dims)
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.ffn_norm = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_hidden_dim, embed_dims),
        )

    def init_weights(self):
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def _attach_ddp_zero_residual(self, query_feat):
        if query_feat is None or not self.training:
            return query_feat

        # Keep this module in the autograd graph even when the current batch has
        # no usable prototype tokens on a given rank. This avoids DDP unused
        # parameter errors during the bank cold-start stage.
        zero = query_feat.new_zeros(())
        for param in self.parameters():
            zero = zero + param.sum() * 0.0
        return query_feat + zero

    def forward(self,
                query_feat,
                cls_score,
                prototype_bank,
                prototype_count,
                min_count=0):
        if query_feat is None:
            return query_feat

        query_feat = self._attach_ddp_zero_residual(query_feat)

        if prototype_bank is None or prototype_count is None or query_feat.numel() == 0:
            return query_feat

        valid_slots = prototype_count >= float(min_count)
        if not torch.any(valid_slots):
            return query_feat

        cls_prob = normalize_query_logits(cls_score)
        valid_classes = valid_slots.any(dim=-1)
        if not torch.any(valid_classes):
            return query_feat

        cls_prob = cls_prob.masked_fill(~valid_classes.view(1, 1, -1), -1e4)
        topk_classes = min(self.topk_classes, cls_prob.shape[-1])
        _, topk_class_inds = cls_prob.topk(topk_classes, dim=-1)

        prototype_bank = F.normalize(prototype_bank.to(query_feat.dtype), dim=-1, eps=1e-6)
        prototype_tokens = prototype_bank[topk_class_inds.reshape(-1)]
        prototype_tokens = prototype_tokens.reshape(
            *query_feat.shape[:2],
            topk_classes * self.num_prototypes,
            self.embed_dims,
        )
        prototype_valid = valid_slots[topk_class_inds.reshape(-1)]
        prototype_valid = prototype_valid.reshape(
            *query_feat.shape[:2],
            topk_classes * self.num_prototypes,
        )

        query_has_proto = prototype_valid.any(dim=-1)
        if not torch.any(query_has_proto):
            return query_feat

        refined_query = query_feat.clone()
        query_tokens = self.query_norm(query_feat[query_has_proto])
        memory_tokens = self.memory_norm(prototype_tokens[query_has_proto])
        key_padding_mask = ~prototype_valid[query_has_proto]

        q = self.q_proj(query_tokens).reshape(-1, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory_tokens).reshape(-1, memory_tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory_tokens).reshape(-1, memory_tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = torch.zeros(
            key_padding_mask.shape[0], 1, 1, key_padding_mask.shape[1],
            device=key_padding_mask.device,
            dtype=q.dtype,
        )
        attn_mask = attn_mask.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(-1, 1, self.embed_dims).squeeze(1)
        attn_out = self.out_proj(attn_out)

        refined_tokens = query_feat[query_has_proto] + attn_out
        refined_tokens = refined_tokens + self.ffn(self.ffn_norm(refined_tokens))
        refined_query[query_has_proto] = refined_tokens
        return refined_query
