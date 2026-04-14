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
                 online_match_threshold=0.75,
                 init_match_threshold=0.55,
                 maintenance_interval=64,
                 alpha_min=0.05,
                 alpha_max=0.20,
                 quality_gamma=0.10,
                 radius_gamma=0.10,
                 recent_score_threshold=0.20,
                 recent_dedup_threshold=0.95,
                 radius_refresh_threshold=0.30):
        super(QueryPrototypeBank, self).__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_prototypes = num_prototypes
        self.memory_size_per_class = memory_size_per_class
        self.recent_buffer_size = memory_size_per_class if recent_buffer_size is None else max(1, int(recent_buffer_size))
        self.momentum = momentum
        self.online_match_threshold = float(online_match_threshold)
        self.init_match_threshold = float(init_match_threshold)
        self.maintenance_interval = max(1, int(maintenance_interval))
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.quality_gamma = float(quality_gamma)
        self.radius_gamma = float(radius_gamma)
        self.recent_score_threshold = float(recent_score_threshold)
        self.recent_dedup_threshold = float(recent_dedup_threshold)
        self.radius_refresh_threshold = float(radius_refresh_threshold)

        self.register_buffer('prototype_bank', torch.zeros(num_classes, num_prototypes, embed_dims))
        self.register_buffer('prototype_count', torch.zeros(num_classes, num_prototypes))
        self.register_buffer('prototype_quality', torch.zeros(num_classes, num_prototypes))
        self.register_buffer('prototype_radius', torch.zeros(num_classes, num_prototypes))
        self.register_buffer('prototype_age', torch.zeros(num_classes, num_prototypes))
        self.register_buffer('prototype_updates', torch.zeros(1, dtype=torch.long))

        self.register_buffer('recent_feats', torch.zeros(num_classes, self.recent_buffer_size, embed_dims))
        self.register_buffer('recent_quality', torch.zeros(num_classes, self.recent_buffer_size))
        self.register_buffer('recent_score', torch.full((num_classes, self.recent_buffer_size), -1e6))
        self.register_buffer('recent_valid', torch.zeros(num_classes, self.recent_buffer_size, dtype=torch.bool))

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
            self._update_class_online(
                cls_idx,
                feats[cls_mask],
                qualities[cls_mask],
            )

        current_update = int(self.prototype_updates.item()) + 1
        for cls_idx in touched_classes:
            self._maybe_maintain_class(cls_idx, current_update)

        self.prototype_updates += 1

    def _quality_to_weight(self, quality):
        quality = quality.float()
        return 0.1 + 0.9 * torch.sigmoid(quality)

    @torch.no_grad()
    def _update_class_online(self, cls_idx, feats, qualities):
        order = qualities.argsort(descending=True)
        for sample_idx in order.tolist():
            feat = feats[sample_idx]
            quality_weight = self._quality_to_weight(qualities[sample_idx]).to(feat.dtype)
            self._update_single_query(cls_idx, feat, quality_weight)

    @torch.no_grad()
    def _update_single_query(self, cls_idx, feat, quality_weight):
        valid_mask = self.prototype_count[cls_idx] > 0
        valid_inds = torch.nonzero(valid_mask, as_tuple=False).flatten()

        if valid_inds.numel() == 0:
            self._init_slot(cls_idx, feat, quality_weight)
            return

        bank = F.normalize(self.prototype_bank[cls_idx, valid_inds], dim=-1, eps=1e-6)
        similarity = torch.matmul(bank, feat)
        best_local = int(similarity.argmax().item())
        best_slot = int(valid_inds[best_local].item())
        best_sim = similarity[best_local].clamp(min=-1.0, max=1.0)
        novelty = 1.0 - best_sim

        if valid_inds.numel() < self.num_prototypes and best_sim.item() < self.init_match_threshold:
            self._init_slot(cls_idx, feat, quality_weight)
            self._add_recent_candidate(cls_idx, feat, quality_weight, novelty)
            return

        self._online_update_slot(cls_idx, best_slot, feat, quality_weight, best_sim)
        self._add_recent_candidate(cls_idx, feat, quality_weight, novelty)

    @torch.no_grad()
    def _init_slot(self, cls_idx, feat, quality_weight):
        empty_inds = torch.nonzero(self.prototype_count[cls_idx] <= 0, as_tuple=False).flatten()
        if empty_inds.numel() == 0:
            return False

        slot_idx = int(empty_inds[0].item())
        self.prototype_bank[cls_idx, slot_idx] = F.normalize(feat, dim=0, eps=1e-6)
        self.prototype_count[cls_idx, slot_idx] = 1.0
        self.prototype_quality[cls_idx, slot_idx] = quality_weight
        self.prototype_radius[cls_idx, slot_idx] = 0.0
        self.prototype_age[cls_idx, slot_idx] = 0.0
        return True

    @torch.no_grad()
    def _online_update_slot(self, cls_idx, slot_idx, feat, quality_weight, best_sim):
        old_center = self.prototype_bank[cls_idx, slot_idx]
        old_count = self.prototype_count[cls_idx, slot_idx]
        old_quality = self.prototype_quality[cls_idx, slot_idx]
        old_radius = self.prototype_radius[cls_idx, slot_idx]

        base_alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * quality_weight
        base_alpha = base_alpha / torch.sqrt(old_count + 1.0)
        if best_sim.item() < self.online_match_threshold:
            base_alpha = base_alpha * 0.5
        alpha_floor = 0.0
        if old_count.item() <= 1.0:
            alpha_floor = self.alpha_min
        elif old_count.item() <= 4.0:
            alpha_floor = max(1.0 - self.momentum, 0.01)
        alpha = base_alpha.clamp(min=alpha_floor, max=self.alpha_max)

        new_center = old_center * (1.0 - alpha) + feat * alpha
        new_center = F.normalize(new_center, dim=0, eps=1e-6)
        sample_radius = 1.0 - torch.matmul(old_center, feat).clamp(min=-1.0, max=1.0)

        self.prototype_bank[cls_idx, slot_idx] = new_center
        self.prototype_count[cls_idx, slot_idx] = old_count + 1.0
        self.prototype_quality[cls_idx, slot_idx] = (
            (1.0 - self.quality_gamma) * old_quality + self.quality_gamma * quality_weight
        )
        self.prototype_radius[cls_idx, slot_idx] = (
            (1.0 - self.radius_gamma) * old_radius + self.radius_gamma * sample_radius
        )
        self.prototype_age[cls_idx, slot_idx] = 0.0

    @torch.no_grad()
    def _add_recent_candidate(self, cls_idx, feat, quality_weight, novelty):
        score = quality_weight * (1.0 + novelty.clamp(min=0.0))
        if score.item() < self.recent_score_threshold:
            return

        valid_mask = self.recent_valid[cls_idx]
        if torch.any(valid_mask):
            existing_feats = self.recent_feats[cls_idx][valid_mask]
            existing_sim = torch.matmul(existing_feats, feat)
            best_existing = int(existing_sim.argmax().item())
            if existing_sim[best_existing].item() >= self.recent_dedup_threshold:
                valid_inds = torch.nonzero(valid_mask, as_tuple=False).flatten()
                dst_idx = int(valid_inds[best_existing].item())
                if score.item() > self.recent_score[cls_idx, dst_idx].item():
                    self.recent_feats[cls_idx, dst_idx] = feat
                    self.recent_quality[cls_idx, dst_idx] = quality_weight
                    self.recent_score[cls_idx, dst_idx] = score
                return

        empty_inds = torch.nonzero(~valid_mask, as_tuple=False).flatten()
        if empty_inds.numel() > 0:
            dst_idx = int(empty_inds[0].item())
        else:
            min_score_idx = int(self.recent_score[cls_idx].argmin().item())
            if score.item() <= self.recent_score[cls_idx, min_score_idx].item():
                return
            dst_idx = min_score_idx

        self.recent_feats[cls_idx, dst_idx] = feat
        self.recent_quality[cls_idx, dst_idx] = quality_weight
        self.recent_score[cls_idx, dst_idx] = score
        self.recent_valid[cls_idx, dst_idx] = True

    @torch.no_grad()
    def _maybe_maintain_class(self, cls_idx, current_update):
        recent_mask = self.recent_valid[cls_idx]
        num_recent = int(recent_mask.sum().item())
        if num_recent == 0:
            return

        valid_mask = self.prototype_count[cls_idx] > 0
        num_valid = int(valid_mask.sum().item())
        need_fill = num_valid < self.num_prototypes
        interval_hit = (current_update % self.maintenance_interval) == 0
        recent_trigger = num_recent >= max(2, self.num_prototypes // 2)
        drift_trigger = False
        if num_valid > 0:
            drift_trigger = self.prototype_radius[cls_idx, valid_mask].max().item() >= self.radius_refresh_threshold

        if not (need_fill or interval_hit or recent_trigger or drift_trigger):
            return

        self._recluster_class(cls_idx)

    @torch.no_grad()
    def _recluster_class(self, cls_idx):
        slot_mask = self.prototype_count[cls_idx] > 0
        recent_mask = self.recent_valid[cls_idx]

        slot_feats = self.prototype_bank[cls_idx][slot_mask]
        slot_count = self.prototype_count[cls_idx][slot_mask]
        slot_quality = self.prototype_quality[cls_idx][slot_mask].clamp(min=0.05)
        recent_feats = self.recent_feats[cls_idx][recent_mask]
        recent_quality = self.recent_quality[cls_idx][recent_mask].clamp(min=0.05)

        if slot_feats.numel() == 0 and recent_feats.numel() == 0:
            return

        support_list = []
        quality_list = []
        feat_list = []
        if slot_feats.numel() > 0:
            feat_list.append(slot_feats)
            support_list.append(slot_count)
            quality_list.append(slot_quality)
        if recent_feats.numel() > 0:
            feat_list.append(recent_feats)
            support_list.append(recent_feats.new_ones((recent_feats.shape[0],)))
            quality_list.append(recent_quality)

        tokens = torch.cat(feat_list, dim=0)
        support = torch.cat(support_list, dim=0)
        quality = torch.cat(quality_list, dim=0)
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

        self.recent_feats[cls_idx].zero_()
        self.recent_quality[cls_idx].zero_()
        self.recent_score[cls_idx].fill_(-1e6)
        self.recent_valid[cls_idx].zero_()


class PrototypeCrossAttention(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_classes,
                 num_prototypes,
                 topk_classes=2,
                 num_heads=8,
                 attn_drop=0.1,
                 ffn_hidden_dim=512,
                 max_memory_tokens=None):
        super(PrototypeCrossAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.topk_classes = max(1, min(num_classes, topk_classes))
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.attn_drop = attn_drop
        self.max_memory_tokens = (
            max(1, int(max_memory_tokens))
            if max_memory_tokens is not None
            else max(1, self.topk_classes * self.num_prototypes)
        )
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

    def forward(self,
                query_feat,
                cls_score,
                prototype_bank,
                prototype_count,
                min_count=0):
        if (prototype_bank is None or prototype_count is None or query_feat is None
                or query_feat.numel() == 0):
            return query_feat

        valid_slots = prototype_count >= float(min_count)
        if not torch.any(valid_slots):
            return query_feat

        cls_prob = normalize_query_logits(cls_score)
        prototype_bank = F.normalize(prototype_bank.to(query_feat.dtype), dim=-1, eps=1e-6)
        flat_bank = prototype_bank.reshape(-1, self.embed_dims)
        flat_valid = valid_slots.reshape(-1)
        num_valid_tokens = int(flat_valid.sum().item())
        if num_valid_tokens <= 0:
            return query_feat

        flat_count = prototype_count.reshape(-1).to(query_feat.dtype)
        count_score = torch.log1p(flat_count)
        valid_count = count_score[flat_valid]
        count_score = count_score / valid_count.max().clamp(min=1.0)
        count_score = 0.5 + 0.5 * count_score

        slot_prior = cls_prob[:, :, :, None].expand(-1, -1, -1, self.num_prototypes)
        slot_prior = slot_prior.reshape(*query_feat.shape[:2], -1)

        query_score_feat = F.normalize(query_feat, dim=-1, eps=1e-6)
        slot_similarity = torch.einsum('bqd,sd->bqs', query_score_feat, flat_bank)
        slot_score = slot_prior * ((slot_similarity + 1.0) * 0.5)
        slot_score = slot_score * count_score.view(1, 1, -1)
        slot_score = slot_score.masked_fill(~flat_valid.view(1, 1, -1), -1e4)

        topk_slots = min(self.max_memory_tokens, num_valid_tokens)
        if topk_slots <= 0:
            return query_feat

        topk_score, topk_slot_inds = slot_score.topk(topk_slots, dim=-1)
        prototype_tokens = flat_bank[topk_slot_inds.reshape(-1)]
        prototype_tokens = prototype_tokens.reshape(*query_feat.shape[:2], topk_slots, self.embed_dims)
        prototype_valid = topk_score > -1e3

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
