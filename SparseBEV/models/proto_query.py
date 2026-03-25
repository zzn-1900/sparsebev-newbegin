import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def normalize_query_logits(cls_score):
    cls_prob = torch.sigmoid(cls_score)
    cls_prob = cls_prob / cls_prob.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return cls_prob


def mix_query_prototypes(cls_score, prototype_bank, prototype_count, min_count=0):
    cls_prob = normalize_query_logits(cls_score)

    if prototype_bank is None or prototype_count is None:
        proto_mix = cls_score.new_zeros(*cls_score.shape[:2], cls_score.shape[-1])
        proto_available = cls_score.new_zeros(*cls_score.shape[:2], 1)
        return proto_mix, proto_available

    valid_proto = (prototype_count >= float(min_count)).to(cls_prob.dtype)
    cls_prob = cls_prob * valid_proto.view(1, 1, -1)

    prob_sum = cls_prob.sum(dim=-1, keepdim=True)
    proto_available = (prob_sum > 0).to(cls_prob.dtype)
    cls_prob = cls_prob / prob_sum.clamp(min=1e-6)

    prototype_bank = F.normalize(prototype_bank, dim=-1, eps=1e-6)
    proto_mix = torch.matmul(cls_prob, prototype_bank)
    proto_mix = F.normalize(proto_mix, dim=-1, eps=1e-6) * proto_available

    return proto_mix, proto_available


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


class QueryPrototypeBank(nn.Module):
    def __init__(self, num_layers, num_classes, embed_dims, momentum=0.99):
        super(QueryPrototypeBank, self).__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.momentum = momentum

        self.register_buffer('prototype_bank', torch.zeros(num_layers, num_classes, embed_dims))
        self.register_buffer('prototype_count', torch.zeros(num_layers, num_classes))
        self.register_buffer('prototype_updates', torch.zeros(1, dtype=torch.long))

    def get_valid_mask(self, layer_idx, min_count=0):
        return self.prototype_count[layer_idx] >= float(min_count)

    def get_normalized_bank(self, layer_idx):
        return F.normalize(self.prototype_bank[layer_idx], dim=-1, eps=1e-6)

    @torch.no_grad()
    def update(self, layer_idx, feats, labels):
        if feats is None or labels is None or feats.numel() == 0 or labels.numel() == 0:
            return

        feats = _all_gather_tensor(feats.detach().float())
        labels = _all_gather_tensor(labels.detach().long())
        if feats is None or labels is None or feats.numel() == 0 or labels.numel() == 0:
            return

        feats = F.normalize(feats, dim=-1, eps=1e-6)
        unique_labels = labels.unique()
        for cls_idx in unique_labels.tolist():
            if cls_idx < 0 or cls_idx >= self.num_classes:
                continue

            cls_mask = labels == cls_idx
            if not torch.any(cls_mask):
                continue

            cls_feat = feats[cls_mask].mean(dim=0)
            cls_feat = F.normalize(cls_feat, dim=0, eps=1e-6)

            if self.prototype_count[layer_idx, cls_idx] <= 0:
                updated = cls_feat
            else:
                updated = self.prototype_bank[layer_idx, cls_idx] * self.momentum
                updated = updated + cls_feat * (1.0 - self.momentum)
                updated = F.normalize(updated, dim=0, eps=1e-6)

            self.prototype_bank[layer_idx, cls_idx] = updated
            self.prototype_count[layer_idx, cls_idx] += cls_mask.sum().item()

        self.prototype_updates += 1


class QueryDifficultyEstimator(nn.Module):
    def __init__(self, num_classes, code_size, hidden_dim=64):
        super(QueryDifficultyEstimator, self).__init__()
        self.num_classes = num_classes
        self.code_size = code_size
        self.entropy_norm = math.log(max(2, num_classes))
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def init_weights(self):
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, query_feat, prev_query_feat, bbox_pred, prev_bbox_pred, cls_score):
        cls_prob = normalize_query_logits(cls_score)

        cls_entropy = -(cls_prob * cls_prob.clamp(min=1e-6).log()).sum(dim=-1)
        cls_entropy = cls_entropy / self.entropy_norm

        if self.num_classes > 1:
            top2 = cls_prob.topk(2, dim=-1).values
            cls_margin = top2[..., 0] - top2[..., 1]
        else:
            cls_margin = cls_prob[..., 0]

        if prev_bbox_pred is None:
            bbox_delta_norm = bbox_pred.new_zeros(bbox_pred.shape[:2])
        else:
            bbox_delta_norm = torch.linalg.vector_norm(bbox_pred - prev_bbox_pred, dim=-1)
            bbox_delta_norm = bbox_delta_norm / math.sqrt(max(1, self.code_size))

        if prev_query_feat is None:
            query_drift = query_feat.new_zeros(query_feat.shape[:2])
        else:
            query_drift = torch.linalg.vector_norm(query_feat - prev_query_feat, dim=-1)
            query_drift = query_drift / math.sqrt(max(1, query_feat.shape[-1]))

        difficulty_feat = torch.stack([
            cls_entropy,
            1.0 - cls_margin,
            bbox_delta_norm,
            query_drift,
        ], dim=-1)

        return torch.sigmoid(self.mlp(difficulty_feat))


class PrototypeRefiner(nn.Module):
    def __init__(self, embed_dims, hidden_dim=256):
        super(PrototypeRefiner, self).__init__()
        input_dim = embed_dims * 2 + 1

        self.delta_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dims),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def init_weights(self):
        nn.init.zeros_(self.delta_mlp[-1].weight)
        nn.init.zeros_(self.delta_mlp[-1].bias)
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, -2.0)

    def forward(self, query_feat, proto_feat, diff_score, proto_available):
        refine_in = torch.cat([query_feat, proto_feat, diff_score], dim=-1)
        delta = self.delta_mlp(refine_in)
        gate = torch.sigmoid(self.gate_mlp(refine_in))
        gate = gate * proto_available

        return query_feat + gate * delta, gate
