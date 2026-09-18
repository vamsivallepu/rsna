"""Mask-safe CoAtNet-compatible baseline and anatomical pooling heads."""
import torch
from torch import nn


def masked_softmax(scores, mask, dim=-1):
    weights = scores.masked_fill(~mask, -1e4).softmax(dim=dim) * mask
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-4)


class AnatomicalHead(nn.Module):
    def __init__(self, feature_dim=1024, variant="baseline", dropout=.2):
        super().__init__()
        if variant not in {"baseline", "position", "hierarchical"}:
            raise ValueError(variant)
        self.variant = variant
        self.norm = nn.LayerNorm(feature_dim)
        self.attention = nn.Sequential(nn.Linear(feature_dim, 256), nn.Tanh(),
                                       nn.Dropout(dropout), nn.Linear(256, 12))
        self.target_weight = nn.Parameter(torch.randn(12, feature_dim) * .02)
        self.target_bias = nn.Parameter(torch.zeros(12))
        if variant != "baseline":
            self.slot_embedding = nn.Embedding(5, feature_dim)
            self.position_projection = nn.Linear(1, feature_dim, bias=False)
            nn.init.zeros_(self.slot_embedding.weight)
            nn.init.zeros_(self.position_projection.weight)
        if variant == "hierarchical":
            self.slot_attention = nn.Linear(feature_dim, 1)
            nn.init.zeros_(self.slot_attention.weight)
            nn.init.zeros_(self.slot_attention.bias)

    def forward(self, features, mask, slots=None, positions=None):
        mask = mask.bool()
        hidden = self.norm(features)
        if self.variant != "baseline":
            if slots is None or positions is None:
                raise ValueError("Anatomical metadata required")
            hidden = hidden + self.slot_embedding(slots.clamp_min(0))
            hidden = hidden + self.position_projection(positions[..., None])
        scores = self.attention(hidden).transpose(1, 2)
        if self.variant != "hierarchical":
            attention = masked_softmax(scores, mask[:, None, :])
            pooled = torch.einsum("btk,bkf->btf", attention, hidden)
        else:
            pools, valid_slots = [], []
            for slot in range(5):
                valid = mask & (slots == slot)
                att = masked_softmax(scores, valid[:, None, :])
                pools.append(torch.einsum("btk,bkf->btf", att, hidden))
                valid_slots.append(valid.any(dim=1))
            stack = torch.stack(pools, dim=2)
            valid = torch.stack(valid_slots, dim=-1)[:, None, :]
            att = masked_softmax(self.slot_attention(stack).squeeze(-1), valid)
            pooled = (stack * att[..., None]).sum(dim=2)
        return (pooled * self.target_weight[None]).sum(dim=-1) + self.target_bias

    def load_adapted(self, state):
        compatible = {k.removeprefix("head."): v for k, v in state.items() if k.startswith("head.")}
        if not compatible:
            compatible = state
        expected = set(AnatomicalHead(self.target_weight.shape[1], "baseline").state_dict())
        if set(compatible) != expected:
            raise ValueError("Unexpected adapted head keys")
        missing, extra = self.load_state_dict(compatible, strict=False)
        if extra or any(not k.startswith(("slot_embedding.", "position_projection.", "slot_attention.")) for k in missing):
            raise ValueError("Incompatible head checkpoint")
