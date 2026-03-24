from __future__ import annotations

import torch
from torch import nn


class PointerPolicyNet(nn.Module):
    """rl_init.md architecture: 9->32 embed + query/key dot-product head + pass key."""

    def __init__(self, max_hand_combos: int, max_keys: int):
        super().__init__()
        self.max_hand_combos = max_hand_combos
        self.max_keys = max_keys
        self.embed = nn.Sequential(nn.Linear(9, 32), nn.ReLU())
        self.query_proj = nn.Linear(32, 32, bias=False)
        self.key_proj = nn.Linear(32, 32, bias=False)
        self.pass_key = nn.Parameter(torch.zeros(32))

        self.value_head = nn.Sequential(nn.Linear(32 * 2 + 6, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hand_vecs = obs["hand_vecs"]  # [B,7,9]
        public_vecs = obs["public_vecs"]  # [B,P,9]
        mask = obs["mask"].bool()  # [B,7,K]
        meta = obs["meta"]  # [B,6]

        hand_emb = self.embed(hand_vecs)  # [B,7,32]
        pub_emb = self.embed(public_vecs)  # [B,P,32]

        b = hand_emb.shape[0]
        pass_key = self.pass_key.view(1, 1, 32).expand(b, 1, 32)
        keys = torch.cat([pub_emb, pass_key], dim=1)  # [B,K,32]

        q = self.query_proj(hand_emb)
        k = self.key_proj(keys)
        logits_matrix = torch.matmul(q, k.transpose(1, 2))  # [B,7,K]
        logits_matrix = logits_matrix.masked_fill(~mask, -1e9)
        logits = logits_matrix.flatten(start_dim=1)  # [B, 7*K]

        # Value head on pooled hand/public embeddings + meta.
        hand_valid = obs["hand_valid"].unsqueeze(-1).float()
        pub_valid = obs["public_valid"].unsqueeze(-1).float()
        hand_pool = (hand_emb * hand_valid).sum(1) / hand_valid.sum(1).clamp(min=1.0)
        pub_pool = (pub_emb * pub_valid).sum(1) / pub_valid.sum(1).clamp(min=1.0)
        value_in = torch.cat([hand_pool, pub_pool, meta.float()], dim=1)
        value = self.value_head(value_in).squeeze(-1)
        return logits, value

