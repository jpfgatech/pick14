from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pick14.rl.encoding import MAX_PLAY_HAND, NUM_GLOBAL_PLAY_CONTEXT_KEYS


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

        hand_valid = obs["hand_valid"].unsqueeze(-1).float()
        pub_valid = obs["public_valid"].unsqueeze(-1).float()
        hand_pool = (hand_emb * hand_valid).sum(1) / hand_valid.sum(1).clamp(min=1.0)
        pub_pool = (pub_emb * pub_valid).sum(1) / pub_valid.sum(1).clamp(min=1.0)
        value_in = torch.cat([hand_pool, pub_pool, meta.float()], dim=1)
        value = self.value_head(value_in).squeeze(-1)
        return logits, value


class DualHeadPolicyNet(nn.Module):
    """
    Shared trunk matches match-head style: Linear(9,D) + ReLU (default D=32).
    Match head: linear Q/K, masked dot-product logits (same pattern as PointerPolicyNet).
    Play head: linear Q/K; keys = public columns + NUM_GLOBAL_PLAY_CONTEXT_KEYS learnable
    vectors. Per-slot logit = sum_k LeakyReLU(0.1)( q·k_k ) over masked keys.
    """

    def __init__(
        self,
        max_hand_combos: int,
        max_match_keys: int,
        max_play_hand: int = MAX_PLAY_HAND,
        hidden_dim: int = 32,
        num_global_play_context_keys: int = NUM_GLOBAL_PLAY_CONTEXT_KEYS,
    ):
        super().__init__()
        self.max_hand_combos = max_hand_combos
        self.max_match_keys = max_match_keys
        self.max_play_hand = max_play_hand
        self.hidden_dim = hidden_dim
        self.num_global_play_context_keys = num_global_play_context_keys
        self.match_flat_dim = max_hand_combos * max_match_keys

        d = hidden_dim
        self.embed = nn.Sequential(nn.Linear(9, d), nn.ReLU())

        self.query_match = nn.Linear(d, d, bias=False)
        self.key_match = nn.Linear(d, d, bias=False)
        self.pass_key_match = nn.Parameter(torch.zeros(d))

        self.query_play = nn.Linear(d, d, bias=False)
        self.key_play = nn.Linear(d, d, bias=False)
        self.global_play_context_keys = nn.Parameter(
            torch.zeros(num_global_play_context_keys, d)
        )

        self.value_head = nn.Sequential(nn.Linear(d * 2 + 6, 64), nn.ReLU(), nn.Linear(64, 1))

        nn.init.normal_(self.global_play_context_keys, mean=0.0, std=0.02)

    def forward(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hand_vecs = obs["hand_vecs"]
        public_vecs = obs["public_vecs"]
        mask_match = obs["mask"].bool()
        play_hand_vecs = obs["play_hand_vecs"]
        play_hand_valid = obs["play_hand_valid"].bool()
        play_key_mask = obs["play_key_mask"].bool()
        meta = obs["meta"]

        hand_emb = self.embed(hand_vecs)
        pub_emb = self.embed(public_vecs)
        play_card_emb = self.embed(play_hand_vecs)

        b = hand_emb.shape[0]
        d = self.hidden_dim
        pass_k = self.pass_key_match.view(1, 1, d).expand(b, 1, d)
        keys_m = torch.cat([pub_emb, pass_k], dim=1)
        qm = self.query_match(hand_emb)
        km = self.key_match(keys_m)
        logits_m = torch.matmul(qm, km.transpose(1, 2))
        logits_m = logits_m.masked_fill(~mask_match, -1e9)
        match_logits = logits_m.flatten(start_dim=1)

        ctx = self.global_play_context_keys.unsqueeze(0).expand(
            b, self.num_global_play_context_keys, d
        )
        keys_p = torch.cat([pub_emb, ctx], dim=1)
        qp = self.query_play(play_card_emb)
        kp = self.key_play(keys_p)
        scores_p = torch.matmul(qp, kp.transpose(1, 2))
        scores_p = scores_p.masked_fill(~play_key_mask, 0.0)
        scores_p = F.leaky_relu(scores_p, negative_slope=0.1)
        play_logits = scores_p.sum(dim=2)
        play_logits = play_logits.masked_fill(~play_hand_valid, -1e9)

        hand_valid = obs["hand_valid"].unsqueeze(-1).float()
        pub_valid = obs["public_valid"].unsqueeze(-1).float()
        hand_pool = (hand_emb * hand_valid).sum(1) / hand_valid.sum(1).clamp(min=1.0)
        pub_pool = (pub_emb * pub_valid).sum(1) / pub_valid.sum(1).clamp(min=1.0)
        value_in = torch.cat([hand_pool, pub_pool, meta.float()], dim=1)
        value = self.value_head(value_in).squeeze(-1)
        return match_logits, play_logits, value
