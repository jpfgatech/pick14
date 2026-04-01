"""
rl.md §2–3: shared Transformer L1, four independent L2 encoders, pointer match head (√d scale),
play head Linear(32,1), dual critics mean-pool + MLP(32→16→1).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from pick14.rl.rlmd_sequences import (
    AGENT_HAND_END,
    AGENT_POOL_END,
    AGENT_POOL_START,
    AGENT_SEQ_LEN,
    CRITIC_SEQ_LEN,
    MAX_HAND_COMBOS,
    MAX_PLAY_HAND,
    NUM_GLOBAL_TOKENS,
    NUM_ROLE_TYPES,
    RLMD_DIM,
    RLMD_POOL_SLOTS,
    ROLE_GLOBAL,
)
from pick14.rl.rlmd_transformer import RLmdEncoderLayer

SQRT_D = math.sqrt(float(RLMD_DIM))


class RLmdPPOAgent(nn.Module):
    """Policy + dual critics per rl.md (2-player default)."""

    max_hand_combos = MAX_HAND_COMBOS
    max_match_keys = RLMD_POOL_SLOTS
    match_flat_dim = MAX_HAND_COMBOS * RLMD_POOL_SLOTS

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        d = RLMD_DIM
        self.card_proj = nn.Linear(9, d)
        self.role_emb = nn.Embedding(NUM_ROLE_TYPES, d)
        self.global_tokens = nn.Parameter(torch.zeros(NUM_GLOBAL_TOKENS, d))
        nn.init.normal_(self.global_tokens, std=0.02)

        self.layer1 = RLmdEncoderLayer(d, nhead=4, dim_ff=128, dropout=dropout)
        self.layer2_match = RLmdEncoderLayer(d, nhead=4, dim_ff=128, dropout=dropout)
        self.layer2_play = RLmdEncoderLayer(d, nhead=4, dim_ff=128, dropout=dropout)
        self.layer2_critic_agent = RLmdEncoderLayer(d, nhead=4, dim_ff=128, dropout=dropout)
        self.layer2_critic_opp = RLmdEncoderLayer(d, nhead=4, dim_ff=128, dropout=dropout)

        self.query_match = nn.Linear(d, d, bias=False)
        self.key_match = nn.Linear(d, d, bias=False)

        self.play_score = nn.Linear(d, 1)

        self.critic_agent_mlp = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 1))
        self.critic_opp_mlp = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 1))

    def set_requires_grad_actor_trunk(self, enabled: bool) -> None:
        """Freeze or unfreeze embeddings, shared L1, match/play L2, and actor heads (curriculum Phase 2)."""
        for name, p in self.named_parameters():
            if any(
                x in name
                for x in (
                    "layer2_critic_agent",
                    "layer2_critic_opp",
                    "critic_agent_mlp",
                    "critic_opp_mlp",
                )
            ):
                continue
            p.requires_grad = enabled

    def set_requires_grad_critics(self, enabled: bool) -> None:
        for name, p in self.named_parameters():
            if any(
                x in name
                for x in (
                    "layer2_critic_agent",
                    "layer2_critic_opp",
                    "critic_agent_mlp",
                    "critic_opp_mlp",
                )
            ):
                p.requires_grad = enabled

    def _body_and_pad_mask(
        self,
        feats: torch.Tensor,
        roles: torch.Tensor,
        body_mask: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """feats [B,Lb,9], roles [B,Lb], body_mask [B,Lb] True=valid → x [B,seq_len,D], pad [B,seq_len] True=pad."""
        B, Lb, _ = feats.shape
        device = feats.device
        tok = self.card_proj(feats) + self.role_emb(roles.clamp(min=0, max=NUM_ROLE_TYPES - 1))
        g = self.global_tokens.unsqueeze(0).expand(B, NUM_GLOBAL_TOKENS, -1)
        g = g + self.role_emb(
            torch.full((B, NUM_GLOBAL_TOKENS), ROLE_GLOBAL, device=device, dtype=torch.long)
        )
        x = torch.cat([tok, g], dim=1)
        assert x.shape[1] == seq_len, (x.shape[1], seq_len)
        glob_valid = torch.ones(B, NUM_GLOBAL_TOKENS, dtype=torch.bool, device=device)
        full_valid = torch.cat([body_mask.bool(), glob_valid], dim=1)
        pad_mask = ~full_valid
        return x, pad_mask

    def forward(
        self,
        obs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns match_logits [B,7*17], play_logits [B,4], v_agent [B], v_opp [B].
        Play logits only meaningful for play-phase rows (mask invalid slots).
        """
        af = obs["seq_agent_feats"]
        ar = obs["seq_agent_roles"].long()
        am = obs["seq_agent_mask"].bool()
        cf = obs["seq_critic_feats"]
        cr = obs["seq_critic_roles"].long()
        cm = obs["seq_critic_mask"].bool()

        xa, pad_a = self._body_and_pad_mask(af, ar, am, AGENT_SEQ_LEN)
        xc, pad_c = self._body_and_pad_mask(cf, cr, cm, CRITIC_SEQ_LEN)

        ha = self.layer1(xa, pad_a)
        hc = self.layer1(xc, pad_c)

        hm = self.layer2_match(ha, pad_a)
        hp = self.layer2_play(ha, pad_a)
        hca = self.layer2_critic_agent(hc, pad_c)
        hco = self.layer2_critic_opp(hc, pad_c)

        # --- Match head (§3.1): Q from hand slice, K from pool slice, /√32
        q = self.query_match(hm[:, 0:AGENT_HAND_END, :])
        k = self.key_match(hm[:, AGENT_POOL_START:AGENT_POOL_END, :])
        scores = torch.matmul(q, k.transpose(-1, -2)) / SQRT_D
        match_mask = obs["mask_match"].bool()
        scores = scores.masked_fill(~match_mask, -1e9)
        match_logits = scores.flatten(1, 2)

        play_h = self.play_score(hp[:, :MAX_PLAY_HAND, :]).squeeze(-1)
        play_logits = play_h.masked_fill(~obs["play_hand_valid"].bool(), -1e9)

        def _pooled(h: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
            valid = ~pad
            denom = valid.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            return (h * valid.unsqueeze(-1).float()).sum(dim=1) / denom

        v_agent = self.critic_agent_mlp(_pooled(hca, pad_c)).squeeze(-1)
        v_opp = self.critic_opp_mlp(_pooled(hco, pad_c)).squeeze(-1)

        return match_logits, play_logits, v_agent, v_opp
