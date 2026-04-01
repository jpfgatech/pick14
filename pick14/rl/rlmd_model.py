"""
rl.md §2–3: one shared Transformer layer, four independent second layers (Match, Play,
Critic-Agent, Critic-Opponent), pointer match head with :math:`1/\\sqrt{d}` scaling,
play head ``Linear(32,1)``, dual critics mean-pool + ``MLP(32→16→1)``.

**Training:** actor + critic losses are summed in the trainer and backpropped once per step, so gradients
flow **recursively** through the shared first layer from all branches (multi-task learning).

Dimensions follow :mod:`pick14.rl.rlmd_sequences` and :class:`~pick14.rl.env.Pick14GymEnv`
(``match_flat_dim = (2**n_hand-1+1) * MAX_PUBLIC_SLOTS``).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from pick14.rl.rlmd_sequences import (
    NUM_GLOBAL_TOKENS,
    NUM_ROLE_TYPES,
    ROLE_GLOBAL,
    agent_hand_token_slots,
    agent_seq_len,
    critic_seq_len,
    max_hand_slots,
    max_match_combo_slots,
)
from pick14.rl.rlmd_transformer import RLmdEncoderLayer
from pick14.rl.sim_core import MAX_PUBLIC_SLOTS

SQRT_D = math.sqrt(32.0)


class RLmdPPOAgent(nn.Module):
    """Policy + dual critics per rl.md (2-player default)."""

    def __init__(self, n_hand: int = 3, dropout: float = 0.0):
        super().__init__()
        if n_hand < 1:
            raise ValueError("n_hand must be >= 1")
        self.n_hand = int(n_hand)
        self.agent_hand_slots = agent_hand_token_slots(self.n_hand)
        self.max_play_hand = max_hand_slots(self.n_hand)
        self.max_match_rows = max_match_combo_slots(self.n_hand) + 1
        self.match_flat_dim = self.max_match_rows * MAX_PUBLIC_SLOTS
        d = 32

        self.card_proj = nn.Linear(9, d)
        self.role_emb = nn.Embedding(NUM_ROLE_TYPES, d)
        self.global_tokens = nn.Parameter(torch.zeros(NUM_GLOBAL_TOKENS, d))
        nn.init.normal_(self.global_tokens, std=0.02)

        # §2.2 — one first layer, shared by agent and critic sequences (two forwards, same weights).
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

    def _embed_sequence(
        self,
        feats: torch.Tensor,
        roles: torch.Tensor,
        body_mask: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``feats`` [B,Lb,9], ``roles`` [B,Lb], ``body_mask`` [B,Lb] True=valid → x [B,seq_len,D], pad [B,seq_len]."""
        B, Lb, _ = feats.shape
        device = feats.device
        tok = self.card_proj(feats) + self.role_emb(roles.clamp(min=0, max=NUM_ROLE_TYPES - 1))
        g = self.global_tokens.unsqueeze(0).expand(B, NUM_GLOBAL_TOKENS, -1)
        g = g + self.role_emb(
            torch.full((B, NUM_GLOBAL_TOKENS), ROLE_GLOBAL, device=device, dtype=torch.long)
        )
        x = torch.cat([tok, g], dim=1)
        if x.shape[1] != seq_len:
            raise RuntimeError(f"sequence length mismatch: got {x.shape[1]}, expected {seq_len}")
        glob_valid = torch.ones(B, NUM_GLOBAL_TOKENS, dtype=torch.bool, device=device)
        full_valid = torch.cat([body_mask.bool(), glob_valid], dim=1)
        pad_mask = ~full_valid
        return x, pad_mask

    def forward(
        self,
        obs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        match_logits
            ``[B, match_flat_dim]`` — row-major ``(match_row, public_col)`` aligned with the env.
        play_logits
            ``[B, max_play_hand]`` — mask with ``play_hand_valid`` outside the model.
        v_agent, v_opp
            Scalar estimates per batch row from the two critic stacks (§3.3).
        """
        af = obs["seq_agent_feats"]
        ar = obs["seq_agent_roles"].long()
        am = obs["seq_agent_mask"].bool()
        cf = obs["seq_critic_feats"]
        cr = obs["seq_critic_roles"].long()
        cm = obs["seq_critic_mask"].bool()

        as_len = agent_seq_len(self.n_hand)
        cs_len = critic_seq_len(self.n_hand)
        xa, pad_a = self._embed_sequence(af, ar, am, as_len)
        xc, pad_c = self._embed_sequence(cf, cr, cm, cs_len)

        ha = self.layer1(xa, pad_a)
        hc = self.layer1(xc, pad_c)

        hm = self.layer2_match(ha, pad_a)
        hp = self.layer2_play(ha, pad_a)
        hca = self.layer2_critic_agent(hc, pad_c)
        hco = self.layer2_critic_opp(hc, pad_c)

        hs = self.agent_hand_slots
        pool_end = hs + MAX_PUBLIC_SLOTS
        q = self.query_match(hm[:, :hs, :])
        k = self.key_match(hm[:, hs:pool_end, :])
        scores = torch.matmul(q, k.transpose(-1, -2)) / SQRT_D
        match_mask = obs["mask_match"].bool()
        scores = scores.masked_fill(~match_mask, -1e9)
        match_logits = scores.flatten(1, 2)

        play_h = self.play_score(hp[:, : self.max_play_hand, :]).squeeze(-1)
        play_logits = play_h.masked_fill(~obs["play_hand_valid"].bool(), -1e9)

        def _pooled(h: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
            valid = ~pad
            denom = valid.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            return (h * valid.unsqueeze(-1).float()).sum(dim=1) / denom

        v_agent = self.critic_agent_mlp(_pooled(hca, pad_c)).squeeze(-1)
        v_opp = self.critic_opp_mlp(_pooled(hco, pad_c)).squeeze(-1)

        return match_logits, play_logits, v_agent, v_opp

    @classmethod
    def from_env(cls, env: Any, dropout: float = 0.0) -> RLmdPPOAgent:
        """Build with ``n_hand`` consistent with a :class:`~pick14.rl.env.Pick14GymEnv`."""
        return cls(n_hand=int(env.n_hand), dropout=dropout)
