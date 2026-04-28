"""
CP2 — Direct-Q network architecture (instructions/05.md §Latent Space / Attention / Q head).

Architecture
------------
Input: (batch, 54, 27)  — one row per canonical card.

1. **Projection**   Linear(27 → 32)  applied per card  → (batch, 54, 32)
2. **Attention**    n_layers × TransformerEncoderLayer(d=32, heads=4, ffn=128)
3. **Q head**       32 learnable pool weights → dot-product collapse → (batch, 54)
                    then MLP  54 → 128 → 1  with ReLU
4. **Opp hand head** separate 32 pool weights → (batch, 54) → Sigmoid
                    predicts P(each card is in opponent's hand)

The two heads use independent collapse weights so the latent space is not
forced to serve both objectives from the same projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CARDS = 54
IN_CHANNELS = 27
LATENT_DIM = 32
N_HEADS = 4
FFN_DIM = 128
N_LAYERS = 2


class AttentionBlock(nn.Module):
    """Single Transformer encoder layer (pre-norm variant for training stability)."""

    def __init__(
        self,
        d_model: int = LATENT_DIM,
        n_heads: int = N_HEADS,
        ffn_dim: int = FFN_DIM,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,   # (batch, seq, d_model)
            norm_first=True,    # pre-norm: more stable for shallow stacks
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class PickQNet(nn.Module):
    """
    Direct Q-network for Pick14 (2-player).

    Parameters
    ----------
    n_layers : int
        Number of stacked attention blocks (default 2 per instructions).
    dropout : float
        Dropout applied inside attention/FFN layers (0 = off, good for small data).
    """

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        latent: int = LATENT_DIM,
        n_heads: int = N_HEADS,
        ffn_dim: int = FFN_DIM,
        n_layers: int = N_LAYERS,
        n_cards: int = N_CARDS,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_cards = n_cards
        self.latent = latent

        # ── Input projection ──────────────────────────────────────────────
        self.proj = nn.Linear(in_channels, latent)

        # ── Attention stack ───────────────────────────────────────────────
        self.attn = nn.ModuleList(
            [AttentionBlock(latent, n_heads, ffn_dim, dropout) for _ in range(n_layers)]
        )

        # ── Q head ────────────────────────────────────────────────────────
        # Learnable per-latent-dim weights to collapse (batch, 54, 32) → (batch, 54)
        # by a dot product along the latent axis.
        self.q_pool_w = nn.Parameter(torch.ones(latent) / latent)
        self.q_mlp = nn.Sequential(
            nn.Linear(n_cards, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, 1),
        )

        # ── Opponent hand head ────────────────────────────────────────────
        # Separate collapse weights, then sigmoid per card.
        self.opp_pool_w = nn.Parameter(torch.ones(latent) / latent)
        # No extra MLP — the per-card sigmoid of the collapsed logits is enough.

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        for lin in [self.q_mlp[0], self.q_mlp[2]]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, 54, 27)

        Returns
        -------
        q_value   : (batch, 1)   — scalar Q estimate
        opp_hand  : (batch, 54)  — opponent hand probability per card
        """
        # (batch, 54, 32)
        h = self.proj(x)
        for block in self.attn:
            h = block(h)

        # Q head: weighted sum over latent dim → (batch, 54)
        q_pooled = h @ self.q_pool_w          # (batch, 54)
        q_value = self.q_mlp(q_pooled)        # (batch, 1)

        # Opp hand head: separate weights → sigmoid → (batch, 54)
        opp_logits = h @ self.opp_pool_w      # (batch, 54)
        opp_hand = torch.sigmoid(opp_logits)  # (batch, 54)

        return q_value, opp_hand

    def predict_q(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: return only the Q value. Shape (batch, 1)."""
        q, _ = self(x)
        return q

    @torch.no_grad()
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
