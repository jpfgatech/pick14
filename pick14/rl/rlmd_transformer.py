"""rl.md §2.2: one Transformer encoder layer (MHA + LN + FFN GELU + LN)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RLmdEncoderLayer(nn.Module):
    def __init__(self, d_model: int = 32, nhead: int = 4, dim_ff: int = 128, dropout: float = 0.0):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must divide nhead")
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True, bias=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        """
        x: [B, L, D]
        key_padding_mask: [B, L] with True = ignore (padding)
        """
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ff = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff))
        return x
