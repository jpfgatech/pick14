"""
Shared training utilities for CP4 and CP5.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pick14.rl.q_targets import TurnSample


def samples_to_tensors(
    samples: list[TurnSample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a list of TurnSamples to three tensors ready for training:
        x         : (N, 54, 27)  — state tensors
        q_targets : (N, 1)       — Q regression targets
        opp_hand  : (N, 54)      — opponent hand binary targets
    """
    import numpy as np

    x   = torch.from_numpy(np.stack([s.state_tensor    for s in samples])).float()
    qt  = torch.tensor([[s.q_target]                   for s in samples], dtype=torch.float32)
    opp = torch.from_numpy(np.stack([s.opp_hand_target for s in samples])).float()
    return x, qt, opp


def train_epoch(
    model: nn.Module,
    x: torch.Tensor,
    qt: torch.Tensor,
    opp: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 16,
    opp_weight: float = 0.1,
) -> dict[str, float]:
    """
    One epoch of training (full pass over the dataset).

    Returns a dict with keys ``q_loss``, ``opp_loss``, ``total_loss``.
    """
    model.train()
    N = x.shape[0]
    perm = torch.randperm(N)
    x, qt, opp = x[perm], qt[perm], opp[perm]

    q_losses, opp_losses = [], []
    for start in range(0, N, batch_size):
        xb  = x[start:start + batch_size]
        qtb = qt[start:start + batch_size]
        ob  = opp[start:start + batch_size]

        optimizer.zero_grad()
        q_pred, opp_pred = model(xb)
        ql = nn.MSELoss()(q_pred, qtb)
        ol = nn.BCELoss()(opp_pred, ob)
        loss = ql + opp_weight * ol
        loss.backward()
        optimizer.step()

        q_losses.append(ql.item())
        opp_losses.append(ol.item())

    return {
        "q_loss":    float(torch.tensor(q_losses).mean()),
        "opp_loss":  float(torch.tensor(opp_losses).mean()),
        "total_loss": float(torch.tensor(q_losses).mean() + opp_weight * torch.tensor(opp_losses).mean()),
    }
