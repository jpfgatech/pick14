"""
Shared training utilities for CP4 and CP5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pick14.rl.q_targets import TurnSample


def load_rollout_shards_numpy(
    rollout_dir: str | Path,
    *,
    load_segment: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all ``shard_*.npz`` files produced by ``scripts/05_rollout_mass.py``.

    Each shard contains arrays ``x`` ((N, 54, 27)), ``qt`` ((N, 1)), ``opp`` ((N, 54)).
    Optional ``seg`` ((N,), uint8): ``0`` = stem row, ``1`` = fork tail row).
    Float16 shards are promoted to float32 for training.

    Parameters
    ----------
    rollout_dir
        Directory containing ``shard_XXXXX.npz`` files.
    load_segment : bool
        If True, load optional ``seg`` column from each shard (must be present).

    Returns
    -------
    x, qt, opp[, seg]
        Concatenated NumPy arrays, row-aligned. ``seg`` only when ``load_segment=True``.
    """
    root = Path(rollout_dir)
    paths = sorted(root.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No shard_*.npz files under {root.resolve()} — run scripts/05_rollout_mass.py first."
        )
    xs: list[np.ndarray] = []
    qts: list[np.ndarray] = []
    opps: list[np.ndarray] = []
    segs: list[np.ndarray] = []
    want_seg = load_segment

    for p in paths:
        z = np.load(p)
        xs.append(np.asarray(z["x"]))
        qts.append(np.asarray(z["qt"]))
        opps.append(np.asarray(z["opp"]))
        if want_seg:
            if "seg" not in z.files:
                raise FileNotFoundError(f"shard missing seg column: {p}")
            segs.append(np.asarray(z["seg"]))
    x = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
    qt = np.concatenate(qts, axis=0).astype(np.float32, copy=False)
    opp = np.concatenate(opps, axis=0).astype(np.float32, copy=False)
    if qt.ndim == 1:
        qt = qt.reshape(-1, 1)
    if want_seg:
        seg = np.concatenate(segs, axis=0).astype(np.uint8, copy=False)
        return x, qt, opp, seg
    return x, qt, opp


def samples_to_tensors(
    samples: list[TurnSample],
    *,
    include_segment: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """
    Convert a list of TurnSamples to tensors ready for training:
        x         : (N, 54, 27)  — state tensors
        q_targets : (N, 1)       — Q regression targets
        opp_hand  : (N, 54)      — opponent hand binary targets
        seg       : (N,) uint8    — optional; ``0`` stem, ``1`` fork tail
    """
    x = torch.from_numpy(np.stack([s.state_tensor for s in samples])).float()
    qt = torch.tensor([[s.q_target] for s in samples], dtype=torch.float32)
    opp = torch.from_numpy(np.stack([s.opp_hand_target for s in samples])).float()
    if not include_segment:
        return x, qt, opp
    seg = torch.tensor(
        [0 if getattr(s, "segment", "stem") == "stem" else 1 for s in samples],
        dtype=torch.uint8,
    )
    return x, qt, opp, seg


def global_val_q_mse(
    model: nn.Module,
    x_va: torch.Tensor,
    qt_va: torch.Tensor,
    microbatch: int = 8192,
) -> float:
    """
    Global validation MSE ``mean((q_pred - qt)^2)`` over all rows, computed in
    micro-batches so large CPU tensors need not occupy GPU VRAM at once.
    """
    model.eval()
    compute_dev = next(model.parameters()).device
    n = x_va.shape[0]
    total = 0.0
    with torch.no_grad():
        for start in range(0, n, microbatch):
            xb = x_va[start:start + microbatch].to(compute_dev, non_blocking=True)
            qtb = qt_va[start:start + microbatch].to(compute_dev, non_blocking=True)
            q_pred, _ = model(xb)
            total += nn.MSELoss(reduction="sum")(q_pred, qtb).item()
    return total / float(n)


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
    compute_dev = next(model.parameters()).device
    perm = torch.randperm(N, device=x.device)
    x, qt, opp = x[perm], qt[perm], opp[perm]

    q_losses, opp_losses = [], []
    for start in range(0, N, batch_size):
        xb = x[start:start + batch_size].to(compute_dev, non_blocking=True)
        qtb = qt[start:start + batch_size].to(compute_dev, non_blocking=True)
        ob = opp[start:start + batch_size].to(compute_dev, non_blocking=True)

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
