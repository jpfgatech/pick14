"""
CP4 — Overfit a single game.

Trains on the ~30 samples from one game for 500 epochs.
Loss must converge to near-zero; failure means the training loop or
architecture is broken before we scale up.

Usage:
    python scripts/05_cp4_overfit.py [seed]
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from random import Random

import torch

from pick14.rl.q_model import PickQNet
from pick14.rl.q_targets import backfill_targets, rollout_game
from pick14.rl.q_train import samples_to_tensors, train_epoch


def run(seed: int = 0, n_epochs: int = 500, lr: float = 1e-3) -> None:
    rng = Random(seed)
    samples = rollout_game(rng=rng)
    backfill_targets(samples)
    print(f"Game samples: {len(samples)}")

    x, qt, opp = samples_to_tensors(samples)
    targets = qt.squeeze()
    print(f"Q target range: [{targets.min():.3f}, {targets.max():.3f}]  std={targets.std():.3f}")

    model = PickQNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nOverfitting {len(samples)} samples for {n_epochs} epochs …\n")
    print(f"  {'Epoch':>6}  {'q_loss':>10}  {'opp_loss':>10}  {'total':>10}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}")

    log_epochs = set([1, 10, 50, 100, 200, 300, 400, 500, n_epochs])
    for epoch in range(1, n_epochs + 1):
        stats = train_epoch(model, x, qt, opp, optimizer, batch_size=len(samples))
        if epoch in log_epochs:
            print(
                f"  {epoch:>6}  {stats['q_loss']:>10.5f}"
                f"  {stats['opp_loss']:>10.5f}  {stats['total_loss']:>10.5f}"
            )

    final_q_loss = stats["q_loss"]
    print(f"\n  Final Q loss: {final_q_loss:.5f}")
    threshold = 0.5
    status = "PASS ✓" if final_q_loss < threshold else f"FAIL ✗ (threshold {threshold})"
    print(f"  CP4 verdict: {status}\n")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run(seed=seed)
