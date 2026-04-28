"""
CP5 — Train on 20 games, print loss curve, compare to naive baseline.

Splits 20 games into train (16) and val (4).  Trains for 50 epochs.
Checks that val Q-loss beats a naive predictor (always predict the mean target).

Usage:
    python scripts/05_cp5_loss_curve.py [n_games [n_epochs]]
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from random import Random

import numpy as np
import torch

from pick14.rl.q_model import PickQNet
from pick14.rl.q_targets import backfill_targets, rollout_game
from pick14.rl.q_train import samples_to_tensors, train_epoch


def run(n_games: int = 20, n_epochs: int = 50, seed: int = 0) -> None:
    rng = Random(seed)

    print(f"Collecting {n_games} games …")
    all_samples = []
    for _ in range(n_games):
        samps = rollout_game(rng=rng)
        backfill_targets(samps)
        all_samples.extend(samps)

    np.random.seed(seed)
    perm = np.random.permutation(len(all_samples))
    split = int(0.8 * len(all_samples))
    train_samps = [all_samples[i] for i in perm[:split]]
    val_samps   = [all_samples[i] for i in perm[split:]]
    print(f"Train samples: {len(train_samps)}   Val samples: {len(val_samps)}")

    x_tr, qt_tr, opp_tr = samples_to_tensors(train_samps)
    x_va, qt_va, opp_va = samples_to_tensors(val_samps)

    # Naive baseline: always predict the training-set mean target.
    mean_target = qt_tr.mean().item()
    baseline_mse = float(((qt_va - mean_target) ** 2).mean())
    print(f"Naive baseline val MSE (predict mean={mean_target:+.3f}): {baseline_mse:.4f}\n")

    model = PickQNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"  {'Epoch':>5}  {'tr_q_loss':>10}  {'val_q_loss':>11}  {'vs_baseline':>12}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*11}  {'─'*12}")

    log_at = {1, 5, 10, 20, 30, 40, 50, n_epochs}
    last_val_q = None
    for epoch in range(1, n_epochs + 1):
        stats = train_epoch(model, x_tr, qt_tr, opp_tr, optimizer, batch_size=32)

        model.eval()
        with torch.no_grad():
            q_pred_val, _ = model(x_va)
            val_q = float(torch.nn.MSELoss()(q_pred_val, qt_va).item())
        model.train()
        last_val_q = val_q

        if epoch in log_at:
            ratio = val_q / baseline_mse if baseline_mse > 0 else float("nan")
            symbol = "▼" if ratio < 1.0 else " "
            print(
                f"  {epoch:>5}  {stats['q_loss']:>10.4f}  {val_q:>11.4f}"
                f"  {symbol}{ratio:>10.3f}x"
            )

    print(f"\n  Final val Q-loss : {last_val_q:.4f}")
    print(f"  Naive baseline   : {baseline_mse:.4f}")
    verdict = "PASS ✓" if last_val_q < baseline_mse else "FAIL ✗  (did not beat baseline)"
    print(f"  CP5 verdict      : {verdict}\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    n_games  = int(args[0]) if len(args) > 0 else 20
    n_epochs = int(args[1]) if len(args) > 1 else 50
    run(n_games=n_games, n_epochs=n_epochs)
