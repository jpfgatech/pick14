"""
Sanity-check a ``pickq_*.pt`` checkpoint produced by ``05_train_timed.py``.

Checks
------
1. **Finite forward** — random tensor batch → ``q_pred``, ``opp_pred`` finite.
2. **Summary JSON** — optional ``--summary`` loads recorded metrics for reference.
3. **Exact validation replay** — if checkpoint contains ``perm_indices_cpu``
   (deterministic split saved since numpy-permutation update), rebuilds the same
   rollout ``(--games``, ``--seed``) as training, reapplies saved row permutation,
   reports ``val_q`` from loaded weights — must match ``final_val_q_loss`` within
   tolerance for bitwise-identical PyTorch loads.

Legacy checkpoints **without** ``perm_indices_cpu`` skip replay and print a warning.

Usage:
    python scripts/05_checkpoint_sanity.py \\
        --checkpoint train_runs_remote/pickq_20260428T143416Z.pt \\
        --summary train_runs_remote/pickq_20260428T143416Z.json \\
        --games 800 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, ".")

from pick14.rl.q_model import PickQNet
from pick14.rl.q_targets import backfill_targets, rollout_game
from pick14.rl.q_train import samples_to_tensors


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--summary", type=str, default="", help="Optional JSON from same run")
    p.add_argument("--games", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-4, help="Match tol for val replay")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    summary_data: dict | None = None
    if args.summary:
        summary_data = json.loads(Path(args.summary).read_text(encoding="utf-8"))

    device = torch.device("cpu")

    print("=== Checkpoint keys ===")
    print(sorted(blob.keys()))

    print("\n=== Finite forward ===")
    model = PickQNet()
    missing, unexpected = model.load_state_dict(blob["model_state"], strict=True)
    assert not missing and not unexpected
    model.eval()
    x_rand = torch.randn(8, 54, 27)
    with torch.no_grad():
        q, opp = model(x_rand)
    assert torch.isfinite(q).all() and torch.isfinite(opp).all()
    print(f"  OK  q_pred shape={tuple(q.shape)} opp_pred shape={tuple(opp.shape)}")

    if summary_data:
        print("\n=== Summary JSON ===")
        for k in (
            "epochs",
            "games",
            "samples",
            "final_train_q_loss",
            "final_val_q_loss",
            "final_val_cv",
            "stop_reason",
            "seed",
        ):
            if k in summary_data:
                print(f"  {k}: {summary_data[k]}")

    perm_cpu = blob.get("perm_indices_cpu")
    if perm_cpu is None:
        print("\n=== Validation replay ===")
        print(
            "  SKIP — checkpoint has no `perm_indices_cpu` (trained before split export)."
            "\n  Re-train once with current ``05_train_timed.py`` for reproducible replay."
        )
        print("\n=== Done (partial sanity only) ===")
        return

    exp_seed = blob.get("seed", args.seed)
    exp_games = blob.get("games", args.games)
    if summary_data:
        exp_seed = summary_data.get("seed", exp_seed)
        exp_games = summary_data.get("games", exp_games)

    from random import Random

    rng = Random(exp_seed)
    samples = []
    for _ in range(exp_games):
        samps = rollout_game(rng=rng)
        backfill_targets(samps)
        samples.extend(samps)

    x, qt, opp = samples_to_tensors(samples)
    perm_cpu = perm_cpu.long().flatten()
    n = x.shape[0]
    if perm_cpu.numel() != n:
        raise RuntimeError(f"perm length {perm_cpu.numel()} != N {n}")

    x_perm = x[perm_cpu]
    qt_perm = qt[perm_cpu]
    opp_perm = opp[perm_cpu]
    split = int(0.9 * n)
    x_va = x_perm[split:]
    qt_va = qt_perm[split:]

    model_cpu = PickQNet()
    model_cpu.load_state_dict(blob["model_state"], strict=True)
    model_cpu.eval()
    with torch.no_grad():
        q_pred, _ = model_cpu(x_va)
    val_q = float(nn.MSELoss()(q_pred, qt_va).item())

    mean_train = float(qt_perm[:split].mean())
    naive_val = float(((qt_va - mean_train) ** 2).mean().item())

    print("\n=== Validation replay (exact split) ===")
    print(f"  recomputed val_q (global MSE): {val_q:.6f}")
    print(f"  naive baseline (predict train mean target): {naive_val:.6f}")

    if summary_data and "final_val_q_loss" in summary_data:
        exp_val = float(summary_data["final_val_q_loss"])
        ok = abs(val_q - exp_val) < args.tol
        print(f"  summary final_val_q_loss:       {exp_val:.6f}")
        print(f"  MATCH within tol={args.tol}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
