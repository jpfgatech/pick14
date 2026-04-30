"""Quick GPU vs CPU timing for PickQNet training (remote sanity)."""

from __future__ import annotations

import argparse
import sys
import time
from random import Random

import torch

sys.path.insert(0, ".")

from pick14.rl.q_model import PickQNet
from pick14.rl.q_targets import backfill_targets, rollout_game
from pick14.rl.q_train import samples_to_tensors, train_epoch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    args = p.parse_args()

    if args.cpu or not torch.cuda.is_available():
        dev = torch.device("cpu")
        print("device: CPU")
    else:
        dev = torch.device("cuda")
        print("device:", torch.cuda.get_device_name(0))

    samples: list = []
    rng = Random(42)
    t_roll = time.perf_counter()
    for _ in range(args.games):
        s = rollout_game(rng=rng)
        backfill_targets(s)
        samples.extend(s)
    print(f"rollout {args.games} games: {len(samples)} samples in {time.perf_counter() - t_roll:.3f}s")

    x, qt, opp = samples_to_tensors(samples)
    x, qt, opp = x.to(dev), qt.to(dev), opp.to(dev)

    model = PickQNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Warmup
    train_epoch(model, x, qt, opp, opt, batch_size=args.batch)
    torch.cuda.synchronize() if dev.type == "cuda" else None

    t0 = time.perf_counter()
    for _ in range(args.epochs):
        train_epoch(model, x, qt, opp, opt, batch_size=args.batch)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"train {args.epochs} epochs (batch={args.batch}): {elapsed:.3f}s")
    print(f"  per epoch: {elapsed / args.epochs * 1000:.1f} ms")


if __name__ == "__main__":
    main()
