"""
Estimate PickQNet training throughput (forward + backward + optimizer).

Run on the training host to compare CPU vs GPU::

    python scripts/05_benchmark_train.py
    CUDA_VISIBLE_DEVICES=0 python scripts/05_benchmark_train.py

Extrapolates wall time for CP4/CP5/CP6-style runs from measured epoch time.

Usage:
    python scripts/05_benchmark_train.py [--epochs N] [--samples N]
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, ".")

import torch
import torch.nn as nn

from pick14.rl.q_model import PickQNet
from pick14.rl.q_targets import backfill_targets, rollout_game
from pick14.rl.q_train import samples_to_tensors, train_epoch
from random import Random


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_forward(
    model: nn.Module,
    device: torch.device,
    batch: int,
    n_iters: int = 100,
    warmup: int = 10,
) -> float:
    """Seconds for n_iters forward passes, batch x (54, 27)."""
    model.eval()
    x = torch.randn(batch, 54, 27, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    _sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(x)
    _sync(device)
    return (time.perf_counter() - t0) / n_iters


def bench_epoch(
    model: nn.Module,
    x: torch.Tensor,
    qt: torch.Tensor,
    opp: torch.Tensor,
    device: torch.device,
    batch_size: int,
    n_epochs: int,
) -> float:
    """Total seconds for n_epochs full passes."""
    model = model.to(device)
    x, qt, opp = x.to(device), qt.to(device), opp.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n_epochs):
        train_epoch(model, x, qt, opp, opt, batch_size=batch_size)
    _sync(device)
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=600, help="Rollout samples to stack (default ~20 games)")
    p.add_argument("--epochs", type=int, default=5, help="Epochs for timing loop")
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    print("=" * 60)
    print("  PickQNet training benchmark")
    print("=" * 60)
    print(f"  torch {torch.__version__}")
    print(f"  device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if cuda else ""))
    print()

    model = PickQNet().to(device)
    for bs in (8, 32, 64):
        t = bench_forward(model, device, bs, n_iters=50, warmup=5)
        print(f"  forward  batch={bs:3d}  {t*1000:.3f} ms/iter")

    print("\n  Building dataset …")
    rng = Random(0)
    samples = []
    while len(samples) < args.samples:
        s = rollout_game(rng=rng)
        backfill_targets(s)
        samples.extend(s)
    samples = samples[: args.samples]
    x, qt, opp = samples_to_tensors(samples)
    print(f"  samples={len(samples)}  tensor shape x={tuple(x.shape)}")

    print(f"\n  Timing {args.epochs} full train epochs (batch={args.batch}) …")
    sec = bench_epoch(PickQNet(), x, qt, opp, device, args.batch, args.epochs)
    per_epoch = sec / args.epochs
    print(f"  total {sec:.3f}s  →  {per_epoch*1000:.1f} ms/epoch")

    # Reference run sizes (from scripts)
    print("\n  Extrapolated wall time (linear in epochs × dataset passes):")
    refs = [
        ("CP4 overfit", 30, 500, args.batch),
        ("CP5 curve", len(samples), 50, args.batch),
        ("CP6 train", 1533, 80, 32),  # typical ~50 games
    ]
    for name, n_samp, n_ep, bs in refs:
        # scale epoch time by (n_samp / len(samples)) * (n_ep / args.epochs) is wrong;
        # epoch time ~ O(ceil(N/bs) * forward_backward) ≈ linear in N for fixed bs.
        ratio = (n_samp / len(samples)) * (n_ep / args.epochs)
        est = per_epoch * ratio
        print(f"    {name:14s}  ~{est:6.1f}s  (N={n_samp}, epochs={n_ep})")

    print("=" * 60)


if __name__ == "__main__":
    main()
