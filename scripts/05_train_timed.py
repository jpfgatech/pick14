"""
Train PickQNet on a fixed rollout dataset until **validation Q-loss stabilizes**
(or a safety cap is hit). Optional wall-clock cap.

Stopping (default)
------------------
After ``--min-epochs``, each epoch we look at the last ``--stable-window``
validation MSE values. If the coefficient of variation
``std(window) / (mean(window)+1e-8)`` falls below ``--stable-rel``, training
stops.

Safety caps (optional): ``--max-epochs`` (default 50_000), ``--max-minutes`` (0 = off).

Uses CUDA when available. Saves checkpoint + JSON summary.

Definitions (precise)
---------------------

**Regression targets**

Each ``TurnSample`` has scalar ``q_target``: **future-only** discounted score-gap
vs the concurrent opponent — ``Σ_{h=1..H} γ^h · gap[j+h]`` in chronological turn
order (default γ = 0.9, ``H`` = ``Q_TARGET_HORIZON_TURNS``, typically **4** future
steps ≈ two rounds). The gap on the **current** timestep ``j`` is excluded, so
immediate match capture points never appear in ``qt``; add those via ``immediate_pts``
in ``direct_q`` at decision time.

**Forward**

``PickQNet(x) -> (q_pred, opp_pred)`` with ``x`` shape ``(batch, 54, 27)``,
``q_pred`` shape ``(batch, 1)``.

**``tr_q``**

From ``train_epoch``: shuffle rows, partition into batches of ``batch_size``.
Per batch::

    Q_batch = mean_i ( q_pred_i − qt_i )²

Then::

    tr_q = mean_batches(Q_batch)

So ``tr_q`` is the **mean of per-batch Q-MSEs** (each batch uses only its rows).
``tr_rmse = sqrt(tr_q)`` is reported next to it (same for ``val_rmse`` vs ``val_q``).
If ``N_train % batch_size != 0``, this differs slightly from one global mean over
all training rows.

Auxiliary **opponent-head BCE** trains in the same backward step but is **not**
included in ``tr_q``.

**``val_q``**

After ``train_epoch``, ``model.eval()`` — single forward on **all** validation rows::

    val_q = mean_j ( q_pred_j − qt_j )²      # global MSE over ``x_va``
    val_rmse = sqrt(val_q)

**``val_cv``**

Let ``v[e]`` be ``val_q`` after epoch ``e``. Let ``W = stable-window``.
While fewer than ``W`` epochs have completed, ``val_cv`` prints ``n/a``.
Otherwise::

    window = v[e-W+1 … e]
    val_cv = std(window) / (mean(window) + 1e-8)

So ``val_cv`` is **relative jitter of the epoch-wise validation curve**, not
variance across validation samples within one epoch.

Usage:
    python scripts/05_train_timed.py --games 800
    python scripts/05_train_timed.py --rollout-dir rollout_data
    python scripts/05_train_timed.py --rollout-dir rollout_data --epochs 50
    python scripts/05_train_timed.py --games 2000 --stable-window 60 --stable-rel 0.015
    python scripts/05_train_timed.py --games 800 --max-minutes 5

If ``--rollout-dir`` is set, loads ``shard_*.npz`` written by ``scripts/05_rollout_mass.py``
and ignores ``--games``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from random import Random

import numpy as np
import torch

sys.path.insert(0, ".")

from pick14.rl.q_model import PickQNet
from pick14.rl.q_targets import (
    GAMMA,
    Q_TARGET_HORIZON_TURNS,
    backfill_targets,
    rollout_game,
)
from pick14.rl.q_train import (
    global_val_opp_bce,
    global_val_q_mse,
    load_rollout_shards_numpy,
    samples_to_tensors,
    train_epoch,
)


def _tee_log(path: Path) -> None:
    """Append stdout/stderr to *path* (line-buffered) so detached runs leave a trace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    log_f = path.open("a", encoding="utf-8", buffering=1)

    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)


def _val_cv(val_losses: list[float], window: int) -> float | None:
    """Coefficient of variation of the last *window* validation losses; None if too short."""
    if len(val_losses) < window:
        return None
    w = np.array(val_losses[-window:], dtype=np.float64)
    m = float(np.mean(w)) + 1e-8
    return float(np.std(w) / m)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--games",
        type=int,
        default=800,
        help="Live rollout_game count (ignored when --rollout-dir is set)",
    )
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--min-epochs",
        type=int,
        default=80,
        help="Do not apply stability stop before this many epochs",
    )
    p.add_argument(
        "--stable-window",
        type=int,
        default=40,
        help="Trailing val-loss window for stability (epochs)",
    )
    p.add_argument(
        "--stable-rel",
        type=float,
        default=0.02,
        help="Stop when std(val[-W:]) / mean(val[-W:]) < this (coeff. of variation)",
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=50_000,
        help="Hard stop even if not stable",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="If >0, run exactly this many epochs (sets max-epochs; disables early "
        "stability stop so the run does not exit early)",
    )
    p.add_argument(
        "--max-minutes",
        type=float,
        default=0.0,
        help="Optional wall-clock cap (0 = no limit)",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print progress every N epochs (1 = each epoch)",
    )
    p.add_argument(
        "--opp-weight",
        type=float,
        default=0.1,
        help="Weight on opponent-hand BCE in the combined training loss",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="train_runs",
        help="Directory for checkpoint + summary (under cwd)",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default="",
        help="If set, tee stdout/stderr to this path (relative to cwd)",
    )
    p.add_argument(
        "--rollout-dir",
        type=str,
        default="",
        help="Directory with shard_*.npz from scripts/05_rollout_mass.py (if set, skips live rollout)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="After loading shards, randomly subsample to at most N rows (0 = use all)",
    )
    p.add_argument(
        "--val-batch",
        type=int,
        default=8192,
        help="Micro-batch size for validation MSE (for large CPU-held tensors)",
    )
    args = p.parse_args()
    if args.epochs > 0:
        args.max_epochs = args.epochs
        # Prevent stability early-stop before the fixed epoch budget completes.
        args.min_epochs = args.epochs + 999_999

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    if args.log_file:
        _tee_log(Path(args.log_file))

    rng = Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{datetime.now(timezone.utc).isoformat()}] device={device}", flush=True)
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    if args.epochs > 0:
        print(
            f"Fixed epoch run: --epochs={args.epochs} (early stability stop disabled)",
            flush=True,
        )

    rollout_from_disk = bool(args.rollout_dir.strip())
    n_shard_files = 0
    rollout_dir_resolved: Path | None = None

    if rollout_from_disk:
        rollout_dir_resolved = Path(args.rollout_dir.strip())
        if not rollout_dir_resolved.is_absolute():
            rollout_dir_resolved = (repo_root / rollout_dir_resolved).resolve()
        else:
            rollout_dir_resolved = rollout_dir_resolved.resolve()
        n_shard_files = len(sorted(rollout_dir_resolved.glob("shard_*.npz")))
        print(
            f"Loading rollout shards from {rollout_dir_resolved} ({n_shard_files} files) ...",
            flush=True,
        )
        t0 = time.perf_counter()
        x_np, qt_np, opp_np = load_rollout_shards_numpy(rollout_dir_resolved)
        n_loaded = x_np.shape[0]
        if args.max_samples > 0 and n_loaded > args.max_samples:
            pick = np.random.RandomState(args.seed).choice(
                n_loaded, size=args.max_samples, replace=False
            )
            x_np, qt_np, opp_np = x_np[pick], qt_np[pick], opp_np[pick]
            print(
                f"  subsampled {args.max_samples} / {n_loaded} rows (seed={args.seed})",
                flush=True,
            )
        load_s = time.perf_counter() - t0
        print(f"Dataset: {x_np.shape[0]} samples from disk in {load_s:.1f}s", flush=True)
        x = torch.from_numpy(x_np)
        qt = torch.from_numpy(qt_np)
        opp = torch.from_numpy(opp_np)
    else:
        print(f"Rolling out {args.games} games ...", flush=True)
        t0 = time.perf_counter()
        all_samples: list = []
        for g in range(args.games):
            samps = rollout_game(rng=rng)
            backfill_targets(samps)
            all_samples.extend(samps)
            if (g + 1) % 200 == 0:
                print(f"  ... {g+1}/{args.games} games, {len(all_samples)} samples", flush=True)
        roll_s = time.perf_counter() - t0
        print(f"Dataset: {len(all_samples)} samples in {roll_s:.1f}s", flush=True)
        x, qt, opp = samples_to_tensors(all_samples)

    if rollout_from_disk:
        # Keep full dataset on CPU; train_epoch moves mini-batches to GPU (fits 8 GB VRAM + multi-M row data).
        x, qt, opp = x.cpu(), qt.cpu(), opp.cpu()
    else:
        x, qt, opp = x.to(device), qt.to(device), opp.to(device)

    n = x.shape[0]
    perm_np = np.random.RandomState(args.seed).permutation(n)
    perm = torch.as_tensor(perm_np, dtype=torch.long, device=x.device)
    split = int(0.9 * n)
    tr_idx, va_idx = perm[:split], perm[split:]
    x_tr, qt_tr, opp_tr = x[tr_idx], qt[tr_idx], opp[tr_idx]
    x_va, qt_va, opp_va = x[va_idx], qt[va_idx], opp[va_idx]

    model = PickQNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ckpt_path = out_dir / f"pickq_{stamp}.pt"
    log_path = out_dir / f"pickq_{stamp}.json"

    epoch = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_opp_losses: list[float] = []
    val_opp_losses: list[float] = []
    wall_start = time.perf_counter()
    stop_reason = "max_epochs"
    budget_s = args.max_minutes * 60.0 if args.max_minutes > 0 else None

    if args.epochs > 0:
        print(
            f"Training for exactly {args.epochs} epochs (stability stop disabled), "
            f"batch={args.batch} ...",
            flush=True,
        )
    else:
        print(
            f"Training until val loss stable (cv < {args.stable_rel} over "
            f"{args.stable_window} epochs after {args.min_epochs} epochs), "
            f"or max_epochs={args.max_epochs}"
            + (f", or max_minutes={args.max_minutes}" if budget_s else "")
            + f", batch={args.batch} ...",
            flush=True,
        )

    while epoch < args.max_epochs:
        if budget_s is not None and (time.perf_counter() - wall_start) >= budget_s:
            stop_reason = "max_minutes"
            break

        stats = train_epoch(
            model,
            x_tr,
            qt_tr,
            opp_tr,
            opt,
            batch_size=args.batch,
            opp_weight=args.opp_weight,
        )
        epoch += 1
        train_losses.append(stats["q_loss"])
        train_opp_losses.append(stats["opp_loss"])

        model.eval()
        val_q = global_val_q_mse(model, x_va, qt_va, microbatch=args.val_batch)
        val_opp = global_val_opp_bce(model, x_va, opp_va, microbatch=args.val_batch)
        val_losses.append(val_q)
        val_opp_losses.append(val_opp)
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()

        cv = _val_cv(val_losses, args.stable_window)
        if (
            epoch >= args.min_epochs
            and cv is not None
            and cv < args.stable_rel
        ):
            stop_reason = "stable_val_cv"
            print(
                f"  Stability reached at epoch={epoch}: val_cv={cv:.5f} < {args.stable_rel} "
                f"(tr_q={stats['q_loss']:.4f} tr_rmse={math.sqrt(max(0.0, stats['q_loss'])):.4f} "
                f"tr_opp={stats['opp_loss']:.4f} "
                f"val_q={val_q:.4f} val_rmse={math.sqrt(max(0.0, val_q)):.4f} "
                f"val_opp={val_opp:.4f})",
                flush=True,
            )
            break

        if epoch % args.log_every == 0:
            elapsed = time.perf_counter() - wall_start
            cv_s = f"{cv:.5f}" if cv is not None else "n/a"
            tr_rmse = math.sqrt(max(0.0, stats["q_loss"]))
            val_rmse = math.sqrt(max(0.0, val_q))
            print(
                f"  {elapsed:6.1f}s  epoch={epoch:5d}  "
                f"tr_q={stats['q_loss']:.4f}  tr_rmse={tr_rmse:.4f}  "
                f"tr_opp={stats['opp_loss']:.4f}  "
                f"val_q={val_q:.4f}  val_rmse={val_rmse:.4f}  "
                f"val_opp={val_opp:.4f}  "
                f"val_cv={cv_s}",
                flush=True,
            )

    wall_total = time.perf_counter() - wall_start
    ckpt_obj: dict = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "samples": int(n),
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "stop_reason": stop_reason,
        "perm_indices_cpu": torch.tensor(perm_np, dtype=torch.long),
        "gamma": GAMMA,
        "q_horizon_turns": Q_TARGET_HORIZON_TURNS,
        "q_target_spec": "future_only_gap_horizon",
        "opp_weight": args.opp_weight,
    }
    if rollout_from_disk:
        assert rollout_dir_resolved is not None
        ckpt_obj["data_source"] = "rollout_shards"
        ckpt_obj["rollout_dir"] = str(rollout_dir_resolved)
        ckpt_obj["n_shard_files"] = n_shard_files
        if args.max_samples > 0:
            ckpt_obj["max_samples_requested"] = args.max_samples
    else:
        ckpt_obj["data_source"] = "live_rollout"
        ckpt_obj["games"] = args.games

    torch.save(ckpt_obj, ckpt_path)

    summary = {
        "utc_end": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(wall_total, 1),
        "stop_reason": stop_reason,
        "epochs": epoch,
        "stable_window": args.stable_window,
        "stable_rel": args.stable_rel,
        "min_epochs": args.min_epochs,
        "max_epochs": args.max_epochs,
        "max_minutes": args.max_minutes,
        "opp_weight": args.opp_weight,
        "samples": int(n),
        "train_samples": int(split),
        "val_samples": int(n - split),
        "final_train_q_loss": train_losses[-1] if train_losses else None,
        "final_val_q_loss": val_losses[-1] if val_losses else None,
        "final_train_opp_loss": train_opp_losses[-1] if train_opp_losses else None,
        "final_val_opp_loss": val_opp_losses[-1] if val_opp_losses else None,
        "final_train_rmse": math.sqrt(max(0.0, train_losses[-1])) if train_losses else None,
        "final_val_rmse": math.sqrt(max(0.0, val_losses[-1])) if val_losses else None,
        "final_val_cv": _val_cv(val_losses, args.stable_window),
        "seed": args.seed,
        "checkpoint": str(ckpt_path.resolve()),
        "data_source": "rollout_shards" if rollout_from_disk else "live_rollout",
        "gamma": GAMMA,
        "q_horizon_turns": Q_TARGET_HORIZON_TURNS,
        "q_target_spec": "future_only_gap_horizon",
    }
    if rollout_from_disk and rollout_dir_resolved is not None:
        summary["rollout_dir"] = str(rollout_dir_resolved)
        summary["n_shard_files"] = n_shard_files
        if args.max_samples > 0:
            summary["max_samples"] = args.max_samples
    else:
        summary["games"] = args.games

    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Done. stop={stop_reason}  epochs={epoch}  wall={wall_total:.1f}s "
        f"({wall_total/60:.2f} min)",
        flush=True,
    )
    print(f"Checkpoint: {ckpt_path}", flush=True)
    print(f"Summary:    {log_path}", flush=True)


if __name__ == "__main__":
    main()
