#!/usr/bin/env python3
"""
Experiment: fresh critic heads, actor + shared trunk frozen from an existing checkpoint.

Match-critic MSE skips batch rows with no legal ``mask_match`` (all zeros): no backward through
those rows (equivalent to zero contribution; we do not target V=0 on them).

Play-critic training unchanged. Then writes the same ~400-point scatter as ``plot_critic_scatter.py``.

Run from repo root: ``python scripts/experiment_critic_scratch.py``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pick14.rl.agents import table_all_baseline
from pick14.rl.env import Pick14GymEnv
from pick14.rl.pretrain_curriculum import (
    CriticDataset,
    PlayCriticEVRow,
    _play_ev_loss,
    collate_critic,
    critic_match_obs_has_legal_action,
    load_critic_bundle,
)
from pick14.rl.rlmd_model import RLmdPPOAgent
from pick14.rl.rlmd_transformer import RLmdEncoderLayer


def reset_critic_stacks(model: RLmdPPOAgent, dropout: float = 0.0) -> None:
    d = 32
    dev = next(model.parameters()).device
    model.layer2_critic_agent = RLmdEncoderLayer(d, nhead=4, dim_ff=128, dropout=dropout).to(dev)
    model.layer2_critic_opp = RLmdEncoderLayer(d, nhead=4, dim_ff=128, dropout=dropout).to(dev)
    model.critic_agent_mlp = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 1)).to(dev)
    model.critic_opp_mlp = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 1)).to(dev)


def freeze_shared_and_actor(model: RLmdPPOAgent) -> None:
    critic_frag = (
        "layer2_critic_agent",
        "layer2_critic_opp",
        "critic_agent_mlp",
        "critic_opp_mlp",
    )
    for name, p in model.named_parameters():
        p.requires_grad = any(x in name for x in critic_frag)


def count_match_rows_no_legal(
    match_rows: list[tuple[dict[str, np.ndarray], float]],
) -> tuple[int, int]:
    n0 = sum(1 for o, _t in match_rows if not critic_match_obs_has_legal_action(o))
    return n0, len(match_rows)


def train_critics_masked_match_frozen_trunk(
    model: RLmdPPOAgent,
    match_rows: list[tuple[dict[str, np.ndarray], float]],
    play_rows: list[PlayCriticEVRow],
    device: torch.device,
    epochs: int,
    batch: int,
    lr: float,
) -> dict[str, list[float]]:
    model.train()
    freeze_shared_and_actor(model)
    ds_m = CriticDataset(match_rows)
    dl_m = DataLoader(ds_m, batch_size=batch, shuffle=True, collate_fn=collate_critic) if ds_m else None
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    hist: dict[str, list[float]] = {"loss_m": [], "loss_p": [], "loss": []}
    n_play = len(play_rows)

    for _ep in range(epochs):
        loss_m_acc = 0.0
        loss_p_acc = 0.0
        n_m_used = 0
        if dl_m is not None:
            for obs_np, tgt_np in dl_m:
                obs_t = {k: torch.as_tensor(v, device=device) for k, v in obs_np.items()}
                y = torch.as_tensor(tgt_np, device=device, dtype=torch.float32)
                B = int(y.shape[0])
                mm = obs_t["mask_match"].reshape(B, -1).bool()
                valid = mm.any(dim=1)
                if not valid.any():
                    continue
                _, _, va, _ = model(obs_t)
                va_v = va[valid]
                y_v = y[valid]
                loss = F.mse_loss(va_v, y_v)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_m_acc += float(loss.detach().item()) * int(valid.sum().item())
                n_m_used += int(valid.sum().item())

        if n_play > 0:
            perm = np.random.permutation(n_play)
            for s in range(0, n_play, batch):
                idxs = perm[s : s + batch]
                opt.zero_grad(set_to_none=True)
                chunk = torch.zeros((), device=device)
                for j in idxs:
                    chunk = chunk + _play_ev_loss(model, device, play_rows[int(j)], use_opp_head=True)
                chunk = chunk / max(1, len(idxs))
                chunk.backward()
                opt.step()
                loss_p_acc += float(chunk.detach().item()) * max(1, len(idxs))

        hist["loss_m"].append(loss_m_acc / max(1, n_m_used))
        hist["loss_p"].append(loss_p_acc / max(1, n_play if n_play else 1))
        hist["loss"].append((loss_m_acc + loss_p_acc) / max(1, n_m_used + (n_play if n_play else 0)))

    return hist


@torch.no_grad()
def play_ev_prediction(model: RLmdPPOAgent, device: torch.device, row: PlayCriticEVRow) -> float:
    parts: list[torch.Tensor] = []
    pi = torch.as_tensor(row.pi, device=device, dtype=torch.float32)
    for hi, op in enumerate(row.post_obs_by_hand):
        if op is None or hi >= pi.numel():
            continue
        w = pi[hi]
        if float(w.item()) == 0.0:
            continue
        oti = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in op.items()}
        _, _, _, vo = model(oti)
        parts.append(w * vo.squeeze(0))
    if not parts:
        return float("nan")
    return float(torch.stack(parts).sum().item())


def write_scatter(
    model: RLmdPPOAgent,
    device: torch.device,
    mr: list,
    pr: list,
    meta: dict,
    bundle_name: str,
    out_path: Path,
    n_each: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    model.eval()
    n_m = min(n_each, len(mr))
    n_p = min(n_each, len(pr))
    mi = rng.choice(len(mr), size=n_m, replace=len(mr) < n_m) if mr else np.array([], dtype=int)
    pi_ix = rng.choice(len(pr), size=n_p, replace=len(pr) < n_p) if pr else np.array([], dtype=int)

    y_match: list[float] = []
    pred_match: list[float] = []
    for j in mi:
        obs, tgt = mr[int(j)]
        batch = collate_critic([(obs, tgt)])
        obs_np, y_np = batch
        obs_t = {k: torch.as_tensor(v, device=device) for k, v in obs_np.items()}
        _, _, va, _ = model(obs_t)
        pred_match.append(float(va.squeeze().item()))
        y_match.append(float(y_np.reshape(-1)[0]))

    y_play: list[float] = []
    pred_play: list[float] = []
    for j in pi_ix:
        row = pr[int(j)]
        pred_play.append(play_ev_prediction(model, device, row))
        y_play.append(float(row.target_neg_opp_gain))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, y, p, title in (
        (
            axes[0],
            np.asarray(y_match, dtype=np.float64),
            np.asarray(pred_match, dtype=np.float64),
            f"Match: V_agent vs A_t (n={len(y_match)})",
        ),
        (
            axes[1],
            np.asarray(y_play, dtype=np.float64),
            np.asarray(pred_play, dtype=np.float64),
            f"Play: Σ π V_opp vs −O_t (n={len(y_play)})",
        ),
    ):
        m = np.isfinite(y) & np.isfinite(p)
        ax.scatter(y[m], p[m], s=8, alpha=0.45, c="#8b4513", edgecolors="none")
        if m.any():
            lo = float(np.min([y[m].min(), p[m].min()]))
            hi = float(np.max([y[m].max(), p[m].max()]))
            pad = 0.05 * (hi - lo + 1e-6)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.5, label="y=x")
        ax.set_xlabel("target (bootstrap)")
        ax.set_ylabel("prediction (scratch critic)")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"Scratch critic + frozen trunk/actor | masked match rows\n{bundle_name} | meta={meta}",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Scatter → {out_path.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="artifacts/rl_joint_run_agent/checkpoint_curriculum.pt")
    ap.add_argument("--critic-bundle", type=str, default="artifacts/rl_joint_run_agent/critic_bundle.pt")
    ap.add_argument("--out-ckpt", type=str, default="artifacts/critic_scratch_locked.pt")
    ap.add_argument("--out-plot", type=str, default="artifacts/critic_scatter_400_scratch_masked.png")
    ap.add_argument("--epochs", type=int, default=36)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-each", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    bundle_path = Path(args.critic_bundle)
    if not ckpt_path.is_file():
        raise SystemExit(f"missing checkpoint: {ckpt_path.resolve()}")
    if not bundle_path.is_file():
        raise SystemExit(f"missing bundle: {bundle_path.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base_env = Pick14GymEnv(table_all_baseline(2), n_hand=3, seed=7)
    model = RLmdPPOAgent.from_env(base_env, dropout=0.0).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    mr, pr, meta = load_critic_bundle(bundle_path)
    n_bad, n_tot = count_match_rows_no_legal(mr)
    print(
        f"Match rows with no legal mask_match: {n_bad}/{n_tot} "
        f"({100.0 * n_bad / max(1, n_tot):.2f}%) — excluded from match-critic loss",
        flush=True,
    )

    reset_critic_stacks(model, dropout=0.0)
    freeze_shared_and_actor(model)
    n_critic_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable (critic-only): {n_critic_params} params | frozen: {n_frozen}", flush=True)

    hist = train_critics_masked_match_frozen_trunk(
        model,
        mr,
        pr,
        device,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
    )
    print(
        f"Last epoch loss_m={hist['loss_m'][-1]:.6f} loss_p={hist['loss_p'][-1]:.6f} "
        f"combined={hist['loss'][-1]:.6f}",
        flush=True,
    )

    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "hist_critic_scratch": hist,
            "source_checkpoint": str(ckpt_path),
            "critic_bundle": str(bundle_path),
            "match_mask_skip_note": "match MSE rows with all-zero mask_match skipped (no backward)",
        },
        out_ckpt,
    )
    print(f"Saved {out_ckpt.resolve()}", flush=True)

    write_scatter(
        model,
        device,
        mr,
        pr,
        meta,
        bundle_path.name,
        Path(args.out_plot),
        n_each=args.n_each,
        seed=args.seed + 1,
    )


if __name__ == "__main__":
    main()
