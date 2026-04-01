"""
curriculum.md Phase 1 (mixed BC) + Phase 2 (freeze actors, MSE both critics vs MC returns).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset

from pick14.rl.env import Pick14GymEnv
from pick14.rl.rlmd_model import RLmdPPOAgent
from pick14.rl.train_bc_static import BC_PHASE_SEMANTICS, Sample, collect_teacher_samples, train_static_bc


def _sanitize_json(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, list):
        return [_sanitize_json(v) for v in x]
    if isinstance(x, dict):
        return {k: _sanitize_json(v) for k, v in x.items()}
    return x


def _to_torch_obs(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}


def collect_critic_bootstrap_data(
    env: Pick14GymEnv,
    model: RLmdPPOAgent,
    device: torch.device,
    episodes: int,
    gamma: float,
    seed_base: int,
    max_steps: int = 512,
) -> tuple[list[tuple[dict[str, np.ndarray], float]], list[tuple[dict[str, np.ndarray], float]]]:
    """Roll out frozen (or any) policy; targets = MC return of atomic rewards (rl.md §4.2 chain)."""
    match_rows: list[tuple[dict[str, np.ndarray], float]] = []
    play_rows: list[tuple[dict[str, np.ndarray], float]] = []
    model.eval()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        trans: list[tuple[bool, float, dict[str, np.ndarray], dict[str, np.ndarray] | None]] = []
        for _ in range(max_steps):
            if env.state is None or env.state.current_player != 0:
                break
            is_match = float(obs["phase"][0]) < 0.5
            with torch.no_grad():
                ml, pl, _, _ = model(_to_torch_obs(obs, device))
                if is_match:
                    dist = Categorical(logits=ml)
                else:
                    dist = Categorical(logits=pl)
                sub = int(dist.sample().item())
            action = sub if is_match else sub + model.match_flat_dim
            next_obs, rew, term, trunc, _ = env.step(action)
            obs_post = None
            if not is_match and env._post_play_obs is not None:
                obs_post = {k: np.asarray(v).copy() for k, v in env._post_play_obs.items()}
            trans.append((is_match, float(rew), {k: v.copy() for k, v in obs.items()}, obs_post))
            obs = next_obs
            if term or trunc:
                break

        g = 0.0
        returns_rev: list[float] = []
        for _, r, _, _ in reversed(trans):
            g = r + gamma * g
            returns_rev.append(g)
        returns_rev.reverse()
        for (_is_m, _r, o_pre, o_post), gret in zip(trans, returns_rev, strict=True):
            if _is_m:
                match_rows.append((o_pre, float(gret)))
            else:
                if o_post is None:
                    continue
                play_rows.append((o_post, float(gret)))
    return match_rows, play_rows


@dataclass(slots=True)
class CriticRow:
    obs: dict[str, np.ndarray]
    target: float


class CriticDataset(Dataset):
    def __init__(self, rows: list[tuple[dict[str, np.ndarray], float]]):
        self.rows = [CriticRow(obs=o, target=t) for o, t in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        return r.obs, r.target


def collate_critic(batch):
    obs_keys = batch[0][0].keys()
    obs = {k: np.stack([x[0][k] for x in batch], axis=0) for k in obs_keys}
    tgt = np.asarray([x[1] for x in batch], dtype=np.float32)
    return obs, tgt


def train_critics_mse(
    model: RLmdPPOAgent,
    match_rows: list[tuple[dict[str, np.ndarray], float]],
    play_rows: list[tuple[dict[str, np.ndarray], float]],
    device: torch.device,
    epochs: int,
    batch: int,
    lr: float,
) -> dict[str, list[float]]:
    model.set_requires_grad_actor_trunk(False)
    model.set_requires_grad_critics(True)
    ds_m = CriticDataset(match_rows)
    ds_p = CriticDataset(play_rows)
    # Joint batches: alternate mini-updates from each stream
    dl_m = DataLoader(ds_m, batch_size=batch, shuffle=True, collate_fn=collate_critic) if ds_m else None
    dl_p = DataLoader(ds_p, batch_size=batch, shuffle=True, collate_fn=collate_critic) if ds_p else None
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    hist = {"loss_m": [], "loss_p": [], "loss": []}
    for _ in range(epochs):
        loss_m_acc = 0.0
        loss_p_acc = 0.0
        n_m = n_p = 0
        if dl_m is not None:
            for obs_np, tgt_np in dl_m:
                obs_t = {k: torch.as_tensor(v, device=device) for k, v in obs_np.items()}
                y = torch.as_tensor(tgt_np, device=device)
                _, _, va, _ = model(obs_t)
                loss = F.mse_loss(va, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_m_acc += float(loss.item()) * y.shape[0]
                n_m += int(y.shape[0])
        if dl_p is not None:
            for obs_np, tgt_np in dl_p:
                obs_t = {k: torch.as_tensor(v, device=device) for k, v in obs_np.items()}
                y = torch.as_tensor(tgt_np, device=device)
                _, _, _, vo = model(obs_t)
                loss = F.mse_loss(vo, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_p_acc += float(loss.item()) * y.shape[0]
                n_p += int(y.shape[0])
        hist["loss_m"].append(loss_m_acc / max(1, n_m))
        hist["loss_p"].append(loss_p_acc / max(1, n_p))
        hist["loss"].append((loss_m_acc + loss_p_acc) / max(1, n_m + n_p))
    model.set_requires_grad_actor_trunk(True)
    model.set_requires_grad_critics(True)
    return hist


def main():
    ap = argparse.ArgumentParser(description="curriculum.md Phase 1 BC + Phase 2 critic bootstrap")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", type=str, default="artifacts/rl_curriculum")
    ap.add_argument("--bc-episodes", type=int, default=360)
    ap.add_argument("--bc-epochs", type=int, default=16)
    ap.add_argument("--bc-batch", type=int, default=128)
    ap.add_argument("--bc-lr", type=float, default=1e-3)
    ap.add_argument("--bc-test-ratio", type=float, default=0.2, help="Held-out fraction per phase (match vs play stratified).")
    ap.add_argument("--play-loss-weight", type=float, default=2.0)
    ap.add_argument(
        "--min-train-play-acc",
        type=float,
        default=0.98,
        help="Abort if final train play-head accuracy is below this (requires enough train play labels).",
    )
    ap.add_argument(
        "--min-test-play-acc",
        type=float,
        default=0.98,
        help="Abort if final test play-head accuracy is below this (requires enough test play labels).",
    )
    ap.add_argument(
        "--min-train-play-labels",
        type=int,
        default=32,
        help="Minimum play-phase rows in the train split to enforce --min-train-play-acc.",
    )
    ap.add_argument(
        "--min-test-play-labels",
        type=int,
        default=32,
        help="Minimum play-phase rows in the test split to enforce --min-test-play-acc.",
    )
    ap.add_argument("--critic-rollout-episodes", type=int, default=400)
    ap.add_argument("--critic-epochs", type=int, default=8)
    ap.add_argument("--critic-batch", type=int, default=64)
    ap.add_argument("--critic-lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Pick14GymEnv(num_players=2, n_hand=3, seed=args.seed)
    mfd = env.match_flat_dim

    print("[Phase 1] Mixed BC (greedy match + stingy play teacher; stratified split + mixed batches per curriculum.md §1.2)")
    samples: list[Sample] = collect_teacher_samples(env, episodes=args.bc_episodes, seed_base=1000)
    total_nm, total_np = (
        sum(1 for s in samples if float(s.obs["phase"][0]) < 0.5),
        sum(1 for s in samples if float(s.obs["phase"][0]) >= 0.5),
    )
    print(
        f"  dataset: {len(samples)} timesteps from {args.bc_episodes} teacher episodes "
        f"(~{total_nm} match-phase, ~{total_np} play-phase labels)",
        flush=True,
    )
    model, hist_bc, n_tr, n_te, tr_nm, tr_np, te_nm, te_np = train_static_bc(
        samples=samples,
        epochs=args.bc_epochs,
        batch=args.bc_batch,
        lr=args.bc_lr,
        device=device,
        seed=args.seed,
        match_flat_dim=mfd,
        play_loss_weight=args.play_loss_weight,
        hidden_dim=32,
        test_ratio=args.bc_test_ratio,
    )
    bc_sizes = {
        "teacher_episodes": args.bc_episodes,
        "total_timesteps": len(samples),
        "total_match_labels": total_nm,
        "total_play_labels": total_np,
        "train_timesteps": n_tr,
        "test_timesteps": n_te,
        "train_match_labels": tr_nm,
        "train_play_labels": tr_np,
        "test_match_labels": te_nm,
        "test_play_labels": te_np,
        "bc_epochs": args.bc_epochs,
        "bc_batch": args.bc_batch,
    }
    print(
        f"  split: train={n_tr} (match={tr_nm}, play={tr_np}) | test={n_te} (match={te_nm}, play={te_np}) | "
        f"last acc train_play={hist_bc['train_acc_play'][-1]} test_play={hist_bc['test_acc_play'][-1]} "
        f"test_overall={hist_bc['test_acc_overall'][-1]:.4f}",
        flush=True,
    )

    tr_p = hist_bc["train_acc_play"][-1]
    te_p = hist_bc["test_acc_play"][-1]
    if tr_np < args.min_train_play_labels:
        raise SystemExit(
            f"Too few train play-phase labels ({tr_np} < {args.min_train_play_labels}); "
            "increase --bc-episodes so play-head accuracy is meaningful."
        )
    if te_np < args.min_test_play_labels:
        raise SystemExit(
            f"Too few test play-phase labels ({te_np} < {args.min_test_play_labels}); "
            "increase --bc-episodes or lower --bc-test-ratio slightly."
        )
    if isinstance(tr_p, float) and not math.isnan(tr_p) and tr_p < args.min_train_play_acc:
        raise SystemExit(
            f"Train play accuracy {tr_p:.4f} < {args.min_train_play_acc}; "
            "increase --bc-epochs / --bc-episodes or tune --play-loss-weight."
        )
    if isinstance(te_p, float) and not math.isnan(te_p) and te_p < args.min_test_play_acc:
        raise SystemExit(
            f"Test play accuracy {te_p:.4f} < {args.min_test_play_acc}; "
            "increase --bc-epochs / --bc-episodes or tune --play-loss-weight."
        )

    print("[Phase 2] Freeze actors, critic MSE on policy rollouts")
    crit_hist = train_critics_mse(
        model,
        *collect_critic_bootstrap_data(
            env,
            model,
            device,
            episodes=args.critic_rollout_episodes,
            gamma=args.gamma,
            seed_base=5000,
        ),
        device=device,
        epochs=args.critic_epochs,
        batch=args.critic_batch,
        lr=args.critic_lr,
    )
    print(f"  critic loss last (joint)={crit_hist['loss'][-1]:.6f}")

    ckpt = {
        "model": model.state_dict(),
        "hist_bc": hist_bc,
        "hist_critic": crit_hist,
        "bc_data_sizes": bc_sizes,
        "bc_phase_semantics": BC_PHASE_SEMANTICS,
        "config": vars(args),
        "match_flat_dim": mfd,
    }
    torch.save(ckpt, out_dir / "checkpoint_curriculum.pt")
    summary_payload = _sanitize_json(
        {
            "bc": hist_bc,
            "critic": crit_hist,
            "bc_data_sizes": bc_sizes,
            "bc_phase_semantics": BC_PHASE_SEMANTICS,
            "config": vars(args),
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, allow_nan=False))
    print(f"Saved {out_dir / 'checkpoint_curriculum.pt'}")


if __name__ == "__main__":
    main()
