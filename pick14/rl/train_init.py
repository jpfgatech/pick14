"""
PPO trial with rl.md §4 atomic rewards, dual critics, and split value losses.
Play advantage uses V_opp(S_post_play) for the chosen action (EV baseline extension in rl.md §4.3 note).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from pick14.rl.env import Pick14GymEnv
from pick14.rl.rlmd_model import RLmdPPOAgent
from pick14.rl.train_bc_static import MAX_KEYS, dual_bc_loss


def to_torch_batch(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}


def to_torch_obs1(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}


def policy_logp_entropy(
    model: RLmdPPOAgent, obs_t: dict[str, torch.Tensor], actions_t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mlog, plog, va, vo = model(obs_t)
    mfd = model.match_flat_dim
    phase_m = obs_t["phase"].squeeze(1) < 0.5
    logp_m = F.log_softmax(mlog, dim=1)
    logp_p = F.log_softmax(plog, dim=1)
    gm = logp_m.gather(1, actions_t.unsqueeze(1).clamp(max=mlog.shape[1] - 1)).squeeze(1)
    rel = (actions_t - mfd).clamp(min=0, max=plog.shape[1] - 1)
    gp = logp_p.gather(1, rel.unsqueeze(1)).squeeze(1)
    logp = torch.where(phase_m, gm, gp)
    p_m = torch.softmax(mlog, dim=1)
    h_m = -(p_m * logp_m).sum(dim=1)
    p_p = torch.softmax(plog, dim=1)
    h_p = -(p_p * logp_p).sum(dim=1)
    entropy = torch.where(phase_m, h_m, h_p)
    return logp, entropy, va, vo


def explained_variance(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    var_y = float(np.var(y))
    if var_y < 1e-12:
        return 0.0
    return float(1.0 - np.var(y - pred) / var_y)


@dataclass
class Transition:
    obs: dict[str, np.ndarray]
    action: int
    reward: float
    done: bool
    is_match: bool
    obs_post_play: dict[str, np.ndarray] | None


def rollout_ppo(
    env: Pick14GymEnv, model: RLmdPPOAgent, device: torch.device, episodes: int, seed_base: int, max_steps: int = 512
) -> tuple[list[list[Transition]], list[float], dict[str, float]]:
    """Collect transitions per episode; episode return = sum of atomic rewards (net-gap macro)."""
    episode_trajs: list[list[Transition]] = []
    ep_returns: list[float] = []
    sum_r_match = sum_r_play = 0.0
    n_m = n_p = 0
    model.eval()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        ep_ret = 0.0
        steps = 0
        cur: list[Transition] = []
        while not done and steps < max_steps:
            if env.state is None:
                break
            is_match = float(obs["phase"][0]) < 0.5
            obs_t = to_torch_obs1(obs, device)
            with torch.no_grad():
                mlog, plog, _, _ = model(obs_t)
                if is_match:
                    dist = Categorical(logits=mlog)
                else:
                    dist = Categorical(logits=plog)
                sub = int(dist.sample().item())
            action = sub if is_match else sub + model.match_flat_dim
            next_obs, rew, term, trunc, _ = env.step(action)
            post = None
            if not is_match and env._post_play_obs is not None:
                post = {k: np.asarray(v).copy() for k, v in env._post_play_obs.items()}
            cur.append(
                Transition(
                    obs={k: v.copy() for k, v in obs.items()},
                    action=action,
                    reward=float(rew),
                    done=bool(term or trunc),
                    is_match=is_match,
                    obs_post_play=post,
                )
            )
            ep_ret += float(rew)
            if is_match:
                sum_r_match += float(rew)
                n_m += 1
            else:
                sum_r_play += float(rew)
                n_p += 1
            obs = next_obs
            done = term or trunc
            steps += 1
        episode_trajs.append(cur)
        ep_returns.append(ep_ret)

    meta = {
        "avg_ep_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
        "avg_match_r": (sum_r_match / max(1, n_m)),
        "avg_play_r": (sum_r_play / max(1, n_p)),
    }
    return episode_trajs, ep_returns, meta


def _flatten_mc_returns(episode_trajs: list[list[Transition]], gamma: float) -> tuple[list[Transition], list[float]]:
    traj: list[Transition] = []
    returns: list[float] = []
    for ep in episode_trajs:
        g = 0.0
        rev: list[float] = []
        for tr in reversed(ep):
            g = tr.reward + gamma * g
            rev.append(g)
        rev.reverse()
        traj.extend(ep)
        returns.extend(rev)
    return traj, returns


def ppo_update(
    model: RLmdPPOAgent,
    episode_trajs: list[list[Transition]],
    device: torch.device,
    gamma: float = 0.99,
    clip_eps: float = 0.2,
    ppo_epochs: int = 4,
    lr: float = 3e-4,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
) -> dict[str, list[float]]:
    traj, returns = _flatten_mc_returns(episode_trajs, gamma)
    if not traj:
        return {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "ev_agent": [],
            "ev_opp": [],
        }

    obs_keys = traj[0].obs.keys()
    obs_np = {k: np.stack([t.obs[k] for t in traj], axis=0) for k in obs_keys}
    obs_t = to_torch_batch(obs_np, device)
    actions_t = torch.as_tensor(np.asarray([t.action for t in traj], dtype=np.int64), device=device)
    ret_t = torch.as_tensor(np.asarray(returns, dtype=np.float32), device=device)
    phase_m = obs_t["phase"].squeeze(1) < 0.5

    with torch.no_grad():
        logp_old, _, va_old, vo_old = policy_logp_entropy(model, obs_t, actions_t)
        # Play baseline: V_opp at post_play (§4.3 EV note: here = chosen action's post state)
        v_old = torch.empty_like(ret_t)
        v_old.copy_(va_old)
        v_old = torch.where(phase_m, va_old, vo_old)
        for i, tr in enumerate(traj):
            if not tr.is_match and tr.obs_post_play is not None:
                po = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in tr.obs_post_play.items()}
                _, _, _, vo_p = model(po)
                v_old[i] = vo_p.squeeze(0)

    adv_t = ret_t - v_old
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    logp_old_det = logp_old.detach()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    kls: list[float] = []
    ev_agent: list[float] = []
    ev_opp: list[float] = []

    y_a = ret_t[phase_m].detach().cpu().numpy()
    y_p = ret_t[~phase_m].detach().cpu().numpy()

    model.train()
    for _ in range(ppo_epochs):
        logp, entropy_vec, va, vo = policy_logp_entropy(model, obs_t, actions_t)
        v_pred = torch.where(phase_m, va, vo)
        for i, tr in enumerate(traj):
            if not tr.is_match and tr.obs_post_play is not None:
                po = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in tr.obs_post_play.items()}
                _, _, _, vo_p = model(po)
                v_pred[i] = vo_p.squeeze(0)

        ratio = torch.exp(logp - logp_old_det)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
        policy_loss = -torch.min(surr1, surr2).mean()

        vl = torch.tensor(0.0, device=device)
        if phase_m.any():
            vl = vl + F.mse_loss(va[phase_m], ret_t[phase_m])
        play_idx = ~phase_m
        if play_idx.any():
            vo_targets = ret_t[play_idx]
            vo_pred_list = []
            idxs = torch.nonzero(play_idx, as_tuple=False).squeeze(1)
            for j in idxs.tolist():
                tr = traj[j]
                if tr.obs_post_play is None:
                    vo_pred_list.append(vo[j])
                else:
                    po = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in tr.obs_post_play.items()}
                    _, _, _, vo_p = model(po)
                    vo_pred_list.append(vo_p.squeeze(0))
            vo_pred_t = torch.stack(vo_pred_list)
            vl = vl + F.mse_loss(vo_pred_t, vo_targets)

        value_loss = 0.5 * vl
        entropy = entropy_vec.mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        approx_kl = float((logp_old_det - logp).mean().item())

        with torch.no_grad():
            ev_a = explained_variance(y_a, va[phase_m].detach().cpu().numpy()) if phase_m.any() else 0.0
            if play_idx.any():
                ev_o = explained_variance(y_p, vo_pred_t.detach().cpu().numpy())
            else:
                ev_o = 0.0

        opt.zero_grad()
        loss.backward()
        opt.step()

        policy_losses.append(float(policy_loss.item()))
        value_losses.append(float(value_loss.item()))
        entropies.append(float(entropy.item()))
        kls.append(approx_kl)
        ev_agent.append(ev_a)
        ev_opp.append(ev_o)

    return {
        "policy_loss": policy_losses,
        "value_loss": value_losses,
        "entropy": entropies,
        "approx_kl": kls,
        "ev_agent": ev_agent,
        "ev_opp": ev_opp,
    }


def run_bc_quick(
    model: RLmdPPOAgent,
    env: Pick14GymEnv,
    device: torch.device,
    episodes: int,
    epochs: int,
    batch: int,
    lr: float,
    mfd: int,
) -> None:
    from pick14.rl.train_bc_static import ImitationDataset, collate, collect_teacher_samples, to_torch

    samples = collect_teacher_samples(env, episodes=episodes, seed_base=2000)
    ds = ImitationDataset(samples)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=collate)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for obs_np, actions_np in dl:
            obs = to_torch(obs_np, device)
            actions = torch.as_tensor(actions_np, device=device)
            ml, pl, _, _ = model(obs)
            loss, _st = dual_bc_loss(ml, pl, actions, obs, match_flat_dim=mfd, max_keys=MAX_KEYS, play_loss_weight=2.0)
            opt.zero_grad()
            loss.backward()
            opt.step()


def main():
    ap = argparse.ArgumentParser(description="PPO trial (rl.md §4) after curriculum checkpoint")
    ap.add_argument("--checkpoint", type=str, default="", help="checkpoint_curriculum.pt from pretrain_curriculum.py")
    ap.add_argument("--episodes-teacher", type=int, default=0, help="If no checkpoint, run this many BC episodes first")
    ap.add_argument("--bc-epochs", type=int, default=12)
    ap.add_argument("--rollout-episodes", type=int, default=32)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--out-dir", type=str, default="artifacts/rl_ppo_trial")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Pick14GymEnv(num_players=2, n_hand=3, seed=args.seed)
    model = RLmdPPOAgent(dropout=0.0).to(device)
    mfd = env.match_flat_dim

    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    if ckpt_path and ckpt_path.is_file():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"Loaded {ckpt_path}")
    elif args.episodes_teacher > 0:
        print(f"[warmup] BC episodes={args.episodes_teacher}")
        run_bc_quick(
            model,
            env,
            device,
            episodes=args.episodes_teacher,
            epochs=args.bc_epochs,
            batch=64,
            lr=1e-3,
            mfd=mfd,
        )
    else:
        print("No --checkpoint and episodes_teacher=0: training from random init (high variance).")

    print("[PPO] Rollout + update")
    ep_trajs, ep_ret, telem = rollout_ppo(env, model, device, episodes=args.rollout_episodes, seed_base=9000)
    stats = ppo_update(model, ep_trajs, device, gamma=0.99, ppo_epochs=args.ppo_epochs)
    print(
        f"  mean_ep_return={telem['avg_ep_return']:.4f} "
        f"avg_match_r={telem['avg_match_r']:.4f} avg_play_r={telem['avg_play_r']:.4f} "
        f"policy_loss={stats['policy_loss'][-1]:.4f} value_loss={stats['value_loss'][-1]:.4f} "
        f"H={stats['entropy'][-1]:.4f} kl={stats['approx_kl'][-1]:.5f} "
        f"EV_agent={stats['ev_agent'][-1]:.4f} EV_opp={stats['ev_opp'][-1]:.4f}"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ppo_stats": stats,
            "episode_returns": ep_ret,
            "telem": telem,
            "config": vars(args),
        },
        out_dir / "checkpoint_ppo_trial.pt",
    )
    (out_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"mean_ep_return={telem['avg_ep_return']:.6f}",
                f"avg_match_r={telem['avg_match_r']:.6f}",
                f"avg_play_r={telem['avg_play_r']:.6f}",
                f"policy_loss_last={stats['policy_loss'][-1]:.6f}",
                f"value_loss_last={stats['value_loss'][-1]:.6f}",
                f"entropy_last={stats['entropy'][-1]:.6f}",
                f"kl_last={stats['approx_kl'][-1]:.6f}",
                f"ev_agent_last={stats['ev_agent'][-1]:.6f}",
                f"ev_opp_last={stats['ev_opp'][-1]:.6f}",
            ]
        )
        + "\n"
    )
    print(f"Artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
