from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset

from pick14.rl.env import Pick14GymEnv
from pick14.rl.model import PointerPolicyNet


def to_torch_batch(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}


@dataclass(slots=True)
class Sample:
    obs: dict[str, np.ndarray]
    action: int
    reward: float


class TeacherDataset(Dataset):
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return s.obs, s.action, s.reward


def collate(batch):
    obs_keys = batch[0][0].keys()
    obs = {k: np.stack([x[0][k] for x in batch], axis=0) for k in obs_keys}
    actions = np.asarray([x[1] for x in batch], dtype=np.int64)
    rewards = np.asarray([x[2] for x in batch], dtype=np.float32)
    return obs, actions, rewards


def collect_teacher_samples(env: Pick14GymEnv, episodes: int, max_steps: int = 256) -> list[Sample]:
    out: list[Sample] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        for _ in range(max_steps):
            action = env.teacher_action()
            out.append(Sample(obs={k: v.copy() for k, v in obs.items()}, action=int(action), reward=0.0))
            obs, rew, done, trunc, _ = env.step(action)
            out[-1].reward = float(rew)
            if done or trunc:
                break
    return out


def run_bc(model: PointerPolicyNet, dataset: TeacherDataset, device: torch.device, epochs: int, batch_size: int, lr: float):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_hist: list[float] = []
    acc_hist: list[float] = []
    model.train()
    for _ in range(epochs):
        ep_loss = 0.0
        ok = 0
        n = 0
        for obs_np, actions_np, _ in dl:
            obs = to_torch_batch(obs_np, device)
            actions = torch.as_tensor(actions_np, device=device)
            logits, _ = model(obs)
            loss = F.cross_entropy(logits, actions)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * actions.shape[0]
            pred = logits.argmax(dim=1)
            ok += int((pred == actions).sum().item())
            n += int(actions.shape[0])
        loss_hist.append(ep_loss / max(1, n))
        acc_hist.append(ok / max(1, n))
    return loss_hist, acc_hist


def rollout_policy(env: Pick14GymEnv, model: PointerPolicyNet, device: torch.device, episodes: int):
    traj = []
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=10000 + ep)
        done = False
        ep_ret = 0.0
        steps = 0
        while not done and steps < 256:
            obs_t = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                logits, value = model(obs_t)
                dist = Categorical(logits=logits)
                action = int(dist.sample().item())
                logp = float(dist.log_prob(torch.tensor([action], device=device)).item())
                val = float(value.item())
            next_obs, rew, term, trunc, _ = env.step(action)
            traj.append((obs, action, rew, logp, val, float(term or trunc)))
            ep_ret += float(rew)
            obs = next_obs
            done = term or trunc
            steps += 1
        returns.append(ep_ret)
    return traj, returns


def ppo_update(
    model: PointerPolicyNet,
    traj,
    device: torch.device,
    gamma: float = 0.99,
    clip_eps: float = 0.2,
    ppo_epochs: int = 3,
    lr: float = 3e-4,
):
    obs_list, act_list, rew_list, old_logp_list, old_val_list, done_list = zip(*traj)
    # Compute returns (simple Monte Carlo).
    returns = []
    g = 0.0
    for r, d in zip(reversed(rew_list), reversed(done_list)):
        g = r + gamma * g * (1.0 - d)
        returns.append(g)
    returns = list(reversed(returns))

    # Build tensors.
    obs_keys = obs_list[0].keys()
    obs_np = {k: np.stack([o[k] for o in obs_list], axis=0) for k in obs_keys}
    obs_t = to_torch_batch(obs_np, device)
    actions_t = torch.as_tensor(np.asarray(act_list, dtype=np.int64), device=device)
    old_logp_t = torch.as_tensor(np.asarray(old_logp_list, dtype=np.float32), device=device)
    old_val_t = torch.as_tensor(np.asarray(old_val_list, dtype=np.float32), device=device)
    ret_t = torch.as_tensor(np.asarray(returns, dtype=np.float32), device=device)
    adv_t = (ret_t - old_val_t)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-6)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    policy_losses = []
    value_losses = []
    entropies = []

    model.train()
    for _ in range(ppo_epochs):
        logits, values = model(obs_t)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(actions_t)
        ratio = torch.exp(logp - old_logp_t)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, ret_t)
        entropy = dist.entropy().mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        opt.zero_grad()
        loss.backward()
        opt.step()

        policy_losses.append(float(policy_loss.item()))
        value_losses.append(float(value_loss.item()))
        entropies.append(float(entropy.item()))
    return policy_losses, value_losses, entropies


def save_plots(out_dir: Path, bc_loss, bc_acc, ppo_policy, ppo_value, returns):
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(bc_loss, label="BC loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy")
    plt.title("Behavior Cloning Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bc_loss.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(bc_acc, label="BC acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Behavior Cloning Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bc_accuracy.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(ppo_policy, label="PPO policy loss")
    plt.plot(ppo_value, label="PPO value loss")
    plt.xlabel("PPO epoch")
    plt.ylabel("Loss")
    plt.title("Initial PPO Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "ppo_losses.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(returns, label="Episode return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Policy Rollout Returns (initial)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "returns.png", dpi=140)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Pick14 RL bootstrap (BC + initial PPO)")
    ap.add_argument("--episodes-teacher", type=int, default=200)
    ap.add_argument("--bc-epochs", type=int, default=16)
    ap.add_argument("--bc-batch", type=int, default=64)
    ap.add_argument("--rollout-episodes", type=int, default=48)
    ap.add_argument("--out-dir", type=str, default="artifacts/rl_init")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Pick14GymEnv(num_players=3, n_hand=3, seed=123)
    model = PointerPolicyNet(max_hand_combos=7, max_keys=17).to(device)

    print("[1/4] Collecting teacher data...")
    samples = collect_teacher_samples(env, episodes=args.episodes_teacher)
    ds = TeacherDataset(samples)
    print(f"  samples={len(ds)}")

    print("[2/4] Behavior cloning...")
    bc_loss, bc_acc = run_bc(model, ds, device, epochs=args.bc_epochs, batch_size=args.bc_batch, lr=1e-3)
    print(f"  final bc_loss={bc_loss[-1]:.4f} acc={bc_acc[-1]:.4f}")

    print("[3/4] PPO bootstrap rollout/update...")
    traj, returns = rollout_policy(env, model, device, episodes=args.rollout_episodes)
    ppo_policy, ppo_value, ppo_entropy = ppo_update(model, traj, device)
    print(
        f"  ppo_policy_last={ppo_policy[-1]:.4f} ppo_value_last={ppo_value[-1]:.4f} "
        f"entropy_last={ppo_entropy[-1]:.4f}"
    )

    print("[4/4] Saving artifacts...")
    out_dir = Path(args.out_dir)
    save_plots(out_dir, bc_loss, bc_acc, ppo_policy, ppo_value, returns)
    ckpt = {
        "model": model.state_dict(),
        "bc_loss": bc_loss,
        "bc_acc": bc_acc,
        "ppo_policy_loss": ppo_policy,
        "ppo_value_loss": ppo_value,
        "returns": returns,
        "config": vars(args),
    }
    torch.save(ckpt, out_dir / "checkpoint.pt")
    (out_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"teacher_samples={len(ds)}",
                f"bc_loss_first={bc_loss[0]:.6f}",
                f"bc_loss_last={bc_loss[-1]:.6f}",
                f"bc_acc_last={bc_acc[-1]:.6f}",
                f"ppo_policy_last={ppo_policy[-1]:.6f}",
                f"ppo_value_last={ppo_value[-1]:.6f}",
                f"returns_mean={float(np.mean(returns)):.6f}",
            ]
        )
        + "\n"
    )
    print(f"  artifacts -> {out_dir}")


if __name__ == "__main__":
    main()

