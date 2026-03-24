from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pick14.rl.env import Pick14GymEnv
from pick14.rl.model import PointerPolicyNet


@dataclass(slots=True)
class Sample:
    obs: dict[str, np.ndarray]
    action: int


class ImitationDataset(Dataset):
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return s.obs, s.action


def collate(batch):
    obs_keys = batch[0][0].keys()
    obs = {k: np.stack([x[0][k] for x in batch], axis=0) for k in obs_keys}
    actions = np.asarray([x[1] for x in batch], dtype=np.int64)
    return obs, actions


def to_torch(obs_np: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in obs_np.items()}


def pass_col_from_obs(obs_t: dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-sample pass column index == number of valid public cards."""
    # public_valid: [B, MAX_PUBLIC]
    return obs_t["public_valid"].sum(dim=1).long()


def grouped_bc_loss_and_acc(logits: torch.Tensor, actions: torch.Tensor, obs_t: dict[str, torch.Tensor], max_keys: int):
    """
    Supervision update:
    - match teacher label -> standard CE on exact flattened action index
    - pass teacher label  -> grouped NLL over all pass actions (one per hand combo)
    Accuracy:
    - exact for match labels
    - pass-group equivalence for pass labels
    """
    bsz = logits.shape[0]
    device = logits.device
    q_idx = actions // max_keys
    k_idx = actions % max_keys
    pass_col = pass_col_from_obs(obs_t)  # [B]
    is_pass_teacher = k_idx.eq(pass_col)

    logp = F.log_softmax(logits, dim=1)
    losses = []
    pred = logits.argmax(dim=1)

    # Match labels: exact CE
    if (~is_pass_teacher).any():
        idx = torch.nonzero(~is_pass_teacher, as_tuple=False).squeeze(1)
        losses.append(F.nll_loss(logp[idx], actions[idx], reduction="mean"))

    # Pass labels: grouped loss over all (hand_combo_i, pass_col)
    if is_pass_teacher.any():
        idx = torch.nonzero(is_pass_teacher, as_tuple=False).squeeze(1)
        pass_col_i = pass_col[idx]  # [Np]
        # Build flattened indices for 7 pass actions in each sample.
        combo_ids = torch.arange(7, device=device).view(1, 7).expand(idx.shape[0], 7)
        pass_flat = combo_ids * max_keys + pass_col_i.view(-1, 1)  # [Np,7]
        selected = logp[idx].gather(1, pass_flat)  # [Np,7]
        # log sum exp over pass-equivalent actions = prob(pass-group)
        group_logp = torch.logsumexp(selected, dim=1)
        losses.append((-group_logp).mean())

    loss = sum(losses) if losses else torch.tensor(0.0, device=device)

    # Grouped accuracy metric
    pred_k = pred % max_keys
    pred_is_pass = pred_k.eq(pass_col)
    correct = torch.zeros((bsz,), dtype=torch.bool, device=device)
    # exact for match labels
    correct = torch.where(~is_pass_teacher, pred.eq(actions), correct)
    # pass-group for pass labels
    correct = torch.where(is_pass_teacher, pred_is_pass, correct)
    acc = correct.float().mean()
    return loss, acc


def collect_teacher_samples(env: Pick14GymEnv, episodes: int, max_steps: int = 256, seed_base: int = 0) -> list[Sample]:
    out: list[Sample] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        for _ in range(max_steps):
            action = env.teacher_action()
            out.append(Sample(obs={k: v.copy() for k, v in obs.items()}, action=int(action)))
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
    return out


def train_epoch(model, dl, opt, device: torch.device):
    model.train()
    total_loss = 0.0
    total_ok = 0
    total = 0
    for obs_np, actions_np in dl:
        obs = to_torch(obs_np, device)
        actions = torch.as_tensor(actions_np, device=device)
        logits, _ = model(obs)
        loss, acc = grouped_bc_loss_and_acc(logits, actions, obs, max_keys=17)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * actions.shape[0]
        total_ok += int(float(acc.item()) * actions.shape[0])
        total += int(actions.shape[0])
    return total_loss / max(1, total), total_ok / max(1, total)


@torch.no_grad()
def eval_epoch(model, dl, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_ok = 0
    total = 0
    for obs_np, actions_np in dl:
        obs = to_torch(obs_np, device)
        actions = torch.as_tensor(actions_np, device=device)
        logits, _ = model(obs)
        loss, acc = grouped_bc_loss_and_acc(logits, actions, obs, max_keys=17)
        total_loss += float(loss.item()) * actions.shape[0]
        total_ok += int(float(acc.item()) * actions.shape[0])
        total += int(actions.shape[0])
    return total_loss / max(1, total), total_ok / max(1, total)


def split_samples(samples: list[Sample], test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_test = max(1, int(len(samples) * test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    train = [samples[i] for i in train_idx]
    test = [samples[i] for i in test_idx]
    return train, test


def train_static_bc(
    samples: list[Sample],
    epochs: int,
    batch: int,
    lr: float,
    device: torch.device,
    seed: int,
):
    train_samples, test_samples = split_samples(samples, test_ratio=0.2, seed=seed)
    train_ds = ImitationDataset(train_samples)
    test_ds = ImitationDataset(test_samples)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate)
    test_dl = DataLoader(test_ds, batch_size=batch, shuffle=False, collate_fn=collate)

    model = PointerPolicyNet(max_hand_combos=7, max_keys=17).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    hist = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    for _ in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_dl, opt, device)
        te_loss, te_acc = eval_epoch(model, test_dl, device)
        hist["train_loss"].append(tr_loss)
        hist["test_loss"].append(te_loss)
        hist["train_acc"].append(tr_acc)
        hist["test_acc"].append(te_acc)
    return model, hist, len(train_ds), len(test_ds)


def save_plots(out_dir: Path, hist: dict[str, list[float]], size_curve: list[tuple[int, float]]):
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(hist["train_loss"], label="train CE loss")
    plt.plot(hist["test_loss"], label="test CE loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Static BC Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bc_static_loss.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(hist["train_acc"], label="train match rate")
    plt.plot(hist["test_acc"], label="test match rate")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Static BC Match Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bc_static_match_rate.png", dpi=140)
    plt.close()

    if size_curve:
        x = [n for n, _ in size_curve]
        y = [acc for _, acc in size_curve]
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, marker="o")
        plt.xlabel("Teacher episodes")
        plt.ylabel("Best test match rate")
        plt.title("Data Size vs Test Match Rate")
        plt.tight_layout()
        plt.savefig(out_dir / "data_size_curve.png", dpi=140)
        plt.close()


def _find_multi_match_case(env: Pick14GymEnv):
    """
    Find a case with >=2 legal match actions (not counting pass) and 7 hand combos.
    """
    for seed in range(20260324, 20260324 + 5000):
        obs, _ = env.reset(seed=seed)
        assert env.state is not None and env._last_encoded is not None
        enc = env._last_encoded
        # Need 3 hand cards -> 7 combos and at least one public.
        if int(obs["hand_valid"].sum()) < 7:
            continue
        public_count = int(obs["public_valid"].sum())
        if public_count <= 0:
            continue
        # Match-valid cells are in columns [0, public_count), pass in public_count.
        match_valid = int(obs["mask"][:, :public_count].sum())
        if match_valid >= 2:
            return seed, obs, env.state, enc, match_valid
    return None


def print_one_case(env: Pick14GymEnv, out_dir: Path):
    found = _find_multi_match_case(env)
    if found is None:
        raise RuntimeError("Could not find a multi-match case in search window")

    seed, obs, st, enc, match_valid = found
    hand_txt = [str(c) for c in st.hands[0]]
    pub_txt = [str(c) for c in st.public]
    case = {
        "seed": seed,
        "match_valid_count": match_valid,
        "hand_cards": hand_txt,
        "public_cards": pub_txt,
        "hand_vecs_9d": obs["hand_vecs"].tolist(),
        "public_vecs_9d": obs["public_vecs"].tolist(),
        "mask_7x17": obs["mask"].astype(int).tolist(),
        "public_count": enc.public_count,
        "notes": "mask[row=hand_combo_i][col=public_j or pass_col=public_count], 1=valid",
    }
    (out_dir / "one_case_encoding.json").write_text(json.dumps(case, indent=2))
    print("=== One case (human-readable, multiple matches) ===")
    print(f"seed={seed} match_valid_count={match_valid}")
    print("hand cards:")
    for i, c in enumerate(hand_txt):
        print(f"  {i}: {c}")
    print("public cards:")
    for i, c in enumerate(pub_txt):
        print(f"  {i}: {c}")
    print("first hand combo vectors (up to valid rows):")
    for i in range(7):
        if int(obs["hand_valid"][i]) == 0:
            break
        print(f"  hand_combo[{i}] = {obs['hand_vecs'][i].tolist()}")
    print("public vectors (valid rows):")
    for j in range(enc.public_count):
        print(f"  public[{j}] = {obs['public_vecs'][j].tolist()}")
    print("valid mask rows (trim to public_count+1):")
    for i in range(7):
        if int(obs["hand_valid"][i]) == 0:
            break
        row = obs["mask"][i][: enc.public_count + 1].astype(int).tolist()
        print(f"  mask[{i}] = {row}")


def main():
    ap = argparse.ArgumentParser(description="Static teacher-labeled BC training for Pick14")
    ap.add_argument("--episodes-start", type=int, default=120)
    ap.add_argument("--episodes-max", type=int, default=1200)
    ap.add_argument("--episodes-step", type=int, default=120)
    ap.add_argument("--target-test-acc", type=float, default=0.9)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-dir", type=str, default="artifacts/rl_static")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Pick14GymEnv(num_players=3, n_hand=3, seed=args.seed)

    size_curve: list[tuple[int, float]] = []
    best = None
    best_model = None
    best_hist = None
    best_eps = None
    samples_cache: list[Sample] = []
    episodes_collected = 0
    seed_base = 1000

    print("[A] Static teacher data + BC loop")
    for eps in range(args.episodes_start, args.episodes_max + 1, args.episodes_step):
        need = eps - episodes_collected
        if need > 0:
            samples_cache.extend(collect_teacher_samples(env, episodes=need, seed_base=seed_base + len(samples_cache)))
            episodes_collected += need

        model, hist, n_train, n_test = train_static_bc(
            samples=samples_cache,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            device=device,
            seed=args.seed,
        )
        test_acc = float(hist["test_acc"][-1])
        size_curve.append((eps, test_acc))
        loss_first = float(hist["test_loss"][0])
        loss_last = float(hist["test_loss"][-1])
        drop = (loss_first - loss_last) / max(1e-9, loss_first)
        print(
            f"  eps={eps:4d} train={n_train:5d} test={n_test:5d} "
            f"test_acc={test_acc:.4f} test_loss_drop={drop:.2%}"
        )

        if best is None or test_acc > best:
            best = test_acc
            best_model = model
            best_hist = hist
            best_eps = eps
        if test_acc >= args.target_test_acc:
            print(f"  target reached at episodes={eps} (test_acc={test_acc:.4f})")
            break

    assert best_model is not None and best_hist is not None and best_eps is not None and best is not None

    print("[B] Save artifacts")
    save_plots(out_dir, best_hist, size_curve)
    torch.save(
        {
            "model": best_model.state_dict(),
            "best_test_acc": best,
            "best_episodes": best_eps,
            "history": best_hist,
            "size_curve": size_curve,
            "config": vars(args),
        },
        out_dir / "checkpoint_bc_static.pt",
    )
    summary_lines = [
        f"best_teacher_episodes={best_eps}",
        f"best_test_match_rate={best:.6f}",
        f"final_train_loss={best_hist['train_loss'][-1]:.6f}",
        f"final_test_loss={best_hist['test_loss'][-1]:.6f}",
        f"final_train_acc={best_hist['train_acc'][-1]:.6f}",
        f"final_test_acc={best_hist['test_acc'][-1]:.6f}",
        f"size_curve={size_curve}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    print("[C] Dump one explicit encoding/mask case")
    print_one_case(env, out_dir)
    print(f"Artifacts: {out_dir}")


if __name__ == "__main__":
    main()

