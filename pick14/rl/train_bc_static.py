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
from pick14.rl.model import DualHeadPolicyNet

MAX_KEYS = 17


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
    return obs_t["public_valid"].sum(dim=1).long()


def grouped_bc_loss_and_acc(logits: torch.Tensor, actions: torch.Tensor, obs_t: dict[str, torch.Tensor], max_keys: int):
    bsz = logits.shape[0]
    device = logits.device
    pass_col = pass_col_from_obs(obs_t)
    k_idx = actions % max_keys
    is_pass_teacher = k_idx.eq(pass_col)

    logp = F.log_softmax(logits, dim=1)
    losses = []
    pred = logits.argmax(dim=1)

    if (~is_pass_teacher).any():
        idx = torch.nonzero(~is_pass_teacher, as_tuple=False).squeeze(1)
        losses.append(F.nll_loss(logp[idx], actions[idx], reduction="mean"))

    if is_pass_teacher.any():
        idx = torch.nonzero(is_pass_teacher, as_tuple=False).squeeze(1)
        pass_col_i = pass_col[idx]
        combo_ids = torch.arange(7, device=device).view(1, 7).expand(idx.shape[0], 7)
        pass_flat = combo_ids * max_keys + pass_col_i.view(-1, 1)
        selected = logp[idx].gather(1, pass_flat)
        group_logp = torch.logsumexp(selected, dim=1)
        losses.append((-group_logp).mean())

    loss = sum(losses) if losses else torch.tensor(0.0, device=device)

    pred_k = pred % max_keys
    pred_is_pass = pred_k.eq(pass_col)
    correct = torch.zeros((bsz,), dtype=torch.bool, device=device)
    correct = torch.where(~is_pass_teacher, pred.eq(actions), correct)
    correct = torch.where(is_pass_teacher, pred_is_pass, correct)
    acc = correct.float().mean()
    return loss, acc


def dual_bc_loss(
    match_logits: torch.Tensor,
    play_logits: torch.Tensor,
    actions: torch.Tensor,
    obs_t: dict[str, torch.Tensor],
    match_flat_dim: int,
    max_keys: int,
    play_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Per-sample mean objective: match grouped BC on match-phase rows, CE on play-phase rows.
    play_loss_weight upweights play gradient (rarer steps) without changing reported per-phase accuracies.
    """
    device = actions.device
    phase_match = obs_t["phase"].squeeze(1) < 0.5
    m_idx = torch.nonzero(phase_match, as_tuple=False).squeeze(1)
    p_idx = torch.nonzero(~phase_match, as_tuple=False).squeeze(1)
    bsz = actions.shape[0]

    loss_parts: list[torch.Tensor] = []
    weights: list[float] = []

    match_ok = match_n = 0.0
    play_ok = play_n = 0.0

    if m_idx.numel() > 0:
        obs_m = {k: v[m_idx] for k, v in obs_t.items()}
        lm, am = grouped_bc_loss_and_acc(
            match_logits[m_idx], actions[m_idx], obs_m, max_keys=max_keys
        )
        loss_parts.append(lm)
        weights.append(float(m_idx.numel()))
        match_ok = float(am.item()) * float(m_idx.numel())
        match_n = float(m_idx.numel())

    if p_idx.numel() > 0:
        tgt = actions[p_idx] - match_flat_dim
        lp = F.cross_entropy(play_logits[p_idx], tgt, reduction="mean")
        loss_parts.append(lp)
        weights.append(float(p_idx.numel()) * play_loss_weight)
        pred = play_logits[p_idx].argmax(dim=1)
        play_ok = float((pred == tgt).float().sum().item())
        play_n = float(p_idx.numel())

    if not loss_parts:
        z = torch.tensor(0.0, device=device)
        return z, {
            "overall_acc": 1.0,
            "match_acc": 1.0,
            "play_acc": 1.0,
            "match_n": 0.0,
            "play_n": 0.0,
        }

    wsum = sum(weights)
    loss = sum(l * w for l, w in zip(loss_parts, weights, strict=True)) / wsum

    overall_ok = match_ok + play_ok
    overall_n = float(bsz)
    stats = {
        "overall_acc": overall_ok / max(1.0, overall_n),
        "match_acc": (match_ok / match_n) if match_n > 0 else 1.0,
        "play_acc": (play_ok / play_n) if play_n > 0 else 1.0,
        "match_n": match_n,
        "play_n": play_n,
    }
    return loss, stats


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


def _accum_epoch(model, dl, opt, device: torch.device, match_flat_dim: int, train: bool, play_loss_weight: float):
    if train:
        model.train()
    else:
        model.eval()

    sum_loss = 0.0
    sum_overall_ok = sum_overall_n = 0.0
    sum_m_ok = sum_m_n = 0.0
    sum_p_ok = sum_p_n = 0.0
    total = 0

    for obs_np, actions_np in dl:
        obs = to_torch(obs_np, device)
        actions = torch.as_tensor(actions_np, device=device)
        with torch.set_grad_enabled(train):
            match_logits, play_logits, _ = model(obs)
            loss, st = dual_bc_loss(
                match_logits,
                play_logits,
                actions,
                obs,
                match_flat_dim=match_flat_dim,
                max_keys=MAX_KEYS,
                play_loss_weight=play_loss_weight,
            )
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        n = float(actions.shape[0])
        sum_loss += float(loss.item()) * n
        sum_overall_ok += st["overall_acc"] * n
        sum_overall_n += n
        if st["match_n"] > 0:
            sum_m_ok += st["match_acc"] * st["match_n"]
            sum_m_n += st["match_n"]
        if st["play_n"] > 0:
            sum_p_ok += st["play_acc"] * st["play_n"]
            sum_p_n += st["play_n"]
        total += int(n)

    denom = max(1, total)
    out = {
        "loss": sum_loss / denom,
        "acc_overall": sum_overall_ok / max(1.0, sum_overall_n),
        "acc_match": (sum_m_ok / sum_m_n) if sum_m_n > 0 else 1.0,
        "acc_play": (sum_p_ok / sum_p_n) if sum_p_n > 0 else 1.0,
    }
    return out


def train_epoch(model, dl, opt, device: torch.device, match_flat_dim: int, play_loss_weight: float):
    return _accum_epoch(model, dl, opt, device, match_flat_dim, train=True, play_loss_weight=play_loss_weight)


@torch.no_grad()
def eval_epoch(model, dl, device: torch.device, match_flat_dim: int, play_loss_weight: float):
    return _accum_epoch(model, dl, None, device, match_flat_dim, train=False, play_loss_weight=play_loss_weight)


def phase_counts(samples: list[Sample]) -> tuple[int, int]:
    n_match = sum(1 for s in samples if float(s.obs["phase"][0]) < 0.5)
    return n_match, len(samples) - n_match


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
    match_flat_dim: int,
    play_loss_weight: float,
    hidden_dim: int = 32,
):
    train_samples, test_samples = split_samples(samples, test_ratio=0.2, seed=seed)
    train_ds = ImitationDataset(train_samples)
    test_ds = ImitationDataset(test_samples)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate)
    test_dl = DataLoader(test_ds, batch_size=batch, shuffle=False, collate_fn=collate)

    model = DualHeadPolicyNet(
        max_hand_combos=7, max_match_keys=MAX_KEYS, hidden_dim=hidden_dim
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    hist: dict[str, list[float]] = {
        "train_loss": [],
        "test_loss": [],
        "train_acc_overall": [],
        "test_acc_overall": [],
        "train_acc_match": [],
        "test_acc_match": [],
        "train_acc_play": [],
        "test_acc_play": [],
    }
    for _ in range(epochs):
        tr = train_epoch(model, train_dl, opt, device, match_flat_dim, play_loss_weight)
        te = eval_epoch(model, test_dl, device, match_flat_dim, play_loss_weight)
        hist["train_loss"].append(tr["loss"])
        hist["test_loss"].append(te["loss"])
        hist["train_acc_overall"].append(tr["acc_overall"])
        hist["test_acc_overall"].append(te["acc_overall"])
        hist["train_acc_match"].append(tr["acc_match"])
        hist["test_acc_match"].append(te["acc_match"])
        hist["train_acc_play"].append(tr["acc_play"])
        hist["test_acc_play"].append(te["acc_play"])
    test_nm, test_np = phase_counts(test_samples)
    return model, hist, len(train_ds), len(test_ds), test_nm, test_np


def save_plots(out_dir: Path, hist: dict[str, list[float]], size_curve: list[tuple[int, float]]):
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(hist["train_loss"], label="train")
    plt.plot(hist["test_loss"], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Static BC (dual-head, weighted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bc_static_loss.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(hist["train_acc_overall"], label="train overall")
    plt.plot(hist["test_acc_overall"], label="test overall")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Static BC Overall Imitation Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bc_static_overall_rate.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(hist["train_acc_match"], label="train match-phase")
    plt.plot(hist["test_acc_match"], label="test match-phase")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Match-Head Imitation (greedy teacher)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bc_static_match_rate.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(hist["train_acc_play"], label="train play-phase")
    plt.plot(hist["test_acc_play"], label="test play-phase")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Play-Head Imitation (stingy teacher)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bc_static_play_rate.png", dpi=140)
    plt.close()

    if size_curve:
        x = [n for n, _ in size_curve]
        y = [acc for _, acc in size_curve]
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, marker="o")
        plt.xlabel("Teacher episodes")
        plt.ylabel("Best test overall accuracy")
        plt.title("Data Size vs Test Overall Accuracy")
        plt.tight_layout()
        plt.savefig(out_dir / "data_size_curve.png", dpi=140)
        plt.close()


def _find_multi_match_case(env: Pick14GymEnv):
    for seed in range(20260324, 20260324 + 5000):
        obs, _ = env.reset(seed=seed)
        assert env.state is not None and env._last_encoded is not None
        enc = env._last_encoded
        if float(obs["phase"][0]) >= 0.5:
            continue
        if int(obs["hand_valid"].sum()) < 7:
            continue
        public_count = int(obs["public_valid"].sum())
        if public_count <= 0:
            continue
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
        "phase": float(obs["phase"][0]),
        "match_valid_count": match_valid,
        "hand_cards": hand_txt,
        "public_cards": pub_txt,
        "hand_vecs_9d": obs["hand_vecs"].tolist(),
        "public_vecs_9d": obs["public_vecs"].tolist(),
        "mask_7x17": obs["mask"].astype(int).tolist(),
        "play_hand_vecs_9d": obs["play_hand_vecs"].tolist(),
        "play_hand_valid": obs["play_hand_valid"].astype(int).tolist(),
        "play_key_mask": obs["play_key_mask"].astype(int).tolist(),
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
    for i in range(7):
        if int(obs["hand_valid"][i]) == 0:
            break
        print(f"  hand_combo[{i}] = {obs['hand_vecs'][i].tolist()}")
    for j in range(enc.public_count):
        print(f"  public[{j}] = {obs['public_vecs'][j].tolist()}")
    for i in range(7):
        if int(obs["hand_valid"][i]) == 0:
            break
        row = obs["mask"][i][: enc.public_count + 1].astype(int).tolist()
        print(f"  mask[{i}] = {row}")


def main():
    ap = argparse.ArgumentParser(description="Static dual-teacher BC (match+play heads, rl_init.md)")
    ap.add_argument("--episodes-start", type=int, default=120)
    ap.add_argument("--episodes-max", type=int, default=1200)
    ap.add_argument("--episodes-step", type=int, default=120)
    ap.add_argument("--target-overall-acc", type=float, default=0.99)
    ap.add_argument("--target-match-acc", type=float, default=0.99)
    ap.add_argument("--target-play-acc", type=float, default=0.99)
    ap.add_argument(
        "--min-test-play-labels",
        type=int,
        default=24,
        help="Require at least this many play-phase labels in the test split before play-acc counts for early stop.",
    )
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="Trunk embedding width D (Linear(9,D)+ReLU; play keys = public + 32 global, LeakyReLU on per-key scores).",
    )
    ap.add_argument(
        "--play-loss-weight",
        type=float,
        default=2.0,
        help="Upweight play-phase CE in the combined objective (rarer than match steps).",
    )
    ap.add_argument("--out-dir", type=str, default="artifacts/rl_static")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Pick14GymEnv(num_players=3, n_hand=3, seed=args.seed)
    mfd = env.match_flat_dim

    size_curve: list[tuple[int, float]] = []
    best = None
    best_model = None
    best_hist = None
    best_eps = None
    samples_cache: list[Sample] = []
    episodes_collected = 0
    seed_base = 1000
    last_test_np = 0

    print("[A] Static teacher data + dual-head BC loop")
    for eps in range(args.episodes_start, args.episodes_max + 1, args.episodes_step):
        need = eps - episodes_collected
        if need > 0:
            samples_cache.extend(collect_teacher_samples(env, episodes=need, seed_base=seed_base + len(samples_cache)))
            episodes_collected += need

        model, hist, n_train, n_test, test_nm, test_np = train_static_bc(
            samples=samples_cache,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            device=device,
            seed=args.seed,
            match_flat_dim=mfd,
            play_loss_weight=args.play_loss_weight,
            hidden_dim=args.hidden_dim,
        )
        te_o = float(hist["test_acc_overall"][-1])
        te_m = float(hist["test_acc_match"][-1])
        te_p = float(hist["test_acc_play"][-1])
        last_test_np = test_np
        size_curve.append((eps, te_o))
        loss_first = float(hist["test_loss"][0])
        loss_last = float(hist["test_loss"][-1])
        drop = (loss_first - loss_last) / max(1e-9, loss_first)
        print(
            f"  eps={eps:4d} train={n_train:5d} test={n_test:5d} "
            f"test_nm={test_nm:4d} test_np={test_np:3d} "
            f"test_overall={te_o:.4f} test_match={te_m:.4f} test_play={te_p:.4f} "
            f"test_loss_drop={drop:.2%}"
        )

        if best is None or te_o > best:
            best = te_o
            best_model = model
            best_hist = hist
            best_eps = eps

        play_eval_ok = test_np >= args.min_test_play_labels and te_p >= args.target_play_acc
        if (
            te_o >= args.target_overall_acc
            and te_m >= args.target_match_acc
            and play_eval_ok
        ):
            print(
                f"  targets reached at episodes={eps} "
                f"(overall={te_o:.4f} match={te_m:.4f} play={te_p:.4f}, test_np={test_np})"
            )
            break

    assert best_model is not None and best_hist is not None and best_eps is not None and best is not None

    if last_test_np < args.min_test_play_labels:
        print(
            f"[note] test split had only {last_test_np} play-phase labels "
            f"(<{args.min_test_play_labels}); increase --episodes-max or lower --min-test-play-labels "
            "for reliable play-head monitoring."
        )

    print("[B] Save artifacts")
    save_plots(out_dir, best_hist, size_curve)
    torch.save(
        {
            "model": best_model.state_dict(),
            "best_test_overall_acc": best,
            "best_episodes": best_eps,
            "history": best_hist,
            "size_curve": size_curve,
            "config": vars(args),
            "match_flat_dim": mfd,
            "hidden_dim": args.hidden_dim,
        },
        out_dir / "checkpoint_bc_static.pt",
    )
    last = -1
    summary_lines = [
        f"best_teacher_episodes={best_eps}",
        f"best_test_overall_acc={best:.6f}",
        f"final_test_overall={best_hist['test_acc_overall'][last]:.6f}",
        f"final_test_match={best_hist['test_acc_match'][last]:.6f}",
        f"final_test_play={best_hist['test_acc_play'][last]:.6f}",
        f"final_train_loss={best_hist['train_loss'][last]:.6f}",
        f"final_test_loss={best_hist['test_loss'][last]:.6f}",
        f"size_curve={size_curve}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    print("[C] Dump one explicit encoding/mask case")
    print_one_case(env, out_dir)
    print(f"Artifacts: {out_dir}")


if __name__ == "__main__":
    main()
