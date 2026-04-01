"""
curriculum.md: Phase 1 mixed BC (match + play, shared trunk) → Phase 2 critic targets from rl.md §4.1.

Phase 2 no longer uses gym ``0``/``-1`` rewards. **Match critic:** MSE of ``V_agent`` vs immediate agent
match capture points ``A_t``. **Play critic:** MSE of the policy-weighted expectation
``sum_a pi(a) V_opp(S_post(a))`` vs atomic ``r_play = -O_t`` (detached ``pi`` from the current actor).

Uses rl.md §1.5 **baseline** teacher for BC. Phase 2 can **freeze** actors (critic-only) or **joint**-train
BC + critic on **static** bundles (no simulation during epochs). Critic rows may be **loaded** from disk
(``torch.save``) after a one-time collection.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import cycle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset

from pick14.rl.env import Pick14GymEnv
from pick14.rl.rlmd_model import RLmdPPOAgent
from pick14.rl.rlmd_obs import RlmdObservationWrapper, encode_rlmd_post_play_from_state
from pick14.rl.sim_core import (
    MAX_PUBLIC_SLOTS,
    apply_play,
    clone_state,
    legal_play_moves,
    total_score_points,
)
from pick14.rl.train_bc_static import (
    BC_PHASE_SEMANTICS,
    ImitationDataset,
    MixedPhaseBatchSampler,
    Sample,
    collate,
    collect_teacher_samples,
    dual_bc_loss,
    eval_epoch,
    phase_counts,
    split_samples_stratified,
    to_torch as bc_to_torch,
    train_static_bc,
)


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


@dataclass(slots=True)
class PlayCriticEVRow:
    """One play-phase decision: counterfactual post-play obs per hand slot, ``pi``, and ``r_play = -O_t``."""

    obs_pre_play: dict[str, np.ndarray]
    post_obs_by_hand: list[dict[str, np.ndarray] | None]
    pi: np.ndarray
    target_neg_opp_gain: float


def _total_opp_score(state: Any, learning_player: int) -> float:
    return float(
        sum(total_score_points(state, j) for j in range(state.num_players) if j != learning_player)
    )


def collect_critic_bootstrap_data(
    env: RlmdObservationWrapper,
    model: RLmdPPOAgent,
    device: torch.device,
    episodes: int,
    gamma: float,
    seed_base: int,
    max_steps: int = 512,
    *,
    policy: str = "sample",
) -> tuple[list[tuple[dict[str, np.ndarray], float]], list[PlayCriticEVRow]]:
    """
    Roll out seat-0 policy; build Phase-2 critic targets per rl.md §4.1 (not gym ``rew``).

    * **Match rows:** ``(S_pre_match, A_t)`` where ``A_t`` is learning-seat score-pile gain from that match
      step (``0`` on pass).
    * **Play rows:** :class:`PlayCriticEVRow` with counterfactual ``S_post`` per legal discard, ``pi(a)``
      from the play head (softmax, masked), and target ``-O_t`` with ``O_t`` = total opponent score gain
      during autoplay after the realized discard.

    ``gamma`` is unused but kept for CLI compatibility (full discounted ``G`` is for RL/PPO, not this bootstrap).

    ``policy``: ``sample`` | ``deterministic`` | ``teacher`` (for eval).
    """
    del gamma  # full G_match/G_play used in PPO, not in this MSE bootstrap
    if policy not in ("sample", "deterministic", "teacher"):
        raise ValueError(policy)
    match_rows: list[tuple[dict[str, np.ndarray], float]] = []
    play_rows: list[PlayCriticEVRow] = []
    model.eval()
    base: Pick14GymEnv = env.unwrapped
    lp = base.learning_player
    n_hand = base.n_hand
    n_slots = model.max_play_hand

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        for _ in range(max_steps):
            if base.state is None or base.state.current_player != lp:
                break
            st = base.state
            is_match = float(obs["phase"][0]) < 0.5

            if is_match:
                obs_pre = {k: v.copy() for k, v in obs.items()}
                s_lp_before = float(total_score_points(st, lp))
                if policy == "teacher":
                    action = int(base.teacher_action())
                else:
                    with torch.no_grad():
                        ml, pl, _, _ = model(_to_torch_obs(obs, device))
                        mm = torch.as_tensor(obs["mask_match"], device=device).reshape(1, -1).bool()
                        mm = mm[:, : ml.shape[1]]
                        logits = ml.masked_fill(~mm, -1e9)
                        if policy == "deterministic":
                            sub = int(torch.argmax(logits, dim=1).item())
                        else:
                            sub = int(Categorical(logits=logits).sample().item())
                    action = sub
                obs, _, term, trunc, _ = env.step(action)
                s_lp_after = float(total_score_points(base.state, lp))
                a_t = s_lp_after - s_lp_before
                match_rows.append((obs_pre, float(a_t)))
                if term or trunc:
                    break
                continue

            # PLAY: counterfactual post-play encodings + policy weights
            opp_before = _total_opp_score(st, lp)
            obs_pre = {k: v.copy() for k, v in obs.items()}
            post_by_hand: list[dict[str, np.ndarray] | None] = [None] * n_slots
            for pm in legal_play_moves(st):
                hi = int(pm.hand_index)
                if hi < 0 or hi >= n_slots:
                    continue
                st_hi = clone_state(st)
                apply_play(st_hi, hi)
                post_by_hand[hi] = encode_rlmd_post_play_from_state(st_hi, lp, n_hand)
                for k, v in post_by_hand[hi].items():
                    post_by_hand[hi][k] = np.asarray(v).copy()

            with torch.no_grad():
                ml, pl, _, _ = model(_to_torch_obs(obs, device))
                pv = obs["play_hand_valid"]
                legal = torch.as_tensor(pv, device=device, dtype=torch.bool).view(1, -1)[:, : pl.shape[1]]
                logits = pl.clone().masked_fill(~legal, -1e9)
                pi = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            if policy == "teacher":
                action = int(base.teacher_action())
            elif policy == "deterministic":
                sub = int(torch.argmax(logits, dim=1).item())
                action = sub + model.match_flat_dim
            else:
                sub = int(Categorical(logits=logits).sample().item())
                action = sub + model.match_flat_dim

            obs, _, term, trunc, _ = env.step(action)
            opp_after = _total_opp_score(base.state, lp)
            o_t = opp_after - opp_before
            r_play = -float(o_t)
            play_rows.append(
                PlayCriticEVRow(
                    obs_pre_play=obs_pre,
                    post_obs_by_hand=post_by_hand,
                    pi=pi.astype(np.float64, copy=False),
                    target_neg_opp_gain=r_play,
                )
            )
            if term or trunc:
                break

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


def critic_match_obs_has_legal_action(obs: dict[str, np.ndarray]) -> bool:
    """True if ``mask_match`` has at least one legal cell (combo or pass row)."""
    mm = np.asarray(obs["mask_match"])
    return bool(mm.reshape(-1).astype(bool).any())


def critic_match_obs_has_scoring_legal(obs: dict[str, np.ndarray], *, pass_row: int) -> bool:
    """
    True if some **combo** cell is legal (rows ``0 .. pass_row-1``).

    Row index ``pass_row`` is the pass row (``env._max_match_combos`` / ``max_match_combo_slots(n_hand)``);
    it is excluded because pass is almost always legal when in match phase.
    """
    mm = np.asarray(obs["mask_match"])
    if mm.ndim != 2 or mm.shape[0] < pass_row:
        return False
    scoring = mm[:pass_row, :]
    return bool(scoring.reshape(-1).astype(bool).any())


def _play_ev_loss(
    model: RLmdPPOAgent,
    device: torch.device,
    row: PlayCriticEVRow,
    *,
    use_opp_head: bool,
) -> torch.Tensor:
    """``use_opp_head`` True → train ``V_opp``; False → ``V_agent`` on post-play states (swap diagnostic)."""
    parts: list[torch.Tensor] = []
    pi = torch.as_tensor(row.pi, device=device, dtype=torch.float32)
    for hi, op in enumerate(row.post_obs_by_hand):
        if op is None or hi >= pi.numel():
            continue
        w = pi[hi].detach()
        if w.item() == 0.0:
            continue
        oti = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in op.items()}
        _, _, va, vo = model(oti)
        v = vo if use_opp_head else va
        parts.append(w * v.squeeze(0))
    if not parts:
        return torch.tensor(0.0, device=device)
    ev = torch.stack(parts).sum()
    y = torch.tensor(row.target_neg_opp_gain, device=device, dtype=torch.float32)
    return F.mse_loss(ev, y)


def train_critics_mse(
    model: RLmdPPOAgent,
    match_rows: list[tuple[dict[str, np.ndarray], float]],
    play_rows: list[PlayCriticEVRow],
    device: torch.device,
    epochs: int,
    batch: int,
    lr: float,
) -> dict[str, list[float]]:
    """
    Phase 2: match critic MSE vs ``A_t``; play critic MSE vs ``sum_a pi(a) V_opp(S_post(a))`` vs ``-O_t``.
    """
    model.set_requires_grad_actor_trunk(False)
    model.set_requires_grad_critics(True)
    ds_m = CriticDataset(match_rows)
    dl_m = DataLoader(ds_m, batch_size=batch, shuffle=True, collate_fn=collate_critic) if ds_m else None
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    hist = {"loss_m": [], "loss_p": [], "loss": []}
    n_play = len(play_rows)

    for _ in range(epochs):
        loss_m_acc = 0.0
        loss_p_acc = 0.0
        n_m = 0
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

        if n_play > 0:
            perm = np.random.permutation(n_play)
            for s in range(0, n_play, batch):
                idxs = perm[s : s + batch]
                opt.zero_grad()
                chunk_loss = torch.zeros((), device=device)
                for j in idxs:
                    row = play_rows[int(j)]
                    chunk_loss = chunk_loss + _play_ev_loss(model, device, row, use_opp_head=True)
                chunk_loss = chunk_loss / max(1, len(idxs))
                chunk_loss.backward()
                opt.step()
                loss_p_acc += float(chunk_loss.item()) * max(1, len(idxs))

        hist["loss_m"].append(loss_m_acc / max(1, n_m))
        hist["loss_p"].append(loss_p_acc / max(1, n_play if n_play else 1))
        hist["loss"].append(
            (loss_m_acc + loss_p_acc) / max(1, n_m + (n_play if n_play else 0))
        )
    model.set_requires_grad_actor_trunk(True)
    model.set_requires_grad_critics(True)
    return hist


def eval_atomic_critic_metrics(
    model: RLmdPPOAgent,
    device: torch.device,
    match_rows: list[tuple[dict[str, np.ndarray], float]],
    play_rows: list[PlayCriticEVRow],
) -> dict[str, float]:
    """Held-out RMSE for match ``A_t`` and play EV vs ``-O_t`` (correct vs swapped play head)."""
    model.eval()
    out: dict[str, float] = {}
    with torch.no_grad():
        if match_rows:
            sse_a = sse_o = 0.0
            for o, t in match_rows:
                obs_t = _to_torch_obs(o, device)
                _, _, va, vo = model(obs_t)
                d_a = float(va.item()) - t
                d_o = float(vo.item()) - t
                sse_a += d_a * d_a
                sse_o += d_o * d_o
            n = len(match_rows)
            out["n_match_rows"] = float(n)
            out["rmse_match_v_agent"] = float(np.sqrt(sse_a / n))
            out["rmse_match_swap_v_opp"] = float(np.sqrt(sse_o / n))
        if play_rows:
            sse_p = sse_swap = 0.0
            for row in play_rows:
                y = row.target_neg_opp_gain
                ev_o = 0.0
                ev_a = 0.0
                for hi, op in enumerate(row.post_obs_by_hand):
                    if op is None or hi >= len(row.pi):
                        continue
                    w = float(row.pi[hi])
                    if w == 0.0:
                        continue
                    oti = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in op.items()}
                    _, _, va, vo = model(oti)
                    ev_o += w * float(vo.item())
                    ev_a += w * float(va.item())
                d_o = ev_o - y
                d_a = ev_a - y
                sse_p += d_o * d_o
                sse_swap += d_a * d_a
            n = len(play_rows)
            out["n_play_rows"] = float(n)
            out["rmse_play_ev_v_opp"] = float(np.sqrt(sse_p / n))
            out["rmse_play_ev_swap_v_agent"] = float(np.sqrt(sse_swap / n))
    return out


CRITIC_PARAM_FRAGMENTS = (
    "layer2_critic_agent",
    "layer2_critic_opp",
    "critic_agent_mlp",
    "critic_opp_mlp",
)


def critic_actor_param_groups(model: RLmdPPOAgent, lr_critic: float, lr_actor: float) -> list[dict[str, Any]]:
    crit: list[torch.nn.Parameter] = []
    act: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if any(x in name for x in CRITIC_PARAM_FRAGMENTS):
            crit.append(p)
        else:
            act.append(p)
    return [{"params": act, "lr": lr_actor}, {"params": crit, "lr": lr_critic}]


def save_critic_bundle(
    path: Path | str,
    match_rows: list[tuple[dict[str, np.ndarray], float]],
    play_rows: list[PlayCriticEVRow],
    meta: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    play_ser = [
        {
            "obs_pre_play": {k: np.asarray(v).copy() for k, v in r.obs_pre_play.items()},
            "post_obs_by_hand": [
                None if x is None else {k: np.asarray(v).copy() for k, v in x.items()}
                for x in r.post_obs_by_hand
            ],
            "pi": np.asarray(r.pi, dtype=np.float64).copy(),
            "target_neg_opp_gain": float(r.target_neg_opp_gain),
        }
        for r in play_rows
    ]
    match_ser = [({k: np.asarray(v).copy() for k, v in o.items()}, float(t)) for o, t in match_rows]
    torch.save(
        {"version": 1, "meta": meta or {}, "match_rows": match_ser, "play_rows": play_ser},
        path,
    )


def load_critic_bundle(
    path: Path | str,
) -> tuple[list[tuple[dict[str, np.ndarray], float]], list[PlayCriticEVRow], dict[str, Any]]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    if int(d.get("version", 0)) != 1:
        raise ValueError(f"unknown critic bundle version: {d.get('version')!r}")
    mr = [(o, float(t)) for o, t in d["match_rows"]]
    pr = [
        PlayCriticEVRow(
            obs_pre_play=x["obs_pre_play"],
            post_obs_by_hand=x["post_obs_by_hand"],
            pi=np.asarray(x["pi"], dtype=np.float64),
            target_neg_opp_gain=float(x["target_neg_opp_gain"]),
        )
        for x in d["play_rows"]
    ]
    return mr, pr, d.get("meta", {})


def train_joint_bc_critic_static(
    model: RLmdPPOAgent,
    train_samples: list[Sample],
    match_rows: list[tuple[dict[str, np.ndarray], float]],
    play_rows: list[PlayCriticEVRow],
    device: torch.device,
    *,
    epochs: int,
    bc_batch: int,
    critic_batch: int,
    lr_actor: float,
    lr_critic: float,
    w_bc: float,
    w_match: float,
    w_play: float,
    match_flat_dim: int,
    pass_row: int,
    public_slots: int,
    play_loss_weight: float,
    seed: int,
) -> dict[str, list[float]]:
    """
    One optimizer step per BC batch: BC imitation + match-critic MSE + mean play-EV MSE.
    All parameters trainable (shared L1 / embeddings get gradients from all terms). Play-head ``pi``
    in the EV loss stays **detached** so the play actor is not pushed by the critic target (BC drives actors).
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    model.set_requires_grad_actor_trunk(True)
    model.set_requires_grad_critics(True)

    train_ds = ImitationDataset(train_samples)
    train_bs = MixedPhaseBatchSampler(train_samples, batch_size=bc_batch, seed=seed)
    train_dl = DataLoader(train_ds, batch_sampler=train_bs, collate_fn=collate)
    dl_m = None
    if match_rows:
        bs_m = min(critic_batch, len(match_rows))
        dl_m = DataLoader(
            CriticDataset(match_rows), batch_size=bs_m, shuffle=True, collate_fn=collate_critic
        )
    opt = torch.optim.Adam(critic_actor_param_groups(model, lr_critic, lr_actor))
    hist: dict[str, list[float]] = {k: [] for k in ("loss_bc", "loss_m", "loss_p", "loss_total")}
    rng = np.random.default_rng(seed & 0xFFFFFFFF)

    for ep in range(epochs):
        match_cycle = cycle(dl_m) if dl_m is not None else None
        sum_bc = sum_m = sum_p = sum_tot = 0.0
        n_steps = 0
        for obs_np, actions_np in train_dl:
            obs_t = bc_to_torch(obs_np, device)
            actions_t = torch.as_tensor(actions_np, device=device, dtype=torch.long)
            ml, pl, _, _ = model(obs_t)
            l_bc, _ = dual_bc_loss(
                ml,
                pl,
                actions_t,
                obs_t,
                match_flat_dim,
                pass_row,
                public_slots,
                play_loss_weight,
            )

            l_m = torch.tensor(0.0, device=device)
            if match_cycle is not None:
                m_obs, m_tgt = next(match_cycle)
                m_obs_t = {k: torch.as_tensor(v, device=device) for k, v in m_obs.items()}
                y = torch.as_tensor(m_tgt, device=device)
                _, _, va, _ = model(m_obs_t)
                l_m = F.mse_loss(va, y)

            l_p = torch.tensor(0.0, device=device)
            if play_rows:
                bsz = min(critic_batch, len(play_rows))
                ix = rng.choice(len(play_rows), size=bsz, replace=False)
                l_p = torch.stack(
                    [_play_ev_loss(model, device, play_rows[int(i)], use_opp_head=True) for i in ix]
                ).mean()

            loss = w_bc * l_bc + w_match * l_m + w_play * l_p
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            sum_bc += float(l_bc.detach().item())
            sum_m += float(l_m.detach().item())
            sum_p += float(l_p.detach().item())
            sum_tot += float(loss.detach().item())
            n_steps += 1

        hist["loss_bc"].append(sum_bc / max(1, n_steps))
        hist["loss_m"].append(sum_m / max(1, n_steps))
        hist["loss_p"].append(sum_p / max(1, n_steps))
        hist["loss_total"].append(sum_tot / max(1, n_steps))

    return hist


def main():
    ap = argparse.ArgumentParser(description="curriculum.md Phase 1 BC + Phase 2 critic bootstrap (baseline teacher)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", type=str, default="artifacts/rl_curriculum")
    ap.add_argument("--bc-episodes", type=int, default=900, help="Teacher rollouts for BC dataset (increase if <95% acc).")
    ap.add_argument("--bc-epochs", type=int, default=48, help="Phase 1 BC epochs (joint match+play, mixed batches).")
    ap.add_argument("--bc-batch", type=int, default=128)
    ap.add_argument("--bc-lr", type=float, default=1e-3)
    ap.add_argument("--bc-test-ratio", type=float, default=0.2, help="Held-out fraction per phase (match vs play stratified).")
    ap.add_argument("--play-loss-weight", type=float, default=2.0)
    ap.add_argument(
        "--min-train-match-acc",
        type=float,
        default=0.95,
        help="Phase 1 gate: train split match-head accuracy (NaN if no rows).",
    )
    ap.add_argument(
        "--min-test-match-acc",
        type=float,
        default=0.95,
        help="Phase 1 gate: test split match-head accuracy.",
    )
    ap.add_argument(
        "--min-train-play-acc",
        type=float,
        default=0.95,
        help="Phase 1 gate: train play-head accuracy.",
    )
    ap.add_argument(
        "--min-test-play-acc",
        type=float,
        default=0.95,
        help="Phase 1 gate: test play-head accuracy.",
    )
    ap.add_argument(
        "--min-test-overall-acc",
        type=float,
        default=0.95,
        help="Phase 1 gate: test overall imitation accuracy.",
    )
    ap.add_argument(
        "--min-train-play-labels",
        type=int,
        default=48,
        help="Minimum play-phase rows in the train split to enforce play accuracy gates.",
    )
    ap.add_argument(
        "--min-test-play-labels",
        type=int,
        default=48,
        help="Minimum play-phase rows in the test split.",
    )
    ap.add_argument("--critic-rollout-episodes", type=int, default=500)
    ap.add_argument("--critic-epochs", type=int, default=12)
    ap.add_argument("--critic-batch", type=int, default=64)
    ap.add_argument("--critic-lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument(
        "--critic-static-in",
        type=str,
        default="",
        help="Load critic bundle (.pt from --critic-static-save); skips on-policy collection.",
    )
    ap.add_argument(
        "--critic-static-save",
        type=str,
        default="",
        help="Save collected critic rows to this path (torch bundle, version=1).",
    )
    ap.add_argument(
        "--phase2-joint",
        action="store_true",
        help="Train BC + critic together on static data (unfreeze shared trunk; no sim during epochs).",
    )
    ap.add_argument("--joint-epochs", type=int, default=48)
    ap.add_argument("--joint-lr-actor", type=float, default=1e-4)
    ap.add_argument("--joint-lr-critic", type=float, default=3e-4)
    ap.add_argument("--joint-w-bc", type=float, default=1.0)
    ap.add_argument("--joint-w-match", type=float, default=0.25)
    ap.add_argument("--joint-w-play", type=float, default=0.25)
    ap.add_argument(
        "--post-bc-eval-batch",
        type=int,
        default=256,
        help="Batch size for BC re-check on held-out samples after Phase 2 (actors unchanged when frozen).",
    )
    ap.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help=(
            "Load model (and BC history) from a prior checkpoint_curriculum.pt; skip Phase 1 BC training. "
            "Use the same --seed and --bc-episodes as the original run so the teacher split matches. "
            "Typical with --critic-static-in and --phase2-joint for more joint epochs on the same bundle."
        ),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from pick14.rl.agents import table_all_baseline

    base_env = Pick14GymEnv(table_all_baseline(2), n_hand=3, seed=args.seed)
    env = RlmdObservationWrapper(base_env)
    mfd = base_env.match_flat_dim
    pass_row = base_env._max_match_combos

    print(
        "[Phase 1] Mixed BC — baseline teacher (greedy-for-public match + caution play); "
        "stratified split + mixed batches (curriculum.md §1.2)",
        flush=True,
    )
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
    resume_path = (args.resume_checkpoint or "").strip()
    ckpt_prev: dict[str, Any] | None = None
    if resume_path:
        rp = Path(resume_path)
        if not rp.is_file():
            raise SystemExit(f"--resume-checkpoint not found: {rp.resolve()}")
        ckpt_prev = torch.load(rp, map_location=device, weights_only=False)
        train_samples, test_samples = split_samples_stratified(
            samples, test_ratio=args.bc_test_ratio, seed=args.seed
        )
        n_tr, n_te = len(train_samples), len(test_samples)
        tr_nm, tr_np = phase_counts(train_samples)
        te_nm, te_np = phase_counts(test_samples)
        model = RLmdPPOAgent.from_env(base_env, dropout=0.0).to(device)
        model.load_state_dict(ckpt_prev["model"])
        hist_bc = ckpt_prev["hist_bc"]
        print(f"[Phase 1] skipped (resume) — loaded model from {rp.resolve()}", flush=True)
        gate_bs = args.post_bc_eval_batch
        tr_dl = DataLoader(
            ImitationDataset(train_samples),
            batch_size=gate_bs,
            shuffle=False,
            collate_fn=collate,
        )
        te_dl = DataLoader(
            ImitationDataset(test_samples),
            batch_size=gate_bs,
            shuffle=False,
            collate_fn=collate,
        )
        tr_eval = eval_epoch(
            model, tr_dl, device, mfd, args.play_loss_weight, pass_row, MAX_PUBLIC_SLOTS
        )
        te_eval = eval_epoch(
            model, te_dl, device, mfd, args.play_loss_weight, pass_row, MAX_PUBLIC_SLOTS
        )
        tr_m, tr_p = float(tr_eval["acc_match"]), float(tr_eval["acc_play"])
        te_m, te_p = float(te_eval["acc_match"]), float(te_eval["acc_play"])
        te_o = float(te_eval["acc_overall"])
        print(
            f"  split: train={n_tr} (match={tr_nm}, play={tr_np}) | test={n_te} (match={te_nm}, play={te_np}) | "
            f"resume eval train match={tr_m} play={tr_p} | test match={te_m} play={te_p} overall={te_o:.4f}",
            flush=True,
        )
    else:
        model, hist_bc, n_tr, n_te, tr_nm, tr_np, te_nm, te_np, test_samples = train_static_bc(
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
            env=base_env,
            pass_row=pass_row,
            public_slots=MAX_PUBLIC_SLOTS,
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
    if not resume_path:
        last = -1
        print(
            f"  split: train={n_tr} (match={tr_nm}, play={tr_np}) | test={n_te} (match={te_nm}, play={te_np}) | "
            f"last train match={hist_bc['train_acc_match'][last]} play={hist_bc['train_acc_play'][last]} | "
            f"test match={hist_bc['test_acc_match'][last]} play={hist_bc['test_acc_play'][last]} "
            f"overall={hist_bc['test_acc_overall'][last]:.4f}",
            flush=True,
        )
        tr_p = float(hist_bc["train_acc_play"][last])
        te_p = float(hist_bc["test_acc_play"][last])
        tr_m = float(hist_bc["train_acc_match"][last])
        te_m = float(hist_bc["test_acc_match"][last])
        te_o = float(hist_bc["test_acc_overall"][last])
        train_samples, _ = split_samples_stratified(samples, test_ratio=args.bc_test_ratio, seed=args.seed)
    # resume_path branch already set tr_*, te_*, train_samples, test_samples, and printed split.

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

    def _check_acc(name: str, val: float, thresh: float) -> None:
        if math.isnan(val):
            return
        if val < thresh:
            raise SystemExit(f"{name}={val:.4f} < {thresh}; increase --bc-epochs / --bc-episodes or tune --play-loss-weight.")

    _check_acc("Train match accuracy", tr_m, args.min_train_match_acc)
    _check_acc("Test match accuracy", te_m, args.min_test_match_acc)
    _check_acc("Train play accuracy", tr_p, args.min_train_play_acc)
    _check_acc("Test play accuracy", te_p, args.min_test_play_acc)
    _check_acc("Test overall accuracy", te_o, args.min_test_overall_acc)

    if args.critic_static_in:
        mr, pr, bundle_meta = load_critic_bundle(args.critic_static_in)
        print(
            f"[Phase 2] loaded critic bundle ({args.critic_static_in}): "
            f"match={len(mr)} play={len(pr)} meta={bundle_meta}",
            flush=True,
        )
    else:
        print(
            "[Phase 2] On-policy critic collection — match A_t, play EV vs -O_t (rl.md §4.1)",
            flush=True,
        )
        mr, pr = collect_critic_bootstrap_data(
            env,
            model,
            device,
            episodes=args.critic_rollout_episodes,
            gamma=args.gamma,
            seed_base=5000,
        )
        print(f"  collected match={len(mr)} play={len(pr)}", flush=True)
        if args.critic_static_save:
            save_critic_bundle(
                Path(args.critic_static_save),
                mr,
                pr,
                {
                    "critic_rollout_episodes": args.critic_rollout_episodes,
                    "seed_base": 5000,
                },
            )
            print(f"  saved critic bundle → {args.critic_static_save}", flush=True)

    if args.phase2_joint:
        print(
            "[Phase 2 joint] Static BC (train split) + static critic; shared Transformer L1 + embeddings trainable.",
            flush=True,
        )
        jhist = train_joint_bc_critic_static(
            model,
            train_samples,
            mr,
            pr,
            device,
            epochs=args.joint_epochs,
            bc_batch=args.bc_batch,
            critic_batch=args.critic_batch,
            lr_actor=args.joint_lr_actor,
            lr_critic=args.joint_lr_critic,
            w_bc=args.joint_w_bc,
            w_match=args.joint_w_match,
            w_play=args.joint_w_play,
            match_flat_dim=mfd,
            pass_row=pass_row,
            public_slots=MAX_PUBLIC_SLOTS,
            play_loss_weight=args.play_loss_weight,
            seed=args.seed,
        )
        if (
            ckpt_prev is not None
            and ckpt_prev.get("hist_critic", {}).get("mode") == "joint_static"
            and isinstance(ckpt_prev["hist_critic"].get("joint"), dict)
        ):
            pj = ckpt_prev["hist_critic"]["joint"]
            for k in jhist:
                if k in pj and isinstance(pj[k], list):
                    jhist[k] = list(pj[k]) + list(jhist[k])
            print(
                f"  appended joint history: +{args.joint_epochs} epochs "
                f"(total joint epochs in summary={len(jhist['loss_total'])})",
                flush=True,
            )
        n_joint_ep = len(jhist["loss_total"])
        log_every = max(1, n_joint_ep // 16)
        for i, (lb, lm, lp_, lt) in enumerate(
            zip(jhist["loss_bc"], jhist["loss_m"], jhist["loss_p"], jhist["loss_total"], strict=True),
            start=1,
        ):
            if i == 1 or i == n_joint_ep or i % log_every == 0:
                print(
                    f"  joint ep {i:3d}: L_bc={lb:.5f} L_match={lm:.5f} (RMSE~{math.sqrt(max(0.0, lm)):.3f}) | "
                    f"L_play={lp_:.5f} (RMSE~{math.sqrt(max(0.0, lp_)):.3f}) | L_total={lt:.5f}",
                    flush=True,
                )
        crit_hist = {
            "mode": "joint_static",
            "joint": jhist,
            "loss_m": jhist["loss_m"],
            "loss_p": jhist["loss_p"],
            "loss": jhist["loss_total"],
        }
        print(f"  joint last L_total={jhist['loss_total'][-1]:.6f}", flush=True)
    else:
        print(
            "[Phase 2 frozen actors] Critic MSE — match vs A_t, play vs sum_a pi(a) V_opp(S_post(a)) vs -O_t",
            flush=True,
        )
        crit_hist = train_critics_mse(
            model,
            mr,
            pr,
            device=device,
            epochs=args.critic_epochs,
            batch=args.critic_batch,
            lr=args.critic_lr,
        )
        crit_hist["mode"] = "critic_only_frozen_actor"
        for i, (lm, lp_, lj) in enumerate(
            zip(crit_hist["loss_m"], crit_hist["loss_p"], crit_hist["loss"], strict=True), start=1
        ):
            print(
                f"  critic epoch {i:3d}: loss_m={lm:.6f} (RMSE~{math.sqrt(max(0.0, lm)):.4f}) | "
                f"loss_p={lp_:.6f} (RMSE~{math.sqrt(max(0.0, lp_)):.4f}) | joint={lj:.6f}",
                flush=True,
            )
        print(f"  critic loss last (joint)={crit_hist['loss'][-1]:.6f}", flush=True)

    test_ds = ImitationDataset(test_samples)
    test_dl = DataLoader(test_ds, batch_size=args.post_bc_eval_batch, shuffle=False, collate_fn=collate)
    post = eval_epoch(
        model,
        test_dl,
        device,
        mfd,
        args.play_loss_weight,
        pass_row,
        MAX_PUBLIC_SLOTS,
    )
    post_tag = "after joint BC+critic" if args.phase2_joint else "actor trunk frozen in Phase 2"
    print(
        f"[Post Phase 2] BC eval on held-out split ({post_tag}): "
        f"loss={post['loss']:.4f} acc_overall={post['acc_overall']:.4f} "
        f"acc_match={post['acc_match']} acc_play={post['acc_play']}",
        flush=True,
    )
    if post["acc_overall"] + 1e-6 < args.min_test_overall_acc:
        raise SystemExit(
            f"Post–Phase 2 overall BC acc {post['acc_overall']:.4f} < {args.min_test_overall_acc} "
            "(unexpected if critics were frozen from actors)."
        )

    ckpt = {
        "model": model.state_dict(),
        "hist_bc": hist_bc,
        "hist_critic": crit_hist,
        "bc_post_phase2_eval": post,
        "bc_data_sizes": bc_sizes,
        "bc_phase_semantics": BC_PHASE_SEMANTICS,
        "config": vars(args),
        "match_flat_dim": mfd,
        "baseline_teacher": "greedy_for_public_caution",
    }
    torch.save(ckpt, out_dir / "checkpoint_curriculum.pt")
    summary_payload = _sanitize_json(
        {
            "bc": hist_bc,
            "critic": crit_hist,
            "bc_post_phase2_eval": post,
            "bc_data_sizes": bc_sizes,
            "bc_phase_semantics": BC_PHASE_SEMANTICS,
            "config": vars(args),
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, allow_nan=False))
    print(f"Saved {out_dir / 'checkpoint_curriculum.pt'}")


if __name__ == "__main__":
    main()
