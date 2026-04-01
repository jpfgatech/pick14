"""Critic bundle I/O and joint static trainer smoke tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("torch")

from pick14.rl.agents import table_all_baseline
from pick14.rl.env import Pick14GymEnv
from pick14.rl.pretrain_curriculum import (
    PlayCriticEVRow,
    collect_critic_bootstrap_data,
    load_critic_bundle,
    save_critic_bundle,
    train_joint_bc_critic_static,
)
from pick14.rl.rlmd_model import RLmdPPOAgent
from pick14.rl.rlmd_obs import RlmdObservationWrapper
from pick14.rl.sim_core import MAX_PUBLIC_SLOTS
from pick14.rl.train_bc_static import Sample, split_samples_stratified


def _tiny_play_row(n_hand: int = 3) -> PlayCriticEVRow:
    """Minimal fake row for bundle I/O (one legal slot)."""
    slots = 4
    z = np.zeros((1,), dtype=np.float32)
    obs = {
        "seq_agent_feats": np.zeros((8, 9), dtype=np.float32),
        "seq_agent_roles": np.zeros((8,), dtype=np.int64),
        "seq_agent_mask": np.zeros((8,), dtype=np.bool_),
        "seq_critic_feats": np.zeros((16, 9), dtype=np.float32),
        "seq_critic_roles": np.zeros((16,), dtype=np.int64),
        "seq_critic_mask": np.zeros((16,), dtype=np.bool_),
        "mask_match": np.zeros((8, MAX_PUBLIC_SLOTS), dtype=np.int8),
        "play_hand_valid": np.zeros((slots,), dtype=np.int8),
        "phase": np.array([1.0], dtype=np.float32),
    }
    post = {k: np.copy(v) for k, v in obs.items()}
    pi = np.zeros((slots,), dtype=np.float64)
    pi[0] = 1.0
    lst: list[dict[str, np.ndarray] | None] = [None] * slots
    lst[0] = post
    return PlayCriticEVRow(
        obs_pre_play=obs,
        post_obs_by_hand=lst,
        pi=pi,
        target_neg_opp_gain=-2.0,
    )


def test_critic_bundle_roundtrip():
    mr = [(_tiny_play_row().obs_pre_play, 3.5)]
    pr = [_tiny_play_row()]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "b.pt"
        save_critic_bundle(p, mr, pr, {"k": 1})
        mr2, pr2, meta = load_critic_bundle(p)
    assert meta.get("k") == 1
    assert len(mr2) == 1 and len(pr2) == 1
    assert mr2[0][1] == 3.5
    assert pr2[0].target_neg_opp_gain == -2.0


def test_joint_train_one_step_smoke():
    base = Pick14GymEnv(table_all_baseline(2), n_hand=3, seed=0)
    env = RlmdObservationWrapper(base)
    model = RLmdPPOAgent.from_env(base)
    device = torch.device("cpu")
    mr, pr = collect_critic_bootstrap_data(
        env, model, device, episodes=1, gamma=0.99, seed_base=0, max_steps=64, policy="teacher"
    )
    assert len(mr) >= 1
    samples = [
        Sample(obs={k: v.copy() for k, v in mr[0][0].items()}, action=0),
        Sample(obs={k: v.copy() for k, v in pr[0].obs_pre_play.items()}, action=base.match_flat_dim),
    ]
    train_s, _ = split_samples_stratified(samples, test_ratio=0.0, seed=0)
    h = train_joint_bc_critic_static(
        model,
        train_s,
        mr[:5],
        pr[:5],
        device,
        epochs=1,
        bc_batch=2,
        critic_batch=2,
        lr_actor=1e-4,
        lr_critic=1e-4,
        w_bc=1.0,
        w_match=0.1,
        w_play=0.1,
        match_flat_dim=base.match_flat_dim,
        pass_row=base._max_match_combos,
        public_slots=MAX_PUBLIC_SLOTS,
        play_loss_weight=2.0,
        seed=0,
    )
    assert len(h["loss_total"]) == 1
    assert np.isfinite(h["loss_total"][0])
