"""Smoke test: load PickQ checkpoint and finish one game vs baseline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


def test_play_one_game_smoke() -> None:
    repo = Path(__file__).resolve().parents[1]
    ck = repo / "train_runs/pickq_20260429T020300Z.pt"
    if not ck.is_file():
        pytest.skip("checkpoint not present")

    spec = importlib.util.spec_from_file_location(
        "pickq_vs_baseline",
        repo / "scripts/05_pickq_vs_baseline.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from random import Random

    from pick14.rl.q_model import PickQNet
    from pick14.rl.q_state import GameHistory

    device = torch.device("cpu")
    state_dict = torch.load(ck, map_location=device, weights_only=False)["model_state"]
    model = PickQNet()
    model.load_state_dict(state_dict)

    sq, sb = mod.play_one_game(
        model,
        GameHistory(n_seats=2),
        device,
        rng=Random(123),
        q_seat=0,
    )
    assert sq >= 0 and sb >= 0
