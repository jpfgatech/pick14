"""
Stress: mixed §1.5 seat bots (``rl.md``) on ``Pick14GymEnv`` — no fixed RNG seed.

Each game picks random table size, random factory per seat, random learning seat, random env seed.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from pick14.rl.agents import SEAT_FACTORY_CHOICES
from pick14.rl.env import Pick14GymEnv
from pick14.rl.sim_core import TurnPhase


def test_randomized_tables_run_to_terminal_without_errors():
    """Many random tables: only legal actions, env must reach ``terminated`` without exceptions."""
    rng = random.Random()
    choices = SEAT_FACTORY_CHOICES()
    max_steps = 20_000

    for _ in range(60):
        n = rng.randint(2, 6)
        agents = [choices[rng.randrange(len(choices))]() for _ in range(n)]
        lp = rng.randrange(n)
        env = Pick14GymEnv(
            agents,
            n_hand=3,
            learning_player=lp,
            seed=rng.randrange(1 << 30),
        )
        obs, info = env.reset()
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated) and steps < max_steps:
            ph = int(round(float(obs["meta"][2])))
            if ph == int(TurnPhase.PLAY):
                idx = np.flatnonzero(info["play_mask"])
                assert idx.size > 0, "PLAY with empty play_mask"
                action = env.match_flat_dim + int(rng.choice(idx))
            else:
                flat = np.flatnonzero(info["legal_mask"].reshape(-1))
                assert flat.size > 0, "non-PLAY with empty legal_mask"
                action = int(rng.choice(flat))

            obs, _reward, terminated, truncated, info = env.step(action)
            steps += 1

        assert terminated, f"expected natural termination within {max_steps} steps"
