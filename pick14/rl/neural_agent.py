"""
Neural seat wired to :class:`~pick14.rl.rlmd_model.RLmdPPOAgent` with legal masked match/play (rl.md §3).

Use with :class:`~pick14.rl.rlmd_obs.RlmdObservationWrapper` on the same underlying
:class:`~pick14.rl.env.Pick14GymEnv` you step in training, or pass the bare env (obs built via
:func:`~pick14.rl.rlmd_obs.merge_rlmd_into_obs`).
"""

from __future__ import annotations

import torch

from pick14.rl.env import Pick14GymEnv
from pick14.rl.rlmd_model import RLmdPPOAgent
from pick14.rl.rlmd_obs import merge_rlmd_into_obs, numpy_legal_action
from pick14.rl.sim_core import Move, RlPick14State


class RlmdNeuralSeatAgent:
    """
    ``act(state) -> Move`` for :class:`~pick14.rl.env.Pick14GymEnv` autoplay / multi-seat setups.

    Sampling respects ``mask_match`` / ``play_hand_valid`` (deck-exhausted play uses whatever slots
    the env marks legal after Draw1 / pass paths).
    """

    def __init__(
        self,
        gym_env: Pick14GymEnv,
        model: RLmdPPOAgent,
        device: torch.device | None = None,
        *,
        deterministic: bool = False,
    ):
        root = gym_env.unwrapped
        if not isinstance(root, Pick14GymEnv):
            raise TypeError("gym_env.unwrapped must be Pick14GymEnv")
        self._env = root
        self.model = model
        self.device = device or torch.device("cpu")
        self.deterministic = deterministic

    def act(self, state: RlPick14State) -> Move:
        if self._env.state is not state:
            raise RuntimeError("RlmdNeuralSeatAgent.act: state must be the wrapped Pick14GymEnv.state")
        obs = merge_rlmd_into_obs(dict(self._env._obs()), self._env)
        idx = numpy_legal_action(
            self.model,
            self.device,
            obs,
            deterministic=self.deterministic,
        )
        return self._env.flat_action_to_move(idx)
