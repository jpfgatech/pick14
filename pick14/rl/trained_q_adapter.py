"""
PickQNet + rolling :class:`~pick14.rl.q_state.GameHistory` for :mod:`pick14.rl.direct_q`.

Shared by rollout (:mod:`pick14.rl.q_targets`), evaluation scripts, and CP demos.
"""

from __future__ import annotations

import numpy as np
import torch

from pick14.rl.q_model import PickQNet
from pick14.rl.q_state import GameHistory, encode_q_state


class TrainedQAdapter:
    """PickQNet + encode_q_state(history) for ``direct_q.decide`` / batched Q."""

    def __init__(self, model: PickQNet, history: GameHistory, device: torch.device) -> None:
        self.model = model
        self.history = history
        self.device = device

    def evaluate(self, state, acting_player: int) -> float:
        return self.evaluate_batch([state], acting_player)[0]

    def evaluate_batch(self, states: list, acting_player: int) -> list[float]:
        if not states:
            return []
        mats = np.stack(
            [encode_q_state(s, agent_seat=acting_player, history=self.history) for s in states],
            axis=0,
        )
        x = torch.from_numpy(mats).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            q, _ = self.model(x)
        return [float(v) for v in q.squeeze(-1).cpu().tolist()]
