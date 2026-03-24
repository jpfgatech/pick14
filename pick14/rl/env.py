from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pick14.agent import choose_dummy_action, greedy_match_move
from pick14.engine import GameState, MatchMove, PlayMove, apply_move, is_finished, new_game, skip_empty_hands
from pick14.rl.encoding import MAX_HAND_COMBOS, MAX_PUBLIC, HandCombo, build_hand_combos, build_public_vectors


def _run_bots_until_human_or_done(state: GameState) -> None:
    for _ in range(1024):
        skip_empty_hands(state)
        if is_finished(state) or state.current_player == 0:
            return
        apply_move(state, choose_dummy_action(state))


@dataclass(slots=True)
class EncodedState:
    obs: dict[str, np.ndarray]
    hand_combos: list[HandCombo]
    public_count: int


class Pick14GymEnv(gym.Env):
    """Single-agent seat-0 env with masked set-action head (rl_init.md style)."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, num_players: int = 3, n_hand: int = 3, seed: int | None = None):
        super().__init__()
        self.num_players = num_players
        self.n_hand = n_hand
        self.base_seed = seed
        self._rng = Random(seed)
        self.state: GameState | None = None
        self._last_encoded: EncodedState | None = None

        # action = flatten(hand_combo_idx, key_idx) where key_idx in [public slots..., pass]
        self.max_keys = MAX_PUBLIC + 1
        self.action_space = spaces.Discrete(MAX_HAND_COMBOS * self.max_keys)
        self.observation_space = spaces.Dict(
            {
                "hand_vecs": spaces.Box(-1e6, 1e6, shape=(MAX_HAND_COMBOS, 9), dtype=np.float32),
                "hand_valid": spaces.MultiBinary(MAX_HAND_COMBOS),
                "public_vecs": spaces.Box(-1e6, 1e6, shape=(MAX_PUBLIC, 9), dtype=np.float32),
                "public_valid": spaces.MultiBinary(MAX_PUBLIC),
                "mask": spaces.MultiBinary((MAX_HAND_COMBOS, self.max_keys)),
                "meta": spaces.Box(-1e6, 1e6, shape=(6,), dtype=np.float32),
            }
        )

    def _encode(self) -> EncodedState:
        assert self.state is not None
        st = self.state
        hand = st.hands[0]
        hand_combos = build_hand_combos(hand)
        hand_vecs = np.zeros((MAX_HAND_COMBOS, 9), dtype=np.float32)
        hand_valid = np.zeros((MAX_HAND_COMBOS,), dtype=np.int8)
        for i, combo in enumerate(hand_combos):
            hand_vecs[i] = combo.vec9
            hand_valid[i] = 1

        public_vecs, public_valid_bool = build_public_vectors(st.public)
        public_valid = public_valid_bool.astype(np.int8)
        public_count = int(public_valid_bool.sum())

        mask = np.zeros((MAX_HAND_COMBOS, self.max_keys), dtype=np.int8)
        if st.current_player == 0 and not is_finished(st):
            for i, combo in enumerate(hand_combos):
                # Pass is always allowed in the network definition;
                # env maps it to a legal play card from selected combo.
                mask[i, public_count] = 1
                if st.must_play_only:
                    continue
                combo_sum = float(combo.vec9[0])
                for j in range(public_count):
                    pub_sum = float(public_vecs[j, 0])
                    if combo_sum + pub_sum == 14.0:
                        mask[i, j] = 1

        meta = np.array(
            [
                float(st.current_player),
                float(len(st.deck)),
                float(st.must_play_only),
                float(sum(c for c in hand_valid)),
                float(public_count),
                float(sum(len(p) for p in st.score_piles)),
            ],
            dtype=np.float32,
        )
        obs = {
            "hand_vecs": hand_vecs,
            "hand_valid": hand_valid,
            "public_vecs": public_vecs,
            "public_valid": public_valid,
            "mask": mask,
            "meta": meta,
        }
        return EncodedState(obs=obs, hand_combos=hand_combos, public_count=public_count)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = Random(seed)
        elif self.base_seed is not None:
            # keep deterministic sequence across resets when seeded in ctor
            self._rng = Random(self._rng.random())
        self.state = new_game(self.num_players, rng=self._rng, n_hand=self.n_hand)
        _run_bots_until_human_or_done(self.state)
        self._last_encoded = self._encode()
        return self._last_encoded.obs, {"legal_mask": self._last_encoded.obs["mask"]}

    def _decode_action(self, action: int) -> tuple[int, int]:
        q = int(action) // self.max_keys
        k = int(action) % self.max_keys
        return q, k

    def _apply_decoded(self, q: int, k: int) -> tuple[float, bool]:
        assert self.state is not None and self._last_encoded is not None
        st = self.state
        enc = self._last_encoded
        mask = enc.obs["mask"]
        if q < 0 or q >= MAX_HAND_COMBOS or k < 0 or k >= self.max_keys or mask[q, k] == 0:
            # Illegal chosen action => strong penalty, continue.
            return -1.0, False

        combo = enc.hand_combos[q]
        score_before = sum(len(p) for p in st.score_piles)
        if k == enc.public_count:
            # Pass branch: select a legal single-card play from the chosen combo.
            # If combo is multi-card, use lowest hand index.
            play_idx = min(combo.hand_indices)
            apply_move(st, PlayMove(play_idx))
        else:
            apply_move(st, MatchMove(k, combo.hand_indices))
            # If forced follow-up play is active, auto-play smallest index from remaining hand.
            if st.must_play_only and st.current_player == 0 and st.hands[0]:
                apply_move(st, PlayMove(0))

        _run_bots_until_human_or_done(st)
        score_after = sum(len(p) for p in st.score_piles)
        # Dense reward: captured-cards delta in all piles approximates points progress.
        reward = float(score_after - score_before)
        done = is_finished(st)
        return reward, done

    def step(self, action: int):
        assert self.state is not None
        q, k = self._decode_action(int(action))
        reward, done = self._apply_decoded(q, k)
        self._last_encoded = self._encode()
        obs = self._last_encoded.obs
        terminated = bool(done)
        truncated = False
        return obs, reward, terminated, truncated, {"legal_mask": obs["mask"]}

    def render(self):
        if self.state is None:
            return "No state"
        return (
            f"p={self.state.current_player} hand0={len(self.state.hands[0])} "
            f"public={len(self.state.public)} deck={len(self.state.deck)} done={is_finished(self.state)}"
        )

    def teacher_action(self) -> int:
        """Map greedy teacher move to rl_init flattened action index."""
        assert self.state is not None
        if self.state.current_player != 0 or is_finished(self.state):
            return 0
        enc = self._encode()
        self._last_encoded = enc
        teacher = greedy_match_move(self.state)
        if teacher is not None:
            hand_set = tuple(sorted(teacher.hand_indices))
            for qi, combo in enumerate(enc.hand_combos):
                if combo.hand_indices == hand_set:
                    return qi * self.max_keys + teacher.public_index
        # fallback to play: deterministic smallest hand index.
        hand = self.state.hands[0]
        best_idx = min(range(len(hand)), key=lambda i: i)
        pass_k = enc.public_count
        for qi, combo in enumerate(enc.hand_combos):
            if combo.hand_indices == (best_idx,):
                return qi * self.max_keys + pass_k
        return pass_k

