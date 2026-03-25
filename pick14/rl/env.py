from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pick14.agent import greedy_match_move, stingy_play_move
from pick14.engine import GameState, MatchMove, PlayMove, apply_move, is_finished, new_game, skip_empty_hands
from pick14.rl.encoding import (
    MAX_HAND_COMBOS,
    MAX_PLAY_HAND,
    MAX_PLAY_KEYS,
    MAX_PUBLIC,
    NUM_GLOBAL_PLAY_CONTEXT_KEYS,
    HandCombo,
    build_hand_combos,
    build_public_vectors,
    combo_vector,
)


def _run_bots_until_human_or_done(state: GameState) -> None:
    from pick14.agent import choose_dummy_action

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
    """
    Seat-0 single-agent env.
    Match phase: flat action over (hand_combo, public|pass) per rl_init.md.
    Forced play phase: flat offset action = match_flat_dim + hand_index (stingy teacher, play head).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, num_players: int = 3, n_hand: int = 3, seed: int | None = None):
        super().__init__()
        self.num_players = num_players
        self.n_hand = n_hand
        self.base_seed = seed
        self._rng = Random(seed)
        self.state: GameState | None = None
        self._last_encoded: EncodedState | None = None

        self.max_keys = MAX_PUBLIC + 1
        self.match_flat_dim = MAX_HAND_COMBOS * self.max_keys
        self.play_action_dim = MAX_PLAY_HAND
        self.action_space = spaces.Discrete(self.match_flat_dim + self.play_action_dim)

        self.observation_space = spaces.Dict(
            {
                "phase": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "hand_vecs": spaces.Box(-1e6, 1e6, shape=(MAX_HAND_COMBOS, 9), dtype=np.float32),
                "hand_valid": spaces.MultiBinary(MAX_HAND_COMBOS),
                "public_vecs": spaces.Box(-1e6, 1e6, shape=(MAX_PUBLIC, 9), dtype=np.float32),
                "public_valid": spaces.MultiBinary(MAX_PUBLIC),
                "mask": spaces.MultiBinary((MAX_HAND_COMBOS, self.max_keys)),
                "play_hand_vecs": spaces.Box(-1e6, 1e6, shape=(MAX_PLAY_HAND, 9), dtype=np.float32),
                "play_hand_valid": spaces.MultiBinary(MAX_PLAY_HAND),
                "play_key_mask": spaces.Box(
                    low=0, high=1, shape=(MAX_PLAY_HAND, MAX_PLAY_KEYS), dtype=np.int8
                ),
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
        play_hand_vecs = np.zeros((MAX_PLAY_HAND, 9), dtype=np.float32)
        play_hand_valid = np.zeros((MAX_PLAY_HAND,), dtype=np.int8)
        play_key_mask = np.zeros((MAX_PLAY_HAND, MAX_PLAY_KEYS), dtype=np.int8)
        phase = np.array([1.0 if st.must_play_only else 0.0], dtype=np.float32)

        if st.current_player == 0 and not is_finished(st):
            if st.must_play_only:
                # Degenerate match mask: one legal cell so flattened match logits stay finite (unused for supervision).
                mask[0, public_count] = 1
                n = len(hand)
                # Trailing key columns = learnable Global Play Context bank (rl_init.md).
                for hi in range(min(n, MAX_PLAY_HAND)):
                    play_hand_vecs[hi] = combo_vector([hand[hi]])
                    play_hand_valid[hi] = 1
                    for j in range(public_count):
                        play_key_mask[hi, j] = 1
                    for g in range(NUM_GLOBAL_PLAY_CONTEXT_KEYS):
                        play_key_mask[hi, MAX_PUBLIC + g] = 1
            else:
                for i, combo in enumerate(hand_combos):
                    mask[i, public_count] = 1
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
            "phase": phase,
            "hand_vecs": hand_vecs,
            "hand_valid": hand_valid,
            "public_vecs": public_vecs,
            "public_valid": public_valid,
            "mask": mask,
            "play_hand_vecs": play_hand_vecs,
            "play_hand_valid": play_hand_valid,
            "play_key_mask": play_key_mask,
            "meta": meta,
        }
        return EncodedState(obs=obs, hand_combos=hand_combos, public_count=public_count)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = Random(seed)
        elif self.base_seed is not None:
            self._rng = Random(self._rng.random())
        self.state = new_game(self.num_players, rng=self._rng, n_hand=self.n_hand)
        _run_bots_until_human_or_done(self.state)
        self._last_encoded = self._encode()
        return self._last_encoded.obs, {"legal_mask": self._last_encoded.obs["mask"]}

    def _decode_match_action(self, action: int) -> tuple[int, int]:
        q = int(action) // self.max_keys
        k = int(action) % self.max_keys
        return q, k

    def _apply_match_decoded(self, q: int, k: int) -> tuple[float, bool]:
        assert self.state is not None and self._last_encoded is not None
        st = self.state
        enc = self._last_encoded
        mask = enc.obs["mask"]
        if q < 0 or q >= MAX_HAND_COMBOS or k < 0 or k >= self.max_keys or mask[q, k] == 0:
            return -1.0, False

        combo = enc.hand_combos[q]
        score_before = sum(len(p) for p in st.score_piles)
        if k == enc.public_count:
            play_idx = min(combo.hand_indices)
            apply_move(st, PlayMove(play_idx))
        else:
            apply_move(st, MatchMove(k, combo.hand_indices))

        _run_bots_until_human_or_done(st)
        score_after = sum(len(p) for p in st.score_piles)
        reward = float(score_after - score_before)
        done = is_finished(st)
        return reward, done

    def _apply_play_decoded(self, hand_index: int) -> tuple[float, bool]:
        assert self.state is not None and self._last_encoded is not None
        st = self.state
        enc = self._last_encoded
        if not st.must_play_only:
            return -1.0, False
        hand = st.hands[0]
        if hand_index < 0 or hand_index >= len(hand) or hand_index >= MAX_PLAY_HAND:
            return -1.0, False
        if enc.obs["play_hand_valid"][hand_index] != 1:
            return -1.0, False

        score_before = sum(len(p) for p in st.score_piles)
        apply_move(st, PlayMove(hand_index))
        _run_bots_until_human_or_done(st)
        score_after = sum(len(p) for p in st.score_piles)
        reward = float(score_after - score_before)
        done = is_finished(st)
        return reward, done

    def step(self, action: int):
        assert self.state is not None
        self._last_encoded = self._encode()
        enc = self._last_encoded
        obs0 = enc.obs
        a = int(action)
        play_phase = float(obs0["phase"][0]) >= 0.5
        if play_phase:
            if a < self.match_flat_dim:
                return obs0, -1.0, False, False, {"legal_mask": obs0["mask"]}
            reward, done = self._apply_play_decoded(a - self.match_flat_dim)
        else:
            if a >= self.match_flat_dim:
                return obs0, -1.0, False, False, {"legal_mask": obs0["mask"]}
            reward, done = self._apply_match_decoded(*self._decode_match_action(a))
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
        assert self.state is not None
        if self.state.current_player != 0 or is_finished(self.state):
            return 0
        enc = self._encode()
        self._last_encoded = enc
        st = self.state

        if st.must_play_only:
            pm = stingy_play_move(st)
            return self.match_flat_dim + int(pm.hand_index)

        teacher = greedy_match_move(st)
        if teacher is not None:
            hand_set = tuple(sorted(teacher.hand_indices))
            for qi, combo in enumerate(enc.hand_combos):
                if combo.hand_indices == hand_set:
                    return qi * self.max_keys + teacher.public_index
        hand = st.hands[0]
        best_idx = min(range(len(hand)), key=lambda i: i)
        pass_k = enc.public_count
        for qi, combo in enumerate(enc.hand_combos):
            if combo.hand_indices == (best_idx,):
                return qi * self.max_keys + pass_k
        return pass_k
