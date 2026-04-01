from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pick14.agent import greedy_match_move, stingy_play_move
from pick14.engine import GameState, MatchMove, PlayMove, apply_move, is_finished, new_game, skip_empty_hands, total_score_points
from pick14.rl.encoding import (
    MAX_HAND_COMBOS,
    MAX_PLAY_HAND,
    MAX_PUBLIC,
    HandCombo,
    build_hand_combos,
    build_public_vectors,
    combo_vector,
)
from pick14.rl.rlmd_sequences import (
    AGENT_BODY_LEN,
    CRITIC_BODY_LEN,
    RLMD_POOL_SLOTS,
    encode_agent_match,
    encode_agent_play,
    encode_critic_match,
    encode_critic_play,
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
    Seat-0 agent vs choose_dummy_action bots. Default **2 players** (rl.md critic scope).

    **Phase bit in ``obs['phase']``** tracks ``GameState.must_play_only`` (see ``pick14.engine``), not the
    informal “whole turn” from the rules text:

    - **Match-phase** (``phase < 0.5``): ``must_play_only`` is False. The agent’s legal *gym* actions are
      the match head (including **pass**, implemented as a ``PlayMove`` on the pass column). Most seat-0
      timesteps look like this: pass-or-match decisions, or any turn that never entered a forced discard.
    - **Play-phase** (``phase >= 0.5``): ``must_play_only`` is True. This happens **only** after the agent
      **scores** a 14-sum match **while the deck is not exhausted**; the engine then draws them to
      ``n_hand+1`` and requires one discard. If the deck is already exhausted when the match resolves,
      the engine advances the turn **without** a play-phase step.

    So **BC “play” labels are much rarer than “match” labels**: they count only those forced-discard
    decisions, not every discard in the card game. That is expected under this MDP, not a dataset bug.

    Rewards (rl.md §4.1):
      match phase: r = A_t (agent score points gained this step)
      play phase:  r = -O_t (negative of opponent score points gained before agent acts again)
    """

    metadata = {"render_modes": ["ansi"]}
    OPP_IDX = 1

    def __init__(self, num_players: int = 2, n_hand: int = 3, seed: int | None = None):
        super().__init__()
        if num_players != 2:
            raise ValueError("Pick14GymEnv rl.md stack currently supports num_players=2 only")
        self.num_players = num_players
        self.n_hand = n_hand
        self.base_seed = seed
        self._rng = Random(seed)
        self.state: GameState | None = None
        self._last_encoded: EncodedState | None = None
        self._post_play_obs: dict[str, np.ndarray] | None = None

        self.max_keys = RLMD_POOL_SLOTS
        self.match_flat_dim = MAX_HAND_COMBOS * self.max_keys
        self.play_action_dim = MAX_PLAY_HAND
        self.action_space = spaces.Discrete(self.match_flat_dim + self.play_action_dim)

        self.observation_space = spaces.Dict(
            {
                "phase": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "seq_agent_feats": spaces.Box(-1e6, 1e6, shape=(AGENT_BODY_LEN, 9), dtype=np.float32),
                "seq_agent_roles": spaces.Box(0, 7, shape=(AGENT_BODY_LEN,), dtype=np.int64),
                "seq_agent_mask": spaces.MultiBinary(AGENT_BODY_LEN),
                "seq_critic_feats": spaces.Box(-1e6, 1e6, shape=(CRITIC_BODY_LEN, 9), dtype=np.float32),
                "seq_critic_roles": spaces.Box(0, 7, shape=(CRITIC_BODY_LEN,), dtype=np.int64),
                "seq_critic_mask": spaces.MultiBinary(CRITIC_BODY_LEN),
                "mask": spaces.MultiBinary((MAX_HAND_COMBOS, self.max_keys)),
                "mask_match": spaces.MultiBinary((MAX_HAND_COMBOS, self.max_keys)),
                "play_hand_valid": spaces.MultiBinary(MAX_PLAY_HAND),
                "meta": spaces.Box(-1e6, 1e6, shape=(4,), dtype=np.float32),
                "public_valid": spaces.MultiBinary(MAX_PUBLIC),
                "hand_valid": spaces.MultiBinary(MAX_HAND_COMBOS),
                "hand_vecs": spaces.Box(-1e6, 1e6, shape=(MAX_HAND_COMBOS, 9), dtype=np.float32),
                "public_vecs": spaces.Box(-1e6, 1e6, shape=(MAX_PUBLIC, 9), dtype=np.float32),
            }
        )

    def _encode(self) -> EncodedState:
        assert self.state is not None
        st = self.state
        hand = st.hands[0]
        hand_combos = build_hand_combos(hand)
        public_vecs, public_valid_bool = build_public_vectors(st.public)
        public_count = int(public_valid_bool.sum())

        mask = np.zeros((MAX_HAND_COMBOS, self.max_keys), dtype=np.int8)
        play_hand_valid = np.zeros((MAX_PLAY_HAND,), dtype=np.int8)
        phase_f = 1.0 if st.must_play_only else 0.0
        phase = np.array([phase_f], dtype=np.float32)

        if st.current_player == 0 and not is_finished(st):
            if st.must_play_only:
                for hi in range(min(len(hand), MAX_PLAY_HAND)):
                    play_hand_valid[hi] = 1
            else:
                for i, combo in enumerate(hand_combos):
                    mask[i, public_count] = 1
                    combo_sum = float(combo.vec9[0])
                    for j in range(public_count):
                        pub_sum = float(public_vecs[j, 0])
                        if combo_sum + pub_sum == 14.0:
                            mask[i, j] = 1

        if phase_f >= 0.5:
            af, ar, am = encode_agent_play(st)
            cf, cr, cm = encode_critic_play(st, self.OPP_IDX)
        else:
            af, ar, am = encode_agent_match(st)
            cf, cr, cm = encode_critic_match(st, self.OPP_IDX)

        gap = float(total_score_points(st, 0) - total_score_points(st, self.OPP_IDX))
        meta = np.array(
            [float(st.current_player), float(len(st.deck)), float(st.must_play_only), gap],
            dtype=np.float32,
        )

        obs = {
            "phase": phase,
            "seq_agent_feats": af,
            "seq_agent_roles": ar.astype(np.int64),
            "seq_agent_mask": am.astype(np.int8),
            "seq_critic_feats": cf,
            "seq_critic_roles": cr.astype(np.int64),
            "seq_critic_mask": cm.astype(np.int8),
            "mask": mask,
            "mask_match": mask,
            "play_hand_valid": play_hand_valid,
            "meta": meta,
            "public_valid": public_valid_bool.astype(np.int8),
            "hand_valid": am[:MAX_HAND_COMBOS].astype(np.int8),
            "hand_vecs": af[:MAX_HAND_COMBOS].copy(),
            "public_vecs": public_vecs.astype(np.float32),
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
        return self._last_encoded.obs, {"legal_mask": self._last_encoded.obs["mask_match"]}

    def _decode_match_action(self, action: int) -> tuple[int, int]:
        q = int(action) // self.max_keys
        k = int(action) % self.max_keys
        return q, k

    def _apply_match_decoded(self, q: int, k: int) -> tuple[float, bool]:
        assert self.state is not None and self._last_encoded is not None
        st = self.state
        enc = self._last_encoded
        mask = enc.obs["mask_match"]
        if q < 0 or q >= MAX_HAND_COMBOS or k < 0 or k >= self.max_keys or mask[q, k] == 0:
            return -1.0, False

        ag0 = total_score_points(st, 0)
        combo = enc.hand_combos[q]
        if k == enc.public_count:
            play_idx = min(combo.hand_indices)
            apply_move(st, PlayMove(play_idx))
        else:
            apply_move(st, MatchMove(k, combo.hand_indices))

        ag1 = total_score_points(st, 0)
        reward = float(ag1 - ag0)
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

        apply_move(st, PlayMove(hand_index))
        # rl.md §4.3: Critic-Opponent sees S_post_play before opponent responds
        snap = self._encode()
        self._post_play_obs = {k: np.asarray(v).copy() for k, v in snap.obs.items()}
        op0 = total_score_points(st, self.OPP_IDX)
        _run_bots_until_human_or_done(st)
        op1 = total_score_points(st, self.OPP_IDX)
        reward = float(-(op1 - op0))
        done = is_finished(st)
        return reward, done

    def step(self, action: int):
        assert self.state is not None
        self._post_play_obs = None
        self._last_encoded = self._encode()
        enc = self._last_encoded
        obs0 = enc.obs
        a = int(action)
        play_phase = float(obs0["phase"][0]) >= 0.5
        if play_phase:
            if a < self.match_flat_dim:
                return obs0, -1.0, False, False, {"legal_mask": obs0["mask_match"]}
            reward, done = self._apply_play_decoded(a - self.match_flat_dim)
        else:
            if a >= self.match_flat_dim:
                return obs0, -1.0, False, False, {"legal_mask": obs0["mask_match"]}
            reward, done = self._apply_match_decoded(*self._decode_match_action(a))
        self._last_encoded = self._encode()
        obs = self._last_encoded.obs
        return obs, reward, bool(done), False, {"legal_mask": obs["mask_match"]}

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
