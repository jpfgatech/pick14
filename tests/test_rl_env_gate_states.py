"""
Gate tests for the revised ``Pick14GymEnv``: injected ``RlPick14State`` plus ``reset(options=…)``.

Turn flow follows ``rl.md`` §1.1 as four phases in ``pick14.rl.sim_core.TurnPhase``: **match** → **draw1**
(after a score, L8) → **play** (L9, same discard protocol after draw1 or after pass) → **draw2** (L10–11,
only after pass-match play). Draw1/draw2 are applied inside the sim when their conditions hold; Gym
``step`` normally observes **match** (``meta[2]==0``) or **play** (``meta[2]==2``).

Each case maps to ``rl.md`` §1 (repo root). The env exposes a **full** observation dict; where §1.2
calls for agent-visible hand + pool only, tests use ``ego_tensors_from_complete_obs`` as the parser
slice. Partial injects may omit cards; unassigned canonical indices appear as zone/slot ``-1`` in obs.
"""

from __future__ import annotations

from random import Random
from typing import Any

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from pick14.cards import Card, Rank, Suit, game_value, score_value
from pick14.rl.agents import table_all_greedy_stingy
from pick14.rl.env import Pick14GymEnv
from pick14.rl.sim_core import MAX_PUBLIC_SLOTS, RlPick14State, TurnPhase, ego_tensors_from_complete_obs


def _T(num_seats: int, **kw: Any) -> Pick14GymEnv:
    """Build env with one greedy–stingy agent per seat (§1.5); kwargs: ``learning_player``, ``autoplay_non_control``, ``n_hand``, ``seed``."""
    lp = kw.pop("learning_player", kw.pop("agent_player", 0))
    anc = kw.pop("autoplay_non_control", kw.pop("autoplay_opponents", True))
    return Pick14GymEnv(
        table_all_greedy_stingy(num_seats),
        learning_player=lp,
        autoplay_non_control=anc,
        **kw,
    )


def _std(rank: Rank, suit: Suit) -> Card:
    """Non-joker card for injected states."""
    return Card(False, rank=rank, suit=suit)


def _joker_red() -> Card:
    """Red joker for §1.3 encoding checks."""
    return Card(True, joker_red=True)


def _inject_env(
    env: Pick14GymEnv,
    st: RlPick14State,
    *,
    bench_log: bool = False,
) -> tuple[dict, dict]:
    """``reset`` with ``inject_state`` + ``inject_skip_advance`` so autoplay does not run before the first ``step``."""
    opts: dict = {"inject_state": st, "inject_skip_advance": True}
    if bench_log:
        opts["bench_log"] = True
    return env.reset(options=opts)


def _match_action(combo_id: int, pool_col: int) -> int:
    """Flat match-phase index: subset row ``combo_id`` × public column ``pool_col`` (see ``decode_match_phase_flat``)."""
    return combo_id * MAX_PUBLIC_SLOTS + pool_col


def _pass_match_flat(env: Pick14GymEnv) -> int:
    """Dedicated pass-match action (rl.md §1.1 L7): row ``_max_match_combos``, column ``0``."""
    return env._max_match_combos * MAX_PUBLIC_SLOTS


def _two_player_state(
    hand0: list[Card],
    hand1: list[Card],
    public: list[Card],
    deck: list[Card],
    *,
    current_player: int = 0,
    phase: TurnPhase = TurnPhase.MATCH,
    passed_match_this_turn: bool = False,
    score_piles: list[list[Card]] | None = None,
    n_hand: int = 3,
    rng: Random | None = None,
) -> RlPick14State:
    """Minimal two-player ``RlPick14State`` for injection (may omit cards not under test; obs fills unassigned as -1)."""
    return RlPick14State(
        n_hand=n_hand,
        hands=[list(hand0), list(hand1)],
        score_piles=score_piles if score_piles is not None else [[], []],
        public=list(public),
        deck=list(deck),
        current_player=current_player,
        phase=phase,
        passed_match_this_turn=passed_match_this_turn,
        rng=rng or Random(0),
    )


class TestRlMd11GameFlow:
    """rl.md §1.1 Game Flow (lines 5–10)."""

    def test_l6_l7_legal_match_requires_sum_14(self):
        """§1.1 L6–L7: match legality requires digit sum 14; ``legal_mask`` admits K♥ + A♣ on pool."""
        k_h = _std(Rank.KING, Suit.HEART)
        a_c = _std(Rank.ACE, Suit.CLUB)
        filler = [_std(Rank.THREE, Suit.DIAMOND)] * 4
        st = _two_player_state(
            [k_h],
            [_std(Rank.TWO, Suit.SPADE)] * 3,
            [a_c],
            filler,
        )
        env = _T(2)
        obs, info = _inject_env(env, st)
        assert float(obs["meta"][2]) == 0.0
        assert info["legal_mask"].sum() >= 1
        assert game_value(k_h) + game_value(a_c) == 14

    def test_l7_pass_match_is_play_single_card_to_pool(self):
        """§1.1 L7 then L8–L10: pass match is its own action; play sub-phase is a second ``step`` (discard + L10 draw)."""
        k_pub = _std(Rank.KING, Suit.CLUB)
        hand = [_std(Rank.TWO, Suit.SPADE), _std(Rank.THREE, Suit.SPADE), _std(Rank.FOUR, Suit.SPADE)]
        assert not any(game_value(k_pub) + game_value(c) == 14 for c in hand)
        deck = [_std(Rank.FIVE, Suit.HEART), _std(Rank.SIX, Suit.HEART)]
        st = _two_player_state(hand, [_std(Rank.SEVEN, Suit.DIAMOND)] * 3, [k_pub], deck)
        env = _T(2, autoplay_non_control=False)
        _, info = _inject_env(env, st)
        pflat = _pass_match_flat(env)
        assert info["legal_mask"][env._max_match_combos, 0] == 1
        obs1, _, _, _, info1 = env.step(pflat)
        assert env.state is not None
        assert env.state.phase == TurnPhase.PLAY
        assert env.state.passed_match_this_turn is True
        assert float(obs1["meta"][2]) == float(TurnPhase.PLAY)
        assert info1["play_mask"].sum() >= 1
        a_discard = env.match_flat_dim + 0
        assert int(info1["play_mask"][0]) == 1
        env.step(a_discard)
        assert env.state is not None
        assert len(env.state.hands[0]) == 3

    def test_l8_l9_after_match_draw_to_four_then_forced_play(self):
        """§1.1 L8–L9: Draw1 (to four cards) runs after match; then ``TurnPhase.PLAY`` with same play protocol as pass path."""
        k_h = _std(Rank.KING, Suit.HEART)
        a_c = _std(Rank.ACE, Suit.CLUB)
        draw = [
            _std(Rank.TWO, Suit.CLUB),
            _std(Rank.THREE, Suit.CLUB),
            _std(Rank.FOUR, Suit.CLUB),
            _std(Rank.FIVE, Suit.CLUB),
        ]
        st = _two_player_state([k_h], [_std(Rank.SIX, Suit.SPADE)] * 3, [a_c], draw)
        env = _T(2, autoplay_non_control=False)
        obs, info = _inject_env(env, st)
        from pick14.rl.sim_core import hand_nonempty_subsets

        subs = hand_nonempty_subsets(1)
        assert subs == [(0,)]
        a_match = _match_action(0, 0)
        assert info["legal_mask"][0, 0] == 1
        obs2, _, _, _, info2 = env.step(a_match)
        assert env.state is not None
        assert len(env.state.hands[0]) == 4
        assert env.state.phase == TurnPhase.PLAY
        assert env.state.passed_match_this_turn is False
        assert float(obs2["meta"][2]) == float(TurnPhase.PLAY)
        assert info2["play_mask"].sum() >= 1

    def test_l10_after_pass_refill_to_n_hand_when_deck_not_empty(self):
        """§1.1 L10: after pass-play to pool, if hand below ``n_hand`` and deck non-empty, refill to ``n_hand``."""
        k_pub = _std(Rank.KING, Suit.CLUB)
        hand = [_std(Rank.TWO, Suit.SPADE), _std(Rank.THREE, Suit.SPADE), _std(Rank.FOUR, Suit.SPADE)]
        deck = [_std(Rank.NINE, Suit.HEART)]
        st = _two_player_state(hand, [_std(Rank.EIGHT, Suit.DIAMOND)] * 3, [k_pub], deck)
        env = _T(2, autoplay_non_control=False)
        _inject_env(env, st)
        env.step(_pass_match_flat(env))
        assert env.state is not None
        assert env.state.phase == TurnPhase.PLAY
        assert env.state.passed_match_this_turn is True
        env.step(env.match_flat_dim + 0)
        assert env.state is not None
        assert len(env.state.hands[0]) == 3

    def test_l8_empty_deck_no_forced_play_after_match_advances_turn(self):
        """§1.1 L8: if deck empty after match, no draw to four / no forced play; turn advances (L5)."""
        k_h = _std(Rank.KING, Suit.HEART)
        a_c = _std(Rank.ACE, Suit.CLUB)
        st = _two_player_state(
            [k_h],
            [_std(Rank.TWO, Suit.SPADE)] * 3,
            [a_c],
            [],
        )
        env = _T(2, autoplay_non_control=False)
        _, info = _inject_env(env, st)
        env.step(_match_action(0, 0))
        assert env.state is not None
        assert env.state.phase == TurnPhase.MATCH
        assert env.state.current_player == 1


class TestRlMd12VisibleObs:
    """rl.md §1.2 Visible state (lines 14–15)."""

    def test_l14_l15_obs_contains_agent_hand_and_public_not_opponent_hand(self):
        """§1.2 L14–L15: agent-visible slice (ego parser) has self hand + public pool, not opponent hand rows."""
        mine = [_std(Rank.ACE, Suit.HEART)]
        theirs = [_std(Rank.KING, Suit.SPADE)]
        pub = [_std(Rank.TWO, Suit.CLUB)]
        st = _two_player_state(mine, theirs, pub, [_std(Rank.THREE, Suit.DIAMOND)])
        env = _T(2)
        obs, _ = _inject_env(env, st)
        ego = ego_tensors_from_complete_obs(obs, 0, 3)
        assert ego["hand_mask"][0] == 1
        assert float(ego["hand"][0, 0]) == float(game_value(mine[0]))
        assert float(ego["hand"][0, 1]) == float(score_value(mine[0]))
        assert ego["public_mask"][0] == 1
        assert float(ego["public"][0, 0]) == float(game_value(pub[0]))
        assert not np.any(np.isclose(ego["hand"][:, 0], float(game_value(theirs[0]))))


class TestRlMd13CardRules:
    """rl.md §1.3 Card rules (lines 20–22)."""

    def test_l21_l22_joker_digit_and_point_encoding_in_obs(self):
        """§1.3 L21–L22: joker digit and point both 5 in the ego hand features derived from full obs."""
        j = _joker_red()
        assert game_value(j) == 5 and score_value(j) == 5
        st = _two_player_state([j], [_std(Rank.TWO, Suit.CLUB)] * 3, [_std(Rank.NINE, Suit.HEART)], [])
        env = _T(2)
        obs, _ = _inject_env(env, st)
        ego = ego_tensors_from_complete_obs(obs, 0, 3)
        assert float(ego["hand"][0, 0]) == 5.0
        assert float(ego["hand"][0, 1]) == 5.0


class TestRlMd14Winning:
    """rl.md §1.4 Winning (lines 24–26)."""

    def test_l25_score_pile_sum_in_obs_and_info(self):
        """§1.4 L25: player score = sum of score-value over cards in that pile; reflected in ``obs['scores']`` and ``info``."""
        pile0 = [_std(Rank.ACE, Suit.HEART), _std(Rank.TWO, Suit.HEART)]
        exp0 = score_value(pile0[0]) + score_value(pile0[1])
        st = _two_player_state(
            [],
            [],
            [_std(Rank.THREE, Suit.CLUB)],
            [],
            score_piles=[pile0, []],
            current_player=0,
        )
        st.hands[1] = [_std(Rank.FOUR, Suit.SPADE)]
        env = _T(2)
        obs, info = _inject_env(env, st)
        assert float(obs["scores"][0]) == float(exp0)
        assert info["scores"][0] == exp0


class TestRlMd15Dummy:
    """rl.md §1.5 Dummy strategy (lines 28–35)."""

    def test_l30_l31_prefers_match_using_more_hand_cards_when_points_tie(self):
        """§1.5 L30–L31: greedy match tie-break — equal capture points → use more hand cards; ``teacher_action`` reflects it."""
        seven_c = _std(Rank.SEVEN, Suit.CLUB)
        seven_d = _std(Rank.SEVEN, Suit.DIAMOND)
        ace_c = _std(Rank.ACE, Suit.CLUB)
        six_c = _std(Rank.SIX, Suit.CLUB)
        assert game_value(seven_c) + game_value(seven_d) == 14
        assert game_value(seven_c) + game_value(ace_c) + game_value(six_c) == 14
        pts_1 = score_value(seven_c) + score_value(seven_d)
        pts_2 = score_value(seven_c) + score_value(ace_c) + score_value(six_c)
        assert pts_1 == pts_2
        hand = [seven_d, ace_c, six_c]
        st = _two_player_state(hand, [_std(Rank.TWO, Suit.SPADE)] * 3, [seven_c], [])
        env = _T(2, autoplay_non_control=False)
        _inject_env(env, st)
        a = env.teacher_action()
        from pick14.rl.sim_core import hand_nonempty_subsets

        subs = hand_nonempty_subsets(len(env.state.hands[0]))
        chosen_pub = a % MAX_PUBLIC_SLOTS
        cid = a // MAX_PUBLIC_SLOTS
        combo = subs[cid]
        assert chosen_pub == 0
        assert set(combo) == {1, 2}

    def test_l32_prefers_higher_public_card_points_when_prior_ties(self):
        """§1.5 L32: further tie-break — highest public card points (K♥+A over 2♣+Q)."""
        two_c = _std(Rank.TWO, Suit.CLUB)
        k_h = _std(Rank.KING, Suit.HEART)
        q_h = _std(Rank.QUEEN, Suit.HEART)
        a_c = _std(Rank.ACE, Suit.CLUB)
        st = _two_player_state(
            [q_h, a_c],
            [_std(Rank.THREE, Suit.SPADE)] * 3,
            [two_c, k_h],
            [],
        )
        env = _T(2, autoplay_non_control=False)
        _inject_env(env, st)
        a = env.teacher_action()
        assert a % MAX_PUBLIC_SLOTS == 1

    def test_l34_l35_forced_play_caution_prefers_largest_digit(self):
        """§1.5 baseline caution play — discard with largest game digit first (5 > 4 > 3 > 2 here)."""
        low_same_pt = _std(Rank.TWO, Suit.CLUB)
        high_digit_same_pt = _std(Rank.THREE, Suit.CLUB)
        assert score_value(low_same_pt) == score_value(high_digit_same_pt)
        assert game_value(high_digit_same_pt) > game_value(low_same_pt)
        extra = [_std(Rank.FOUR, Suit.HEART), _std(Rank.FIVE, Suit.HEART)]
        st = _two_player_state(
            [low_same_pt, high_digit_same_pt, *extra],
            [_std(Rank.SIX, Suit.SPADE)] * 3,
            [_std(Rank.EIGHT, Suit.DIAMOND)],
            [],
            phase=TurnPhase.PLAY,
            passed_match_this_turn=False,
        )
        env = _T(2, autoplay_non_control=False)
        _inject_env(env, st)
        a = env.teacher_action()
        hi = a - env.match_flat_dim
        assert hi == 3


class TestRlMd16Bench:
    """rl.md §1.6 Game Bench (lines 37–40)."""

    def test_l39_multiplayer_inject_and_env_player_count(self):
        """§1.6 L39: environment supports more than two seats; inject and obs report three players."""
        hands = [
            [_std(Rank.ACE, Suit.HEART)],
            [_std(Rank.TWO, Suit.CLUB)],
            [_std(Rank.THREE, Suit.DIAMOND)],
        ]
        st = RlPick14State(
            n_hand=3,
            hands=hands,
            score_piles=[[], [], []],
            public=[_std(Rank.FOUR, Suit.SPADE)],
            deck=[_std(Rank.FIVE, Suit.HEART)],
            current_player=0,
            phase=TurnPhase.MATCH,
            passed_match_this_turn=False,
            rng=Random(0),
        )
        env = _T(3)
        obs, info = _inject_env(env, st)
        assert int(obs["num_players"][0]) == 3
        assert len(info["scores"]) == 3

    def test_l40_bench_snapshot_includes_ordered_deck_and_all_hands(self):
        """§1.6 L40: ``info['bench']`` records full snapshot including ordered deck (draw order) and all hands."""
        d0, d1 = _std(Rank.NINE, Suit.HEART), _std(Rank.TEN, Suit.HEART)
        st = _two_player_state(
            [_std(Rank.JACK, Suit.CLUB)],
            [_std(Rank.QUEEN, Suit.CLUB)],
            [_std(Rank.KING, Suit.DIAMOND)],
            [d0, d1],
        )
        env = _T(2)
        _, info = _inject_env(env, st, bench_log=True)
        bench = info["bench"]
        assert len(bench["deck"]) == 2
        assert bench["deck"][-1]["rank"] == d1.rank.name
        assert bench["deck"][-2]["rank"] == d0.rank.name
        assert len(bench["hands"]) == 2


class TestCornerCases:
    """Gym semantics and guards; turn order aligns with rl.md L5 (alternating turns)."""

    def test_step_when_not_agent_turn_returns_negative_reward(self):
        """§1.1 L5: alternating turns — ``learning_player`` alone may ``step``; off-turn action gets negative reward."""
        st = _two_player_state(
            [_std(Rank.ACE, Suit.HEART)] * 3,
            [],
            [_std(Rank.TWO, Suit.CLUB)],
            [],
            current_player=1,
        )
        st.hands[1] = [_std(Rank.THREE, Suit.SPADE)] * 2
        env = _T(2, autoplay_non_control=False)
        _inject_env(env, st)
        obs, r, _, _, _ = env.step(0)
        assert r == -1.0
        assert env.state is not None
        assert env.state.current_player == 1

    def test_illegal_match_action_penalised(self):
        """§1.1 L6–L7: illegal match (not summing to 14) rejected with penalty; state unchanged for control seat."""
        st = _two_player_state(
            [_std(Rank.TWO, Suit.HEART)],
            [_std(Rank.THREE, Suit.SPADE)] * 3,
            [_std(Rank.FOUR, Suit.CLUB)],
            [],
        )
        env = _T(2, autoplay_non_control=False)
        _inject_env(env, st)
        obs, r, term, trunc, _ = env.step(0)
        assert r == -1.0
        assert term is False and trunc is False
        assert env.state is not None
        assert env.state.current_player == 0

    def test_finished_game_terminated_observation_stable(self):
        """§1.4 L24–L26: terminal when no cards in hands (engine); further ``step`` terminates with penalty."""
        st = _two_player_state([], [], [], [], score_piles=[[], []], current_player=0)
        st.hands[0] = []
        st.hands[1] = []
        env = _T(2, autoplay_non_control=False)
        obs, info = _inject_env(env, st)
        assert info["play_mask"].sum() == 0
        _, r, term, _, _ = env.step(0)
        assert term is True
        assert r == -1.0

    def test_inject_state_player_count_must_match_env(self):
        """§1.6 L39: injected ``RlPick14State.num_players`` must match ``len(env.agents)``."""
        st = _two_player_state([_std(Rank.ACE, Suit.HEART)], [], [_std(Rank.TWO, Suit.CLUB)], [])
        env = _T(3)
        with pytest.raises(ValueError, match="num_players"):
            env.reset(options={"inject_state": st, "inject_skip_advance": True})

    def test_agent_seat_nonzero_with_inject(self):
        """§1.2 L14–L15: ego slice for ``learning_player=2`` shows that seat's hand, not seat 0."""
        h2 = [_std(Rank.ACE, Suit.SPADE)] * 3
        h0 = [_std(Rank.TWO, Suit.CLUB)] * 3
        h1 = [_std(Rank.THREE, Suit.HEART)] * 3
        st = RlPick14State(
            n_hand=3,
            hands=[h0, h1, h2],
            score_piles=[[], [], []],
            public=[_std(Rank.FOUR, Suit.DIAMOND)],
            deck=[],
            current_player=2,
            phase=TurnPhase.MATCH,
            passed_match_this_turn=False,
            rng=Random(1),
        )
        env = _T(3, learning_player=2)
        obs, _ = _inject_env(env, st)
        ego = ego_tensors_from_complete_obs(obs, 2, 3)
        assert float(ego["hand"][0, 0]) == float(game_value(h2[0]))

