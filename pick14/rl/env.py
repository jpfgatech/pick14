"""
Gymnasium environment for Pick14 — rl.md §1 simulation in sim_core, seat-0 agent vs §1.5 dummies.

--------------------------------------------------------------------------------
Gymnasium primer (how this file fits in)
--------------------------------------------------------------------------------

**What Gymnasium is:** A small API standard for "environments" used in RL: each env is a
``gymnasium.Env`` subclass. Libraries (Stable-Baselines3, RLlib, custom trainers) call the
same three operations: ``reset`` → ``step`` → ``step`` → … until the episode ends.

**Core types from this package:**

- ``gymnasium.Env`` (imported as ``gym.Env``): abstract base class. Subclasses **must** define
  ``observation_space``, ``action_space``, ``reset``, and ``step``. Optional: ``render``,
  ``close``. Pick14 implements these on ``Pick14GymEnv``.

- ``gymnasium.spaces`` (imported as ``spaces``): **schemas** for observations and actions.
  They describe shape, dtype, and bounds so generic code can allocate networks or validate
  samples. They do **not** compute the next state; your ``step``/``reset`` logic does.

  - ``spaces.Discrete(n)``: a single integer in ``{0, …, n-1}`` (here: flattened match/play).
  - ``spaces.Box(low, high, shape, dtype)``: a real-valued tensor with fixed shape (here:
    card features, scores). Bounds are declared for documentation/sampling; our obs is built
    in code, not sampled from the space.
  - ``spaces.MultiBinary(n)``: length-``n`` vector of 0/1 flags (masks).
  - ``spaces.Dict({...})``: observation is a **dictionary** of arrays; keys must match what
    ``reset``/``step`` return.

**``numpy``:** Gymnasium spaces and observations are usually NumPy arrays. This env returns
observations as ``dict[str, np.ndarray]``.

**Dependency graph (conceptual):**

  ``gym.Env``  ← you subclass this
       ↑
  uses  ``spaces.*``  to declare ``observation_space`` / ``action_space``
       ↑
  ``reset`` / ``step`` return obs dicts that **should** be contained in ``observation_space``
  (same keys, shapes, dtypes).

**Not used here but common elsewhere:** ``gymnasium.register`` + ``gym.make("Name-v0")`` for
env factories; ``wrappers`` to clip rewards or stack frames. Pick14 is constructed directly:
``Pick14GymEnv(...)``.

--------------------------------------------------------------------------------
Agent vs critic ("god mode") — FAQ
--------------------------------------------------------------------------------

Gymnasium exposes **one** ``observation_space``. Here ``observation`` is the **full** table state:
per-card digit/point plus ``card_zone`` / ``card_slot`` for all 54 cards (deck order, public, each
hand, each score pile), plus ``scores``, ``num_players``, ``learning_player``, and ``meta``. A
parser or ``gym.Wrapper`` slices this dict for actor, critic (rl.md §2–3), or future memory (e.g.
used-card history when added). Helpers: ``pick14.rl.sim_core.ego_tensors_from_complete_obs`` for a
legacy hand/public layout. ``info["bench"]`` remains a JSON debug snapshot.

--------------------------------------------------------------------------------
Why actions are one flat ``Discrete`` — FAQ
--------------------------------------------------------------------------------

Match sub-phase, play-after-pass, and play-after-match use **different** action meanings, but Gymnasium
needs a **fixed** ``action_space``. Indices in ``[0, match_flat_dim)`` cover real matches plus a
dedicated **pass-match** slot (rl.md §1.1 L7). ``meta[2]`` is ``float(``:class:`~pick14.rl.sim_core.TurnPhase` ``)``:
``0`` match, ``1`` draw1, ``2`` play, ``3`` draw2 (draw1/draw2 usually resolve inside the same ``step`` as
match or play). Use ``meta[2] == 2`` for the play head. Alternatives: factored heads + action adapter.

**Phase-specific heads (rl.md §3.1–3.2):** The network can use **narrower** outputs (e.g. match
``H×P`` logits + separate play logits over hand slots). An **action adapter** then maps the sampled
structured choice to the single flat ``Discrete`` index this env expects (inverting the same
encoding used inside ``step``) before calling ``env.step``.

--------------------------------------------------------------------------------
``render`` — FAQ
--------------------------------------------------------------------------------

``render()`` returns a **short debug string**, not the observation dict and not a full serialized
state. For full information use ``info["bench"]`` after ``reset``/``step``, or ``bench_snapshot``.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pick14.rl.agents import (
    CautionPlayPolicy,
    CompositeSeatAgent,
    GreedyForPublicMatchPolicy,
)
from pick14.rl.sim_core import (
    MAX_PLAYERS_OBS,
    MAX_PUBLIC_SLOTS,
    MatchMove,
    Move,
    N_CANONICAL_CARDS,
    PassMatch,
    PlayMove,
    RlPick14State,
    TurnPhase,
    apply_move,
    bench_snapshot,
    build_match_legality_mask,
    build_play_legality_mask,
    clone_state,
    decode_match_phase_flat,
    encode_complete_observation,
    hand_nonempty_subsets,
    is_finished,
    max_hand_slots,
    max_match_combo_slots,
    new_game,
    skip_empty_hands,
    total_score_points,
)

# Expert labelling for ``teacher_action`` (rl.md §1.5 baseline: greedy-for-public + caution).
_EXPERT_SEAT = CompositeSeatAgent(GreedyForPublicMatchPolicy(), CautionPlayPolicy())


def _run_autoplay_until_learning(
    state: RlPick14State,
    learning_player: int,
    bench_log: list[dict[str, Any]] | None,
    agents: list[Any],
) -> None:
    """Non-control seats move via ``agents[seat].act`` until ``learning_player`` acts or done."""
    for _ in range(4096):
        skip_empty_hands(state)
        if is_finished(state) or state.current_player == learning_player:
            return
        if bench_log is not None:
            bench_log.append(bench_snapshot(clone_state(state)))
        apply_move(state, agents[state.current_player].act(state))


@dataclass(slots=True)
class _StepContext:
    """Per-step legality masks for decoding Gym actions (not part of the public Gym API)."""

    match_mask: np.ndarray
    play_mask: np.ndarray
    hand_subs: list[tuple[int, ...]]


class Pick14GymEnv(gym.Env):
    """
    Pick14 as a Gymnasium environment (``gymnasium.Env``).

    **Gymnasium contract**

    - ``observation_space`` / ``action_space``: fixed for the lifetime of the env instance;
      trainers use them to build policies. Declared in ``__init__`` via ``spaces.*``.
    - ``reset()`` → ``(observation, info)``: start (or restart) an episode. ``observation`` must
      match ``observation_space`` (here: a dict of NumPy arrays). ``info`` is a plain dict for
      extras (legal masks, bench); the Gym API does not type-check it.
    - ``step(action)`` → ``(observation, reward, terminated, truncated, info)``: apply one
      agent action, advance the world (including opponent autoplay if enabled), return the next
      observation. ``terminated`` means the episode ended by **env rules** (here: game over).
      ``truncated`` means an external time limit (we always return ``False``; wrappers often set
      this). ``reward`` is a scalar float.

    **Pick14-specific**

    - **``agents``:** one **seat agent** per player; each must implement ``act(state) -> Move``.
      Compose match + play policies via ``pick14.rl.agents.CompositeSeatAgent`` (see factories like
      ``table_all_baseline`` / ``table_all_greedy_stingy``). Mix seat types (dummy vs trainable) by passing different agent
      objects per seat.
    - **``learning_player``:** seat index that receives ``step(action)``; other seats auto-play via
      their ``agents[seat]`` when ``autoplay_non_control`` is true.
    - **Observation:** full 54-card placement + scores + ``meta``; slice in your parser or wrapper.
    - **``match_flat_dim``:** ``((2**n_hand - 1) + 1) * MAX_PUBLIC_SLOTS`` — match rows × pool columns, plus one
      **pass-match** row (only column 0 legal) for rl.md §1.1 L7 before the play sub-phase.
    - **Bench** (rl.md §1.6): optional ``reset(..., options={'bench_log': True})``.
    - **Injected states** (tests): ``options={'inject_state': ..., 'inject_skip_advance': True}``.
    - **``autoplay_non_control``:** if ``False``, no auto-play after ``step`` (tests / manual multi-agent).

    **Action encoding** (single ``Discrete``; see ``action_space`` in ``__init__``)

    - Indices ``[0, match_flat_dim)``: **match sub-phase** — subset row and public column for a true match,
      or the dedicated **pass-match** flat index (end play on match sub-phase; next ``step`` uses play head).
    - Indices ``[match_flat_dim, …)``: **PLAY** phase (L9) — discard one card; same encoding after Draw1 or after pass-match (``meta[2] == 2``).

    **Rewards:** sparse (``0.0`` on success, ``-1.0`` on illegal/off-turn); use ``info`` for scores/bench.

    **``metadata``:** optional dict read by some libraries; ``render_modes`` lists what ``render`` supports.
    """

    # Used by Gymnasium / third-party code that checks supported render modes (we only implement "ansi"-style text).
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        agents: list[Any],
        n_hand: int = 3,
        seed: int | None = None,
        learning_player: int = 0,
        autoplay_non_control: bool = True,
    ):
        """
        Parameters
        ----------
        agents
            Length ≥ 2; entry ``p`` moves when ``current_player == p`` during autoplay.
        learning_player
            Seat that receives ``step(action)`` from Gym.
        autoplay_non_control
            After a legal control move, run other seats' ``.act`` until control's turn or done.
        """
        super().__init__()
        if len(agents) < 2:
            raise ValueError("agents must have length >= 2")
        if not (0 <= learning_player < len(agents)):
            raise ValueError("learning_player out of range")
        for i, ag in enumerate(agents):
            if not callable(getattr(ag, "act", None)):
                raise TypeError(f"agents[{i}] must have act(RlPick14State) -> Move")
        self.agents = list(agents)
        self.num_players = len(agents)
        self.n_hand = n_hand
        self.learning_player = learning_player
        self.autoplay_non_control = autoplay_non_control
        self.base_seed = seed
        self._rng = Random(seed)
        self.state: RlPick14State | None = None
        self._bench_log: list[dict[str, Any]] | None = None
        self._last_ctx: _StepContext | None = None
        #: Set only immediately after a legal learning-seat PLAY, before opponent autoplay (rl.md §3.3 / §4.3).
        self._post_play_obs: dict[str, Any] | None = None

        self._max_match_combos = max_match_combo_slots(n_hand)
        self._hand_slots = max_hand_slots(n_hand)
        self.match_flat_dim = (self._max_match_combos + 1) * MAX_PUBLIC_SLOTS
        self.action_space = spaces.Discrete(self.match_flat_dim + self._hand_slots)

        zm = 2.0 + float(MAX_PLAYERS_OBS) + float(MAX_PLAYERS_OBS)
        self.observation_space = spaces.Dict(
            {
                "card_digit": spaces.Box(0.0, 20.0, shape=(N_CANONICAL_CARDS,), dtype=np.float32),
                "card_point": spaces.Box(0.0, 20.0, shape=(N_CANONICAL_CARDS,), dtype=np.float32),
                "card_zone": spaces.Box(-1.0, zm, shape=(N_CANONICAL_CARDS,), dtype=np.float32),
                "card_slot": spaces.Box(-1.0, 60.0, shape=(N_CANONICAL_CARDS,), dtype=np.float32),
                "scores": spaces.Box(0.0, 500.0, shape=(MAX_PLAYERS_OBS,), dtype=np.float32),
                "num_players": spaces.Box(0.0, 32.0, shape=(1,), dtype=np.float32),
                "learning_player": spaces.Box(0.0, 32.0, shape=(1,), dtype=np.float32),
                "meta": spaces.Box(-1e6, 1e6, shape=(4,), dtype=np.float32),
            }
        )

    def _obs(self) -> dict[str, np.ndarray]:
        assert self.state is not None
        o = encode_complete_observation(self.state, self.learning_player)
        return {k: np.asarray(v) for k, v in o.items()}

    def _flat_action_for_move(self, st: RlPick14State, move: Move) -> int:
        if isinstance(move, PassMatch):
            return self._max_match_combos * MAX_PUBLIC_SLOTS
        if isinstance(move, PlayMove):
            return self.match_flat_dim + int(move.hand_index)
        assert isinstance(move, MatchMove)
        hand = st.hands[st.current_player]
        subs = hand_nonempty_subsets(len(hand))
        cid = subs.index(tuple(sorted(move.hand_indices)))
        return cid * MAX_PUBLIC_SLOTS + move.public_index

    def flat_action_to_move(self, action: int) -> Move:
        """Inverse of :meth:`_flat_action_for_move` for the current ``state`` and phase (for neural seats)."""
        assert self.state is not None
        st = self.state
        a = int(action)
        if st.phase == TurnPhase.PLAY:
            return PlayMove(a - self.match_flat_dim)
        is_pass, row, col = decode_match_phase_flat(self._max_match_combos, a)
        if is_pass:
            return PassMatch()
        hand = st.hands[st.current_player]
        subs = hand_nonempty_subsets(len(hand))
        return MatchMove(col, subs[row])

    def _build_ctx(self) -> _StepContext:
        """Recompute legality masks for the current simulation state."""
        assert self.state is not None
        mm, subs = build_match_legality_mask(self.state)
        pm = build_play_legality_mask(self.state)
        return _StepContext(match_mask=mm, play_mask=pm, hand_subs=subs)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Gymnasium API: start a new episode.

        Returns
        -------
        observation
            First observation after setup (same structure as ``step``).
        info
            ``legal_mask``, ``play_mask``, ``scores``, optional ``bench_log``, and ``bench`` (JSON).

        ``super().reset(seed=seed)`` updates the env's internal PRNG used by some wrappers;
        we still drive card shuffling with ``random.Random`` for the game itself.
        """
        super().reset(seed=seed)
        opts = options or {}
        self._bench_log = [] if opts.get("bench_log") else None

        if seed is not None:
            self._rng = Random(seed)
        elif self.base_seed is not None:
            self._rng = Random(self._rng.random())

        self._post_play_obs = None
        if "inject_state" in opts:
            inj = opts["inject_state"]
            if not isinstance(inj, RlPick14State):
                raise TypeError("options['inject_state'] must be an RlPick14State")
            self.state = clone_state(inj)
            if self.state.num_players != self.num_players:
                raise ValueError("inject_state.num_players must match env.num_players")
            if self.state.n_hand != self.n_hand:
                raise ValueError("inject_state.n_hand must match env.n_hand (observation_space sizing)")
            if self.autoplay_non_control and not opts.get("inject_skip_advance", False):
                _run_autoplay_until_learning(self.state, self.learning_player, self._bench_log, self.agents)
        else:
            self.state = new_game(self.num_players, rng=self._rng, n_hand=self.n_hand)
            if self.autoplay_non_control:
                _run_autoplay_until_learning(self.state, self.learning_player, self._bench_log, self.agents)

        ctx = self._build_ctx()
        self._last_ctx = ctx
        obs = self._obs()
        info: dict[str, Any] = {
            "legal_mask": ctx.match_mask
            if self.state.phase == TurnPhase.MATCH
            else np.zeros_like(ctx.match_mask),
            "play_mask": ctx.play_mask,
            "scores": [total_score_points(self.state, p) for p in range(self.state.num_players)],
        }
        if self._bench_log is not None:
            info["bench_log"] = self._bench_log
        info["bench"] = bench_snapshot(clone_state(self.state))
        return obs, info

    def _apply_agent_match(self, action: int) -> tuple[float, bool]:
        assert self.state is not None and self._last_ctx is not None
        st = self.state
        ctx = self._last_ctx
        is_pass, row, col = decode_match_phase_flat(self._max_match_combos, action)
        pr = self._max_match_combos
        if is_pass:
            if col != 0 or ctx.match_mask[row, col] != 1:
                return -1.0, False
            apply_move(st, PassMatch())
            return 0.0, is_finished(st)
        hand = st.hands[st.current_player]
        subs = hand_nonempty_subsets(len(hand))
        combo_id = row
        pool_col = col
        if combo_id < 0 or combo_id >= len(subs):
            return -1.0, False
        if pool_col >= len(st.public):
            return -1.0, False
        if ctx.match_mask[combo_id, pool_col] != 1:
            return -1.0, False
        combo = subs[combo_id]
        apply_move(st, MatchMove(pool_col, combo))
        return 0.0, is_finished(st)

    def _apply_agent_play(self, hand_index: int) -> tuple[float, bool]:
        assert self.state is not None and self._last_ctx is not None
        st = self.state
        if st.current_player != self.learning_player:
            return -1.0, False
        if self._last_ctx.play_mask[hand_index] != 1:
            return -1.0, False
        apply_move(st, PlayMove(hand_index))
        return 0.0, is_finished(st)

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Gymnasium API: one transition.

        Parameters
        ----------
        action
            Integer in ``action_space`` (see class docstring for encoding).

        Returns
        -------
        observation, reward, terminated, truncated, info

        After a **legal** control move, other seats may auto-play (if ``autoplay_non_control``).
        """
        assert self.state is not None
        st = self.state
        self._post_play_obs = None
        self._last_ctx = self._build_ctx()
        ctx = self._last_ctx

        if st.current_player != self.learning_player or is_finished(st):
            obs = self._obs()
            inf = self._info_dict(ctx)
            return obs, -1.0, is_finished(st), False, inf

        a = int(action)
        was_play_phase = st.phase == TurnPhase.PLAY
        if st.phase == TurnPhase.PLAY:
            if a < self.match_flat_dim:
                inf = self._info_dict(ctx)
                return self._obs(), -1.0, False, False, inf
            hi = a - self.match_flat_dim
            reward, done = self._apply_agent_play(hi)
        else:
            if a >= self.match_flat_dim:
                inf = self._info_dict(ctx)
                return self._obs(), -1.0, False, False, inf
            reward, done = self._apply_agent_match(a)

        if reward < 0:
            ctx_bad = self._build_ctx()
            inf = self._info_dict(ctx_bad)
            return self._obs(), reward, False, False, inf

        if was_play_phase:
            from pick14.rl.rlmd_obs import encode_rlmd_post_play

            self._post_play_obs = encode_rlmd_post_play(self)

        if self.autoplay_non_control:
            _run_autoplay_until_learning(st, self.learning_player, self._bench_log, self.agents)
        ctx2 = self._build_ctx()
        self._last_ctx = ctx2
        obs = self._obs()
        terminated = is_finished(st)
        info = self._info_dict(ctx2)
        info["bench"] = bench_snapshot(clone_state(st))
        return obs, reward, terminated, False, info

    def _info_dict(self, ctx: _StepContext) -> dict[str, Any]:
        assert self.state is not None
        st = self.state
        if st.phase == TurnPhase.PLAY:
            legal = np.zeros((self._max_match_combos + 1, MAX_PUBLIC_SLOTS), dtype=np.int8)
        else:
            legal = ctx.match_mask
        out: dict[str, Any] = {
            "legal_mask": legal,
            "play_mask": ctx.play_mask,
            "scores": [total_score_points(st, p) for p in range(st.num_players)],
        }
        if self._bench_log is not None:
            out["bench_log"] = self._bench_log
        return out

    def render(self) -> str:
        """
        Gymnasium API: human-readable snapshot (optional).

        This is **not** the observation tensor and **not** full state — only a one-line debug summary.
        For omniscient logging use ``info["bench"]`` / ``bench_snapshot`` (see module FAQ).
        """
        if self.state is None:
            return "No state"
        st = self.state
        return (
            f"p={st.current_player} learning={self.learning_player} "
            f"hand0={len(st.hands[0])} public={len(st.public)} "
            f"deck={len(st.deck)} done={is_finished(st)} phase={st.phase.name}"
        )

    def teacher_action(self) -> int:
        """
        Not part of the Gymnasium API: returns an action index for the built-in expert (rl.md §1.5).

        Useful for behaviour cloning or sanity checks; RL trainers only need ``reset``/``step``.
        """
        assert self.state is not None
        st = self.state
        if st.current_player != self.learning_player or is_finished(st):
            return 0
        mv = _EXPERT_SEAT.act(st)
        return self._flat_action_for_move(st, mv)

    def action_for_move(self, move: Move) -> int:
        """Encode a rules-level ``Move`` into this env's flat ``Discrete`` index for ``state``'s current player."""
        assert self.state is not None
        return self._flat_action_for_move(self.state, move)
