"""
Seat agents for ``Pick14GymEnv``: composable **match** and **play** policies per rl.md §1.5-style bots.

Each seat is an object with ``act(state) -> Move``. Combine ``MatchPolicy`` + ``PlayPolicy`` in
``CompositeSeatAgent`` (e.g. greedy–stingy match + stingy play, or always-pass match + stingy play).
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from pick14.rl.sim_core import (
    MatchMove,
    Move,
    PassMatch,
    PlayMove,
    RlPick14State,
    TurnPhase,
    caution_play,
    greedy_for_public_match,
    greedy_stingy_match,
    greedy_stingy_play,
)


@runtime_checkable
class MatchPolicy(Protocol):
    """In match sub-phase: return a match or ``None`` to take :class:`PassMatch` (separate play ``step``)."""

    def choose_match(self, state: RlPick14State) -> MatchMove | None: ...


@runtime_checkable
class PlayPolicy(Protocol):
    """Choose a hand index for both pass-to-pool and forced discard after a match."""

    def choose_play(self, state: RlPick14State) -> PlayMove: ...


class CompositeSeatAgent:
    """
    Assemble independent match and play strategies (two Gym steps for pass-then-discard).

    Flow: if ``TurnPhase.PLAY`` → ``play_policy`` (same after Draw1 or after pass-match); else match or :class:`PassMatch`.
    """

    def __init__(self, match_policy: MatchPolicy, play_policy: PlayPolicy) -> None:
        self.match_policy = match_policy
        self.play_policy = play_policy

    def act(self, state: RlPick14State) -> Move:
        if state.phase == TurnPhase.PLAY:
            return self.play_policy.choose_play(state)
        mm = self.match_policy.choose_match(state)
        if mm is not None:
            return mm
        return PassMatch()


class GreedyStingyMatchPolicy:
    """rl.md §1.5 greedy–stingy match (total capture, then hand size, then public points)."""

    def choose_match(self, state: RlPick14State) -> MatchMove | None:
        return greedy_stingy_match(state)


class GreedyForPublicMatchPolicy:
    """rl.md §1.5 greedy-for-public match (public points, then total capture, then hand size)."""

    def choose_match(self, state: RlPick14State) -> MatchMove | None:
        return greedy_for_public_match(state)


class StingyPlayPolicy:
    """rl.md §1.5 stingy play (lowest point, then highest digit)."""

    def choose_play(self, state: RlPick14State) -> PlayMove:
        return greedy_stingy_play(state)


class CautionPlayPolicy:
    """rl.md §1.5 caution play (highest digit, then lowest point)."""

    def choose_play(self, state: RlPick14State) -> PlayMove:
        return caution_play(state)


class AlwaysPassMatchPolicy:
    """Never match; always defer to ``PlayPolicy`` (pass / play to pool)."""

    def choose_match(self, state: RlPick14State) -> MatchMove | None:
        return None


def greedy_stingy_seat() -> CompositeSeatAgent:
    """Classic greedy–stingy seat (rl.md §1.5); for ablations / matrix row ``greedy_stingy``."""
    return CompositeSeatAgent(GreedyStingyMatchPolicy(), StingyPlayPolicy())


def pass_stingy_seat() -> CompositeSeatAgent:
    """Always pass match, stingy play (useful for ablations)."""
    return CompositeSeatAgent(AlwaysPassMatchPolicy(), StingyPlayPolicy())


def table_of(factory: Callable[[], Any], num_seats: int) -> list[Any]:
    """Build ``num_seats`` agents from ``factory()`` (fresh instance per seat)."""
    if num_seats < 2:
        raise ValueError("need at least 2 seats")
    return [factory() for _ in range(num_seats)]


def table_all_greedy_stingy(num_seats: int) -> list[Any]:
    """Shorthand: every seat uses ``greedy_stingy_seat()``."""
    return table_of(greedy_stingy_seat, num_seats)


def greedy_for_public_stingy_seat() -> CompositeSeatAgent:
    """Greedy-for-public match + stingy play."""
    return CompositeSeatAgent(GreedyForPublicMatchPolicy(), StingyPlayPolicy())


def greedy_stingy_caution_seat() -> CompositeSeatAgent:
    """Greedy–stingy match + caution play."""
    return CompositeSeatAgent(GreedyStingyMatchPolicy(), CautionPlayPolicy())


def pass_caution_seat() -> CompositeSeatAgent:
    """Always-pass match + caution play."""
    return CompositeSeatAgent(AlwaysPassMatchPolicy(), CautionPlayPolicy())


def greedy_for_public_caution_seat() -> CompositeSeatAgent:
    """Greedy-for-public match + caution play (rl.md §1.5 default baseline; best 1v1 in strategy matrix)."""
    return CompositeSeatAgent(GreedyForPublicMatchPolicy(), CautionPlayPolicy())


def baseline_seat() -> CompositeSeatAgent:
    """Alias for :func:`greedy_for_public_caution_seat` (recommended teacher / opponent for curriculum)."""
    return greedy_for_public_caution_seat()


def table_all_baseline(num_seats: int) -> list[Any]:
    """Every seat uses the §1.5 baseline (greedy-for-public match + caution play)."""
    return table_of(baseline_seat, num_seats)


def SEAT_FACTORY_CHOICES() -> list[Callable[[], CompositeSeatAgent]]:
    """All bundled seat factories (for random mix tests)."""
    return [
        greedy_for_public_caution_seat,
        greedy_stingy_seat,
        pass_stingy_seat,
        greedy_for_public_stingy_seat,
        greedy_stingy_caution_seat,
        pass_caution_seat,
    ]
