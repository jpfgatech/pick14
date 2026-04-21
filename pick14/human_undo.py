from __future__ import annotations

from dataclasses import dataclass, field

from pick14.engine import GameState, clone, is_finished


@dataclass
class HumanSegmentUndo:
    """One `r` reverts the current human segment (match + forced play = one segment).

    Checkpoints for earlier segments are kept on ``segments_stack`` (newest last).
    """

    human_seat: int = 0
    segments_stack: list[GameState] = field(default_factory=list)
    segment_base: GameState | None = None
    dirty: bool = False

    def on_human_turn_begin(self, state: GameState) -> None:
        """Call when it is the human seat's turn and the UI is about to prompt.

        Forced-play continuation (``must_play_only``) keeps the existing ``segment_base``.
        """
        if state.current_player != self.human_seat:
            return
        if state.must_play_only:
            return
        self.segment_base = clone(state)
        self.dirty = False

    def on_human_committed(self, state_after: GameState) -> None:
        """Call after a human ``apply_move`` succeeds."""
        self.dirty = True
        if self._segment_finished(state_after) and self.segment_base is not None:
            self.segments_stack.append(self.segment_base)
            self.segment_base = None
            self.dirty = False

    def _segment_finished(self, state: GameState) -> bool:
        if is_finished(state):
            return True
        return state.current_player != self.human_seat

    def regret(self) -> tuple[GameState | None, int]:
        """Return ``(new_state, remaining_completed_segments)`` or ``(None, n)``."""
        if self.dirty and self.segment_base is not None:
            self.dirty = False
            return clone(self.segment_base), len(self.segments_stack)
        if self.segments_stack:
            restored = clone(self.segments_stack.pop())
            self.segment_base = None
            self.dirty = False
            return restored, len(self.segments_stack)
        return None, len(self.segments_stack)
