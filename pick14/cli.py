from __future__ import annotations

import sys
from random import Random

from pick14.agent import choose_dummy_action, match_capture_points
from pick14.cards import format_card, score_value
from pick14.engine import (
    GameState,
    MatchMove,
    PlayMove,
    apply_move,
    clone,
    is_finished,
    match_moves,
    new_game,
    play_moves,
    score_as_sets_and_points,
    skip_empty_hands,
    total_score_points,
)
from pick14.human_undo import HumanSegmentUndo


def _points_phrase(n: int) -> str:
    """Align the numeric part to 2 columns, e.g. '( 9 points)' vs '(10 points)'."""
    return f"({n:>2} points)"


def _sorted_matches(state: GameState) -> list[MatchMove]:
    ms = match_moves(state)

    def sort_key(m: MatchMove) -> tuple[int, int, tuple[int, ...]]:
        pts = match_capture_points(state, m)
        return (-pts, m.public_index, m.hand_indices)

    return sorted(ms, key=sort_key)


def _sorted_plays(state: GameState) -> list[PlayMove]:
    ps = play_moves(state)
    hand = state.hands[state.current_player]

    def sort_key(p: PlayMove) -> tuple[int, int]:
        return (score_value(hand[p.hand_index]), p.hand_index)

    return sorted(ps, key=sort_key)


def _move_points_before(state: GameState, move: PlayMove | MatchMove) -> int:
    if isinstance(move, PlayMove):
        c = state.hands[state.current_player][move.hand_index]
        return score_value(c)
    return match_capture_points(state, move)


def _print_scores(state: GameState) -> None:
    print("\n=== Game over — scores ===")
    for i in range(state.num_players):
        pts = total_score_points(state, i)
        sets, rem = score_as_sets_and_points(pts)
        label = "You" if i == 0 else f"Player {i}"
        print(f"{label}: {pts} points ({sets} sets and {rem} points)")


def _describe_move(state: GameState, move: PlayMove | MatchMove) -> str:
    if isinstance(move, PlayMove):
        c = state.hands[state.current_player][move.hand_index]
        return f"play {format_card(c)}"
    pub = state.public[move.public_index]
    hand = state.hands[state.current_player]
    parts = ", ".join(format_card(hand[i]) for i in move.hand_indices)
    return f"pick {format_card(pub)} by {parts}"


def _describe_move_with_points(state: GameState, move: PlayMove | MatchMove) -> str:
    pts = _move_points_before(state, move)
    return f"{_points_phrase(pts)} {_describe_move(state, move)}"


def _print_table(state: GameState) -> None:
    print("\nPublic pool:")
    for c in state.public:
        print(f"  {format_card(c)}")
    print("\nYour hand:")
    for c in state.hands[0]:
        print(f"  {format_card(c)}")
    if state.must_play_only:
        print("\n(You must play one card to the public pool.)")


def _format_match_line(state: GameState, m: MatchMove, idx: int) -> str:
    pts = match_capture_points(state, m)
    pub = state.public[m.public_index]
    hand = state.hands[state.current_player]
    parts = ", ".join(format_card(hand[i]) for i in m.hand_indices)
    return f"[{idx}] {_points_phrase(pts)} pick {format_card(pub)} by {parts}"


def _format_play_line(state: GameState, p: PlayMove, idx: int) -> str:
    c = state.hands[state.current_player][p.hand_index]
    pts = score_value(c)
    return f"[{idx}] {_points_phrase(pts)} play {format_card(c)}"


def _read_token() -> str:
    return input("> ").strip().lower()


def _human_turn(state: GameState, commit_human) -> str:
    """Apply a human move via `commit_human`, or return quit/new/regret."""
    _print_table(state)

    if not state.must_play_only:
        ms = _sorted_matches(state)
        if ms:
            print("\nMatch (optional):")
            print(f"  [0] {_points_phrase(0)} skip")
            for i, m in enumerate(ms, start=1):
                print(f"  {_format_match_line(state, m, i)}")
            tok = _read_token()
            if tok == "q":
                return "quit"
            if tok == "n":
                return "new"
            if tok == "r":
                return "regret"
            if tok.isdigit():
                choice = int(tok)
                if 1 <= choice <= len(ms):
                    commit_human(ms[choice - 1])
                    return "ok"
                # 0 or out of range → fall through to plays

    ps = _sorted_plays(state)
    if not ps:
        return "ok"

    print("\nPlay a card from your hand (0 .. {}):".format(len(ps) - 1))
    for i, p in enumerate(ps):
        print(f"  {_format_play_line(state, p, i)}")

    while True:
        tok = _read_token()
        if tok == "q":
            return "quit"
        if tok == "n":
            return "new"
        if tok == "r":
            return "regret"
        if tok.isdigit():
            choice = int(tok)
            if 0 <= choice < len(ps):
                commit_human(ps[choice])
                return "ok"
        print("Invalid choice; enter a listed index, or q / n / r.")


def run_session(num_players: int, rng: Random) -> str:
    state = new_game(num_players, rng=rng)
    undo_ctl = HumanSegmentUndo()
    consecutive_undos = 0

    def commit_human(move: PlayMove | MatchMove) -> None:
        nonlocal state
        apply_move(state, move)
        undo_ctl.on_human_committed(state)

    while True:
        skip_empty_hands(state)
        if is_finished(state):
            _print_scores(state)
            return "done"

        cp = state.current_player
        if cp != 0:
            st_before = clone(state)
            mv = choose_dummy_action(state)
            apply_move(state, mv)
            who = f"Player {cp}"
            print(f"\n{who}: {_describe_move_with_points(st_before, mv)}")
            continue

        undo_ctl.on_human_turn_begin(state)
        cmd = _human_turn(state, commit_human)
        if cmd == "quit":
            return "quit"
        if cmd == "new":
            return "new"
        if cmd == "regret":
            restored, remaining = undo_ctl.regret()
            if restored is not None:
                state = restored
                consecutive_undos += 1
                print(
                    f"Reverted your last human turn segment "
                    f"({consecutive_undos} undo(s) since your last forward play; "
                    f"{remaining} completed segment(s) still reversible)."
                )
            else:
                print("Nothing to undo.")
            continue
        if cmd == "ok":
            consecutive_undos = 0


def main() -> None:
    rng = Random()
    print("Pick14 — you are seat 0; other seats use the dummy agent.")
    print("Commands any time: q quit, n new game, r regret.")
    print(
        "Regret: each r restores to the start of your current turn segment, "
        "or to before a prior completed segment (match + forced play count as one segment). "
        "Bot moves after your segment are rolled back with it."
    )
    while True:
        raw = input("\nHow many players (2+)? ").strip().lower()
        if raw == "q":
            break
        try:
            n = int(raw)
        except ValueError:
            print("Enter a number ≥ 2, or q.")
            continue
        if n < 2:
            print("Need at least 2 players.")
            continue

        while True:
            out = run_session(n, rng)
            if out == "quit":
                return
            if out == "new":
                break
            if out == "done":
                again = input("\nAnother round with same player count? [Y/n] ").strip().lower()
                if again in ("n", "no"):
                    break
                continue


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
        sys.exit(0)
