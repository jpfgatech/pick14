"""04-1 side-branch record schema (no simulator yet)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "m41", _root / "scripts" / "04-1_side_branch_eval.py",
)
assert _spec and _spec.loader
m = importlib.util.module_from_spec(_spec)
sys.modules["m41"] = m
_spec.loader.exec_module(m)


def test_header_json_round_trip() -> None:
    h = m.GameRunHeader(
        schema_version=m.SCHEMA_VERSION,
        seed=1,
        n_players=4,
        n_hand=3,
        explorer_seat=0,
        policy_match="greedy_stingy",
        policy_play="caution",
    )
    d = h.to_json_dict()
    s = json.dumps(d)
    back = json.loads(s)
    assert back["schema_version"] == m.SCHEMA_VERSION
    assert back["policy"]["match"] == "greedy_stingy"


def test_snapshot_fields() -> None:
    sn = m.LayerSnapshot(
        fork_id=0,
        branch_id=1,
        layer=0,
        truncated_after_layer=False,
        per_player_total_score=[0, 0, 0, 0],
    )
    assert sn.n_plus_one_gap is None


def test_collect_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        m.collect_side_branch_dataset(0)
