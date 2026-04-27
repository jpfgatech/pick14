"""Sanity for 03-3 hand features (raw counts / score pts) and snap kinds."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "q3", _root / "scripts" / "03-3_q_feature_analysis.py",
)
assert _spec and _spec.loader
_m = importlib.util.module_from_spec(_spec)
sys.modules["q3"] = _m
_spec.loader.exec_module(_m)

from pick14.cards import Card, Rank, Suit  # noqa: E402


def test_hand_features_empty_hand() -> None:
    h: list[Card] = []
    public = [Card(is_joker=False, rank=Rank.FIVE, suit=Suit.HEART)]
    f = _m._hand_features(h, public)  # type: ignore[attr-defined]
    assert f[0] == 0.0  # f1: no non-empty subset sums
    assert f[1] == 0.0  # f2
    assert f[2] == 0.0  # f3: no sum < 14 with complement in public that matches
    assert f[5] == 0.0  # f6
    assert f[6] == 0.0  # f7


def test_collect_returns_kind_and_length_match() -> None:
    feats, gap, kind = _m.collect_samples(50, 4, 3, 42)  # type: ignore[attr-defined]
    if len(feats) == 0:
        pytest.skip("no samples in short run")
    assert len(feats) == len(gap) == len(kind)
    assert int(kind.min()) >= 0 and int(kind.max()) <= 3


def test_f6_at_most_max_match_points() -> None:
    # Quick run: f6 and f7 should never exceed MAX_MATCH_PTS
    feats, _, _ = _m.collect_samples(200, 4, 3, 0)  # type: ignore[attr-defined]
    if len(feats) == 0:
        pytest.skip("no samples")
    assert np.max(feats[:, 5]) <= _m.MAX_MATCH_PTS + 0.5  # type: ignore[attr-defined]
    assert np.max(feats[:, 6]) <= _m.MAX_MATCH_PTS + 0.5  # type: ignore[attr-defined]
