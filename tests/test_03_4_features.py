"""Unit tests for 03-4 per-sum feature vectors and regression helpers."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "m34", _root / "scripts" / "03-4_linear_regression.py"
)
assert _spec and _spec.loader
m = importlib.util.module_from_spec(_spec)
sys.modules["m34"] = m
_spec.loader.exec_module(m)

from pick14.cards import Card, Rank, Suit  # noqa: E402


def _card(rank: int) -> Card:
    return Card(is_joker=False, rank=Rank(rank), suit=Suit.CLUB)


def test_feat_vector_lengths() -> None:
    h = [_card(1), _card(4), _card(5)]
    pub: list[Card] = []
    fa, fb = m.build_feature_vectors(h, pub)
    assert len(fa) == m.N_FEAT_A  # 27
    assert len(fb) == m.N_FEAT_B  # 53


def test_x1_x2_ace_four_five() -> None:
    """Hand A/4/5: subsets summing to s=1..13 (ignore game_value capping)."""
    h = [_card(1), _card(4), _card(5)]
    fa, fb = m.build_feature_vectors(h, [])
    # sum=5 (A+4): x1[4]=1 and x2[4]>=1
    assert fa[4] == 1.0   # x1[s=5] — index 4
    assert fa[m.N_SUMS + 4] >= 1.0  # x2[s=5]
    # sum=10 (A+4+5=10): x1[9]=1
    assert fa[9] == 1.0


def test_x3_x4_with_public() -> None:
    """x3/x4 for a hand of [5] and public [9H, 9S] (complement of 5 is 9).
    score_value: HEART=4, SPADE=3.
    """
    hand = [_card(5)]
    pub = [
        Card(is_joker=False, rank=Rank.NINE, suit=Suit.HEART),   # score 4
        Card(is_joker=False, rank=Rank.NINE, suit=Suit.SPADE),   # score 3
    ]
    _fa, fb = m.build_feature_vectors(hand, pub)
    # s=5 -> comp=9; x3[4] = best pub score = 4; x4[4] = second = 3
    x3_idx = 2 * m.N_SUMS + 4   # x1(13) + x2(13) + index-4
    x4_idx = 3 * m.N_SUMS + 4
    assert fb[x3_idx] == 4.0
    assert fb[x4_idx] == 3.0


def test_x3_zero_if_no_public_complement() -> None:
    hand = [_card(5)]
    _fa, fb = m.build_feature_vectors(hand, [])
    x3_idx = 2 * m.N_SUMS + 4
    assert fb[x3_idx] == 0.0


def test_bias_last() -> None:
    fa, fb = m.build_feature_vectors([], [])
    assert fa[-1] == 1.0
    assert fb[-1] == 1.0


def test_collect_small_run() -> None:
    fa, fb, g1, g2 = m.collect_samples(30, 4, 3, 0, report_every=1000)
    assert fa.shape[1] == m.N_FEAT_A
    assert fb.shape[1] == m.N_FEAT_B
    assert len(fa) == len(g1) == len(g2)
    # g1 has no NaN; g2 may
    assert not np.any(np.isnan(g1))
