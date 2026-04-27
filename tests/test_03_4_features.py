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

from pick14.cards import Card, Rank, Suit, score_value  # noqa: E402


def _card(rank: int) -> Card:
    return Card(is_joker=False, rank=Rank(rank), suit=Suit.CLUB)


def test_feat_vector_lengths() -> None:
    h = [_card(1), _card(4), _card(5)]
    fa, fb = m.build_feature_vectors(h, [])
    assert len(fa) == m.N_FEAT_A
    assert len(fb) == m.N_FEAT_B


def test_x1_max_hand_pts_x2_extra_occ() -> None:
    """A/4/5: s=5 from {A+4} and {5} -> k=2 -> x2=2; x1 = max hand pts of those."""
    h = [_card(1), _card(4), _card(5)]
    fa, _fb = m.build_feature_vectors(h, [])
    p14 = float(score_value(_card(1)) + score_value(_card(4)))
    p5o = float(score_value(_card(5)))
    assert fa[4] == max(p14, p5o)
    assert fa[m.N_SUMS + 4] == 2.0


def test_x2_multiple_subsets() -> None:
    """Force same sum s from two distinct hand subsets: k=2 -> x2=2."""
    # 4+4+4 with three 4s: sums 4 (three singles), 8 (pair), 12 (triplet)
    # s=4: 3 one-card subsets, k=3 -> x2 = 2*2=4
    h = [
        Card(is_joker=False, rank=Rank.FOUR, suit=Suit.CLUB),
        Card(is_joker=False, rank=Rank.FOUR, suit=Suit.HEART),
        Card(is_joker=False, rank=Rank.FOUR, suit=Suit.SPADE),
    ]
    fa, _ = m.build_feature_vectors(h, [])
    assert fa[3] > 0  # x1[s=4] at index 3
    assert fa[m.N_SUMS + 3] == 4.0  # 2 * (3 - 1)


def test_x3_x4_category_totals() -> None:
    """Hand [5C], public 9H+9S: s=5, max_hp=1, comp=9, x3=1+4, x4=1+3."""
    hand = [_card(5)]
    pub = [
        Card(is_joker=False, rank=Rank.NINE, suit=Suit.HEART),
        Card(is_joker=False, rank=Rank.NINE, suit=Suit.SPADE),
    ]
    _fa, fb = m.build_feature_vectors(hand, pub)
    i = 2 * m.N_SUMS + 4
    j = 3 * m.N_SUMS + 4
    assert fb[i] == 1.0 + 4.0
    assert fb[j] == 1.0 + 3.0


def test_x3_zero_if_no_complement() -> None:
    hand = [_card(5)]
    _fa, fb = m.build_feature_vectors(hand, [])
    assert fb[2 * m.N_SUMS + 4] == 0.0
    assert fb[3 * m.N_SUMS + 4] == 0.0


def test_x4_zero_if_one_public() -> None:
    hand = [_card(5)]
    pub = [Card(is_joker=False, rank=Rank.NINE, suit=Suit.HEART)]
    _fa, fb = m.build_feature_vectors(hand, pub)
    i = 2 * m.N_SUMS + 4
    j = 3 * m.N_SUMS + 4
    assert fb[i] == 1.0 + 4.0
    assert fb[j] == 0.0


def test_bias_last() -> None:
    fa, fb = m.build_feature_vectors([], [])
    assert fa[-1] == 1.0
    assert fb[-1] == 1.0


def test_collect_uses_self_score() -> None:
    fa, fb, s1, s2 = m.collect_samples(20, 4, 3, 0, report_every=1000)
    assert not np.any(s1 < 0)
    assert not np.any(np.isnan(s1))
