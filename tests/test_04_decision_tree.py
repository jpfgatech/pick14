"""Smoke tests for 04 decision-tree DFS."""
import importlib.util
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("q4", _root / "scripts" / "04_decision_tree.py")
assert _spec and _spec.loader
_m = importlib.util.module_from_spec(_spec)
sys.modules["q4"] = _m
_spec.loader.exec_module(_m)


def test_single_pass_completes() -> None:
    ms, decisions, checkpoints = _m.run_single_pass(4, 3, 42)
    assert ms > 0
    assert decisions > 0
    assert 0 < checkpoints <= decisions


def test_dfs_fixed_deck_small_limit() -> None:
    stats = _m.run_dfs(4, 3, 42, node_limit=500, n_reshuffles=0)
    assert stats.n_nodes <= 500
    assert stats.n_terminals <= stats.n_nodes
    assert stats.n_checkpoints >= 0
    # With a 500-node limit the tree is always capped.
    assert stats.node_limit_hit


def test_dfs_reshuffle_small_limit() -> None:
    stats = _m.run_dfs(4, 3, 42, node_limit=500, n_reshuffles=1)
    assert stats.n_nodes <= 500
    # reshuffles = 1 means each checkpoint has at most B*(1+1) children.
    assert stats.node_limit_hit


def test_single_pass_2p_small() -> None:
    ms, decisions, checkpoints = _m.run_single_pass(2, 2, 0)
    assert decisions > 0
