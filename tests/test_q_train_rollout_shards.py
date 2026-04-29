"""Tests for loading mass-rollout NPZ shards into training tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pick14.rl.q_train import load_rollout_shards_numpy


def test_load_rollout_shards_concat_and_fp16(tmp_path: Path) -> None:
    x1 = np.random.randn(3, 54, 27).astype(np.float32)
    qt1 = np.random.randn(3, 1).astype(np.float32)
    opp1 = np.random.rand(3, 54).astype(np.float32)
    x2 = np.random.randn(2, 54, 27).astype(np.float16)
    qt2 = np.random.randn(2, 1).astype(np.float16)
    opp2 = np.random.rand(2, 54).astype(np.float16)
    np.savez_compressed(tmp_path / "shard_00000.npz", x=x1, qt=qt1, opp=opp1)
    np.savez_compressed(tmp_path / "shard_00001.npz", x=x2, qt=qt2, opp=opp2)

    x, qt, opp = load_rollout_shards_numpy(tmp_path)
    assert x.shape == (5, 54, 27)
    assert qt.shape == (5, 1)
    assert opp.shape == (5, 54)
    assert x.dtype == np.float32 and qt.dtype == np.float32 and opp.dtype == np.float32


def test_load_rollout_shards_missing_dir_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        load_rollout_shards_numpy(empty)


def test_load_rollout_qt_1d_expanded(tmp_path: Path) -> None:
    x = np.zeros((2, 54, 27), dtype=np.float32)
    qt = np.array([1.0, 2.0], dtype=np.float32)
    opp = np.zeros((2, 54), dtype=np.float32)
    np.savez_compressed(tmp_path / "shard_00000.npz", x=x, qt=qt, opp=opp)
    _, qt2, _ = load_rollout_shards_numpy(tmp_path)
    assert qt2.shape == (2, 1)
