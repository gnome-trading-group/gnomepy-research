"""Tests for walk-forward fold layout.

`step_mode` was accepted and never read, so `--mode expanding` and `--mode rolling`
produced byte-identical windows.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from gnomepy_research.validation.walk_forward import (
    WalkForwardConfig,
    _generate_fold_windows,
)

START = datetime(2026, 2, 1)
END = datetime(2026, 3, 1)


def _cfg(mode: str, folds: int = 4) -> WalkForwardConfig:
    return WalkForwardConfig(total_start=START, total_end=END, n_folds=folds, step_mode=mode)


def test_modes_produce_different_windows():
    assert _generate_fold_windows(_cfg("expanding")) != _generate_fold_windows(_cfg("rolling"))


def test_rolling_windows_are_disjoint_and_contiguous():
    windows = _generate_fold_windows(_cfg("rolling"))
    assert windows[0][0] == START and windows[-1][1] == END
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
        assert prev_end == next_start


def test_expanding_windows_share_a_start_and_grow():
    windows = _generate_fold_windows(_cfg("expanding"))
    assert all(start == START for start, _ in windows)
    ends = [end for _, end in windows]
    assert ends == sorted(ends) and len(set(ends)) == len(ends)
    assert ends[-1] == END


@pytest.mark.parametrize("mode", ["expanding", "rolling"])
def test_fold_count_is_respected(mode):
    assert len(_generate_fold_windows(_cfg(mode, folds=7))) == 7


def test_invalid_step_mode_is_rejected():
    with pytest.raises(ValueError, match="step_mode"):
        _generate_fold_windows(_cfg("sideways"))


@pytest.mark.parametrize("folds", [0, -1])
def test_nonpositive_fold_count_is_rejected(folds):
    with pytest.raises(ValueError, match="n_folds"):
        _generate_fold_windows(_cfg("rolling", folds=folds))


def test_inverted_range_is_rejected():
    cfg = WalkForwardConfig(total_start=END, total_end=START, n_folds=3)
    with pytest.raises(ValueError, match="total_end"):
        _generate_fold_windows(cfg)
