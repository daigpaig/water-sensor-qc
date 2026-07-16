"""Unit tests for pull_usgs helpers that need no network access.

Focus: `longest_unbroken_run_days` (the unbroken-stretch filter that drives
gauge selection, CLAUDE.md §9) and config validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pull_usgs import (
    PullConfig,
    _validate,
    longest_unbroken_run_days,
)


def _series(start: str, periods: int, freq: str = "15min") -> pd.Series:
    idx = pd.date_range(start, periods=periods, freq=freq)
    return pd.Series(np.arange(periods, dtype=float), index=idx)


def test_fully_continuous_run_equals_full_span() -> None:
    # 10 days of clean 15-min data -> longest run == the whole span.
    s = _series("2024-01-01", periods=10 * 96 + 1)  # +1 so span is exactly 10 days
    got = longest_unbroken_run_days(s, pd.Timedelta("3h"))
    assert got == pytest.approx(10.0, abs=1e-6)


def test_small_gaps_do_not_break_the_run() -> None:
    # Drop a single interior sample -> a 30-min gap, below the 3h tolerance.
    s = _series("2024-01-01", periods=10 * 96 + 1)
    s = s.drop(s.index[500])
    got = longest_unbroken_run_days(s, pd.Timedelta("3h"))
    assert got == pytest.approx(10.0, abs=1e-6)


def test_large_gap_splits_and_returns_longer_segment() -> None:
    # Two clean blocks (2 days, then 5 days) separated by a 6h hole.
    a = _series("2024-01-01", periods=2 * 96 + 1)          # 2-day block
    b = _series("2024-01-03 06:00", periods=5 * 96 + 1)    # 5-day block, 6h after a ends
    s = pd.concat([a, b])
    # 3h tolerance: the 6h hole breaks it -> longer (5-day) segment wins.
    assert longest_unbroken_run_days(s, pd.Timedelta("3h")) == pytest.approx(5.0, abs=1e-6)
    # 12h tolerance: the 6h hole is bridged -> one run across the whole span.
    span = (s.index.max() - s.index.min()).total_seconds() / 86400.0
    assert longest_unbroken_run_days(s, pd.Timedelta("12h")) == pytest.approx(span, abs=1e-6)


def test_nan_values_count_as_gaps() -> None:
    # A run of NaN values wider than max_gap should break the stretch.
    s = _series("2024-01-01", periods=10 * 96 + 1)
    s.iloc[300:340] = np.nan          # 40 * 15min = 10h of NaN, > 3h
    got = longest_unbroken_run_days(s, pd.Timedelta("3h"))
    assert got < 10.0                 # the NaN block split the run
    assert got > 0.0


def test_fewer_than_two_valid_points_returns_zero() -> None:
    idx = pd.date_range("2024-01-01", periods=1, freq="15min")
    assert longest_unbroken_run_days(pd.Series([1.0], index=idx), pd.Timedelta("3h")) == 0.0
    empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
    assert longest_unbroken_run_days(empty, pd.Timedelta("3h")) == 0.0


def test_unit_agnostic_microsecond_resolution() -> None:
    # datetime64[us] index must give the same answer as the default resolution.
    s = _series("2024-01-01", periods=5 * 96 + 1)
    s_us = pd.Series(s.to_numpy(), index=pd.DatetimeIndex(s.index).as_unit("us"))
    assert longest_unbroken_run_days(s_us, pd.Timedelta("3h")) == pytest.approx(5.0, abs=1e-6)


def test_validate_rejects_bad_max_gap() -> None:
    with pytest.raises(ValueError, match="max_gap"):
        _validate(PullConfig(max_gap="not-a-duration"))


def test_validate_rejects_negative_min_unbroken_days() -> None:
    with pytest.raises(ValueError, match="min_unbroken_days"):
        _validate(PullConfig(min_unbroken_days=-1.0))
