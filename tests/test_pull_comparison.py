"""Unit tests for pull_comparison helpers that need no network access.

Focus: the approval split (which must keep modified ``A*`` codes on the approved
side and produce two non-overlapping sides) and the season-matched window used to
control for seasonality when comparing the two spans (CLAUDE.md §9.3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.datasets.pull_comparison import (
    ComparisonConfig,
    _validate,
    season_matched_window,
    split_by_approval,
    summarise_side,
)


def _tidy(qualifiers: list[str], start: str = "2024-01-01", freq: str = "15min") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(qualifiers), freq=freq)
    return pd.DataFrame(
        {
            "datetime": idx,
            "value": np.arange(len(qualifiers), dtype=float),
            "qualifier": qualifiers,
        }
    )


# --------------------------------------------------------------------------
# split_by_approval
# --------------------------------------------------------------------------
def test_split_keeps_modified_approved_codes_on_the_approved_side() -> None:
    # 'A e' (estimated), 'A, >' (over-range) and 'A, R' (revised) are all approved
    # readings -- USGS approved them -- so they must not leak into provisional.
    df = _tidy(["A", "A e", "A, >", "A, R", "P"])
    approved, provisional = split_by_approval(df)
    assert len(approved) == 4
    assert list(provisional["qualifier"]) == ["P"]


def test_split_treats_blank_and_missing_qualifiers_as_not_approved() -> None:
    df = _tidy(["A", "P", ""])
    df.loc[3] = {"datetime": pd.Timestamp("2024-01-01 00:45"), "value": 3.0,
                 "qualifier": None}
    approved, provisional = split_by_approval(df)
    assert len(approved) == 1
    assert len(provisional) == 3


def test_split_sides_are_disjoint_and_exhaustive() -> None:
    df = _tidy(["A"] * 50 + ["P"] * 20)
    approved, provisional = split_by_approval(df)
    assert len(approved) + len(provisional) == len(df)
    assert set(approved["datetime"]).isdisjoint(set(provisional["datetime"]))


def test_split_without_qualifier_column_raises() -> None:
    df = _tidy(["A", "P"]).drop(columns=["qualifier"])
    with pytest.raises(ValueError, match="qualifier"):
        split_by_approval(df)


# --------------------------------------------------------------------------
# summarise_side
# --------------------------------------------------------------------------
def test_summarise_side_reports_step_span_and_completeness() -> None:
    df = _tidy(["A"] * (4 * 24 + 1))  # exactly one day of 15-min data
    s = summarise_side(df, "approved")
    assert s.state == "approved"
    assert s.n_rows == 4 * 24 + 1
    assert s.median_dt_min == pytest.approx(15.0)
    assert s.span_days == pytest.approx(1.0, abs=1e-6)
    assert s.completeness_pct == pytest.approx(100.0, rel=0.02)
    assert s.qualifier_counts == {"A": 4 * 24 + 1}


def test_summarise_side_handles_a_single_row_without_raising() -> None:
    s = summarise_side(_tidy(["P"]), "provisional")
    assert s.n_rows == 1
    assert np.isnan(s.median_dt_min)


# --------------------------------------------------------------------------
# season_matched_window
# --------------------------------------------------------------------------
def test_season_matched_window_shifts_back_one_year_when_that_fits() -> None:
    approved = _tidy(["A"] * 96, start="2024-01-01", freq="1D")     # 2024-01-01 -> 2024-04-05
    provisional = _tidy(["P"] * 10, start="2025-02-01", freq="1D")  # 2025-02-01 -> 2025-02-10
    got = season_matched_window(provisional, approved)
    assert got is not None
    assert got["years_back"] == "1"
    assert got["start"].startswith("2024-02-01")
    assert int(got["n_approved_rows"]) == 10


def test_season_matched_window_returns_none_when_no_shift_fits() -> None:
    # Approved span is far too short to contain the provisional span at any offset.
    approved = _tidy(["A"] * 5, start="2024-01-01", freq="1D")
    provisional = _tidy(["P"] * 10, start="2025-06-01", freq="1D")
    assert season_matched_window(provisional, approved) is None


def test_season_matched_window_returns_none_on_an_empty_side() -> None:
    assert season_matched_window(_tidy([]), _tidy(["A"] * 5)) is None


# --------------------------------------------------------------------------
# config validation
# --------------------------------------------------------------------------
def test_validate_rejects_bad_site_numbers_and_windows() -> None:
    with pytest.raises(ValueError, match="site number"):
        _validate(ComparisonConfig(sites=("abc",)))
    with pytest.raises(ValueError, match="No sites"):
        _validate(ComparisonConfig(sites=()))
    with pytest.raises(ValueError, match="must be after"):
        _validate(ComparisonConfig(start="2025-01-01", end="2024-01-01"))


def test_validate_accepts_the_default_config() -> None:
    _validate(ComparisonConfig())  # must not raise
