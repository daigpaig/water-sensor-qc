"""Tests for the injected-dataset visualiser (src/tools/visualize_injected.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tools.visualize_injected import (
    ANOMALY_COLORS,
    build_figure,
    find_gauges,
    load_injected,
)


def _write_dataset(d: Path, gauge: str, level: int) -> None:
    """A tiny injected dataset + labels: one spike, one injected gap, rest clean."""
    idx = pd.date_range("2024-01-01", periods=6, freq="15min")
    value = pd.Series([1.0, 2.0, 50.0, 4.0, np.nan, 6.0], index=idx)   # spike @2, gap @4
    is_anom = [False, False, True, False, True, False]
    atype = ["", "", "spike", "", "gap", ""]
    true_value = [np.nan, np.nan, 3.0, np.nan, 5.0, np.nan]            # gap's removed reading
    source = ["", "", "injected", "", "injected", ""]
    pd.DataFrame({"datetime": idx, "value": value.to_numpy()}).to_csv(
        d / f"{gauge}_l{level}.csv", index=False
    )
    pd.DataFrame(
        {"datetime": idx, "is_anomaly": is_anom, "anomaly_type": atype,
         "true_value": true_value, "source": source}
    ).to_csv(d / f"{gauge}_l{level}_labels.csv", index=False)


def test_find_gauges(tmp_path: Path) -> None:
    _write_dataset(tmp_path, "12340500", 1)
    _write_dataset(tmp_path, "12340500", 3)
    _write_dataset(tmp_path, "06818000", 2)
    assert find_gauges(tmp_path) == ["06818000", "12340500"]


def test_load_injected_marks_gap_at_true_value(tmp_path: Path) -> None:
    _write_dataset(tmp_path, "12340500", 1)
    df = load_injected("12340500", 1, tmp_path)
    # The injected gap row has NaN value but is marked at its removed true_value.
    gap = df[df["anomaly_type"] == "gap"]
    assert len(gap) == 1
    assert np.isnan(gap["value"].iloc[0])
    assert gap["mark_y"].iloc[0] == 5.0
    # The spike is marked at its contaminated value, not its true_value.
    spike = df[df["anomaly_type"] == "spike"]
    assert spike["mark_y"].iloc[0] == 50.0


def test_load_injected_reconstructs_clean_base(tmp_path: Path) -> None:
    _write_dataset(tmp_path, "12340500", 1)
    df = load_injected("12340500", 1, tmp_path)
    # clean = true_value at anomalies, value elsewhere -> the pre-injection series.
    assert df["clean"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # ...whereas the injected value has the spike (50) and the gap (NaN).
    assert df["value"].iloc[2] == 50.0 and np.isnan(df["value"].iloc[4])


def test_build_figure_has_line_plus_present_type_traces(tmp_path: Path) -> None:
    _write_dataset(tmp_path, "12340500", 1)
    levels = {1: load_injected("12340500", 1, tmp_path)}
    fig = build_figure("12340500", levels)
    names = [tr.name for tr in fig.data]
    assert names.count("clean base (uninjected)") == 1   # one top base panel
    assert names.count("injected value") == 1            # one injected line (1 level)
    assert "spike" in names and "gap" in names
    assert "plateau" not in names             # absent types get no trace


def test_build_figure_base_panel_on_top(tmp_path: Path) -> None:
    # With all three levels, the base is drawn once, as the first (top) panel.
    for lvl in (1, 2, 3):
        _write_dataset(tmp_path, "12340500", lvl)
    levels = {n: load_injected("12340500", n, tmp_path) for n in (1, 2, 3)}
    fig = build_figure("12340500", levels)
    names = [tr.name for tr in fig.data]
    assert names[0] == "clean base (uninjected)"         # top panel, drawn first
    assert names.count("clean base (uninjected)") == 1   # base shown once, not per level
    assert names.count("injected value") == 3            # one per level
    assert "xaxis4" in fig.layout                        # 1 base + 3 level panels
    # Every marker trace uses its type's colour.
    for tr in fig.data:
        if tr.name in ANOMALY_COLORS:
            assert tr.marker.color == ANOMALY_COLORS[tr.name]
