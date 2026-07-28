"""Tests for the SaQC-wrapping tool functions and SaQC 2.8 presence."""

import json
import numpy as np
import pandas as pd
import pytest
import saqc

from src.tools import wrappers


def test_saqc_version():
    assert saqc.__version__ == "2.8.0"


def test_saqc_required_methods_exist():
    """Smoke test that required methods exist on the SaQC object."""
    data = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-01-01", periods=3, freq="15min"))
    qc = saqc.SaQC(data)

    assert hasattr(qc, "flagRange")
    assert hasattr(qc, "flagConstants")
    assert hasattr(qc, "flagPlateau")
    assert hasattr(qc, "flagUniLOF")
    assert hasattr(qc, "flagZScore")
    assert hasattr(qc, "flagJumps")
    assert hasattr(qc, "flagNAN")
    assert hasattr(qc, "interpolateByRolling")
    assert hasattr(qc, "correctDrift")


def _toy_qc():
    idx = pd.date_range("2024-01-01", periods=100, freq="15min")
    rng = np.random.default_rng(42)
    v = 10 + np.sin(np.arange(100) / 10) + rng.normal(0, 0.1, 100)
    # Add a spike
    v[50] = 50.0
    # Add a gap
    v[80:85] = np.nan
    data = pd.DataFrame({"value": v}, index=idx)
    return saqc.SaQC(data)


def test_inspect_dataset_returns_json():
    qc = _toy_qc()
    result = wrappers.inspect_dataset(qc, field="value")
    assert "tool" in result
    assert result["tool"] == "inspect_dataset"
    assert "params" in result
    assert "message" in result
    # Shouldn't error if JSON serialized
    json.dumps({k: v for k, v in result.items() if k not in ["qc", "df"]})


def test_flag_spike_unilof_returns_json():
    qc = _toy_qc()
    result = wrappers.flag_spike_unilof(qc, field="value", n=20, thresh=1.5)

    assert result["tool"] == "flag_spike_unilof"
    assert "params" in result
    assert "n_flagged" in result
    assert "pct_flagged" in result
    assert "flagged_datetimes" in result
    assert "message" in result
    json.dumps({k: v for k, v in result.items() if k not in ["qc", "df"]})


def test_flag_nan_returns_json():
    qc = _toy_qc()
    result = wrappers.flag_nan(qc, field="value")

    assert result["tool"] == "flag_nan"
    assert result["n_flagged"] == 5
    assert len(result["flagged_datetimes"]) == 5
    json.dumps({k: v for k, v in result.items() if k not in ["qc", "df"]})


def test_impute_rolling_returns_json():
    qc = _toy_qc()
    result = wrappers.impute_rolling(qc, field="value", window="2h")

    assert result["tool"] == "impute_rolling"
    assert result["n_flagged"] > 0
    assert result["message"]
    json.dumps({k: v for k, v in result.items() if k not in ["qc", "df"]})


def test_correct_drift_preserves_trailing():
    idx = pd.date_range("2024-01-01", periods=100, freq="15min")
    v = np.linspace(10, 15, 100)
    data = pd.DataFrame({"value": v}, index=idx)

    # 3 maintenance points
    maint_starts = [idx[10], idx[50], idx[90]]
    maint_ends = [idx[11], idx[51], idx[91]]
    maint = pd.DataFrame({"maintenance": pd.Series(maint_ends, index=pd.DatetimeIndex(maint_starts))})

    # Needs to be passed into saqc constructor correctly
    qc_input = saqc.SaQC(pd.concat([data, maint], axis=1).sort_index())

    # Should not lose data at the end!
    result = wrappers.correct_drift(qc_input, maint, field="value", model="linear", cal_range=5)
    assert result["tool"] == "correct_drift"

    qc_out = result["qc"] # Assuming we return the qc object for chainability in CLI/Agent
    out_data = qc_out.data["value"]

    # 91 to 99 should not be nan
    assert not out_data.iloc[-5:].isna().all() # Should not be NaN at the end!
