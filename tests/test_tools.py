"""Tests for the SaQC-wrapping tool functions and SaQC 2.8 presence."""

import json
import numpy as np
import pandas as pd
import pytest
import saqc

from src.agent_tools import wrappers


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
    # correctDrift is deliberately NOT asserted here — drift is removed (§9.2).


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
    result = wrappers.impute_rolling(qc, field="value", window="2h", max_gap="1h")

    assert result["tool"] == "impute_rolling"
    assert "n_imputed" in result
    assert "n_gaps_total" in result
    assert "gaps_summary" in result
    assert result["message"]
    # Check JSON serialisability
    json.dumps({k: v for k, v in result.items() if k not in ["qc", "df"]})


def test_all_detection_tools_return_json():
    qc = _toy_qc()
    tools_to_test = [
        (wrappers.flag_range, {"min": 0, "max": 100}),
        (wrappers.flag_constants, {"thresh": 0.0, "window": "1h"}),
        (wrappers.flag_plateau, {"max_length": "2h"}),
        (wrappers.flag_spike_unilof, {"n": 20}),
        (wrappers.flag_zscore, {"window": "2h"}),
        (wrappers.flag_jumps, {"thresh": 1.0, "window": "1h"}),
        (wrappers.flag_nan, {}),
    ]
    for func, kwargs in tools_to_test:
        result = func(qc, field="value", **kwargs)
        assert result["tool"] == func.__name__
        assert "n_flagged" in result
        # Check JSON serialisability
        json.dumps({k: v for k, v in result.items() if k not in ["qc", "df"]})


def test_flag_plateau_survives_its_own_crash():
    """flagPlateau raises on ordinary input (§7.1); the wrapper must absorb that.

    A 100-row toy series is short enough to trip SaQC's window-shape error. The
    caller should get a normal result saying nothing was flagged and why, not an
    exception that ends the agent's run.
    """
    qc = _toy_qc()
    result = wrappers.flag_plateau(qc, field="value", min_length="1h", max_length="2h")

    assert result["tool"] == "flag_plateau"
    assert result["n_flagged"] == 0
    assert result.get("failed") is True
    assert "flagged nothing" in result["message"]
    assert result["qc"] is qc          # unchanged — nothing was flagged
    json.dumps({k: v for k, v in result.items() if k not in ["qc", "df"]})


# ---------------------------------------------------------------------------
# Re-running a detector with different parameters (CLAUDE.md §7.1)
# ---------------------------------------------------------------------------

def _rerun_qc() -> saqc.SaQC:
    """A calm series with three obvious spikes, big enough for LOF to need a window."""
    idx = pd.date_range("2024-01-01", periods=600, freq="15min")
    rng = np.random.default_rng(0)
    v = 10 + rng.normal(0, 0.2, 600)
    for i in (100, 250, 400):
        v[i] = 30.0
    return saqc.SaQC(pd.DataFrame({"value": v}, index=idx))


def test_rerun_reports_only_newly_flagged_rows_and_a_running_total():
    """A second call at a looser threshold counts what it ADDED, not its whole result.

    SaQC never re-flags a row an earlier test already flagged, so ``n_flagged`` is
    per-call. ``n_flagged_total`` is what the agent needs to judge the overall share.
    """
    strict = wrappers.flag_spike_unilof(_rerun_qc(), field="value", thresh=2.0)
    assert strict["n_flagged"] == strict["n_flagged_total"]  # nothing flagged before it

    loose = wrappers.flag_spike_unilof(strict["qc"], field="value", thresh=1.1)
    assert loose["n_flagged_total"] >= strict["n_flagged_total"]
    assert loose["n_flagged"] == loose["n_flagged_total"] - strict["n_flagged_total"]
    # The message has to say so, or the agent reads a re-run count as the total.
    if loose["n_flagged"] != loose["n_flagged_total"]:
        assert "in total" in loose["message"]


def test_a_stricter_rerun_cannot_take_flags_back():
    """Detection is additive: tightening a threshold afterwards un-flags nothing.

    This is why the agent must start strict and loosen (§7.1) — and why an
    over-flagged segment has to be handled with a 'keep' decision, not a re-run.
    """
    loose = wrappers.flag_spike_unilof(_rerun_qc(), field="value", thresh=1.1)
    strict = wrappers.flag_spike_unilof(loose["qc"], field="value", thresh=3.0)

    assert strict["n_flagged"] == 0
    assert strict["n_flagged_total"] == loose["n_flagged_total"]
    assert int(wrappers.export_clean_data(strict["qc"])["df"]["flag"].notna().sum()) == (
        loose["n_flagged_total"]
    )


# ---------------------------------------------------------------------------
# Context tools (CLAUDE.md §7.3)
# ---------------------------------------------------------------------------

def test_describe_point_returns_json_and_leaves_qc_untouched():
    qc = _toy_qc()
    at = qc.data.to_pandas().index[50]          # the injected spike
    result = wrappers.describe_point(qc, at=str(at), field="value")

    assert result["tool"] == "describe_point"
    assert result["reads_like"] == "spike"
    assert "message" in result
    for block in ("slope", "excursion", "recovery", "level_shift",
                  "flatness", "neighbourhood", "gap", "history"):
        assert block in result
    # An observing tool returns no SaQC object, so the agent's state cannot drift.
    assert "qc" not in result
    json.dumps(result)


def test_describe_points_describes_a_detector_output():
    qc = _toy_qc()
    idx = qc.data.to_pandas().index
    result = wrappers.describe_points(
        qc, ats=[str(idx[50]), str(idx[82]), str(idx[10])], field="value"
    )

    assert result["tool"] == "describe_points"
    assert result["n_described"] == 3
    assert {row["reads_like"] for row in result["points"]} >= {"spike", "gap"}
    assert "qc" not in result
    json.dumps(result)


def test_describe_points_truncates_loudly():
    qc = _toy_qc()
    idx = qc.data.to_pandas().index
    result = wrappers.describe_points(
        qc, ats=[str(t) for t in idx[:10]], field="value", max_points=3
    )

    assert result["n_described"] == 3
    assert result["n_truncated"] == 7
    assert "max_points" in result["message"]


def test_describe_point_rejects_a_timestamp_outside_the_series():
    qc = _toy_qc()
    with pytest.raises(ValueError):
        wrappers.describe_point(qc, at="1999-01-01T00:00:00", field="value")
