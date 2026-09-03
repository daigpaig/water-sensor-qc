"""Tests for the turbidity-vs-rainfall page (`src/workbench/precip_overlay.py`).

Everything asserted here is assembled in Python rather than in JS, which is the
reason the payload is built that way at all (§10.1). The traps guarded:

* coverage is measured from the timestamps, so "no data" can never be rendered as
  "no rain" (§7.7) -- the most dangerous failure in this part of the project;
* dropping zero-rain rows from the bars is lossless, so it cannot quietly change
  a lag-window sum;
* timestamps leave Python as naive ISO strings, never Date-parseable objects with
  a timezone (§9.1).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.workbench import precip_overlay as po


@pytest.fixture
def series() -> pd.Series:
    idx = pd.date_range("2024-05-01", periods=600, freq="5min")
    return pd.Series(np.linspace(2.0, 8.0, 600), index=idx)


def _write_station(root: Path, gauge: str, site: str, km: float,
                   index: pd.DatetimeIndex, values) -> None:
    d = root / gauge
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"datetime": index, "precip_in": values}).to_csv(d / f"{site}.csv", index=False)
    manifest_path = d / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "gauge": gauge, "stations": [], "note": "test manifest"}
    manifest["stations"].append({"site": site, "name": f"station {site}",
                                 "km": km, "file": f"{site}.csv"})
    manifest_path.write_text(json.dumps(manifest))


def test_coverage_intervals_split_on_a_real_break():
    """A station that goes quiet for hours reports two intervals, not one."""
    a = pd.date_range("2024-05-01 00:00", periods=20, freq="15min")
    b = pd.date_range("2024-05-01 12:00", periods=20, freq="15min")
    intervals = po.coverage_intervals(pd.DatetimeIndex(a.append(b)))
    assert len(intervals) == 2
    assert intervals[0] == ["2024-05-01 00:00:00", "2024-05-01 04:45:00"]
    assert intervals[1][0] == "2024-05-01 12:00:00"


def test_coverage_intervals_are_one_run_when_the_record_is_dense():
    idx = pd.date_range("2024-05-01", periods=100, freq="5min")
    assert len(po.coverage_intervals(idx)) == 1


def test_coverage_gap_is_judged_against_the_station_own_cadence():
    """A 15-min bucket and a 5-min met station must not share one gap constant."""
    # Each run ends well before the next begins: the coarse station is silent
    # from 02:15 to 06:00, the fine one from 00:45 to 06:00.
    coarse = pd.DatetimeIndex(
        pd.date_range("2024-05-01 00:00", periods=10, freq="15min").append(
            pd.date_range("2024-05-01 06:00", periods=10, freq="15min")))
    fine = pd.DatetimeIndex(
        pd.date_range("2024-05-01 00:00", periods=10, freq="5min").append(
            pd.date_range("2024-05-01 06:00", periods=10, freq="5min")))
    # A multi-hour absence is a break at either cadence.
    assert len(po.coverage_intervals(coarse)) == 2
    assert len(po.coverage_intervals(fine)) == 2

    # A 40-min absence is not: it is under 6 steps for the 15-min bucket (90 min)
    # but well over 6 steps for the 5-min station (30 min). A shared constant
    # would have to call it the same way for both, and would be wrong for one.
    coarse_near = pd.DatetimeIndex(
        pd.date_range("2024-05-01 00:00", periods=10, freq="15min").append(
            pd.date_range("2024-05-01 02:55", periods=10, freq="15min")))
    fine_near = pd.DatetimeIndex(
        pd.date_range("2024-05-01 00:00", periods=10, freq="5min").append(
            pd.date_range("2024-05-01 01:25", periods=10, freq="5min")))
    assert len(po.coverage_intervals(coarse_near)) == 1
    assert len(po.coverage_intervals(fine_near)) == 2


def test_station_payload_keeps_only_wet_bars_but_full_coverage(tmp_path):
    """Dropping zeros is what makes full-resolution rain affordable; it must be
    lossless for both the plot and the sums, and must not shrink coverage."""
    idx = pd.date_range("2024-05-01", periods=288, freq="5min")
    values = np.zeros(288)
    values[[10, 11, 50]] = [0.01, 0.02, 0.05]
    station = {"site": "S1", "name": "n", "km": 4.0, "note": "",
               "series": pd.Series(values, index=idx)}
    out = po._station_payload(station)
    assert len(out["bar_x"]) == 3
    assert pytest.approx(sum(out["bar_y"])) == 0.08
    assert pytest.approx(out["total_in"]) == 0.08
    # Coverage comes from the FULL index, so the quiet stretches stay covered.
    assert out["coverage"] == [["2024-05-01 00:00:00", "2024-05-01 23:55:00"]]


def test_build_payload_emits_naive_iso_strings(series):
    """Plotly renders a Date in the viewer's timezone (§9.1); only naive strings
    survive the trip to the page unchanged."""
    payload = po.build_payload(series, None, "01467200", "t", [], None)
    for stamp in payload["series_x"][:5]:
        assert isinstance(stamp, str)
        assert "T" not in stamp and "Z" not in stamp and "+" not in stamp
        assert pd.Timestamp(stamp).tz is None


def test_build_payload_marks_nan_values_as_null_not_nan(series):
    """`json.dumps` emits bare NaN, which is not valid JSON and breaks the page."""
    holed = series.copy()
    holed.iloc[100:110] = np.nan
    payload = po.build_payload(holed, None, "g", "t", [], None)
    assert None in payload["series_y"]
    json.loads(json.dumps(payload))          # would raise on a bare NaN


def test_build_payload_marks_gaps_at_true_value_and_skips_natural_ones(series, tmp_path):
    """§5: only an injected gap has a true_value, so a natural gap has nowhere to
    draw and must be absent rather than drawn at zero."""
    holed = series.copy()
    holed.iloc[10:13] = np.nan               # injected gap, true_value known
    holed.iloc[20:23] = np.nan               # natural gap, true_value blank
    labels = pd.DataFrame({
        "is_anomaly": False, "anomaly_type": "", "true_value": series.to_numpy(),
        "source": "",
    }, index=series.index)
    labels.iloc[10:13, labels.columns.get_loc("anomaly_type")] = "gap"
    labels.iloc[20:23, labels.columns.get_loc("anomaly_type")] = "gap"
    labels.iloc[20:23, labels.columns.get_loc("true_value")] = np.nan
    payload = po.build_payload(holed, labels, "g", "t", [], None)
    assert len(payload["anomalies"]["gap"]["x"]) == 3
    assert payload["anomalies"]["gap"]["x"][0] == "2024-05-01 00:50:00"


def test_load_stations_returns_nearest_first(tmp_path):
    idx = pd.date_range("2024-05-01", periods=50, freq="15min")
    _write_station(tmp_path, "G", "far", 30.0, idx, np.zeros(50))
    _write_station(tmp_path, "G", "near", 5.0, idx, np.zeros(50))
    stations, manifest = po.load_stations("G", tmp_path)
    assert [s["site"] for s in stations] == ["near", "far"]
    assert manifest["note"] == "test manifest"


def test_load_stations_is_empty_when_nothing_was_pulled(tmp_path):
    """The page must state the absence, so this returns empty rather than raising."""
    assert po.load_stations("02054550", tmp_path) == ([], None)


def test_page_states_missing_precip_rather_than_rendering_an_empty_panel(series):
    """The §7.7 failure this guards is 'no data' reading as 'no rain'."""
    html = po.build_html(po.build_payload(series, None, "02054550", "t", [], None))
    assert "NOT the same as" in html
    assert "pull_precip" in html


def test_html_is_self_contained_and_carries_the_payload(series, tmp_path):
    idx = pd.date_range("2024-05-01", periods=600, freq="5min")
    rain = np.zeros(600)
    rain[[5, 6]] = 0.03
    _write_station(tmp_path, "G", "S1", 6.5, idx, rain)
    stations, manifest = po.load_stations("G", tmp_path)
    html = po.build_html(po.build_payload(series, None, "G", "t", stations, manifest))
    # plotly.js is inlined, not fetched: no <script src=...> anywhere on the page.
    # (Do not grep for "cdn.plot.ly" -- the bundle carries that string itself, as
    # the default value of its own topojson config.)
    assert "<script src" not in html
    assert "Plotly.newPlot" in html
    assert "/*PAYLOAD*/" not in html            # the placeholder was substituted
    assert '"site": "S1"' in html or '"site":"S1"' in html


def test_resolve_target_maps_a_gauge_id_to_its_dataset(tmp_path):
    name = "01467200_l2"
    d = tmp_path / "01467200" / "l2"
    d.mkdir(parents=True)
    (d / f"{name}.csv").write_text("datetime,value\n2024-05-01 00:00:00,1.0\n")
    path, gauge = po.resolve_target("01467200", 2, tmp_path)
    assert path.name == f"{name}.csv"
    assert gauge == "01467200"


def test_resolve_target_infers_the_gauge_from_a_file_stem(tmp_path):
    p = tmp_path / "040851385_turbidity_63680.csv"
    p.write_text("datetime,value\n2024-05-01 00:00:00,1.0\n")
    _, gauge = po.resolve_target(str(p), 1, tmp_path)
    assert gauge == "040851385"
