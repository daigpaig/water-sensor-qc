"""Rainfall as outside evidence (src/agent_tools/precipitation.py).

The dangerous failure here is silence: if precipitation data is missing and the tool
answers "no rain", that reads as evidence FOR deleting a point. Most of these tests
are about that distinction.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.agent_tools import precipitation as precip


@pytest.fixture
def gauge_dir(tmp_path):
    """A two-station precipitation set: one near but short, one far but complete."""
    root = tmp_path / "precip"
    d = root / "TESTGAUGE"
    d.mkdir(parents=True)

    far = pd.date_range("2024-01-01", periods=24 * 12, freq="1h")
    rain = pd.Series(0.0, index=far)
    rain.loc["2024-01-05 00:00":"2024-01-05 06:00"] = 0.2      # a storm
    pd.DataFrame({"datetime": far, "precip_in": rain.to_numpy()}).to_csv(
        d / "FAR.csv", index=False)

    near = pd.date_range("2024-01-08", periods=48, freq="1h")
    pd.DataFrame({"datetime": near, "precip_in": 0.0}).to_csv(d / "NEAR.csv", index=False)

    (d / "manifest.json").write_text(json.dumps({
        "gauge": "TESTGAUGE",
        "stations": [
            {"site": "NEAR", "name": "near but short", "km": 5.0, "file": "NEAR.csv"},
            {"site": "FAR", "name": "far but complete", "km": 30.0, "file": "FAR.csv"},
        ],
    }))
    precip._load.cache_clear()
    return root


def test_rain_before_a_point_is_reported_as_a_physical_cause(gauge_dir):
    r = precip.precip_context(None, "2024-01-05 08:00", gauge="TESTGAUGE", root=gauge_dir)
    assert r["available"] and r["rained"]
    assert r["total_before_12h_in"] > 0
    assert "physical cause" in r["message"]
    # the lag windows are reported separately, because rain leads turbidity
    assert set(r["rain_before_in"]) == {"0-1h", "1-3h", "3-12h"}


def test_absence_of_rain_is_reported_as_WEAK_evidence(gauge_dir):
    """A storm cell can miss a station 30 km away, so 'dry' must not read as proof."""
    r = precip.precip_context(None, "2024-01-02 12:00", gauge="TESTGAUGE", root=gauge_dir)
    assert r["available"] and not r["rained"]
    assert "WEAK evidence" in r["message"]
    assert "does not license a deletion" in r["message"]
    assert r["distance_km"] == 30.0          # and it says how far away it was


def test_missing_data_is_never_reported_as_dry_weather(tmp_path):
    """The failure that would matter most: silence reading as evidence to delete."""
    precip._load.cache_clear()
    r = precip.precip_context(None, "2024-01-05 08:00", gauge="NOSUCH", root=tmp_path)
    assert r["available"] is False
    assert "rained" not in r
    assert "NOT the same as 'it did not rain'" in r["message"]
    assert "pull_precip" in r["message"]


def test_a_timestamp_no_station_covers_says_so(gauge_dir):
    r = precip.precip_context(None, "2030-06-01 00:00", gauge="TESTGAUGE", root=gauge_dir)
    assert r["available"] is False
    assert "not evidence of dry weather" in r["message"]


def test_the_nearest_station_with_data_answers(gauge_dir):
    """Nearest-first, but a station with no data at that time must not veto a further one."""
    covered_by_near = precip.precip_context(
        None, "2024-01-08 12:00", gauge="TESTGAUGE", root=gauge_dir)
    assert covered_by_near["station"] == "NEAR"
    # NEAR has no data in January 5; the answer must fall through to FAR, not go blank
    only_far = precip.precip_context(None, "2024-01-05 08:00", gauge="TESTGAUGE", root=gauge_dir)
    assert only_far["station"] == "FAR" and only_far["rained"]


def test_the_batch_form_tallies_and_reports_truncation(gauge_dir):
    ats = ["2024-01-05 08:00", "2024-01-02 12:00", "2030-06-01 00:00"]
    r = precip.precip_context_points(None, ats, gauge="TESTGAUGE", root=gauge_dir)
    assert r["n_checked"] == 3
    assert r["tally"] == {"rained": 1, "dry": 1, "unknown": 1}
    assert "could not be checked" in r["message"]

    capped = precip.precip_context_points(None, ats, gauge="TESTGAUGE",
                                          root=gauge_dir, max_points=2)
    assert capped["n_truncated"] == 1
    assert "were NOT checked" in capped["message"]
