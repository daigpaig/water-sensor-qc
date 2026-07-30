"""Tests for the point-context tools (src/agent_tools/context.py).

Each fixture builds a series whose shape is known by construction — a one-sample
spike, a stuck run, a sustained step, a gap — so the assertions test the
measurement, not a label file.
"""

import json

import numpy as np
import pandas as pd
import pytest
import saqc

from src.agent_tools import context as ctx

FREQ = "15min"


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq=FREQ)


def _calm(n: int = 400, level: float = 10.0, noise: float = 0.05, seed: int = 0) -> pd.Series:
    """A calm baseline: small Gaussian noise around a constant level."""
    rng = np.random.default_rng(seed)
    return pd.Series(level + rng.normal(0, noise, n), index=_index(n))


@pytest.fixture
def spike_series() -> pd.Series:
    s = _calm()
    s.iloc[200] = 40.0          # one sample, far above; neighbours untouched
    return s


@pytest.fixture
def storm_series() -> pd.Series:
    """A broad event: 4 h rise, 4 h at the top, ~20 h recession."""
    s = _calm()
    rise = np.linspace(0, 30, 16)
    fall = np.linspace(30, 0, 80)
    s.iloc[150:166] += rise
    s.iloc[166:182] += 30
    s.iloc[182:262] += fall
    return s


@pytest.fixture
def stuck_series() -> pd.Series:
    s = _calm()
    s.iloc[200:240] = 10.0      # 40 identical samples
    return s


@pytest.fixture
def shift_series() -> pd.Series:
    s = _calm()
    s.iloc[200:] += 5.0         # permanent step, ~100x the noise sd
    return s


@pytest.fixture
def gap_series() -> pd.Series:
    s = _calm()
    s.iloc[200:210] = np.nan
    return s


# ---------------------------------------------------------------------------
# Input plumbing
# ---------------------------------------------------------------------------
def test_as_series_accepts_saqc_frame_and_series(spike_series):
    qc = saqc.SaQC(pd.DataFrame({"value": spike_series}))
    frame = spike_series.rename("value").rename_axis("datetime").reset_index()

    from_qc = ctx.as_series(qc, "value")
    from_frame = ctx.as_series(frame, "value")
    from_series = ctx.as_series(spike_series)

    # check_freq=False: round-tripping through a column drops the index freq attr.
    pd.testing.assert_series_equal(from_qc, from_series, check_names=False, check_freq=False)
    pd.testing.assert_series_equal(from_frame, from_series, check_names=False, check_freq=False)


def test_as_series_sorts_an_unsorted_index(spike_series):
    shuffled = spike_series.sample(frac=1.0, random_state=1)
    assert ctx.as_series(shuffled).index.is_monotonic_increasing


def test_resolve_timestamp_exact_nearest_and_out_of_range(spike_series):
    exact = spike_series.index[200]
    assert ctx.resolve_timestamp(spike_series, exact) == exact
    # Within half a step, snap to the sample.
    assert ctx.resolve_timestamp(spike_series, exact + pd.Timedelta("5min")) == exact
    # Beyond the series: fail loudly rather than silently clamp (§13).
    with pytest.raises(ValueError, match="not in the series"):
        ctx.resolve_timestamp(spike_series, "2025-06-01")


# ---------------------------------------------------------------------------
# Individual contexts
# ---------------------------------------------------------------------------
def test_slope_context_reads_a_peak(storm_series):
    peak = storm_series.index[170]
    r = ctx.slope_context(storm_series, peak, n_before=16, n_after=16)
    assert r["slope_before_per_hour"] > 0
    assert r["slope_after_per_hour"] < 0
    assert r["shape"] == "peak"
    assert r["fall_rise_ratio"] is not None


def test_slope_context_single_sample_steps(spike_series):
    at = spike_series.index[200]
    r = ctx.slope_context(spike_series, at)
    # The spike is ~30 units above a 0.05-sd baseline, up then straight back down.
    assert r["delta_before"] > 25
    assert r["delta_after"] < -25
    assert abs(r["delta_before_sigmas"]) > 10


def test_excursion_context_isolates_a_one_sample_spike(spike_series):
    r = ctx.excursion_context(spike_series, spike_series.index[200])
    assert r["n_samples"] == 1
    assert r["isolated"] is True
    assert r["direction"] == "up"
    assert r["excursion"] == pytest.approx(30, abs=1)
    # Essentially the whole excursion happens in one sample.
    assert r["peak_sharpness"] > 0.9


def test_excursion_context_measures_a_broad_event(storm_series):
    r = ctx.excursion_context(storm_series, storm_series.index[170])
    assert r["n_samples"] > 20            # hours wide, not a point
    assert r["isolated"] is False
    assert r["peak_sharpness"] < 0.5      # built gradually


def test_recovery_context_spike_recovers_immediately(spike_series):
    r = ctx.recovery_context(spike_series, spike_series.index[200])
    assert r["recovered"] is True
    assert r["n_samples_to_recover"] <= 2


def test_recovery_context_step_never_recovers(shift_series):
    r = ctx.recovery_context(shift_series, shift_series.index[200])
    assert r["recovered"] is False
    assert r["n_samples_to_recover"] is None


def test_recovery_context_anchors_the_baseline_before_the_event(storm_series):
    """Mid-event, a point-anchored baseline is already elevated; the excursion
    anchor reaches back to the pre-event level and reports the real recovery."""
    mid = storm_series.index[175]
    anchored = ctx.recovery_context(storm_series, mid, anchor="excursion")
    naive = ctx.recovery_context(storm_series, mid, anchor="point")
    assert anchored["baseline"] < naive["baseline"]
    assert anchored["n_samples_to_recover"] > naive["n_samples_to_recover"]


def test_level_shift_context_detects_a_sustained_step(shift_series):
    r = ctx.level_shift_context(shift_series, shift_series.index[200])
    assert r["step"] == pytest.approx(5, abs=0.5)
    assert r["step_sigmas"] > 3
    assert r["hold_minutes"] >= 6 * 60
    assert r["holds_to_end_of_record"] is True


def test_level_shift_context_flat_series_has_no_step(spike_series):
    r = ctx.level_shift_context(spike_series, spike_series.index[100])
    assert abs(r["step_sigmas"]) < 3


def test_flatness_context_finds_the_stuck_run(stuck_series):
    r = ctx.flatness_context(stuck_series, stuck_series.index[220], window="6h")
    assert r["run_length_at_point"] >= 20
    assert r["longest_run"] >= 20
    assert r["frac_unchanged"] > 0.5


def test_flatness_context_quiet_noise_is_not_flat(spike_series):
    r = ctx.flatness_context(spike_series, spike_series.index[100])
    assert r["run_length_at_point"] == 1


def test_neighbourhood_stats_excludes_the_point_from_its_own_reference(spike_series):
    r = ctx.neighbourhood_stats(spike_series, spike_series.index[200])
    assert r["local_median"] == pytest.approx(10, abs=0.2)   # unmoved by the spike
    assert r["robust_z"] > 10
    assert r["percentile_in_window"] == 100.0


def test_gap_context_inside_and_beside_a_gap(gap_series):
    inside = ctx.gap_context(gap_series, gap_series.index[205])
    assert inside["is_missing"] is True
    assert inside["enclosing_gap"]["n_rows"] == 10

    beside = ctx.gap_context(gap_series, gap_series.index[210])
    assert beside["is_missing"] is False
    assert beside["adjacent_to_gap"] is True
    assert beside["samples_to_previous_gap"] == 1

    far = ctx.gap_context(gap_series, gap_series.index[100])
    assert far["adjacent_to_gap"] is False


def test_historical_context_counts_episodes_not_rows(spike_series):
    # The spike level is unique in the record; the baseline level is not.
    unique = ctx.historical_context(spike_series, spike_series.index[200])
    assert unique["n_episodes_at_or_beyond"] == 0
    assert unique["percentile_in_record"] == 100.0

    common = ctx.historical_context(spike_series, spike_series.index[100])
    assert common["n_episodes_at_or_beyond"] > 5


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fixture_name, position, expected",
    [
        ("spike_series", 200, "spike"),
        ("stuck_series", 220, "plateau/stuck"),
        ("gap_series", 205, "gap"),
        ("shift_series", 200, "level_shift"),
    ],
)
def test_describe_point_reads_like(request, fixture_name, position, expected):
    series = request.getfixturevalue(fixture_name)
    r = ctx.describe_point(series, series.index[position])
    assert r["reads_like"] == expected
    assert r["reads_like_reason"]


def test_describe_point_is_json_serialisable(spike_series):
    r = ctx.describe_point(spike_series, spike_series.index[200])
    json.dumps(r)          # raises if any numpy scalar or Timestamp leaked through


def test_every_context_keeps_its_key_set_on_a_nan_point(gap_series):
    """A NaN point must not return a different-shaped dict from a valid one."""
    nan_at = gap_series.index[205]
    ok_at = gap_series.index[100]
    for fn in (
        ctx.slope_context,
        ctx.excursion_context,
        ctx.recovery_context,
        ctx.level_shift_context,
        ctx.flatness_context,
        ctx.neighbourhood_stats,
        ctx.gap_context,
        ctx.historical_context,
    ):
        assert set(fn(gap_series, nan_at)) == set(fn(gap_series, ok_at)), fn.__name__


def test_describe_points_tallies_and_reports_truncation(spike_series):
    stamps = [spike_series.index[i] for i in (100, 150, 200, 250, 300)]
    r = ctx.describe_points(spike_series, stamps, max_points=3)

    assert r["n_described"] == 3
    assert r["n_truncated"] == 2
    assert "NOT described" in r["message"]      # never a silent cap
    assert sum(r["tally"].values()) == 3
    json.dumps(r)


def test_describe_points_records_unresolvable_timestamps(spike_series):
    r = ctx.describe_points(spike_series, [spike_series.index[200], "2030-01-01"])
    assert r["n_described"] == 1
    assert len(r["errors"]) == 1
    assert "2030" in r["errors"][0]["at"]


def test_describe_points_accepts_a_saqc_object(spike_series):
    qc = saqc.SaQC(pd.DataFrame({"value": spike_series}))
    r = ctx.describe_points(qc, [spike_series.index[200]], field="value")
    assert r["points"][0]["reads_like"] == "spike"
