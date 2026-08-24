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


@pytest.fixture
def noisy_stretch_series() -> pd.Series:
    """Calm everywhere except rows 200-260, where the sensor thrashes.

    The stretch carries no anomaly at all — it is the same water, measured badly —
    so every detector hit inside it is a false positive. This is the fixture for
    the failure noise_context exists to prevent: judged one point at a time
    against the record's scale, each of these rows looks extreme.
    """
    rng = np.random.default_rng(7)
    s = _calm()
    s.iloc[200:260] += rng.normal(0, 1.0, 60)      # 20x the baseline noise sd
    return s


@pytest.fixture
def accelerating_ramp_series() -> pd.Series:
    """A smooth, quickening rise — real water moving fast, not a noisy sensor.

    Variable in the same sample-to-sample sense as the noisy stretch, but the
    moves are all in one direction, which is what ``variation_kind`` separates.
    """
    s = _calm()
    s.iloc[200:260] += np.linspace(0, 1, 60) ** 2 * 40
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


def test_noise_context_calm_stretch_leaves_a_spike_standing_out(spike_series):
    """In calm surroundings the spike is exactly what it looks like."""
    r = ctx.noise_context(spike_series, spike_series.index[200])
    assert r["noise_ratio"] < 3                       # its neighbours are quiet
    assert r["point_step_sigmas_local"] > 5           # it is not
    assert r["point_unremarkable_here"] is False
    assert r["noise_regime"] in ("quiet", "typical")


def test_noise_context_flags_a_point_that_is_typical_for_its_noisy_stretch(
    noisy_stretch_series,
):
    """The whole point of the tool: this row is extreme by the RECORD's scale and
    unremarkable by its own neighbourhood's."""
    r = ctx.noise_context(noisy_stretch_series, noisy_stretch_series.index[230])
    assert r["noise_ratio"] > 3
    assert r["noise_regime"] in ("elevated", "severe")
    assert r["point_step_sigmas_local"] < 5
    assert r["point_unremarkable_here"] is True
    # The two denominators must actually disagree, or the tool has added nothing.
    assert r["point_step_sigmas_global"] > 3 * r["point_step_sigmas_local"]
    assert r["variation_kind"] == "noise-like"


def test_noise_context_separates_a_fast_rise_from_a_noisy_sensor(
    accelerating_ramp_series,
):
    """Busy for a different reason: the moves are directional, so it is real water."""
    r = ctx.noise_context(accelerating_ramp_series, accelerating_ramp_series.index[240])
    assert r["variation_kind"] == "directional"
    assert r["turning_fraction"] < 0.35


def test_noise_context_bounds_the_noisy_episode(noisy_stretch_series):
    """The episode bounds are what let a run write ONE decision span (§5)."""
    r = ctx.noise_context(noisy_stretch_series, noisy_stretch_series.index[230])
    start = pd.Timestamp(r["episode_start"])
    end = pd.Timestamp(r["episode_end"])
    assert start <= noisy_stretch_series.index[230] <= end
    # Bounded by the stretch it describes, not the whole record.
    assert start >= noisy_stretch_series.index[0]
    assert end <= noisy_stretch_series.index[-1]
    assert r["episode_hours"] > 0


def test_noise_context_reports_a_quiet_record_as_quiet(spike_series):
    """No episode is reported when nothing is elevated — no false alarm."""
    r = ctx.noise_context(spike_series, spike_series.index[50])
    assert r["point_unremarkable_here"] is False
    assert r["episode_start"] is None


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


def test_describe_point_does_not_call_a_noisy_stretch_a_spike(noisy_stretch_series):
    """The guard that stops one noisy hour being reported as many sensor failures.

    Without it every one of these rows reads 'spike' — narrow, and far from the
    local median by the record's scale — and a run deletes them one by one.
    Measured on real data, the guard drops 34.6% of the wrong 'spike' hints on
    detector false positives and costs 3.0% of the right ones on true spikes
    (scratchpad/probe_noise_reads_like.py).
    """
    labels = [
        ctx.describe_point(noisy_stretch_series, noisy_stretch_series.index[i])["reads_like"]
        for i in range(215, 250)
    ]
    assert "noisy-stretch" in labels
    assert labels.count("spike") < labels.count("noisy-stretch")

    # ...and it must not disarm the spike label where the surroundings ARE calm.
    calm = ctx.describe_point(noisy_stretch_series, noisy_stretch_series.index[100])
    assert calm["reads_like"] != "noisy-stretch"


def test_describe_points_carries_the_noise_columns(noisy_stretch_series):
    """Triage is where a noisy stretch is visible — one row cannot show it."""
    stamps = [noisy_stretch_series.index[i] for i in (215, 225, 235, 245)]
    rows = ctx.describe_points(noisy_stretch_series, stamps)["points"]
    assert all("noise_ratio" in r and "step_sigmas_local" in r for r in rows)
    assert all(r["noise_ratio"] > 3 for r in rows)


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
        ctx.noise_context,
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

def test_describe_points_batches_a_hundred_by_default_and_clamps_beyond_the_ceiling():
    """The 20-point default was the binding limit on how much a run ever measured.

    On 01467200_l1 the agent measured 45 of 10,463 decided rows and deleted 367
    spikes having inspected 33 — 194 of those deletions were normal water. Cost is
    ~90 tokens/point and flat with batch size, so the default is 100. It stays
    agent-settable, but clamped: MAX_FLAGGED_DATETIMES is 1000, and an uncapped
    call would return ~90k tokens in a single tool result.
    """
    index = pd.date_range("2024-01-01", periods=1200, freq="5min")
    series = pd.Series(np.linspace(5.0, 9.0, 1200), index=index)
    ats = [str(t) for t in index]

    default = ctx.describe_points(series, ats=ats)
    assert default["n_described"] == ctx.DEFAULT_MAX_POINTS == 100
    assert default["n_truncated"] == 1200 - 100
    # Truncation is reported, never silent.
    assert "NOT described" in default["message"]

    raised = ctx.describe_points(series, ats=ats, max_points=250)
    assert raised["n_described"] == 250, "max_points must stay agent-settable"

    clamped = ctx.describe_points(series, ats=ats, max_points=5000)
    assert clamped["n_described"] == ctx.MAX_POINTS_CEILING == 300
    # The message must name the cap that was APPLIED, not the one that was asked
    # for — otherwise a clamped call reads as if 5000 points were considered.
    assert "max_points=300" in clamped["message"]
    assert "clamped from the 5000" in clamped["message"]
    assert clamped["params"]["max_points_requested"] == 5000


# ---------------------------------------------------------------------------
# jump_scale (§7.6) -- flag_jumps' threshold, measured on the series
# ---------------------------------------------------------------------------
def _stepped(n: int = 800, level: float = 10.0, step: float = 8.0) -> pd.Series:
    """Calm baseline with one sustained step half way through."""
    s = _calm(n, level=level)
    s.iloc[n // 2:] += step
    return s


def test_jump_scale_reproduces_the_statistic_flagjumps_thresholds():
    """The suggested thresh must be on the same axis as flagJumps' own decision.

    A threshold set at the p99 of the measured statistic must therefore leave
    roughly the top 1% of change points flagged -- far fewer than the series has
    rows, which a threshold guessed in data units does not guarantee.
    """
    s = _stepped()
    out = ctx.jump_scale(s)
    assert out["recommended_window"] == ctx.JUMP_RECOMMENDED_WINDOW
    thresh = out["recommended_thresh"]

    qc = saqc.SaQC(pd.DataFrame({"value": s}))
    flagged = qc.flagJumps("value", thresh=thresh, window=out["recommended_window"])
    n = int((flagged.flags["value"] > 0).sum())
    assert 0 < n < 0.05 * len(s), f"{n} flags on {len(s)} rows at the p99 threshold"


def test_jump_scale_quantiles_are_ordered_and_finite():
    out = ctx.jump_scale(_stepped())
    for window, block in out["by_window"].items():
        assert block["p50"] <= block["p90"] <= block["p99"] <= block["p99_9"], window
        assert block["suggested_thresh"] > 0, window


def test_jump_scale_scales_with_the_series_not_with_a_constant():
    """The whole point of the block: the same shape at 100x the scale gives 100x thresh.

    A fixed "1-5 FNU" suggestion cannot do this, which is why a run set thresh at
    roughly the 60th percentile of ordinary movement and flagged 2,865 rows.
    """
    small = ctx.jump_scale(_stepped(level=10.0, step=8.0))["recommended_thresh"]
    large = ctx.jump_scale(_stepped(level=1000.0, step=800.0))["recommended_thresh"]
    assert large > 10 * small


def test_jump_scale_observes_only():
    """No flags, no mutation -- it is a measurement (§7.3)."""
    qc = saqc.SaQC(pd.DataFrame({"value": _stepped()}))
    before = qc.flags["value"].copy()
    out = ctx.jump_scale(qc)
    assert "qc" not in out
    pd.testing.assert_series_equal(before, qc.flags["value"])


def test_jump_scale_survives_a_constant_series():
    flat = pd.Series(5.0, index=_index(200))
    out = ctx.jump_scale(flat)
    assert out["recommended_thresh"] in (None, 0.0)
    assert "message" in out

def test_the_slope_window_is_a_duration_so_it_survives_a_frequency_change():
    """3 samples meant 45 min at 15-min sampling and 15 min at 5-min — a 3x error.

    §7.3 measured the fall/rise ratio's optimum at 45 MINUTES and warns the signal
    REVERSES at the wrong width. When the project moved to 5-minute bases the
    sample-count default silently became a third of the intended window. Measured on
    01467200_l1 at 2023-07-09 22:50, a gradual 2.5h rise the run deleted as a spike:
    the 15-min view called it "symmetric" (ratio 0.54), the 45-min view called it
    "gradual_before" (ratio 3.09).
    """
    for freq, expected in (("15min", 3), ("5min", 9), ("1min", 45)):
        index = pd.date_range("2024-01-01", periods=400, freq=freq)
        series = pd.Series(np.linspace(1.0, 9.0, 400), index=index)
        params = ctx.slope_context(series, at=index[200])["params"]
        assert params["n_before"] == params["n_after"] == expected, freq
        assert params["window"] == ctx.SLOPE_WINDOW

    # an explicit sample count still wins, for a caller that really wants one
    index = pd.date_range("2024-01-01", periods=100, freq="5min")
    series = pd.Series(np.linspace(1.0, 9.0, 100), index=index)
    assert ctx.slope_context(series, at=index[50], n_before=4, n_after=4)["params"]["n_before"] == 4


def test_ramp_context_measures_the_climb_a_45_minute_window_cannot_see():
    """The shape above slope_context's scale: hours, not minutes."""
    index = pd.date_range("2024-01-01", periods=200, freq="5min")
    values = np.full(200, 5.0)
    values[64:100] = np.linspace(5.0, 20.0, 36)      # 3h climb
    values[100:124] = np.linspace(20.0, 5.0, 24)     # 2h recession
    series = pd.Series(values, index=index)

    ramped = ctx.ramp_context(series, at=index[99])
    assert ramped["rise"]["minutes"] >= 120, ramped["rise"]
    # A clean synthetic climb is maximally DIRECT — and directness separates the two
    # populations the opposite way to the intuition (see ramp_context's docstring): a
    # straight climb is the injected-displacement shape, a meandering one is water.
    assert ramped["rise"]["directness"] == 1.0
    assert ramped["reads_like"] == "no-ramp"

    # The invariant that matters is RELATIVE: the same net rise, approached by wandering,
    # must score lower than one approached in a straight line. (An exact threshold is not
    # asserted here — RAMP_DIRECTNESS is fitted on real gauges, not on a sine wave, and
    # the wobble also has to stay slower than the 20-min smoother and below the peak.)
    wandering = series.copy()
    wobble = 4.0 * np.sin(np.arange(36) * 2 * np.pi / 36)
    wandering.iloc[64:100] = wandering.iloc[64:100].to_numpy() + wobble
    wobbly = ctx.ramp_context(wandering, at=index[99])
    assert wobbly["rise"]["directness"] < ramped["rise"]["directness"]
    assert wobbly["rise"]["minutes"] >= 120

    # a lone spike on a flat baseline is climbing to nothing
    flat = pd.Series(np.full(200, 5.0), index=index)
    flat.iloc[100] = 40.0
    spike = ctx.ramp_context(flat, at=index[100])
    assert spike["reads_like"] == "no-ramp"
    assert "Nothing long and meandering leads into this point" in spike["message"]
    # and it still refuses to decide on its own
    assert "check width, recovery and rainfall" in spike["message"]


def test_a_shift_window_is_judged_on_its_edges_not_just_its_elevation():
    """Elevation alone cannot separate a level shift from a storm.

    Measured on 01467200_l1: of 24 windows elevated between two jumps, ONE was the
    injected shift and 23 were storms — and the largest storm scored z=+8.6 against
    the real shift's +4.5, so ranking by elevation puts them the wrong way round.
    §9.1's lever is edge sharpness: a recalibration moves in one sample, a storm ramps.
    """
    index = pd.date_range("2024-01-01", periods=600, freq="5min")

    # a true shift: instantaneous step up, holds, instantaneous step back
    shifted = np.full(600, 5.0)
    shifted[200:400] = 15.0
    step = ctx.shift_window_context(pd.Series(shifted, index=index),
                                    start=index[200], end=index[399])
    assert step["reads_like"] == "level-shift-like"
    assert step["interior_sigmas"] is not None and abs(step["interior_sigmas"]) >= 3
    assert step["onset_sharpness"] >= ctx.SHIFT_EDGE_SHARPNESS
    assert "recalibration or sensor swap" in step["message"]

    # a storm: same elevation, but it ramps in and out over hours
    storm = np.full(600, 5.0)
    storm[200:260] = np.linspace(5, 15, 60)
    storm[260:340] = 15.0
    storm[340:400] = np.linspace(15, 5, 60)
    ramped = ctx.shift_window_context(pd.Series(storm, index=index),
                                      start=index[200], end=index[399])
    assert ramped["reads_like"] == "event-like", ramped["message"]
    assert ramped["onset_sharpness"] < ctx.SHIFT_EDGE_SHARPNESS
    assert "ramps rather than steps" in ramped["message"].lower()

    # flat: nothing to report
    flat = ctx.shift_window_context(pd.Series(np.full(600, 5.0), index=index),
                                    start=index[200], end=index[399])
    assert flat["reads_like"] == "not-shifted"


def test_find_shift_windows_reports_one_event_not_thirty_sliding_ones():
    """A jump list produces many overlapping pairs; the tool must merge them."""
    index = pd.date_range("2024-01-01", periods=600, freq="5min")
    values = np.full(600, 5.0)
    values[200:400] = 15.0
    series = pd.Series(values, index=index)

    # jump timestamps as a detector would give them: a cluster at each edge
    ats = [str(index[i]) for i in (198, 199, 200, 201, 398, 399, 400, 401)]
    found = ctx.find_shift_windows(series, ats)

    assert found["n_windows"] <= 2, found["windows"]
    assert found["n_level_shift_like"] >= 1
    best = found["windows"][0]
    assert best["reads_like"] == "level-shift-like"
    # and it spans the WHOLE shifted region, which is the point of the tool
    assert pd.Timestamp(best["start"]) <= index[201]
    assert pd.Timestamp(best["end"]) >= index[398]
    assert "WHOLE span" in found["message"]
