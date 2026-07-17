"""Tests for synthetic anomaly injection (CLAUDE.md Phase 1 gate).

The gate asks that injected anomalies be recoverable from the label file, which
is the central property here: restoring ``true_value`` at every ``is_anomaly``
row must reproduce the clean base exactly.

These tests deliberately assert *properties* rather than the specific numbers in
``LEVELS`` — the level constants are tuning knobs, and pinning them would mean a
test churn every time we adjust contamination.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.inject import (
    DRIFT,
    GAP,
    LEVEL_SHIFT,
    LEVELS,
    PLATEAU,
    SOURCE_INJECTED,
    SOURCE_NATURAL,
    SPIKE,
    inject_drift,
    inject_gap,
    inject_level_shift,
    inject_plateau,
    inject_series,
    inject_spike,
    write_result,
)
from src.inspect_data import DATETIME_COL, ContractError, validate_labels_frame

STEP = pd.Timedelta(minutes=15)


def make_base(days: float = 240.0, *, seed: int = 0, gaps: bool = True) -> pd.DataFrame:
    """A synthetic clean base: smooth seasonal signal + noise, on a 15-min grid.

    Long enough to fit the drift episodes every level asks for. ``gaps`` punches
    two natural NaN runs, mimicking the small dropouts the real clean bases carry.
    """
    n = int(days * 24 * 4)
    idx = pd.date_range("2024-01-01", periods=n, freq=STEP)
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    signal = 20.0 + 5.0 * np.sin(2 * np.pi * t / (4 * 24 * 30)) + rng.normal(0, 1.0, n)
    values = np.maximum(signal, 0.1)
    if gaps:
        values[1000:1050] = np.nan
        values[5000:5008] = np.nan
    return pd.DataFrame({DATETIME_COL: idx, "value": values})


@pytest.fixture(scope="module")
def base() -> pd.DataFrame:
    return make_base()


# ---------------------------------------------------------------------------
# The five injectors, in isolation
# ---------------------------------------------------------------------------
def test_spike_offsets_only_its_window() -> None:
    values = np.full(100, 10.0)
    out, seg = inject_spike(values, 50, magnitude=25.0, length=2)

    assert seg.anomaly_type == SPIKE
    assert (seg.start_idx, seg.end_idx) == (50, 52)
    assert out[50] == out[51] == 35.0
    # Everything outside the window is untouched.
    assert np.array_equal(np.delete(out, [50, 51]), np.delete(values, [50, 51]))


def test_spike_clamps_at_the_physical_floor() -> None:
    values = np.full(10, 3.0)
    out, _ = inject_spike(values, 5, magnitude=-99.0)
    # Turbidity cannot go negative, so a large downward spike saturates at 0.
    assert out[5] == 0.0


def test_plateau_freezes_at_the_reading_it_stuck_on() -> None:
    values = np.arange(100, dtype=float)
    out, seg = inject_plateau(values, 40, length=10)

    assert seg.anomaly_type == PLATEAU
    assert np.all(out[40:50] == 40.0)
    assert out[50] == 50.0  # released


def test_plateau_refuses_to_stick_at_nan() -> None:
    values = np.arange(20, dtype=float)
    values[5] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        inject_plateau(values, 5, length=3)


def test_level_shift_returns_to_true_level_after_its_window() -> None:
    values = np.full(100, 10.0)
    out, seg = inject_level_shift(values, 30, length=20, magnitude=5.0)

    assert seg.anomaly_type == LEVEL_SHIFT
    assert np.all(out[30:50] == 15.0)
    assert np.all(out[50:] == 10.0)  # bounded, not permanent
    assert np.all(out[:30] == 10.0)


def test_gap_blanks_only_its_window() -> None:
    values = np.full(50, 7.0)
    out, seg = inject_gap(values, 10, length=5)

    assert seg.anomaly_type == GAP
    assert np.all(np.isnan(out[10:15]))
    assert np.isfinite(out[:10]).all() and np.isfinite(out[15:]).all()


def test_drift_ramps_from_zero_to_full_offset() -> None:
    values = np.full(100, 10.0)
    out, seg = inject_drift(values, 0, length=100, magnitude=8.0, model="linear")

    assert seg.anomaly_type == DRIFT
    assert out[0] == pytest.approx(10.0)   # no offset at the start of the episode
    assert out[-1] == pytest.approx(18.0)  # full offset where maintenance lands
    # A linear ramp is monotone in between.
    assert np.all(np.diff(out) > 0)


def test_exponential_drift_lags_the_linear_ramp_early() -> None:
    values = np.full(100, 10.0)
    lin, _ = inject_drift(values, 0, length=100, magnitude=8.0, model="linear")
    exp, _ = inject_drift(values, 0, length=100, magnitude=8.0, model="exponential")

    # Fouling creeps slowly then accelerates, so it sits below linear mid-episode
    # but reaches the same offset at the maintenance reset.
    assert exp[50] < lin[50]
    assert exp[-1] == pytest.approx(lin[-1])


def test_drift_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="linear|exponential"):
        inject_drift(np.full(10, 1.0), 0, length=10, magnitude=1.0, model="quadratic")


@pytest.mark.parametrize(
    "call",
    [
        lambda v: inject_spike(v, 95, magnitude=1.0, length=10),
        lambda v: inject_plateau(v, 95, length=10),
        lambda v: inject_gap(v, 98, length=5),
        lambda v: inject_drift(v, 0, length=200, magnitude=1.0),
    ],
)
def test_injectors_reject_windows_that_run_off_the_end(call) -> None:
    with pytest.raises(ValueError, match="does not fit"):
        call(np.full(100, 1.0))


@pytest.mark.parametrize(
    "call",
    [
        lambda v: inject_spike(v, 5, magnitude=0.0),
        lambda v: inject_level_shift(v, 5, length=2, magnitude=0.0),
        lambda v: inject_drift(v, 5, length=2, magnitude=0.0),
    ],
)
def test_injectors_reject_zero_magnitude(call) -> None:
    with pytest.raises(ValueError, match="magnitude"):
        call(np.full(100, 1.0))


# ---------------------------------------------------------------------------
# The Phase 1 gate: anomalies are recoverable from the labels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("level", sorted(LEVELS))
def test_true_value_restores_the_clean_base_exactly(base: pd.DataFrame, level: int) -> None:
    result = inject_series(base, level=level, seed=1, name="t")

    restored = result.data["value"].to_numpy().copy()
    anomalous = result.labels["is_anomaly"].to_numpy()
    restored[anomalous] = result.labels["true_value"].to_numpy()[anomalous]

    np.testing.assert_allclose(
        restored, base["value"].to_numpy(), equal_nan=True, rtol=0, atol=0
    )


@pytest.mark.parametrize("level", sorted(LEVELS))
def test_every_modified_row_is_labelled(base: pd.DataFrame, level: int) -> None:
    """No silent contamination: anything we changed carries a label.

    The converse does not hold and should not be asserted — a drift episode's
    first row has zero offset, and a plateau's first row is the value it stuck
    on, so some labelled rows legitimately equal the base.
    """
    result = inject_series(base, level=level, seed=1, name="t")

    injected = result.data["value"].to_numpy()
    original = base["value"].to_numpy()
    changed = ~(
        (injected == original) | (np.isnan(injected) & np.isnan(original))
    )
    assert result.labels["is_anomaly"].to_numpy()[changed].all()


def test_all_five_types_appear_at_every_level(base: pd.DataFrame) -> None:
    # Per-type recall (§10) is undefined for a type with no positives, so every
    # level must produce every type on a usable base.
    for level in sorted(LEVELS):
        result = inject_series(base, level=level, seed=3, name="t")
        present = set(result.labels.loc[result.labels["is_anomaly"], "anomaly_type"])
        assert present == {SPIKE, PLATEAU, LEVEL_SHIFT, GAP, DRIFT}
        assert not result.manifest["types_missing"]
        assert result.manifest["scoreable"]


def test_a_base_that_cannot_host_a_type_is_marked_unscoreable() -> None:
    """06818000 at level 3 places zero level shifts, and must not hide it.

    A dataset silently missing a type would be scored as if it covered all five,
    quietly corrupting macro-F1 (§11) with an undefined per-type recall.
    """
    values = make_base(days=240, gaps=False)
    v = values["value"].to_numpy(copy=True)
    # Islands far too short for a 6-72h level shift (24-288 rows).
    for start in range(0, len(v) - 12, 14):
        v[start : start + 10] = np.nan
    values["value"] = v

    result = inject_series(values, level=3, seed=1, name="fragmented")
    assert LEVEL_SHIFT in result.manifest["types_missing"]
    assert not result.manifest["scoreable"]


# ---------------------------------------------------------------------------
# Labels contract + gap sourcing
# ---------------------------------------------------------------------------
def test_labels_satisfy_the_contract(base: pd.DataFrame) -> None:
    result = inject_series(base, level=2, seed=1, name="t")
    validated = validate_labels_frame(result.labels)
    assert len(validated) == len(base)


def test_labels_are_row_aligned_with_the_data(base: pd.DataFrame) -> None:
    result = inject_series(base, level=2, seed=1, name="t")
    assert result.data[DATETIME_COL].equals(result.labels[DATETIME_COL])


def test_natural_gaps_are_labelled_but_carry_no_true_value(base: pd.DataFrame) -> None:
    result = inject_series(base, level=1, seed=1, name="t")
    natural = result.labels["source"] == SOURCE_NATURAL

    # The base is not pristine, and the labels say so rather than pretending.
    assert natural.sum() == int(base["value"].isna().sum())
    assert result.labels.loc[natural, "is_anomaly"].all()
    assert (result.labels.loc[natural, "anomaly_type"] == GAP).all()
    # No recorded truth, so these cannot be scored for imputation error.
    assert result.labels.loc[natural, "true_value"].isna().all()


def test_injected_gaps_are_scoreable_for_imputation(base: pd.DataFrame) -> None:
    result = inject_series(base, level=2, seed=1, name="t")
    injected_gaps = (result.labels["source"] == SOURCE_INJECTED) & (
        result.labels["anomaly_type"] == GAP
    )

    assert injected_gaps.any()
    # We deleted values we know, so every injected gap has a truth to score against.
    assert result.labels.loc[injected_gaps, "true_value"].notna().all()
    assert result.data.loc[injected_gaps, "value"].isna().all()


def test_point_anomalies_never_land_on_missing_data(base: pd.DataFrame) -> None:
    """Spikes and injected gaps need a real reading in every row they touch.

    A spike displaces a value, and an injected gap must delete a value we know —
    otherwise its true_value is unknown and `source=injected` would be a lie.
    """
    result = inject_series(base, level=3, seed=1, name="t")
    natural = ~np.isfinite(base["value"].to_numpy())
    for seg in result.segments:
        if seg.anomaly_type in (SPIKE, GAP):
            assert not natural[seg.start_idx : seg.end_idx].any(), seg


def test_segment_anomalies_may_span_isolated_dropouts() -> None:
    """A stuck sensor with one absent sample mid-window is entirely plausible.

    Refusing to span isolated dropouts left only 16.8% of 06818000 usable and
    starved level 3 of plateaus and level shifts, even though that record's
    longest missing run in 253 days is 10 samples.
    """
    df = make_base(days=240, gaps=False)
    v = df["value"].to_numpy(copy=True)
    # A single-sample dropout every 25 rows: invisible on a plot, but under the
    # old rule it blocked every plateau and level shift on the series.
    v[np.arange(500, len(v) - 100, 25)] = np.nan
    df["value"] = v

    result = inject_series(df, level=3, seed=1, name="dotty")

    assert result.manifest["scoreable"], result.manifest["types_missing"]
    spanning = [
        s
        for s in result.segments
        if s.anomaly_type in (PLATEAU, LEVEL_SHIFT, DRIFT)
        and not np.isfinite(df["value"].to_numpy()[s.start_idx : s.end_idx]).all()
    ]
    assert spanning, "segment anomalies should be able to span isolated dropouts"


def test_plateau_does_not_resurrect_missing_data() -> None:
    # A stuck sensor and a dropped sample are independent failures: freezing the
    # reading must not invent data the logger never recorded.
    values = np.arange(50, dtype=float)
    values[10] = np.nan
    out, _ = inject_plateau(values, 8, length=6)

    assert np.isnan(out[10])
    assert out[8] == out[9] == out[11] == 8.0


def test_non_anomalous_rows_have_empty_type_and_source(base: pd.DataFrame) -> None:
    result = inject_series(base, level=1, seed=1, name="t")
    clean = ~result.labels["is_anomaly"]
    assert (result.labels.loc[clean, "anomaly_type"] == "").all()
    assert (result.labels.loc[clean, "source"] == "").all()


# ---------------------------------------------------------------------------
# Reproducibility (§2: seed everything)
# ---------------------------------------------------------------------------
def test_same_seed_reproduces_identical_output(base: pd.DataFrame) -> None:
    a = inject_series(base, level=2, seed=42, name="t")
    b = inject_series(base, level=2, seed=42, name="t")
    pd.testing.assert_frame_equal(a.data, b.data)
    pd.testing.assert_frame_equal(a.labels, b.labels)
    pd.testing.assert_frame_equal(a.maintenance, b.maintenance)


def test_different_seed_gives_different_placement(base: pd.DataFrame) -> None:
    a = inject_series(base, level=2, seed=1, name="t")
    b = inject_series(base, level=2, seed=2, name="t")
    assert [s.start_idx for s in a.segments] != [s.start_idx for s in b.segments]


def test_streams_are_independent_across_names_and_levels(base: pd.DataFrame) -> None:
    # Seeds derive per (name, level), so adding a dataset or a level must not
    # shift the anomalies in the others.
    a = inject_series(base, level=1, seed=5, name="site_a")
    b = inject_series(base, level=1, seed=5, name="site_b")
    assert [s.start_idx for s in a.segments] != [s.start_idx for s in b.segments]

    l1 = inject_series(base, level=1, seed=5, name="site_a")
    pd.testing.assert_frame_equal(a.data, l1.data)


# ---------------------------------------------------------------------------
# Contamination levels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("level", sorted(LEVELS))
def test_point_budget_lands_near_its_target(base: pd.DataFrame, level: int) -> None:
    result = inject_series(base, level=level, seed=1, name="t")
    actual = result.manifest["actual_point_pct"]
    assert actual == pytest.approx(LEVELS[level].point_pct, abs=0.5)
    assert result.manifest["point_budget_met"]


def test_budget_is_met_on_a_base_riddled_with_small_natural_gaps() -> None:
    """Placement must survive fragmentation, not give up on it.

    06818000 is 14.3% missing, scattered as ~2,700 tiny dropouts. A rejection
    sampler gives up on that long before the budget is spent, even though half the
    series is free; exact free-run placement finds the room that is actually there.
    """
    frag = make_base(days=240, gaps=False)
    values = frag["value"].to_numpy(copy=True)
    # A 2-hour dropout every 20 hours: ~10% missing, split into ~290 free runs.
    # Fragmented enough to defeat guess-and-check, but the runs still comfortably
    # fit the longest plateau, so the budget is genuinely placeable.
    for start in range(500, len(values) - 100, 80):
        values[start : start + 8] = np.nan
    frag["value"] = values

    result = inject_series(frag, level=2, seed=1, name="frag")
    assert result.manifest["point_budget_met"], result.manifest["actual_point_pct"]


def test_budget_shortfall_is_reported_rather_than_hidden() -> None:
    # If a base genuinely cannot host the budget, the dataset must say so instead
    # of quietly claiming a level it does not carry.
    values = make_base(days=240, gaps=False)
    v = values["value"].to_numpy(copy=True)
    # Leave only tiny islands of usable data: nothing long enough for a plateau.
    for start in range(0, len(v) - 10, 12):
        v[start : start + 9] = np.nan
    values["value"] = v

    result = inject_series(values, level=3, seed=1, name="hostile")
    assert not result.manifest["point_budget_met"]
    assert result.manifest["actual_point_pct"] < LEVELS[3].point_pct


def test_contamination_increases_with_level(base: pd.DataFrame) -> None:
    pcts = [
        inject_series(base, level=lv, seed=1, name="t").manifest["total_anomalous_pct"]
        for lv in sorted(LEVELS)
    ]
    assert pcts == sorted(pcts)


def test_drift_share_is_comparable_across_bases_of_different_length() -> None:
    """The reason drift is driven by maintenance *interval* and not by count.

    A fixed episode count would make level 3 mean ~37% drift on a short base but
    ~16% on a long one, and the levels would not be comparable across datasets.
    """
    short = inject_series(make_base(days=240), level=3, seed=1, name="short")
    long = inject_series(make_base(days=560), level=3, seed=1, name="long")

    short_pct = short.manifest["by_type"][DRIFT]["pct_rows"]
    long_pct = long.manifest["by_type"][DRIFT]["pct_rows"]
    assert short_pct == pytest.approx(long_pct, abs=8.0)
    # The longer record simply contains more maintenance cycles.
    assert (
        long.manifest["by_type"][DRIFT]["n_events"]
        > short.manifest["by_type"][DRIFT]["n_events"]
    )


def test_magnitudes_stay_physical_on_a_flashy_storm_driven_series() -> None:
    """Storms must not size the anomalies.

    03447687 is quiet (median ~7 FNU) but spikes to ~1000 in storms. Sizing
    magnitudes off a rolling *standard deviation* let the storm's own variability
    set the scale and produced a 2,973 FNU spike — 400x the median, far outside
    anything the river does, and trivially detectable. A rolling MAD with a cap
    keeps injected anomalies inside the plausible range.
    """
    n = 240 * 24 * 4
    idx = pd.date_range("2024-01-01", periods=n, freq=STEP)
    rng = np.random.default_rng(0)
    values = np.abs(rng.normal(7.0, 1.0, n))
    # Three storms: quiet baseline punctuated by excursions to ~900 FNU.
    for start in (8000, 16000, 20000):
        storm = np.linspace(0, np.pi, 600)
        values[start : start + 600] += 900 * np.sin(storm)
    flashy = pd.DataFrame({DATETIME_COL: idx, "value": values})

    result = inject_series(flashy, level=3, seed=1, name="flashy")
    injected = result.data["value"].to_numpy()

    # Nothing injected may exceed what the river itself plausibly reaches.
    assert np.nanmax(injected) < 3.0 * np.nanmax(values), (
        f"injected max {np.nanmax(injected):.0f} vs series max {np.nanmax(values):.0f}"
    )


def test_local_scale_is_bounded_by_the_global_scale() -> None:
    from src.inject import (
        LOCAL_SCALE_CAP_K,
        LOCAL_SCALE_FLOOR_K,
        LOCAL_SCALE_WINDOW_ROWS,
        _local_scale,
        _robust_scale,
    )

    values = np.abs(np.random.default_rng(0).normal(10.0, 2.0, 5000))
    values[2000:2100] += 500.0  # a storm
    scale = _local_scale(values, LOCAL_SCALE_WINDOW_ROWS)
    g = _robust_scale(values)

    assert np.isfinite(scale).all()
    assert scale.min() >= LOCAL_SCALE_FLOOR_K * g - 1e-9
    assert scale.max() <= LOCAL_SCALE_CAP_K * g + 1e-9


def test_robust_scale_survives_a_flat_series() -> None:
    from src.inject import _robust_scale

    # A flat base has no spread, but injection still needs a non-zero scale or
    # every magnitude would be zero and rejected by the injectors.
    assert _robust_scale(np.full(100, 5.0)) > 0
    assert _robust_scale(np.full(100, np.nan)) > 0


def test_unknown_level_is_rejected(base: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="level must be one of"):
        inject_series(base, level=9, seed=1, name="t")


def test_base_too_short_for_realistic_drift_is_rejected() -> None:
    # Rather than silently shrinking drift into something that is not "creep over
    # weeks" any more, injection refuses (§13: fail loudly on nonsensical input).
    tiny = make_base(days=5, gaps=False)
    with pytest.raises(ValueError, match="too short"):
        inject_series(tiny, level=3, seed=1, name="tiny")


def test_missing_value_column_is_rejected(base: pd.DataFrame) -> None:
    with pytest.raises(ContractError, match="value column"):
        inject_series(base.rename(columns={"value": "turbidity"}), level=1, seed=1, name="t")


# ---------------------------------------------------------------------------
# Maintenance schedule (the support points correctDrift needs)
# ---------------------------------------------------------------------------
def test_maintenance_event_follows_each_drift_episode(base: pd.DataFrame) -> None:
    result = inject_series(base, level=3, seed=1, name="t")
    drifts = [s for s in result.segments if s.anomaly_type == DRIFT]

    assert len(result.maintenance) == len(drifts)
    # correctDrift reads the index as the start of a maintenance event and the
    # value as its end, so start must precede end.
    assert (result.maintenance["end"] > result.maintenance["start"]).all()

    times = pd.DatetimeIndex(base[DATETIME_COL])
    for seg, (_, event) in zip(drifts, result.maintenance.iterrows()):
        assert event["start"] == times[seg.end_idx]


def test_maintenance_schedule_is_ordered_and_disjoint(base: pd.DataFrame) -> None:
    result = inject_series(base, level=3, seed=1, name="t")
    m = result.maintenance
    assert (m["start"].diff().dropna() > pd.Timedelta(0)).all()
    assert (m["start"].to_numpy()[1:] > m["end"].to_numpy()[:-1]).all()


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def test_write_result_emits_the_full_quartet(base: pd.DataFrame, tmp_path) -> None:
    result = inject_series(base, level=1, seed=1, name="site_l1")
    paths = write_result(result, tmp_path)

    for key in ("data", "labels", "maintenance", "manifest"):
        assert paths[key].is_file(), key

    # Round-trips through CSV and still satisfies the contract.
    reloaded = pd.read_csv(paths["labels"], keep_default_na=True)
    assert len(validate_labels_frame(reloaded)) == len(base)
