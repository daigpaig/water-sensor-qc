"""Tests for src/tools/candidates.py and src/tools/review.py.

The series built here contain *planted* anomalies at known positions, so a test
can assert that a candidate actually lands on the thing that was planted rather
than merely that some candidate exists.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tools.candidates import (
    LABEL_TYPES,
    REVIEW_TYPES,
    Candidate,
    DetectConfig,
    _runs,
    find_candidates,
    merge_decisions,
    robust_scales,
)
from src.tools.review import build_payload, build_review_html, main

# Planted positions in the fixture series (see `planted_csv`).
SPIKE_POS = 900
PLATEAU_SLICE = (1500, 1560)  # 15h stuck at one value on a 15-min grid
GAP_SLICE = (2000, 2040)  # 10h of NaN
SHIFT_POS = 2500


@pytest.fixture(scope="module")
def planted_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 3000-row 15-min series with one of each failure type planted in it."""
    rng = np.random.default_rng(0)
    n = 3000
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    values = 20.0 + rng.normal(0, 0.4, n)

    values[SPIKE_POS] = 95.0  # spike: far from neighbours
    values[slice(*PLATEAU_SLICE)] = 21.0  # plateau: exactly constant
    values[SHIFT_POS:] += 18.0  # level shift: permanent step
    values[slice(*GAP_SLICE)] = np.nan  # gap: NaN run

    path = tmp_path_factory.mktemp("planted") / "planted.csv"
    pd.DataFrame({"datetime": idx, "value": values}).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def result(planted_csv: Path):
    return find_candidates(planted_csv)


# --------------------------------------------------------------------------- units
def test_runs_finds_contiguous_blocks():
    mask = np.array([0, 1, 1, 0, 0, 1, 0], dtype=bool)
    assert _runs(mask) == [(1, 3), (5, 6)]


def test_runs_bridges_short_breaks_but_not_long_ones():
    mask = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1], dtype=bool)
    assert _runs(mask, bridge=1) == [(0, 5), (8, 9)]
    assert _runs(mask, bridge=0) == [(0, 2), (3, 5), (8, 9)]


def test_runs_on_empty_mask():
    assert _runs(np.zeros(10, dtype=bool)) == []


def test_robust_scales_are_positive_on_a_constant_series():
    """A zero MAD must not produce a zero threshold, or detectors flag everything."""
    level, step = robust_scales(pd.Series([5.0] * 50))
    assert level > 0 and step > 0


def test_robust_scales_ignore_outliers():
    """MAD-based scale barely moves when a few wild values are added."""
    clean = pd.Series(np.random.default_rng(1).normal(0, 1, 500))
    dirty = clean.copy()
    dirty.iloc[:5] = 1000.0
    assert robust_scales(dirty)[0] == pytest.approx(robust_scales(clean)[0], rel=0.1)


# --------------------------------------------------------------------------- detect
def _covering(result, atype: str, pos: int) -> list[Candidate]:
    return [
        c for c in result.candidates
        if c.anomaly_type == atype and c.start_pos <= pos < c.end_pos
    ]


def test_every_detector_ran_without_error(result):
    assert result.errors == {}, f"detector(s) failed: {result.errors}"


def test_finds_the_planted_spike(result):
    assert _covering(result, "spike", SPIKE_POS)


def test_finds_the_planted_plateau(result):
    mid = (PLATEAU_SLICE[0] + PLATEAU_SLICE[1]) // 2
    assert _covering(result, "plateau", mid)


def test_gaps_are_never_queued_for_review(result):
    """Whether a value is missing is not a judgement call — §5 settles it."""
    assert not any(c.anomaly_type == "gap" for c in result.candidates)
    assert "gap" not in REVIEW_TYPES


def test_gap_totals_count_every_missing_run(result):
    n_runs, n_rows = result.gap_totals()
    assert n_rows == int(result.series.isna().sum())
    assert n_runs >= 1


def test_finds_the_planted_level_shift(result):
    """An abrupt step survives the sharpness filter; flagJumps marks the edge."""
    assert any(
        c.anomaly_type == "level_shift" and abs(c.start_pos - SHIFT_POS) < 100
        for c in result.candidates
    )


def test_gradual_slope_is_not_flagged_as_a_level_shift(tmp_path: Path):
    """A ramp of the same total size as a step must be rejected (the user's bug).

    flagJumps fires on a gradual rise the same as on an abrupt one; the sharpness
    filter is what tells them apart. This is a storm limb, not a sensor step.
    """
    rng = np.random.default_rng(3)
    n = 3000
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    values = 20.0 + rng.normal(0, 0.4, n)
    # +25 rise spread over 48 samples (~12h), then hold: steep enough for
    # flagJumps to fire, but each step is ~0.5 — no single sharp edge.
    ramp = np.linspace(0, 25, 48)
    values[1000:1048] += ramp
    values[1048:] += 25.0
    path = tmp_path / "ramp.csv"
    pd.DataFrame({"datetime": idx, "value": values}).to_csv(path, index=False)

    res = find_candidates(path)
    on_ramp = [
        c for c in res.candidates
        if c.anomaly_type == "level_shift" and c.start_pos < 1200 and c.end_pos > 1000
    ]
    assert not on_ramp, f"gradual slope wrongly flagged: {[c.candidate_id for c in on_ramp]}"


def test_sharpness_filter_can_be_relaxed(tmp_path: Path):
    """The filter is a knob, not a hard rule — 0.0 keeps flagJumps' raw output."""
    rng = np.random.default_rng(4)
    n = 3000
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    values = 20.0 + rng.normal(0, 0.4, n)
    values[1000:1048] += np.linspace(0, 25, 48)
    values[1048:] += 25.0
    path = tmp_path / "ramp2.csv"
    pd.DataFrame({"datetime": idx, "value": values}).to_csv(path, index=False)

    strict = find_candidates(path)
    loose = find_candidates(path, config=DetectConfig(shift_min_sharpness=0.0))
    n_strict = sum(c.anomaly_type == "level_shift" for c in strict.candidates)
    n_loose = sum(c.anomaly_type == "level_shift" for c in loose.candidates)
    assert n_loose > n_strict


def test_even_a_single_missing_sample_is_labelled_a_gap(planted_csv: Path, tmp_path: Path):
    """§5: *every* missing run is a gap. No length threshold — a 1-row hole counts.

    The earlier design filtered gaps to runs >= 1h, which omitted 93% of the
    missing rows on 06818000.
    """
    df = pd.read_csv(planted_csv)
    df.loc[100, "value"] = np.nan  # a single missing sample
    path = tmp_path / "dropout.csv"
    df.to_csv(path, index=False)

    res = find_candidates(path)
    labels = merge_decisions(res, _decisions(res, {}))
    assert labels.loc[100, "is_anomaly"]
    assert labels.loc[100, "anomaly_type"] == "gap"


def test_candidate_spans_are_the_detected_extent(result):
    """Positions must bound the flagged rows exactly — merge() labels this span."""
    for c in result.candidates:
        assert 0 <= c.start_pos < c.end_pos <= len(result.series)
        assert c.n_rows == c.end_pos - c.start_pos
        assert c.start == result.series.index[c.start_pos]


def test_candidate_ids_are_unique_and_typed(result):
    ids = [c.candidate_id for c in result.candidates]
    assert len(ids) == len(set(ids))
    for c in result.candidates:
        assert c.candidate_id.startswith(c.anomaly_type)


def test_candidates_are_ordered_by_time_within_type(result):
    for atype in REVIEW_TYPES:
        starts = [c.start_pos for c in result.candidates if c.anomaly_type == atype]
        assert starts == sorted(starts)


def test_max_per_type_caps_and_reports_the_remainder(planted_csv: Path):
    res = find_candidates(planted_csv, config=DetectConfig(max_per_type=1))
    counts = res.counts()
    assert all(v <= 1 for v in counts.values())
    # anything dropped must be accounted for, not silently discarded
    full = find_candidates(planted_csv)
    for atype, n in full.counts().items():
        if n > 1:
            assert res.truncated[atype] == n - 1


def test_a_failing_detector_is_recorded_not_raised(planted_csv: Path, monkeypatch):
    """flagPlateau raises on some real inputs (§7.1) — one failure must not sink the run."""
    import src.tools.candidates as mod

    real = mod._run_saqc

    def flaky(series, method, params):
        if method == "flagPlateau":
            raise ValueError("attempt to get argmin of an empty sequence")
        return real(series, method, params)

    monkeypatch.setattr(mod, "_run_saqc", flaky)
    res = find_candidates(planted_csv)
    assert "flagPlateau" in res.errors
    assert res.candidates  # the other detectors still produced candidates


def test_frame_matches_the_candidate_list(result):
    frame = result.frame()
    assert len(frame) == len(result.candidates)
    assert frame["decision"].eq("").all()  # reviewer fills this in


def test_summary_mentions_every_type(result):
    text = result.summary()
    for atype in REVIEW_TYPES:
        assert atype in text


def test_spike_candidates_contain_no_unflagged_filler(result):
    """Spikes are never bridged: every row in the span was actually flagged.

    Bridging at 2h glued distinct spikes together across the rows between them,
    putting 12-27% never-flagged filler inside spike spans on the real gauges —
    rows a confirmed candidate would then label as spikes.
    """
    for c in result.candidates:
        if c.anomaly_type != "spike":
            continue
        body = result.series.iloc[c.start_pos : c.end_pos]
        assert not body.isna().any(), f"{c.candidate_id} spans a missing row"


def test_end_timestamp_is_the_last_included_row(result):
    """`end` is inclusive so the exported span reads correctly and can be edited."""
    for c in result.candidates:
        assert c.end == result.series.index[c.end_pos - 1]
        assert c.start <= c.end


# --------------------------------------------------------------------------- merge
def _decisions(result, mapping: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"candidate_id": c.candidate_id, "decision": mapping.get(c.candidate_id, "")}
            for c in result.candidates
        ]
    )


def _reviewed(labels: pd.DataFrame) -> pd.DataFrame:
    """Rows labelled by a human decision — gaps are added unconditionally."""
    return labels[labels["is_anomaly"] & (labels["anomaly_type"] != "gap")]


def test_merge_labels_only_confirmed_candidates(result):
    spike = _covering(result, "spike", SPIKE_POS)[0]
    labels = merge_decisions(result, _decisions(result, {spike.candidate_id: "anomaly"}))

    assert labels.loc[spike.start_pos, "is_anomaly"]
    assert labels.loc[spike.start_pos, "anomaly_type"] == "spike"
    assert len(_reviewed(labels)) == spike.n_rows


def test_merge_ignores_rejected_and_undecided(result):
    labels = merge_decisions(
        result, _decisions(result, {c.candidate_id: "normal" for c in result.candidates})
    )
    assert _reviewed(labels).empty


def test_merge_can_include_unsure(result):
    cid = result.candidates[0].candidate_id
    dec = _decisions(result, {cid: "unsure"})
    assert _reviewed(merge_decisions(result, dec)).empty
    assert not _reviewed(merge_decisions(result, dec, include_unsure=True)).empty


def test_merge_labels_every_missing_row_as_a_gap(result):
    """§5, unconditionally — even with no decisions at all."""
    labels = merge_decisions(result, _decisions(result, {}))
    missing = result.series.isna().to_numpy()

    assert (labels.loc[missing, "anomaly_type"] == "gap").all()
    assert labels.loc[missing, "is_anomaly"].all()
    assert int((labels["anomaly_type"] == "gap").sum()) == int(missing.sum())


def test_gap_wins_over_an_overlapping_confirmed_candidate(result):
    """A missing value is missing whatever else a detector thought was there."""
    series = result.series
    covering_gap = Candidate(
        candidate_id="plateau_gapoverlap", anomaly_type="plateau", detectors=("x",),
        start_pos=GAP_SLICE[0] - 5, end_pos=GAP_SLICE[1] + 5,
        start=series.index[GAP_SLICE[0] - 5], end=series.index[GAP_SLICE[1] + 5],
        n_rows=(GAP_SLICE[1] - GAP_SLICE[0]) + 10, duration_hours=1.0,
        score=1.0, v_min=0.0, v_max=1.0,
    )
    result.candidates.append(covering_gap)
    try:
        labels = merge_decisions(
            result, _decisions(result, {covering_gap.candidate_id: "anomaly"})
        )
        mid = (GAP_SLICE[0] + GAP_SLICE[1]) // 2
        assert labels.loc[mid, "anomaly_type"] == "gap"
        # the non-missing shoulders keep the reviewed label
        assert labels.loc[GAP_SLICE[0] - 3, "anomaly_type"] == "plateau"
    finally:
        result.candidates.remove(covering_gap)


def test_merge_output_satisfies_the_labels_contract(result, tmp_path: Path):
    from src.inspect_data import validate_labels_csv

    cid = result.candidates[0].candidate_id
    labels = merge_decisions(result, _decisions(result, {cid: "anomaly"}))
    path = tmp_path / "out_labels.csv"
    labels.to_csv(path, index=False)

    loaded = validate_labels_csv(path)
    assert len(loaded) == len(result.series)
    # natural anomalies have no known clean value, so imputation is unscoreable (§5)
    flagged = loaded[loaded["is_anomaly"]]
    assert (flagged["source"] == "natural").all()
    assert flagged["true_value"].isna().all()


def test_merge_resolves_overlaps_by_type_priority(result):
    """A spike inside a confirmed level_shift stays labelled spike."""
    series = result.series
    shift = Candidate(
        candidate_id="level_shift_900", anomaly_type="level_shift", detectors=("x",),
        start_pos=800, end_pos=1000, start=series.index[800], end=series.index[999],
        n_rows=200, duration_hours=50.0, score=1.0, v_min=0.0, v_max=1.0,
    )
    spike = _covering(result, "spike", SPIKE_POS)[0]
    result.candidates.append(shift)
    try:
        labels = merge_decisions(
            result,
            _decisions(result, {shift.candidate_id: "anomaly", spike.candidate_id: "anomaly"}),
        )
        assert labels.loc[SPIKE_POS, "anomaly_type"] == "spike"
        assert labels.loc[805, "anomaly_type"] == "level_shift"
    finally:
        result.candidates.remove(shift)


def _decisions_with_span(result, cid: str, start, end) -> pd.DataFrame:
    """Decisions frame carrying an explicit (possibly narrowed) span."""
    rows = []
    for c in result.candidates:
        chosen = c.candidate_id == cid
        rows.append({
            "candidate_id": c.candidate_id,
            "decision": "anomaly" if chosen else "",
            "start": start if chosen else c.start,
            "end": end if chosen else c.end,
        })
    return pd.DataFrame(rows)


def test_merge_honours_a_narrowed_span(result):
    """Only the endorsed sample is labelled, not the detector's whole window."""
    wide = max(result.candidates, key=lambda c: c.n_rows)
    assert wide.n_rows > 1, "need a multi-row candidate to narrow"
    peak = result.series.index[wide.start_pos]

    labels = merge_decisions(
        result, _decisions_with_span(result, wide.candidate_id, peak, peak)
    )
    assert len(_reviewed(labels)) == 1
    assert labels.loc[wide.start_pos, "anomaly_type"] == wide.anomaly_type
    assert not labels.loc[wide.start_pos + 1, "is_anomaly"]


def test_merge_rejects_a_span_outside_the_candidate(result):
    """A decision may narrow a proposal, never extend it."""
    c = result.candidates[0]
    outside = result.series.index[c.end_pos + 50]
    with pytest.raises(ValueError, match="outside the candidate"):
        merge_decisions(
            result, _decisions_with_span(result, c.candidate_id, c.start, outside)
        )


def test_merge_without_span_columns_uses_the_full_candidate(result):
    """The narrowing columns are optional — a bare id/decision CSV still works."""
    c = result.candidates[0]
    labels = merge_decisions(result, _decisions(result, {c.candidate_id: "anomaly"}))
    assert len(_reviewed(labels)) == c.n_rows


def test_merge_rejects_a_frame_without_the_required_columns(result):
    with pytest.raises(ValueError, match="candidate_id"):
        merge_decisions(result, pd.DataFrame({"foo": [1]}))


# --------------------------------------------------------------------------- page
def test_payload_reconstructs_the_series_timestamps(result):
    payload = build_payload(result)
    assert payload["n"] == len(result.series)
    assert len(payload["values"]) == len(result.series)
    last = payload["t0"] + (payload["n"] - 1) * payload["step_ms"]
    assert pd.Timestamp(last, unit="ms") == result.series.index[-1]


def test_payload_encodes_missing_values_as_null(result):
    payload = build_payload(result)
    assert payload["values"][GAP_SLICE[0] + 1] is None
    # must be strict JSON: NaN would parse as a syntax error in the browser
    json.dumps(payload, allow_nan=False)


def test_page_is_self_contained_and_carries_its_data(result):
    html = build_review_html(result)
    assert "plotly" in html.lower()
    assert "<script src=" not in html  # no CDN, no network
    assert 'id="payload"' in html
    for c in result.candidates[:5]:
        assert c.candidate_id in html


def test_payload_cannot_break_out_of_its_script_tag(result):
    """A '</script>' inside the data would end the host element early."""
    result.errors["evil"] = "</script><script>alert(1)</script>"
    try:
        assert "</script><script>alert" not in build_review_html(result)
    finally:
        result.errors.pop("evil")


# --------------------------------------------------------------------------- CLI
def test_detect_cli_writes_candidates_and_page(
    planted_csv: Path, tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)  # DEFAULT_DATA_DIR is relative to the cwd
    out = tmp_path / "review.html"
    assert main(["detect", str(planted_csv), "--out", str(out), "--no-open"]) == 0
    assert out.is_file()
    assert (tmp_path / "data/review" / f"{planted_csv.stem}_candidates.csv").is_file()


def test_detect_cli_does_not_write_beside_the_series(
    planted_csv: Path, tmp_path: Path, monkeypatch
):
    """data/clean/ is globbed by name elsewhere; a stray CSV there breaks lookups."""
    monkeypatch.chdir(tmp_path)
    before = set(planted_csv.parent.iterdir())
    main(["detect", str(planted_csv), "--out", str(tmp_path / "r.html"), "--no-open"])
    assert set(planted_csv.parent.iterdir()) == before


def test_merge_cli_round_trips_through_the_exported_csv(
    planted_csv: Path, tmp_path: Path, result
):
    spike = _covering(result, "spike", SPIKE_POS)[0]
    dec_path = tmp_path / "decisions.csv"
    _decisions(result, {spike.candidate_id: "anomaly"}).to_csv(dec_path, index=False)

    out = tmp_path / "labels.csv"
    assert main(["merge", str(planted_csv), str(dec_path), "--out", str(out)]) == 0

    labels = pd.read_csv(out)
    assert len(_reviewed(labels)) == spike.n_rows
    # the planted NaN run is labelled too, without anyone reviewing it
    assert int((labels["anomaly_type"] == "gap").sum()) == GAP_SLICE[1] - GAP_SLICE[0]


def test_merge_cli_refuses_decisions_from_a_different_run(planted_csv: Path, tmp_path: Path):
    """Ids come from the detector settings; a mismatch would silently drop verdicts."""
    dec_path = tmp_path / "stale.csv"
    pd.DataFrame(
        {"candidate_id": ["spike_999"], "decision": ["anomaly"]}
    ).to_csv(dec_path, index=False)

    with pytest.raises(SystemExit, match="do not exist"):
        main(["merge", str(planted_csv), str(dec_path), "--out", str(tmp_path / "l.csv")])


def test_merge_cli_reports_a_malformed_decisions_csv(planted_csv: Path, tmp_path: Path, capsys):
    dec_path = tmp_path / "wrong.csv"
    pd.DataFrame({"id": ["spike_001"], "verdict": ["anomaly"]}).to_csv(dec_path, index=False)

    assert main(["merge", str(planted_csv), str(dec_path), "--out", str(tmp_path / "l.csv")]) == 1
    assert "missing column(s)" in capsys.readouterr().err
