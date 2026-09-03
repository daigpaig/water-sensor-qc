"""Tests for the SaQC-wrapping tool functions and SaQC 2.8 presence."""

import json
import numpy as np
import pandas as pd
import pytest
import saqc

from pathlib import Path

from src.agent_tools import wrappers
from src.inspect_data import DATETIME_COL


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


def test_impute_linear_returns_json():
    qc = _toy_qc()
    result = wrappers.impute_linear(qc, field="value", max_gap="1h")

    assert result["tool"] == "impute_linear"
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
# export_clean_data and the §5 flag log
# ---------------------------------------------------------------------------

def _flagged_qc():
    """A toy series with a spike flagged and a gap flagged + partly imputed."""
    qc = wrappers.flag_spike_unilof(_toy_qc(), field="value", thresh=1.5)["qc"]
    qc = wrappers.flag_nan(qc, field="value")["qc"]
    return qc


def test_export_writes_the_flag_log_with_the_agents_decisions(tmp_path):
    """§5: export_clean_data is what turns flags into `{datetime, flagged_by, action, reason}`."""
    qc = _flagged_qc()
    spike_at = qc.data.to_pandas().index[50]

    result = wrappers.export_clean_data(
        qc,
        decisions=[
            {"start": str(spike_at), "difficulty": "clear", "verdict": "anomaly", "anomaly_type": "spike",
             "action": "delete", "reason": "robust_z 40, 1 sample wide"},
            {"start": "2024-01-01T20:00:00", "end": "2024-01-01T21:00:00",
             "difficulty": "clear", "verdict": "anomaly", "anomaly_type": "gap",
             "action": "impute", "reason": "5-sample gap, well under max_gap"},
        ],
        output_dir=tmp_path,
        stem="toy_l1",
    )

    written = tmp_path / "toy_l1_flags.json"
    assert result["flags_path"] == str(written)
    entries = json.loads(written.read_text())

    assert entries == result["flags"]
    assert {e["action"] for e in entries} <= {"delete", "impute", "keep", "correct", "undecided"}
    by_stamp = {e["datetime"]: e for e in entries}
    deleted = by_stamp[spike_at.strftime("%Y-%m-%dT%H:%M:%S")]
    assert deleted["action"] == "delete"
    assert deleted["reason"].startswith("robust_z")
    assert "flagUniLOF" in deleted["flagged_by"]
    assert sum(e["action"] == "impute" for e in entries) == 5   # the whole NaN run
    assert result["n_undecided"] == 0
    assert result["decisions_matching_no_flagged_row"] == []


def test_export_marks_flagged_rows_the_agent_never_judged_as_undecided(tmp_path):
    """A flag with no decision must not pass silently as a deliberate keep.

    `evaluate.apply_decisions` scores an undecided row as no claim either way, so the
    only defence against a run that flags everything and concludes nothing is that the
    export says so — loudly, in the message the agent reads.
    """
    result = wrappers.export_clean_data(_flagged_qc(), output_dir=tmp_path, stem="toy_l1")

    entries = json.loads((tmp_path / "toy_l1_flags.json").read_text())
    assert entries, "a flagged series must not produce an empty flag log"
    assert all(e["action"] == wrappers.UNDECIDED for e in entries)
    assert result["n_undecided"] == len(entries)
    assert "WARNING" in result["message"] and "undecided" in result["message"]


def test_export_reports_a_decision_that_matched_no_flagged_row(tmp_path):
    """A mistyped timestamp is a silent loss of a verdict unless it is reported back."""
    result = wrappers.export_clean_data(
        _flagged_qc(),
        decisions=[{"start": "2024-06-01T00:00:00", "difficulty": "clear", "verdict": "normal",
                    "action": "keep", "reason": "storm"}],
        output_dir=tmp_path,
        stem="toy_l1",
    )

    assert len(result["decisions_matching_no_flagged_row"]) == 1
    assert "matched no flagged row" in result["message"]


def test_export_rejects_an_action_outside_the_contract():
    """§13: fail loudly rather than write a log evaluate.py cannot read."""
    with pytest.raises(ValueError, match="delete"):
        wrappers.export_clean_data(
            _flagged_qc(),
            decisions=[{"start": "2024-01-01T12:30:00", "difficulty": "clear", "verdict": "anomaly",
                        "anomaly_type": "spike", "action": "flag", "reason": "x"}],
        )


def test_export_without_an_output_path_writes_nothing(tmp_path):
    """The runner owns the path; ablation and tests call the tool with no output at all."""
    result = wrappers.export_clean_data(_flagged_qc())

    assert result["flags_path"] is None
    assert result["flags"]
    assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# The verdict: the agent's actual answer, separate from the treatment (§5, §10)
# ---------------------------------------------------------------------------

def test_the_verdict_is_recorded_per_row_and_carries_the_agents_own_type():
    """§10 scores the verdict, so it must reach the log intact and typed by the AGENT.

    The type deliberately does not come from `flagged_by`: a row flagUniLOF found but
    the agent classified as a plateau is a plateau claim, because classifying it is
    the agent's job and grading it on the detector's guess measures the detector.
    """
    qc = _flagged_qc()
    idx = qc.data.to_pandas().index
    spike_at = str(idx[50])

    result = wrappers.export_clean_data(qc, decisions=[
        {"start": str(idx[0]), "end": str(idx[-1]), "difficulty": "clear", "verdict": "normal",
         "action": "keep", "reason": "storm limb, 14 samples wide"},
        {"start": spike_at, "difficulty": "clear", "verdict": "anomaly", "anomaly_type": "plateau",
         "action": "delete", "reason": "flagged by UniLOF but 9 samples unchanged"},
    ])

    entry = next(e for e in result["flags"]
                 if e["datetime"] == pd.Timestamp(spike_at).strftime("%Y-%m-%dT%H:%M:%S"))
    assert entry["verdict"] == "anomaly"
    assert entry["anomaly_type"] == "plateau"      # the agent's call, not flagUniLOF's
    assert "flagUniLOF" in entry["flagged_by"]

    assert result["n_by_verdict"]["anomaly"] >= 1
    assert result["n_by_verdict"]["normal"] > 0
    assert result["n_by_anomaly_type"]["plateau"] == 1
    assert "VERDICTS" in result["message"]


def test_a_normal_verdict_may_not_delete_the_value():
    """Calling a value real water and then removing it is not a defensible pair.

    Enforced one-directionally (wrappers.UNTREATED_ANOMALY_ACTION): if this were
    allowed, a run could score as having REJECTED a candidate while the value it
    claimed to keep is gone from the exported file.
    """
    qc = _flagged_qc()
    at = str(qc.data.to_pandas().index[50])

    for action in ("delete", "correct", "impute"):
        with pytest.raises(ValueError, match="verdict='normal'"):
            wrappers.export_clean_data(qc, decisions=[
                {"start": at, "difficulty": "clear", "verdict": "normal", "action": action, "reason": "storm"},
            ])


def test_an_anomaly_may_be_left_untreated_if_the_agent_says_why():
    """The one allowed divergence: a gap too long for any defensible fill window.

    Forcing consistency here would make the agent either lie about the verdict or
    impute a gap it had just judged unfillable, so `anomaly` + `keep` is legal — but
    only as a stated choice, since the pair claims a detection without touching the
    value.
    """
    qc = _flagged_qc()
    at = str(qc.data.to_pandas().index[50])

    with pytest.raises(ValueError, match="cannot be treated"):
        wrappers.export_clean_data(qc, decisions=[
            {"start": at, "difficulty": "clear", "verdict": "anomaly", "anomaly_type": "gap",
             "action": "keep", "reason": "too long"},          # no real justification
        ])

    result = wrappers.export_clean_data(qc, decisions=[
        {"start": at, "difficulty": "clear", "verdict": "anomaly", "anomaly_type": "gap", "action": "keep",
         "reason": "47-sample outage, longer than any window I could defend; left NaN"},
    ])
    entry = next(e for e in result["flags"]
                 if e["datetime"] == pd.Timestamp(at).strftime("%Y-%m-%dT%H:%M:%S"))
    assert (entry["verdict"], entry["action"]) == ("anomaly", "keep")
    assert result["n_anomalies_left_untreated"] == 1
    assert "kept as recorded" in result["message"]


def test_export_rejects_a_decision_with_no_verdict_or_a_bad_one():
    """§13: the verdict is the answer, so an export cannot quietly omit it."""
    qc = _flagged_qc()
    at = str(qc.data.to_pandas().index[50])

    with pytest.raises(ValueError, match="verdict"):
        wrappers.export_clean_data(qc, decisions=[
            {"start": at, "difficulty": "clear", "action": "delete", "reason": "spike"},
        ])
    with pytest.raises(ValueError, match="verdict"):
        wrappers.export_clean_data(qc, decisions=[
            {"start": at, "difficulty": "clear", "verdict": "suspicious", "action": "delete", "reason": "x"},
        ])
    with pytest.raises(ValueError, match="anomaly_type"):
        wrappers.export_clean_data(qc, decisions=[
            {"start": at, "difficulty": "clear", "verdict": "anomaly", "action": "delete", "reason": "x"},
        ])
    with pytest.raises(ValueError, match="anomaly_type"):
        wrappers.export_clean_data(qc, decisions=[
            {"start": at, "difficulty": "clear", "verdict": "anomaly", "anomaly_type": "drift",     # §9.2
             "action": "delete", "reason": "x"},
        ])
    with pytest.raises(ValueError, match="no failure type"):
        wrappers.export_clean_data(qc, decisions=[
            {"start": at, "difficulty": "clear", "verdict": "normal", "anomaly_type": "spike",
             "action": "keep", "reason": "x"},
        ])


def test_export_rejects_a_decision_that_does_not_say_how_hard_the_call_was():
    """`difficulty` is required, with no default — it used to default to 'clear'.

    Measured on the 2026-08-20 run: 145 of 147 spans omitted it, so 98.6% of the
    run read as clear-cut, and those deletions were wrong 37.8% of the time. A
    default that manufactures a confidence claim is worse than no field, because
    the review queue is built from exactly this signal.
    """
    qc = _flagged_qc()
    at = str(qc.data.to_pandas().index[50])

    with pytest.raises(ValueError, match="difficulty"):
        wrappers.export_clean_data(qc, decisions=[
            {"start": at, "verdict": "anomaly", "anomaly_type": "spike",
             "action": "delete", "reason": "robust_z 40, 1 sample wide"},
        ])
    # Present and valid is fine; the omission is what fails.
    ok = wrappers.export_clean_data(qc, decisions=[
        {"start": at, "difficulty": "clear", "verdict": "anomaly", "anomaly_type": "spike",
         "action": "delete", "reason": "robust_z 40, 1 sample wide"},
    ])
    assert ok["n_by_verdict"]["anomaly"] >= 1


def test_an_imputed_row_is_recorded_as_a_gap_the_agent_found():
    """A filled row was missing, and §5 says every missing run is a gap.

    That is a fact about the data rather than a call the agent makes, so the verdict
    follows the forced `impute` action. Leaving it blank would let a run fill 3,403
    rows and record no claim about any of them.
    """
    qc = wrappers.flag_nan(_toy_qc(), field="value")["qc"]
    qc = wrappers.impute_linear(qc, field="value", max_gap="6h")["qc"]

    result = wrappers.export_clean_data(qc)      # no decisions at all
    filled = [e for e in result["flags"] if "impute_linear" in e["flagged_by"]]

    assert filled
    assert all(e["verdict"] == "anomaly" and e["anomaly_type"] == "gap" for e in filled)


def test_an_undecided_row_is_neither_a_detection_nor_a_rejection():
    """`undecided` is a third verdict the agent may not choose (§5).

    It must not read as a considered "normal": one is a judgement, the other is a row
    nobody looked at, and §10 scores it as no claim in either direction.
    """
    result = wrappers.export_clean_data(_flagged_qc())     # nothing adjudicated

    assert all(e["verdict"] == wrappers.UNDECIDED for e in result["flags"])
    assert all(e["anomaly_type"] == "" for e in result["flags"])
    assert result["n_by_verdict"][wrappers.UNDECIDED] == len(result["flags"])
    assert "anomaly" not in result["n_by_verdict"]


def test_every_entry_records_which_span_claimed_it():
    """`decided_by` is what lets an audit page tell a judgement from an absorption.

    `rationale_source` already says "blanket", but not by WHICH span or how wide it
    was — so a reader could not see that a row they were looking at had been swept
    up by a decision spanning the whole record.
    """
    qc = _flagged_qc()
    spike_at = qc.data.to_pandas().index[50]

    result = wrappers.export_clean_data(
        qc,
        decisions=[
            {"start": "2024-01-01T00:00:00", "end": "2024-01-03T00:00:00",
             "difficulty": "clear", "verdict": "normal", "action": "keep",
             "reason": "catch-all for everything not judged individually"},
            {"start": str(spike_at), "difficulty": "clear", "verdict": "anomaly", "anomaly_type": "spike",
             "action": "delete", "reason": "robust_z 40, 1 sample wide"},
        ],
    )
    by_time = {e["datetime"]: e for e in result["flags"]}
    spike = by_time[spike_at.strftime("%Y-%m-%dT%H:%M:%S")]

    # The narrow span won the row, and the entry names it rather than the catch-all.
    assert spike["decided_by"]["start"] == spike["decided_by"]["end"]
    assert spike["decided_by"]["n_rows_claimed"] == 1
    assert spike["rationale_source"] == "agent-reason"

    swept = [e for e in result["flags"] if e["rationale_source"] == "blanket"]
    assert swept, "the wide span should have claimed the rest"
    span = swept[0]["decided_by"]
    # Its reach is every flagged row it covers, INCLUDING the one the narrow span took —
    # that is what makes it a blanket — while n_rows_claimed is what it actually got.
    assert span["n_flagged_rows_in_span"] > span["n_rows_claimed"]
    assert span["start"] == "2024-01-01T00:00:00"

    # A row code decided is attributed to nobody: pinning it on a span the agent wrote
    # would credit it with a judgement it never made.
    assert all(e["decided_by"] is None
               for e in result["flags"] if e["rationale_source"] == "deterministic")


def test_an_anomaly_span_claims_every_row_it_covers_not_just_the_flagged_ones():
    """A windowed anomaly could not be reported at all before this (§5, 2026-08-24).

    A level shift is a SPAN sitting at the wrong level, but flagJumps flags its two
    EDGES. With one entry per flagged row, the interior was never in the log and no
    decision could reach it. Measured on 01467200_l1: the agent found the shift exactly
    and claimed 2 of 117 rows.
    """
    index = pd.date_range("2024-01-01", periods=400, freq="5min")
    values = np.full(400, 5.0)
    values[100:300] = 15.0
    qc = saqc.SaQC(pd.DataFrame({"value": pd.Series(values, index=index)}))
    qc = wrappers.flag_jumps(qc, field="value", thresh=3.0, window="1h")["qc"]

    result = wrappers.export_clean_data(qc, decisions=[
        {"start": str(index[100]), "end": str(index[299]), "difficulty": "clear",
         "verdict": "anomaly", "anomaly_type": "level_shift", "action": "delete",
         "reason": "interior 4.5 sigmas above surroundings, both edges sharp"},
    ])
    claimed = [e for e in result["flags"] if e["anomaly_type"] == "level_shift"]
    assert len(claimed) == 200, "the whole window must be claimed, not just its edges"
    assert result["n_rows_claimed_by_span_not_flagged"] == 199

    # Provenance stays honest: a row a detector found is distinguishable from a row a
    # decision reached.
    by_source = {e["flagged_by"] for e in claimed}
    assert wrappers.SPAN_CLAIMED in by_source and "flagJumps" in by_source


def test_a_wide_gap_span_does_not_materialise_the_whole_series():
    """The regression this guard exists for (2026-08-24).

    Restricting materialisation to `anomaly` spans was not enough: the agent wrote a
    perfectly reasonable whole-record catch-all — anomaly / gap / impute over two years
    — and the flag log came back with 210,816 entries, one per row of the series. Only
    `level_shift` has the edge-vs-window mismatch that needs materialising; every other
    type is already flagged row-for-row by its own detector.
    """
    index = pd.date_range("2024-01-01", periods=2000, freq="5min")
    values = np.full(2000, 5.0)
    values[500:520] = np.nan                       # a real gap, which flagNAN will flag
    qc = saqc.SaQC(pd.DataFrame({"value": pd.Series(values, index=index)}))
    qc = wrappers.flag_nan(qc, field="value")["qc"]

    result = wrappers.export_clean_data(qc, decisions=[
        {"start": str(index[0]), "end": str(index[-1]), "difficulty": "clear",
         "verdict": "anomaly", "anomaly_type": "gap", "action": "keep",
         "reason": "catch-all for every missing run in the record, none fillable"},
    ])
    assert result["n_rows_claimed_by_span_not_flagged"] == 0
    assert len(result["flags"]) == 20, "only the rows flagNAN flagged belong in the log"


def test_an_oversized_level_shift_span_is_refused_and_reported():
    """A mistyped end timestamp must not expand the log by a whole series."""
    index = pd.date_range("2024-01-01", periods=8000, freq="5min")
    qc = saqc.SaQC(pd.DataFrame({"value": pd.Series(np.full(8000, 5.0), index=index)}))
    qc = wrappers.flag_jumps(qc, field="value", thresh=0.5, window="1h")["qc"]

    result = wrappers.export_clean_data(qc, decisions=[
        {"start": str(index[0]), "end": str(index[-1]), "difficulty": "clear",
         "verdict": "anomaly", "anomaly_type": "level_shift", "action": "delete",
         "reason": "a span far wider than any level shift §9 injects (4-24h)"},
    ])
    assert result["n_rows_claimed_by_span_not_flagged"] == 0
    assert result["spans_too_wide_to_materialise"], "an oversized span must be reported"


def test_a_normal_span_does_not_materialise_rows():
    """Otherwise a whole-record catch-all keep writes an entry for every row."""
    qc = _flagged_qc()
    frame = qc.data.to_pandas()
    result = wrappers.export_clean_data(qc, decisions=[
        {"start": str(frame.index[0]), "end": str(frame.index[-1]), "difficulty": "clear",
         "verdict": "normal", "action": "keep",
         "reason": "catch-all for everything not judged individually"},
    ])
    assert result["n_rows_claimed_by_span_not_flagged"] == 0
    assert len(result["flags"]) < len(frame), "a keep span must not expand the log"


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


def test_a_broad_keep_span_cannot_relabel_rows_the_imputer_filled(tmp_path):
    """A filled row is 'impute' even under a catch-all 'keep' — the value was replaced.

    Measured on a real run (03447687_l1, 2026-08-10): the agent swept up its
    remaining spike/jump flags with one whole-series 'keep' span, which also covered
    every gap the imputer had filled. The log then said "left untouched" about 3,429
    rows whose values had been rewritten, and gap F1 fell from 1.0 to 0.
    """
    qc = wrappers.flag_nan(_toy_qc(), field="value")["qc"]
    qc = wrappers.impute_linear(qc, field="value", max_gap="6h")["qc"]
    idx = qc.data.to_pandas().index

    result = wrappers.export_clean_data(
        qc,
        decisions=[{
            "start": str(idx[0]), "end": str(idx[-1]), "difficulty": "clear", "verdict": "normal",
            "action": "keep", "reason": "catch-all: everything else is genuine water",
        }],
        output_dir=tmp_path,
        stem="toy_l1",
    )

    entries = json.loads((tmp_path / "toy_l1_flags.json").read_text())
    filled = [e for e in entries if "impute_linear" in e["flagged_by"]]
    assert filled, "the toy gap should have been filled"
    assert all(e["action"] == "impute" for e in filled)
    assert result["n_keep_rewritten_to_impute"] == len(filled)
    assert "recorded as 'impute'" in result["message"]


def test_an_explicit_delete_still_wins_over_the_impute_default(tmp_path):
    """Acting further on a filled value is a real decision; only 'keep' is false."""
    qc = wrappers.flag_nan(_toy_qc(), field="value")["qc"]
    qc = wrappers.impute_linear(qc, field="value", max_gap="6h")["qc"]
    filled_at = next(
        e["datetime"] for e in wrappers.export_clean_data(qc)["flags"]
        if "impute_linear" in e["flagged_by"]
    )

    result = wrappers.export_clean_data(
        qc,
        decisions=[{"start": filled_at, "difficulty": "clear", "verdict": "anomaly", "anomaly_type": "gap",
                    "action": "delete", "reason": "fill is unreliable"}],
    )

    entry = next(e for e in result["flags"] if e["datetime"] == filled_at)
    assert entry["action"] == "delete"


# ---------------------------------------------------------------------------
# Payload size (context cost)
# ---------------------------------------------------------------------------

def test_flagged_datetimes_are_sampled_not_truncated_to_the_head():
    """A detector must not send thousands of timestamps back on every turn.

    Measured on 03447687_l1 (2026-08-10): the timestamp lists were 34% of a 149k-token
    request and input tokens were 88% of the run's cost. The sample must be spread
    across the record — a head would put every point the agent inspects in the first
    weeks of a two-year series.
    """
    n = wrappers.MAX_FLAGGED_DATETIMES * 4
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(7)
    v = 10 + rng.normal(0, 0.05, n)
    v[::2] = 60.0                                   # flag roughly half the series
    qc = saqc.SaQC(pd.DataFrame({"value": v}, index=idx))

    result = wrappers.flag_range(qc, field="value", min=0, max=50)

    assert result["n_flagged"] > wrappers.MAX_FLAGGED_DATETIMES
    assert len(result["flagged_datetimes"]) <= wrappers.MAX_FLAGGED_DATETIMES
    assert result["n_flagged_datetimes_shown"] == len(result["flagged_datetimes"])
    # Loudly, or the agent writes decisions that silently miss thousands of rows.
    assert "showing" in result["message"] and str(result["n_flagged"]) in result["message"]
    # Spread, not a head: the sample must reach the end of the record.
    shown = pd.DatetimeIndex(result["flagged_datetimes"])
    assert shown.max() > idx[int(n * 0.9)]


def test_a_short_flag_list_is_returned_whole():
    """The cap must not cost anything on the ordinary case."""
    result = wrappers.flag_nan(_toy_qc(), field="value")

    assert result["n_flagged"] == 5
    assert len(result["flagged_datetimes"]) == 5
    assert "showing" not in result["message"]


def test_a_specific_decision_beats_a_broad_catch_all_whatever_the_order(tmp_path):
    """The narrowest span covering a row wins, not the first one listed.

    Measured on 03447687_l1 (2026-08-11): the agent measured a point, read
    `reads_like=spike, robust_z=7.4, width=3`, wrote a `delete` for it, and the log
    recorded `keep` — because a whole-series catch-all appeared earlier in the list
    and first-covering-span-wins handed it the row. A broad span is a statement about
    what is left over; it must lose to anything more specific.
    """
    qc = _flagged_qc()
    idx = qc.data.to_pandas().index
    spike_at = str(idx[50])

    result = wrappers.export_clean_data(
        qc,
        decisions=[
            # Catch-all FIRST — the ordering that used to silently win.
            {"start": str(idx[0]), "end": str(idx[-1]), "difficulty": "clear", "verdict": "normal",
             "action": "keep", "reason": "everything else is genuine water"},
            {"start": spike_at, "difficulty": "clear", "verdict": "anomaly", "anomaly_type": "spike",
             "action": "delete", "reason": "robust_z 7.4, width 3"},
        ],
    )

    entry = next(e for e in result["flags"]
                 if e["datetime"] == pd.Timestamp(spike_at).strftime("%Y-%m-%dT%H:%M:%S"))
    assert entry["action"] == "delete"
    assert "7.4" in entry["reason"]
    # The catch-all still does its job on every row the specific span did not claim.
    assert result["n_by_action"]["keep"] > 0


def test_a_broad_span_whose_rows_are_all_claimed_is_not_reported_as_unmatched(tmp_path):
    """Only a span matching NO flagged row is an error worth telling the agent about."""
    qc = _flagged_qc()
    idx = qc.data.to_pandas().index

    result = wrappers.export_clean_data(
        qc,
        decisions=[
            {"start": str(idx[0]), "end": str(idx[-1]), "difficulty": "clear", "verdict": "normal",
             "action": "keep", "reason": "rest"},
            {"start": str(idx[50]), "difficulty": "clear", "verdict": "anomaly", "anomaly_type": "spike",
             "action": "delete", "reason": "spike"},
            {"start": "2019-01-01T00:00:00", "difficulty": "clear", "verdict": "normal",
             "action": "keep", "reason": "typo, matches nothing"},
        ],
    )

    unmatched = result["decisions_matching_no_flagged_row"]
    assert len(unmatched) == 1 and "2019" in unmatched[0]


def test_slope_ratio_separates_a_flush_event_from_an_artifact():
    """The fall/rise ratio must be measured at the scale the decay happens on.

    A small first-flush event and a debris strike are both 1-3 samples wide and the
    flush often has the HIGHER robust_z, so width and z cannot tell them apart. What
    separates them is that the flush decays over 3-5 samples while the artifact snaps
    back. That only shows at a ~45-minute window: past ~90 min the decay is over, the
    window fills with flat surroundings, and the ordering reverses (CLAUDE.md §7.3).
    """
    idx = pd.date_range("2024-01-01", periods=200, freq="15min")
    base = 10 + np.zeros(200)

    flush = base.copy()
    flush[100:106] = [40.0, 28.0, 21.0, 16.0, 13.0, 11.0]   # one-sample rise, long decay
    artifact = base.copy()
    artifact[100] = 40.0                                     # rises and snaps back

    def ratio(values, n):
        qc = saqc.SaQC(pd.DataFrame({"value": values}, index=idx))
        return wrappers.describe_point(
            qc, at=str(idx[100]), n_before=n, n_after=n
        )["slope"]["fall_rise_ratio"]

    # At the default 45-minute window the flush is clearly gentler on the way down.
    assert ratio(flush, 3) < 0.8 < ratio(artifact, 3)
    # And the compact row the agent actually reads carries it.
    qc = saqc.SaQC(pd.DataFrame({"value": flush}, index=idx))
    row = wrappers.describe_points(qc, ats=[str(idx[100])])["points"][0]
    assert row["fall_rise_ratio"] < 0.8
    assert row["samples_to_recover"] >= 3


def test_a_judgement_call_must_carry_its_reasoning(tmp_path):
    """Marking a decision 'judgement-call' obliges the agent to show its working."""
    qc = _flagged_qc()
    at = str(qc.data.to_pandas().index[50])

    with pytest.raises(ValueError, match="judgement-call"):
        wrappers.export_clean_data(qc, decisions=[
            {"start": at, "verdict": "anomaly", "anomaly_type": "spike",
             "action": "delete", "reason": "spike",
             "difficulty": "judgement-call"},          # no deliberation
        ])

    ok = wrappers.export_clean_data(qc, decisions=[
        {"start": at, "verdict": "anomaly", "anomaly_type": "spike",
         "action": "delete", "reason": "spike",
         "difficulty": "judgement-call",
         "deliberation": "Width 1 and robust_z 14 say artifact, but the fall decays over "
                         "3 samples which argues flush. Deleted because the recovery is "
                         "within noise; a 5-sample decay would have changed my mind."},
    ])
    entry = next(e for e in ok["flags"]
                 if e["datetime"] == pd.Timestamp(at).strftime("%Y-%m-%dT%H:%M:%S"))
    assert entry["rationale_source"] == "agent-deliberation"
    assert "changed my mind" in entry["deliberation"]


def test_a_blanket_span_is_labelled_as_one_and_reported(tmp_path):
    """A span covering most of the flags speaks for the remainder, not for any one row.

    Every run so far had one of these absorb points the agent had actually measured,
    and the log could not distinguish that from a considered call (§5).
    """
    qc = _flagged_qc()
    idx = qc.data.to_pandas().index

    result = wrappers.export_clean_data(qc, decisions=[
        {"start": str(idx[0]), "end": str(idx[-1]), "difficulty": "clear", "verdict": "normal",
         "action": "keep", "reason": "the rest"},
        {"start": str(idx[50]), "difficulty": "clear", "verdict": "anomaly", "anomaly_type": "spike",
         "action": "delete", "reason": "robust_z 40, width 1"},
    ])

    by_source = result["n_by_rationale_source"]
    assert by_source.get("blanket", 0) > 0
    assert by_source.get("agent-reason", 0) == 1
    assert "blanket" in result["message"]

    blanketed = next(e for e in result["flags"] if e["rationale_source"] == "blanket")
    assert "blanket" in blanketed["rationale"]
    # And a row nothing covered says so in its own words rather than looking decided.
    plain = wrappers.export_clean_data(qc)["flags"][0]
    assert plain["rationale_source"] == "deterministic"
    assert "never adjudicated" in plain["rationale"]


def test_decision_audit_does_not_hide_an_overwritten_anomaly_as_a_success():
    """An anomaly the imputer overwrote is a miss, and must not be filed as `imputed`.

    On 03447687_l1 the page reported missed=0 while 10 injected spikes and 3 level
    shifts carried action=impute — overwritten by interpolateByRolling (the §7.1
    dfilter bug) before the agent judged them. Folding those into `imputed` made a
    silent failure look like successful gap-filling.
    """
    from src.workbench.decision_audit import categorise, UNHANDLED_ANOMALY

    assert categorise("impute", "gap", flagged=True) == "imputed"
    assert categorise("impute", "spike", flagged=True) == "overwritten-anomaly"
    assert categorise("impute", "", flagged=True) == "overwritten-water"
    assert categorise("keep", "spike", flagged=True) == "missed"
    assert categorise("keep", "gap", flagged=True) == "left-missing"
    assert categorise("", "spike", flagged=False) == "undetected"
    assert categorise("delete", "", flagged=True) == "wrongly-deleted"

    # The three ways an anomaly can go unhandled must all be counted as such.
    assert set(UNHANDLED_ANOMALY) == {"missed", "overwritten-anomaly", "undetected"}


def test_impute_linear_does_not_overwrite_a_flagged_reading():
    """The imputer must fill genuine gaps only, never a row a detector flagged.

    SaQC masks rows whose flag is >= `dfilter` before a function runs, so without
    dfilter=inf an already-flagged reading looks missing to the imputer and gets
    replaced by a rolling median. Measured on 03447687_l1 (2026-08-10): 1,887 real
    readings silently rewritten, and 13 injected anomalies never judged because the
    overwrite gave them action=impute (§7.1).
    """
    idx = pd.date_range("2024-01-01", periods=60, freq="15min")
    values = 10 + np.zeros(60)
    values[20] = 80.0            # a real reading a detector will flag
    values[40:43] = np.nan       # a genuine gap

    qc = saqc.SaQC(pd.DataFrame({"value": values}, index=idx))
    qc = wrappers.flag_range(qc, field="value", min=0, max=50)["qc"]
    result = wrappers.impute_linear(qc, field="value", max_gap="3h")

    got = result["qc"].data.to_pandas()["value"]
    assert got.iloc[20] == 80.0, "the flagged reading was overwritten by the imputer"
    assert got.iloc[40:43].notna().all(), "the genuine gap was not filled"
    # The imputer's own flag history must now count only what it actually filled.
    assert result["n_flagged"] == result["n_imputed"] == 3


def test_flag_jumps_rejects_a_threshold_that_flags_everything():
    """§13 fail loudly: thresh<=0 or a missing window is a mis-set parameter, not a run.

    thresh has no portable default -- it is 6.2 / 13.8 / 75.7 FNU on the three project
    gauges -- so silently defaulting it produced runs that flagged thousands of storm
    limbs and concluded nothing (§7.6).
    """
    idx = pd.date_range("2024-01-01", periods=300, freq="15min")
    qc = saqc.SaQC(pd.DataFrame({"value": np.linspace(1.0, 2.0, 300)}, index=idx))
    with pytest.raises(ValueError, match="must be > 0"):
        wrappers.flag_jumps(qc, thresh=0.0, window="1h")
    with pytest.raises(ValueError, match="jump_scale"):
        wrappers.flag_jumps(qc, thresh=1.0)


def test_inspect_dataset_measures_the_flag_jumps_threshold():
    """inspect_dataset must hand the run a usable flag_jumps.thresh (§7.6)."""
    idx = pd.date_range("2024-01-01", periods=800, freq="15min")
    rng = np.random.default_rng(0)
    values = 10.0 + rng.normal(0, 0.05, 800)
    values[400:] += 8.0
    qc = saqc.SaQC(pd.DataFrame({"value": values}, index=idx))

    res = wrappers.inspect_dataset(qc)
    block = res["jump_scale"]
    thresh = block["recommended_thresh"]
    assert thresh > 0
    assert block["recommended_window"] in block["by_window"]

    # It has to be a threshold flag_jumps can actually be run with.
    out = wrappers.flag_jumps(res["qc"], thresh=thresh,
                              window=block["recommended_window"])
    assert 0 < out["n_flagged"] < 0.05 * len(idx)

    # And the summary must carry the robust pair the prompt tells the agent to
    # scale from -- `std` alone is what produced the 2,865-flag run.
    stats = res["summary"]["columns"]["value"]
    assert stats["median"] is not None and stats["robust_sigma"] > 0


# --- §7.7 rainfall gate + the anomaly/keep restriction (2026-08-25) ------------

def _qc_with_one_flagged_spike():
    import numpy as np
    import pandas as pd
    import saqc
    idx = pd.date_range("2024-01-01", periods=60, freq="5min")
    values = pd.Series(np.full(60, 5.0), index=idx)
    values.iloc[20] = 500.0
    qc = saqc.SaQC(pd.DataFrame({"value": values}))
    return qc.flagRange("value", min=0, max=100), idx[20].strftime("%Y-%m-%dT%H:%M:%S")


def test_deleting_a_spike_never_checked_against_rain_is_refused():
    """§7.7. The prompt has required this since v0.13 and it never once happened —
    the dispatch branch was missing, so the tool always errored. Prompt text cannot
    enforce a requirement; this can."""
    from src.agent_tools.wrappers import export_clean_data
    qc, at = _qc_with_one_flagged_spike()
    decisions = [{"start": at, "difficulty": "clear", "verdict": "anomaly",
                  "anomaly_type": "spike", "action": "delete", "reason": "500 NTU spike"}]
    with pytest.raises(ValueError, match="without ever being checked against rainfall"):
        export_clean_data(qc=qc, decisions=decisions, precip_audited=frozenset())

    # audited -> allowed
    out = export_clean_data(qc=qc, decisions=decisions, precip_audited=frozenset({at}))
    assert out["n_entries"] == 1


def test_the_rain_gate_covers_the_obvious_deletions_too_but_not_other_verdicts():
    """The user's point, and §7.7's: a clear-cut-looking spike is exactly the kind of
    call rainfall overturns, so `difficulty` does not exempt it. Equally, the gate is
    about DESTROYING a value — keeping one, or judging another type, is untouched."""
    from src.agent_tools.wrappers import export_clean_data
    qc, at = _qc_with_one_flagged_spike()

    def export(**over):
        d = {"start": at, "difficulty": "clear", "verdict": "anomaly",
             "anomaly_type": "spike", "action": "delete",
             "reason": "clear-cut single-sample excursion"}
        d.update(over)
        return export_clean_data(qc=qc, decisions=[d], precip_audited=frozenset())

    with pytest.raises(ValueError, match="rainfall"):
        export()                                    # "clear" is not an exemption
    with pytest.raises(ValueError, match="rainfall"):
        export(action="correct")                    # correcting destroys it too
    # rejecting the detector needs no rain check: nothing is removed
    assert export(verdict="normal", anomaly_type="", action="keep")["n_entries"] == 1


def test_an_anomaly_verdict_may_not_ship_the_values_it_calls_faulty():
    """Run I called the injected level shift an artifact — both edges moving in one
    sample, no elevated-noise stretch on that date — and then kept all 118 rows,
    because §6 makes `keep` the default and nothing made it revisit that. Only a gap
    genuinely has no treatment available."""
    from src.agent_tools.wrappers import export_clean_data
    qc, at = _qc_with_one_flagged_spike()
    kept = {"start": at, "difficulty": "judgement-call", "verdict": "anomaly",
            "anomaly_type": "level_shift", "action": "keep",
            "deliberation": "edges move in one sample; not in any noisy stretch",
            "reason": "sharp step, both edges one sample, no storm on this date"}
    with pytest.raises(ValueError, match="ships data you have just said is faulty"):
        export_clean_data(qc=qc, decisions=[kept], precip_audited=None)

    # a gap is the one type it stays legal for
    gap = {**kept, "anomaly_type": "gap",
           "reason": "11-day outage, far longer than any defensible imputation window"}
    assert export_clean_data(qc=qc, decisions=[gap], precip_audited=None)["n_entries"] == 1


# --- level shift is CORRECTED, not deleted (2026-08-25) -----------------------

def test_correcting_a_level_shift_recovers_the_water_under_it():
    """A shift is an OFFSET: the water underneath moved normally, so the shape inside
    the window is real data at the wrong height. Measured on 01467200_l1's injected
    shift, correction takes interior error from 6.95 FNU to 0.78 FNU against the true
    values — deleting those 117 rows would have thrown all of it away."""
    import numpy as np
    import pandas as pd
    import saqc

    from src.agent_tools.wrappers import correct_level_shift

    root = Path("data/injected/01467200/l1")
    if not (root / "01467200_l1.csv").exists():
        pytest.skip("injected dataset not present")
    s_ = pd.read_csv(root / "01467200_l1.csv", parse_dates=["datetime"]
                     ).set_index("datetime")["value"]
    lab = pd.read_csv(root / "01467200_l1_labels.csv", parse_dates=["datetime"])
    sh = lab.loc[lab.anomaly_type == "level_shift", "datetime"]
    lo, hi = sh.min(), sh.max()

    out = correct_level_shift(qc=saqc.SaQC(pd.DataFrame({"value": s_})),
                              start=str(lo), end=str(hi))
    got = out["qc"].data.to_pandas()["value"]
    truth = lab.set_index("datetime")["true_value"]
    inside = (got.index >= lo) & (got.index <= hi)

    def rmse(v):
        return float(np.sqrt(((v[inside] - truth[inside]) ** 2).mean()))

    assert rmse(got) < rmse(s_) / 4, "correction should recover most of the offset"
    # and it must not touch anything else
    changed = (got - s_).abs() > 1e-9
    assert int(changed[(changed.index < lo) | (changed.index > hi)].sum()) == 0


def test_the_correction_writes_through_saqcs_flag_mask():
    """§7.1's dfilter trap, third appearance. SaQC masks rows whose flag is >= dfilter
    before a function runs, and processGeneric's DEFAULT (-inf) masks everything — so
    the window, which flag_jumps has by definition already flagged, arrives as NaN and
    the correction silently writes nothing while returning a normal result dict.
    Probed: interior median 16.00 at the default, 10.00 at inf."""
    import numpy as np
    import pandas as pd
    import saqc

    from src.agent_tools.wrappers import correct_level_shift

    idx = pd.date_range("2024-01-01", periods=300, freq="5min")
    v = pd.Series(np.full(300, 10.0), index=idx)
    v.iloc[100:200] += 6.0
    qc = saqc.SaQC(pd.DataFrame({"value": v})).flagRange("value", min=0, max=14)
    out = correct_level_shift(qc=qc, start=str(idx[100]), end=str(idx[199]))
    got = out["qc"].data.to_pandas()["value"]
    assert abs(float(got.iloc[100:200].median()) - 10.0) < 0.5, (
        "the flagged window was masked out and the correction was lost")


def test_a_window_whose_edges_disagree_is_reported_not_silently_corrected():
    """A clean level shift steps up and back down by the same amount. A storm does not,
    and a confident-looking offset on one is how real water gets rewritten."""
    import numpy as np
    import pandas as pd
    import saqc

    from src.agent_tools.wrappers import correct_level_shift

    idx = pd.date_range("2024-01-01", periods=300, freq="5min")
    v = pd.Series(np.full(300, 10.0), index=idx)
    v.iloc[100:200] += 6.0
    v.iloc[200:] += 6.0                      # steps up and never comes back
    out = correct_level_shift(qc=saqc.SaQC(pd.DataFrame({"value": v})),
                              start=str(idx[100]), end=str(idx[199]))
    assert out["edges_agree"] is False
    assert "disagree" in out["message"]


def test_deleted_values_are_actually_gone_from_the_cleaned_file():
    """§5 says the cleaned file carries deleted values as NaN and §1 promises a cleaned
    dataset — but the frame came straight off qc.data with no action applied, so every
    deleted value was still there at its original reading. Verified on run L:
    2023-07-05T13:45 was action=delete in the flag log and read 16.1 in the clean CSV."""
    from src.agent_tools.wrappers import export_clean_data
    qc, at = _qc_with_one_flagged_spike()
    out = export_clean_data(
        qc=qc, precip_audited=frozenset({at}),
        decisions=[{"start": at, "difficulty": "clear", "verdict": "anomaly",
                    "anomaly_type": "spike", "action": "delete",
                    "reason": "500 NTU single-sample excursion"}])
    row = out["df"].loc[pd.Timestamp(at), "value"]
    assert out["n_values_deleted"] == 1
    assert row != 500.0, "the deleted spike must not survive into the cleaned file"
    # ...and a deleted value IS a gap, so a 1-sample hole is interpolated like any other
    # (§: one rule, MAX_FILL_GAP). Run N left 218 permanent holes because deletions were
    # applied after the imputer had already run.
    assert out["n_rows_refilled_after_delete"] == 1
    assert abs(float(row) - 5.0) < 0.01, f"expected the interpolated baseline, got {row}"


def test_correcting_does_not_shadow_builtins_in_the_wrappers_module():
    """SaQC's processGeneric injects its 35-name evaluation environment into the
    __globals__ of whatever callable it is handed, and six of those names shadow
    builtins: abs, id, len, max, min, sum. One call with a function defined in
    wrappers.py therefore replaces `max` and `len` for every OTHER function in that
    module for the rest of the process — which made `_build_flag_log`'s
    `max(len(stamps), 1)` raise `AxisError: axis 1 is out of bounds` inside a test
    that never touched this tool. The failure is remote, silent and order-dependent,
    so it gets its own guard.
    """
    import numpy as np
    import pandas as pd
    import saqc

    from src.agent_tools import wrappers as w

    idx = pd.date_range("2024-01-01", periods=200, freq="5min")
    v = pd.Series(np.full(200, 10.0), index=idx)
    v.iloc[50:150] += 5.0
    w.correct_level_shift(qc=saqc.SaQC(pd.DataFrame({"value": v})),
                          start=str(idx[50]), end=str(idx[149]))

    for name in ("max", "min", "len", "sum", "abs", "id"):
        assert name not in vars(w), (
            f"processGeneric leaked `{name}` into the wrappers module globals, "
            "shadowing the builtin for every function in it")


# --- one fill rule, applied everywhere (2026-08-25) ---------------------------

def test_a_gap_is_filled_whole_or_not_at_all():
    """The rolling median filled only where its window found context, so it repaired
    the ENDS of a gap and left the middle — 23 of 27, 23 of 35 and 23 of 55 rows on
    three of the four injected gap events on 01467200_l1. That biased the error metric
    optimistically, because the rows it declined were the ones furthest from any real
    reading."""
    import numpy as np
    import pandas as pd
    import saqc

    idx = pd.date_range("2024-01-01", periods=200, freq="5min")
    v = pd.Series(np.linspace(10.0, 20.0, 200), index=idx)
    v.iloc[100:110] = np.nan                       # 50 min — inside the 1h cap
    out = wrappers.impute_linear(qc=saqc.SaQC(pd.DataFrame({"value": v})))
    got = out["qc"].data.to_pandas()["value"]
    assert int(got.iloc[100:110].isna().sum()) == 0, "the gap must be filled whole"
    # and the fill follows the TREND, which is why it beats a median on a slope:
    # a centred median would return the local level, not the value on the ramp
    expected = np.linspace(10.0, 20.0, 200)[105]
    assert abs(float(got.iloc[105]) - expected) < 0.05


def test_a_gap_longer_than_the_cap_is_left_entirely_alone():
    import numpy as np
    import pandas as pd
    import saqc

    idx = pd.date_range("2024-01-01", periods=400, freq="5min")
    v = pd.Series(np.full(400, 10.0), index=idx)
    v.iloc[100:150] = np.nan                       # 4h10m — well over the 1h cap
    out = wrappers.impute_linear(qc=saqc.SaQC(pd.DataFrame({"value": v})))
    got = out["qc"].data.to_pandas()["value"]
    assert int(got.iloc[100:150].isna().sum()) == 50, "no partial fill at the edges"
    assert out["n_gaps_skipped_too_long"] == 1
    assert out["n_imputed"] == 0


def test_the_imputer_never_extrapolates_off_the_end_of_the_record():
    """A NaN run with data on only one side has no two points to interpolate between."""
    import numpy as np
    import pandas as pd
    import saqc

    idx = pd.date_range("2024-01-01", periods=100, freq="5min")
    v = pd.Series(np.full(100, 10.0), index=idx)
    v.iloc[:6] = np.nan
    v.iloc[-6:] = np.nan
    out = wrappers.impute_linear(qc=saqc.SaQC(pd.DataFrame({"value": v})))
    got = out["qc"].data.to_pandas()["value"]
    assert int(got.iloc[:6].isna().sum()) == 6
    assert int(got.iloc[-6:].isna().sum()) == 6


def test_a_deleted_plateau_is_too_long_to_refill_and_stays_an_honest_hole():
    """The same one-hour rule that refills a deleted spike leaves a deleted plateau
    missing. That is the whole point of having one rule rather than a rule per type."""
    import numpy as np
    import pandas as pd
    import saqc

    idx = pd.date_range("2024-01-01", periods=400, freq="5min")
    v = pd.Series(np.linspace(5.0, 9.0, 400), index=idx)
    v.iloc[100:150] = 7.0                          # a stuck sensor, 4h10m
    qc = saqc.SaQC(pd.DataFrame({"value": v})).flagConstants(
        "value", thresh=0.01, window="1h", min_periods=2)
    out = wrappers.export_clean_data(
        qc=qc, precip_audited=None,
        decisions=[{"start": str(idx[100]), "end": str(idx[149]), "difficulty": "clear",
                    "verdict": "anomaly", "anomaly_type": "plateau", "action": "delete",
                    "reason": "sensor stuck at 7.0 for over four hours"}])
    vals = out["df"]["value"]
    assert int(vals.iloc[100:150].isna().sum()) == 50, (
        "a multi-hour deleted plateau must stay NaN, not be interpolated")
    assert out["n_rows_refilled_after_delete"] == 0


def test_one_junk_reading_at_a_gap_edge_does_not_smear_across_the_whole_gap():
    """§6 notes artifacts cluster at gap EDGES. A line anchored on the two individual
    readings either side is hostage to both: measured across all nine datasets, one
    corrupted edge reading took point-anchored linear from RMSE 1.659 to 54.562, while
    the median anchor moved 1.719 -> 1.720. That robustness is the entire reason the
    anchors are medians rather than points."""
    import numpy as np
    import pandas as pd

    from src.agent_tools.wrappers import _linear_fill

    idx = pd.date_range("2024-01-01", periods=120, freq="5min")
    true = pd.Series(np.linspace(10.0, 16.0, 120), index=idx)
    obs = true.copy()
    obs.iloc[60:66] = np.nan            # 30 min, inside the cap
    obs.iloc[59] = 900.0                # telemetry junk immediately before the gap

    filled, touched, _ = _linear_fill(obs, "1h")
    assert len(touched) == 6
    err = float(np.abs(filled.iloc[60:66] - true.iloc[60:66]).max())
    assert err < 2.0, f"the junk edge reading leaked into the fill (max err {err:.1f})"


def test_spike_scale_sizes_both_detectors_from_the_record_itself():
    """§7.10. No summary statistic predicts flagUniLOF's thresh: measured across the nine
    datasets it tracks the CONTAMINATION LEVEL (Spearman +0.78), which a run cannot know,
    and only +0.57 with the best record statistic on three gauges. So the threshold is
    read off this record's own |LOF| distribution and lands the wanted candidate count on
    any gauge — 1.95 on one, 3.45 on another, both ~175 candidates."""
    import warnings

    import numpy as np
    import pandas as pd
    import saqc

    from src.agent_tools import context as ctx

    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=6000, freq="5min")
    v = pd.Series(10 + rng.normal(0, 0.4, 6000), index=idx)
    for i in range(200, 6000, 400):
        v.iloc[i] += 8.0
    out = ctx.spike_scale(saqc.SaQC(pd.DataFrame({"value": v})),
                          lof_budget=60, zscore_budget=40)
    assert out["lof_n"] == ctx.SPIKE_LOF_N == 10
    assert out["recommended_lof_thresh"] == ctx.SPIKE_LOF_THRESH
    # the floor that stops a collapsed windowed MAD making thresh meaningless (§7.1)
    assert out["zscore_min_residuals"] > 0
    assert "Run BOTH" in out["message"]


def test_the_lof_neighbourhood_default_is_not_the_stale_15_minute_one():
    """n counts SAMPLES, so n=20 meant 5 hours on the retired 15-min bases and means 100
    minutes now — a different question about the data. At equal candidate budget n=10 beat
    n=20 on 8 of 9 datasets and tied on the ninth (median recall 0.688 vs 0.521)."""
    import inspect

    from src.agent_tools import context as ctx
    from src.agent_tools import wrappers as w

    assert inspect.signature(w.flag_spike_unilof).parameters["n"].default == ctx.SPIKE_LOF_N
    assert ctx.SPIKE_LOF_N == 10


# --- §7.11 difficulty is DERIVED, and a blanket cannot settle a contested point ------

def _conflicted_fixture():
    """Ten flagged spikes, one of which has shape and rainfall disagreeing.

    Ten and not one: a span covering the only flagged row is 100% of the run's flags and
    is recorded as a blanket by definition (BLANKET_SHARE), so a one-row fixture cannot
    tell "judged on its own" from "swept up".
    """
    import numpy as np
    import pandas as pd
    import saqc

    idx = pd.date_range("2024-01-01", periods=200, freq="5min")
    v = pd.Series(np.full(200, 5.0), index=idx)
    for i in range(20, 120, 10):
        v.iloc[i] = 500.0
    qc = saqc.SaQC(pd.DataFrame({"value": v})).flagRange("value", min=0, max=100)
    at = idx[20].strftime("%Y-%m-%dT%H:%M:%S")
    return qc, idx, at, {at: {"reads_like": "spike", "rained": True, "noise_ratio": 1.4}}


def _conflicted_export(**over):
    """A catch-all over everything, unless the caller narrows it."""
    qc, idx, at, evidence = _conflicted_fixture()
    decision = {"start": str(idx[0]), "end": str(idx[-1]), "difficulty": "clear",
                "verdict": "normal", "action": "keep",
                "reason": "catch-all for everything flagged in this stretch"}
    decision.update(over)
    return wrappers.export_clean_data(
        qc=qc, precip_audited=None, decisions=[decision], evidence=evidence)


def test_a_blanket_cannot_settle_a_point_whose_evidence_disagrees():
    """Run P swept 21 of the 26 points whose shape and rainfall disagreed into a
    whole-record catch-all marked 'clear'. One was an injected spike it had itself
    measured at 8.8 robust sigmas. Conflicted points are decided wrong 69.4% of the
    time against 16.6% elsewhere — they are the last rows a catch-all should claim."""
    with pytest.raises(ValueError, match="swept up by a catch-all"):
        _conflicted_export()


def _export_with_own_span(**over):
    """The conflicted point gets its own narrow span; everything else a catch-all."""
    qc, idx, at, evidence = _conflicted_fixture()
    own = {"start": at, "end": at, "difficulty": "clear", "verdict": "anomaly",
           "anomaly_type": "spike", "action": "delete",
           "reason": "single-sample excursion at 30 robust sigmas"}
    own.update(over)
    rest = {"start": str(idx[0]), "end": str(idx[-1]), "difficulty": "clear",
            "verdict": "normal", "action": "keep", "reason": "the rest is real water"}
    return wrappers.export_clean_data(
        qc=qc, precip_audited=None, decisions=[own, rest], evidence=evidence)


def test_a_contested_point_cannot_be_called_clear_cut():
    """`difficulty` was self-declared and unchecked, so a span covering 1,977 rows
    including a genuine spike was recorded 'clear'. §10.1's review queue is built from
    exactly that field."""
    with pytest.raises(ValueError, match="difficulty='clear' on points whose"):
        _export_with_own_span()


def test_a_contested_point_judged_on_its_own_is_accepted():
    out = _export_with_own_span(
        difficulty="judgement-call",
        deliberation="Shape says artifact: one sample, 30 sigmas, instant recovery. "
                     "Rain says maybe real, but only 0.19in and none within 3h. Shape wins.")
    assert out["n_entries"] >= 1


def test_the_conflict_rules_are_the_ones_measured():
    assert wrappers.evidence_conflicts({"reads_like": "spike", "rained": True})
    assert wrappers.evidence_conflicts({"reads_like": "spike", "noise_ratio": 3.0})
    assert not wrappers.evidence_conflicts({"reads_like": "spike", "rained": False,
                                            "noise_ratio": 0.9})
    assert not wrappers.evidence_conflicts(None)


def test_a_library_call_without_measurements_is_not_gated():
    """`evidence=None` is what a test or library call gets — the check needs measurements
    to mean anything, and failing without them would break every non-agent caller."""
    import numpy as np
    import pandas as pd
    import saqc
    idx = pd.date_range("2024-01-01", periods=60, freq="5min")
    v = pd.Series(np.full(60, 5.0), index=idx); v.iloc[20] = 500.0
    qc = saqc.SaQC(pd.DataFrame({"value": v})).flagRange("value", min=0, max=100)
    out = wrappers.export_clean_data(qc=qc, precip_audited=None, evidence=None,
        decisions=[{"start": str(idx[0]), "end": str(idx[-1]), "difficulty": "clear",
                    "verdict": "normal", "action": "keep", "reason": "all normal water"}])
    assert out["n_entries"] >= 1


def test_a_missing_row_stays_a_gap_whatever_span_covers_it():
    """§9 lets a plateau or level_shift span dropouts; §5 says every missing run IS a
    gap. Run P's six gap 'errors' were all NaN rows retyped by a covering span."""
    import numpy as np
    import pandas as pd
    import saqc
    idx = pd.date_range("2024-01-01", periods=200, freq="5min")
    v = pd.Series(np.linspace(5, 9, 200), index=idx)
    v.iloc[60:100] = 7.0
    v.iloc[70:73] = np.nan
    qc = saqc.SaQC(pd.DataFrame({"value": v})).flagConstants(
        "value", thresh=0.01, window="1h", min_periods=2)
    qc = wrappers.flag_nan(qc, field="value")["qc"]
    out = wrappers.export_clean_data(qc=qc, precip_audited=None, decisions=[
        {"start": str(idx[60]), "end": str(idx[99]), "difficulty": "clear",
         "verdict": "anomaly", "anomaly_type": "plateau", "action": "delete",
         "reason": "sensor stuck at 7.0 for over three hours"}])
    types = {e["datetime"]: e["anomaly_type"] for e in out["flags"]}
    for i in (70, 71, 72):
        k = idx[i].strftime("%Y-%m-%dT%H:%M:%S")
        assert types[k] == "gap", f"{k} was retyped to {types[k]}"
    assert out["n_missing_rows_retyped_to_gap"] == 3


def test_the_candidate_count_tracks_contamination_rather_than_a_quota():
    """The threshold is ABSOLUTE, so a record with few anomalies must yield few
    candidates. The quota it replaced emitted ~175 on every record whatever it held:
    measured over 18 datasets its count correlated -0.29 with the number of injected
    spikes, against +0.72 for this rule, and on a gauge with 21 real spikes it produced
    168 candidates and capped precision at 0.13 before any judgement was made."""
    import warnings

    import numpy as np
    import pandas as pd
    import saqc

    from src.agent_tools import context as ctx

    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=6000, freq="5min")

    def scale_for(every):
        v = pd.Series(10 + rng.normal(0, 0.4, 6000), index=idx)
        for i in range(200, 6000, every):
            v.iloc[i] += 9.0
        return ctx.spike_scale(saqc.SaQC(pd.DataFrame({"value": v})))

    sparse, dense = scale_for(900), scale_for(120)
    assert sparse["recommended_lof_thresh"] == dense["recommended_lof_thresh"], \
        "the THRESHOLD is what should stay put between records"
    assert dense["lof_candidates_at_thresh"] > sparse["lof_candidates_at_thresh"], \
        "the COUNT is what should move with contamination"


def test_the_budget_survives_only_as_a_ceiling_and_says_when_it_binds():
    import warnings

    import numpy as np
    import pandas as pd
    import saqc

    from src.agent_tools import context as ctx

    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=4000, freq="5min")
    v = pd.Series(10 + rng.normal(0, 0.4, 4000), index=idx)
    for i in range(50, 4000, 6):
        v.iloc[i] += 9.0
    out = ctx.spike_scale(saqc.SaQC(pd.DataFrame({"value": v})), lof_budget=40)
    assert out["lof_thresh_capped"] is True
    assert out["lof_candidates_at_thresh"] <= 60
    assert "ceiling" in out["message"]
