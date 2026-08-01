"""Tests for the detection metrics in ``src.evaluate``."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import evaluate as ev


def _index(n: int = 10) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="15min")


def _labels(index: pd.DatetimeIndex, marks: dict[int, str]) -> pd.DataFrame:
    """§5 labels frame where ``marks`` maps row position -> anomaly_type."""
    types = [marks.get(i, "") for i in range(len(index))]
    return pd.DataFrame({
        ev.DATETIME_COL: index,
        "is_anomaly": [bool(t) for t in types],
        "anomaly_type": types,
        "true_value": [None] * len(index),
        "source": ["injected" if t else "" for t in types],
    })


def test_score_computes_per_type_prf(tmp_path):
    """A hand-checkable case: 2 true spikes, 1 hit, 1 miss, 1 false alarm."""
    index = _index()
    labels = _labels(index, {2: "spike", 3: "spike"})
    predictions = {"spike": {index[2], index[7]}}  # hits row 2, misses 3, invents 7

    scores, macro_f1 = ev.score(predictions, labels, index)
    spike = next(s for s in scores if s.anomaly_type == "spike")

    assert (spike.n_true, spike.n_pred) == (2, 2)
    assert spike.precision == pytest.approx(0.5)   # 1 of 2 flagged rows is a spike
    assert spike.recall == pytest.approx(0.5)      # 1 of 2 spike rows was flagged
    assert spike.f1 == pytest.approx(0.5)
    # Types with no labels and no predictions score 0 and still drag macro-F1 down.
    assert macro_f1 == pytest.approx(0.5 / 4)


def test_score_handles_a_type_with_no_predictions(tmp_path):
    """A detector that never fired scores 0, not NaN or a crash."""
    index = _index()
    labels = _labels(index, {4: "plateau"})
    scores, _ = ev.score({}, labels, index)
    plateau = next(s for s in scores if s.anomaly_type == "plateau")
    assert (plateau.n_pred, plateau.precision, plateau.recall, plateau.f1) == (0, 0.0, 0.0, 0.0)


def test_imputation_is_not_counted_as_gap_detection():
    """interpolateByRolling is an action; scoring its rows would double-count gaps."""
    assert "interpolateByRolling" not in ev.TOOL_TO_TYPE
    assert "impute_rolling" not in ev._WRAPPER_TO_TYPE


def test_wrapper_and_method_tables_agree():
    """The two lookup tables are derived, so they cannot drift apart."""
    for wrapper, method in ev._WRAPPER_TO_METHOD.items():
        assert ev._WRAPPER_TO_TYPE[wrapper] == ev.TOOL_TO_TYPE[method]
    assert set(ev._WRAPPER_TO_TYPE.values()) <= set(ev.SCORED_TYPES)


def test_a_flagged_row_the_agent_kept_is_not_a_false_positive():
    """The whole point of decision-aware scoring (§6: keep genuine extremes).

    Row 2 is a real spike the agent deleted; row 7 is a storm peak it flagged,
    inspected and kept. Scoring flags punishes the keep; scoring decisions does
    not — and the storm row becomes a true negative rather than a false positive.
    """
    index = _index()
    labels = _labels(index, {2: "spike"})          # row 7 is real water, not an anomaly
    predictions = {"spike": {index[2], index[7]}}  # the detector fired on both

    flags_only, _ = ev.score(predictions, labels, index)
    spike_flags = next(s for s in flags_only if s.anomaly_type == "spike")
    assert spike_flags.precision == pytest.approx(0.5)  # the kept storm counts against it

    decided, _ = ev.score(predictions, labels, index, decisions={
        index[2]: "delete",
        index[7]: "keep",
    })
    spike_decided = next(s for s in decided if s.anomaly_type == "spike")
    assert spike_decided.n_pred == 1                    # only the deleted row is a claim
    assert spike_decided.precision == pytest.approx(1.0)
    assert spike_decided.recall == pytest.approx(1.0)
    assert spike_decided.f1 == pytest.approx(1.0)


def test_keeping_a_real_anomaly_is_a_miss_not_a_free_pass():
    """`keep` must not be a way to score well by never committing."""
    index = _index()
    labels = _labels(index, {2: "spike", 3: "spike"})
    predictions = {"spike": {index[2], index[3]}}

    scores, _ = ev.score(predictions, labels, index, decisions={
        index[2]: "delete",
        index[3]: "keep",  # a genuine spike waved through
    })
    spike = next(s for s in scores if s.anomaly_type == "spike")
    assert spike.recall == pytest.approx(0.5)  # the kept spike is a false negative


def test_deleting_real_water_is_a_false_positive():
    """§1's costly error: only decision-aware scoring can see it."""
    index = _index()
    labels = _labels(index, {})  # nothing is an anomaly; it is all real water
    predictions = {"spike": {index[4]}}

    scores, _ = ev.score(predictions, labels, index, decisions={index[4]: "delete"})
    spike = next(s for s in scores if s.anomaly_type == "spike")
    assert (spike.n_pred, spike.precision) == (1, 0.0)


def test_an_undecided_flag_is_not_a_claim():
    """A flag the agent never ruled on is not an assertion it made."""
    index = _index()
    labels = _labels(index, {})
    narrowed = ev.apply_decisions({"spike": {index[1], index[2]}}, {index[1]: "delete"})
    assert narrowed["spike"] == {index[1]}


def test_impute_counts_as_acting_on_a_gap():
    """Imputing a gap is agreeing it is a gap, not declining to judge."""
    index = _index()
    labels = _labels(index, {5: "gap"})
    scores, _ = ev.score({"gap": {index[5]}}, labels, index, decisions={index[5]: "impute"})
    gap = next(s for s in scores if s.anomaly_type == "gap")
    assert gap.f1 == pytest.approx(1.0)


def test_load_decisions_keeps_the_stronger_verdict(tmp_path):
    """One tool's delete must not be undone by another tool's keep on the same row."""
    path = tmp_path / "x_flags.json"
    path.write_text(json.dumps([
        {"datetime": "2024-01-01T00:30:00", "flagged_by": "flagUniLOF",
         "action": "delete", "reason": "width=1 spike"},
        {"datetime": "2024-01-01T00:30:00", "flagged_by": "flagJumps",
         "action": "keep", "reason": "storm limb"},
    ]))
    assert ev.load_decisions(path) == {pd.Timestamp("2024-01-01T00:30:00"): "delete"}


def _log(tmp_path, results: list[dict], repeats: int = 1):
    """Write a JSONL log that re-sends its history, like agent.py does."""
    blocks = [
        {"type": "tool_result", "tool_use_id": f"c{i}", "content": json.dumps(r)}
        for i, r in enumerate(results)
    ]
    lines = []
    for _ in range(repeats):
        lines.append(json.dumps({
            "event": "api_call", "step": 1,
            "messages": [{"role": "user", "content": blocks}],
        }))
    path = tmp_path / "run_x.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_predictions_from_log_maps_tools_and_dedupes(tmp_path):
    """The same tool_result appears on every step; it must be counted once."""
    path = _log(tmp_path, [
        {"tool": "flag_spike_unilof", "flagged_datetimes": ["2024-01-01T00:30:00"]},
        {"tool": "flag_nan", "flagged_datetimes": ["2024-01-01T01:00:00"]},
        {"tool": "impute_rolling", "flagged_datetimes": ["2024-01-01T01:15:00"]},
    ], repeats=4)  # history re-logged four times

    predictions = ev.predictions_from_log(path)

    assert predictions["spike"] == {pd.Timestamp("2024-01-01T00:30:00")}
    assert predictions["gap"] == {pd.Timestamp("2024-01-01T01:00:00")}
    assert predictions["plateau"] == set()
    # The imputed row belongs to no type — it was an action, not a detection.
    assert pd.Timestamp("2024-01-01T01:15:00") not in predictions["gap"]


def test_predictions_from_log_survives_an_errored_result(tmp_path):
    """A crashed tool returns plain text, not JSON; that must not kill scoring."""
    path = tmp_path / "run_x.jsonl"
    path.write_text(json.dumps({
        "event": "api_call", "step": 1,
        "messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c0", "is_error": True,
             "content": "Error executing flag_plateau: argmin of an empty sequence"},
            {"type": "tool_result", "tool_use_id": "c1",
             "content": json.dumps({"tool": "flag_nan",
                                    "flagged_datetimes": ["2024-01-01T00:15:00"]})},
        ]}],
    }) + "\n")

    assert ev.predictions_from_log(path)["gap"] == {pd.Timestamp("2024-01-01T00:15:00")}


def test_test_split_scores_only_the_tail():
    """§10 holds out the last 20%, never shuffled."""
    index = _index(100)
    labels = _labels(index, {5: "spike", 95: "spike"})
    predictions = {"spike": {index[5], index[95]}}

    tail = index[int(len(index) * (1 - ev.TEST_FRACTION)):]
    scores, _ = ev.score(predictions, labels, tail)
    spike = next(s for s in scores if s.anomaly_type == "spike")

    assert len(tail) == 20
    assert spike.n_true == 1  # the row-5 spike is in the training portion
    assert spike.recall == pytest.approx(1.0)
