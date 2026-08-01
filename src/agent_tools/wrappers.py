"""SaQC-wrapping tool functions.

Each function wraps a SaQC 2.8 method and returns the tool-result dict defined in
CLAUDE.md §5. Utility (inspect_dataset, get_flag_summary, export_clean_data),
detection (flag_range, flag_constants, flag_plateau, flag_spike_unilof, flag_zscore,
flag_jumps, flag_nan), action (impute_rolling), and context (describe_point,
describe_points -- thin pass-throughs to context.py, see §7.3) tools.

NOTE: verify every SaQC method name/signature against the SaQC 2.8 API before use.

Implemented in Phase 2
"""

import pandas as pd
import saqc

from src.agent_tools import context
from src.inspect_data import summarise_series, DATETIME_COL


def _find_nan_runs(series: pd.Series) -> list[dict]:
    """Return one dict per contiguous NaN run in *series*.

    Each dict contains:
      start        -- first NaN timestamp
      end          -- last NaN timestamp
      duration_td  -- pd.Timedelta (end - start; zero for single-row gaps)
      n_rows       -- number of NaN rows in the run
    """
    runs: list[dict] = []
    if not series.isna().any():
        return runs

    is_nan = series.isna()
    # Label each contiguous block of the same boolean
    group_ids = (is_nan != is_nan.shift()).cumsum()
    for _, grp in is_nan.groupby(group_ids):
        if not grp.iloc[0]:          # skip non-NaN blocks
            continue
        start = grp.index[0]
        end   = grp.index[-1]
        runs.append({
            "start":       start,
            "end":         end,
            "duration_td": end - start,
            "n_rows":      len(grp),
        })
    return runs


def _build_result(
    tool_name: str,
    params: dict,
    qc_input: saqc.SaQC,
    qc_output: saqc.SaQC,
    field: str,
    custom_msg: str = None
) -> dict:
    """Helper to construct the standardized JSON-serializable tool result."""
    # Note: qc_output.flags is a DictOfSeries, we need to find what was newly flagged
    # by looking at the history.
    history = qc_output._flags.history[field]

    if len(history.hist.columns) > 0:
        # The last column in the history corresponds to the most recently applied test
        last_test = history.hist.columns[-1]

        # Flags in history are floats (UNFLAGGED=-inf, GOOD=0, DOUBTFUL=25, BAD=255)
        # We consider anything > 0 as flagged for detection/actions.
        flagged_mask = history.hist[last_test] > 0

        flagged_datetimes = history.hist.index[flagged_mask].strftime('%Y-%m-%dT%H:%M:%S').tolist()
        n_flagged = len(flagged_datetimes)
        n_total = len(history.hist)
        pct_flagged = round(n_flagged / n_total, 4) if n_total > 0 else 0.0
    else:
        n_flagged = 0
        pct_flagged = 0.0
        flagged_datetimes = []

    # Rows flagged on this field by ANY call so far, not just this one. SaQC never
    # re-flags a row a previous test already flagged, so `n_flagged` above counts only
    # what THIS call added; without the running total, an agent that re-runs a detector
    # with looser parameters cannot tell how much is flagged altogether (§7.1).
    n_flagged_total = int((qc_output.flags[field] > 0).sum())

    msg = custom_msg or f"Flagged {n_flagged} values ({pct_flagged*100:.1f}%) using {tool_name}."
    if custom_msg is None and n_flagged_total != n_flagged:
        msg += (
            f" These are rows not already flagged by an earlier call; "
            f"{n_flagged_total} row(s) are now flagged on '{field}' in total."
        )

    return {
        "tool": tool_name,
        "params": params,
        "n_flagged": n_flagged,
        "pct_flagged": pct_flagged,
        "n_flagged_total": n_flagged_total,
        "flagged_datetimes": flagged_datetimes,
        "message": msg,
        "qc": qc_output  # Keep the qc object for the next steps
    }


def inspect_dataset(qc: saqc.SaQC, field: str = "value") -> dict:
    """
    Summarizes the dataset. It looks at the data and counts the rows, missing values,
    and checks the start/end times.
    """
    df = qc.data.to_pandas()
    df = df.reset_index()
    df.rename(columns={"index": DATETIME_COL}, inplace=True)
    summary = summarise_series(df, value_col=field)
    return {
        "tool": "inspect_dataset",
        "params": {"field": field},
        "n_rows": summary.n_rows,
        "message": "Dataset inspected.",
        "qc": qc,
        "summary": summary.to_dict()
    }


def get_flag_summary(qc: saqc.SaQC, field: str = "value") -> dict:
    """
    Looks at the history of the data and counts how many bad data points were found
    by each tool that the robot used so far.
    """
    history = qc._flags.history[field]
    summary = {}
    for col in history.hist.columns:
        test_name = history.meta[col].get("func", col)
        flagged = (history.hist[col] > 0).sum()
        summary[test_name] = summary.get(test_name, 0) + int(flagged)

    return {
        "tool": "get_flag_summary",
        "params": {"field": field},
        "message": f"Flag summary retrieved: {summary}",
        "qc": qc,
        "summary": summary
    }


def export_clean_data(qc: saqc.SaQC, field: str = "value") -> dict:
    """
    Takes the final, cleaned data and gives it back as a simple spreadsheet-like format,
    marking which points were flagged by the tools.
    """
    df = qc.data.to_pandas()
    flags = qc.flags[field]

    # Contract: 'flag' column naming the action, if any
    # Since saqc just returns float flags, we can map > 0 to 'flagged'
    # and we can deduce actions from history if needed, but for now we'll
    # just create a generic flag column if it's flagged.
    history = qc._flags.history[field]
    df['flag'] = None

    for col in history.hist.columns:
        test_name = history.meta[col].get("func", col)
        mask = history.hist[col] > 0
        df.loc[mask, 'flag'] = test_name

    return {
        "tool": "export_clean_data",
        "params": {"field": field},
        "message": "Data exported.",
        "qc": qc,
        "df": df
    }


def flag_range(qc: saqc.SaQC, field: str = "value", min=None, max=None) -> dict:
    """
    Flags any data points that are too high or too low based on a set minimum and maximum limit.
    """
    params = {"min": min, "max": max}
    qc_out = qc.flagRange(field, min=min, max=max)
    return _build_result("flag_range", params, qc, qc_out, field)


def flag_constants(qc: saqc.SaQC, field: str = "value", thresh=0.0, window=None, min_periods=2) -> dict:
    """
    Flags data points that get "stuck" (like a broken thermometer showing the exact same
    number for hours). It checks if values stay completely flat for a certain time window.
    """
    params = {"thresh": thresh, "window": window, "min_periods": min_periods}
    qc_out = qc.flagConstants(field, thresh=thresh, window=window, min_periods=min_periods)
    return _build_result("flag_constants", params, qc, qc_out, field)


def flag_plateau(qc: saqc.SaQC, field: str = "value", min_length="1h", max_length=None, min_jump=None, granularity=None) -> dict:
    """
    Flags a "plateau" - when the data suddenly jumps up, stays flat for a while, and then
    drops back down. This happens when debris gets stuck on the sensor temporarily.
    """
    params = {"min_length": min_length, "max_length": max_length, "min_jump": min_jump, "granularity": granularity}
    # min_jump is not a valid argument for flagPlateau in saqc 2.8 maybe? Wait.
    # CLAUDE.md §7: `flagPlateau` | `min_length`, `max_length`, `min_jump`, `granularity`
    # We will pass kwargs dynamically to avoid None defaults if they aren't accepted.
    kwargs = {}
    if min_length is not None: kwargs["min_length"] = min_length
    if max_length is not None: kwargs["max_length"] = max_length
    if min_jump is not None: kwargs["min_jump"] = min_jump
    if granularity is not None: kwargs["granularity"] = granularity

    # flagPlateau is the one §7 method that raises on perfectly ordinary input, and it
    # is data-dependent rather than length-monotonic (§7.1): 'attempt to get argmin of
    # an empty sequence' from _getAnomalyCenter, or a numpy window-shape error when the
    # window outruns the array. One crashy detector must not sink the whole run, so the
    # failure comes back as a normal result saying it found nothing and why.
    try:
        qc_out = qc.flagPlateau(field, **kwargs)
    except ValueError as exc:
        return {
            "tool": "flag_plateau",
            "params": params,
            "n_flagged": 0,
            "pct_flagged": 0.0,
            "n_flagged_total": int((qc.flags[field] > 0).sum()),
            "flagged_datetimes": [],
            "message": (
                f"flag_plateau could not run on this series and flagged nothing: {exc}. "
                "This is a known SaQC 2.8 defect, not a statement about the data — it says "
                "nothing about whether plateaus are present. Do not retry it with the same "
                "parameters; rely on flag_constants for stuck-sensor detection instead."
            ),
            "failed": True,
            "qc": qc,  # unchanged: nothing was flagged
        }
    return _build_result("flag_plateau", params, qc, qc_out, field)


def flag_spike_unilof(qc: saqc.SaQC, field: str = "value", n=20, thresh=None, density='auto', slope_correct=True) -> dict:
    """
    Flags sudden, sharp "spikes" in the data (outliers) using a smart math trick called
    Local Outlier Factor. It looks for points that are very different from their neighbors.
    """
    params = {"n": n, "thresh": thresh, "density": density, "slope_correct": slope_correct}
    qc_out = qc.flagUniLOF(field, n=n, thresh=thresh, density=density, slope_correct=slope_correct)
    return _build_result("flag_spike_unilof", params, qc, qc_out, field)


def flag_zscore(qc: saqc.SaQC, field: str = "value", method='standard', window=None, thresh=3.0) -> dict:
    """
    Another way to find spikes. It calculates an average over a rolling window of time,
    and flags any data points that stray too far away from that local average.
    """
    params = {"method": method, "window": window, "thresh": thresh}
    qc_out = qc.flagZScore(field, method=method, window=window, thresh=thresh)
    return _build_result("flag_zscore", params, qc, qc_out, field)


def flag_jumps(qc: saqc.SaQC, field: str = "value", thresh=0.0, window=None) -> dict:
    """
    Flags permanent jumps in the data. For example, if the sensor is bumped into a different
    position and the readings suddenly jump up and stay there forever.
    """
    params = {"thresh": thresh, "window": window}
    qc_out = qc.flagJumps(field, thresh=thresh, window=window)
    return _build_result("flag_jumps", params, qc, qc_out, field)


def flag_nan(qc: saqc.SaQC, field: str = "value") -> dict:
    """
    Flags places where the data is completely missing (NaN - Not a Number).
    """
    params = {}
    qc_out = qc.flagNAN(field)
    return _build_result("flag_nan", params, qc, qc_out, field)


def impute_rolling(
    qc: saqc.SaQC,
    field: str = "value",
    window=None,
    func: str = "median",
    min_periods: int = 0,
    max_gap: str | None = None,
) -> dict:
    """Fill NaN gaps using a rolling window median (or other aggregation).

    Only fills gaps whose duration is <= max_gap. Longer gaps are left as NaN
    and reported in the result so the agent knows they were skipped.

    max_gap: pandas offset string, e.g. '3h'. If None, all gaps are imputed
    up to what the window can reach. Always set max_gap to the longest gap you
    are willing to accept; do not impute multi-day outages.
    """
    params = {"window": window, "func": func, "min_periods": min_periods, "max_gap": max_gap}

    # --- 1. Analyse gaps before touching the data ---
    pre_series = qc.data.to_pandas()[field]
    nan_runs   = _find_nan_runs(pre_series)

    max_gap_td = pd.Timedelta(max_gap) if max_gap is not None else None

    fillable_runs  = []
    too_large_runs = []
    for run in nan_runs:
        if max_gap_td is not None and run["duration_td"] > max_gap_td:
            too_large_runs.append(run)
        else:
            fillable_runs.append(run)

    # --- 2. Call SaQC's interpolateByRolling ---
    # flag=25 (DOUBTFUL) so imputed rows appear in the flag history (§7.1).
    pre_nans = int(pre_series.isna().sum())
    qc_out   = qc.interpolateByRolling(
        field, window=window, func=func, min_periods=min_periods, flag=25
    )
    post_series = qc_out.data.to_pandas()[field]
    post_nans   = int(post_series.isna().sum())
    n_imputed   = pre_nans - post_nans
    n_total     = len(pre_series)

    # --- 3. Warn if any too-large gap was partially filled ---
    # SaQC's rolling window naturally can't bridge a gap wider than `window`,
    # but it will fill the edges.  Report every such case explicitly.
    partial_fill_warnings: list[str] = []
    for run in too_large_runs:
        run_slice = post_series.loc[run["start"]:run["end"]]
        n_partial  = int(run_slice.notna().sum())
        if n_partial > 0:
            partial_fill_warnings.append(
                f"Gap {run['start'].isoformat()}–{run['end'].isoformat()} "
                f"({run['duration_td']}) exceeds max_gap='{max_gap}': "
                f"{n_partial} edge row(s) were partially filled."
            )

    # --- 4. Build gap summary (JSON-serialisable, for agent context) ---
    gaps_summary = [
        {
            "start":           run["start"].isoformat(),
            "end":             run["end"].isoformat(),
            "duration":        str(run["duration_td"]),
            "n_rows":          run["n_rows"],
            "skipped_too_large": (max_gap_td is not None and run["duration_td"] > max_gap_td),
        }
        for run in nan_runs
    ]

    # --- 5. Compose message ---
    pct_imputed = round(n_imputed / n_total, 4) if n_total > 0 else 0.0
    msg_parts   = [
        f"Imputed {n_imputed} values ({pct_imputed * 100:.1f}%) across "
        f"{len(fillable_runs)} of {len(nan_runs)} gap(s). "
        f"{post_nans} NaN(s) remain."
    ]
    if too_large_runs:
        msg_parts.append(
            f"{len(too_large_runs)} gap(s) exceeded max_gap='{max_gap}' and were not imputed."
        )
    if partial_fill_warnings:
        msg_parts.append("PARTIAL FILL WARNING: " + " | ".join(partial_fill_warnings))

    msg = " ".join(msg_parts)

    result = _build_result("impute_rolling", params, qc, qc_out, field, custom_msg=msg)
    result["n_imputed"]            = n_imputed
    result["n_gaps_total"]         = len(nan_runs)
    result["n_gaps_filled"]        = len(fillable_runs)
    result["n_gaps_skipped_large"] = len(too_large_runs)
    result["gaps_summary"]         = gaps_summary
    return result


# ---------------------------------------------------------------------------
# Context tools (CLAUDE.md §7.3)
#
# Thin pass-throughs to src/agent_tools/context.py. The measurement lives there;
# these exist so every tool the agent can call is reachable from one module with
# one calling convention -- `qc` first, like every wrapper above. They OBSERVE:
# nothing here flags or mutates, so the results carry no `qc` key and the caller's
# SaQC object is unchanged (`inspect_dataset` sets that precedent in §5).
#
# Only the two aggregators are wrapped. The eight primitives behind them stay
# library functions: nine near-identical tools would eat the 25-call cap, and
# describe_point already returns all of them at once.
# ---------------------------------------------------------------------------

def describe_point(
    qc: saqc.SaQC,
    at: str,
    field: str = "value",
    n_before: int = 8,
    n_after: int = 8,
    window: str = "6h",
    shift_window: str = "24h",
) -> dict:
    """Measure the shape of the series around ONE timestamp.

    Answers the question a detector cannot: is this excursion real water or a
    sensor artifact? Returns the eight nested measurement blocks from
    :mod:`src.agent_tools.context` (slope, excursion, recovery, level_shift,
    flatness, neighbourhood, gap, history) plus a ``reads_like`` hint.

    Raises ValueError if *at* is not a timestamp in the series -- deliberately,
    rather than rounding silently to a neighbour (§13).
    """
    return context.describe_point(
        qc,
        at,
        field=field,
        n_before=n_before,
        n_after=n_after,
        window=window,
        shift_window=shift_window,
    )


def describe_points(
    qc: saqc.SaQC,
    ats: list[str],
    field: str = "value",
    max_points: int = 20,
    window: str = "6h",
) -> dict:
    """Compact shape measurements for a LIST of timestamps -- e.g. a detector's output.

    One row per timestamp (value, reads_like, robust_z, width_samples,
    peak_sharpness, fall_rise_ratio, samples_to_recover, recovered) plus a tally
    by label, so a whole detector result can be judged in a single call.

    Timestamps beyond *max_points* are reported in ``n_truncated`` rather than
    dropped silently; unresolvable ones land in ``errors`` instead of raising, so
    one bad timestamp cannot sink the batch.
    """
    return context.describe_points(
        qc,
        ats,
        field=field,
        max_points=max_points,
        window=window,
    )
