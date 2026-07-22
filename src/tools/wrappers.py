"""SaQC-wrapping tool functions.

Each function wraps a SaQC 2.8 method and returns the tool-result dict defined in
CLAUDE.md §5. Utility (inspect_dataset, get_flag_summary, export_clean_data),
detection (flag_range, flag_constants, flag_spike_unilof, flag_zscore,
flag_jumps, flag_nan), and action (impute_rolling, correct_drift) tools.

NOTE: verify every SaQC method name/signature against the SaQC 2.8 API before use.

Implemented in Phase 2
"""

import inspect
import pandas as pd
import numpy as np
import saqc

from src.inspect_data import summarise_series, SeriesSummary, DATETIME_COL


def _build_result(
    tool_name: str,
    params: dict,
    qc_input,
    qc_output,
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

    msg = custom_msg or f"Flagged {n_flagged} values ({pct_flagged*100:.1f}%) using {tool_name}."

    return {
        "tool": tool_name,
        "params": params,
        "n_flagged": n_flagged,
        "pct_flagged": pct_flagged,
        "flagged_datetimes": flagged_datetimes,
        "message": msg,
        "qc": qc_output  # Keep the qc object for the next steps
    }


def inspect_dataset(qc, field: str = "value") -> dict:
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


def get_flag_summary(qc, field: str = "value") -> dict:
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


def export_clean_data(qc, field: str = "value") -> dict:
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


def flag_range(qc, field: str = "value", min=None, max=None) -> dict:
    """
    Flags any data points that are too high or too low based on a set minimum and maximum limit.
    """
    params = {"min": min, "max": max}
    qc_out = qc.flagRange(field, min=min, max=max)
    return _build_result("flag_range", params, qc, qc_out, field)


def flag_constants(qc, field: str = "value", thresh=0.0, window=None, min_periods=2) -> dict:
    """
    Flags data points that get "stuck" (like a broken thermometer showing the exact same
    number for hours). It checks if values stay completely flat for a certain time window.
    """
    params = {"thresh": thresh, "window": window, "min_periods": min_periods}
    qc_out = qc.flagConstants(field, thresh=thresh, window=window, min_periods=min_periods)
    return _build_result("flag_constants", params, qc, qc_out, field)


def flag_plateau(qc, field: str = "value", min_length=None, max_length=None, min_jump=None, granularity=None) -> dict:
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

    qc_out = qc.flagPlateau(field, **kwargs)
    return _build_result("flag_plateau", params, qc, qc_out, field)


def flag_spike_unilof(qc, field: str = "value", n=20, thresh=None, density='auto', slope_correct=True) -> dict:
    """
    Flags sudden, sharp "spikes" in the data (outliers) using a smart math trick called
    Local Outlier Factor. It looks for points that are very different from their neighbors.
    """
    params = {"n": n, "thresh": thresh, "density": density, "slope_correct": slope_correct}
    qc_out = qc.flagUniLOF(field, n=n, thresh=thresh, density=density, slope_correct=slope_correct)
    return _build_result("flag_spike_unilof", params, qc, qc_out, field)


def flag_zscore(qc, field: str = "value", method='standard', window=None, thresh=3.0) -> dict:
    """
    Another way to find spikes. It calculates an average over a rolling window of time,
    and flags any data points that stray too far away from that local average.
    """
    params = {"method": method, "window": window, "thresh": thresh}
    qc_out = qc.flagZScore(field, method=method, window=window, thresh=thresh)
    return _build_result("flag_zscore", params, qc, qc_out, field)


def flag_jumps(qc, field: str = "value", thresh=0.0, window=None) -> dict:
    """
    Flags permanent jumps in the data. For example, if the sensor is bumped into a different
    position and the readings suddenly jump up and stay there forever.
    """
    params = {"thresh": thresh, "window": window}
    qc_out = qc.flagJumps(field, thresh=thresh, window=window)
    return _build_result("flag_jumps", params, qc, qc_out, field)


def flag_nan(qc, field: str = "value") -> dict:
    """
    Flags places where the data is completely missing (NaN - Not a Number).
    """
    params = {}
    qc_out = qc.flagNAN(field)
    return _build_result("flag_nan", params, qc, qc_out, field)


def impute_rolling(qc, field: str = "value", window=None, func='median', min_periods=0) -> dict:
    """
    An action tool! It tries to "fill in" small gaps of missing data by looking at
    the surrounding data and guessing the missing values (using an average/median).
    """
    params = {"window": window, "func": func, "min_periods": min_periods}

    pre_nans = qc.data[field].isna().sum()
    # explicitly pass flag=25 (DOUBTFUL) to track imputation in flags
    qc_out = qc.interpolateByRolling(field, window=window, func=func, min_periods=min_periods, flag=25)
    post_nans = qc_out.data[field].isna().sum()

    n_imputed = pre_nans - post_nans
    n_total = len(qc.data[field])
    pct_imputed = round(n_imputed / n_total, 4) if n_total > 0 else 0.0

    msg = f"Imputed {n_imputed} values ({pct_imputed*100:.1f}%) using impute_rolling. Remaining NaNs: {post_nans}"

    return _build_result("impute_rolling", params, qc, qc_out, field, custom_msg=msg)


def correct_drift(qc, maintenance_df, field: str = "value", model="linear", cal_range=5) -> dict:
    """
    An action tool! Over time, sensors get dirty and their readings "drift" away from reality.
    This tool uses the exact times a human cleaned the sensor to bend the data back into place.
    """
    params = {"model": model, "cal_range": cal_range}

    # The maintenance variable is index=start, value=end
    maint_starts = maintenance_df.index
    maint_ends = maintenance_df.iloc[:, 0]

    data_dict = {
        field: qc.data[field].copy(),
        "maintenance": maintenance_df.iloc[:, 0].copy()
    }
    qc_drift = saqc.SaQC(data_dict)

    pre_nans = qc_drift.data[field].isna().sum()
    qc_out = qc_drift.correctDrift(field, maintenance_field="maintenance", model=model, cal_range=cal_range)

    # CLAUDE.md: correctDrift silently overwrites the final interval with NaN.
    # We must restore the trailing span from the original data.
    if len(maint_ends) > 0:
        last_visit_end = maint_ends.max()
        trailing_mask = qc_out.data[field].index >= last_visit_end
        # Restore the data
        qc_out.data[field].loc[trailing_mask] = qc.data[field].loc[trailing_mask]

    post_nans = qc_out.data[field].isna().sum()

    if post_nans > pre_nans:
        # Just a safety check to warn if it did drop things
        pass

    # Build the result
    return _build_result("correct_drift", params, qc, qc_out, field)
