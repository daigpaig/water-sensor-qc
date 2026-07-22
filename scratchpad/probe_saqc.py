"""Probe the installed SaQC API.

Verifies, for every method named in CLAUDE.md §7 (plus near-miss candidates):
  1. the method exists under that exact name,
  2. its real signature (parameter names, defaults),
  3. that it runs on toy data and what it returns,
  4. any DeprecationWarning it raises.

Run:  .venv/bin/python scratchpad/probe_saqc.py
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd

import saqc

FIELD = "value"
MAINT = "maintenance"

# CLAUDE.md §7 names -> what we call the wrapper
CLAIMED = {
    "flagRange": "flag_range",
    "flagConstants": "flag_constants",
    "flagUniLOF": "flag_spike_unilof",
    "flagZScore": "flag_zscore",
    "flagJumps": "flag_jumps",
    "flagNAN": "flag_nan",
    "interpolateByRolling": "impute_rolling",
    "correctDrift": "correct_drift",
}

# Methods worth comparing against the claimed ones before we commit to a wrapper.
CANDIDATES = [
    "flagMissing",
    "flagPlateau",
    "flagDriftFromNorm",
    "flagDriftFromReference",
    "flagOffset",
    "flagChangePoints",
    "flagLOF",
    "assignUniLOF",
    "assignZScore",
]


def toy_data() -> pd.DataFrame:
    """15-min series with one of each §6 failure type at known positions."""
    idx = pd.date_range("2024-01-01", periods=2000, freq="15min")
    rng = np.random.default_rng(0)
    v = 10 + np.sin(np.arange(2000) / 50) + rng.normal(0, 0.05, 2000)

    v[500] = 60.0  # spike
    v[800:860] = v[799]  # plateau / stuck
    v[1200:] += 5.0  # level shift
    v[1500:1520] = np.nan  # gap
    v[1600:1800] += np.linspace(0, 3, 200)  # drift

    return pd.DataFrame({FIELD: v}, index=idx)


def maintenance_frame(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """correctDrift's support points: index = visit start, value = visit end."""
    starts = [idx[1800], idx[1990]]
    ends = [idx[1810], idx[1999]]
    return pd.DataFrame({MAINT: pd.Series(ends, index=pd.DatetimeIndex(starts))})


def show_signature(name: str) -> bool:
    if not hasattr(saqc.SaQC, name):
        print(f"  MISSING: SaQC has no method {name!r}")
        return False
    fn = getattr(saqc.SaQC, name)
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        print(f"  {name}: signature unavailable")
        return True
    params = [p for p in sig.parameters.values() if p.name != "self"]
    print(f"  signature: {name}(")
    for p in params:
        default = "" if p.default is inspect.Parameter.empty else f" = {p.default!r}"
        ann = "" if p.annotation is inspect.Parameter.empty else f": {p.annotation}"
        ann = ann.replace("typing.", "")
        print(f"      {p.name}{ann}{default},")
    print("  )")
    doc = (inspect.getdoc(fn) or "").strip().splitlines()
    if doc:
        print(f"  doc[0]: {doc[0]}")
    return True


def run_call(label: str, fn, *args, **kwargs) -> None:
    """Call a SaQC method on toy data and report flags / values / warnings."""
    data = toy_data()
    qc = saqc.SaQC(data)
    if label == "correctDrift":
        qc = saqc.SaQC(
            pd.concat([data, maintenance_frame(data.index)], axis=1).sort_index()
        )

    before_nan = int(data[FIELD].isna().sum())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = fn(qc)
        except Exception as exc:  # noqa: BLE001 - probing, we want the message
            print(f"  CALL FAILED: {type(exc).__name__}: {exc}")
            return

    print(f"  returns: {type(out).__module__}.{type(out).__qualname__}")
    flags = out.flags[FIELD]
    vals = out.data[FIELD]
    n_flagged = int((flags > -np.inf).sum())
    print(
        f"  flags: dtype={flags.dtype} n_flagged={n_flagged} "
        f"distinct={sorted(set(np.unique(flags)) - {-np.inf})}"
    )
    print(
        f"  data: n_nan {before_nan} -> {int(vals.isna().sum())} "
        f"changed_values={int((~np.isclose(vals.to_numpy(dtype=float), data[FIELD].to_numpy(dtype=float), equal_nan=True)).sum())}"
    )
    for w in caught:
        if issubclass(w.category, (DeprecationWarning, FutureWarning)):
            print(f"  WARNING [{w.category.__name__}]: {w.message}")


CALLS = {
    "flagRange": lambda qc: qc.flagRange(FIELD, min=0, max=50),
    "flagConstants": lambda qc: qc.flagConstants(FIELD, thresh=0.01, window="6h"),
    "flagUniLOF": lambda qc: qc.flagUniLOF(FIELD, n=20, thresh=1.5),
    "flagZScore": lambda qc: qc.flagZScore(FIELD, window="12h", thresh=3.0),
    "flagJumps": lambda qc: qc.flagJumps(FIELD, thresh=2.0, window="3h"),
    "flagNAN": lambda qc: qc.flagNAN(FIELD),
    "flagMissing": lambda qc: qc.flagMissing(FIELD),
    "flagPlateau": lambda qc: qc.flagPlateau(FIELD, min_length="1h", max_length="24h"),
    "interpolateByRolling": lambda qc: qc.interpolateByRolling(FIELD, window="3h"),
    "correctDrift": lambda qc: qc.correctDrift(
        FIELD, maintenance_field=MAINT, model="linear"
    ),
    "flagOffset": lambda qc: qc.flagOffset(FIELD, tolerance=1.0, thresh=5.0, window="3h"),
    "flagChangePoints": lambda qc: qc.flagChangePoints(
        FIELD,
        stat_func=lambda x, y: np.abs(np.mean(x) - np.mean(y)),
        thresh_func=lambda x, y: 2.0,
        window="12h",
        min_periods=5,
    ),
}


def main() -> None:
    print(f"saqc {saqc.__version__}   pandas {pd.__version__}   numpy {np.__version__}")

    print("\n" + "=" * 72)
    print("CLAIMED IN CLAUDE.md §7")
    print("=" * 72)
    for name, wrapper in CLAIMED.items():
        print(f"\n--- {name}  (wrapper: {wrapper}) ---")
        if show_signature(name) and name in CALLS:
            run_call(name, CALLS[name])

    print("\n" + "=" * 72)
    print("CANDIDATES NOT IN §7")
    print("=" * 72)
    for name in CANDIDATES:
        print(f"\n--- {name} ---")
        if show_signature(name) and name in CALLS:
            run_call(name, CALLS[name])


if __name__ == "__main__":
    main()
