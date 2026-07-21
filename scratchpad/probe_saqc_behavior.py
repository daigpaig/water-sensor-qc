"""Probe SaQC 2.8 behaviours that the signature alone does not reveal.

Follow-up to probe_saqc.py. Answers:
  A. correctDrift — where does it actually correct, and what does it need?
  B. Is there a *univariate* drift detector at all?
  C. flagPlateau vs flagConstants — which detects a stuck sensor?
  D. interpolateByRolling — which gaps does it refuse to fill?
  E. flag constants + how to read "which rows did this tool flag".
  F. Does SaQC require a regular / sorted DatetimeIndex?

Run:  .venv/bin/python scratchpad/probe_saqc_behavior.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import saqc

warnings.filterwarnings("ignore", category=DeprecationWarning, module="scipy")

F = "value"
M = "maintenance"


def hdr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def base_series(n: int = 2000) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(0)
    return pd.Series(10 + rng.normal(0, 0.05, n), index=idx, name=F)


# --------------------------------------------------------------------------- A
VISIT_ROWS = [400, 1000, 1600, 2200, 2800]


def drifting_series(n: int = 4000) -> pd.Series:
    """Fouling that accrues inside each inter-visit interval and resets at the visit."""
    s = base_series(n)
    for a, b in zip(VISIT_ROWS[:-1], VISIT_ROWS[1:]):
        s.iloc[a:b] += np.linspace(0, 3, b - a)
    return s


def maint_var(idx: pd.DatetimeIndex, n_visits: int) -> pd.Series:
    """index = visit start, value = visit end. Keeps its OWN short index."""
    starts = [idx[r] for r in VISIT_ROWS[:n_visits]]
    ends = [idx[r + 10] for r in VISIT_ROWS[:n_visits]]
    return pd.Series(pd.DatetimeIndex(ends), index=pd.DatetimeIndex(starts), name=M)


def probe_correct_drift() -> None:
    hdr("A. correctDrift — support points, and what it does to the last interval")

    s = drifting_series()
    print(f"  input NaNs: {int(s.isna().sum())}")
    print("  visits at rows", VISIT_ROWS, "\n")

    for n_visits in [2, 3, 4, 5]:
        # Pass a DICT so `value` and `maintenance` keep independent indexes.
        out = saqc.SaQC({F: s, M: maint_var(s.index, n_visits)}).correctDrift(
            F, maintenance_field=M, model="linear"
        )
        v = out.data[F]
        diff = (v - s).abs()
        changed = np.flatnonzero((diff > 1e-9).to_numpy())
        nan_rows = np.flatnonzero(v.isna().to_numpy())
        cspan = f"rows {changed.min()}..{changed.max()}" if changed.size else "nowhere"
        nspan = f"rows {nan_rows.min()}..{nan_rows.max()}" if nan_rows.size else "-"
        flags_on_nan = (
            set(np.unique(out.flags[F].to_numpy()[nan_rows])) if nan_rows.size else set()
        )
        print(
            f"  {n_visits} visits -> {n_visits - 1} interval(s):\n"
            f"      corrected {changed.size:>5} vals ({cspan})\n"
            f"      NaN'd out {nan_rows.size:>5} vals ({nspan})  flags there: {flags_on_nan or '-'}"
        )

    # The failure mode: maintenance aligned onto the data index instead of standalone.
    try:
        joined = pd.Series(np.nan, index=s.index, name=M)
        joined.loc[s.index[400]] = s.index[410]
        print("\n  maintenance NaN-padded onto the data index: constructed OK")
    except Exception as exc:  # noqa: BLE001
        print(f"\n  maintenance NaN-padded onto the data index: {type(exc).__name__}: {exc}")

    try:
        saqc.SaQC(pd.DataFrame({F: s})).correctDrift(
            F, maintenance_field=M, model="linear"
        )
        print("  maintenance_field absent from the object: accepted (!)")
    except Exception as exc:  # noqa: BLE001
        print(f"  maintenance_field absent: {type(exc).__name__}: {str(exc).splitlines()[0]}")


# --------------------------------------------------------------------------- B
def probe_drift_detection() -> None:
    hdr("B. Is there a UNIVARIATE drift detector?")

    idx = base_series().index
    rng = np.random.default_rng(1)

    def flat():
        return pd.Series(10 + rng.normal(0, 0.05, len(idx)), index=idx)

    drifter = flat()
    drifter.iloc[1000:2000] += np.linspace(0, 5, 1000)
    cols = {"drifter": drifter, "a": flat(), "b": flat(), "c": flat()}

    for fields in [["drifter"], ["drifter", "a"], ["drifter", "a", "b", "c"]]:
        out = saqc.SaQC(pd.DataFrame(cols)).flagDriftFromNorm(
            fields, window="3d", spread=2.0
        )
        n = int((out.flags["drifter"] > -np.inf).sum())
        print(f"  flagDriftFromNorm(fields={fields}): drifter n_flagged={n}")

    out = saqc.SaQC(pd.DataFrame({F: drifter})).flagDriftFromReference(
        F, reference=F, freq="3d", thresh=1.0
    )
    print(f"  flagDriftFromReference(self as reference): n={int((out.flags[F] > -np.inf).sum())}")

    out = saqc.SaQC(pd.DataFrame({F: drifter})).flagJumps(F, thresh=0.5, window="12h")
    print(f"  flagJumps on a slow drift: n={int((out.flags[F] > -np.inf).sum())}")


# --------------------------------------------------------------------------- C
def probe_plateau() -> None:
    hdr("C. flagConstants vs flagPlateau on a stuck sensor")

    for label, level in {
        "stuck AT the local level (no offset)": 10.0,
        "stuck at +2 ABOVE the local level": 12.0,
    }.items():
        s = base_series(3000)
        s.iloc[800:900] = level  # 100 rows = 25h
        qc = saqc.SaQC(pd.DataFrame({F: s}))
        print(f"\n  {label}  (plateau = rows 800..899, 25h)")

        out = qc.flagConstants(F, thresh=0.01, window="6h")
        w = np.flatnonzero((out.flags[F] > -np.inf).to_numpy())
        span = f"rows {w.min()}..{w.max()}" if w.size else "-"
        print(f"      flagConstants(thresh=0.01, window='6h'): n={w.size} {span}")

        for min_length in ["1h", "3h", "6h", "12h"]:
            out = qc.flagPlateau(F, min_length=min_length)
            w = np.flatnonzero((out.flags[F] > -np.inf).to_numpy())
            span = f"rows {w.min()}..{w.max()}" if w.size else "-"
            print(f"      flagPlateau(min_length={min_length!r}): n={w.size} {span}")

    # flagConstants with a loose threshold swallows the whole series.
    s = base_series(3000)
    out = saqc.SaQC(pd.DataFrame({F: s})).flagConstants(F, thresh=0.5, window="6h")
    print(
        f"\n  flagConstants(thresh=0.5) on pure noise (sd=0.05): "
        f"n={int((out.flags[F] > -np.inf).sum())} of {len(s)}  <- thresh must be << signal sd"
    )


# --------------------------------------------------------------------------- D
def probe_interpolate() -> None:
    hdr("D. interpolateByRolling — which gaps get filled?")

    s = base_series()
    # gaps of 1, 4, 12, 40 samples (15min grid => 0.25h, 1h, 3h, 10h)
    for start, length in [(200, 1), (400, 4), (600, 12), (900, 40)]:
        s.iloc[start : start + length] = np.nan
    total = int(s.isna().sum())

    for window in ["1h", "3h", "12h"]:
        for min_periods in [0, 2]:
            out = saqc.SaQC(pd.DataFrame({F: s})).interpolateByRolling(
                F, window=window, min_periods=min_periods
            )
            v = out.data[F]
            filled_by_gap = {
                f"{length}pt": int(s.iloc[start : start + length].isna().sum()
                                   - v.iloc[start : start + length].isna().sum())
                for start, length in [(200, 1), (400, 4), (600, 12), (900, 40)]
            }
            print(
                f"  window={window:>4} min_periods={min_periods}: "
                f"filled {total - int(v.isna().sum())}/{total}  {filled_by_gap}"
            )

    # Does it flag what it filled?
    out = saqc.SaQC(pd.DataFrame({F: s})).interpolateByRolling(F, window="3h")
    print(f"  default flag= -inf -> n_flagged={int((out.flags[F] > -np.inf).sum())}")
    out = saqc.SaQC(pd.DataFrame({F: s})).interpolateByRolling(
        F, window="3h", flag=saqc.BAD
    )
    print(f"  flag=saqc.BAD    -> n_flagged={int((out.flags[F] > -np.inf).sum())}")


# --------------------------------------------------------------------------- E
def probe_flags() -> None:
    hdr("E. Flag constants and reading back 'which rows did THIS tool flag'")

    print(f"  saqc.UNFLAGGED={saqc.UNFLAGGED}  GOOD={saqc.GOOD}  "
          f"DOUBTFUL={saqc.DOUBTFUL}  BAD={saqc.BAD}")

    s = base_series()
    s.iloc[500] = 60.0
    s.iloc[700] = 55.0

    qc = saqc.SaQC(pd.DataFrame({F: s}))
    after_a = qc.flagRange(F, min=0, max=50)
    after_b = after_a.flagUniLOF(F, n=20, thresh=1.5)

    fa = after_a.flags[F]
    fb = after_b.flags[F]
    print(f"  after flagRange:  n={int((fa > -np.inf).sum())}")
    print(f"  after +flagUniLOF: n={int((fb > -np.inf).sum())}")
    newly = int(((fb > -np.inf) & ~(fa > -np.inf)).sum())
    print(f"  => attribution requires diffing successive flag frames; newly={newly}")

    # Is the flags object a DataFrame? Can we get a history?
    print(f"  type(qc.flags)={type(after_b.flags).__name__}")
    hist = after_b._flags.history[F] if hasattr(after_b, "_flags") else None
    if hist is not None:
        print(f"  history columns (one per applied test): {hist.hist.shape[1]}")
        print(f"  history meta funcs: {[m.get('func') for m in hist.meta]}")


# --------------------------------------------------------------------------- F
def probe_index_requirements() -> None:
    hdr("F. Index requirements")

    s = base_series(500)

    # unsorted
    shuffled = s.sample(frac=1, random_state=0)
    try:
        saqc.SaQC(pd.DataFrame({F: shuffled})).flagConstants(F, thresh=0.01, window="6h")
        print("  unsorted DatetimeIndex: accepted")
    except Exception as exc:  # noqa: BLE001
        print(f"  unsorted DatetimeIndex: {type(exc).__name__}: {exc}")

    # irregular (drop random rows) with an offset-string window
    irregular = s.drop(s.index[[10, 11, 12, 100, 250]])
    for label, call in {
        "flagConstants(window='6h')": lambda d: saqc.SaQC(d).flagConstants(
            F, thresh=0.01, window="6h"
        ),
        "flagConstants(window=10 rows)": lambda d: saqc.SaQC(d).flagConstants(
            F, thresh=0.01, window=10
        ),
        "flagJumps(window='6h')": lambda d: saqc.SaQC(d).flagJumps(
            F, thresh=1.0, window="6h"
        ),
        "flagUniLOF()": lambda d: saqc.SaQC(d).flagUniLOF(F),
    }.items():
        try:
            call(pd.DataFrame({F: irregular}))
            print(f"  irregular index, {label}: accepted")
        except Exception as exc:  # noqa: BLE001
            print(f"  irregular index, {label}: {type(exc).__name__}: {exc}")

    # non-datetime index
    try:
        saqc.SaQC(pd.DataFrame({F: s.to_numpy()})).flagRange(F, min=0, max=50)
        print("  RangeIndex (no datetime): accepted")
    except Exception as exc:  # noqa: BLE001
        print(f"  RangeIndex (no datetime): {type(exc).__name__}: {exc}")

    # duplicate timestamps
    dup = pd.concat([s, s.iloc[:5]]).sort_index()
    try:
        saqc.SaQC(pd.DataFrame({F: dup})).flagRange(F, min=0, max=50)
        print("  duplicate timestamps: accepted")
    except Exception as exc:  # noqa: BLE001
        print(f"  duplicate timestamps: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    print(f"saqc {saqc.__version__}  pandas {pd.__version__}  numpy {np.__version__}")
    probe_correct_drift()
    probe_drift_detection()
    probe_plateau()
    probe_interpolate()
    probe_flags()
    probe_index_requirements()
