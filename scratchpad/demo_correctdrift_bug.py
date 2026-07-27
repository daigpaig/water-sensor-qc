"""Demo: SaQC 2.8.0 `correctDrift` silently destroys the last inter-visit interval.

Toy data with an exactly-known truth, so nothing here is a judgement call:

  clean signal      = 10.0 everywhere (no noise)
  fouling           = a linear ramp 0 -> +3.0 inside each inter-visit interval,
                      reset to 0 at each maintenance visit
  perfect correction = returns every value to exactly 10.0

Run:  .venv/bin/python scratchpad/demo_correctdrift_bug.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import saqc

warnings.filterwarnings("ignore", category=DeprecationWarning)

F, M = "value", "maintenance"
CLEAN_LEVEL = 10.0
DRIFT_MAGNITUDE = 3.0

# Hourly data, 5 maintenance visits every 240 h (10 days), each lasting 3 h.
N_ROWS = 1440
VISIT_ROWS = [240, 480, 720, 960, 1200]
VISIT_LEN = 3


def build() -> tuple[pd.Series, pd.Series]:
    """Return (drifted series, clean truth). Both free of NaN."""
    idx = pd.date_range("2024-01-01", periods=N_ROWS, freq="h")
    clean = pd.Series(CLEAN_LEVEL, index=idx, dtype=float)

    drifted = clean.copy()
    for a, b in zip(VISIT_ROWS[:-1], VISIT_ROWS[1:]):
        drifted.iloc[a:b] += np.linspace(0, DRIFT_MAGNITUDE, b - a)
    return drifted, clean


def maintenance(idx: pd.DatetimeIndex, n_visits: int) -> pd.Series:
    """index = visit start, value = visit end (SaQC's support-point format)."""
    starts = [idx[r] for r in VISIT_ROWS[:n_visits]]
    ends = [idx[r + VISIT_LEN] for r in VISIT_ROWS[:n_visits]]
    return pd.Series(pd.DatetimeIndex(ends), index=pd.DatetimeIndex(starts), name=M)


def run(drifted: pd.Series, n_visits: int) -> tuple[pd.Series, pd.Series]:
    """Apply correctDrift; return (corrected values, flags)."""
    maint = maintenance(drifted.index, n_visits)
    # dict -> the two variables keep independent indexes
    out = saqc.SaQC({F: drifted, M: maint}).correctDrift(
        F, maintenance_field=M, model="linear"
    )
    return out.data[F], out.flags[F]


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------- part 1
def part1_the_bug() -> None:
    drifted, clean = build()
    corrected, flags = run(drifted, n_visits=5)
    rule(f"PART 1 — Input has ZERO NaNs. Output has {int(corrected.isna().sum())}.")

    print(f"  input  : {len(drifted):>5} rows, {int(drifted.isna().sum()):>4} NaN, "
          f"range {drifted.min():.2f} .. {drifted.max():.2f}")
    print(f"  output : {len(corrected):>5} rows, {int(corrected.isna().sum()):>4} NaN  "
          f"<-- correctDrift INVENTED these")

    lost = np.flatnonzero(corrected.isna().to_numpy() & ~drifted.isna().to_numpy())
    print(f"\n  {lost.size} values destroyed, rows {lost.min()}..{lost.max()}")
    print(f"  their flags: {set(np.unique(flags.to_numpy()[lost]))} "
          f"(saqc.UNFLAGGED = {saqc.UNFLAGGED})")
    print("  -> nothing in the flag log would ever explain the loss.")

    print("\n  Per-interval outcome (visits at rows "
          f"{VISIT_ROWS}, each {VISIT_LEN}h long):\n")
    print(f"    {'interval':<26} {'rows':>12} {'corrected':>10} {'destroyed':>10}  verdict")
    print(f"    {'-' * 26} {'-' * 12} {'-' * 10} {'-' * 10}  {'-' * 22}")

    bounds = [(VISIT_ROWS[k] + VISIT_LEN, VISIT_ROWS[k + 1])
              for k in range(len(VISIT_ROWS) - 1)]
    for k, (a, b) in enumerate(bounds):
        seg_out, seg_in = corrected.iloc[a:b], drifted.iloc[a:b]
        n_nan = int(seg_out.isna().sum())
        n_fixed = int(((seg_out - seg_in).abs() > 1e-9).sum())
        verdict = "WIPED TO NaN" if n_nan else "corrected"
        print(f"    {f'{k}: end v{k} -> start v{k+1}':<26} {f'{a}..{b - 1}':>12} "
              f"{n_fixed:>10} {n_nan:>10}  {verdict}")

    print("\n  Accuracy where it DID work (truth is exactly 10.0):")
    for k, (a, b) in enumerate(bounds):
        seg = corrected.iloc[a:b]
        if seg.isna().all():
            continue
        before = float(np.sqrt(((drifted.iloc[a:b] - clean.iloc[a:b]) ** 2).mean()))
        after = float(np.sqrt(((seg - clean.iloc[a:b]) ** 2).mean()))
        print(f"    interval {k}: RMSE {before:.3f} -> {after:.3f}   "
              f"({before / after:.1f}x better — the maths is sound, only the "
              f"last interval is broken)")


# ---------------------------------------------------------------- part 2
def part2_scaling() -> None:
    rule("PART 2 — It is not an edge case: N visits => only N-2 intervals survive.")

    drifted, _ = build()
    print(f"    {'visits':>6} {'intervals':>10} {'corrected':>10} {'destroyed':>10}  outcome")
    print(f"    {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 10}  {'-' * 30}")

    for n in range(2, 6):
        corrected, _ = run(drifted, n_visits=n)
        n_nan = int(corrected.isna().sum())
        n_fixed = int(((corrected - drifted).abs() > 1e-9).sum())
        note = "nothing corrected, data lost" if n_fixed == 0 else f"{n - 2} of {n - 1} intervals kept"
        print(f"    {n:>6} {n - 1:>10} {n_fixed:>10} {n_nan:>10}  {note}")

    print("\n  With 2 visits correctDrift corrects NOTHING and still eats the data.")


# ---------------------------------------------------------------- part 3
def part3_root_cause() -> None:
    rule("PART 3 — Root cause, reproduced in 4 lines of pandas.")

    print("""  saqc/funcs/drift.py, correctDrift:

      shift_targets = drift_grouper.aggregate(...).shift(-1)
                                                  ^^^^^^^^^
  Each interval is corrected toward the calibration level of the NEXT
  interval. The last interval has no next one, so its target is NaN, and
  `data + (NaN - fit)` = NaN for every row in it.
""")

    groups = pd.DataFrame({"level": [10.0, 11.0, 12.0, 13.0]}, index=[0, 1, 2, 3])
    groups["target = .shift(-1)"] = groups["level"].shift(-1)
    groups["result"] = np.where(
        groups["target = .shift(-1)"].isna(), "ALL ROWS -> NaN", "corrected"
    )
    print(groups.to_string())


# ---------------------------------------------------------------- part 4
def part4_guard() -> None:
    rule("PART 4 — The guard the wrapper must apply.")

    drifted, clean = build()
    corrected, _ = run(drifted, n_visits=5)

    # Restore anything correctDrift NaN'd out that was present on input.
    destroyed = corrected.isna() & ~drifted.isna()
    guarded = corrected.copy()
    guarded[destroyed] = drifted[destroyed]

    print(f"  rows destroyed by correctDrift : {int(destroyed.sum())}")
    print(f"  NaN after restore              : {int(guarded.isna().sum())}")
    print(f"  restored rows are UNCORRECTED, so report them as such:")
    a = int(np.flatnonzero(destroyed.to_numpy()).min())
    b = int(np.flatnonzero(destroyed.to_numpy()).max())
    rmse = float(np.sqrt(((guarded.iloc[a:b + 1] - clean.iloc[a:b + 1]) ** 2).mean()))
    print(f"      rows {a}..{b} still carry their drift (RMSE vs truth {rmse:.3f})")

    print("\n  So the wrapper must, every time:")
    print("    1. record the NaN mask before correctDrift,")
    print("    2. restore any value it destroyed,")
    print("    3. mark that trailing span uncorrected in the flag log,")
    print("    4. refuse the call outright when there are fewer than 3 visits.")


if __name__ == "__main__":
    print(f"saqc {saqc.__version__} | pandas {pd.__version__} | numpy {np.__version__}")
    part1_the_bug()
    part2_scaling()
    part3_root_cause()
    part4_guard()
