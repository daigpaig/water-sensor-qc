"""Fit the 'gradual approach' rule on gauges we do NOT report on.

`markdowns/spike_error_profiles.md` §5: `excursion_context`'s width is measured at
HALF height, so a mountain that is steep at the summit and broad at the foot
returns 1-2 samples — the artifact signature. `scratchpad/audit_recovery_reference.py`
measured that on the 01467200_l1 run of 2026-09-02: 58 of 65 wrongly-deleted
points read <= 1 sample wide, and 15 of those were approached over >= 3 rising
samples. This fits the threshold for that rule.

Fitting it on 01467200, the gauge we report on, would be fitting to the test set —
§7.4 records what that costs (60.2% where fitted, 33% held out). So the fit runs
on 02054550 and 040851385, and 01467200_l1 is scored afterwards as the held-out
check, never as an input.

No agent involved: the chain that matters is detector -> measurement -> label, and
reproducing it directly removes run-to-run agent variability from the fit. Same
structure as `tune_noise_spare_rule.py`, so the two rules stay comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import saqc

from src.agent_tools import context as ctx
from src.agent_tools import wrappers

# The detector settings a run actually gets today (§7.10): an ABSOLUTE LOF
# threshold, n=10. Fitting against the retired quota-based candidate set would
# fit a population no run now sees.
UNILOF = {"thresh": ctx.SPIKE_LOF_THRESH, "n": ctx.SPIKE_LOF_N}

TUNE = [("02054550", "l1"), ("02054550", "l2"), ("040851385", "l1"), ("040851385", "l2")]
HELD_OUT = [("01467200", "l1")]


def candidates(gauge: str, level: str) -> tuple[list[dict], list[dict]]:
    """(true-positive rows, false-positive rows) as describe_points measurement rows."""
    base = f"data/turbidity/injected/{gauge}/{level}/{gauge}_{level}"
    frame = pd.read_csv(f"{base}.csv", parse_dates=["datetime"]).sort_values("datetime")
    series = frame.set_index("datetime")["value"]
    labels = pd.read_csv(f"{base}_labels.csv", parse_dates=["datetime"]).set_index("datetime")
    kind = labels["anomaly_type"].fillna("")

    qc = saqc.SaQC(pd.DataFrame({"value": series}))
    result = wrappers.flag_spike_unilof(qc, field="value", **UNILOF)
    flagged = pd.DatetimeIndex(result["flagged_datetimes"])

    # Cover EVERY flagged point, in chunks: describe_points truncates to the FIRST
    # max_points timestamps, so one capped call would fit on an early slice of the
    # record rather than on the candidate population. Offline, so coverage is free.
    ats = [str(t) for t in flagged]
    rows: list[dict] = []
    step = ctx.MAX_POINTS_CEILING
    for i in range(0, len(ats), step):
        rows += ctx.describe_points(series, ats=ats[i:i + step], max_points=step)["points"]

    tp, fp = [], []
    for row in rows:
        at = pd.Timestamp(row["at"])
        (tp if kind.get(at, "") == "spike" else fp).append(row)
    print(f"  {gauge}_{level}: {len(flagged)} flagged"
          f" -> {len(tp)} true spike / {len(fp)} not")
    return tp, fp


def _mono(row: dict) -> int:
    """Longest monotone run either side, 0 when the measurement is unavailable."""
    before = row.get("n_monotone_before") or 0
    after = row.get("n_monotone_after") or 0
    return max(before, after)


def _sweep(name: str, tp: list[dict], fp: list[dict], rule) -> None:
    spared = sum(1 for r in fp if rule(r))
    lost = sum(1 for r in tp if rule(r))
    fpct = 100 * spared / max(len(fp), 1)
    tpct = 100 * lost / max(len(tp), 1)
    print(f"{name:<34}{spared:>7}{lost:>8}{fpct:>8.1f}%{tpct:>8.1f}%")


def main() -> None:
    all_tp: list[dict] = []
    all_fp: list[dict] = []
    print("fitting on (never scored):")
    for gauge, level in TUNE:
        tp, fp = candidates(gauge, level)
        all_tp += tp
        all_fp += fp
    print(f"\npooled: {len(all_tp)} true spikes, {len(all_fp)} false positives\n")

    print(f"{'spare the point when...':<34}{'FP':>7}{'TP':>8}{'FP%':>9}{'TP%':>9}")
    print("-" * 66)
    best = None
    for k in (2, 3, 4, 5, 6):
        rule = lambda r, k=k: _mono(r) >= k
        _sweep(f"monotone run >= {k}", all_tp, all_fp, rule)
        spared = sum(1 for r in all_fp if rule(r))
        tpct = 100 * sum(1 for r in all_tp if rule(r)) / max(len(all_tp), 1)
        # Same criterion as §7.4's rule: most FPs spared at <= 5% of true spikes.
        if tpct <= 5.0 and (best is None or spared > best[1]):
            best = (k, spared, tpct)

    print()
    # Does the summit-vs-base ratio add anything the monotone run does not?
    for a in (0.4, 0.5, 0.6):
        _sweep(f"apex_ratio <= {a}", all_tp, all_fp,
               lambda r, a=a: (r.get("apex_ratio") or 1.0) <= a)
    if best:
        k = best[0]
        _sweep(f"monotone >= {k} OR apex <= 0.5", all_tp, all_fp,
               lambda r, k=k: _mono(r) >= k or (r.get("apex_ratio") or 1.0) <= 0.5)
        _sweep(f"monotone >= {k} AND apex <= 0.5", all_tp, all_fp,
               lambda r, k=k: _mono(r) >= k and (r.get("apex_ratio") or 1.0) <= 0.5)

    if not best:
        print("\nNo threshold spares anything at <= 5% of true spikes.")
        return

    k, spared, tpct = best
    print(f"\nCHOSEN (max FP spared, <=5% true spikes lost):")
    print(f"  longest monotone run >= {k}")
    print(f"  spares {100 * spared / max(len(all_fp), 1):.1f}% of false positives, "
          f"costs {tpct:.1f}% of true spikes")

    print("\nHELD OUT (never fitted on) — expect the direction to carry, not the magnitude:")
    print(f"{'spare the point when...':<34}{'FP':>7}{'TP':>8}{'FP%':>9}{'TP%':>9}")
    print("-" * 66)
    for gauge, level in HELD_OUT:
        tp, fp = candidates(gauge, level)
        _sweep(f"  {gauge}_{level}: monotone >= {k}", tp, fp, lambda r: _mono(r) >= k)


if __name__ == "__main__":
    main()
