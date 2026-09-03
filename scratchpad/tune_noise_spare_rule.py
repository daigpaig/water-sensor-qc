"""Fit the 'spare this deletion' rule on gauges we do NOT report on.

Part A left 104 spike false positives, all of them measured. §7.4's published rule
(noise_ratio > 3 AND step_sigmas_local < 5) spares 0 of them: that rule was fitted
on the retired 15-min gauges, whose FP population was storm limbs at noise_ratio
~11, and it does not transfer to the 5-min bases (FP median noise_ratio 2.2).

The separator is still real — FP median noise_ratio 2.2 vs TP 0.9 — so the
threshold needs refitting. Fitting it on 01467200, the gauge we report on, would
be fitting to the test set. This fits on 02054550 and 040851385 instead.

No agent involved: the chain that matters is detector -> measurement -> label, and
reproducing it directly removes run-to-run agent variability from the fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import saqc

from src.agent_tools import context as ctx
from src.agent_tools import wrappers

# The parameters the Part A run actually used, so the candidate population matches.
UNILOF = {"thresh": 1.5, "n": 20}
TUNE = [("02054550", "l1"), ("02054550", "l2"), ("040851385", "l1"), ("040851385", "l2")]


def candidates(gauge: str, level: str) -> tuple[list[dict], list[dict]]:
    """(true-positive rows, false-positive rows) as describe_points measurement rows."""
    base = f"data/injected/{gauge}/{level}/{gauge}_{level}"
    frame = pd.read_csv(f"{base}.csv", parse_dates=["datetime"]).sort_values("datetime")
    series = frame.set_index("datetime")["value"]
    labels = pd.read_csv(f"{base}_labels.csv", parse_dates=["datetime"]).set_index("datetime")
    kind = labels["anomaly_type"].fillna("")

    qc = saqc.SaQC(pd.DataFrame({"value": series}))
    result = wrappers.flag_spike_unilof(qc, field="value", **UNILOF)
    flagged = pd.DatetimeIndex(result["flagged_datetimes"])

    # Cover EVERY flagged point, in chunks. describe_points truncates to the FIRST
    # max_points timestamps, so a single capped call would fit the rule on an early
    # slice of the record (summer only, on a 2-year series) rather than on the
    # candidate population as a whole. This is offline, so coverage is free.
    ats = [str(t) for t in flagged]
    rows = []
    step = ctx.MAX_POINTS_CEILING
    for i in range(0, len(ats), step):
        chunk = ats[i:i + step]
        rows += ctx.describe_points(series, ats=chunk, max_points=step)["points"]

    tp, fp = [], []
    for row in rows:
        at = pd.Timestamp(row["at"])
        (tp if kind.get(at, "") == "spike" else fp).append(row)
    assert len(rows) == len(flagged), f"{len(rows)} described of {len(flagged)} flagged"
    print(f"  {gauge}_{level}: {len(flagged)} flagged, all described"
          f" -> {len(tp)} true spike / {len(fp)} not")
    return tp, fp


def main() -> None:
    all_tp: list[dict] = []
    all_fp: list[dict] = []
    print("fitting on (never scored):")
    for gauge, level in TUNE:
        tp, fp = candidates(gauge, level)
        all_tp += tp
        all_fp += fp
    print(f"\npooled: {len(all_tp)} true spikes, {len(all_fp)} false positives\n")

    get = lambda r, k, d: (r.get(k) if r.get(k) is not None else d)
    print(f"{'noise>':>7}{'step<':>7}{'FP spared':>11}{'TP lost':>9}{'FP%':>7}{'TP%':>7}")
    best = None
    for n in (1.2, 1.5, 1.8, 2.0, 2.5, 3.0):
        for m in (6, 8, 10, 12, 99):
            spared = sum(1 for r in all_fp
                         if get(r, "noise_ratio", 0) > n and get(r, "step_sigmas_local", 99) < m)
            lost = sum(1 for r in all_tp
                       if get(r, "noise_ratio", 0) > n and get(r, "step_sigmas_local", 99) < m)
            fpct = 100 * spared / max(len(all_fp), 1)
            tpct = 100 * lost / max(len(all_tp), 1)
            print(f"{n:>7}{m:>7}{spared:>11}{lost:>9}{fpct:>6.1f}%{tpct:>6.1f}%")
            # criterion: most FPs spared while losing at most 5% of true spikes
            if tpct <= 5.0 and (best is None or spared > best[2]):
                best = (n, m, spared, lost, fpct, tpct)
    if best:
        n, m, spared, lost, fpct, tpct = best
        print(f"\nCHOSEN (max FP spared, <=5% true spikes lost):")
        print(f"  noise_ratio > {n} AND step_sigmas_local < {m}")
        print(f"  spares {fpct:.1f}% of false positives, costs {tpct:.1f}% of true spikes")


if __name__ == "__main__":
    main()
