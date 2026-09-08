"""Fit ramp_context's thresholds on gauges we do NOT report on.

`slope_context` looks 45 minutes either side; `ramp_context` looks hours. The
question this fits: how long, and how steadily, does a climb have to run before
the point on top of it is better explained as the peak of an event than as an
artifact?

Same discipline as scratchpad/tune_noise_spare_rule.py — fitted on 02054550 and
040851385, never on 01467200 — and the same criterion: spare as many false
positives as possible while losing at most 5% of true spikes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import saqc

from src.agent_tools import context as ctx
from src.agent_tools import wrappers

UNILOF = {"thresh": 1.5, "n": 20}
TUNE = [("02054550", "l1"), ("02054550", "l2"), ("040851385", "l1"), ("040851385", "l2")]


def features(gauge: str, level: str) -> tuple[list[dict], list[dict]]:
    base = f"data/turbidity/injected/{gauge}/{level}/{gauge}_{level}"
    frame = pd.read_csv(f"{base}.csv", parse_dates=["datetime"]).sort_values("datetime")
    series = frame.set_index("datetime")["value"]
    kind = pd.read_csv(f"{base}_labels.csv", parse_dates=["datetime"]
                       ).set_index("datetime")["anomaly_type"].fillna("")

    qc = saqc.SaQC(pd.DataFrame({"value": series}))
    flagged = pd.DatetimeIndex(
        wrappers.flag_spike_unilof(qc, field="value", **UNILOF)["flagged_datetimes"])

    tp, fp = [], []
    for at in flagged:
        r = ctx.ramp_context(series, at=at)
        rise = r.get("rise") or {}
        if rise.get("minutes") is None:
            continue
        row = {"minutes": rise["minutes"] or 0.0,
               "monotonic": rise["monotonic_fraction"] or 0.0}
        (tp if kind.get(at, "") == "spike" else fp).append(row)
    print(f"  {gauge}_{level}: {len(flagged)} flagged -> {len(tp)} true spike / {len(fp)} not")
    return tp, fp


def main() -> None:
    all_tp: list[dict] = []
    all_fp: list[dict] = []
    print("fitting on (never scored):")
    for gauge, level in TUNE:
        tp, fp = features(gauge, level)
        all_tp += tp
        all_fp += fp
    print(f"\npooled: {len(all_tp)} true spikes, {len(all_fp)} false positives\n")

    for name, rows in (("true spike", all_tp), ("false positive", all_fp)):
        for key in ("minutes", "monotonic"):
            vals = sorted(r[key] for r in rows)
            if not vals:
                continue
            q = lambda p: vals[min(int(p * len(vals)), len(vals) - 1)]
            print(f"  {name:<16}{key:<11} p10={q(.1):>7.2f} median={np.median(vals):>7.2f} p90={q(.9):>7.2f}")
    print()

    # NOTE THE DIRECTION. Directness separates the two populations the opposite way to
    # the intuition: a true injected spike reads 0.75 (a clean single jump straight up
    # from its foot) and a false positive reads 0.28 (a climb that wandered). So the
    # rule that spares real water is a LOW directness over a LONG rise — the point sits
    # on top of a meandering climb, which is what water does and what an injected
    # displacement does not.
    print(f"{'rise>=min':>10}{'directness<=':>14}{'FP spared':>11}{'TP lost':>9}{'FP%':>7}{'TP%':>7}")
    best = None
    for mins in (30, 45, 60, 90, 120, 180):
        for mono in (0.20, 0.30, 0.40, 0.50, 0.60):
            spared = sum(1 for r in all_fp if r["minutes"] >= mins and r["monotonic"] <= mono)
            lost = sum(1 for r in all_tp if r["minutes"] >= mins and r["monotonic"] <= mono)
            fpct = 100 * spared / max(len(all_fp), 1)
            tpct = 100 * lost / max(len(all_tp), 1)
            print(f"{mins:>10}{mono:>13}{spared:>11}{lost:>9}{fpct:>6.1f}%{tpct:>6.1f}%")
            if tpct <= 5.0 and (best is None or spared > best[2]):
                best = (mins, mono, spared, lost, fpct, tpct)
    if best:
        mins, mono, spared, lost, fpct, tpct = best
        print(f"\nCHOSEN (max FP spared, <=5% true spikes lost):")
        print(f"  rise >= {mins} min AND directness <= {mono}")
        print(f"  spares {fpct:.1f}% of false positives, costs {tpct:.1f}% of true spikes")


if __name__ == "__main__":
    main()
