"""Fit the level-shift window rule on gauges we do NOT report on.

The definition under test: a level shift is the span between two flag_jumps points whose
interior sits well away from its surroundings AND whose edges are abrupt. The first half
alone does not work — measured on 01467200_l1, 1 of 24 elevated windows was the injected
shift and the largest storm outscored it (z=+8.6 vs +4.5). §9.1 found edge sharpness to
be the lever; this fits its threshold.

Fitted on 02054550 + 040851385, never on 01467200. Reports episode-level recall (did we
find the event at all?) and row-level recall (did we claim the right span?), because an
edge detector scored per row is what produced 0/117 in the first place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import saqc

from src.agent_tools import context as ctx
from src.agent_tools import wrappers

TUNE = [("02054550", "l1"), ("02054550", "l2"), ("02054550", "l3"),
        ("040851385", "l1"), ("040851385", "l2"), ("040851385", "l3")]
JUMPS = {"thresh": 6.0, "window": "6h"}


def episodes(kind: pd.Series) -> list[tuple]:
    """Contiguous level_shift runs, bridged across gaps (§9.1: one event, not fragments)."""
    ls = kind == "level_shift"
    if not ls.any():
        return []
    grp = (ls != ls.shift()).cumsum()[ls]
    runs = [(sub.index[0], sub.index[-1]) for _, sub in ls[ls].groupby(grp)]
    merged = [runs[0]]
    for a, b in runs[1:]:
        if (a - merged[-1][1]) <= pd.Timedelta("6h"):
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def main() -> None:
    print(f"{'sharp>=':>8}{'episodes found':>16}{'windows kept':>14}{'row recall':>13}"
          f"{'row precision':>15}")
    per_thresh = {t: [0, 0, 0, 0, 0] for t in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)}
    for gauge, lvl in TUNE:
        base = f"data/turbidity/injected/{gauge}/{lvl}/{gauge}_{lvl}"
        s = pd.read_csv(f"{base}.csv", parse_dates=["datetime"]).set_index("datetime")["value"]
        kind = pd.read_csv(f"{base}_labels.csv", parse_dates=["datetime"]
                           ).set_index("datetime")["anomaly_type"].fillna("")
        truth_rows = set(kind[kind == "level_shift"].index)
        eps = episodes(kind)
        qc = saqc.SaQC(pd.DataFrame({"value": s}))
        jumps = wrappers.flag_jumps(qc, field="value", **JUMPS)["flagged_datetimes"]

        found = ctx.find_shift_windows(s, jumps, max_windows=999)
        for t in per_thresh:
            kept = [w for w in found["windows"]
                    if (w["onset_sharpness"] or 0) >= t and (w["end_sharpness"] or 0) >= t
                    and abs(w["interior_sigmas"] or 0) >= ctx.SHIFT_INTERIOR_SIGMAS]
            claimed = set()
            for w in kept:
                claimed |= set(s.loc[pd.Timestamp(w["start"]):pd.Timestamp(w["end"])].index)
            hit = sum(1 for a, b in eps
                      if any(pd.Timestamp(w["start"]) <= b and pd.Timestamp(w["end"]) >= a
                             for w in kept))
            acc = per_thresh[t]
            acc[0] += hit; acc[1] += len(eps); acc[2] += len(kept)
            acc[3] += len(claimed & truth_rows); acc[4] += len(claimed)

    for t, (hit, n_eps, kept, right_rows, claimed_rows) in per_thresh.items():
        rr = 100 * right_rows / max(sum_truth, 1) if (sum_truth := SUM_TRUTH) else 0
        rp = 100 * right_rows / max(claimed_rows, 1)
        print(f"{t:>8}{hit}/{n_eps:<14}{kept:>14}{rr:>12.1f}%{rp:>14.1f}%")


SUM_TRUTH = 0
if __name__ == "__main__":
    # total labelled level_shift rows across the tuning set, for row recall
    for g, l in TUNE:
        k = pd.read_csv(f"data/turbidity/injected/{g}/{l}/{g}_{l}_labels.csv", parse_dates=["datetime"]
                        ).set_index("datetime")["anomaly_type"].fillna("")
        SUM_TRUTH += int((k == "level_shift").sum())
    main()
