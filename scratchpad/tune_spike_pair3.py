"""UniLOF vs flagZScore vs their union, at a MATCHED candidate budget.

The first pass targeted each detector's budget by quantile of a hand-computed statistic.
That worked for LOF (assignUniLOF exposes the real score) and failed for the z-score,
whose SaQC implementation differs from the rolling median/MAD approximation -- it landed
505-1114 candidates against a 150 target, so the z-score was judged with 3-6x the budget.
Here both are bisected on the ACTUAL detector call so the comparison is like for like.
"""
import glob
import warnings

import numpy as np
import pandas as pd
import saqc

warnings.filterwarnings("ignore")
EACH = 150


def fit_budget(call, lo, hi, budget, steps=22):
    """Highest thresh whose candidate count is <= budget (recall falls as thresh rises)."""
    best = (np.nan, 0, set())
    for _ in range(steps):
        mid = (lo + hi) / 2
        idx = call(round(mid, 3))
        if len(idx) <= budget:
            if len(idx) > best[1]:
                best = (round(mid, 3), len(idx), idx)
            hi = mid
        else:
            lo = mid
    return best


rows = []
for path in sorted(glob.glob("data/injected/*/l*/*_l?.csv")):
    stem = path.rsplit("/", 1)[-1][:-4]
    v = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")["value"]
    lab = pd.read_csv(path.replace(".csv", "_labels.csv"),
                      parse_dates=["datetime"]).set_index("datetime")
    truth = set(lab.index[lab.anomaly_type == "spike"])
    if not truth:
        continue
    qc = saqc.SaQC(pd.DataFrame({"value": v}))
    mad_step = 1.4826 * float(v.diff().abs().median())

    lof_th, lof_n, lof_hit = fit_budget(
        lambda t: set(v.index[(qc.flagUniLOF("value", n=10, thresh=t)
                               .flags["value"] > 0).to_numpy()]), 1.0, 8.0, EACH)
    z_th, z_n, z_hit = fit_budget(
        lambda t: set(v.index[(qc.flagZScore("value", method="modified", window="6h",
                                             thresh=t, min_residuals=3 * mad_step)
                               .flags["value"] > 0).to_numpy()]), 2.0, 400.0, EACH)
    both = lof_hit | z_hit
    solo_th, solo_n, solo_hit = fit_budget(
        lambda t: set(v.index[(qc.flagUniLOF("value", n=10, thresh=t)
                               .flags["value"] > 0).to_numpy()]), 1.0, 8.0, 300)
    rows.append({
        "dataset": stem, "spikes": len(truth),
        "lof_th": lof_th, "lof_n": lof_n, "lof_rec": len(lof_hit & truth) / len(truth),
        "z_th": z_th, "z_n": z_n, "z_rec": len(z_hit & truth) / len(truth),
        "union_n": len(both), "union_rec": len(both & truth) / len(truth),
        "z_only": len((z_hit - lof_hit) & truth), "lof_only": len((lof_hit - z_hit) & truth),
        "solo300_n": solo_n, "solo300_rec": len(solo_hit & truth) / len(truth),
    })
    print(f"  done {stem}", flush=True)

df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print("\n", df.to_string(index=False))
print(f"\nmedian recall at ~{EACH} candidates EACH:  "
      f"lof {df.lof_rec.median():.3f}   zscore {df.z_rec.median():.3f}   "
      f"UNION {df.union_rec.median():.3f}")
print(f"union candidate range {df.union_n.min()}-{df.union_n.max()} "
      f"(measure cap is 300)")
print(f"LOF ALONE at a 300 budget: median {df.solo300_rec.median():.3f}  "
      f"(union uses {df.union_n.median():.0f} candidates for {df.union_rec.median():.3f})")
print(f"union beats LOF-alone-at-300 on {int((df.union_rec > df.solo300_rec).sum())}/9 datasets")
print(f"spikes only z finds: {df.z_only.tolist()}\nspikes only LOF finds: {df.lof_only.tolist()}")
