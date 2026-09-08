"""UniLOF + flagZScore together, each sized by a QUANTILE of its own statistic (§7.6).

No summary stat of the record predicts flagUniLOF's thresh: contamination dominates
(Spearman +0.78 against level, which is unknowable at runtime) and only three gauges
exist to fit anything else on. §7.6 hit the identical wall on flag_jumps and the answer
was to stop guessing -- reproduce the statistic the detector thresholds, on THIS record,
and quote a quantile. The quantile is self-calibrating: it lands the same candidate
count on every gauge whatever the absolute scale.
"""
import glob
import warnings

import numpy as np
import pandas as pd
import saqc

warnings.filterwarnings("ignore")
BUDGET = 300


def thresh_for(qc, v, budget, **kw):
    """Highest thresh whose candidate count still fits *budget*."""
    best = (np.nan, 0)
    for th in np.arange(1.05, 6.01, 0.05):
        hit = (qc.flagUniLOF("value", thresh=round(th, 2), **kw).flags["value"] > 0).to_numpy()
        n = int(hit.sum())
        if n <= budget and n > best[1]:
            best = (round(th, 2), n)
        if n == 0:
            break
    return best


rows = []
for path in sorted(glob.glob("data/turbidity/injected/*/l*/*_l?.csv")):
    stem = path.rsplit("/", 1)[-1][:-4]
    v = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")["value"]
    lab = pd.read_csv(path.replace(".csv", "_labels.csv"),
                      parse_dates=["datetime"]).set_index("datetime")
    truth = set(lab.index[lab.anomaly_type == "spike"])
    if not truth:
        continue
    qc = saqc.SaQC(pd.DataFrame({"value": v}))
    mad_step = 1.4826 * float(v.diff().abs().median())

    # --- LOF, sized by a quantile of this record's own |LOF| scores ---------
    lof = qc.assignUniLOF("value", target="lof", n=10).data.to_pandas()["lof"].abs()
    q_lof = float(lof.quantile(1 - 200 / len(v)))          # aim ~200 candidates
    lof_hit = set(v.index[(qc.flagUniLOF("value", n=10, thresh=round(q_lof, 2))
                           .flags["value"] > 0).to_numpy()])

    # --- z-score, sized by a quantile of its own modified-z statistic -------
    med = v.rolling("6h", center=True, min_periods=5).median()
    mad = 1.4826 * (v - med).abs().rolling("6h", center=True, min_periods=5).median()
    z = ((v - med).abs() / np.maximum(mad, 3 * mad_step)).replace([np.inf], np.nan)
    q_z = float(z.quantile(1 - 150 / len(v)))              # aim ~150 candidates
    z_hit = set(v.index[(qc.flagZScore("value", method="modified", window="6h",
                                       thresh=round(q_z, 2), min_residuals=3 * mad_step)
                         .flags["value"] > 0).to_numpy()])

    both = lof_hit | z_hit
    rows.append({
        "dataset": stem, "spikes": len(truth),
        "lof_th": round(q_lof, 2), "lof_n": len(lof_hit),
        "lof_rec": len(lof_hit & truth) / len(truth),
        "z_th": round(q_z, 2), "z_n": len(z_hit),
        "z_rec": len(z_hit & truth) / len(truth),
        "union_n": len(both), "union_rec": len(both & truth) / len(truth),
        "z_adds": len((z_hit - lof_hit) & truth),
    })
    print(f"  done {stem}", flush=True)

df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print("\n", df.to_string(index=False))
print(f"\nmedian: lof {df.lof_rec.median():.3f}  zscore {df.z_rec.median():.3f}  "
      f"UNION {df.union_rec.median():.3f}   union candidates "
      f"{df.union_n.min()}-{df.union_n.max()}")
print(f"true spikes the z-score adds that LOF missed: {df.z_adds.tolist()}")
