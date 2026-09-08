"""What dataset characteristic should set flagUniLOF's thresh? (§7.2, §13)

For each of the nine injected datasets: measure the record's own scale, then find the
HIGHEST thresh still reaching 90% / 95% / 100% recall of the injected spikes. Highest,
because recall falls monotonically with thresh, so the highest one meeting a target is
the most selective setting that meets it -- fewest candidates for the same recall.

Candidate count is reported at every step because it is the binding constraint, not a
footnote: describe_points caps at MAX_POINTS_CEILING (300), and §7.5 measured that the
run's largest precision gain came from measuring EVERY flagged point. A threshold that
buys recall by emitting 600 candidates puts the agent back to deleting rows it never
looked at, which is how spike precision was 0.227.
"""
import glob
import warnings

import numpy as np
import pandas as pd
import saqc

warnings.filterwarnings("ignore")

GRID = [round(x, 2) for x in np.arange(1.02, 3.01, 0.02)]
TARGETS = (0.90, 0.95, 1.00)


def characterise(v: pd.Series) -> dict:
    d = v.dropna()
    med = float(d.median())
    robust_sigma = 1.4826 * float((d - med).abs().median())
    step_sigma = 1.4826 * float(d.diff().abs().median())      # §7.4's noise scale
    return {
        "median": med,
        "robust_sigma": robust_sigma,
        "step_sigma": step_sigma,
        "rel_noise": step_sigma / med if med else np.nan,     # noise per unit level
        "spread_ratio": robust_sigma / step_sigma if step_sigma else np.nan,
    }


rows = []
for path in sorted(glob.glob("data/turbidity/injected/*/l*/*_l?.csv")):
    stem = path.rsplit("/", 1)[-1][:-4]
    v = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")["value"]
    lab = pd.read_csv(path.replace(".csv", "_labels.csv"),
                      parse_dates=["datetime"]).set_index("datetime")
    truth = set(lab.index[lab.anomaly_type == "spike"])
    if not truth:
        continue
    ch = characterise(v)
    qc = saqc.SaQC(pd.DataFrame({"value": v}))

    curve = []
    for th in GRID:
        hit = (qc.flagUniLOF("value", n=20, thresh=th).flags["value"] > 0).to_numpy()
        idx = set(v.index[hit])
        curve.append((th, len(idx), len(idx & truth) / len(truth)))

    row = {"dataset": stem, **ch, "n_spikes": len(truth)}
    for t in TARGETS:
        ok = [c for c in curve if c[2] >= t - 1e-9]
        if ok:
            th, n, rec = max(ok, key=lambda c: c[0])
            row[f"th{int(t*100)}"], row[f"n{int(t*100)}"] = th, n
        else:
            row[f"th{int(t*100)}"], row[f"n{int(t*100)}"] = np.nan, np.nan
    rows.append(row)
    print(f"  done {stem}", flush=True)

df = pd.DataFrame(rows)
df.to_csv("scratchpad/out/unilof_by_noise.csv", index=False)
pd.set_option("display.width", 200)
print("\n", df.to_string(index=False))
