"""Is `n` the parameter that matters, not `thresh`? (§9.1 found n=20 -> 10 for spikes)

n is a NEIGHBOURHOOD SIZE IN SAMPLES, so at 5-min cadence n=20 spans 100 minutes and a
1-3 sample spike is a small perturbation of its own neighbourhood's density. §9.1's
candidate tuner already found n=10 far better for spike recall; this asks the same
question of the agent-facing detector, at a candidate budget the pipeline can measure.
"""
import glob
import warnings

import numpy as np
import pandas as pd
import saqc

warnings.filterwarnings("ignore")
GRID = [round(x, 2) for x in np.arange(1.05, 3.51, 0.15)]
BUDGET = 300

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
    row = {"dataset": stem, "n_spikes": len(truth)}
    for n in (5, 10, 20):
        best = (np.nan, 0, 0.0)
        for th in GRID:
            hit = (qc.flagUniLOF("value", n=n, thresh=th).flags["value"] > 0).to_numpy()
            idx = set(v.index[hit])
            if len(idx) <= BUDGET and len(idx) > best[1]:
                best = (th, len(idx), len(idx & truth) / len(truth))
        row[f"n={n}"] = best[2]
        row[f"th{n}"] = best[0]
    rows.append(row)
    print(f"  done {stem}", flush=True)

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(f"\nspike recall at a {BUDGET}-candidate budget (what describe_points can measure)")
print(df[["dataset", "n_spikes", "n=5", "n=10", "n=20", "th5", "th10", "th20"]].to_string(index=False))
print("\nmedian recall:  n=5 %.3f   n=10 %.3f   n=20 %.3f"
      % (df["n=5"].median(), df["n=10"].median(), df["n=20"].median()))
print("min recall:     n=5 %.3f   n=10 %.3f   n=20 %.3f"
      % (df["n=5"].min(), df["n=10"].min(), df["n=20"].min()))
