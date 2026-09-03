"""Does a CANDIDATE BUDGET transfer where a fixed threshold does not? (§7.6's lesson)

No static summary stat predicts the right flagUniLOF thresh (tune_unilof_by_noise.py).
§7.6 hit the same wall on flag_jumps and the answer was to quote a quantile of the
detector's own statistic. Here the equivalent is a budget: take the top-N points by LOF
score. That is self-normalising, and N is the unit the pipeline actually constrains --
describe_points caps at 300, so a budget is a promise the run can keep.
"""
import glob
import warnings

import numpy as np
import pandas as pd
import saqc

warnings.filterwarnings("ignore")
BUDGETS = (150, 300, 600, 1200)
GRID = [round(x, 2) for x in np.arange(1.02, 3.01, 0.02)]

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
    curve = []
    for th in GRID:
        hit = (qc.flagUniLOF("value", n=20, thresh=th).flags["value"] > 0).to_numpy()
        idx = set(v.index[hit])
        curve.append((th, len(idx), len(idx & truth) / len(truth)))

    row = {"dataset": stem, "n_spikes": len(truth)}
    for b in BUDGETS:
        under = [c for c in curve if c[1] <= b]
        th, n, rec = max(under, key=lambda c: c[1]) if under else (np.nan, 0, 0.0)
        row[f"th@{b}"], row[f"n@{b}"], row[f"rec@{b}"] = th, n, rec
    rows.append(row)
    print(f"  done {stem}", flush=True)

df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print("\n", df.to_string(index=False))
print("\nrecall at each budget — median across the nine datasets:")
for b in BUDGETS:
    print(f"  top-{b:<5} recall median {df[f'rec@{b}'].median():.3f}   "
          f"min {df[f'rec@{b}'].min():.3f}   thresh range "
          f"{df[f'th@{b}'].min():.2f}-{df[f'th@{b}'].max():.2f}")
