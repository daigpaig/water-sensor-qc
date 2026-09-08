"""Why does flagUniLOF miss locally-obvious spikes? (§13)"""
import warnings

import numpy as np
import pandas as pd
import saqc

warnings.filterwarnings("ignore")

inj = pd.read_csv("data/turbidity/injected/01467200/l1/01467200_l1.csv",
                  parse_dates=["datetime"]).set_index("datetime")["value"]
lab = pd.read_csv("data/turbidity/injected/01467200/l1/01467200_l1_labels.csv",
                  parse_dates=["datetime"]).set_index("datetime")
truth = lab.index[lab.anomaly_type == "spike"]
at = pd.Timestamp("2024-04-17 08:30:00")

print(f"record: median {inj.median():.2f}  robust_sigma "
      f"{1.4826 * (inj - inj.median()).abs().median():.2f} FNU")

# the LOF score this point actually gets
qc = saqc.SaQC(pd.DataFrame({"value": inj})).assignUniLOF("value", target="lof", n=20)
lof = qc.data.to_pandas()["lof"]
print(f"\nLOF score at {at}: {lof[at]:.3f}   (a run flags when |score| >= thresh)")
print(f"LOF percentile of that score: {(lof.abs() < abs(lof[at])).mean():.4%}")

print("\nthresh sweep — flagUniLOF(n=20) against the injected spike labels")
print(f"{'thresh':>7}{'flagged':>10}{'TP':>6}{'recall':>8}{'precision':>11}   caught 04-17?")
for th in (1.1, 1.2, 1.3, 1.5, 1.8, 2.5):
    out = saqc.SaQC(pd.DataFrame({"value": inj})).flagUniLOF("value", n=20, thresh=th)
    hit = out.flags["value"] > 0
    idx = inj.index[hit.to_numpy()]
    tp = len(set(idx) & set(truth))
    print(f"{th:>7}{len(idx):>10}{tp:>6}{tp/len(truth):>8.3f}"
          f"{tp/max(len(idx),1):>11.3f}   {'YES' if at in set(idx) else 'no'}")

# does a second detector see it?
print("\nsecond opinions on that one point:")
mad = 1.4826 * (inj.diff().abs().median())
for name, out in (
    ("flagZScore(modified, 12h, thresh=8)",
     saqc.SaQC(pd.DataFrame({"value": inj})).flagZScore(
         "value", method="modified", window="12h", thresh=8, min_residuals=3 * mad)),
    ("flagZScore(modified, 3h, thresh=6)",
     saqc.SaQC(pd.DataFrame({"value": inj})).flagZScore(
         "value", method="modified", window="3h", thresh=6, min_residuals=3 * mad)),
):
    hit = out.flags["value"] > 0
    idx = set(inj.index[hit.to_numpy()])
    tp = len(idx & set(truth))
    print(f"  {name:<38} flagged {len(idx):>6}  spike recall {tp/len(truth):.3f}"
          f"  caught 04-17? {'YES' if at in idx else 'no'}")
