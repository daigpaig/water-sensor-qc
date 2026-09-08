"""How should flagUniLOF's threshold be set, if not by a fixed candidate count? (§13)

The budget rule pins the candidate count at ~175 on every record, so on a gauge with 21
injected spikes it emits 168 candidates and caps detector precision at 0.13 before the
agent looks at anything (08041770_l1, run R). A record with 5 spikes and one with 300
should not produce the same number of candidates.

LOF is a LOCAL DENSITY RATIO and therefore already scale-free -- |LOF| = 2 means "twice
as sparse as its neighbours" on any river at any cadence -- so unlike flag_jumps' FNU
threshold (§7.6) it has no units problem for a quantile to solve.

The property being tested is NOT recall or precision alone: it is whether the candidate
count TRACKS the number of anomalies actually present. That is what the budget destroys.

|LOF| is computed ONCE per dataset; every rule is then just a threshold on the same
vector, so the whole sweep costs one assignUniLOF per file.
"""
import glob
import warnings

import numpy as np
import pandas as pd
import saqc

from src.agent_tools import context as C

warnings.filterwarnings("ignore")
MAD = 1.4826


def rules(lof: pd.Series, n_rows: int) -> dict:
    """threshold -> label, for each candidate rule."""
    med, mad = float(lof.median()), MAD * float((lof - lof.median()).abs().median())
    out = {}
    for k in (1.3, 1.5, 1.8, 2.2):
        out[f"abs {k}"] = k
    for k in (4, 6, 8):
        out[f"mad {k}"] = med + k * mad
    # the current rule, for comparison
    out["budget 175"] = float(lof.quantile(max(0.0, 1 - 175 / n_rows)))
    return out


rows = []
for path in sorted(glob.glob("data/turbidity/injected/*/l*/*_l?.csv")
                   + glob.glob("data/legacy_15min/injected/*/l*/*_l?.csv")):
    stem = path.rsplit("/", 1)[-1][:-4]
    cadence = "15min" if "legacy" in path else "5min"
    v = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")["value"]
    lab = pd.read_csv(path.replace(".csv", "_labels.csv"),
                      parse_dates=["datetime"]).set_index("datetime")
    truth = set(lab.index[lab.anomaly_type == "spike"])
    if not truth:
        continue
    qc = saqc.SaQC(pd.DataFrame({"value": v}))
    lof = qc.assignUniLOF("value", target="_l", n=C.SPIKE_LOF_N).data.to_pandas()["_l"].abs()
    lof = lof.replace([np.inf, -np.inf], np.nan).dropna()
    for name, th in rules(lof, len(v)).items():
        hit = set(lof.index[lof >= th])
        tp = len(hit & truth)
        rows.append({"dataset": stem, "cadence": cadence, "rule": name,
                     "n_true": len(truth), "n_cand": len(hit),
                     "recall": tp / len(truth),
                     "prec": tp / max(len(hit), 1)})
    print(f"  done {stem}", flush=True)

df = pd.DataFrame(rows)
df.to_csv("scratchpad/out/lof_rule.csv", index=False)
pd.set_option("display.width", 200)

print("\n=== does the candidate count TRACK the number of true spikes? ===")
print(f"{'rule':<12}{'corr(n_cand, n_true)':>22}{'median cand':>13}{'cand range':>16}"
      f"{'median recall':>15}{'median prec':>13}")
for name, g in df.groupby("rule"):
    corr = g["n_cand"].corr(g["n_true"], method="spearman")
    print(f"{name:<12}{corr:>22.2f}{g.n_cand.median():>13.0f}"
          f"{(str(int(g.n_cand.min()))+'-'+str(int(g.n_cand.max()))):>16}"
          f"{g.recall.median():>15.3f}{g.prec.median():>13.3f}")

print("\n=== the sparse gauge that exposed this (21 true spikes) ===")
sub = df[df.dataset == "08041770_l1"]
print(sub[["rule", "n_cand", "recall", "prec"]].to_string(index=False))
