"""How do you write corrected values into a SaQC object without losing flag history? (§13)

§7.1 burned an hour on `dfilter` once already (interpolateByRolling silently overwrote
1,887 real readings), so the masking behaviour is probed here rather than assumed.
"""
import numpy as np
import pandas as pd
import saqc

rng = np.random.default_rng(0)
idx = pd.date_range("2024-01-01", periods=300, freq="5min")
s = pd.Series(10 + rng.normal(0, 0.3, 300), index=idx)
s.iloc[100:200] += 6.0
qc = saqc.SaQC(pd.DataFrame({"value": s})).flagRange("value", min=0, max=14)
n_flagged = int((qc.flags["value"] > 0).sum())

lo, hi = idx[100], idx[199]
def shift_back(v):
    out = v.copy()
    out[(out.index >= lo) & (out.index <= hi)] -= 6.0
    return out

for dfilter in (-np.inf, np.inf):
    out = qc.processGeneric("value", func=shift_back, dfilter=dfilter)
    got = out.data.to_pandas()["value"]
    nan_made = int(got.isna().sum()) - int(s.isna().sum())
    print(f"dfilter={dfilter}: interior median {got.iloc[100:200].median():.2f} "
          f"(target ~10.0)  new NaNs {nan_made}  "
          f"flags kept {int((out.flags['value'] > 0).sum())}/{n_flagged}  "
          f"hist cols {out._flags.history['value'].hist.shape[1]}")
