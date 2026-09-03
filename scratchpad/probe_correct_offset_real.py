"""Is SaQC's correctOffset safe to turn loose on a storm-driven record? (§13)"""
import numpy as np
import pandas as pd
import saqc

df = pd.read_csv("data/injected/01467200/l1/01467200_l1.csv", parse_dates=["datetime"])
s = df.set_index("datetime")["value"]
qc = saqc.SaQC(pd.DataFrame({"value": s}))
lab = pd.read_csv("data/injected/01467200/l1/01467200_l1_labels.csv", parse_dates=["datetime"])
shift = lab.loc[lab.anomaly_type == "level_shift", "datetime"]
lo, hi = shift.min(), shift.max()

for max_jump in (3.0, 6.0):
    out = qc.correctOffset("value", max_jump=max_jump, spread=1.0,
                           window="12h", min_periods=5)
    got = out.data.to_pandas()["value"]
    diff = (got - s).abs()
    changed = diff > 1e-9
    inside = changed[(changed.index >= lo) & (changed.index <= hi)].sum()
    outside = int(changed.sum()) - int(inside)
    print(f"max_jump={max_jump}: changed {int(changed.sum()):,} rows "
          f"({int(inside)} inside the real shift, {outside:,} elsewhere); "
          f"largest change {diff.max():.1f} FNU")
