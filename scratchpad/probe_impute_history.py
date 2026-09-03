"""What does the flag history record for setFlags/processGeneric, and can we name it? (§13)

The flag log identifies imputed rows by matching the test's `func` name in the history
(§7.1). Replacing interpolateByRolling means whatever writes the fill must still be
identifiable there.
"""
import numpy as np
import pandas as pd
import saqc

idx = pd.date_range("2024-01-01", periods=60, freq="5min")
v = pd.Series(np.full(60, 10.0), index=idx)
v.iloc[20:24] = np.nan
qc = saqc.SaQC(pd.DataFrame({"value": v})).flagRange("value", min=0, max=100)

filled = pd.Series(False, index=idx)
filled.iloc[20:24] = True

for kwargs in ({}, {"label": "impute_linear"}):
    out = qc.setFlags("value", data=list(idx[20:24]), flag=25, **kwargs)
    hist = out._flags.history["value"]
    metas = [m.get("func") for m in hist.meta]
    print(f"setFlags(**{kwargs}) -> history funcs {metas}")
