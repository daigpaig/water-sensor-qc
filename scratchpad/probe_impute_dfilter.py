"""Can interpolateByRolling be stopped from overwriting already-flagged rows?

SaQC masks rows whose flag is >= `dfilter` before a function runs, so a row an earlier
detector flagged looks MISSING to the imputer, which duly fills it — silently replacing
a real reading. Measured on 03447687_l1: 1,879 rows that were never NaN were rewritten
with a rolling median (§7.1). This probes whether raising `dfilter` prevents that while
still filling genuine gaps.
"""
import numpy as np
import pandas as pd
import saqc

idx = pd.date_range("2024-01-01", periods=60, freq="15min")
values = 10 + np.zeros(60)
values[20] = 80.0          # a spike: real reading, will be flagged, must NOT be filled
values[40:43] = np.nan     # a genuine gap: must be filled

for label, kwargs in (
    ("default (dfilter unset)", {}),
    ("dfilter=np.inf", {"dfilter": np.inf}),
):
    qc = saqc.SaQC(pd.DataFrame({"value": values}, index=idx))
    qc = qc.flagRange("value", min=0, max=50)              # flags the spike
    out = qc.interpolateByRolling("value", window="3h", func="median",
                                  min_periods=0, flag=25, **kwargs)
    got = out.data.to_pandas()["value"]
    spike_kept = np.isclose(got.iloc[20], 80.0)
    gap_filled = got.iloc[40:43].notna().all()
    print(f"{label:26s} spike preserved: {str(spike_kept):5s}   gap filled: {gap_filled}")
    print(f"{'':26s} value at the spike: {got.iloc[20]:.2f} (was 80.00)")
