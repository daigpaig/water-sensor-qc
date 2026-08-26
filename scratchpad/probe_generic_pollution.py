"""Does processGeneric write into the DEFINING MODULE's globals? (§13)"""
import numpy as np
import pandas as pd
import saqc

BEFORE = set(globals())

def _shift_back(v):
    return v - 1.0

idx = pd.date_range("2024-01-01", periods=50, freq="5min")
qc = saqc.SaQC(pd.DataFrame({"value": pd.Series(np.full(50, 10.0), index=idx)}))
qc.processGeneric("value", func=_shift_back, dfilter=np.inf)

added = sorted(set(globals()) - BEFORE - {"BEFORE", "_shift_back", "idx", "qc", "added"})
print(f"{len(added)} names injected into this module's globals by processGeneric:")
print(" ", ", ".join(added[:40]))
print("\nshadowed builtins:", [n for n in added if n in dir(__builtins__)])
print("globals()['max'] is now:", globals().get("max"))
