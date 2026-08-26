"""Can SaQC 2.8 shift a level-shifted window back to its surroundings? (§13: probe, never guess)"""
import inspect
import numpy as np
import pandas as pd
import saqc

for name in ("correctOffset", "correctRegimeAnomaly", "flagOffset"):
    print(f"--- {name}\n    {inspect.signature(getattr(saqc.SaQC, name))}\n")

rng = np.random.default_rng(0)
idx = pd.date_range("2024-01-01", periods=600, freq="5min")
base = pd.Series(10 + rng.normal(0, 0.3, 600), index=idx)
shifted = base.copy()
shifted.iloc[200:400] += 6.0                      # a 6-unit level shift
qc = saqc.SaQC(pd.DataFrame({"value": shifted}))

for name, kwargs in (
    ("correctOffset", dict(max_jump=3.0, spread=1.0, window="6h", min_periods=5)),
    ("correctRegimeAnomaly", {}),
):
    try:
        out = getattr(qc, name)("value", **kwargs)
        got = out.data.to_pandas()["value"]
        interior = got.iloc[200:400]
        print(f"{name}: interior median {interior.median():.2f} "
              f"(true {base.iloc[200:400].median():.2f}, uncorrected "
              f"{shifted.iloc[200:400].median():.2f})  "
              f"max resid {np.nanmax(np.abs(got - base)):.2f}")
    except Exception as exc:
        print(f"{name}: {type(exc).__name__}: {exc}")
