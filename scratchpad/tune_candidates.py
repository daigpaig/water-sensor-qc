"""Pick review-friendly default thresholds for src/workbench/candidates.py.

The review page is only usable if each detector proposes tens of segments, not
thousands. This prints, per clean gauge, how many flagged rows and how many
grouped segments each setting yields. Excludes flagPlateau (see
probe_plateau_cost.py).
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import saqc

from src.inspect_data import DATETIME_COL, load_series, reindex_to_grid
from src.workbench.candidates import _runs, robust_scales


def flagged(s: pd.Series, func: str, **kw) -> np.ndarray:
    qc = saqc.SaQC(pd.DataFrame({"value": s}))
    out = getattr(qc, func)("value", **kw)
    m = out.flags["value"] > saqc.UNFLAGGED
    return m.reindex(s.index, fill_value=False).fillna(False).to_numpy()


def report(tag: str, mask: np.ndarray, bridge: int) -> None:
    segs = _runs(mask, bridge=bridge)
    print(f"    {tag:<44} rows={mask.sum():>6}  segments={len(segs):>5}")


def main() -> None:
    for path in sorted(glob.glob("data/raw/approved/*.csv")):
        df = reindex_to_grid(load_series(path))
        s = df.set_index(DATETIME_COL)["value"].astype(float)
        level, step = robust_scales(s)
        nan = s.isna().to_numpy()
        bridge = 8  # 2h on a 15min grid
        print(f"\n{path.split('/')[-1]}  n={len(s)}  level={level:.3g} step={step:.3g}")

        print("  spike / flagUniLOF")
        for thresh in (1.2, 1.5, 2.0, 3.0):
            report(f"thresh={thresh}", flagged(s, "flagUniLOF", n=20, thresh=thresh) & ~nan, bridge)

        print("  spike / flagZScore(modified, 12h)")
        for thresh in (4.0, 8.0, 12.0, 20.0, 30.0):
            report(f"thresh={thresh}", flagged(s, "flagZScore", method="modified",
                                               window="12h", thresh=thresh) & ~nan, bridge)

        print("  plateau / flagConstants(window=3h)")
        for mult in (0.0, 0.1, 0.5):
            report(f"thresh={mult}*step={mult * step:.4g}",
                   flagged(s, "flagConstants", thresh=mult * step, window="3h",
                           min_periods=2) & ~nan, bridge)

        print("  level_shift / flagJumps(window=12h)")
        for mult in (4.0, 8.0, 16.0, 32.0):
            report(f"thresh={mult}*step={mult * step:.4g}",
                   flagged(s, "flagJumps", thresh=mult * step, window="12h") & ~nan, bridge)


if __name__ == "__main__":
    main()
