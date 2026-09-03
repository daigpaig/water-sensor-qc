"""Does min_residuals fix flagZScore's quantisation blow-up?

On 11501000 (turbidity quantised to 0.1, step MAD 0.148) flagZScore(modified)
saturates near 195 segments no matter how high `thresh` goes: in a 12h window
where the MAD is ~0, any wiggle scores an enormous z. `min_residuals` sets a
floor in DATA units, which should kill exactly that pathology.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import saqc

from src.inspect_data import DATETIME_COL, load_series, reindex_to_grid
from src.workbench.candidates import _runs, robust_scales


def main() -> None:
    for path in sorted(glob.glob("data/raw/approved/*.csv")):
        df = reindex_to_grid(load_series(path))
        s = df.set_index(DATETIME_COL)["value"].astype(float)
        level, step = robust_scales(s)
        nan = s.isna().to_numpy()
        print(f"\n{path.split('/')[-1]}  n={len(s)}  level={level:.3g} step={step:.3g}")
        for thresh in (6.0, 10.0):
            for mult in (0.0, 1.0, 3.0, 6.0):
                qc = saqc.SaQC(pd.DataFrame({"value": s}))
                out = qc.flagZScore(
                    "value", method="modified", window="12h", thresh=thresh,
                    min_residuals=mult * step,
                )
                m = (out.flags["value"] > saqc.UNFLAGGED).reindex(
                    s.index, fill_value=False
                ).fillna(False).to_numpy() & ~nan
                print(f"  thresh={thresh:<5} min_residuals={mult}*step={mult * step:<8.4g}"
                      f" rows={m.sum():>5} segments={len(_runs(m, 8)):>4}")


if __name__ == "__main__":
    main()
