"""Where does flagPlateau break in SaQC 2.8.0?

Motivation: flagPlateau appeared to hang on a 21k-row turbidity series while
every other §7 detector returned in under a second.

Run this from a FILE, not `python -` : flagPlateau spawns multiprocessing
workers, and each worker re-imports __main__. When __main__ is stdin the import
fails, the pool respawns, and the "hang" is actually an infinite spawn loop.
"""
from __future__ import annotations

import time

import pandas as pd
import saqc

from src.inspect_data import DATETIME_COL, load_series, reindex_to_grid

PATH = "data/clean/11501000_clean_20231226_20240803.csv"


def main() -> None:
    df = reindex_to_grid(load_series(PATH))
    full = df.set_index(DATETIME_COL)["value"].astype(float)
    print(f"full series: {len(full)} rows")

    lengths = (500, 1000, 2000, 3000, 4000, 8000, len(full))
    for n in lengths:
        s = full.iloc[:n]
        for kw in (
            {"min_length": "1h"},
            {"min_length": "1h", "granularity": 10},
        ):
            t = time.time()
            try:
                qc = saqc.SaQC(pd.DataFrame({"value": s}))
                out = qc.flagPlateau("value", **kw)
                flagged = int((out.flags["value"] > saqc.UNFLAGGED).sum())
                print(f"  n={n:<6} {kw}  {time.time() - t:7.2f}s  flagged={flagged}")
            except Exception as exc:
                print(
                    f"  n={n:<6} {kw}  {time.time() - t:7.2f}s  "
                    f"RAISED {type(exc).__name__}: {exc}"
                )


if __name__ == "__main__":
    main()
