"""Where does flagPlateau break in SaQC 2.8.0?

Motivation: flagPlateau appeared to hang on a 21k-row turbidity series while
every other §7 detector returned in under a second.

Run this from a FILE, not `python -` : flagPlateau spawns multiprocessing
workers, and each worker re-imports __main__. When __main__ is stdin the import
fails, the pool respawns, and the "hang" is actually an infinite spawn loop.

    PYTHONPATH=. python scratchpad/probe_plateau_cost.py

Sweeps growing prefixes of each series, because the crash is NOT length-
monotonic: the question is *where* it breaks, not *above what size*. Every call
is wrapped, so one failure does not end the sweep.
"""
from __future__ import annotations

import glob
import time

import pandas as pd
import saqc

from src.inspect_data import DATETIME_COL, load_series, reindex_to_grid

# The three approved bases (CLAUDE.md §9). The original probe ran on a clean
# segment of 11501000, a gauge since retired as mostly provisional.
PATHS = sorted(glob.glob("data/raw/approved/*_turbidity_*.csv"))


def probe(path: str) -> None:
    df = reindex_to_grid(load_series(path))
    full = df.set_index(DATETIME_COL)["value"].astype(float)
    print(f"\n{path}  ({len(full):,} rows)")

    lengths = (500, 1000, 2000, 3000, 4000, 8000, 16000, 32000, len(full))
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
                print(f"  n={n:<6} {str(kw):<45} {time.time() - t:7.2f}s  flagged={flagged}")
            except Exception as exc:
                print(
                    f"  n={n:<6} {str(kw):<45} {time.time() - t:7.2f}s  "
                    f"RAISED {type(exc).__name__}: {exc}"
                )


def main() -> None:
    if not PATHS:
        raise SystemExit("no approved bases found in data/raw/approved/")
    for path in PATHS:
        probe(path)


if __name__ == "__main__":
    main()
