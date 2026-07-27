"""Why level_shift over-flags, and what the sharpness filter does about it.

flagJumps flags any change of `thresh` within `window`. On storm-driven
turbidity that is every rising and falling limb of every storm — "gradual
slopes" a reviewer rejects. This probe established:

- Neither `window` nor `thresh` fixes it: a shorter window catches MORE (every
  fast wiggle), a longer one still catches gradual storms.
- Persistence (before/after medians differ days later) does not fix it either:
  turbidity storms have multi-week recessions, and wet seasons genuinely shift
  the baseline for weeks — both indistinguishable from a sensor step by
  persistence alone.
- Sharpness — the largest single-sample move as a fraction of the net step —
  does target the stated complaint: it keeps abrupt steps and drops gradual
  slopes. It cut candidates 85/23/5 -> 8/1/0 across the three gauges.
- BUT the survivors are flash-flood ONSETS (e.g. 06818000 has none; 03447687's
  survivors are all storm rising limbs that jump in one sample then keep
  climbing). A real sensor step and a sudden storm are identical in one series.
  This is the level_shift analogue of the drift problem (CLAUDE.md §9.2).

Run from a file / -m, never `python -` (flagPlateau's multiprocessing trap, §7.1
— flagJumps is fine, but keep the habit).
"""
from __future__ import annotations

import pandas as pd

from src.inspect_data import DATETIME_COL, load_series, reindex_to_grid
from src.tools.candidates import DetectConfig, _run_saqc, _runs, _shift_metrics, robust_scales

GAUGES = (
    "data/clean/03447687_clean_20230807_20250101.csv",
    "data/clean/06818000_clean_20240515_20250123.csv",
    "data/clean/11501000_clean_20231226_20240803.csv",
)


def _rows(td: str) -> int:
    return int(pd.Timedelta(td) / pd.Timedelta("15min"))


def main() -> None:
    cfg = DetectConfig()
    persist = _rows(cfg.shift_persist_window)
    for path in GAUGES:
        s = reindex_to_grid(load_series(path)).set_index(DATETIME_COL)["value"].astype(float)
        _, step = robust_scales(s)
        nan = s.isna().to_numpy()
        mask = _run_saqc(
            s, "flagJumps", {"thresh": cfg.jumps_thresh_sigmas * step, "window": cfg.jumps_window}
        ).to_numpy() & ~nan
        runs = _runs(mask, bridge=_rows(cfg.segment_bridge))

        kept = 0
        for a, b in runs:
            net, sharp = _shift_metrics(s, slice(a, b), persist)
            if net >= cfg.jumps_thresh_sigmas * step and sharp >= cfg.shift_min_sharpness:
                kept += 1
        print(f"{path.split('/')[-1][:12]}  step={step:.3g}  "
              f"raw flagJumps={len(runs):>3}  after sharpness filter={kept:>3}")


if __name__ == "__main__":
    main()
