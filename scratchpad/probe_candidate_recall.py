"""candidates.py recall against the injected labels: detectors vs the cap.

This is what established that `max_per_type=50` — not the detectors — was the
binding constraint on spike recall, so the cap is now opt-in (§9.1). Reports
event-level recall (an injected event counts as found if ANY candidate touches
it) as well as row-level.

    .venv/bin/python -m scratchpad.measure_candidate_recall2
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.workbench.candidates import REVIEW_TYPES, DetectConfig, _runs, find_candidates

ROOT = Path("data/injected")


def masks(result):
    n = len(result.series)
    any_mask = result.series.isna().to_numpy().copy()
    for c in result.candidates:
        any_mask[c.start_pos : c.end_pos] = True
    return any_mask


def report(tag, result, is_anom, atype, source):
    any_mask = masks(result)
    inj = is_anom & (source == "injected")
    row_hits = int((inj & any_mask).sum())
    line = [f"  {tag:<9} rows {100 * row_hits / inj.sum():5.1f}%"]
    for t in (*REVIEW_TYPES, "gap"):
        sel = inj & (atype == t)
        if not sel.any():
            line.append(f"{t}: n/a")
            continue
        # event level: contiguous runs of this type
        events = _runs(sel)
        found = sum(1 for a, b in events if any_mask[a:b].any())
        rows = int((sel & any_mask).sum())
        line.append(
            f"{t[:5]} row {100 * rows / sel.sum():5.1f}% ev {found}/{len(events)}"
        )
    print("   ".join(line))


def main() -> None:
    for path in sorted(ROOT.glob("*/l*/*.csv")):
        if path.stem.endswith("_labels"):
            continue
        labels = pd.read_csv(path.with_name(f"{path.stem}_labels.csv"),
                             parse_dates=["datetime"])
        print(f"\n=== {path.stem} ===")
        for tag, cfg in (("default", None),  # uncapped since 2026-07-29
                         ("capped50", DetectConfig(max_per_type=50))):
            res = find_candidates(path, config=cfg)
            lab = labels.set_index("datetime").reindex(res.series.index)
            is_anom = lab["is_anomaly"].fillna(False).to_numpy(dtype=bool)
            atype = lab["anomaly_type"].fillna("").to_numpy()
            source = lab["source"].fillna("").to_numpy()
            report(tag, res, is_anom, atype, source)


if __name__ == "__main__":
    main()
