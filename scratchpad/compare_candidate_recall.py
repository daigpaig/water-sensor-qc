"""Attribute the §9.1 candidate-recall failures: 5-min bases vs the retired 15-min ones.

`tests/test_candidates.py` asserts >=95% per-type recall of *injected* rows by the
candidate proposer. After the move to 5-minute bases (§9) some of those assertions
fail, and the question that matters is whether the cadence change caused it or
whether the same assertions already failed on the 15-min datasets. This runs the
identical measurement over both sets and prints them side by side.

Run:  .venv/bin/python -u -m scratchpad.compare_candidate_recall
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.workbench.candidates import REVIEW_TYPES, find_candidates  # noqa: E402

RECALL_TYPES: tuple[str, ...] = (*REVIEW_TYPES, "gap")
NEW = Path("data/injected")
LEGACY = Path("data/legacy_15min/injected")


def recall(path: Path) -> tuple[float, dict[str, tuple[int, int]]]:
    """Identical to tests/test_candidates.py::_recall, on one dataset."""
    result = find_candidates(path)
    labels = (
        pd.read_csv(path.with_name(f"{path.stem}_labels.csv"), parse_dates=["datetime"])
        .set_index("datetime")
        .reindex(result.series.index)
    )
    covered = np.zeros(len(result.series), dtype=bool)
    for c in result.candidates:
        covered[c.start_pos : c.end_pos] = True
    covered |= result.series.isna().to_numpy()

    injected = (
        labels["is_anomaly"].fillna(False).to_numpy(dtype=bool)
        & (labels["source"].fillna("").to_numpy() == "injected")
    )
    atype = labels["anomaly_type"].fillna("").to_numpy()
    per_type = {
        t: (int((injected & (atype == t) & covered).sum()),
            int((injected & (atype == t)).sum()))
        for t in RECALL_TYPES
    }
    total = int(injected.sum())
    return (int((injected & covered).sum()) / total if total else 1.0), per_type


def run(root: Path, tag: str) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/l[1-3]/*_l[1-3].csv")):
        r, per = recall(path)
        row = {"set": tag, "dataset": path.stem, "overall": round(100 * r, 1)}
        for t, (h, n) in per.items():
            row[t] = round(100 * h / n, 1) if n else np.nan
            row[f"{t}_n"] = n
        rows.append(row)
        print(f"  {tag:<7} {path.stem:<16} overall={row['overall']:5.1f}%  " + "  ".join(
            f"{t}={row[t]}% (n={row[f'{t}_n']})" for t in RECALL_TYPES))
    return rows


def main() -> int:
    rows: list[dict] = []
    print("=== 5-min bases (current) ===")
    rows += run(NEW, "5min")
    print("\n=== 15-min bases (retired) ===")
    rows += run(LEGACY, "15min")
    df = pd.DataFrame(rows)
    df.to_csv("scratchpad/out/candidate_recall_compare.csv", index=False)
    print("\n=== median per-type recall by set ===")
    print(df.groupby("set")[list(RECALL_TYPES)].median().round(1).to_string())
    print("\n-> scratchpad/out/candidate_recall_compare.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
