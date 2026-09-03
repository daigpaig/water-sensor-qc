"""flag_jumps.thresh set from the series' OWN jump-statistic distribution.

`tune_jumps_scale.py` showed no fixed multiple of the value MAD (or std) transfers:
the 99th-percentile 3h window-mean-difference is 2.9 / 32.9 / 1.4 x MAD on the three
5-min gauges. What DOES transfer is a quantile of the statistic flagJumps itself
thresholds, because that is measured on the same axis as the decision.

This sweeps thresh = quantile(|mean(bwd) - mean(fwd)|, q) over q and window, and
reports event recall, candidate count, and the per-dataset raw threshold it implies.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import saqc

ROOT = Path(__file__).resolve().parents[1]
INJECTED = ROOT / "data" / "injected"

WINDOWS = ["1h", "3h", "6h", "12h"]
QS = [0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995]


def jump_stat(s: pd.Series, window: str) -> np.ndarray:
    s = s.dropna()
    bwd = s.rolling(window, min_periods=1).mean()
    fwd = s[::-1].rolling(window, min_periods=1, closed="left").mean()[::-1]
    return np.abs(bwd.to_numpy() - fwd.to_numpy())


def label_spans(labels: pd.DataFrame, kind: str):
    m = (labels["anomaly_type"] == kind).to_numpy()
    idx = pd.DatetimeIndex(labels["datetime"])
    spans, start = [], None
    for i, v in enumerate(m):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append((idx[start], idx[i - 1]))
            start = None
    if start is not None:
        spans.append((idx[start], idx[-1]))
    return spans


def main() -> None:
    rows = []
    for csv in sorted(INJECTED.glob("*/l*/*.csv")):
        if csv.name.endswith("_labels.csv"):
            continue
        stem = csv.stem
        df = pd.read_csv(csv, parse_dates=["datetime"]).set_index("datetime").sort_index()
        lab = pd.read_csv(csv.with_name(f"{stem}_labels.csv"), parse_dates=["datetime"])
        truth = label_spans(lab, "level_shift")
        v = df["value"]

        for window in WINDOWS:
            tol = pd.Timedelta(window)
            st = jump_stat(v, window)
            for q in QS:
                thresh = float(np.nanquantile(st, q))
                qc = saqc.SaQC(df[["value"]].copy())
                try:
                    out = qc.flagJumps("value", thresh=thresh, window=window)
                except Exception as exc:  # noqa: BLE001
                    print(f"{stem} {window} q={q}: FAILED {exc}")
                    continue
                fl = out.flags["value"]
                hit = pd.DatetimeIndex(fl.index[fl.to_numpy() > 0])
                detected = sum(
                    any((h >= s - tol) and (h <= e + tol) for h in hit) for s, e in truth
                )
                rows.append(dict(dataset=stem, window=window, q=q, thresh=thresh,
                                 n_events=len(truth), detected=detected,
                                 n_flagged=len(hit),
                                 per_month=len(hit) / 24.0))
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "scratchpad" / "tune_jumps_quantile.csv", index=False)
    pd.set_option("display.width", 220)
    for window in WINDOWS:
        w = res[res.window == window]
        print(f"\n=== window={window} ===")
        g = w.groupby("q").apply(
            lambda x: pd.Series({
                "events": f"{x.detected.sum()}/{x.n_events.sum()}",
                "recall": x.detected.sum() / x.n_events.sum(),
                "flags_min": x.n_flagged.min(),
                "flags_med": x.n_flagged.median(),
                "flags_max": x.n_flagged.max(),
            }), include_groups=False)
        print(g.round(3).to_string())
    print("\n=== per-dataset recall at window=3h ===")
    p = res[res.window == "3h"].pivot_table(index="dataset", columns="q",
                                            values="detected", aggfunc="sum")
    n = res[res.window == "3h"].groupby("dataset").n_events.first()
    print(pd.concat([n.rename("n_events"), p], axis=1).to_string())
    print("\n=== flags at window=3h ===")
    print(res[res.window == "3h"].pivot_table(index="dataset", columns="q",
                                              values="n_flagged").to_string())
    print("\n=== implied raw thresh (FNU) at window=3h ===")
    print(res[res.window == "3h"].pivot_table(index="dataset", columns="q",
                                              values="thresh").round(2).to_string())


if __name__ == "__main__":
    main()
