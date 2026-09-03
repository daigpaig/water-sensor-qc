"""Where is the recall/candidate knee for the quantile rule? Per dataset, not pooled."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, saqc

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ["3h", "6h", "12h"]
QS = [0.98, 0.99, 0.995]


def jump_stat(s, window):
    s = s.dropna()
    bwd = s.rolling(window, min_periods=1).mean()
    fwd = s[::-1].rolling(window, min_periods=1, closed="left").mean()[::-1]
    return np.abs(bwd.to_numpy() - fwd.to_numpy())


def label_spans(lab, kind):
    m = (lab["anomaly_type"] == kind).to_numpy()
    idx = pd.DatetimeIndex(lab["datetime"])
    out, s = [], None
    for i, v in enumerate(m):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((idx[s], idx[i - 1])); s = None
    if s is not None:
        out.append((idx[s], idx[-1]))
    return out


rows = []
for csv in sorted((ROOT / "data/injected").glob("*/l*/*.csv")):
    if csv.name.endswith("_labels.csv"):
        continue
    df = pd.read_csv(csv, parse_dates=["datetime"]).set_index("datetime").sort_index()
    lab = pd.read_csv(csv.with_name(f"{csv.stem}_labels.csv"), parse_dates=["datetime"])
    truth = label_spans(lab, "level_shift")
    v = df["value"]
    for w in WINDOWS:
        tol = pd.Timedelta(w)
        st = jump_stat(v, w)
        for q in QS:
            th = float(np.nanquantile(st, q))
            out = saqc.SaQC(df[["value"]].copy()).flagJumps("value", thresh=th, window=w)
            fl = out.flags["value"]
            hit = pd.DatetimeIndex(fl.index[fl.to_numpy() > 0])
            det = sum(any((h >= s - tol) and (h <= e + tol) for h in hit) for s, e in truth)
            rows.append(dict(dataset=csv.stem, gauge=csv.stem.split("_")[0], window=w, q=q,
                             thresh=th, n_events=len(truth), detected=det, n_flagged=len(hit)))
            print(f"{csv.stem:14s} {w:>3s} q={q} thresh={th:8.2f} {det}/{len(truth)} flags={len(hit)}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(ROOT / "scratchpad/tune_jumps_knee.csv", index=False)
pd.set_option("display.width", 220)
print("\n=== pooled ===")
print(res.groupby(["window", "q"]).apply(
    lambda g: pd.Series({"events": f"{g.detected.sum()}/{g.n_events.sum()}",
                         "recall": g.detected.sum() / g.n_events.sum(),
                         "flags_min": g.n_flagged.min(), "flags_med": g.n_flagged.median(),
                         "flags_max": g.n_flagged.max()}), include_groups=False).round(3).to_string())
print("\n=== by gauge ===")
print(res.groupby(["gauge", "window", "q"]).apply(
    lambda g: pd.Series({"events": f"{g.detected.sum()}/{g.n_events.sum()}",
                         "flags_med": g.n_flagged.median()}), include_groups=False).to_string())
