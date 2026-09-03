"""Does a step-vs-noise guard separate a real level shift from a storm? (§13)

`_edge_sharpness` is max(single-sample move) / |net step| -- a pure RATIO with no scale
check. On a noisy storm limb the net step is the same size as the ordinary jitter, so the
ratio parks near 1.0 and reports "the level moved in one sample" when nothing sharp
happened. Measured on run P's false positive: net step 3.35 FNU against local noise 1.93.
The real shift reads 6.75 against 0.59. This sweeps the cutoff on all nine datasets.
"""
import glob
import warnings

import numpy as np
import pandas as pd
import saqc

from src.agent_tools import context as C
from src.agent_tools.wrappers import flag_jumps

warnings.filterwarnings("ignore")


def step_over_noise(v, at, k=6, span="6h"):
    t = pd.Timestamp(at)
    pos = v.index.get_indexer([t], method="nearest")[0]
    seg = v.iloc[max(0, pos - k): pos + k + 1].to_numpy(float)
    if len(seg) < 4:
        return None
    b = np.nanmedian(seg[: max(1, len(seg) // 3)])
    a = np.nanmedian(seg[-max(1, len(seg) // 3):])
    net = abs(a - b)
    w = v.loc[t - pd.Timedelta(span): t + pd.Timedelta(span)]
    sig = 1.4826 * float(w.diff().abs().median())
    if not np.isfinite(net) or sig <= 0:
        return None
    return net / sig


real, false = [], []
for path in sorted(glob.glob("data/injected/*/l*/*_l?.csv")):
    stem = path.rsplit("/", 1)[-1][:-4]
    v = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")["value"]
    lab = pd.read_csv(path.replace(".csv", "_labels.csv"),
                      parse_dates=["datetime"]).set_index("datetime")
    shifts = lab.index[lab.anomaly_type == "level_shift"]
    if not len(shifts):
        continue
    # true shift windows, as contiguous label runs
    step = pd.Series(v.index).diff().median()
    runs, cur = [], [shifts[0]]
    for a, b in zip(shifts, shifts[1:]):
        if b - a == step:
            cur.append(b)
        else:
            runs.append((cur[0], cur[-1])); cur = [b]
    runs.append((cur[0], cur[-1]))

    qc = saqc.SaQC(pd.DataFrame({"value": v}))
    js = C.jump_scale(v)
    th = js.get("recommended_thresh")
    if not th:
        continue
    qc = flag_jumps(qc=qc, thresh=th, window="6h")["qc"]
    out = C.find_shift_windows(source=qc)
    for w in out["windows"]:
        if w["reads_like"] != "level-shift-like":
            continue
        ws, we = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        hit = any(not (we < a or ws > b) for a, b in runs)
        for edge in (ws, we):
            r = step_over_noise(v, edge)
            if r is not None:
                (real if hit else false).append(r)
    print(f"  done {stem}", flush=True)

real, false = np.array(real), np.array(false)
print(f"\nedges of REAL shift windows : n={len(real)}  median {np.median(real):.2f}"
      f"  p10 {np.percentile(real,10):.2f}")
print(f"edges of FALSE windows      : n={len(false)}  median {np.median(false):.2f}"
      f"  p90 {np.percentile(false,90):.2f}")
print(f"\n{'cutoff':>8}{'real kept':>12}{'false killed':>15}")
for c in (2, 3, 4, 5, 6, 8):
    print(f"{c:>8}{(real>=c).mean():>12.1%}{(false<c).mean():>15.1%}")
