"""median-anchored vs point-anchored linear vs rolling median, all nine datasets (§13).

Two questions:
  1. Does anchoring cost accuracy on clean gaps?
  2. Does it survive a JUNK READING at the gap edge, which is the failure a rolling
     median shrugs off and a two-point line does not (§6: artifacts cluster at gap edges).
"""
import glob
import warnings

import numpy as np
import pandas as pd
import saqc

from src.agent_tools.wrappers import _linear_fill

warnings.filterwarnings("ignore")


def rmse(pred, truth, idx):
    e = (pred.reindex(idx) - truth).dropna()
    return float(np.sqrt((e ** 2).mean())) if len(e) else float("nan")


clean_rows, junk_rows = [], []
for path in sorted(glob.glob("data/injected/*/l*/*_l?.csv")):
    stem = path.rsplit("/", 1)[-1][:-4]
    ser = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")["value"]
    lab = pd.read_csv(path.replace(".csv", "_labels.csv"),
                      parse_dates=["datetime"]).set_index("datetime")
    gap = lab[(lab.source == "injected") & (lab.anomaly_type == "gap")]
    if gap.empty:
        continue
    truth, idx = gap["true_value"].astype(float), gap.index

    anchored, _, _ = _linear_fill(ser, "6h")
    point = ser.interpolate(method="time", limit_area="inside")
    roll = saqc.SaQC(pd.DataFrame({"value": ser})).interpolateByRolling(
        "value", window="6h", func="median", min_periods=0, flag=25, dfilter=np.inf
    ).data.to_pandas()["value"]
    clean_rows.append({"dataset": stem, "anchored": rmse(anchored, truth, idx),
                       "point": rmse(point, truth, idx), "rolling": rmse(roll, truth, idx)})

    # now corrupt the single reading immediately before each injected gap
    dirty = ser.copy()
    runs = (gap.index.to_series().diff() != pd.Timedelta("5min")).cumsum()
    scale = float(ser.dropna().std())
    for _, grp in gap.groupby(runs):
        before = dirty.loc[:grp.index[0]].dropna()
        if len(before) > 1:
            dirty.loc[before.index[-1]] = before.iloc[-1] + 8 * scale   # telemetry junk
    a2, _, _ = _linear_fill(dirty, "6h")
    p2 = dirty.interpolate(method="time", limit_area="inside")
    r2 = saqc.SaQC(pd.DataFrame({"value": dirty})).interpolateByRolling(
        "value", window="6h", func="median", min_periods=0, flag=25, dfilter=np.inf
    ).data.to_pandas()["value"]
    junk_rows.append({"dataset": stem, "anchored": rmse(a2, truth, idx),
                      "point": rmse(p2, truth, idx), "rolling": rmse(r2, truth, idx)})

for name, rows in (("CLEAN gap edges", clean_rows), ("ONE JUNK READING at each gap edge", junk_rows)):
    df = pd.DataFrame(rows)
    print(f"\n=== {name} — RMSE ===")
    print(f"{'dataset':<16}{'anchored':>10}{'point-lin':>11}{'rolling':>10}")
    for _, r in df.iterrows():
        print(f"{r['dataset']:<16}{r['anchored']:>10.3f}{r['point']:>11.3f}{r['rolling']:>10.3f}")
    print(f"{'MEDIAN':<16}{df['anchored'].median():>10.3f}"
          f"{df['point'].median():>11.3f}{df['rolling'].median():>10.3f}")
