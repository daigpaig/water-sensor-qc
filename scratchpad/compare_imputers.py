"""linear vs rolling median on EVERY injected gap in all nine datasets (§13).

Scored on identical rows: only where BOTH methods produced a value, so coverage
differences cannot flatter either one.
"""
import glob
import warnings

import numpy as np
import pandas as pd
import saqc

warnings.filterwarnings("ignore")
rows = []
for path in sorted(glob.glob("data/turbidity/injected/*/l*/*_l?.csv")):
    stem = path.rsplit("/", 1)[-1][:-4]
    ser = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")["value"]
    lab = pd.read_csv(path.replace(".csv", "_labels.csv"),
                      parse_dates=["datetime"]).set_index("datetime")
    gap = lab[(lab.source == "injected") & (lab.anomaly_type == "gap")]
    if gap.empty:
        continue
    truth = gap["true_value"].astype(float)
    idx = gap.index

    lin = ser.interpolate(method="time", limit_area="inside").reindex(idx)
    for win in ("2h", "6h"):
        roll = saqc.SaQC(pd.DataFrame({"value": ser})).interpolateByRolling(
            "value", window=win, func="median", min_periods=0, flag=25, dfilter=np.inf
        ).data.to_pandas()["value"].reindex(idx)
        both = lin.notna() & roll.notna()
        if not both.any():
            continue
        el = (lin[both] - truth[both]); er = (roll[both] - truth[both])
        rows.append({
            "dataset": stem, "window": win, "n_scored": int(both.sum()),
            "roll_cov": float(roll.notna().mean()), "lin_cov": float(lin.notna().mean()),
            "linear": float(np.sqrt((el ** 2).mean())),
            "rolling": float(np.sqrt((er ** 2).mean())),
        })

df = pd.DataFrame(rows)
for win, g in df.groupby("window"):
    g = g.sort_values("dataset")
    print(f"\n=== rolling window={win} — RMSE on rows BOTH filled ===")
    print(f"{'dataset':<18}{'n':>5}{'linear':>9}{'rolling':>9}   winner")
    for _, r in g.iterrows():
        win_name = "linear" if r["linear"] < r["rolling"] else "ROLLING"
        print(f"{r["dataset"]:<18}{r["n_scored"]:>5}{r["linear"]:>9.3f}{r["rolling"]:>9.3f}   {win_name}")
    print(f"{'MEDIAN':<18}{'':>5}{g["linear"].median():>9.3f}{g["rolling"].median():>9.3f}"
          f"   linear wins {int((g["linear"] < g["rolling"]).sum())}/{len(g)}")
    print(f"coverage: rolling {g["roll_cov"].mean():.0%}   linear {g["lin_cov"].mean():.0%}")
