"""How big is a real level shift vs the record's own window-to-window movement?

flag_jumps' statistic is |mean(window before) - mean(window after)|. Setting `thresh`
in raw FNU cannot transfer between gauges, so this measures the statistic's own
distribution per series and asks which normaliser makes the workable threshold a
constant: the value MAD, the value std, or a high quantile of the statistic itself.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INJECTED = ROOT / "data" / "injected"
WINDOWS = ["1h", "3h", "6h", "12h"]


def jump_stat(s: pd.Series, window: str) -> np.ndarray:
    """|mean(bwd window) - mean(fwd window)| at every timestamp, as SaQC computes it."""
    s = s.dropna()
    bwd = s.rolling(window, min_periods=1).mean()
    fwd = s[::-1].rolling(window, min_periods=1, closed="left").mean()[::-1]
    return np.abs(bwd.to_numpy() - fwd.to_numpy())


def main() -> None:
    rows = []
    for csv in sorted(INJECTED.glob("*/l*/*.csv")):
        if csv.name.endswith("_labels.csv"):
            continue
        stem = csv.stem
        df = pd.read_csv(csv, parse_dates=["datetime"]).set_index("datetime").sort_index()
        lab = pd.read_csv(csv.with_name(f"{stem}_labels.csv"), parse_dates=["datetime"])
        v = df["value"]
        arr = v.to_numpy(dtype=float)
        f = arr[np.isfinite(arr)]
        mad = float(np.median(np.abs(f - np.median(f))) * 1.4826)
        std = float(np.nanstd(arr, ddof=1))

        # true shift magnitude = |value - true_value| inside each level_shift span
        m = (lab["anomaly_type"] == "level_shift").to_numpy()
        mags = np.abs(
            pd.to_numeric(lab.loc[m, "true_value"], errors="coerce").to_numpy()
            - v.to_numpy()[m]
        )
        mags = mags[np.isfinite(mags)]

        rec = dict(dataset=stem, median=float(np.nanmedian(arr)), mad=mad, std=std,
                   shift_mag_min=float(mags.min()) if mags.size else np.nan,
                   shift_mag_med=float(np.median(mags)) if mags.size else np.nan,
                   shift_mag_max=float(mags.max()) if mags.size else np.nan)
        for w in WINDOWS:
            st = jump_stat(v, w)
            for q in (0.99, 0.999, 0.9995, 0.9999):
                rec[f"q{q}_{w}"] = float(np.nanquantile(st, q))
        rows.append(rec)

    res = pd.DataFrame(rows).set_index("dataset")
    pd.set_option("display.width", 250)
    print("\n=== scales and true shift magnitudes (FNU) ===")
    print(res[["median", "mad", "std", "shift_mag_min", "shift_mag_med", "shift_mag_max"]].round(3).to_string())
    print("\n=== shift magnitude in units of each candidate normaliser ===")
    out = pd.DataFrame({
        "mag_med/mad": res["shift_mag_med"] / res["mad"],
        "mag_min/mad": res["shift_mag_min"] / res["mad"],
        "mag_med/std": res["shift_mag_med"] / res["std"],
        "mag_min/std": res["shift_mag_min"] / res["std"],
    })
    print(out.round(2).to_string())
    for w in WINDOWS:
        print(f"\n=== window={w}: jump-stat quantiles, raw and /mad ===")
        cols = [c for c in res.columns if c.endswith(f"_{w}")]
        t = res[cols].copy()
        for c in cols:
            t[c + "/mad"] = res[c] / res["mad"]
        print(t.round(2).to_string())
    res.to_csv(ROOT / "scratchpad" / "tune_jumps_scale.csv")


if __name__ == "__main__":
    main()
