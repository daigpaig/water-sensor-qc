"""Generate mock data for testing the Streamlit UI's per-decision reasoning view.

Creates:
  data/mock/63680.csv            — synthetic turbidity time series (~1 week, 15-min)
  data/mock/63680_decisions.json — fake per-decision agent reasoning data
  data/mock/63680_clean.csv      — cleaned version with flag column

Run:
    python data/mock/generate_mock_data.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(63680)
    n = 672  # 1 week at 15-min intervals
    start = pd.Timestamp("2023-09-06 00:00:00")
    times = pd.date_range(start, periods=n, freq="15min")

    # Base signal: slowly varying turbidity around 5 FNU with diurnal pattern
    t = np.arange(n)
    base = 5.0 + 1.5 * np.sin(2 * np.pi * t / 96)  # 96 samples = 1 day
    noise = np.random.normal(0, 0.25, n)
    values = base + noise

    # --- Inject anomalies ---

    # Spike 1: single-point spike at index 248 (Sep 8 ~14:00)
    spike1_idx = 248
    values[spike1_idx] = 11.8

    # Spike 2: 2-sample excursion at index 330 (Sep 9 ~10:30)
    spike2_idx = 330
    values[spike2_idx] = 10.5
    values[spike2_idx + 1] = 9.2

    # Plateau: stuck sensor at 4.2 for ~3 hours (indices 450-462)
    plateau_start = 450
    plateau_end = 463
    values[plateau_start:plateau_end] = 4.2

    # Gap: missing data for ~2 hours (indices 520-527)
    gap_start = 520
    gap_end = 528
    values[gap_start:gap_end] = np.nan

    # Storm event (REAL — should be kept): indices 370-395
    storm_start = 370
    storm_len = 26
    storm = np.concatenate([
        np.linspace(0, 4.5, 12),   # rapid rise
        np.linspace(4.5, 0.8, 14), # gradual recession
    ])
    values[storm_start:storm_start + storm_len] += storm

    # The storm peak (borderline — flagged but should be kept)
    borderline_idx = 381  # near storm peak

    df = pd.DataFrame({
        "datetime": times.strftime("%Y-%m-%d %H:%M:%S"),
        "value": np.round(values, 2),
    })

    csv_path = out_dir / "63680.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} rows)")

    # --- Build a clean_df with flags for mock mode ---
    clean_df = df.copy()
    clean_df["flag"] = None
    clean_df.loc[spike1_idx, "flag"] = "flag_spike_unilof"
    clean_df.loc[spike2_idx, "flag"] = "flag_spike_unilof"
    clean_df.loc[spike2_idx + 1, "flag"] = "flag_spike_unilof"
    clean_df.loc[borderline_idx, "flag"] = "flag_spike_unilof"
    for i in range(plateau_start, plateau_end):
        clean_df.loc[i, "flag"] = "flag_constants"
    for i in range(gap_start, gap_end):
        clean_df.loc[i, "flag"] = "flag_nan"

    clean_path = out_dir / "63680_clean.csv"
    clean_df.to_csv(clean_path, index=False)
    print(f"Wrote {clean_path}")

    # --- Decisions JSON ---
    decisions = [
        {
            "id": 1,
            "segment_start": str(times[spike1_idx]),
            "segment_end": str(times[spike1_idx]),
            "value": round(float(values[spike1_idx]), 2),
            "action": "delete",
            "anomaly_type": "spike",
            "flagged_by": "flag_spike_unilof",
            "n_points": 1,
            "reasoning": (
                "1-sample excursion, robust_z=8.3, recovers in 1 sample (15 min), "
                "step_sigmas_local=14.1. Extremely high z and instant recovery "
                "\u2014 textbook spike. No storm context: the surrounding 6 h window "
                "is flat at 5.2\u00b10.3 FNU."
            ),
            "thinking": (
                "Width=1 sample and z=8.3 are clearly spike-like. Recovery=1 sample "
                "(15 min) is instant \u2014 far too fast for any physical event. "
                "step_sigmas_local=14.1 shows the point moves 14\u00d7 its immediate "
                "neighbours. For a real flush event at 15-minute resolution I would "
                "expect recovery \u22659 samples (2.25 h) with a wider excursion. The "
                "instant recovery and width of 1 sample make this unambiguous. Deleted."
            ),
            "neighborhood_start": str(times[spike1_idx - 16]),
            "neighborhood_end": str(times[min(spike1_idx + 16, n - 1)]),
        },
        {
            "id": 2,
            "segment_start": str(times[spike2_idx]),
            "segment_end": str(times[spike2_idx + 1]),
            "value": round(float(values[spike2_idx]), 2),
            "action": "delete",
            "anomaly_type": "spike",
            "flagged_by": "flag_spike_unilof",
            "n_points": 2,
            "reasoning": (
                "2-sample excursion, robust_z=7.7, recovers in 6 samples (30 min), "
                "step_sigmas_local=12.8. High z and locally prominent despite the "
                "slightly extended recovery."
            ),
            "thinking": (
                "Width=2 samples and z=7.7 are clearly spike-like. Recovery=6 "
                "samples=30 min is borderline at 5-minute resolution. However, "
                "step_sigmas_local=12.8 shows the point moves 12\u00d7 its immediate "
                "neighbours \u2014 far above the ~1.6 level of a false positive. For a "
                "real flush event at 5-minute resolution I would expect recovery "
                "\u22659 samples (45 min) with a wider excursion. The 30-min recovery is "
                "fast and the width is narrow, so the spike interpretation wins. "
                "Deleted. Would have kept if recovery were \u22659 samples or width "
                "\u22654 samples."
            ),
            "neighborhood_start": str(times[spike2_idx - 16]),
            "neighborhood_end": str(times[min(spike2_idx + 17, n - 1)]),
        },
        {
            "id": 3,
            "segment_start": str(times[borderline_idx]),
            "segment_end": str(times[borderline_idx]),
            "value": round(float(values[borderline_idx]), 2),
            "action": "keep",
            "anomaly_type": "storm_peak",
            "flagged_by": "flag_spike_unilof",
            "n_points": 1,
            "reasoning": (
                "Flagged by UniLOF (thresh=1.5), but describe_point returns: "
                "width=14 samples, recovery=22 samples (5.5 h), robust_z=1.8, "
                "reads_like='storm_peak'. The excursion is 14 samples wide and "
                "takes over 5 hours to recover \u2014 this is a genuine storm "
                "first-flush event, not a sensor artifact. Kept as real signal."
            ),
            "thinking": (
                "This point is at the top of what looks like a storm hydrograph. "
                "The width (14 samples) and recovery time (22 samples, 5.5 h) are "
                "both well outside spike range. robust_z=1.8 is elevated but not "
                "extreme \u2014 within the range I\u2019d expect for a moderate storm peak. "
                "The surrounding hours show a gradual rise over ~3 h followed by a "
                "slow recession, which is the textbook shape of a turbidity response "
                "to a rainfall event. UniLOF flagged it because the local density is "
                "low at the peak, but that is exactly what a storm peak looks like. "
                "If I deleted this I would be removing real water-quality signal. "
                "Overriding the default \u2018delete\u2019 action \u2014 keeping this as genuine "
                "extreme event."
            ),
            "neighborhood_start": str(times[borderline_idx - 24]),
            "neighborhood_end": str(times[min(borderline_idx + 24, n - 1)]),
        },
        {
            "id": 4,
            "segment_start": str(times[plateau_start]),
            "segment_end": str(times[plateau_end - 1]),
            "value": 4.2,
            "action": "delete",
            "anomaly_type": "plateau",
            "flagged_by": "flag_constants",
            "n_points": plateau_end - plateau_start,
            "reasoning": (
                "13-sample constant run at exactly 4.2 FNU (3 h 15 min). "
                "thresh=0.01, so any reading with <0.01 variation over the window "
                "is flagged. The surrounding noise sd is 0.25 FNU, so 13 identical "
                "readings is impossible from real water \u2014 stuck sensor. Deleted."
            ),
            "thinking": (
                "flagConstants found a 13-sample run at exactly 4.2 with zero "
                "variation. The series\u2019 noise sd in this window is ~0.25 FNU, so "
                "the probability of 13 consecutive readings within 0.01 of each "
                "other by chance is vanishingly small. This is a stuck sensor \u2014 "
                "the reading froze, probably due to a fouled wiper or a firmware "
                "hang. The value 4.2 is within the normal range for this sensor, "
                "so it is not obviously wrong by magnitude alone, but the "
                "zero-variation signature is definitive. Deleting the entire "
                "3.25 h window."
            ),
            "neighborhood_start": str(times[plateau_start - 16]),
            "neighborhood_end": str(times[min(plateau_end + 16, n - 1)]),
        },
        {
            "id": 5,
            "segment_start": str(times[gap_start]),
            "segment_end": str(times[gap_end - 1]),
            "value": None,
            "action": "impute",
            "anomaly_type": "gap",
            "flagged_by": "flag_nan",
            "n_points": gap_end - gap_start,
            "reasoning": (
                "8-sample gap (2 h). Short enough to impute defensibly with rolling "
                "median (window=3h). Surrounding values are stable at 5.1\u00b10.3 FNU "
                "with no trend \u2014 interpolation will land in the right range. "
                "Filled 8 of 8 samples."
            ),
            "thinking": (
                "flag_nan found 8 consecutive NaN rows spanning 2 hours. The values "
                "on either side are stable (5.0\u20135.4 FNU, no trend, no storm in "
                "progress), so a rolling median fill will produce reasonable values. "
                "window=\u20183h\u2019 is wider than the gap (2 h) so interpolateByRolling "
                "will have context on both sides. I would NOT impute if the gap were "
                "during a storm recession or if the surrounding values showed a trend, "
                "because the median would smooth out real dynamics. Here the signal "
                "is flat, so imputation is safe."
            ),
            "neighborhood_start": str(times[gap_start - 16]),
            "neighborhood_end": str(times[min(gap_end + 16, n - 1)]),
        },
    ]

    decisions_path = out_dir / "63680_decisions.json"
    decisions_path.write_text(json.dumps(decisions, indent=2, default=str))
    print(f"Wrote {decisions_path} ({len(decisions)} decisions)")


if __name__ == "__main__":
    main()
