"""Tune the noise-context measurement: does local noise separate FPs from real spikes?

The problem this exists to answer (CLAUDE.md §7.3): in a stretch where the sensor
is noisy, the spike detectors fire on a great many points and the agent deletes
them all as separate sensor failures. Nothing the agent can currently call tells
it that the *stretch* is noisy — every measurement it has is about one point in
isolation, and a point that is 8 global sigmas out is 8 global sigmas out whether
its neighbours are calm or thrashing.

So: build candidate measurements of local noise, and check whether they separate

  A  injected spikes            (labels: is_anomaly & anomaly_type == 'spike')
  B  false positives            (flagged by a sensitive UniLOF, labelled normal)
  C  ordinary normal rows       (random sample of labelled-normal rows)

Run from a file, never a heredoc (§7.1 multiprocessing trap does not apply here,
but keep the habit).

    python -m scratchpad.tune_noise_context
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import saqc

from src.inspect_data import load_series

MAD_TO_SIGMA = 1.4826
DATASETS = [
    ("03447687", "l1"), ("03447687", "l2"),
    ("02198840", "l2"),
    ("08041770", "l2"),
]


def mad_sigma(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if clean.size < 2:
        return 0.0
    return float(MAD_TO_SIGMA * np.median(np.abs(clean - np.median(clean))))


def block_noise(diffs: np.ndarray, block: int) -> np.ndarray:
    """Robust first-difference scale of every non-overlapping block of *diffs*.

    Vectorised: reshape to (n_blocks, block) and take the MAD along axis 1. This is
    the cheap way to get a whole-record distribution of "how noisy is a window".
    """
    n = (len(diffs) // block) * block
    if n < block:
        return np.array([])
    grid = diffs[:n].reshape(-1, block)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(grid, axis=1, keepdims=True)
        sigma = MAD_TO_SIGMA * np.nanmedian(np.abs(grid - med), axis=1)
    return sigma[np.isfinite(sigma)]


def residuals(series: pd.Series, smooth: int = 5) -> pd.Series:
    """Value minus a short centred rolling median.

    This is the measurement that separates NOISE from MOVEMENT. A first-difference
    scale counts a smooth storm limb as "variable" because consecutive samples
    differ a lot; a residual around a local smooth curve does not — on a ramp the
    residual is ~0, while a thrashing sensor scatters around its own curve.
    """
    return series - series.rolling(smooth, center=True, min_periods=1).median()


def measure(
    series: pd.Series,
    resid: pd.Series,
    pos: int,
    half: int,
    baseline: float,
    global_step: float,
    resid_baseline: float,
    global_resid: float,
) -> dict:
    """Candidate noise measurements around position *pos*, window +/- half samples."""
    lo, hi = max(0, pos - half), min(len(series), pos + half + 1)
    win = series.iloc[lo:hi]
    d = win.diff().to_numpy()
    local = mad_sigma(d)
    local = max(local, 1e-9)

    # Residual-based noise: scatter around the local smooth curve.
    rwin = resid.iloc[lo:hi].to_numpy()
    local_resid = max(mad_sigma(rwin), 1e-9)
    point_resid = float(resid.iloc[pos])

    # If you delete this point for being N global sigmas out, how many OTHER
    # samples in the window clear the same bar? A large number means the bar is
    # picking out the stretch, not the point.
    finite_r = np.abs(rwin[np.isfinite(rwin)])
    pct_beyond = (
        float((finite_r >= 3 * global_resid).mean() * 100) if finite_r.size else np.nan
    )

    # Lag-1 autocorrelation of the first differences: white noise ~ -0.5,
    # a smooth directional ramp ~ 0 or positive.
    fd = d[np.isfinite(d)]
    if fd.size >= 4 and np.std(fd) > 0:
        acf1 = float(np.corrcoef(fd[:-1], fd[1:])[0, 1])
    else:
        acf1 = np.nan

    # The point's own single-sample move, scored against local vs global scale.
    prev_v = series.iloc[pos - 1] if pos > 0 else np.nan
    next_v = series.iloc[pos + 1] if pos + 1 < len(series) else np.nan
    v = series.iloc[pos]
    steps = [abs(v - prev_v), abs(next_v - v)]
    step = float(np.nanmax(steps)) if np.isfinite(steps).any() else np.nan

    # How many OTHER samples in the window move at least as much as this one? If
    # many do, deleting this one and not those is arbitrary.
    other = np.abs(d[np.isfinite(d)])
    as_extreme = float((other >= step).mean() * 100) if other.size and np.isfinite(step) else np.nan

    # Turning-point fraction: white noise reverses direction ~2/3 of the time, a
    # smooth storm limb almost never. Separates "noisy" from "moving fast".
    finite = d[np.isfinite(d)]
    if finite.size >= 3:
        sign = np.sign(finite)
        turns = float((sign[1:] * sign[:-1] < 0).mean())
    else:
        turns = np.nan

    return {
        "noise_ratio": local / baseline,
        "residual_ratio": local_resid / resid_baseline,
        "step_sigmas_global": step / global_step,
        "step_sigmas_local": step / local,
        "resid_sigmas_global": abs(point_resid) / global_resid,
        "resid_sigmas_local": abs(point_resid) / local_resid,
        "pct_window_beyond_3g": pct_beyond,
        "pct_window_as_extreme": as_extreme,
        "turning_fraction": turns,
        "diff_acf1": acf1,
    }


def quantiles(rows: list[dict], key: str) -> str:
    vals = np.array([r[key] for r in rows if np.isfinite(r.get(key, np.nan))])
    if vals.size == 0:
        return "     n/a"
    return f"{np.percentile(vals, 25):7.2f} {np.median(vals):7.2f} {np.percentile(vals, 75):7.2f}"


def separation(a: list[dict], b: list[dict], key: str) -> float:
    """P(a > b) for random draws — 0.5 is a coin flip, 1.0 is perfect separation."""
    x = np.array([r[key] for r in a if np.isfinite(r.get(key, np.nan))])
    y = np.array([r[key] for r in b if np.isfinite(r.get(key, np.nan))])
    if x.size == 0 or y.size == 0:
        return float("nan")
    return float((x[:, None] > y[None, :]).mean())


def run_one(gauge: str, level: str, half_hours: float) -> tuple[list, list, list]:
    root = f"data/injected/{gauge}/{level}/{gauge}_{level}"
    df = load_series(f"{root}.csv")
    labels = pd.read_csv(f"{root}_labels.csv", parse_dates=["datetime"])
    series = df.set_index("datetime")["value"].astype(float).sort_index()

    step_td = series.index.to_series().diff().median()
    half = max(4, int(round(pd.Timedelta(hours=half_hours) / step_td)))

    diffs = series.diff().to_numpy()
    global_step = max(mad_sigma(diffs), 1e-9)
    blocks = block_noise(diffs, half * 2 + 1)
    baseline = max(float(np.median(blocks)), 1e-9) if blocks.size else global_step

    resid = residuals(series)
    resid_arr = resid.to_numpy()
    global_resid = max(mad_sigma(resid_arr), 1e-9)
    rblocks = block_noise(resid_arr, half * 2 + 1)
    resid_baseline = max(float(np.median(rblocks)), 1e-9) if rblocks.size else global_resid

    # A sensitive spike run, so B is the population that actually causes the
    # complaint: points a detector fires on that are not anomalies at all.
    qc = saqc.SaQC(pd.DataFrame({"value": series}))
    qc = qc.flagUniLOF("value", n=20, thresh=1.5)
    flags = qc.flags["value"]
    flagged = set(flags[flags > 0].index)

    lab = labels.set_index("datetime")
    is_spike = lab["is_anomaly"].fillna(False).astype(bool) & (lab["anomaly_type"] == "spike")
    spike_ts = set(is_spike[is_spike].index)
    normal_ts = set(lab.index[~lab["is_anomaly"].fillna(False).astype(bool)])

    rng = np.random.default_rng(0)
    pos_of = {ts: i for i, ts in enumerate(series.index)}

    def sample(stamps, limit):
        stamps = [s for s in stamps if s in pos_of and np.isfinite(series.iloc[pos_of[s]])]
        if len(stamps) > limit:
            stamps = [stamps[i] for i in rng.choice(len(stamps), limit, replace=False)]
        return [
            measure(series, resid, pos_of[s], half, baseline, global_step,
                    resid_baseline, global_resid)
            for s in stamps
        ]

    a = sample(sorted(spike_ts & flagged), 400)                    # true spikes, detected
    b = sample(sorted(flagged & normal_ts), 400)                   # false positives
    c = sample(sorted(normal_ts - flagged), 400)                   # ordinary normal rows
    return a, b, c


def main() -> None:
    for half_hours in (0.75, 1.0, 1.5, 2.0, 3.0):
        print(f"\n{'='*78}\nwindow = +/-{half_hours} h\n{'='*78}")
        all_a, all_b, all_c = [], [], []
        for gauge, level in DATASETS:
            a, b, c = run_one(gauge, level, half_hours)
            all_a += a
            all_b += b
            all_c += c
            print(f"  {gauge}_{level}: {len(a)} true spikes, {len(b)} false positives, {len(c)} normal")

        keys = ["noise_ratio", "residual_ratio", "step_sigmas_global", "step_sigmas_local",
                "resid_sigmas_global", "resid_sigmas_local", "pct_window_beyond_3g",
                "pct_window_as_extreme", "turning_fraction", "diff_acf1"]
        print(f"\n  {'metric':<24}{'true spike (q1/med/q3)':<26}{'false pos':<26}{'normal':<26}{'sep'}")
        for k in keys:
            print(f"  {k:<24}{quantiles(all_a, k):<26}{quantiles(all_b, k):<26}"
                  f"{quantiles(all_c, k):<26}{separation(all_a, all_b, k):.2f}")

        # Concrete rules: "spare this point" tests, scored as what they cost and buy.
        print(f"\n  {'rule (spare the point when...)':<44}{'FPs spared':>12}{'spikes lost':>13}")
        rules = [
            ("step_sigmas_local < 3", lambda r: r["step_sigmas_local"] < 3),
            ("step_sigmas_local < 5", lambda r: r["step_sigmas_local"] < 5),
            ("step_sigmas_local < 8", lambda r: r["step_sigmas_local"] < 8),
            ("noise_ratio > 3", lambda r: r["noise_ratio"] > 3),
            ("noise_ratio > 5", lambda r: r["noise_ratio"] > 5),
            ("noise_ratio > 3 and step_sigmas_local < 5",
             lambda r: r["noise_ratio"] > 3 and r["step_sigmas_local"] < 5),
            ("turning_fraction < 0.4", lambda r: r["turning_fraction"] < 0.4),
        ]
        for name, rule in rules:
            def hits(rows):
                ok = [r for r in rows if all(np.isfinite(v) for v in r.values() if isinstance(v, float))]
                return 100 * np.mean([rule(r) for r in ok]) if ok else float("nan")
            print(f"  {name:<44}{hits(all_b):>11.1f}%{hits(all_a):>12.1f}%")


if __name__ == "__main__":
    main()
