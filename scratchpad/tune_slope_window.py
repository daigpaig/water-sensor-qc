"""Does slope_context's fall_rise_ratio separate flush events from artifact spikes
at ANY window, and which one?

§7.3 recorded the ratio as useless, measured at 1h/2h/6h/24h on storm peaks vs
injected spikes. This re-asks the question on the population that actually matters:
the rows a run DELETED that the labels call normal water (which the shape says are
small first-flush events) against the injected spikes it caught.

Separation is reported as AUC — the probability that a randomly chosen flush event
has a higher fall/rise ratio than a randomly chosen artifact. 0.5 is a coin flip;
>=0.7 is a usable signal. The decay-run length is included as the benchmark to beat.
"""
import json

import numpy as np
import pandas as pd

from src.agent_tools import context

RAW = "data/raw/approved/03447687_turbidity_63680.csv"
SERIES = "data/injected/03447687/l1/03447687_l1.csv"
FLAGS = "data/agent_runs/03447687_l1_flags.json"
SPIKE_FUNCS = ("flagUniLOF", "flagZScore", "flagRange")


def auc(pos: pd.Series, neg: pd.Series) -> float:
    """P(pos > neg), ties counted as half — Mann-Whitney U normalised."""
    pos, neg = pos.dropna(), neg.dropna()
    if pos.empty or neg.empty:
        return float("nan")
    grid = pos.to_numpy()[:, None] - neg.to_numpy()[None, :]
    return float(((grid > 0).sum() + 0.5 * (grid == 0).sum()) / grid.size)


def decay_run(series: pd.Series, ts: pd.Timestamp, n: int = 6) -> int | None:
    i = series.index.get_loc(ts)
    after = series.iloc[i: i + n + 1].to_numpy()
    if len(after) < 3 or not np.isfinite(after).all():
        return None
    run = 0
    for a, b in zip(after, after[1:]):
        if b < a:
            run += 1
        else:
            break
    return run


def main() -> None:
    raw = pd.read_csv(RAW, parse_dates=["datetime"]).set_index("datetime")["value"]
    series = pd.read_csv(SERIES, parse_dates=["datetime"]).set_index("datetime")["value"]
    labels = pd.read_csv(SERIES.replace(".csv", "_labels.csv"),
                         parse_dates=["datetime"]).set_index("datetime")
    flags = json.loads(open(FLAGS).read())

    flush, artifact = [], []
    for entry in flags:
        ts = pd.Timestamp(entry["datetime"])
        if not any(f in entry["flagged_by"] for f in SPIKE_FUNCS):
            continue
        if entry["action"] not in ("delete", "correct"):
            continue
        (artifact if labels["anomaly_type"].get(ts) == "spike" else flush).append(ts)

    print(f"flush events (deleted-natural, measured on the raw base): {len(flush)}")
    print(f"artifact spikes (injected, caught):                       {len(artifact)}\n")
    print(f"{'window':>18} {'flush med':>10} {'artifact med':>13} {'AUC':>7}")
    print("-" * 52)

    for n in (2, 3, 4, 5, 6, 8, 12, 24):
        f = pd.Series([context.slope_context(raw, t, n_before=n, n_after=n)
                       .get("fall_rise_ratio") for t in flush if t in raw.index], dtype=float)
        a = pd.Series([context.slope_context(series, t, n_before=n, n_after=n)
                       .get("fall_rise_ratio") for t in artifact if t in series.index], dtype=float)
        label = f"{n} samples ({n * 15}min)"
        print(f"{label:>18} {f.median():>10.2f} {a.median():>13.2f} {auc(f, a):>7.2f}")

    fd = pd.Series([decay_run(raw, t) for t in flush if t in raw.index], dtype=float)
    ad = pd.Series([decay_run(series, t) for t in artifact if t in series.index], dtype=float)
    print(f"\n{'decay-run length':>18} {fd.median():>10.2f} {ad.median():>13.2f} {auc(fd, ad):>7.2f}"
          "   <- benchmark to beat")


if __name__ == "__main__":
    main()
