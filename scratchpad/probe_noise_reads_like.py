"""Hit rates for the `reads_like` noise guard (CLAUDE.md §7.3).

The guard turns a `spike` label into `noisy-stretch` when noise_context says the
point does not stand out from its own surroundings. This prints what that costs
on true injected spikes and what it buys on the false positives that motivated
it, so the §7.3 table is a measurement rather than a claim.

    .venv/bin/python -m scratchpad.probe_noise_reads_like
"""
import numpy as np, pandas as pd, saqc
from src.inspect_data import load_series
from src.agent_tools import context as ctx

DATASETS = [("03447687","l1"),("03447687","l2"),("02198840","l2"),("08041770","l2")]
LIMIT = 250

def run(gauge, level):
    root = f"data/turbidity/injected/{gauge}/{level}/{gauge}_{level}"
    s = load_series(f"{root}.csv").set_index("datetime")["value"].astype(float).sort_index()
    lab = pd.read_csv(f"{root}_labels.csv", parse_dates=["datetime"]).set_index("datetime")
    qc = saqc.SaQC(pd.DataFrame({"value": s})).flagUniLOF("value", n=20, thresh=1.5)
    f = qc.flags["value"]; flagged = set(f[f > 0].index)
    anom = lab["is_anomaly"].fillna(False).astype(bool)
    spike_ts = set(lab.index[anom & (lab["anomaly_type"] == "spike")])
    normal_ts = set(lab.index[~anom])
    rng = np.random.default_rng(0)
    def pick(st):
        st = [t for t in sorted(st) if t in s.index and np.isfinite(s.loc[t])]
        if len(st) > LIMIT:
            st = [st[i] for i in sorted(rng.choice(len(st), LIMIT, replace=False))]
        return st
    return s, {
        "true spike (detected)": pick(spike_ts & flagged),
        "FALSE POSITIVE":        pick(flagged & normal_ts),
        "normal (unflagged)":    pick(normal_ts - flagged),
    }

tally = {}
for gauge, level in DATASETS:
    s, pops = run(gauge, level)
    for name, stamps in pops.items():
        d = tally.setdefault(name, {"n": 0, "spike_before": 0, "noisy": 0})
        for t in stamps:
            r = ctx.describe_point(s, t)
            d["n"] += 1
            if r["reads_like"] in ("spike", "noisy-stretch"):
                d["spike_before"] += 1
            if r["reads_like"] == "noisy-stretch":
                d["noisy"] += 1

print(f"\n{'population':<24}{'n':>6}{'read spike BEFORE':>20}{'still spike AFTER':>20}{'-> noisy-stretch':>19}")
for name, d in tally.items():
    before, noisy, n = d["spike_before"], d["noisy"], d["n"]
    print(f"{name:<24}{n:>6}{before:>13} ({100*before/n:4.1f}%){before-noisy:>13} "
          f"({100*(before-noisy)/n:4.1f}%){noisy:>12} ({100*noisy/max(before,1):4.1f}% of them)")
