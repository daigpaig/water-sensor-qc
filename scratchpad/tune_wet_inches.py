"""Does rainfall discriminate a real spike from real water? At what threshold? (§7.7, §13)

The decision the agent actually faces is: this point LOOKS like a spike -- is it an
artifact, or water the rain washed in? So the population is spike CANDIDATES, split by
ground truth, not all rows. If rain informs the call, candidates that are really water
should have had more rain than candidates that are really injected spikes.
"""
import glob
import warnings

import numpy as np
import pandas as pd
import saqc

from src.agent_tools import precipitation as P
from src.agent_tools import context as C

warnings.filterwarnings("ignore")
GRID = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40)

rows = []
for path in sorted(glob.glob("data/injected/*/l*/*_l?.csv")):
    stem = path.rsplit("/", 1)[-1][:-4]
    gauge = stem.split("_")[0]
    v = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")["value"]
    lab = pd.read_csv(path.replace(".csv", "_labels.csv"),
                      parse_dates=["datetime"]).set_index("datetime")
    truth = set(lab.index[lab.anomaly_type == "spike"])
    qc = saqc.SaQC(pd.DataFrame({"value": v}))
    scale = C.spike_scale(qc, field="value")
    th = scale.get("recommended_lof_thresh")
    if th is None:
        continue
    cand = list(v.index[(qc.flagUniLOF("value", n=C.SPIKE_LOF_N, thresh=th)
                         .flags["value"] > 0).to_numpy()])
    res = P.precip_context_points(qc, [t.strftime("%Y-%m-%dT%H:%M:%S") for t in cand],
                                  gauge=gauge, max_points=400)
    for t, p in zip(cand, res["points"]):
        tot = p.get("total_before_12h_in")
        if tot is None:
            continue
        rows.append({"dataset": stem, "in12h": float(tot), "is_spike": t in truth})
    print(f"  done {stem}", flush=True)

df = pd.DataFrame(rows)
print(f"\nspike candidates with rain coverage: {len(df)} "
      f"({int(df.is_spike.sum())} really injected spikes, "
      f"{int((~df.is_spike).sum())} really water)\n")
print(f"{'WET_INCHES':>11}{'P(rain|spike)':>15}{'P(rain|water)':>15}{'separation':>12}"
      f"{'  what rained=true would imply'}")
for w in GRID:
    a = float((df.loc[df.is_spike, "in12h"] > w).mean())
    b = float((df.loc[~df.is_spike, "in12h"] > w).mean())
    # P(really water | rained) -- the number the agent's "keep it" decision rests on
    n_r = ((df.in12h > w)).sum()
    pw = float((~df.loc[df.in12h > w, "is_spike"]).mean()) if n_r else float("nan")
    print(f"{w:>11.2f}{a:>15.1%}{b:>15.1%}{b-a:>12.1%}"
          f"   P(water|rained)={pw:.1%}  n={n_r}")
base = float((~df.is_spike).mean())
print(f"\nbase rate P(really water) with no rain information: {base:.1%}")
