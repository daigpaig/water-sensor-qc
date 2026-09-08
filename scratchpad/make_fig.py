"""Before/after evidence for the spike-magnitude change in ``src/datasets/inject.py``.

One-off figure, not part of any pipeline. It exists to justify one decision:
sampling a spike's magnitude **log-uniformly over 2-15x** the local robust scale
instead of **uniformly over 4-10x** (``SPIKE_MAGNITUDE_SCALE``, src/datasets/inject.py).
A flat U(4, 10) made every spike a similar, uniformly-large size — an
unrealistically easy, narrow target for the detectors. Log-uniform over a wider
band puts most spikes just above the local noise with a long tail of big ones.

Writes ``scratchpad/spike_comparison.png``, a 3-panel figure for one gauge:

- **left column** — the same 10-day window at four contamination levels: the
  uninjected base (reconstructed from the labels' ``true_value``), then L1/L2/L3
  with every labelled spike row marked in red. The window is chosen automatically
  as the densest 10 days of spikes in L3, so the comparison lands where there is
  something to see. Read it for *realism*: do the injected spikes look like a
  plausible sensor record next to the clean base above them?
- **top right** — the OLD vs NEW magnitude distributions. Read it for *spread*:
  the old draw spans 2.5x min-to-max, the new one 7.5x.
- **bottom right** — the *realized* spike magnitudes ``|value - true_value|`` per
  level, log x-axis. This is what actually landed in the datasets, as opposed to
  the distribution it was drawn from.

Caveat: the top-right panel samples both distributions from a seeded RNG here
rather than reading them from the injector, so it documents the change as of
writing and will not follow later edits to ``SPIKE_MAGNITUDE_SCALE``. The other
two panels read the real datasets and do follow. Level labels and target rates
come from each dataset's manifest for the same reason — hardcoding them left the
figure claiming 3/7/12% long after the rates were retuned to 0.8/2/3.5 (§9).

Run from the repo root, after ``python -m src.datasets.inject``:

    python scratchpad/make_fig.py
"""
import json
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SITE = "03447687"  # French Broad R nr Fletcher, NC; moderate, median ~8 FNU
def _dir(lvl):
    return f"data/turbidity/injected/{SITE}/l{lvl}"

def load(lvl):
    d = pd.read_csv(f"{_dir(lvl)}/{SITE}_l{lvl}.csv", parse_dates=["datetime"])
    lab = pd.read_csv(f"{_dir(lvl)}/{SITE}_l{lvl}_labels.csv", parse_dates=["datetime"])
    return d, lab

def level_label(lvl):
    """``L2 (medium, 2.0%)`` — from the manifest, so it cannot go stale."""
    m = json.loads(open(f"{_dir(lvl)}/{SITE}_l{lvl}_manifest.json").read())
    return f"L{lvl} ({m['level_name']}, {m['target_point_pct']}%)"

d3, lab3 = load(3)
spk3 = lab3["anomaly_type"].eq("spike").to_numpy()
win = 96 * 10  # 10 days at 15-min
counts = pd.Series(spk3.astype(int)).rolling(win).sum()
end = int(counts.idxmax()); start = end - win
t0, t1 = d3["datetime"].iloc[start], d3["datetime"].iloc[end]

fig = plt.figure(figsize=(13, 11))
gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.25], hspace=0.5, wspace=0.22)

titles = ["Uninjected (true clean base)"] + [level_label(l) for l in (1, 2, 3)]
truth = lab3["true_value"].to_numpy()
ts_axes = []
for i, lvl in enumerate([None, 1, 2, 3]):
    ax = fig.add_subplot(gs[i, 0]); ts_axes.append(ax)
    if lvl is None:
        d = d3; y = truth; sp = np.zeros(len(y), bool)
    else:
        d, lab = load(lvl); y = d["value"].to_numpy(); sp = lab["anomaly_type"].eq("spike").to_numpy()
    tt = d["datetime"]; m = (tt >= t0) & (tt <= t1)
    ax.plot(tt[m], y[m], lw=0.6, color="#3b6ea5")
    if lvl is not None:
        smask = (m & sp).to_numpy()
        ax.scatter(tt[smask], y[smask], s=16, color="#d1495b", zorder=5)
    ax.set_title(titles[i], fontsize=10, loc="left")
    ax.set_ylabel("FNU", fontsize=8)
    ax.set_xlim(t0, t1)
    ax.tick_params(labelsize=7)
    if lvl is None:
        ax.set_ylim(bottom=0)

# old vs new scale-factor distribution
rng = np.random.default_rng(0); N = 200_000
old = rng.uniform(4, 10, N); new = np.exp(rng.uniform(np.log(2), np.log(15), N))
axd = fig.add_subplot(gs[0:2, 1])
bins = np.linspace(0, 16, 50)
axd.hist(old, bins=bins, alpha=0.55, color="#8d99ae", label="OLD  U(4, 10)", density=True)
axd.hist(new, bins=bins, alpha=0.65, color="#e07a5f", label="NEW  logU(2, 15)", density=True)
axd.axvline(np.median(old), color="#4a4e69", ls="--", lw=1)
axd.axvline(np.median(new), color="#bc4b2a", ls="--", lw=1)
axd.set_title("Spike magnitude scale factor  (x local robust scale)", fontsize=10, loc="left")
axd.set_xlabel("scale factor"); axd.set_ylabel("density")
axd.legend(fontsize=8); axd.tick_params(labelsize=7)
axd.text(0.98, 0.55, "max/min\nold 2.5x\nnew 7.5x", transform=axd.transAxes,
         ha="right", va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="#ccc"))

# realized magnitudes per level (log x)
axr = fig.add_subplot(gs[2:4, 1])
colors = {1: "#8ecae6", 2: "#219ebc", 3: "#023047"}
for lvl in [1, 2, 3]:
    d, lab = load(lvl)
    m = lab["anomaly_type"].eq("spike").to_numpy()
    mag = np.abs(d["value"].to_numpy()[m] - lab["true_value"].to_numpy()[m])
    mag = mag[np.isfinite(mag) & (mag > 0)]
    axr.hist(mag, bins=np.logspace(np.log10(0.5), np.log10(300), 40),
             alpha=0.55, color=colors[lvl], label=f"L{lvl} (n={mag.size})")
axr.set_xscale("log")
axr.set_title(f"Realized spike magnitude |value - true|  ({SITE})", fontsize=10, loc="left")
axr.set_xlabel("FNU (log scale)"); axr.set_ylabel("count")
axr.legend(fontsize=8); axr.tick_params(labelsize=7)

fig.suptitle(f"Spike magnitude variation after the change - {SITE}   "
             f"(window {t0:%Y-%m-%d} -> {t1:%Y-%m-%d})", fontsize=13, y=0.995)
os.makedirs("scratchpad", exist_ok=True)
out = "scratchpad/spike_comparison.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
