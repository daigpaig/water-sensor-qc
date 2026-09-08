"""Do earlier detectors starve later ones? (§7.1's additive-flagging trap, measured)

SaQC masks rows whose flag is >= dfilter before a function runs, and no detector wrapper
passes dfilter. So every detector after the first examines a MASKED series, and the set of
detectors a run happens to call earlier silently changes what the later ones can find.
Run Q surfaced this: find_shift_windows read 76 jumps from the history where flag_jumps
alone finds 144, and the injected shift's own edges were among the missing.

MUST be run from a file, never a heredoc: flagPlateau uses multiprocessing and its workers
re-import __main__ (§7.1).
"""
import warnings

import numpy as np
import pandas as pd
import saqc

warnings.filterwarnings("ignore")

v = pd.read_csv("data/turbidity/injected/01467200/l1/01467200_l1.csv",
                parse_dates=["datetime"]).set_index("datetime")["value"]
lab = pd.read_csv("data/turbidity/injected/01467200/l1/01467200_l1_labels.csv",
                  parse_dates=["datetime"]).set_index("datetime")
shift = lab.index[lab.anomaly_type == "level_shift"]
mad = 1.4826 * float(v.diff().abs().median())


def jump_hits(qc, **kw):
    out = qc.flagJumps("value", thresh=5.995, window="6h", **kw)
    hist = out._flags.history["value"]
    col = [c for c in hist.hist.columns
           if "jump" in str(hist.meta[c].get("func", "")).lower()][-1]
    hits = set(hist.hist.index[hist.hist[col] > 0])
    edges = sum(1 for t in hits
                if min(abs((t - shift.min()).total_seconds()),
                       abs((t - shift.max()).total_seconds())) <= 900)
    return len(hits), edges


base = saqc.SaQC(pd.DataFrame({"value": v}))
n, e = jump_hits(base)
print(f"{'flag_jumps ALONE':<46}{n:>6} jumps, {e} real-shift edges")

q = base.flagRange("value", min=0, max=1000)
q = q.flagConstants("value", thresh=0.01, window="1h", min_periods=2)
q = q.flagUniLOF("value", n=10, thresh=2.32)
q = q.flagZScore("value", method="modified", window="6h", thresh=10.41,
                 min_residuals=3 * mad)
if False:  # flagPlateau is slow here and not needed for the point being measured
    pass

n, e = jump_hits(q)
print(f"{'after run Q earlier detectors (current)':<46}{n:>6} jumps, {e} real-shift edges")
n, e = jump_hits(q, dfilter=np.inf)
print(f"{'same, with dfilter=inf':<46}{n:>6} jumps, {e} real-shift edges")
