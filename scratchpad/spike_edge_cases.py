"""Pull the spike decisions that went wrong out of one run, with the evidence.

Two questions: which injected spikes did the agent KEEP (false negatives), and which
genuine water did it DELETE (false positives)? For each, report what the agent
actually measured (the describe_point/describe_points row it saw, if any) and the
reason it recorded — so a wrong call can be traced to a wrong measurement, a missing
measurement, or a wrong inference from a right measurement.

    python -m scratchpad.spike_edge_cases <log.jsonl> <flags.json> <series.csv>
"""
import json
import sys
from pathlib import Path

import pandas as pd

log_path, flags_path, series_path = (Path(a) for a in sys.argv[1:4])

# ---- the agent's decisions, and every point it measured -------------------------
decisions, measured = [], {}
for line in log_path.read_text().splitlines():
    ev = json.loads(line)
    if ev.get("event") == "api_response":
        for b in ev["response"]["content"]:
            if b.get("type") == "tool_use" and b["name"] == "export_clean_data":
                decisions = b["input"].get("decisions", []) or decisions
    if ev.get("event") != "api_call":
        continue
    for msg in ev.get("messages") or []:
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                continue
            try:
                r = json.loads(block["content"])
            except (json.JSONDecodeError, TypeError):
                continue
            if r.get("tool") == "describe_points":
                for row in r.get("points", []):
                    measured.setdefault(pd.Timestamp(row["at"]), row)
            elif r.get("tool") == "describe_point":
                measured.setdefault(pd.Timestamp(r["at"]), r)

# ---- labels, series, and the flag log --------------------------------------------
series = pd.read_csv(series_path, parse_dates=["datetime"]).set_index("datetime")["value"]
lbl = pd.read_csv(series_path.with_name(series_path.stem + "_labels.csv"),
                  parse_dates=["datetime"]).set_index("datetime")
flags = json.loads(flags_path.read_text())
spike_flagged = pd.DatetimeIndex(
    [pd.Timestamp(e["datetime"]) for e in flags
     if any(t in e["flagged_by"] for t in ("flagUniLOF", "flagZScore", "flagRange"))]
)

action = pd.Series("(undecided)", index=spike_flagged, dtype=object)
reason = pd.Series("", index=spike_flagged, dtype=object)
for d in decisions:
    s = pd.Timestamp(d["start"]); e = pd.Timestamp(d.get("end") or d["start"])
    m = (spike_flagged >= s) & (spike_flagged <= e) & (action == "(undecided)").to_numpy()
    action[m] = d["action"]
    reason[m] = d.get("reason", "")

truth = lbl["anomaly_type"].reindex(spike_flagged).fillna("real water")


def _fmt(ts):
    row = measured.get(ts)
    if row is None:
        return "        NOT MEASURED — no describe_point(s) call covered this timestamp"
    exc = row.get("excursion") or {}
    rec = row.get("recovery") or {}
    nb = row.get("neighbourhood") or {}
    if "n_samples" not in exc:                      # compact describe_points row
        return (f"        reads_like={row.get('reads_like')}  "
                + "  ".join(f"{k}={row[k]}" for k in
                            ("robust_z", "width_samples", "recovery_samples", "step_sharpness")
                            if k in row))
    return (f"        reads_like={row.get('reads_like')}  robust_z={nb.get('robust_z')}  "
            f"width={exc.get('n_samples')}  recovered_in={rec.get('samples_to_recover')}")


for title, mask in (
    (f"INJECTED SPIKES THE AGENT KEPT (false negatives)", (truth == "spike") & (action == "keep")),
    (f"GENUINE WATER THE AGENT DELETED (false positives)", (truth == "real water") & (action == "delete")),
):
    stamps = spike_flagged[mask.to_numpy()]
    print(f"\n{'=' * 78}\n{title} — {len(stamps)}\n{'=' * 78}")
    for ts in stamps:
        val = series.get(ts, float("nan"))
        true_val = lbl["true_value"].get(ts, "")
        extra = f"  (true value {true_val})" if str(true_val) not in ("", "nan") else ""
        print(f"\n  {ts}   value={val:.2f}{extra}")
        print(_fmt(ts))
        if reason[ts]:
            print(f"        reason: {reason[ts][:150]}")
