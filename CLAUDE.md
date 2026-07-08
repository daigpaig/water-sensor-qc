# CLAUDE.md — Agentic Water-Quality Data Quality Control Tool

This file is the durable brief for this project. Read it fully at the start of every
session and follow it. If a decision here needs to change, update this file in the same
commit so it never drifts from the code.

---

## 1. What we are building (in one paragraph)

A **standalone agentic data quality control (QC) tool** for continuous water-quality
sensor time series. Physical sensors drift, foul, stick, spike, and drop out. The tool
uses an LLM (Anthropic Claude) as a reasoning engine: given an uploaded time series it
**inspects** the data, **adaptively selects and runs** QC operations from the SaQC
library, **decides** what to do with each problem segment (delete / correct / keep),
**imputes** gaps where appropriate, and returns a **cleaned dataset plus a plain-language
report and a machine-readable flag log** explaining what it did and why. This is a
research prototype. **Quality control is more than anomaly detection** — detection is one
stage; deciding, correcting, and documenting matter equally.

This project is **independent** of the lab's separate water-quality detection-model
effort. It is not a baseline for that model, a data provider to it, or a layer it plugs
into. Do not add any coupling to it.

---

## 2. Golden rules (hard constraints — do not violate)

- **SaQC is pinned to `2.8`.** All QC operations go through SaQC unless SaQC has no
  equivalent. SaQC's method names and signatures differ across versions — **always verify
  each method against the SaQC 2.8 API before using it**; do not trust older examples.
- **Model: `claude-sonnet-4-6`** via the Anthropic Messages API, for all agent reasoning
  and tool-call decisions.
- **Never hard-code the API key.** Load `ANTHROPIC_API_KEY` from `.env` via
  `python-dotenv`. `.env` must be in `.gitignore` _before the first commit_.
- **One agent, one reasoning loop.** No multi-agent systems.
- **Max 25 tool calls per session**, enforced in code (hard cap, not a suggestion).
- **One variable per run.** Input: a CSV with a datetime column + one or more numeric
  columns (process one numeric column per run). Output: a cleaned CSV + a JSON flag log.
- **Python 3.11+**, backend + Streamlit UI only. No other languages.
- **Log every Anthropic API call** to `logs/*.jsonl` (one JSON object per line):
  timestamp, model, input messages, output, stop reason, token counts.
- **Write a test alongside every tool wrapper.** Do not advance a phase with failing tests.
- **Prefer simple, readable, tested code over cleverness.** This is a prototype; do not
  add premature abstraction, plugin systems, or config frameworks.

### Explicitly OUT OF SCOPE — do not build

Spatial / multi-sensor relationship modelling; real-time streaming; forecasting;
multi-agent systems; training any ML model; production hardening (authentication, cloud
deployment, Docker, CI beyond a basic test run).

---

## 3. Tech stack

`saqc==2.8`, `anthropic`, `pandas`, `numpy`, `scikit-learn`, `dataretrieval` (USGS NWIS),
`plotly`, `streamlit`, `python-dotenv`, `pytest`.

---

## 4. Repo structure

```
.
├── CLAUDE.md              # this file
├── README.md
├── requirements.txt
├── .gitignore            # must include .env, .venv/, data/raw/, logs/
├── .env.example          # ANTHROPIC_API_KEY=
├── data/
│   ├── raw/              # downloaded USGS/ECCC (gitignored)
│   ├── clean/            # inspected clean segments used for injection
│   └── injected/         # synthetic datasets + label files
├── src/
│   ├── inspect_data.py   # load + summarise a series
│   ├── inject.py         # synthetic anomaly injection (5 types, 3 levels, seeded)
│   ├── evaluate.py       # metrics, fixed-pipeline baseline, ablation
│   ├── agent.py          # ReAct loop + API logger
│   └── tools/
│       ├── schemas.py    # JSON tool schemas for the Messages API
│       └── wrappers.py   # SaQC-wrapping tool functions
├── app/
│   └── streamlit_app.py  # UI
├── tests/
│   └── test_tools.py
└── logs/                 # JSONL API logs (gitignored)
```

---

## 5. Data contracts (single source of truth — keep code consistent with this)

**Injected dataset CSV** (`data/injected/<name>.csv`)

- `datetime` (ISO 8601), `value` (float; may be NaN for gaps).

**Ground-truth labels** (`data/injected/<name>_labels.csv`, row-aligned by `datetime`)

- `datetime`, `is_anomaly` (bool), `anomaly_type` (one of: `spike`, `plateau`,
  `level_shift`, `gap`, `drift`, or empty), `true_value` (the original clean value).

**Tool result dict** (returned by every tool wrapper; JSON-serialisable)

```json
{
  "tool": "flag_spike_unilof",
  "params": { "...": "..." },
  "n_flagged": 47,
  "pct_flagged": 0.8,
  "flagged_datetimes": ["2024-06-01T03:00:00", "..."],
  "message": "Flagged 47 values (0.8%) as spikes."
}
```

**Cleaned output** (`*_clean.csv`): `datetime`, `value` (corrected/imputed; deleted values
as NaN or removed per config), plus a `flag` column naming the action, if any.

**Flag log** (`*_flags.json`): a list of
`{ "datetime": "...", "flagged_by": "<tool>", "action": "delete|correct|keep|impute", "reason": "<short text>" }`.

---

## 6. The five failure types and their default action

| Type               | Signature in the data                      | Default action                                     |
| ------------------ | ------------------------------------------ | -------------------------------------------------- |
| Spike              | one/few values far from neighbours         | delete                                             |
| Plateau / stuck    | identical value repeated for a long window | delete or flag                                     |
| Level shift / jump | permanent step to a new level              | flag; keep unless clearly erroneous (agent judges) |
| Gap (missing)      | NaN run                                    | impute (short gaps only)                           |
| Drift              | slow creep from truth over weeks           | correct                                            |

The agent reasons about each flagged segment and picks the action; these are defaults, not
hard rules. Genuine extreme events must be **kept**, not "corrected" away.

---

## 7. QC tool inventory

Each is a Python function in `src/tools/wrappers.py` wrapping a SaQC 2.8 method, returning
the tool-result dict from §5. `inspect_dataset` must be callable first. **Verify every
SaQC method name/signature against the 2.8 docs — the names below are from the proposal and
may need adjustment.**

Utility: `inspect_dataset` (summary: rows, time range, inferred frequency, NaN count/%,
per-column min/max/mean/std), `get_flag_summary` (counts by tool), `export_clean_data`.

Detection: `flag_range` (flagRange), `flag_constants` (flagConstants),
`flag_spike_unilof` (flagUniLOF), `flag_zscore` (flagZScore), `flag_jumps` (flagJumps),
`flag_nan` (flag missing values — confirm the 2.8 method name).

Action: `impute_rolling` (rolling-window imputation — confirm 2.8 name),
`correct_drift` (correctDrift; needs maintenance dates).

---

## 8. The agent loop (ReAct)

1. **Inspect** — always call `inspect_dataset` first.
2. **Reason** — read the summary; decide which checks to run, in what order, with what params.
3. **Act** — call tools one at a time; use each result to decide the next call. Respect the
   25-call cap; if reached, stop and summarise.
4. **Summarise** — call `get_flag_summary` and `export_clean_data`.
5. **Report** — write a plain-language report: what was found (by type), what was done, and
   any caveats.

The system prompt (in `src/agent.py`, versioned in git — commit changes with a note) states
the agent's role, the golden rules, the tool list, and that it must justify each action.
Use the Anthropic Messages API multi-turn tool-use pattern (assistant emits tool_use →
we run the tool → we return tool_result → loop).

---

## 9. Data strategy

- **Real data:** USGS NWIS via the `dataretrieval` package (primary — cleanly scriptable);
  ECCC (secondary — may be a manual download). Target ~2 years, 15-min/hourly, 2–3 gauges
  per variable. Start with **one variable** (turbidity or specific conductance).
- **Clean segments:** visually inspect before use — real data may already contain anomalies.
- **Synthetic injection:** inject all five types at recorded locations into clean segments;
  save the labels (§5). Build **three contamination levels (~3%, 7%, 12%)** with a **fixed
  random seed** for reproducibility.

---

## 10. Evaluation

- **Detection:** precision / recall / F1 per anomaly type + macro-F1 (scikit-learn), vs the
  injected labels.
- **Imputation:** RMSE / MAE on filled values vs true values, compared to a
  linear-interpolation baseline.
- **Decision quality:** for each flagged segment, does the chosen action match the known
  correct action?
- **Fixed-pipeline baseline:** the same SaQC methods in a set order with default params and
  no agent reasoning. The agent should beat it; if not, that is a finding to explain.
- **Ablation:** disable tool subsets to show which matter.
- **Splitting:** first 80% of each series as context, last 20% as held-out test. Never shuffle.
- Also report on a small **manually labelled real** segment and discuss synthetic-vs-real gap.

---

## 11. Definition of done (success criteria)

- Macro-F1 ≥ 0.70 across the five types on held-out test data.
- Imputation RMSE beats linear interpolation on ≥ 2 of 3 datasets.
- Decision action reported and compared to the correct action for every flagged segment.
- A non-programmer can upload → run → view flags → download in < 5 minutes.
- Every API call logged; a run is reproducible from the log.
- Tests cover ≥ 80% of tool wrappers and pass.

---

## 12. Build order — work phase by phase, STOP at each gate for review

Do not jump ahead. After each phase, run the checks, commit, and summarise what was built
before continuing.

- **Phase 0 — Scaffold.** Create the repo tree (§4), `requirements.txt`, `.gitignore` (with
  `.env`), `.env.example`, `README.md`. **Gate:** `pip install -r requirements.txt` succeeds;
  `pytest` runs (zero tests OK). Do not proceed until confirmed.
- **Phase 1 — Data + injection.** `inspect_data.py`; pull USGS data via `dataretrieval`;
  `inject.py` implementing all five anomaly types, labels, seed, and the three levels.
  **Gate:** three labelled datasets exist; a test confirms injected anomalies are recoverable
  from the label file.
- **Phase 2 — Tools.** `schemas.py` (with parameter ranges) + `wrappers.py` for every tool,
  each returning the result dict; a unit test per wrapper; a smoke test that the required
  SaQC 2.8 methods exist. **Gate:** all wrappers tested and passing.
- **Phase 3 — Agent (CLI).** `agent.py` ReAct loop, Messages API tool-use, dispatch, 25-call
  cap, JSONL logger, versioned system prompt. **Gate:** an end-to-end CLI run on one labelled
  dataset produces a cleaned CSV, flag log, and report; the JSONL log looks sensible.
- **Phase 4 — Evaluation.** `evaluate.py` metrics + fixed-pipeline baseline + ablation; run on
  held-out segments. **Gate:** results, baseline, and ablation tables generated.
- **Phase 5 — UI.** `streamlit_app.py` with upload / preview / run / Plotly plot / report /
  download panels, wired to the agent. **Gate:** the non-coder flow works locally.
- **Phase 6 — Polish.** Docstrings, README run instructions, cleanup. (Paper is written
  outside the repo.)

---

## 13. Coding conventions

- Small, typed, documented functions; structured returns (dataclasses or TypedDicts).
- Deterministic where possible (seed everything random).
- Validate tool parameters and fail loudly on nonsensical input.
- If unsure how a SaQC 2.8 method behaves, write a tiny probe script and check — do not guess.
- Keep the diff per phase reviewable; commit at every gate.
