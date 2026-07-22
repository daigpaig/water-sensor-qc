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

- **SaQC is pinned to `2.8`, which resolves to `2.8.0`.** The current PyPI release is
  `2.9.1` — **do not upgrade**, and do not trust 2.9 examples, docs, or answers. All QC
  operations go through SaQC unless SaQC has no equivalent. SaQC's method names and
  signatures differ across versions — **always verify each method against the installed
  2.8.0 API before using it** (`scratchpad/probe_saqc.py` prints every signature).
  Probed environment: saqc 2.8.0, pandas 3.0.3, numpy 2.2.6, Python 3.13.
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
├── scratchpad/
│   ├── probe_saqc.py          # signature + toy-call probe for every §7 method
│   └── probe_saqc_behavior.py # reproduces the §7.1 constraints
└── logs/                 # JSONL API logs (gitignored)
```

---

## 5. Data contracts (single source of truth — keep code consistent with this)

**Injected dataset CSV** (`data/injected/<name>.csv`)

- `datetime` (ISO 8601), `value` (float; may be NaN for gaps).

**Ground-truth labels** (`data/injected/<name>_labels.csv`, row-aligned by `datetime`)

- `datetime`, `is_anomaly` (bool), `anomaly_type` (one of: `spike`, `plateau`,
  `level_shift`, `gap`, `drift`, or empty), `true_value` (the original clean value),
  `source` (`natural` | `injected`, or empty on non-anomalous rows).
- **Every** missing run is labelled `is_anomaly=True, anomaly_type=gap` — including
  gaps already present in the "clean" base — so the ground truth is honest rather than
  pretending the base is pristine. `source` separates the two: only `injected` gaps have
  a known `true_value`, so **imputation RMSE/MAE (§10) is scored on injected gaps only**,
  while both count for detection precision/recall.

**Maintenance schedule** (`data/injected/<name>_maintenance.csv`)

- `start`, `end` — one row per maintenance visit. Drift accrues between visits and resets
  at each one, so this is the ground truth for the drift episodes _and_ the support-point
  input SaQC 2.8's `correctDrift` requires (its `maintenance_field` reads the index as an
  event's start and the value as its end).
- **A file with fewer than 3 visits is unusable for correction** — `correctDrift` corrects
  the first `N-2` of `N-1` inter-visit intervals and NaNs out the last (§7.1). Injection
  must emit at least 3 visits per base or record that drift correction is unscoreable for
  that dataset. Watch the §9 interval settings here: 200 days between visits on a ~250-day
  base yields ~2 visits and therefore **zero** correctable drift.

**Injection manifest** (`data/injected/<name>_manifest.json`)

- Per-type event/row counts, the seed, and target-vs-actual point contamination.
  `evaluate.py` must read per-type counts from here or from the labels — never infer them
  from the level number (see §9).

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

| Type               | Signature in the data                      | Detected by                          | Default action                                     |
| ------------------ | ------------------------------------------ | ------------------------------------ | -------------------------------------------------- |
| Spike              | one/few values far from neighbours         | `flagUniLOF`, `flagZScore`, `flagRange` | delete                                          |
| Plateau / stuck    | identical value repeated for a long window | `flagConstants` (+ `flagPlateau` if offset) | delete or flag                             |
| Level shift / jump | permanent step to a new level              | `flagJumps`                          | flag; keep unless clearly erroneous (agent judges) |
| Gap (missing)      | NaN run                                    | `flagNAN`                            | impute (short gaps only)                           |
| Drift              | slow creep from truth over weeks           | **no detector in 2.8** — §7.1        | correct (`correctDrift`, from the schedule)        |

The agent reasons about each flagged segment and picks the action; these are defaults, not
hard rules. Genuine extreme events must be **kept**, not "corrected" away.

Drift is the odd one out: SaQC 2.8 has no univariate drift detector (§7.1), so the agent
cannot *find* drift the way it finds the other four. It applies `correctDrift` using the
maintenance schedule's support points instead.

---

## 7. QC tool inventory

Each is a Python function in `src/tools/wrappers.py` wrapping a SaQC 2.8 method, returning
the tool-result dict from §5. `inspect_dataset` must be callable first.

**All method names below were probed against the installed 2.8.0 and are correct as
written** — `scratchpad/probe_saqc.py` prints each signature and runs each on toy data;
`scratchpad/probe_saqc_behavior.py` reproduces every constraint in §7.1. Re-run both
before changing a wrapper. Signatures below are abridged to the parameters we pass.

Utility: `inspect_dataset` (summary: rows, time range, inferred frequency, NaN count/%,
per-column min/max/mean/std), `get_flag_summary` (counts by tool), `export_clean_data`.

Detection:

| Wrapper             | SaQC 2.8 method  | Key parameters                                             |
| ------------------- | ---------------- | ---------------------------------------------------------- |
| `flag_range`        | `flagRange`      | `min`, `max` (both default `None`)                          |
| `flag_constants`    | `flagConstants`  | `thresh`, `window`, `min_periods=2` — both required          |
| `flag_plateau`      | `flagPlateau`    | `min_length`, `max_length`, `min_jump`, `granularity`        |
| `flag_spike_unilof` | `flagUniLOF`     | `n=20`, `thresh=None`, `density='auto'`, `slope_correct=True`|
| `flag_zscore`       | `flagZScore`     | `method='standard'\|'modified'`, `window`, `thresh=3`        |
| `flag_jumps`        | `flagJumps`      | `thresh`, `window` — both required                           |
| `flag_nan`          | `flagNAN`        | (field only)                                                 |

Action: `impute_rolling` (`interpolateByRolling`; `window` required, `func='median'`,
`min_periods=0`), `correct_drift` (`correctDrift`; `maintenance_field`, `model`,
`cal_range=5` — see §7.1, it is the sharpest edge in the library).

`flag_plateau` is an addition the probe justified: `flagConstants` and `flagPlateau`
detect **different** things and we want both (§7.1).

### 7.1 Probed constraints — these bit us, do not rediscover them

**`flagNAN` is the correct 2.8 name.** `flagMissing` also exists but is **deprecated since
2.7.0**. Do not "modernise" `flagNAN` into `flagMissing`.

**`correctDrift` is dangerous and needs a guard in the wrapper.** Probed behaviour:

- The maintenance variable must be a **standalone variable with its own short index**
  (index = visit start, value = visit end, dtype `datetime64`). Construct the object as
  `saqc.SaQC({"value": series, "maintenance": maint})` — passing a **dict**, so the two
  variables keep independent indexes. NaN-padding the visits onto the data index does not
  work at all: pandas 3 raises `TypeError: Invalid value ... for dtype 'float64'`.
- It corrects only the spans **between** support points — from the end of visit `k` to the
  start of visit `k+1`. Data before the first visit ends and after the last visit starts is
  never touched.
- **`N` visits yield `N-1` intervals, of which only the first `N-2` are corrected**, because
  the implementation calls `.shift(-1)` to get each interval's target level and the last
  interval has no successor. **Fewer than 3 visits corrects nothing.**
- **The final interval is silently overwritten with `NaN` and left `UNFLAGGED` (`-inf`).**
  Verified: a 4000-row input with zero NaNs comes back with 591 NaNs and no flag explaining
  them. This is silent data loss that the §5 flag log would not record. The wrapper **must**
  either append a sentinel trailing visit or restore-and-flag that trailing span, and must
  assert that the output NaN count did not rise unexpectedly.

**There is no univariate drift *detector* in 2.8** — which is why §7 lists none, and it is a
constraint, not an oversight. `flagDriftFromNorm` clusters a **group** of fields and needs
**≥3** to form a "norm" (probed: 1 field → 0 flagged, 2 → 0, 4 → 560). `flagDriftFromReference`
needs a separate reference variable. Both collide with the "one variable per run" golden rule.
`flagJumps` does not catch slow drift either (probed: 0 flagged on a 1000-row ramp). So drift
is located **out of band from the maintenance schedule** (§5), and `correctDrift` is the
action applied to it. See §10 for what this means for scoring.

**`flagConstants` vs `flagPlateau` detect different failures — keep both.**

- `flagConstants` catches a **stuck sensor**: a run of near-identical values, at any level.
  Exact and reliable (probed: rows 799–899 for a 100-row stuck run).
- `flagPlateau` catches an **offset** plateau — a segment displaced from its surroundings,
  whose values need not be constant. It flags **nothing** when the stuck run sits at the
  local level, so it cannot replace `flagConstants`.
- `flagPlateau`'s `min_length` must be set **well below** the true plateau length. Probed on
  a 25 h plateau: `min_length` of `1h`/`3h` hit it exactly; `6h` and `12h` found nothing.
- `flagConstants`' `thresh` must be **much smaller than the signal's noise sd**, or it
  swallows the series: on noise with sd 0.05, `thresh=0.5` flagged 2999 of 3000 rows.

**`interpolateByRolling` will half-fill a gap.** It fills only where the rolling window finds
context, so a window narrower than the gap leaves the gap **partly** filled rather than
skipping it — probed, a 40-sample gap: `window='3h'` filled 11 of 40, `window='12h'` filled
all 40. Choose `window` > the longest gap to fill, and have the wrapper report filled-vs-
remaining per gap instead of a single total. Its default is **`flag=-inf`, so imputed values
are not flagged at all**; pass `flag` explicitly (or track imputation separately) or the §5
flag log will silently omit every `"action": "impute"` row.

**Flag attribution needs the history, not the flag frame.** `qc.flags` is a `DictOfSeries` of
floats (`UNFLAGGED=-inf`, `GOOD=0`, `DOUBTFUL=25`, `BAD=255`). A later test does not overwrite
an existing flag, so diffing successive flag frames **undercounts** — a row both tools flag is
attributed only to the first. For the §5 `flagged_datetimes` contract and `get_flag_summary`,
read `qc._flags.history[field]`, whose `.hist` has one column per applied test and whose
`.meta` carries each test's `func` name.

**Index requirements.** The index **must be monotonic** — unsorted raises
`ValueError: index values must be monotonic`, so sort on load. An **irregular** index is
accepted by every method we use, including offset-string windows. Note that SaQC silently
accepts two things we should reject ourselves in `inspect_dataset`: a non-datetime
`RangeIndex`, and **duplicate timestamps**.

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
  save the labels (§5). Build **three contamination levels** with a **fixed random seed**
  for reproducibility.
- **What the level scales.** The `~3% / 7% / 12%` knob applies to the **point-like types
  only** (spike, plateau, gap), where "percent of rows" is a natural unit. Drift and
  level_shift are driven by episode structure instead, and their row-share is a reported
  consequence rather than a target — so a dataset's **total** anomalous share exceeds its
  headline level (materially: level 3 lands near 50%). Read per-type counts from the
  manifest/labels; never infer them from the level number.
- **Drift is set by maintenance interval, not by episode count.** Drift recurs — it is
  fouling that accrues until the sensor is serviced and then resets — so the level sets
  days-between-visits (200 / 100 / 55) and the episode count follows from the base's
  length. A fixed _count_ would make level 3 mean 37% drift on a 220-day base but 16% on a
  513-day one, leaving the levels non-comparable across datasets.
- **Level shift is injected as a bounded window**, not a literal permanent step: a
  permanent step would either label every subsequent row anomalous (one mid-series shift
  ≈ 50% contamination) or leave post-step rows with `true_value != value` while marked
  not-anomalous, which the §5 label contract cannot express.
- **Isolated dropouts are not gaps.** Re-gridding a raw NWIS file onto its 15-min grid
  makes absent samples explicit, and the resulting NaN share looks alarming (06818000:
  14.3%) — but that record's _longest_ missing run across 253 days is 10 samples (2.5h),
  and 2,216 of its 2,744 missing runs are a single sample. Nothing there is visible on a
  plot, and `longest_unbroken_run_days` is right to pass it. Injection therefore lets
  segment anomalies (plateau, level_shift, drift) span dropouts up to
  `SPANNABLE_DROPOUT_ROWS`, as they do in reality; only spikes and injected gaps require
  a real reading in every row. Treating every isolated dropout as blocking left just
  16.8% of that series usable and starved level 3 of plateaus and level shifts entirely.
  If a base genuinely cannot host its budget, injection reports `point_budget_met: false`
  and `types_missing` rather than silently under-filling.

---

## 10. Evaluation

- **Detection:** precision / recall / F1 per anomaly type + macro-F1 (scikit-learn), vs the
  injected labels. **Drift is excluded from the detection metrics** — SaQC 2.8 ships no
  univariate drift detector (§7.1), so drift episodes are supplied by the maintenance
  schedule rather than discovered. Report macro-F1 over the **four** detectable types
  (spike, plateau, level_shift, gap), say so explicitly next to the number, and score drift
  under correction quality instead: RMSE of the corrected series vs `true_value` over drift
  rows, against the uncorrected series as the baseline. The §11 macro-F1 ≥ 0.70 target
  therefore refers to those four types.
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

- Macro-F1 ≥ 0.70 across the **four detectable** types on held-out test data (drift is
  scored as correction quality, not detection — §10).
- Imputation RMSE beats linear interpolation on ≥ 2 of 3 datasets.
- Drift correction beats the uncorrected series on drift rows, on datasets with ≥ 3
  maintenance visits (§5).
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
  `scratchpad/probe_saqc.py` (every signature + a toy call) and
  `scratchpad/probe_saqc_behavior.py` (the §7.1 constraints) already exist — extend them
  rather than starting over, and re-run both if the pin ever moves.
- Keep the diff per phase reviewable; commit at every gate.
