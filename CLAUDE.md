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
│   ├── raw/              # downloaded USGS series, split by approval (gitignored)
│   │   ├── approved/     #   APPROVED = the clean bases; the ONLY dir inject globs (§9)
│   │   └── provisional/  #   unapproved pulls; audit scratch only (§9.1)
│   ├── injected/         # synthetic datasets + label files (injected straight from raw)
│   │   └── <gauge>/l<level>/  # the §5 triple, filed by gauge then level
│   ├── comparison/       # same sensor either side of the approval boundary (§9.3)
│   │   └── <gauge>/      #   *_approved.csv, *_provisional.csv, *_manifest.json
│   └── review/           # candidate proposals (gitignored) + reviewed labels (§9.1)
├── src/                  # grouped by AUDIENCE — see the note below the tree
│   ├── inspect_data.py   # load + validate the §5 contracts + summarise (shared foundation)
│   ├── evaluate.py       # metrics, fixed-pipeline baseline, ablation
│   ├── agent.py          # ReAct loop + API logger
│   ├── datasets/         # writes everything under data/
│   │   ├── pull_usgs.py  # pull approved-only turbidity from NWIS -> data/raw/approved (§9)
│   │   ├── pull_comparison.py # pull one series in BOTH approval states -> data/comparison (§9.3)
│   │   └── inject.py     # synthetic anomaly injection (4 types, 3 levels, seeded)
│   ├── agent_tools/      # AGENT-facing: the §7 inventory, handed to the Messages API
│   │   ├── schemas.py    # JSON tool schemas for the Messages API
│   │   ├── wrappers.py   # SaQC-wrapping tool functions (§5 result dict)
│   │   └── context.py    # point-context measurements: storm vs artifact (§7.3)
│   └── workbench/        # HUMAN-facing: run by a person; CLI/HTML/plot output
│       ├── visualize.py  # series/overview plots
│       ├── visualize_injected.py # plot injected datasets with anomaly labels
│       ├── visualize_log.py # replay one logs/*.jsonl run: flags per iteration (§8)
│       ├── param_sweep.py # sweep one param, score vs labels, plot (§7.2)
│       ├── candidates.py # propose anomalies in a "clean" series for review (§9.1)
│       └── review.py     # keyboard-driven labelling page + merge back to labels (§9.1)
├── app/
│   └── streamlit_app.py  # UI
├── tests/
│   └── test_tools.py
├── scratchpad/
│   ├── probe_saqc.py          # signature + toy-call probe for every §7 method
│   ├── probe_saqc_behavior.py # reproduces the §7.1 constraints
│   ├── probe_plateau_cost.py  # flagPlateau's crash + multiprocessing trap (§7.1)
│   ├── screen_prov_vs_approved.py # screens gauges for an approved+provisional split (§9.3)
│   ├── tune_candidates.py     # picks the §9.1 proposal thresholds
│   ├── tune_spike_recall.py   # sweeps the §9.1 spike params for >=95% recall
│   ├── probe_candidate_recall.py # candidates.py recall vs the injected labels
│   └── tune_zscore.py         # flagZScore min_residuals on quantised data (§7.1)
└── logs/                 # JSONL API logs (gitignored)
```

**`src/` is grouped by audience, because the three audiences impose different contracts**
(reorganised 2026-07-29; there is no longer a `src/tools/`):

- **`src/agent_tools/`** — what the agent calls. Every function here returns the §5
  tool-result dict and is described by a schema in `schemas.py`. Adding a tool means
  touching both files. Nothing here prints, plots, or writes to `data/`.
- **`src/workbench/`** — what a *person* runs. Free to emit CLIs, HTML pages, and Plotly
  figures; bound by no result-dict contract. These are how we tune and audit (§7.2, §9.1),
  not part of an agent run.
- **`src/datasets/`** — what produces `data/`. The §5 file layout is these modules'
  output contract.
- **Top level** — `inspect_data.py` is deliberately *not* in a subpackage: it owns the §5
  column contracts and validators and is imported by all three groups (`wrappers.py` builds
  the `inspect_dataset` tool on top of its `summarise_series`). `agent.py` and `evaluate.py`
  are single-purpose entry points.

Dependencies point one way — everything → `inspect_data`, `workbench` → `datasets` — and
must stay acyclic. If you find yourself needing `agent_tools` → `workbench`, the shared
piece belongs at the top level instead.

---

## 5. Data contracts (single source of truth — keep code consistent with this)

**Where a dataset lives.** `<name>` is always `<gauge>_l<level>`, and the §5 triple below
is filed together in `data/injected/<gauge>/l<level>/`. Keeping the three files in one
directory is load-bearing: `param_sweep` and `visualize_injected` find the labels *beside*
the series (`path.with_name(f"{stem}_labels.csv")`), so never split them across
directories. `src.datasets.inject.dataset_dir(root, name)` is the single place that maps a name to
its directory — derive paths from it rather than rebuilding the layout by hand.

**Injected dataset CSV** (`data/injected/<gauge>/l<level>/<name>.csv`)

- `datetime` (ISO 8601), `value` (float; may be NaN for gaps).

**Ground-truth labels** (`<name>_labels.csv`, beside the series, row-aligned by `datetime`)

- `datetime`, `is_anomaly` (bool), `anomaly_type` (one of: `spike`, `plateau`,
  `level_shift`, `gap`, or empty), `true_value` (the original clean value),
  `source` (`natural` | `injected`, or empty on non-anomalous rows).
- **Every** missing run is labelled `is_anomaly=True, anomaly_type=gap` — including
  gaps already present in the "clean" base — so the ground truth is honest rather than
  pretending the base is pristine. `source` separates the two: only `injected` gaps have
  a known `true_value`, so **imputation RMSE/MAE (§10) is scored on injected gaps only**,
  while both count for detection precision/recall.

**Maintenance schedule** — **removed.** `*_maintenance.csv` no longer exists and is no
longer emitted; drift is removed and this file existed only to feed `correctDrift` (§9.2).

**Injection manifest** (`<name>_manifest.json`, beside the series)

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

## 6. The four failure types and their default action

| Type               | Signature in the data                      | Detected by                          | Default action                                     |
| ------------------ | ------------------------------------------ | ------------------------------------ | -------------------------------------------------- |
| Spike              | one/few values far from neighbours         | `flagUniLOF`, `flagZScore`, `flagRange` | delete                                          |
| Plateau / stuck    | identical value repeated for a long window | `flagConstants` (+ `flagPlateau` if offset) | delete or flag                             |
| Level shift / jump | permanent step to a new level              | `flagJumps`                          | flag; keep unless clearly erroneous (agent judges) |
| Gap (missing)      | NaN run                                    | `flagNAN`                            | impute (short gaps only)                           |

The agent reasons about each flagged segment and picks the action; these are defaults, not
hard rules. Genuine extreme events must be **kept**, not "corrected" away.

**Drift was a fifth type and has been removed** (2026-07-22, §9.2). It is not injected, not
detected, not scored, and not in the label vocabulary. Do not add it back without reading
§9.2 first.

---

## 7. QC tool inventory

Each is a Python function in `src/agent_tools/wrappers.py` wrapping a SaQC 2.8 method, returning
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
| `flag_plateau`      | `flagPlateau`    | `min_length` **required by SaQC** (wrapper defaults `'1h'`), `max_length`, `min_jump`, `granularity` |
| `flag_spike_unilof` | `flagUniLOF`     | `n=20`, `thresh=None`, `density='auto'`, `slope_correct=True`|
| `flag_zscore`       | `flagZScore`     | `method='standard'\|'modified'`, `window`, `thresh=3`        |
| `flag_jumps`        | `flagJumps`      | `thresh`, `window` — both required                           |
| `flag_nan`          | `flagNAN`        | (field only)                                                 |

Action: `impute_rolling` (`interpolateByRolling`; `window` required, `func='median'`,
`min_periods=0`). There is no `correct_drift` wrapper — drift is removed (§9.2).

Context: `describe_point`, `describe_points` — thin pass-throughs in `wrappers.py` to the
§7.3 measurements. They observe only: no `qc` key in the result, caller's SaQC unchanged.

`flag_plateau` is an addition the probe justified: `flagConstants` and `flagPlateau`
detect **different** things and we want both (§7.1).

### 7.1 Probed constraints — these bit us, do not rediscover them

**`flagNAN` is the correct 2.8 name.** `flagMissing` also exists but is **deprecated since
2.7.0**. Do not "modernise" `flagNAN` into `flagMissing`.

**Drift methods (`correctDrift`, `flagDriftFromNorm`, `flagDriftFromReference`) are not used
and their probed constraints have moved to §9.2**, so this section stays about live methods.
Read §9.2 before touching any of them.

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

**`flagPlateau` is the one method that can crash or appear to hang. Wrap every call.**
Probed via `scratchpad/probe_plateau_cost.py` on real turbidity:

- It **raises `ValueError: attempt to get argmin of an empty sequence`** from
  `saqc/funcs/pattern.py::_getAnomalyCenter` on some inputs — data-dependent, not
  length-monotonic: on 11501000 it succeeded at n=3000 and n=21151 but raised at n=4000 and
  n=8000. A caller must catch this per-method and carry on, not let one detector sink a run.
- It is the only §7 method that uses **multiprocessing**, and its workers re-import
  `__main__`. Called from a `python -` / heredoc script, `__main__` is stdin, the re-import
  fails, the pool respawns forever, and it looks exactly like a hang — that cost an hour.
  **Run anything that touches `flagPlateau` from a real file or `python -m`.** From a file it
  is fast: 3.35 s on 21k rows.
- `min_length` is **required** — omitting it raises `TypeError: missing a required argument`
  before SaQC runs at all, so the wrapper defaults it to `'1h'` (§7.2) rather than `None`.
- Short series raise a *second* `ValueError` (`window shape cannot be larger than input array
  shape`, from numpy's stride tricks) — a 100-row toy series with `max_length='2h'` trips it.
- **The guard is now implemented** (2026-07-31): `flag_plateau` catches `ValueError` and
  returns a normal result dict with `n_flagged=0`, `failed=True`, the exception text, and the
  **unchanged** `qc`. It had been promised in this file and in the tool schema for a while but
  never written, so every crash reached the agent as a raw exception.
  `tests/test_tools.py::test_flag_plateau_survives_its_own_crash` holds the line.

**`flagZScore(method='modified')` needs `min_residuals` on quantised data.** In a window
where the MAD is ~0, any wiggle scores an enormous modified z, so `thresh` stops doing
anything. On 11501000 (turbidity quantised to 0.1 NTU) it flagged ~195 segments at
`thresh=30` just as at `thresh=6`. `min_residuals` sets a floor in **data units** and fixes
it: at 3 robust step-sigmas the same series drops to 7 segments while the other two gauges
barely move. Scale it to the series, never hard-code it (`scratchpad/tune_zscore.py`).

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

**Flagging is additive: a re-run cannot un-flag, and its count is only what it added**
(2026-07-31, measured on `03447687_l2`, first 8000 rows; the agent re-tunes detectors at run
time (§8), so this governs what it can and cannot do).

- A stricter second call **takes nothing back**. `flagUniLOF` at `thresh=1.1` (78 rows) then
  at `thresh=3.0` leaves all 78 flagged and reports **0 new**. Tightening is not an undo, so
  an agent must **start strict and loosen**; an over-flagged segment can only be handled with
  a `"keep"` decision in the §5 flag log, never by re-running.
- A re-run reports **only the rows it added**, since SaQC never re-flags an already-flagged
  row. `_build_result` therefore also returns **`n_flagged_total`** — the union for that field
  — and appends it to the message when the two differ. Compare *that* against the §9 sparsity
  prior, not the per-call count.
- A re-run is **not equivalent to a fresh run** at the new parameters: already-flagged rows
  are filtered out of the input for later tests (SaQC's `dfilter`), so strict→loose ended at
  70 flagged rows where a single loose run finds 78. Never report a re-run's count as what
  that parameter would have found alone.

**Index requirements.** The index **must be monotonic** — unsorted raises
`ValueError: index values must be monotonic`, so sort on load. An **irregular** index is
accepted by every method we use, including offset-string windows. Note that SaQC silently
accepts two things we should reject ourselves in `inspect_dataset`: a non-datetime
`RangeIndex`, and **duplicate timestamps**.

### 7.2 Practical parameter ranges (measured via `param_sweep`)

Swept against the injected labels with `src/workbench/param_sweep.py` across all three gauges
(median 8 / 13 / 28 FNU) and levels 1–3. Ranges are wide because the optimum shifts with
**turbidity scale** and **anomaly density**. These are starting ranges + directional rules
for the agent (§8), not hard bounds — re-run the sweep if the data changes. Overriding
principle: **raise a sensitivity threshold when the base is spiky/variable or anomalies are
sparse (favor precision); lower it when anomalies are dense or the base is calm (favor
recall).**

> ⚠ **These ranges were swept against the pre-2026-07-31 datasets and have NOT been re-run**
> since the §9 frequency cut (`0.8/2/3.5` → `0.15/0.4/0.8`) and the spike-magnitude floor
> raise (2× → 5×). Both changes push the same direction the table's own rule predicts:
> anomalies are now ~5× sparser, which favors **higher** thresholds for precision, while
> spikes are uniformly larger, which means a higher `flag_spike_unilof`/`flag_zscore`
> threshold now costs less recall than it used to. Treat every number below as a stale
> starting point and re-run `param_sweep` before quoting any of it as tuned.

| Tool · param | range (default) | increase (↑) when | decrease (↓) when |
| --- | --- | --- | --- |
| `flag_spike_unilof` · `thresh` (`n≈20`) | **1.2–2.0** (1.5) | many false positives; sparse anomalies (low level); base naturally spiky | missing spikes; dense spikes (high level); calm base. *LOF ratio → same range every gauge.* |
| `flag_zscore` · `thresh` (`modified`, `window≈12h`) | **6–12** (8) | spikier/more-variable base (French Broad → 12); too many FP | calm base; missing spikes. *Backup to UniLOF, which usually wins.* |
| `flag_range` · `min`/`max` | `min=0`, `max` **1000–2000** | raise `max` if it clips real extremes | lower `max` only to catch a known over-range fault. *Physical gate, not a sensitivity knob.* |
| `flag_constants` · `thresh` (`window` 3–12h) | **≤0.05** (0.01) | only if the noise sd is unusually large | keep small — `>0.5` swallows the series (§7.1). *≤0.05 robust on every gauge.* |
| `flag_plateau` · `min_length` | **1–3h** (1h) | — | — *Crash-prone/data-dependent (§7.1); finds little. Wrap it; rely on `flag_constants`.* |
| `flag_jumps` · `thresh` | **1–5** (2) | to cut false positives (precision stays ~1–5% regardless) | to catch all episodes at low thresh. *"Look here" aid only — can't be tuned reliably (~3 level_shift episodes/dataset; fires on storm limbs, §9.1).* |
| `impute_rolling` · `window` (`func='median'`) | **1–6h** (3h) | to fill longer gaps (↑coverage but ↑RMSE) | for accuracy on short gaps. *Beats linear only on the calmest gauge at 3h — open issue (§11).* |

Data-unit thresholds (`flag_zscore`, `flag_constants`, `flag_jumps`) don't transfer between
gauges of different scale — rescale by the series' robust (MAD) spread when moving to a new
gauge (§7.1). Measured across 3 gauges × 3 levels, but level_shift and imputation remain
under-powered, so treat those two rows as guidance-to-flag, not tuned optima.

### 7.3 Point-context tools (`src/agent_tools/context.py`, 2026-07-30)

Detectors say **where** a rule fired; they cannot say **whether it is real water**, which is
the §6 decision the agent actually has to make. `context.py` closes that gap: given one
timestamp it measures the shape around it and returns JSON-serialisable dicts. Nothing in it
flags or mutates — it observes. Eight primitives (`slope_context`, `excursion_context`,
`recovery_context`, `level_shift_context`, `flatness_context`, `neighbourhood_stats`,
`gap_context`, `historical_context`) plus two aggregators, `describe_point` and
`describe_points`. **Only the aggregators get schemas** — nine near-identical tools would
eat the 25-call cap; the primitives stay library functions.

Every threshold is in **robust sigmas of the series**, floored by the series-wide robust
first-difference scale, because a windowed MAD collapses to 0 on quantised turbidity and
blows up any local z (§7.1).

**What separates a storm peak from a spike (measured, 3 gauges × level 2, injected labels
as truth; `scratchpad/demo_point_context.py` reproduces the table):**

| metric | spike | storm peak | normal | plateau |
| --- | --- | --- | --- | --- |
| `|robust_z|` (±6h) | 4.8 | 1.3 | 0.4 | 0.0 |
| excursion width (samples at half height) | 2–3 | 7–25 | 2–14 | 22–43 |
| samples to recover | 2 | 12–30 | 1 | 1 |
| `step_sharpness` | 5–10 | 0.6 | 1.2 | 0.5 |

- **The rise-vs-fall gradient ratio does NOT work — do not rediscover this.** The intuition
  (storms recede gradually, spikes are symmetric) fails on this data: injected upward spikes
  and genuine storm peaks sit at median `fall_rise_ratio` 1.00 vs 0.97 at a 2 h window, and
  overlap just as badly at 1 h, 6 h and 24 h, with or without the point excluded from the
  fits. A storm's asymmetry is an **event-scale** property (rise in hours, recession over
  days), not a local gradient either side of one sample. `slope_context` is kept for shape
  and direction; **width and recovery time are what actually separate them.**
- **Recovery must be anchored at the excursion's onset, not at the point.** Mid-storm, the
  six hours before the *point* are already storm, so a point-anchored baseline reports an
  instant recovery for an event nowhere near over. `recovery_context(anchor='excursion')`
  is the default for this reason.
- **`reads_like` is a hint, not a verdict**, and it is deliberately conservative. Measured
  hit rates: spike 87/120 with 11 storm-peak false positives and 0 on normal rows; plateau
  exact; gap exact; **level_shift weak (6/9 onsets, 6 storm-peak FP)** — the same wall §9.1
  hit from a different direction, so treat that label as "look here".
- **Level shift is measured as a duration, not a persistence ratio**, because §9 injects
  bounded 4–24 h windows rather than permanent steps; a "does it hold forever" test would
  score every injected shift as transient. (The window was 6–72 h when this was measured;
  shortened 2026-07-31 with the frequency cut, so shifts are now shorter than the durations
  the table above was calibrated on — expect the weak level_shift hit rate to be no better.)

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

**Adaptive thinking is ON** (`thinking={"type": "adaptive"}`, 2026-08-01). Without it the
log records only the prose the model writes for the reader, not its reasoning, and
`src/workbench/visualize_log.py` has an empty thinking panel. Four things this pins down:

- **Adaptive, never a fixed `budget_tokens`.** The fixed-budget form is deprecated on
  `claude-sonnet-4-6` and *removed* (400) on 4.7 and later. Adaptive also enables
  **interleaved** thinking automatically — the agent reasons *between* tool calls, which is
  what makes the per-iteration trace worth reading. No beta header.
- **`display` is deliberately unset.** It defaults to `summarized` on 4.6 and the parameter
  only arrived with 4.7. If `MODEL` ever moves to 4.7+ the default flips to `omitted` and
  the thinking text comes back **empty** — pass `display="summarized"` at the same time.
- **`max_tokens` caps thinking + response together**, so it was raised 4096 → 16000. A run
  that truncates mid-report is the symptom of setting this too low.
- **Thinking blocks must be returned to the API unchanged** on the next turn. `agent.py`
  appends the whole `response.content`, which is already correct — do not "optimise" it into
  extracting just the text blocks, or the next request errors.

The model id is a §2 golden rule and lives in one place, `agent.py::MODEL`. It had drifted to
`claude-3-5-sonnet-20240620`, which predates adaptive thinking; that is fixed.

---

## 9. Data strategy

- **Real data:** USGS NWIS via the `dataretrieval` package (primary — cleanly scriptable);
  ECCC (secondary — may be a manual download). Target ~2 years, 15-min/hourly, 2–3 gauges
  per variable. Start with **one variable** (turbidity or specific conductance).
- **Approved data is the base — but "approved" ≠ "no spikes".** Record processing (TM 1-D3,
  §9.2) applies fouling and calibration-drift corrections and deletes *clearly erroneous*
  data, but it **keeps real turbidity spikes** — a storm first-flush or resuspension event is
  genuine signal, not an error. So a *flashy* river's approved record is still full of sharp
  excursions, and those unlabelled base spikes would score as **false positives** against the
  injected labels (§9.1). We therefore require the base to be **approved AND calm** — a
  baseline with almost no real spikes — rather than assuming approval alone makes it clean.
  `pull_usgs.py` keeps approved-only rows (drops `P`/blank, which become gaps) and writes to
  **`data/raw/approved/`**; `inject.py` globs that directory and only that one. The split is
  load-bearing, not cosmetic: `data/raw/provisional/` (and `expert_flagged/`) sit alongside it
  holding *unvetted* series, and the non-recursive glob is what keeps them out of the
  injection bases. `--keep-unapproved` therefore redirects `pull_usgs.py`'s default output to
  `data/raw/provisional/`. There is no `data/clean/` folder and no
  by-eye clean-segment selection. The §9.1 candidate/review tooling stays available for
  auditing a spikier or provisional base if one is ever used, but the default calm bases don't
  need it.
- **The three default gauges** (all **100% approved, consistent 15-min, ≥95% complete, and
  calm** — ≤0.04% real base spikes — over 2023-07→2025-07; regime spread moderate→high):
  `03447687` French Broad R nr Fletcher, NC (S. Appalachia; moderate, median ~8 FNU; ~95%
  complete; ~25 base spikes); `02198840` Savannah R at I-95 nr Port Wentworth, GA (tidal
  river; moderate-high, ~13 FNU, ~9× range; ~99% complete; 0 base spikes); `08041770` LNVA
  Canal at Beaumont, TX (managed canal; high, ~28 FNU, ~5× range; ~99% complete; ~1 base
  spike). Replaced `02203603` (South R, Atlanta) and `02198955` (Middle R, tidal): 100%
  approved and dense but too **flashy** — 120 (0.18%) and 637 (0.91%) real base spikes.
  Earlier retired: `12340500` (Blackfoot, 35% missing), `06818000` (Missouri, 14% missing),
  `11501000` (Sprague, mostly provisional). No clean *low/clear* base survived all filters —
  clear rivers are spring/mountain-fed (winter gaps or provisional), so the spread is
  moderate→high, not low→high.
- **Synthetic injection:** inject all four types at recorded locations into the approved
  bases; save the labels (§5). Build **three contamination levels** with a **fixed random
  seed** for reproducibility.
- **What the level scales.** The `0.15% / 0.4% / 0.8%` knob applies to the **point-like types
  only** (spike, plateau, gap), where "percent of rows" is a natural unit. `level_shift` is
  driven by episode count instead (1 / 2 / 3), and its row-share is a reported consequence
  rather than a target — so a dataset's **total** anomalous share exceeds its headline level,
  partly via *natural* gaps carried in from the base. Level 3 now lands at 5.6 / 2.4 / 1.7%
  on 03447687 / 02198840 / 08041770 (natural-gap share is 4.6 / 1.4 / 0.6%, which now
  *dominates* the total on 03447687). (These per-type rates were toned down in stages —
  `3 / 7 / 12` → `2 / 5 / 9` → `0.8 / 2 / 3.5` → the current `0.15 / 0.4 / 0.8`
  (2026-07-31) — to look like real, sparsely-anomalous records. At 3.5% the level-3 series
  was visibly speckled on a plot, which makes detection easier than the real problem.)
  Read per-type counts from the manifest/labels; never infer them from the level number.
- **Level 1 is now thin by design — check `n_events` before trusting a per-type score.**
  At 0.15% a dataset carries ~11–13 spike events, **1–2 plateau events**, and 3–4 injected
  gap events across two years. That is the intended realism, but per-type precision/recall
  on plateau at level 1 rests on one or two events, so it is noisy and a single miss swings
  it to 0. Score plateau on levels 2–3, or pool levels. All nine datasets still report
  `point_budget_met: true` and `types_missing: []`.
- **Spike magnitude is log-uniform 5–15× the local robust scale** (`SPIKE_MAGNITUDE_SCALE`).
  The floor was raised from 2× (2026-07-31): below ~5× an injected spike sits inside the
  local noise band and is not identifiable as an anomaly even by eye, so scoring a detector
  on it measures luck. The ceiling stayed at 15× deliberately — that is already the edge of
  physical plausibility for turbidity, and raising it would only add trivially-detectable
  outliers that flatter the metrics. **A downward spike that would clip at `VALUE_FLOOR` is
  flipped upward, not clipped**: turbidity cannot go negative, so on a low reading a −30 FNU
  draw used to land as a −3 FNU displacement, putting 9.9% of spike rows back under the
  floor (some at 0.7× local scale) and silently defeating the floor. Realised spread across
  all nine datasets is now p10 5.7× / median 9.0× / p90 13.5×.
- **Level shift is injected as a bounded window**, not a literal permanent step: a
  permanent step would either label every subsequent row anomalous (one mid-series shift
  ≈ 50% contamination) or leave post-step rows with `true_value != value` while marked
  not-anomalous, which the §5 label contract cannot express.
- **Isolated dropouts are not gaps.** Re-gridding a raw NWIS file onto its 15-min grid
  makes absent samples explicit, and the resulting NaN share looks alarming (06818000:
  14.3%) — but that record's _longest_ missing run across 253 days is 10 samples (2.5h),
  and 2,216 of its 2,744 missing runs are a single sample. Nothing there is visible on a
  plot, and `longest_unbroken_run_days` is right to pass it. Injection therefore lets
  segment anomalies (plateau, level_shift) span dropouts up to
  `SPANNABLE_DROPOUT_ROWS`, as they do in reality; only spikes and injected gaps require
  a real reading in every row. Treating every isolated dropout as blocking left just
  16.8% of that series usable and starved level 3 of plateaus and level shifts entirely.
  If a base genuinely cannot host its budget, injection reports `point_budget_met: false`
  and `types_missing` rather than silently under-filling.

### 9.1 Auditing a base (candidate proposal + human review)

**Not needed for the current approved bases.** Since we now inject into USGS *approved*
series (§9), which are clean apart from gaps, there is no by-eye base to audit and this
tooling is dormant. It is retained for one case: auditing a **provisional/unapproved** series
pulled with `--keep-unapproved`, where the "assumed anomaly-free" premise below does not hold.
The reviewed labels still in `data/review/` were made against the retired `data/clean/`
segments and are kept only for provenance; they are not part of the approved-base pipeline.

The original rationale (applies to any un-audited base): a base chosen **by eye** may still
hold real anomalies — which would silently become false positives when scoring detection
against injected labels (§10), because the base is assumed anomaly-free everywhere the label
file says nothing. `src/workbench/candidates.py` + `src/workbench/review.py` exist to check
that assumption:

```
python -m src.workbench.review detect data/raw/provisional/<gauge>.csv   # propose + open page
python -m src.workbench.review merge  data/raw/provisional/<gauge>.csv <decisions.csv>
```

- **Proposal is tuned for recall, not precision.** Detectors run at deliberately sensitive
  settings; false positives are expected and are what the review step removes. Every
  threshold in data units is derived from the series' own robust (MAD) scale, so the
  defaults transfer across gauges spanning 0–40 and 0–1000 NTU.
- **Recall is measured, and the target is >95% per type** (2026-07-29). The old defaults
  were picked to land in the *tens of segments* per type, on the reasoning that a detector
  proposing 800 segments cannot be reviewed by a human at all. Measured against the injected
  labels that costs far too much: spike recall was **59–81%** of injected spike rows, and the
  `max_per_type=50` cap alone accounted for most of it — a dataset proposes thousands of spike
  segments, so the cap discarded ~98% of them and a *truncated candidate never reaches the
  reviewer*. Both were changed:
  - `unilof_n` 20 → **10** (an injected spike is 1–3 rows; a 20-sample LOF neighbourhood blurs
    a 3-row burst into its own local density), `unilof_thresh` 1.5 → **1.1**, `zscore_thresh`
    10 → **3**, `zscore_min_residual_sigmas` 3.0 → **2.0**. Spike recall is now **96.8–99.1%**
    on all nine datasets. (`scratchpad/tune_spike_recall.py`.)
  - `max_per_type` 50 → **None** (opt-in, `--max-per-type`). It is a *presentation* limit for
    the review page, not a detection setting, and defaulting it on made the cap rather than the
    detectors the binding constraint on recall.
  - The cost is real and accepted: a proposal now flags 4–10% of a two-year series across
    ~2,500 spike segments, which is past one-by-one human review. Reviewing a provisional base
    means setting `--max-per-type` explicitly and accepting the recall that buys.
  `tests/test_candidates.py::test_candidate_type_recall_on_injected_datasets` holds the line at
  95% per type per dataset. **These recall figures predate the 2026-07-31 regeneration and
  have not been re-measured** — the whole `tests/test_candidates.py` file is currently failing
  for an unrelated environment reason (see the §9 note below), so the numbers here are the last
  known-good ones, not a current reading. As recorded then: **green for spike, gap; red for
  level_shift** (2.8–27.9% — `flagJumps` marks a step's *edge* while the §5 label covers the
  whole injected window, so row-recall cannot be high; open, see the level_shift bullet below)
  **and for plateau on 03447687_l1** (88.6%; every other dataset is 95.2–100%). Note the
  frequency cut makes plateau recall at level 1 rest on 1–2 events, so that figure will be far
  noisier than it was.
- **Only spike, plateau and level_shift are reviewed — gaps are never queued.** Whether a
  value is missing is not a judgement call, and §5 is explicit that *every* missing run is
  `anomaly_type=gap`, so putting gaps in the queue only invites a reviewer to press "normal"
  on one and produce labels that contradict the contract. `merge` labels them from the NaN
  mask instead, with **no length threshold**. The first version filtered gaps to runs ≥ 1 h,
  which looked reasonable and was badly wrong: it omitted 91% / 93% / 59% of the missing
  rows on 03447687 / 06818000 / 11501000, because those series are dominated by sub-hour
  dropouts. §9's "isolated dropouts are not gaps" governs where *injection* may place
  anomalies — it does not govern *labelling*, and importing it here was the mistake.
  (The one case where a NaN run is not a real gap: a series whose sampling rate changes
  mid-record, where re-gridding to the modal step manufactures phantom NaN. None of the
  three gauges does this. The guard for that is a periodicity check in `inspect_data`, not
  a review queue.)
- **level_shift is `flagJumps` + a sharpness filter — it was drowning in storms.**
  `flagJumps` flags any change of `thresh` within its window, so on storm-driven turbidity
  it fires on every rising and falling limb: gradual slopes that are normal water behaviour.
  Neither `window` nor `thresh` nor a persistence test fixes this (a shorter window catches
  more; storms recede over weeks; wet seasons shift the baseline for weeks). The lever that
  works is **sharpness** — the largest single-sample move as a fraction of the net step. A
  recalibration / sensor swap moves most of its magnitude in one sample; a storm spreads it
  over hours. `shift_min_sharpness=0.5` cut candidates 85/23/5 → 8/1/0 on the three gauges.
  **It does not, and cannot, reject a flash-flood onset** — sharp and sustained, identical
  to a real step in one series. This is the level_shift analogue of the drift wall (§9.2):
  across all 78 level_shift candidates reviewed before the filter, the human rejected every
  one. Treat surviving candidates as "look here", not "this is an artifact".
  (`scratchpad/tune_level_shift.py`.)
- **Spikes are never bridged; segment types are.** `segment_bridge` (2h) stitches a
  patchily-detected plateau or level_shift back into one event. Applying it to spikes was a
  bug: §6 defines a spike as one/few values far from neighbours, so bridging glued distinct
  spikes together across the normal rows between them — 12% / 19% / 27% of the rows inside
  spike spans on the three gauges had never been flagged at all, and confirming such a
  candidate would have labelled them spikes. One real case on 06818000: an 11-row "spike"
  spanning 12:30–15:15 on 2024-07-16 was actually an isolated 66.7 NTU point at 12:30, a
  NaN run, and a separate 52–55 bump at 14:30 — now three separate decisions.
- **A decision may narrow a candidate, never extend it.** A detector marks a *window*;
  often one sample in it is the anomaly. Clicking a point in the page narrows the label to
  that sample, and the exported `start`/`end` (end **inclusive**) carry it through to
  `merge_decisions`, which validates the span lies inside the original proposal. The CSV is
  therefore hand-editable, and `start`/`end` are the source of truth for what was endorsed.
- **The review page is a single self-contained HTML file** — plotly.js inlined, no server,
  no Streamlit, works offline. One candidate at a time, `A`/`N`/`U` to judge, auto-advance,
  `Z` to undo, progress mirrored to `localStorage` so closing the tab loses nothing.
  That store is keyed on the dataset name **and candidate count**, so re-running `detect`
  with different options orphans an in-progress review. Finish a pass before retuning.
- **Two page traps, both verified in a browser, both silent failures.** Plotly renders a
  `Date` object in the *viewer's* timezone, so an axis built from Dates printed hours away
  from the timestamps in the side panel and the CSV; the series is timezone-naive, so x
  values must be naive ISO **strings**. And `gd.on(...)` does not exist until Plotly has
  plotted into that div — registering `plotly_click` at start-up throws and takes the rest
  of the init down with it, keyboard handlers included. Wire it after the first draw.
- **`merge` writes the §5 labels contract** with `source=natural` and an empty `true_value`:
  these anomalies were already in the record, so no uncontaminated value exists for them
  and they are scoreable for detection but not for imputation (§5, §10). It **refuses**
  decisions whose `candidate_id`s do not match the current proposal run, since ids come
  from the detector settings and a silent mismatch would drop verdicts.
- Reviewed labels are **hand-made ground truth — commit them**. The `*_candidates.csv`
  proposals are regenerable and gitignored.

### 9.2 Drift is REMOVED (decision, 2026-07-22)

**Drift is gone from the project — injection, detection, correction, scoring, and the label
vocabulary. This project handles four types: spike, plateau, level_shift, gap.** Do not
reintroduce drift without reading this section end to end.

What was removed, and where it used to live:

| Removed | From |
| ------- | ---- |
| `inject_drift`, `_place_drift_episodes`, `_n_drift_episodes`, `_inject_drift_and_maintenance`, `MaintenanceEvent`, `DRIFT_*` / `MAINTENANCE_*` constants, `ContaminationLevel.maintenance_interval_days`, `InjectionResult.maintenance` | `src/datasets/inject.py` |
| `_detect_drift` heuristic proposer, `drift_*` config fields, the `drift` type/colour | `src/workbench/candidates.py` |
| the `drift` review category and `--drift-min-days` | `src/workbench/review.py` |
| the `correct_drift` ToolSpec, `MAINT_FIELD`, `maintenance_variable`, the whole `correct` tool kind | `src/workbench/param_sweep.py` |
| `"drift"` from `ANOMALY_TYPES` | `src/inspect_data.py` |
| all nine `data/injected/*_maintenance.csv`; all nine datasets regenerated without drift | `data/injected/` |

The knowledge below is kept deliberately: it was expensive to establish, and it is the
reason this decision was made rather than an argument for revisiting it cheaply.

**Why drift was removed.** It cost more than the other four types combined, structurally:

- **No detector exists.** SaQC 2.8 has no univariate drift detector, so drift could only ever
  be handed in out of band via a maintenance schedule — a poor fit for a project whose thesis
  is an agent *adaptively selecting* detectors. Probed: `flagDriftFromNorm` clusters a
  **group** of fields and needs **≥3** to form a "norm" (1 field → 0 flagged, 2 → 0, 4 → 560);
  `flagDriftFromReference` needs a separate reference variable. Both collide with the "one
  variable per run" golden rule. `flagJumps` does not catch slow drift either (0 flagged on a
  1000-row ramp).
- **`correctDrift` was the sharpest edge in the library.** Probed behaviour, all verified:
  - The maintenance variable had to be a **standalone variable with its own short index**
    (index = visit start, value = visit end, `datetime64`), built as
    `saqc.SaQC({"value": series, "maintenance": maint})` — a **dict**, so the two variables
    keep independent indexes. NaN-padding visits onto the data index fails outright: pandas 3
    raises `TypeError: Invalid value ... for dtype 'float64'`.
  - It corrects only spans **between** support points; data before the first visit ends and
    after the last visit starts is never touched.
  - **`N` visits yield `N-1` intervals, of which only the first `N-2` are corrected** — the
    implementation calls `.shift(-1)` for each interval's target level and the last has no
    successor. **Fewer than 3 visits corrects nothing.**
  - **The final interval is silently overwritten with `NaN` and left `UNFLAGGED` (`-inf`).**
    A 4000-row input with zero NaNs came back with 591 NaNs and no flag explaining them —
    silent data loss the §5 flag log would never record.
    (`scratchpad/demo_correctdrift_bug.py` reproduces this.)
- **The contamination knob was never resolved.** Levels set the maintenance interval
  (200/100/55 days), but drift row-share ≈ `DRIFT_DURATION_DAYS / interval`, so a *shorter*
  interval yielded *more* drift. Measured on the three bases: 21 d → 93–95% drift rows,
  28 d → 76–79%, vs 200 d → 8–13%. Two consequences we hit head-on:
  - Real USGS cadence is 2–4 weeks (below), so any realistic interval floods the series with
    drift. The honest fix was to move the severity knob from *interval* to drift *rate*
    (magnitude = rate × episode duration) — a redesign that regenerates every dataset and
    invalidates tuning done against them (§9.1 thresholds, §7.1 z-score work).
  - Mapping level 1 → 14 days **raised `ValueError`** on 03447687 and 11501000:
    `_place_drift_episodes` needed each equal block to hold a ≥14-day episode plus its
    maintenance window, which a 14-day block cannot.

**What we learned about real maintenance dates (keep — it was expensive to establish).**
USGS gives you **no** sensor service record. `nwis.get_iv` returns `datetime, value,
qualifier`, and `qualifier` is approval status only (`A`/`P`/`A, >`) — none of the three
gauges carries a `Mnt`/`Eqp`/`Ice` code. Two public proxies exist, both partial:

- `waterdata.get_field_measurements(monitoring_location_id='USGS-<site>')` — hydrographer
  site visits via `field_visit_id`. In our clean windows: 11 visits (03447687), 23
  (06818000), 8 (11501000). But these are discharge/gage-height trips (params `00060`,
  `00065`), so station-level, not turbidity-sensor-level. Note `nwis.get_discharge_
  measurements` now raises and redirects here; it returns a **tuple**.
- `waterdata.get_samples(...)` — discrete WQ samples, which usually accompany WQ sensor
  servicing. Inconsistent across gauges: **03447687 has none at all**, 06818000 has 2
  turbidity dates in-window (both on field-visit dates), and 11501000 has 16 on a clean
  ~14-day cadence, 4 of which fall exactly on its field-visit dates.

So real cadence is **2–4 weeks** where observable, but no source proves *service*. The only
route to certainty is emailing the Water Science Center for field notes (SIMS/Aquarius).
Moot now that drift is removed; recorded so it is not re-derived.

**Also relevant: approved data is already drift-corrected.** Per USGS TM 1-D3 (Wagner et
al. 2006), record processing applies fouling and calibration-drift corrections, prorated
between field visits, before publication — so `A`-qualified series are a *poor* place to
look for real drift. Residual drift survives only where fouling was non-linear, below
threshold, or between sparse visits. 11501000's clean segment is **provisional** (`P`,
20,717 of 20,853 rows), i.e. not yet through final processing, making it the one plausible
place to observe real uncorrected drift if this is ever resumed.

**To reintroduce drift**, in order: (1) pick the severity knob — drift *rate*, not interval;
(2) make the episode duration a fraction of the inter-visit block so placement cannot trip at
realistic cadence; (3) write the `correctDrift` guard the probe findings above demand;
(4) restore `"drift"` to `ANOMALY_TYPES`, the injector, and the maintenance schedule;
(5) regenerate all nine datasets and re-check §9.1 tuning; (6) restore the §10 and §11 drift
criteria. Recover the deleted code from git history rather than rewriting it — branch
`no_drift_detection`, the commit before the removal.

### 9.3 Provisional vs approved (`data/comparison/`, 2026-07-29)

**It is a split, not a pair — do not go looking for the archived provisional values.**
NWIS serves the *current* state of a record: when USGS approves a period, the provisional
values it carried are **overwritten in place**. There is no public archive of superseded
provisional values, and the instantaneous-values service has no revision / as-of parameter,
so **the same timestamps cannot be retrieved in two states**. A row-for-row before/after
diff would have to have been captured live, pre-approval, and kept. (Checked 2026-07-29.
The `63680_final` / `63680_from multiparameter sonde…` column names some sites return are
*time-series labels*, not a raw-vs-corrected pair — do not mistake them for one.)

What is retrievable is the moving **approval boundary**: older rows are `A` (record
processing applied per TM 1-D3 — fouling/drift corrections prorated between field visits,
clearly-erroneous data deleted), newer rows are `P` (essentially as the sensor reported).
Same site, same sensor, same 15-min grid, opposite sides of the processing step.
`src/datasets/pull_comparison.py` splits one pull on that boundary and writes both sides plus a
manifest to `data/comparison/<gauge>/`.

- **Compare distributions, not counterparts.** The two sides are different calendar
  periods, so seasonality confounds a naive comparison. Each manifest carries a
  `season_matched_approved_window` — the same calendar dates one year earlier — as the
  fairer slice of the approved side.
- **The three gauges** (screened from a 24-gauge pool by
  `scratchpad/screen_prov_vs_approved.py` on: ≥1 yr approved, ≥3 mo provisional, consistent
  15-min, dense on *both* sides): `03447687` French Broad, NC (S. Appalachian mountain;
  moderate ~8 FNU; 1018 d / 96.2% approved vs 106 d / 98.9% provisional); `02198840`
  Savannah at I-95, GA (Atlantic tidal; moderate-high ~13 FNU; 894 d / 98.8% vs 231 d /
  98.0%); `06818000` Missouri at St Joseph, MO (Great Plains; high ~24 FNU; 926 d / 85.4%
  vs 198 d / 90.4%). All three split cleanly — zero approved rows after the boundary, zero
  timestamp overlap.
- **`08041770` (LNVA Canal, TX) cannot be used here** — its record ends 2025-11-19 with
  **zero** provisional rows, so the third injection base has no comparison counterpart.
  `06818000` stands in: it was retired as an *injection base* for 14% missing rows, but §9
  establishes those are overwhelmingly isolated single-sample dropouts, which does not
  disqualify it for a distribution comparison.
- **Modified approved codes stay on the approved side** (`A e` estimated, `A, >`
  over-range, `A, R` revised) — USGS approved those readings. `P`, blank and NaN are
  not-approved.
- This is **reference material, not an injection base**. Nothing in `data/comparison/`
  feeds `src.datasets.inject`, which globs `data/raw/approved/` and only that (§9). The CSVs are
  gitignored and regenerable; `data/comparison/README.md` is committed.

---

## 10. Evaluation

- **Detection:** precision / recall / F1 per anomaly type + macro-F1 (scikit-learn), vs the
  injected labels, over the **four** types (spike, plateau, level_shift, gap). The §11
  macro-F1 ≥ 0.70 target refers to those four. Drift is not scored at all — it is removed
  (§9.2); it was already excluded from macro-F1 when it existed, so the headline number is
  unchanged by its removal, and there is no longer a drift correction-quality metric.
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
- ~~Drift correction beats the uncorrected series on drift rows, on datasets with ≥ 3
  maintenance visits (§5).~~ **Dropped — drift is removed (§9.2).** Restore this criterion
  only if drift is reintroduced.
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
- **Phase 1 — Data + injection.** `inspect_data.py`; pull approved USGS data via
  `pull_usgs.py` (`dataretrieval`); `inject.py` implementing all four anomaly types, labels,
  seed, and the three levels.
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
