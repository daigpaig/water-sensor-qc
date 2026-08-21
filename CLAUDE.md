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
│   ├── legacy_15min/     # the RETIRED 15-min bases + their injected datasets (§9)
│   │   ├── raw/          #   approved/ + provisional/ 15-min pulls (gitignored)
│   │   └── injected/     #   the nine 15-min §5 triples (tracked; see its README)
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
│       ├── spike_audit.py # one run's SPIKE decisions case by case, with evidence (§10.1)
│       ├── decision_audit.py # click ANY point: why it got the verdict it got (§10.1)
│       ├── provenance.py # rebuilds one point's story from the run log (§10.1)
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
│   ├── screen_5min_turbidity.py # national catalog sweep for 5-min turbidity (§9)
│   ├── screen_5min_stage2.py  # measures each candidate's ACTUAL cadence (§9)
│   ├── screen_5min_stage3.py  # approval/completeness/calmness per candidate (§9)
│   ├── check_cadence_stability.py # guards the §9.1 phantom-NaN trap
│   ├── find_precip.py         # nearest instantaneous rain gauges to a site (§9.4)
│   ├── verify_precip.py       # confirms those series really cover the window (§9.4)
│   ├── compare_candidate_recall.py # §9.1 recall, 5-min vs the retired 15-min set
│   ├── tune_noise_context.py  # picks the §7.4 noise window + rule thresholds
│   ├── tune_noise_spare_rule.py # refits §7.4 for 5-min data, on gauges we don't score
│   ├── probe_cache_ttl.py     # is `ttl` accepted on top-level cache_control? (§8)
│   ├── probe_noise_reads_like.py # the §7.4 noisy-stretch guard's hit rates
│   ├── tune_zscore.py         # flagZScore min_residuals on quantised data (§7.1)
│   ├── tune_jumps.py          # flag_jumps thresh as a multiple of the value MAD (§7.6)
│   ├── tune_jumps_scale.py    # why no static summary stat can size that thresh (§7.6)
│   ├── tune_jumps_quantile.py # thresh as a quantile of flagJumps' OWN statistic (§7.6)
│   └── tune_jumps_knee.py     # the recall-vs-candidate knee that set the default (§7.6)
└── logs/                 # JSONL API logs (gitignored)
```

**`src/` is grouped by audience, because the three audiences impose different contracts**
(reorganised 2026-07-29; there is no longer a `src/tools/`):

- **`src/agent_tools/`** — what the agent calls. Every function here returns the §5
  tool-result dict and is described by a schema in `schemas.py`. Adding a tool means
  touching both files. Nothing here prints or plots. **One exception to "writes nothing":
  `export_clean_data` writes the §5 flag log** (2026-08-10) — it is the tool that *is* the
  export step, and the log has to be assembled from the agent's `decisions` argument at
  the moment it is passed. The path is not the agent's to choose: `output_dir`/`stem` are
  injected by the runner in `agent.py`'s dispatch and are deliberately **absent from the
  tool schema**, so the model decides what goes in the log and never where it lands. Given
  neither, the tool writes nothing and returns the entries only.
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

**`flagged_datetimes` is a SAMPLE, not the whole set** (2026-08-10). It is capped at
`wrappers.MAX_FLAGGED_DATETIMES` (**1000**, not the 250 this line claimed until 2026-08-20)
and **evenly spaced across the record** — a head would
put every point the agent inspects in the first weeks of a two-year series. `n_flagged` /
`n_flagged_total` stay exact; `n_flagged_datetimes_shown` says how many came back, and the
`message` says so loudly whenever it truncates, because an agent that believes it received
every timestamp writes decision spans that silently miss thousands of rows. Same for
`impute_rolling`'s `gaps_summary`, capped at 40 and sorted longest-first (which gaps were too
long to fill is the decision it informs). Measured justification in §8.

**Cleaned output** (`*_clean.csv`): `datetime`, `value` (corrected/imputed; deleted values
as NaN or removed per config), plus a `flag` column naming the action, if any.

**Flag log** (`*_flags.json`): a list of
`{ "datetime": "...", "flagged_by": "<tool>", "verdict": "anomaly|normal", "anomaly_type": "spike|plateau|level_shift|gap", "action": "delete|correct|keep|impute", "reason": "<short text>" }`,
**one entry per flagged row**, written by `export_clean_data` (§4) from the `decisions`
the agent passes it — a list of
`{start, end (inclusive, optional), verdict, anomaly_type, action, reason}` spans,
narrowest covering span wins. Details the contract above did not say:

- `flagged_by` is **`+`-joined** when several tests flagged the same row (`flagUniLOF+flagRange`),
  read from the history, because `qc.flags` attributes such a row to the first test only (§7.1).
- **Each entry also carries `rationale`, `rationale_source` and `deliberation`** (2026-08-11).
  `rationale_source` is one of `agent-deliberation` (a close call, with the agent's reasoning
  attached), `agent-reason` (a specific decision it judged clear-cut), `blanket` (swept up by a
  span covering >20% of all flagged rows — `wrappers.BLANKET_SHARE`), or `deterministic` (code
  decided: filled by the imputer, or never adjudicated). This exists because the first three
  produce identically-shaped entries otherwise, and **every run so far has had a blanket absorb
  points the agent had actually measured** — the log could not show that. `rationale` is a
  sentence a reader can act on; for the deterministic cases it is generated in code.
- **Each entry also carries `decided_by`** (2026-08-18): the span that claimed the row —
  its `start`/`end`, its `difficulty`, `n_flagged_rows_in_span` (its whole reach) and
  `n_rows_claimed` (what survived narrower spans). `rationale_source` already said *that*
  a row was swept up by a blanket; this says by **which** span and how wide it was, which
  is what an auditor needs in order to tell a judgement from an absorption. It is `null`
  wherever `rationale_source` is `deterministic`, because pinning a code decision on a
  span the agent wrote would credit it with a judgement it never made.
- **`difficulty` is REQUIRED and has no default** (2026-08-20). It defaulted to `"clear"`,
  and the field was consequently inert: measured on the 2026-08-19 run, **145 of 147 spans
  omitted it**, so 98.6% of the run read as clear-cut — and those deletions were **wrong
  37.8% of the time**. A default that manufactures a confidence claim nobody made is worse
  than no field, because §10.1's review queue is built from exactly this signal. Required,
  it calibrates: on the next run 12.1% of spans were judgement calls, and those were wrong
  66.7% vs 27.8% for `clear` — 2.4x, which is what makes the queue worth reviewing. Note
  `clear` is still wrong 27.8% of the time: directionally calibrated, not yet sufficient.
- **A decision marked `difficulty: "judgement-call"` MUST carry a `deliberation`** and
  `export_clean_data` raises without one (§13, fail loudly). `reason` states the conclusion;
  `deliberation` shows the working — what pointed which way, what was weighed, what would have
  changed the agent's mind. This is the §1 requirement that the tool explain itself, made
  enforceable rather than aspirational.
- **`action` has a fifth value, `undecided`** (2026-08-10), which the agent may *not* choose:
  it is what a flagged row gets when no decision covered it. §10 scores an undecided row and a
  `normal` row identically (neither is a positive claim), but they are not the same thing —
  one is a judgement, the other is a row nobody looked at — and writing them the same way hides
  a run that flagged 2,595 rows and concluded nothing. `verdict` carries the same third value
  for the same reason. The tool's result and `message` report `n_undecided` and any decision
  span that matched **no** flagged row (a mistyped timestamp), so the agent can spend one more
  call fixing it.

### 5.1 `verdict` is the run's answer; `action` is the treatment (2026-08-13)

**Detection metrics are scored on `verdict`, not on `action`.** Until this date the two were
one field: §10 inferred the claim from whether the action was destructive
(`delete`/`correct`/`impute` = "the agent says this is an anomaly"). That conflated two
genuinely different questions and cost real recall — a gap correctly identified but too long
to fill defensibly was recorded `keep`, and scored as the agent claiming the data was fine.

- **`verdict`** ∈ `anomaly` | `normal` (+ `undecided`, code-assigned). The agent's claim about
  what the data **is**. This and nothing else is what precision/recall measure.
- **`anomaly_type`** — required when `verdict='anomaly'`, forbidden otherwise, drawn from
  `inspect_data.ANOMALY_TYPES` so it cannot drift from the label vocabulary. It is **the
  agent's classification, scored as such**: a row `flagJumps` found but the agent calls a spike
  is a spike claim. Typing predictions by *which detector fired* (`evaluate.TOOL_TO_TYPE`)
  measures the detector, not the agent, and remains only as the fallback for older logs.
- **`action`** ∈ `delete` | `correct` | `keep` | `impute`. Unchanged, and still what
  `report_decisions` grades.

**Consistency is enforced one-directionally** (`wrappers._validate_verdict`), because the two
directions are not symmetric:

- `verdict='normal'` **must** take `action='keep'`. Calling a value real water and then
  deleting it is not a defensible pair, and allowing it would let a run score as having
  *rejected* a candidate whose value is gone from the exported file.
- `verdict='anomaly'` normally takes `delete`/`correct`/`impute` but **may** take `keep`,
  provided the `reason` says why the segment cannot be treated (≥20 chars, enforced). The
  case this exists for is a gap longer than any defensible imputation window: forcing
  consistency there would make the agent either lie about the verdict or impute a gap it had
  just judged unfillable. `n_anomalies_left_untreated` surfaces the count, since the pair
  claims a detection without touching the value.

A row the imputer filled gets `verdict='anomaly', anomaly_type='gap'` deterministically, for
the same reason its `action` is forced to `impute`: §5 says every missing run is a gap, so
this is a fact about the data rather than a call the agent makes.

**Every flagged row needs a decision or the run scores as if it claimed nothing.** §10 counts
only `verdict='anomaly'` as a positive claim, so an export without `decisions` produces a log
that is syntactically valid and evidentially empty.

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
per-column min/max/mean/std), `get_flag_summary` (counts by tool), `export_clean_data`
(`decisions` — the agent's per-segment verdicts; emits the cleaned frame **and writes the §5
flag log**).

Detection:

| Wrapper             | SaQC 2.8 method  | Key parameters                                             |
| ------------------- | ---------------- | ---------------------------------------------------------- |
| `flag_range`        | `flagRange`      | `min`, `max` (both default `None`)                          |
| `flag_constants`    | `flagConstants`  | `thresh`, `window`, `min_periods=2` — both required          |
| `flag_plateau`      | `flagPlateau`    | `min_length` **required by SaQC** (wrapper defaults `'1h'`), `max_length`, `min_jump`, `granularity` |
| `flag_spike_unilof` | `flagUniLOF`     | `n=20`, `thresh=None`, `density='auto'`, `slope_correct=True`|
| `flag_zscore`       | `flagZScore`     | `method='standard'\|'modified'`, `window`, `thresh=3`        |
| `flag_jumps`        | `flagJumps`      | `thresh`, `window` — both required, no defaults; take them from `inspect_dataset`'s `jump_scale` (§7.6) |
| `flag_nan`          | `flagNAN`        | (field only)                                                 |

Action: `impute_rolling` (`interpolateByRolling`; `window` required, `func='median'`,
`min_periods=0`). There is no `correct_drift` wrapper — drift is removed (§9.2).

Context: `describe_point`, `describe_points` — thin pass-throughs in `wrappers.py` to the
§7.3 measurements. They observe only: no `qc` key in the result, caller's SaQC unchanged.
The five exposed primitives (`slope_context`, `excursion_context`, `recovery_context`,
`level_shift_context`, `noise_context`) have no wrapper at all — `agent.py` resolves them
straight out of `context.py`, so a schema is the whole of what exposing one costs (§7.3).
`noise_context` answers a different question from the rest: not "what shape is this point"
but "how noisy is this STRETCH, and does the point stand out within it" (§7.4).

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

**`interpolateByRolling` overwrites already-flagged readings unless you pass
`dfilter=np.inf`** (fixed 2026-08-13). SaQC masks every row whose flag is >= `dfilter`
before a function runs, and the default masks BAD — so a row an earlier detector flagged
looks *missing* to the imputer, which fills it, silently replacing a reading that was
never absent. Measured on 03447687_l1: **1,887 real readings rewritten with a rolling
median**, 1,866 of them ordinary water and 13 injected anomalies the agent therefore
never judged. It corrupted three separate measurements before anyone noticed, because an
overwritten row carries `action=impute` and reads as a successful gap fill: `evaluate.py`
reported spike precision 0.061 where 261 of its 277 "spike predictions" were overwrites
rather than agent decisions, and both audit pages reported `missed=0` while ten spikes
sat unjudged. With `dfilter=np.inf` the same sequence overwrites **0** rows and the
imputer's `n_flagged` finally equals its `n_imputed`. `scratchpad/probe_impute_dfilter.py`
reproduces both behaviours; a test holds the line.

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

> ⚠ **Every number below was swept on the retired 15-min gauges and is now doubly stale.**
> First, it predates the 2026-07-31 frequency cut (`0.8/2/3.5` → `0.15/0.4/0.8`) and
> spike-magnitude floor raise (2× → 5×) — both push the same way the table's own rule
> predicts: anomalies are ~5× sparser, favouring **higher** thresholds for precision, and
> spikes are uniformly larger, so a higher `flag_spike_unilof`/`flag_zscore` threshold costs
> less recall than it used to. Second, and more disruptive, **the bases moved to a 5-minute
> step on 2026-08-18 (§9)**. The `window` entries are durations and carry over unchanged,
> but anything counted in **samples** now spans a third of the time it did: `flag_spike_unilof`'s
> `n≈20` neighbourhood was 5 hours and is now 100 minutes, which is a different question about
> the data, not the same one measured more finely. The gauges are new too, so the data-unit
> thresholds (`flag_zscore`, `flag_constants`, `flag_jumps`) need rescaling to the new series'
> robust spread regardless. Re-run `param_sweep` before quoting any of this as tuned.

| Tool · param | range (default) | increase (↑) when | decrease (↓) when |
| --- | --- | --- | --- |
| `flag_spike_unilof` · `thresh` (`n≈20`) | **1.2–2.0** (1.5) | many false positives; sparse anomalies (low level); base naturally spiky | missing spikes; dense spikes (high level); calm base. *LOF ratio → same range every gauge.* |
| `flag_zscore` · `thresh` (`modified`, `window≈12h`) | **6–12** (8) | spikier/more-variable base (French Broad → 12); too many FP | calm base; missing spikes. *Backup to UniLOF, which usually wins.* |
| `flag_range` · `min`/`max` | `min=0`, `max` **1000–2000** | raise `max` if it clips real extremes | lower `max` only to catch a known over-range fault. *Physical gate, not a sensitivity knob.* |
| `flag_constants` · `thresh` (`window` 3–12h) | **≤0.05** (0.01) | only if the noise sd is unusually large | keep small — `>0.5` swallows the series (§7.1). *≤0.05 robust on every gauge.* |
| `flag_plateau` · `min_length` | **1–3h** (1h) | — | — *Crash-prone/data-dependent (§7.1); finds little. Wrap it; rely on `flag_constants`.* |
| `flag_jumps` · `thresh` | **`jump_scale.recommended_thresh`** from `inspect_dataset` — 6.2 / 13.8 / 75.7 FNU on the three gauges (§7.6) | to `p99_9` if there are more candidates than you can triage | only if the result is empty. *Never a raw number: the old "1–5" row is what produced a 2,865-flag run. "Look here" aid regardless (~1–3 level_shift episodes/dataset; fires on storm limbs, §9.1).* |
| `impute_rolling` · `window` (`func='median'`) | **1–6h** (3h) | to fill longer gaps (↑coverage but ↑RMSE) | for accuracy on short gaps. *Beats linear only on the calmest gauge at 3h — open issue (§11).* |

Data-unit thresholds (`flag_zscore`, `flag_constants`, `flag_jumps`) don't transfer between
gauges of different scale — rescale by the series' robust (MAD) spread when moving to a new
gauge (§7.1). Measured across 3 gauges × 3 levels, but level_shift and imputation remain
under-powered, so treat those two rows as guidance-to-flag, not tuned optima.

### 7.3 Point-context tools (`src/agent_tools/context.py`, 2026-07-30)

Detectors say **where** a rule fired; they cannot say **whether it is real water**, which is
the §6 decision the agent actually has to make. `context.py` closes that gap: given one
timestamp it measures the shape around it and returns JSON-serialisable dicts. Nothing in it
flags or mutates — it observes. Nine primitives (`slope_context`, `excursion_context`,
`recovery_context`, `level_shift_context`, `flatness_context`, `neighbourhood_stats`,
`noise_context`, `gap_context`, `historical_context`) plus two aggregators, `describe_point`
and `describe_points`. **The two aggregators and five primitives get schemas** — the other
four stay library functions, because eleven near-identical tools would eat the 25-call cap
and `describe_point` already returns all nine blocks at once.

The exposed primitives are `excursion_context`, `recovery_context` and `level_shift_context`
(2026-08-10), `slope_context` (2026-08-11, retuned to 45 min) and `noise_context`
(2026-08-13, §7.4): **width and recovery time are what
actually separate a storm peak from a spike** (the measured table below), and `level_shift` is
the weakest `reads_like` label, so the agent needs to interrogate it directly rather than trust
the hint. Exposing them costs nothing but a schema — `agent.py`'s dispatch already resolves any
`context.py` function by name and passes `source=`, so **a primitive needs no wrapper**; add
its schema to `schemas.py::TOOL_SCHEMAS` and it is callable. They are single-question
instruments: one call, one timestamp, one measurement, so the prompt tells the agent to triage
with `describe_points` first and spend one of these only where a decision turns on a specific
number. **`describe_points` describes up to `DEFAULT_MAX_POINTS` (100) per call, ceiling
`MAX_POINTS_CEILING` (300)** — raised from 20 on 2026-08-20, which was the binding limit on
how much of a detector's output a run ever measured (§7.5). Cost is ~90 tokens/point and
**flat with batch size** (measured 20/100/263 points: 93/91/90), so measuring a detector's
whole output costs a fraction of one wrong deletion. It is agent-settable but capped:
`MAX_FLAGGED_DATETIMES` is 1000, so an uncapped call could return ~90k tokens in one result.
Truncation and clamping are both reported, never silent, and the message names the cap that
was APPLIED rather than the one requested
number. `noise_context` is the exception to "one timestamp": one call anywhere inside a busy
stretch characterises the whole stretch and returns its bounds.

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

- **The rise-vs-fall gradient ratio DOES work — but only at a 45-minute window, and the
  signal reverses if you widen it.** This entry previously said the ratio was useless and
  told you not to rediscover it. That was wrong, and the way it was wrong is worth keeping:
  the original measurement compared injected spikes against **storm peaks** at a **2 h**
  window and found 1.00 vs 0.97. Both choices hid the effect. Re-measured 2026-08-11 on the
  population that actually causes false positives — 31 **small flush events** (rows a run
  deleted that the labels call normal water, verified against the raw approved base) vs 14
  confirmed injected spikes — with `n_before`/`n_after` swept:

  | window | flush median | artifact median | separation |
  | --- | ---: | ---: | ---: |
  | 2 samples (30 min) | 0.58 | 0.99 | 0.56 |
  | **3 samples (45 min)** | **0.75** | **0.99** | **0.84** |
  | 4 samples (60 min) | 0.83 | 1.00 | 0.71 |
  | 6 samples (90 min) | 1.11 | 0.98 | 0.42 |
  | 8 samples (120 min) | 1.24 | 0.99 | 0.34 |
  | 12 samples (180 min) | 1.43 | 0.93 | 0.27 |

  (Separation is P(artifact ratio > flush ratio); 0.5 is a coin flip.) A flush rises in one
  sample and decays over 3–5, so at 45 minutes its fall is visibly gentler than its rise.
  Past ~90 minutes the decay is over, the window fills with flat surroundings, and the
  **ordering flips** — which is exactly what makes this look like noise if you only sample a
  few wide windows. `slope_context`'s default is therefore **3 samples**, and so is
  `describe_point`'s, so the compact `describe_points` row carries a usable ratio.
  `scratchpad/tune_slope_window.py` reproduces the sweep.
- **Width and robust_z cannot separate a small flush from an artifact** — this is why the
  ratio matters. Both are 1–3 samples wide, and the flush often has the *higher* z (measured:
  a flush at robust_z 21.8 against an artifact at 14.1). Width and recovery remain the right
  test for a **storm peak**, which is genuinely 7–25 samples wide; they are the wrong test for
  the small events that produce most of the false positives.
- **Recovery must be anchored at the excursion's onset, not at the point.** Mid-storm, the
  six hours before the *point* are already storm, so a point-anchored baseline reports an
  instant recovery for an event nowhere near over. `recovery_context(anchor='excursion')`
  is the default for this reason.
- **`reads_like` is a hint, not a verdict**, and it is deliberately conservative. Measured
  hit rates: spike 87/120 with 11 storm-peak false positives and 0 on normal rows; plateau
  exact; gap exact; **level_shift weak (6/9 onsets, 6 storm-peak FP)** — the same wall §9.1
  hit from a different direction, so treat that label as "look here". (The spike branch was
  since guarded by `noise_context` and now also emits `noisy-stretch` — see §7.4 for the
  re-measured rates.)
- **Level shift is measured as a duration, not a persistence ratio**, because §9 injects
  bounded 4–24 h windows rather than permanent steps; a "does it hold forever" test would
  score every injected shift as transient. (The window was 6–72 h when this was measured;
  shortened 2026-07-31 with the frequency cut, so shifts are now shorter than the durations
  the table above was calibrated on — expect the weak level_shift hit rate to be no better.)

### 7.4 `noise_context` — the denominator was wrong (2026-08-13)

**Every measurement in §7.3 scores one point against a *record-wide* scale.** So a point
reads as extreme whether its neighbours are flat or thrashing, and in a busy stretch the
spike detectors fire on dozens of points that each look extreme by that standard. Runs were
deleting them as dozens of separate sensor failures. They are one stretch, and the right
output is one decision about the stretch.

`noise_context` measures the **surroundings** rather than the point. Two numbers carry it:
`noise_ratio` (the window's robust first-difference scale ÷ the record-typical window's) and
`point_step_sigmas_local` (the point's own largest single-sample move ÷ that *local* scale).

**Measured** (`scratchpad/tune_noise_context.py`; 4 datasets across all three gauges,
injected labels as truth). 168 injected spikes a `flagUniLOF(thresh=1.5)` run found, 662
**false positives** from the same run — points it flagged that the labels call normal water,
i.e. the population that motivated this — and 1600 ordinary normal rows. Medians:

| metric | true spike | false positive | normal | separation |
| --- | ---: | ---: | ---: | ---: |
| `noise_ratio` | 1.2 | 11.3 | 1.0 | 0.91 (inverted) |
| `point_step_sigmas_local` | 17.1 | 1.6 | 1.1 | **0.87** |
| `point_step_sigmas_global` | 18.7 | 16.7 | 1.0 | 0.53 |
| `turning_fraction` | 0.55 | 0.27 | 0.55 | 0.77 |

**Read the third row first.** The measurement the agent already had — `slope_context`'s
`delta_before_sigmas`, scaled by the record step sigma — reads 18.7 on a real spike and 16.7
on a false positive. It cannot tell them apart at all. Rescaling the *same* move against the
local neighbourhood gives 17.1 vs 1.6. The information was always there; the denominator was
wrong. Rules, as what they buy and cost:

| spare the point when… | FPs spared | spikes lost |
| --- | ---: | ---: |
| `point_step_sigmas_local < 5` | 83.1% | 16.7% |
| `noise_ratio > 5` | 70.7% | 7.1% |
| **`noise_ratio > 3` and `step_sigmas_local < 5`** | **76.7%** | **3.0%** |

- **The window is ±90 min and that is the peak — do not widen it.** Swept, the combined rule
  spares 83.7 / 76.7 / 72.4 / 61.0% of FPs at 6.5 / 3.0 / 1.2 / 2.4% of true spikes at
  ±1 / 1.5 / 2 / 3 h. Widen it and the window fills with calm surroundings, the local scale
  collapses back toward the record scale, and the measurement degrades into the global one it
  exists to replace. It is therefore **not** passed `describe_point`'s `window`.
- **`turning_fraction` splits the two ways a stretch can be busy, and they need different
  decisions.** It is the share of samples where the series reverses direction: white noise
  reverses about half the time (a thrashing sensor — and also a calm baseline, both ≈0.55),
  a storm limb climbing steadily almost never does (≈0.27). High `noise_ratio` + high turning
  = a noisy **sensor**; high `noise_ratio` + low turning = water genuinely moving fast. Both
  mean "do not delete these one at a time"; only the first is a data-quality problem at all.
  In this FP population the low turning fraction says most of them are **storm limbs**.
- **Residual-based noise measures were tried and dropped.** Scatter around a short centred
  rolling median is the textbook noise estimate and looks better on paper, but on quantised
  turbidity the residual MAD collapses to exactly 0 in a large share of windows and the ratio
  explodes — the §7.1 trap in a new place. First differences do not have this failure mode.
- **`reads_like` gained a `noisy-stretch` label**, returned instead of `spike` when the guard
  trips. Measured (`scratchpad/probe_noise_reads_like.py`): on detector false positives it
  drops **34.6%** of the wrong `spike` hints (179 → 117 of 621), on true injected spikes it
  costs **3.0%** (164 → 159 of 168), and on unflagged normal rows it changes **nothing**.
  This supersedes the "spike 87/120, 11 storm-peak FP" figure above for the spike branch.
- **RE-TUNED FOR THE 5-MIN BASES (2026-08-20), and the old numbers fire on nothing.** The
  rule above (`noise_ratio > 3 and step_sigmas_local < 5`) spares **0 of 104** surviving
  false positives on 01467200_l1: it was fitted on the retired 15-min gauges, whose FP
  population was storm limbs at median `noise_ratio` **11.3**, and the 5-min FP population
  sits at median **2.2**. The separator still holds — FP median `noise_ratio` 2.2 vs true
  spike 0.9, FP median `step_sigmas_local` 8.4 vs 14.9 — so only the thresholds moved.
  Refitted over 2,892 FPs / 392 true spikes pooled from 02054550 l1/l2 and 040851385 l1/l2,
  **never the gauge reported on** (`scratchpad/tune_noise_spare_rule.py`), maximising FPs
  spared subject to losing ≤5% of true spikes:

  | rule | FPs spared | spikes lost |
  | --- | ---: | ---: |
  | `noise_ratio > 2.0 and step_sigmas_local < 8` | 60.2% | 4.1% |

  Live as `context.ELEVATED_NOISE_RATIO` / `LOCAL_STEP_UNREMARKABLE`. **Mind the transfer
  gap**: 60.2% sparing where fitted, **33%** held out on 01467200_l1 (34 of 104 FPs, and
  **0** of 57 true spikes). The direction carries; the magnitude does not fully. Fitting
  these on the gauge you then score is fitting to the test set — don't.
- **`inspect_dataset` returns a `noise_profile`** (2026-08-20): which calendar stretches of
  the record are noisier than its own typical window, via `context.noise_profile`. **Rolling,
  not fixed blocks** — a fixed grid dilutes a stretch that straddles a boundary and can push
  it under threshold in both halves. Contiguous runs are **bridged at the window width**,
  because a rolling threshold flickers across one storm and reports it as hundreds of
  fragments (796 → 419 episodes on 01467200_l1, widest 3.7 days). ~116 ms and ~800 tokens
  for a 210k-row series. The §7.4 warning that `rolling().apply(mad)` costs seconds does not
  apply: this runs once per run and uses pandas' native rolling median, not `.apply`.
  It is a *returned measurement* rather than a prompt instruction on purpose — the agent
  already had per-point `noise_ratio` in every `describe_points` row and did not act on it.
- **`noise_ratio` and `step_sigmas_local` ride in every `describe_points` row**, because this
  failure is only visible *across* a cluster — no single row shows it. `episode_start` /
  `episode_end` bound the elevated stretch so the agent can write one §5 decision span over
  it instead of one per flagged row.
- `_block_noise` computes the reference distribution by reshaping the whole record's diffs
  into non-overlapping blocks and taking a `nanmedian` along an axis. The obvious
  `rolling().apply(mad)` is O(n·w) in Python and takes seconds per call on a two-year 15-min
  series; this is ~10 ms, which is what makes 100 points in one `describe_points` call viable.

### 7.5 Spike precision: what actually moved it (2026-08-20)

Spike precision was **0.227** — the run deleted 194 real readings for every 173 correct
ones. Two causes, fixed and **measured separately** so each is attributable. Recall never
moved (0.905 throughout) and true positives held at 173, so both fixes are pure precision.

| | baseline | A: tool | B: decision |
| --- | ---: | ---: | ---: |
| spike precision | 0.227 | 0.354 | **0.460** |
| spike recall | 0.905 | 0.905 | **0.905** |
| false positives | 194 | 104 | **67** |
| macro-F1 | 0.577 | 0.614 | **0.639** |

**A — the tool limit.** `describe_points` capped at 20 points/call, so the run measured
**45 of 10,463 decided rows (0.43%)** and deleted 367 spikes having inspected 33. Raising
the default to 100 took measurement to **263 of 263 flagged points in two calls**, and
removed 90 FPs. The diagnostic that proved the mechanism: after the fix, **100% of the
surviving FPs had been measured** (104/104) versus 33% of the true positives — the "deleted
a point it never looked at" failure was gone, and what remained was misjudgement.

**B — the decision layer.** Three changes, of which the guard re-tune (§7.4) carries most
of the weight: it relabels 34 FPs `noisy-stretch` and **0** true spikes. Plus the §7.4
`noise_profile` at step 0, and `difficulty` made required (§5).

**What is NOT the cause, having been checked:** `level_shift` scores 0/117 in every run and
is untouched by either fix — it is the single largest macro-F1 drag (0.639 would be ~0.85
without that zero) and remains open. And the surviving 67 FPs are not a coverage problem:
they were measured, cited real numbers, and were still wrong.

### 7.6 `flag_jumps.thresh` — the unit was wrong, not just the number (2026-08-20)

**A run set `thresh=3.0` FNU on 01467200, got 2,865 flags spread evenly over two years,
correctly read them as storm limbs, and blanket-`keep`ed every one** — so a dataset with
three injected level shifts scored zero recall, and the flag log recorded the agent
concluding there were no level shifts at all. Its stated reasoning was that 3.0 FNU "is
only 0.57 std for this gauge (mean=7.72, std=5.23)". That reasoning is the bug, twice over.

- **`std` is not the series' spread on a storm-driven river.** 02054550 reports std **28.1**
  FNU against a median of **1.9** and a robust sigma of **1.3**, because a handful of storm
  peaks reach 800. A threshold set at "half a std" there sits at twenty times the water's
  ordinary movement. `inspect_dataset` now reports `median` and `robust_sigma` alongside
  `mean`/`std` for exactly this, and §2 of the prompt tells the agent to scale from the
  robust pair.
- **But no static summary stat can size this parameter at all** — including the robust one.
  `flagJumps` thresholds `|mean(previous window) − mean(next window)|`, and at a 3 h window
  the p99 of that statistic is **2.9 × `robust_sigma`** on 01467200 and **1.4 ×** on
  040851385 — but **33 ×** on 02054550. The gauge needing the *highest*
  threshold is the one with the *lowest* median: Roanoke sits at 1.9 FNU and swings hundreds
  in storms. A multiplier tuned on any one gauge is wrong by an order of magnitude on another.
  (`scratchpad/tune_jumps.py`, `tune_jumps_scale.py`.)

**The fix is to measure the statistic flagJumps actually thresholds and quote a quantile of
it**, because that is the only normaliser on the same axis as the decision.
`context.jump_scale` reproduces the statistic (`_getChangePoints` with
`stat_func=|mean(x)−mean(y)|`; backward window closed right, forward open left) and reports
its p50/p90/p99/p99.9 per window, plus `recommended_thresh` = **p99 at a 6 h window**.
`inspect_dataset` returns the block, so the agent reads the number rather than deriving it.

**What the quantile buys, measured** (`scratchpad/tune_jumps_quantile.py`, `tune_jumps_knee.py`;
all nine 5-min datasets, event-level recall against the injected `level_shift` labels):

| rule | candidates per dataset | events found |
| --- | ---: | ---: |
| `thresh = 2 FNU` (the old suggestion) | 704 – 4,054 | agent gave up; 0 |
| `thresh = 1 × robust_sigma`, 6 h | 670 – 6,674 | 22/33 |
| `thresh = 3 × robust_sigma`, 6 h | 34 – 3,754 | 6/33 |
| **`thresh = p99` of the 6 h jump statistic** | **131 – 162** | **14/33** |
| `thresh = p99.5`, 6 h | 45 – 80 | 8/33 |

**Read the candidate column, not the recall column.** The quantile rule's contribution is
that the candidate count is *stable across gauges* — 131–162 everywhere, against a 12×
spread in the raw threshold (6.2 / 13.8 / 75.7 FNU) — which is what makes the output
triageable at all. Every fixed multiple of a static stat gives a count that varies by two
orders of magnitude between gauges, and on at least one of them it is always unusable.
6 h is the knee: 3 h and 12 h both cost recall at the same candidate count.

**Recall is capped by the injection, not by the tuning, and the cap is severe.** §9 sizes an
injected shift at 2–5 × the *local* scale, itself clamped to `[0.1, 3] × global MAD` — so a
shift placed in a calm week is a fraction of the global scale. Realised magnitudes:

| gauge | injected shift | p99 of its 6 h jump statistic | best events found |
| --- | ---: | ---: | ---: |
| 01467200 Delaware | 3.4 – 16.4 FNU | 6.2 | **13/14** |
| 040851385 Fox | 2.3 – 9.8 FNU | 13.8 | 1/11 (5/11 at 3 h, p98) |
| 02054550 Roanoke | 0.4 – 1.5 FNU | 75.7 | **0/8 at every setting swept** |

On Roanoke the injected shifts are ~1/50 of the record's ordinary window-to-window movement.
No threshold can find them, and one that flagged them would flag tens of thousands of storm
rows first. **That is an injection-design problem, not a detector-tuning one**, and it is the
same wall §9.1 hit from the review side — do not spend another pass tuning `thresh` against
those eight events. If level_shift recall is to improve further, the lever is sizing the
injected magnitude against the series' *jump* scale rather than its local value scale.

**Expect ~150 candidates and at most a handful of real ones even when correctly tuned.** That
is this detector's nature on storm-driven turbidity (§9.1: across 78 reviewed level_shift
candidates the human rejected every one). The prompt now says so, and tells the agent to
triage with `describe_points` / `level_shift_context` — `step_sharpness` is the
discriminator — rather than treating the count as a tuning signal. What it must *not* do is
conclude "there are no level shifts here" from an over-flagged run: that is the parameter
talking, and it is what happened.

**`flag_jumps` now rejects `thresh <= 0` and a missing `window`** rather than defaulting
(§13). The wrapper's old `thresh=0.0` default would have flagged every change point in the
record, and the schema's `default: 2.0` is what the model reached for.

---

## 8. The agent loop (ReAct)

1. **Inspect** — always call `inspect_dataset` first.
2. **Reason** — read the summary; decide which checks to run, in what order, with what params.
3. **Act** — call tools one at a time; use each result to decide the next call. Respect the
   25-call cap; if reached, stop and summarise.
4. **Summarise** — call `get_flag_summary`, then `export_clean_data` **with `decisions`
   covering every flagged segment** (§5). Each decision states **two separate things**: a
   `verdict` (is this segment anomalous, and of which type — the answer §10 scores) and an
   `action` (what to do with the values). This is the step where the run's reasoning becomes
   the record; without it the flag log says the agent concluded nothing. The result reports
   `n_undecided` and any span that matched no flagged row — budget a spare call to fix it.
5. **Report** — write a plain-language report: what was found (by type), what was done, and
   any caveats.

The system prompt (in `src/agent.py`, versioned in git — commit changes with a note) states
the agent's role, the golden rules, the tool list, and that it must justify each action.
Use the Anthropic Messages API multi-turn tool-use pattern (assistant emits tool_use →
we run the tool → we return tool_result → loop).

**The prompt describes PHASES, not a script** (v0.5, 2026-08-10). It used to lay out STEP 1–8
with a fixed detector order and required a full plan up front — which, given §0 (the agent
cannot see the data), is a guess dressed up as a decision, and it produced runs that executed
their plan instead of reading their results. Only three orderings are actually forced, and
each is technical rather than stylistic: `inspect_dataset` first, `flag_range` before the
spike detectors (so impossible values do not distort the neighbourhood statistics they
depend on), and `flag_nan` before `impute_rolling` (so `max_gap` comes from the gap
distribution). Everything else is a menu the agent sequences from evidence, and it is told
explicitly that a detector it has no reason to expect anything from is a wasted call.

**Cost: the levers are prompt caching and payload size, not thinking or a cheaper model.**
Measured on the 2026-08-10 run (15 steps, $4.91):

- **88% of the bill was input tokens** (1.45M in / 38k out) and `cache_read_input_tokens` was
  **0 on every step** — the whole conversation was re-sent at full price 15 times. Thinking
  was 12% of the bill, so turning it off saves little and costs the trace `visualize_log.py`
  renders. **Prompt caching is now on** via top-level `cache_control={"type": "ephemeral"}`,
  which auto-places the breakpoint on the last cacheable block — the standard multi-turn
  pattern. Verified: `scratchpad/probe_prompt_cache.py` shows 15,546 tokens (tools + system)
  written then read back. Cache reads bill at ~0.1×, writes at ~1.25×, and `RunSummary` now
  tracks all three and **warns when a multi-step run records zero cache reads**.
- **The cache TTL is `1h`, not the 5-minute default** (2026-08-20). A single step here can
  spend minutes generating — one emitted 25,179 output tokens — and when a step outlives the
  TTL the next request re-writes the WHOLE prefix at 1.25x instead of reading it at 0.1x.
  Measured on the 2026-08-19 run: step 9 wrote 122,796 tokens with **zero** cache reads,
  $0.46 of a $2.21 run. A 1h write costs 2x rather than 1.25x, which is the cheaper trade
  the moment one such miss is avoided; the next run had **0** misses with two inter-call
  gaps over 300s. `ttl` is accepted on the TOP-LEVEL `cache_control` (the docs show it on
  content blocks) — `scratchpad/probe_cache_ttl.py` confirms before a run depends on it.
- **Every logged event carries a `timestamp`** (2026-08-20). §2 required it and it was
  missing, which is why the miss above could not be told from a TTL expiry after the fact.
- **This only holds while the prefix is byte-stable.** `tools` renders first and `system`
  second, so interpolating a timestamp, run id or dataset name into `SYSTEM_PROMPT`, or
  varying `TOOL_SCHEMAS` between calls, silently invalidates everything. Re-run the probe
  after touching either.
- **34% of the payload was raw `flagged_datetimes`** — 8,900 timestamps. Now sampled (§5).
- **A cheaper model is not available for this workload.** Haiku 4.5's context window is 200K;
  that run's final request was **263,345 input tokens**. It would not fit, quite apart from
  the §2 golden rule pinning `claude-sonnet-4-6` (1M context).

**A run is 15+ sequential API calls, so one failure anywhere throws away everything
before it.** Three layers guard that, and they cover different failures — do not collapse
them:

- `max_retries=8` on the client (above the SDK default of 2), for a request that fails
  *before* the response starts. Measured 2026-08-13: an `overloaded_error` at step 15 of
  15, one call before `export_clean_data`, discarded a complete run.
- **`_stream_message` retries a failure that happens DURING the stream**, because the SDK
  cannot: once bytes are arriving, `max_retries` no longer applies. Measured 2026-08-18 on
  01467200_l1 — an `httpx.ReadTimeout` while the long final export response was streaming
  ended the process with 14 completed steps unexported. Re-issuing is safe: the request is
  the whole conversation so far, so a retry re-asks rather than resuming a half-received
  answer.
- The `except` around the call catches **`httpx.HTTPError` as well as
  `anthropic.APIError`** and `break`s, so the run exports what it has. This is why the
  above mattered twice over: `httpx.ReadTimeout` is not an `anthropic.APIError`, so it
  escaped the handler written for exactly this situation and killed the process instead.

Note the limit: the §5 flag log is produced *by* `export_clean_data`, so a run that dies
before the agent calls it has no flag log to salvage, however gracefully it stops.

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
- **`max_tokens` caps thinking + response together**, so it was raised 4096 → 16000 →
  32000 → **64000** (2026-08-20). The EXPORT turn is the longest of the run — the agent
  plans one decision span per flagged segment and emits them in a single call — and at
  32k it spent the whole budget deliberating over ~177 spans and never emitted the call,
  ending a 15-step run with nothing exported. `claude-sonnet-4-6` accepts **128,000**
  output tokens and this client already streams (which large `max_tokens` requires), so
  the headroom is free; `max_tokens` is a ceiling, not a reservation. A run
  that truncates mid-report is the symptom of setting this too low.
- **Thinking blocks must be returned to the API unchanged** on the next turn. `agent.py`
  appends the whole `response.content`, which is already correct — do not "optimise" it into
  extracting just the text blocks, or the next request errors.

The model id is a §2 golden rule and lives in one place, `agent.py::MODEL`. It had drifted to
`claude-3-5-sonnet-20240620`, which predates adaptive thinking; that is fixed.

---

## 9. Data strategy

- **Real data:** USGS NWIS via the `dataretrieval` package (primary — cleanly scriptable);
  ECCC (secondary — may be a manual download). Target ~2 years at **5-min** sampling, 2–3
  gauges per variable. Start with **one variable** (turbidity or specific conductance).
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
- **Sampling is 5-minute** (2026-08-18). The bases were 15-min until then; the retired
  series and the nine datasets injected into them are kept under `data/legacy_15min/`
  (see its README). Cadence is not cosmetic: several §7 measurements are expressed in
  **samples** rather than time — `flag_spike_unilof`'s `n≈20` neighbourhood,
  `slope_context`'s 3-sample window — so at 5 minutes each of those spans a third of the
  time it used to. Every number in §7.2, §7.3 and §7.4 was measured on 15-min series and
  is now a **stale starting point**, not a tuned value.
- **The three default gauges** (all **100% approved, consistent 5-min, ≥95% complete, and
  calm** over 2023-07→2025-07; regime spread low→moderate):
  `02054550` Roanoke R at Salem, VA (Blue Ridge Appalachian headwater; **low**, median
  ~1.9 FNU; 96.0% complete; 0.31% base spikes); `01467200` Delaware R at Penn's Landing,
  Philadelphia, PA (Mid-Atlantic tidal urban estuary; **moderate**, ~6.5 FNU; 96.6%
  complete; 0.10% base spikes); `040851385` Fox R at Green Bay, WI (Great Lakes river
  mouth; **high**, ~11.3 FNU; 95.4% complete; 0.41% base spikes). All three hold a 5-min
  modal step in **every one of the 25 months** in the window (≥98.7% of steps), so
  re-gridding cannot manufacture the phantom gaps §9.1 warns about.
- **How they were chosen, and why the pool is this small.** `scratchpad/screen_5min_*.py`
  swept all 52 states' NWIS series catalogs: **1,737** turbidity (`63680`) instantaneous
  series nationwide, 700 with a ≥2-year record still active mid-2025. Cadence **cannot be
  read from the catalog** — `count_nu` is *days* of record for a unit-value series, not a
  sample count, so every series computes to a nonsense 1440-min step; it has to be measured
  from the timestamps. Doing so found **519 at 15-min and only 57 at ≤5 min**, of which
  **13** also clear approval + completeness + a 2-year span. That is the entire national
  pool, and it is why the spread is now low→moderate: **no ≤5-min gauge clears the gates
  with a median above ~12 FNU**, so the retired set's ~28 FNU high end has no 5-min
  counterpart. Base-spike share is measured identically for every candidate —
  `flagUniLOF(n=20, thresh=1.5)` over the re-gridded approved series — which is *not* the
  method behind the retired gauges' quoted "≤0.04%"; re-measured that way the retired three
  score 0.41 / 0.12 / 0.06%, so the new three sit inside the same band rather than being
  three times worse.
- Retired as 5-min candidates: `072632996` (Lk Maumelle AR) is the calmest series in the
  country at 0.034% and 98.4% complete, but **no USGS rain gauge within ~110 km covers our
  window** (every nearby one begins 2026-04), and rainfall is decision-relevant context
  (§9.4); `01585075` (Foster Branch MD, 11.6 FNU) and `01579550` (Susquehanna MD) are
  strong but would have put two of three gauges in the same Mid-Atlantic region.
  Earlier 15-min retirements, kept for the record: `02203603` / `02198955` (too **flashy** —
  0.18% and 0.91% real base spikes), `12340500` (Blackfoot, 35% missing), `06818000`
  (Missouri, 14% missing), `11501000` (Sprague, mostly provisional).
- **Synthetic injection:** inject all four types at recorded locations into the approved
  bases; save the labels (§5). Build **three contamination levels** with a **fixed random
  seed** for reproducibility.
- **What the level scales.** The `0.15% / 0.4% / 0.8%` knob applies to the **point-like types
  only** (spike, plateau, gap), where "percent of rows" is a natural unit. `level_shift` is
  driven by episode count instead (1 / 2 / 3), and its row-share is a reported consequence
  rather than a target — so a dataset's **total** anomalous share exceeds its headline level,
  partly via *natural* gaps carried in from the base. On the 5-min bases level 3 lands at
  5.1 / 4.5 / 5.6% on 02054550 / 01467200 / 040851385, and the natural-gap share
  (4.1 / 3.4 / 4.6%) now *dominates* the total on all three — the injected point budget is
  0.8%. (On the retired 15-min bases the same figures were 5.6 / 2.4 / 1.7%.)
  (These per-type rates were toned down in stages —
  `3 / 7 / 12` → `2 / 5 / 9` → `0.8 / 2 / 3.5` → the current `0.15 / 0.4 / 0.8`
  (2026-07-31) — to look like real, sparsely-anomalous records. At 3.5% the level-3 series
  was visibly speckled on a plot, which makes detection easier than the real problem.)
  Read per-type counts from the manifest/labels; never infer them from the level number.
- **Level 1 is thin by design — check `n_events` before trusting a per-type score.**
  The level is a share of *rows*, so tripling the sampling rate tripled the event counts at
  the same contamination: on the 5-min bases level 1 carries **~30 spike events, 1–2 plateau
  events, and ~4 injected gap events** across two years (it was ~11–13 / 1–2 / 3–4 at
  15-min). Spike is now comfortably scoreable at level 1; **plateau still is not** — it rests
  on one or two events, so a single miss swings its precision/recall to 0. Score plateau on
  levels 2–3, or pool levels. All nine datasets report `point_budget_met: true` and
  `types_missing: []`.
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
  95% per type per dataset. **Re-measured 2026-08-18** on the 5-min bases *and* on the retired
  15-min ones, with the identical metric, so the two are comparable
  (`scratchpad/compare_candidate_recall.py`; medians across the nine datasets in each set):

  | type | 15-min (retired) | 5-min (current) | verdict |
  | --- | ---: | ---: | --- |
  | spike | 100% | 100% | green (98.2–100% on every dataset) |
  | gap | 100% | 100% | green by construction (NaN mask, not a detector) |
  | plateau | 100% | 96.1% | **amber** — under 95% on 4/9 now, was 3/9 |
  | level_shift | 16.1% | 35.8% | **red**, and structurally so |

  **These failures are not caused by the move to 5-min data — they pre-date it.** The same
  assertions fail on the legacy 15-min datasets: **21 failing parametrisations there against
  18 now**, and level_shift's median recall more than doubled. So the cadence change slightly
  *improved* the picture; it did not break it. Do not spend time bisecting the 5-min switch
  looking for the cause.

  level_shift is red for the reason the level_shift bullet below gives, and no threshold fixes
  it: `flagJumps` marks a step's **edge** while the §5 label covers the **whole injected
  window**, so row-recall is bounded by (edge width / window length) and cannot approach 95%.
  Scoring it per *row* is measuring the wrong thing — an event-level or onset-tolerance metric
  is the real repair, and until that exists the assertion is a known-red reminder rather than a
  regression signal. Plateau's amber is thinner than it looks: level 1 carries only 1–2 plateau
  events (§9), so a single clipped event moves a dataset's recall by tens of percent.
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
- **Three page traps, all verified in a browser, all silent failures** — they apply to every
  page in `workbench/`, not just `review.py`. Plotly renders a `Date` object in the
  *viewer's* timezone, so an axis built from Dates printed hours away from the timestamps in
  the side panel and the CSV; the series is timezone-naive, so x values must be naive ISO
  **strings**. `gd.on(...)` does not exist until Plotly has plotted into that div —
  registering `plotly_click` at start-up throws and takes the rest of the init down with it,
  keyboard handlers included; wire it after the first draw. And **`DatetimeIndex.view("int64")`
  does not return nanoseconds** (2026-08-11): these CSVs parse to `datetime64[us]`, so the
  usual `// 1_000_000` yielded *seconds* and `spike_audit.py` plotted a two-year series
  entirely inside 1970. Use `index.as_unit("ms").astype("int64")`, which is explicit about
  the unit whatever the source resolution. The axis is the only place this shows, so it
  survives every test that checks the data rather than the picture. **A context window
  expressed in SAMPLES is the same class of bug** (2026-08-18): `CONTEXT_SAMPLES = 96`
  read "24 h" on the 15-min bases and became 8 h the moment the project moved to 5-min
  ones, so the detail plot silently stopped showing the surroundings a storm-vs-spike
  call depends on. Both audit pages now derive it from the series' own median step.
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

### 9.4 Precipitation (checked 2026-08-18 — available, not yet pulled)

Rain is the one piece of evidence that settles the hardest call the agent makes. §7.3
measured that a storm peak and a spike are separated by *width* and *recovery time*, and
§7.4 that a whole noisy stretch is one event rather than dozens — but all of that is inferred
from the turbidity series' own shape. A rain record is **exogenous**: it can confirm that
water genuinely moved, which no amount of looking at the sensor's own trace can.

**No 5-min turbidity gauge in the country records precipitation at the same site** — checked
against the national series catalog, **0 of 57**. So it has to come from a nearby station.
`scratchpad/find_precip.py` searches NWIS by bounding box for instantaneous `00045`
("precipitation, total, inches") series and ranks by great-circle distance;
`scratchpad/verify_precip.py` then **pulls the window and measures it**, which is not
optional — many gauges near our sites advertise a long period of record but only began
reporting instantaneous values in **2026-04**, well after our window. The catalog alone
would have reported those as available.

Verified over 2023-07-01 → 2025-07-01, all **100% approved**:

| Turbidity gauge | Nearest usable rain gauge | Dist | Step | Complete |
| --- | --- | ---: | ---: | ---: |
| `02054550` Roanoke, VA | `371520080015100` MET STN Hidden Valley, Roanoke | 7.0 km | **5-min** | 99.8% |
| `01467200` Delaware R, PA | `01473169` Valley Creek at Valley Forge | 31.1 km | 15-min | 99.8% |
| `040851385` Fox R, WI | `04085108` East River nr Greenleaf | 18.6 km | **5-min** | 98.7% |

- **Roanoke is the strong case**: three independent 5-min met stations within 11 km, all
  ≥99.5% complete, so a rain signal can be corroborated across stations rather than trusted
  from one bucket.
- **Delaware is the weak case** and the weakness is twofold. The nearest full-window gauge is
  31 km away across a metropolitan area, and the site is a **tidal** estuary whose turbidity
  is driven by tidal resuspension as much as by runoff — so rainfall is *less* diagnostic
  there than at the other two, not merely harder to source. The 19.9 km Bridgeboro NJ station
  is 6-min but starts 2024-10 and is provisional; it covers barely a third of the window.
- **A rain gauge kilometres away is a proxy, not a measurement.** Convective summer storms
  are patchy at exactly the scale of these separations, so "no rain recorded" is much weaker
  evidence than "rain recorded" — an asymmetry any tool built on this must respect rather
  than treating absence as refutation.

**Nothing has been pulled.** This is a documented availability check; wiring precipitation
into the pipeline would mean a `src/datasets/pull_precip.py`, a place for it in the §5
contracts, and a §7 context tool that reads it — none of which exist, and the last would
change what the agent is given rather than how it reasons. Note also that the "one variable
per run" golden rule (§2) governs the *QC target*; rain would be read-only context, not a
second variable to clean, but that reading should be confirmed before building on it.

---

## 10. Evaluation

- **Detection:** precision / recall / F1 per anomaly type + macro-F1 (scikit-learn), vs the
  injected labels, over the **four** types (spike, plateau, level_shift, gap). The §11
  macro-F1 ≥ 0.70 target refers to those four. Drift is not scored at all — it is removed
  (§9.2); it was already excluded from macro-F1 when it existed, so the headline number is
  unchanged by its removal, and there is no longer a drift correction-quality metric.
  **Scored on the agent's `verdict`, and typed by the agent's `anomaly_type`** (§5.1):
  flagging is candidate generation, not a claim. `evaluate.load_verdicts` returns `None` for a
  log written before the field existed, and scoring falls back to inferring the claim from
  `POSITIVE_ACTIONS` + `TOOL_TO_TYPE`; the printed table's `mode:` line always says which of
  the three paths ran (verdicts / actions / raw flags), so the two are never confused.
- **Imputation:** RMSE / MAE on filled values vs true values, compared to a
  linear-interpolation baseline.
- **Decision quality:** for each flagged segment, does the chosen action match the known
  correct action?
- **Fixed-pipeline baseline:** the same SaQC methods in a set order with default params and
  no agent reasoning. The agent should beat it; if not, that is a finding to explain.
- **Ablation:** disable tool subsets to show which matter.
- **Splitting:** first 80% of each series as context, last 20% as held-out test. Never shuffle.
- Also report on a small **manually labelled real** segment and discuss synthetic-vs-real gap.

### 10.1 Auditing one point: why did it get that verdict? (2026-08-18)

A macro-F1 says the run was wrong somewhere. **§1 also requires the tool to explain
itself per point**, and that is a different artefact from a score: a reader must be able
to click any timestamp — flagged or not — and get, in plain language, how the run
arrived at its answer there. `src/workbench/decision_audit.py` is that page and
`src/workbench/provenance.py` is the engine; `spike_audit.py` is the narrower,
spike-only ancestor of both.

**The verdict is the headline, not the action.** The page used to show only what was
*done* to a value, which cannot express the §5.1 split — a gap correctly identified and
left unfilled (`anomaly` + `keep`) read identically to real water the agent rejected
(`normal` + `keep`). Both the header and the panel now carry `verdict`/`anomaly_type`.

**Four situations, deliberately worded differently** — the failure mode this replaces is
all four rendering as a blank or as the same blank-ish row:

| the point was… | what the page says |
| --- | --- |
| flagged, called an anomaly | which detector fired, at which step, **with which parameters**; which `describe_point(s)` call measured it and the numbers it returned; which decision span claimed it; the agent's own words |
| flagged, called normal | the same trail, ending in the agent rejecting its own detector — §6 asks for exactly this, so it must read as a decision, not an omission |
| flagged, undecided | no span covered it; no claim in either direction |
| **never flagged** | a **roll-call**: every detector that ran, with its parameters, and the statement that none of them fired — so a detector miss is visibly upstream of the agent rather than an empty panel |

- **The roll-call is the reason `provenance.py` exists.** A flag log only holds flagged
  rows, so "no entry" silently covered both ordinary water and a labelled anomaly every
  detector walked past. Naming the detectors that ran and stayed silent separates them.
- **A blanket says so.** Where `rationale_source` is `blanket`, the panel states the
  span's reach as a share of all flagged rows and that the point was swept up rather than
  assessed — on the 2026-08-13 run that is **364 of 703 rows**, so without it most points
  would show a paragraph of reasoning that was never about them.
- **"Never measured" is reported as a different failure from "judged wrong."** No
  `describe_point`/`describe_points`/context call covering the timestamp means the
  decision was made without looking, which is a coverage problem, not a judgement one.
- The narrative is generated in **Python, not JS**, so it is testable
  (`tests/test_provenance.py`); the page only escapes it and renders `**bold**`. Roles are
  emitted once per point as an array aligned with a single global call table — repeating
  the 15-step trace per clickable point multiplied the page size for no added information.
- Every field it reads degrades to a weaker story rather than a traceback when a log
  predates it (`verdict` 2026-08-13, `decided_by` 2026-08-18).

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
