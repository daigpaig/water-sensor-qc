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
│   ├── raw/              # downloaded USGS APPROVED series = the clean bases (gitignored)
│   ├── injected/         # synthetic datasets + label files (injected straight from raw)
│   └── review/           # candidate proposals (gitignored) + reviewed labels (§9.1)
├── src/
│   ├── pull_usgs.py      # pull approved-only turbidity from NWIS -> data/raw (§9)
│   ├── inspect_data.py   # load + summarise a series
│   ├── inject.py         # synthetic anomaly injection (5 types, 3 levels, seeded)
│   ├── evaluate.py       # metrics, fixed-pipeline baseline, ablation
│   ├── agent.py          # ReAct loop + API logger
│   └── tools/
│       ├── schemas.py    # JSON tool schemas for the Messages API
│       ├── wrappers.py   # SaQC-wrapping tool functions
│       ├── visualize.py  # series/overview plots
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
│   ├── tune_candidates.py     # picks the §9.1 proposal thresholds
│   └── tune_zscore.py         # flagZScore min_residuals on quantised data (§7.1)
└── logs/                 # JSONL API logs (gitignored)
```

---

## 5. Data contracts (single source of truth — keep code consistent with this)

**Injected dataset CSV** (`data/injected/<name>.csv`)

- `datetime` (ISO 8601), `value` (float; may be NaN for gaps).

**Ground-truth labels** (`data/injected/<name>_labels.csv`, row-aligned by `datetime`)

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
`min_periods=0`). There is no `correct_drift` wrapper — drift is removed (§9.2).

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

**Index requirements.** The index **must be monotonic** — unsorted raises
`ValueError: index values must be monotonic`, so sort on load. An **irregular** index is
accepted by every method we use, including offset-string windows. Note that SaQC silently
accepts two things we should reject ourselves in `inspect_dataset`: a non-datetime
`RangeIndex`, and **duplicate timestamps**.

### 7.2 Practical parameter ranges (measured via `param_sweep`)

Swept against the injected labels with `src/tools/param_sweep.py` across all three gauges
(median 8 / 13 / 28 FNU) and levels 1–3. Ranges are wide because the optimum shifts with
**turbidity scale** and **anomaly density**. These are starting ranges + directional rules
for the agent (§8), not hard bounds — re-run the sweep if the data changes. Overriding
principle: **raise a sensitivity threshold when the base is spiky/variable or anomalies are
sparse (favor precision); lower it when anomalies are dense or the base is calm (favor
recall).**

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
- **Approved data is the base — but "approved" ≠ "no spikes".** Record processing (TM 1-D3,
  §9.2) applies fouling and calibration-drift corrections and deletes *clearly erroneous*
  data, but it **keeps real turbidity spikes** — a storm first-flush or resuspension event is
  genuine signal, not an error. So a *flashy* river's approved record is still full of sharp
  excursions, and those unlabelled base spikes would score as **false positives** against the
  injected labels (§9.1). We therefore require the base to be **approved AND calm** — a
  baseline with almost no real spikes — rather than assuming approval alone makes it clean.
  `pull_usgs.py` keeps approved-only rows (drops `P`/blank, which become gaps) and writes to
  `data/raw`; `inject.py` reads `data/raw` directly. There is no `data/clean/` folder and no
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
- **What the level scales.** The `0.8% / 2% / 3.5%` knob applies to the **point-like types
  only** (spike, plateau, gap), where "percent of rows" is a natural unit. `level_shift` is
  driven by episode count instead (1 / 2 / 3), and its row-share is a reported consequence
  rather than a target — so a dataset's **total** anomalous share exceeds its headline level,
  partly via *natural* gaps carried in from the base. Level 3 now lands at 8.7 / 5.5 / 5.0%
  on 03447687 / 02198840 / 08041770 (natural-gap share is only 0.6–4.6%). (These per-type
  rates were deliberately toned down in stages — from `3 / 7 / 12` to `2 / 5 / 9` to the
  current `0.8 / 2 / 3.5` — to look like real, sparsely-anomalous records.) Read per-type
  counts from the manifest/labels; never infer them from the level number.
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
file says nothing. `src/tools/candidates.py` + `src/tools/review.py` exist to check that
assumption:

```
python -m src.tools.review detect data/raw/<gauge>.csv        # propose + open the page
python -m src.tools.review merge  data/raw/<gauge>.csv <decisions.csv>
```

- **Proposal is tuned for recall, not precision.** Detectors run at deliberately sensitive
  settings; false positives are expected and are what the review step removes. Every
  threshold in data units is derived from the series' own robust (MAD) scale, so the
  defaults transfer across gauges spanning 0–40 and 0–1000 NTU. Defaults were picked in
  `scratchpad/tune_candidates.py` to land in the tens of segments per type, because a
  detector that proposes 800 segments cannot be reviewed by a human at all.
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
| `inject_drift`, `_place_drift_episodes`, `_n_drift_episodes`, `_inject_drift_and_maintenance`, `MaintenanceEvent`, `DRIFT_*` / `MAINTENANCE_*` constants, `ContaminationLevel.maintenance_interval_days`, `InjectionResult.maintenance` | `src/inject.py` |
| `_detect_drift` heuristic proposer, `drift_*` config fields, the `drift` type/colour | `src/tools/candidates.py` |
| the `drift` review category and `--drift-min-days` | `src/tools/review.py` |
| the `correct_drift` ToolSpec, `MAINT_FIELD`, `maintenance_variable`, the whole `correct` tool kind | `src/tools/param_sweep.py` |
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
