# Agentic Water-Quality Data Quality Control Tool

A standalone **agentic data quality control (QC) tool** for continuous water-quality
sensor time series. Physical sensors drift, foul, stick, spike, and drop out. This tool
uses an LLM (Anthropic Claude) as a reasoning engine: given an uploaded time series it
**inspects** the data, **adaptively selects and runs** QC operations from the
[SaQC](https://rdm-software.pages.ufz.de/saqc/) library, **decides** what to do with each
problem segment (delete / correct / keep), **imputes** gaps where appropriate, and returns
a **cleaned dataset plus a plain-language report and a machine-readable flag log**.

This is a research prototype. Quality control is more than anomaly detection — detection is
one stage; deciding, correcting, and documenting matter equally.

> See [`CLAUDE.md`](./CLAUDE.md) for the full project brief, constraints, and build plan.
> That file is the single source of truth; this README is the quick-start.

---

## Requirements

- Python **3.11+**
- An Anthropic API key

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY=...
```

## Verify the install

```bash
pytest        # zero tests is OK at this stage
python -c "import saqc; print(saqc.__version__)"   # should print 2.8.0
```

### Troubleshooting

**`ModuleNotFoundError: No module named '_tkinter'` when importing `saqc`.**
SaQC 2.8 imports `tkinter` at module load. Homebrew's Python 3.13 ships without the
Tk C-extension, so `import saqc` fails until it is installed separately:

```bash
brew install python-tk@3.13     # match your Homebrew Python's minor version
```

Non-Homebrew Python builds (python.org installer, conda) already bundle Tk.

**Prefer Python 3.11 or 3.12** (not 3.14). On 3.14, `pip install -r requirements.txt`
may fail building `scipy` (pulled in by `saqc`). For pull + inspect only you can install
a smaller set instead:

```bash
pip install pandas numpy dataretrieval pytest
```

---

## Pull raw USGS data

No Anthropic API key needed. Downloads **approved-only** turbidity (`63680`) for the default
3 gauges into `data/turbidity/approved/` (gitignored). Approved USGS data has already had
fouling/drift corrections applied (TM 1-D3), so it is clean apart from gaps and serves as
the injection base directly (CLAUDE.md §9):

```bash
python -m src.datasets.pull_usgs --dry-run       # show what would be downloaded
python -m src.datasets.pull_usgs                 # approved-only CSVs -> data/turbidity/approved/
python -m src.datasets.pull_usgs --keep-unapproved  # provisional too -> data/turbidity/provisional/
```

`data/turbidity/` is partitioned by approval status, and the split matters: **only
`data/turbidity/approved/` is globbed as an injection base**, so a provisional pull can never be
mistaken for a clean base. `--keep-unapproved` redirects the default output directory
accordingly; an explicit `--outdir` always wins.

Each raw CSV has: `datetime`, `value` (turbidity, FNU), `qualifier` (only approved codes —
those starting `A` — are kept by default; `P`/blank rows are dropped and become gaps).

---

## Inspect + validate CSVs

`src/inspect_data.py` loads a series, enforces the `datetime` / `value` contract
(CLAUDE.md §5), and prints a summary (rows, time range, frequency, NaN count/%,
min/max/mean/std).

```bash
# Check the series contract (datetime + value; extras like qualifier are kept)
PYTHONPATH=. python -m src.inspect_data validate data/turbidity/approved/02054550_turbidity_63680.csv

# Print a human-readable summary
PYTHONPATH=. python -m src.inspect_data summarise data/turbidity/approved/02054550_turbidity_63680.csv

# Same summary as JSON; --reindex turns missing timestamps into NaN rows
PYTHONPATH=. python -m src.inspect_data summarise data/turbidity/approved/02054550_turbidity_63680.csv --json
PYTHONPATH=. python -m src.inspect_data summarise data/turbidity/approved/02054550_turbidity_63680.csv --reindex

# Validate an injected labels file (when those exist)
PYTHONPATH=. python -m src.inspect_data validate-labels data/turbidity/injected/<gauge>/l<level>/<name>_labels.csv
```

Run the unit tests:

```bash
PYTHONPATH=. pytest tests/test_inspect_data.py -v
```

---

## Phase 2: Tools

In Phase 2, we build the toolkit the LLM agent calls at runtime.

- **Wrappers** (`src/agent_tools/wrappers.py`): each of the 11 QC functions wraps a SaQC 2.8
  method and returns a JSON-serialisable result dict (CLAUDE.md §5).
- **Schemas** (`src/agent_tools/schemas.py`): Anthropic Messages API tool-use schemas for every
  wrapper — name, description, parameter types, and valid ranges sourced from the
  param-sweep results (CLAUDE.md §7.2). Pass `TOOL_SCHEMAS` directly to the API `tools=`
  argument.
- **Tests** (`tests/test_tools.py`, `tests/test_schemas.py`): `test_tools.py` requires SaQC
  (Python 3.11/3.12 venv); `test_schemas.py` has no external dependencies and runs in any
  environment:

```bash
PYTHONPATH=. pytest tests/test_schemas.py -v   # no SaQC needed
PYTHONPATH=. pytest tests/test_tools.py -v     # requires saqc==2.8 installed
```

---

## Phase 3: Agent (CLI)

The agent runs an autonomous ReAct loop on a dataset, deciding which tools to call, inspecting results, and deciding whether to keep or delete data. It tracks token usage and outputs the final cleaned CSV, decision flags, and a text report.

```bash
# Run the agent on a dataset (requires ANTHROPIC_API_KEY in .env)
PYTHONPATH=. python -m src.agent data/turbidity/injected/02054550/l1/02054550_l1.csv

# View the tests for the agent loop and token tracking
PYTHONPATH=. pytest tests/test_agent.py -v
```

---

## Phase 4: Evaluation

We evaluate the agent's performance by scoring its anomaly detection (F1) and gap imputation (RMSE/MAE) against synthetic labels.

```bash
# Score a specific agent run's log against the dataset
PYTHONPATH=. python -m src.evaluate data/turbidity/injected/02054550/l1/02054550_l1.csv \
    --log logs/run_20260801_131008.jsonl \
    --decisions data/turbidity/injected/02054550/l1/02054550_l1_flags.json \
    --clean data/turbidity/injected/02054550/l1/02054550_l1_clean.csv

# Run the fixed-pipeline "dumb" SaQC baseline for comparison
PYTHONPATH=. python -m src.evaluate data/turbidity/injected/02054550/l1/02054550_l1.csv --baseline

# Run an ablation study (disabling specific tools)
PYTHONPATH=. python -m src.workbench.ablation data/turbidity/injected/02054550/l1/02054550_l1.csv \
    --disable flag_spike_unilof impute_rolling
```

---

## Phase 5: Streamlit UI

Run the interactive web application to perform data QC without writing any code:

```bash
PYTHONPATH=. streamlit run app/streamlit_app.py
```

This will launch a browser window where you can:
1. Upload a CSV time series
2. Preview the summary statistics
3. Run the Agent (using your API key or checking "Mock Mode" if you don't have one)
4. View an interactive Plotly visualization of flagged anomalies
5. Download the clean dataset and the JSON flag log


---

## Project layout

```
.
├── CLAUDE.md                   # durable project brief (read this first)
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example                # ANTHROPIC_API_KEY=
├── data/
│   ├── raw/                    # downloaded USGS series (gitignored)
│   │   ├── approved/           #   APPROVED series = the injection bases (§9)
│   │   └── provisional/        #   unapproved pulls, for auditing only (§9.1)
│   └── injected/               # synthetic datasets, filed by gauge then level
│       └── <gauge>/l<level>/   #   <name>.csv + <name>_labels.csv + <name>_manifest.json
├── src/                        # grouped by audience (CLAUDE.md §4)
│   ├── inspect_data.py         # load, validate contracts, summarise — shared foundation
│   ├── evaluate.py             # metrics, fixed-pipeline baseline, ablation
│   ├── agent.py                # ReAct loop + API logger
│   ├── datasets/               # writes everything under data/
│   │   ├── pull_usgs.py        # download APPROVED turbidity from USGS NWIS
│   │   ├── pull_comparison.py  # one series in BOTH approval states (§9.3)
│   │   └── inject.py           # synthetic anomaly injection (4 types, 3 levels, seeded)
│   ├── agent_tools/            # AGENT-facing: the §7 tool inventory
│   │   ├── wrappers.py         # SaQC-wrapping tool functions (§5 result dict)
│   │   └── schemas.py          # Anthropic Messages API tool schemas (§7 + §7.2)
│   └── workbench/              # HUMAN-facing: CLIs, HTML pages, plots
│       ├── visualize.py        # interactive raw-series explorer
│       ├── visualize_injected.py  # plot injected datasets with anomaly labels
│       ├── param_sweep.py      # sweep one param, score vs labels (§7.2)
│       ├── candidates.py       # propose candidate anomalies for human review
│       ├── review.py           # keyboard-driven HTML review + label merge
│       └── ablation.py         # run agent with disabled tools for ablation studies
├── app/
│   └── streamlit_app.py        # Streamlit UI
├── scratchpad/                 # one-off probe/tune scripts (not imported)
├── tests/
│   ├── test_inspect_data.py
│   ├── test_tools.py           # requires saqc==2.8
│   ├── test_schemas.py         # no external deps — runs in any Python
│   ├── test_inject.py
│   ├── test_pull_usgs.py
│   ├── test_pull_comparison.py
│   ├── test_candidates.py
│   ├── test_visualize_injected.py
│   ├── test_evaluate.py
│   └── test_agent.py           # ReAct loop + tool dispatch tests
└── logs/                       # JSONL API logs (gitignored)
```

## Build status

The project is built phase by phase:

- [x] Phase 0 — Scaffold
- [x] Phase 1 — Data + injection
- [x] Phase 2 — Tools
- [ ] Phase 3 — Agent (CLI)
- [x] Phase 4 — Evaluation
- [x] Phase 5 — UI
- [x] Phase 6 — Polish

