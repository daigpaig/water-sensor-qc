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

No Anthropic API key needed. Downloads turbidity (`63680`) for the default 3 gauges into
`data/raw/` (gitignored):

```bash
python -m src.pull_usgs --dry-run    # show what would be downloaded
python -m src.pull_usgs              # write CSVs under data/raw/
```

Each raw CSV has: `datetime`, `value` (turbidity, FNU), `qualifier` (e.g. `A` = approved;
approved ≠ clean).

---

## Inspect + validate CSVs

`src/inspect_data.py` loads a series, enforces the `datetime` / `value` contract
(CLAUDE.md §5), and prints a summary (rows, time range, frequency, NaN count/%,
min/max/mean/std).

```bash
# Check the series contract (datetime + value; extras like qualifier are kept)
PYTHONPATH=. python -m src.inspect_data validate data/raw/02336000_turbidity_63680.csv

# Print a human-readable summary
PYTHONPATH=. python -m src.inspect_data summarise data/raw/02336000_turbidity_63680.csv

# Same summary as JSON; --reindex turns missing timestamps into NaN rows
PYTHONPATH=. python -m src.inspect_data summarise data/raw/02336000_turbidity_63680.csv --json
PYTHONPATH=. python -m src.inspect_data summarise data/raw/02336000_turbidity_63680.csv --reindex

# Validate an injected labels file (when those exist)
PYTHONPATH=. python -m src.inspect_data validate-labels data/injected/<name>_labels.csv
```

Run the unit tests:

```bash
PYTHONPATH=. pytest tests/test_inspect_data.py -v
```

---

## Phase 2: Tools

In Phase 2, we build the "toolkit" for our smart robot (the LLM) using the SaQC library.

-  Testing the tools: We wrote automated tests (in `tests/test_tools.py`) to double-check that the SaQC tools work on our data without crashing.
- Translators ("Wrappers"): Our robot cannot use the SaQC tools directly, so we built "wrappers" around them (in `src/tools/wrappers.py`). A wrapper is like a translator: the robot says "find a spike", the wrapper tells SaQC to do the math, and then the wrapper hands the robot back a simple report card (a JSON dictionary) of what happened.

---

## Project layout

```
.
├── CLAUDE.md              # durable project brief (read this first)
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example           # ANTHROPIC_API_KEY=
├── data/
│   ├── raw/               # downloaded USGS/ECCC (gitignored)
│   ├── clean/             # inspected clean segments used for injection
│   └── injected/          # synthetic datasets + label files
├── src/
│   ├── pull_usgs.py       # download continuous turbidity from USGS NWIS
│   ├── inspect_data.py    # load, validate contracts, summarise a series
│   ├── inject.py          # synthetic anomaly injection (5 types, 3 levels, seeded)
│   ├── evaluate.py        # metrics, fixed-pipeline baseline, ablation
│   ├── agent.py           # ReAct loop + API logger
│   └── tools/
│       └── wrappers.py    # SaQC-wrapping tool functions
├── app/
│   └── streamlit_app.py   # Streamlit UI
├── tests/
│   ├── test_inspect_data.py
│   └── test_tools.py
└── logs/                  # JSONL API logs (gitignored)
```

## Build status

The project is built phase by phase (see §12 of `CLAUDE.md`).

- [x] **Phase 0 — Scaffold**
- [ ] Phase 1 — Data + injection
- [ ] Phase 2 — Tools
- [ ] Phase 3 — Agent (CLI)
- [ ] Phase 4 — Evaluation
- [ ] Phase 5 — UI
- [ ] Phase 6 — Polish
