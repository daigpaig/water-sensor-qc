"""Pull continuous USGS water-quality time series from NWIS, by variable.

Downloads instantaneous-value ("iv", sub-hourly — the project bases are 5-min)
series for one or more stream gauges via the ``dataretrieval`` package and
writes one tidy CSV per site under that variable's directory.

``data/`` is partitioned by **variable** first and **approval status** second::

    data/turbidity/{approved,provisional}/
    data/specific_conductance/{approved,provisional}/

Both splits are load-bearing. ``<variable>/approved/`` holds the clean bases
that ``src.datasets.inject`` globs, so a provisional pull must never land
there — ``_validate`` refuses the combination outright rather than trusting the
caller to pass the right ``--outdir``.

Variables (see :class:`Variable` and :data:`VARIABLES`)
------------------------------------------------------
``turbidity`` — param ``63680``, *Turbidity, water, unfiltered, monochrome near
infra-red LED light, 780-900 nm, detection angle 90 +-2.5 degrees, formazin
nephelometric units (FNU).* The standard **continuous optical-sensor** code; we
deliberately avoid ``00076`` (NTU), which is more often discrete / lab data.

``specific_conductance`` — param ``00095``, *Specific conductance, water,
unfiltered, microsiemens per centimetre at 25 degC.* Again the continuous code;
``90095`` is the same measurement from a lab and so is discrete.

Approval modes
--------------
**Approved data is our clean base (CLAUDE.md §9).** By default we keep only rows
whose USGS qualifier is *approved* (code starts with ``A``). Approved records
have been through USGS record processing — fouling and calibration-drift
corrections applied and prorated between field visits (TM 1-D3) — so an approved
series is clean apart from gaps. That is what lets us inject synthetic anomalies
straight into these files. Dropped rows simply become missing rows, i.e. gaps,
once the series is re-gridded downstream.

``--approval provisional`` keeps exactly the **complement** — the rows USGS has
not yet vetted. This is a real filter and not the absence of one: a pull over a
recent window still carries approved rows at its head, and letting them through
would put vetted data in ``provisional/`` and misrepresent what the file is.
Because USGS approves a record from the past forward, provisional rows only
exist near the END of a series, so a variable carries a separate, more recent
provisional window (:attr:`Variable.provisional_start`).

``--approval all`` keeps every row with its qualifier intact.

Output CSV schema (one file per site,
``data/<variable>/<approval>/<site>_<variable>_<param>.csv``)
-------------------------------------------------------------------------------
- ``datetime``  : ISO-8601, **UTC, timezone-naive** (converted from the site's
  local reporting zone so multiple gauges share one clock).
- ``value``     : the measurement in the variable's unit (float; NaN where the
  sensor reported a NaN).
- ``qualifier`` : USGS approval/qualifier code for the reading, e.g. ``A``
  (approved), ``P`` (provisional), ``A e`` (approved estimated).

Note the raw file only contains rows that USGS actually reported; real gaps show
up as *missing rows*, not NaN. Down-stream inspection (Phase 1 ``inspect_data``)
re-indexes onto a regular grid to expose those gaps.

CLI
---
    # See what would be pulled, but download nothing:
    python -m src.datasets.pull_usgs --dry-run

    # Pull the default 3-gauge, 2-year, 5-minute turbidity set:
    python -m src.datasets.pull_usgs

    # Specific conductance: 3 approved bases, then the 5 provisional gauges
    # over their recent tail (both site lists live in VARIABLES):
    python -m src.datasets.pull_usgs --variable specific_conductance
    python -m src.datasets.pull_usgs --variable specific_conductance --approval provisional

    # Custom sites / window:
    python -m src.datasets.pull_usgs --sites 06818000 11501000 --start 2022-07-01 --end 2024-07-01

    # Screen candidates and only keep gauges with a >=120-day unbroken stretch:
    python -m src.datasets.pull_usgs --sites 06818000 11501000 --min-unbroken-days 120 --drop-unqualified

Usage from Python
-----------------
    from src.datasets.pull_usgs import PullConfig, pull_all
    results = pull_all(PullConfig())
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults (recommended gauges — see the discovery/verification notes).
# Every default gauge over the default window samples at a consistent **5-minute**
# step, is 100% USGS-APPROVED, is >=95% complete, AND has a CALM approved baseline.
# "Approved" corrects fouling/drift and deletes clearly-bad data but KEEPS real
# storm spikes, so a flashy river's approved record is still spiky; unlabelled base
# spikes would score as false positives (§9.1), hence the calm-baseline requirement.
#
# The 5-min requirement is a hard filter on a small pool: of 1,737 turbidity (63680)
# instantaneous series nationwide, only 57 sample at <=5 min, and only 13 of those
# also clear approval + completeness + a 2-year span (scratchpad/screen_5min_*.py).
# Regime spread (low / moderate / high), with base-spike share measured identically
# for all candidates via flagUniLOF(n=20, thresh=1.5):
#   02054550   Roanoke R at Salem, VA        -> Blue Ridge Appalachian headwater;
#                                               LOW regime (median ~1.9 FNU);
#                                               96.0% complete; 0.31% base spikes.
#   01467200   Delaware R at Penn's Landing, PA -> Mid-Atlantic tidal urban estuary;
#                                               MODERATE (median ~6.5 FNU);
#                                               96.6% complete; 0.10% base spikes.
#   040851385  Fox R at Green Bay, WI        -> Great Lakes river mouth; HIGH
#                                               (median ~11.3 FNU); 95.4% complete;
#                                               0.41% base spikes.
# All three hold a 5-min modal step in every one of the 25 months in the window
# (>=98.7% of steps), so re-gridding cannot manufacture phantom gaps (§9.1).
#
# Retired 2026-08-18 when the project moved to 5-min data; the 15-min series and the
# datasets injected into them are kept under data/legacy_15min/ (see its README):
#   03447687 French Broad R, NC (~8 FNU) | 02198840 Savannah R, GA (~13 FNU)
#   08041770 LNVA Canal, TX (~28 FNU).
# Note the 5-min pool tops out near ~12 FNU median: no gauge sampling at <=5 min
# clears the §9 gates with a median above that, so the new spread is low->moderate
# rather than the retired set's moderate->high.
# Earlier retired: 02203603 / 02198955 (too flashy), 12340500 (35% missing),
# 06818000 (14% missing), 11501000 (mostly provisional).
# ---------------------------------------------------------------------------
TURBIDITY_PARAM = "63680"
DEFAULT_SITES: tuple[str, ...] = ("02054550", "01467200", "040851385")
DEFAULT_START = "2023-07-01"
DEFAULT_END = "2025-07-01"
# Approval status partitions each variable's directory: approved/ is what
# src.datasets.inject globs as its clean bases, provisional/ is scratch for
# auditing (CLAUDE.md §9, §9.1).
DEFAULT_OUTDIR = Path("data/turbidity/approved")
DEFAULT_UNAPPROVED_OUTDIR = Path("data/turbidity/provisional")

# USGS qualifier codes starting with this prefix are "approved" (e.g. ``A``,
# ``A e``, ``A, >``); ``P`` (provisional), blank, and NaN are not. Approved rows
# are our clean base (CLAUDE.md §9); everything else is dropped by default.
APPROVED_PREFIX = "A"

#: How a pull treats the qualifier column. ``approved`` keeps only rows USGS has
#: signed off (the injection bases); ``provisional`` keeps only rows it has NOT
#: (the unvetted tail — the opposite filter, not the absence of one); ``all``
#: keeps every row with the qualifier intact, which is what
#: ``pull_comparison`` needs to see the boundary inside one file.
APPROVAL_MODES = ("approved", "provisional", "all")


# ---------------------------------------------------------------------------
# Variables. `data/` is partitioned by VARIABLE first and approval second, so a
# variable owns its parameter code, its unit, its directory root, and the gauges
# screened for it. Adding one means adding an entry here and nothing else.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Variable:
    """One measured quantity, its NWIS parameter code, and where it lands.

    ``name`` is both the directory under ``data/`` and the middle field of every
    filename, so ``turbidity``/``63680`` reproduces the existing
    ``<site>_turbidity_63680.csv`` exactly — the rename is a no-op for files
    already on disk.

    A variable carries TWO windows because the two halves of ``data/<var>/``
    answer different questions. ``start``/``end`` is the approved window: a
    fixed two-year span, identical across gauges, so the bases are comparable.
    ``provisional_start``/``provisional_end`` is the unapproved tail, which is
    necessarily RECENT — USGS approves a record from the past forward, so
    provisional rows only exist near the end of a series and a pull over the
    approved window would come back empty (CLAUDE.md §9.6).
    """

    name: str
    param_cd: str
    unit: str
    #: Directory root under ``data/``; ``approved/`` and ``provisional/`` hang off it.
    root: Path
    #: Gauges screened as clean injection bases (approved window).
    approved_sites: tuple[str, ...] = ()
    #: Gauges pulled for their UNAPPROVED tail (provisional window).
    provisional_sites: tuple[str, ...] = ()
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    provisional_start: str = ""
    provisional_end: str = ""

    @property
    def approved_dir(self) -> Path:
        return self.root / "approved"

    @property
    def provisional_dir(self) -> Path:
        return self.root / "provisional"

    def outdir_for(self, approval: str) -> Path:
        """Where a pull in this approval mode belongs.

        ``all`` keeps both states in one file, which is not an injection base by
        any reading, so it lands in ``provisional/`` rather than beside the
        clean bases — the split is load-bearing (CLAUDE.md §9).
        """
        return self.approved_dir if approval == "approved" else self.provisional_dir

    def filename(self, site: str, param_cd: str | None = None) -> str:
        """``<site>_<variable>_<param>.csv`` — the param code records what was pulled."""
        return f"{site}_{self.name}_{param_cd or self.param_cd}.csv"


TURBIDITY = Variable(
    name="turbidity",
    param_cd=TURBIDITY_PARAM,
    unit="FNU",
    root=Path("data/turbidity"),
    approved_sites=DEFAULT_SITES,
)

# Specific conductance, water, unfiltered, microsiemens per centimetre at 25 degC.
# 00095 is the standard CONTINUOUS sensor code; 90095 is the same measurement
# reported by a lab, so it is discrete and deliberately not used here.
#
# Site lists are filled in by `scratchpad/screen_5min_conductance.py`, which
# repeats the §9 screen for this parameter: a national catalog sweep, then a
# MEASURED cadence check (the catalog's `count_nu` is days of record for a
# unit-value series, not a sample count, so cadence cannot be read from it), then
# approval/completeness per candidate. Empty until that has actually run — a
# guessed site list would pull the wrong river and look perfectly well-formed.
CONDUCTANCE_PARAM = "00095"
CONDUCTANCE_APPROVED_SITES: tuple[str, ...] = ()
CONDUCTANCE_PROVISIONAL_SITES: tuple[str, ...] = ()
# The provisional tail. USGS approves from the past forward, so unapproved rows
# live at the END of a record; a 12-month window is wide enough to be worth
# plotting and recent enough that most of it is genuinely still provisional.
CONDUCTANCE_PROVISIONAL_START = "2025-09-01"
CONDUCTANCE_PROVISIONAL_END = "2026-09-01"

SPECIFIC_CONDUCTANCE = Variable(
    name="specific_conductance",
    param_cd=CONDUCTANCE_PARAM,
    unit="uS/cm",
    root=Path("data/specific_conductance"),
    approved_sites=CONDUCTANCE_APPROVED_SITES,
    provisional_sites=CONDUCTANCE_PROVISIONAL_SITES,
    provisional_start=CONDUCTANCE_PROVISIONAL_START,
    provisional_end=CONDUCTANCE_PROVISIONAL_END,
)

VARIABLES: dict[str, Variable] = {v.name: v for v in (TURBIDITY, SPECIFIC_CONDUCTANCE)}

# "Unbroken-stretch" reporting defaults. A stretch stays "unbroken" as long as no
# internal gap exceeds `max_gap`, so a few scattered single-sample dropouts don't break
# an otherwise continuous span. This is now an informational gauge-quality report
# (we inject into the whole approved series), not a carving step.
DEFAULT_MAX_GAP = "3h"
DEFAULT_MIN_UNBROKEN_DAYS = 90.0


@dataclass(frozen=True)
class PullConfig:
    """Configuration for one pull run of one variable."""

    sites: tuple[str, ...] = DEFAULT_SITES
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    param_cd: str = TURBIDITY_PARAM
    outdir: Path = DEFAULT_OUTDIR
    max_retries: int = 3
    retry_wait_s: float = 5.0
    #: Which side of the approval boundary to keep; see :data:`APPROVAL_MODES`.
    approval: str = "approved"
    # Unbroken-stretch report (see `longest_unbroken_run_days`).
    max_gap: str = DEFAULT_MAX_GAP
    min_unbroken_days: float = DEFAULT_MIN_UNBROKEN_DAYS
    drop_unqualified: bool = False
    #: Names the output file and supplies the unit for printed summaries.
    variable: Variable = TURBIDITY

    @property
    def approved_only(self) -> bool:
        """Back-compat read-only view of :attr:`approval`."""
        return self.approval == "approved"


@dataclass
class SiteResult:
    """Structured summary of one site's pulled series."""

    site_no: str
    station_nm: str
    n_obs: int
    n_raw: int
    n_dropped_by_approval: int
    median_dt_min: float
    span_days: float
    completeness_pct: float
    nan_pct: float
    longest_gap_hr: float
    value_min: float
    value_median: float
    value_max: float
    longest_unbroken_days: float
    max_gap: str
    meets_unbroken: bool
    out_path: Path | None = None


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without network access).
# ---------------------------------------------------------------------------
def select_value_column(df: pd.DataFrame, param_cd: str = TURBIDITY_PARAM) -> str:
    """Return the name of the value column for ``param_cd`` in an NWIS frame.

    NWIS frames carry a value column (e.g. ``"63680"``) and a paired qualifier
    column (``"63680_cd"``). Some sites expose more than one sensor series
    (``"63680"`` plus ``"63680.1"`` / suffixed names); we take the first value
    column and leave a note for the caller to inspect if that happens.

    Raises ``ValueError`` if no value column for ``param_cd`` is present.
    """
    value_cols = [
        c for c in df.columns
        if param_cd in str(c) and not str(c).endswith("_cd")
    ]
    if not value_cols:
        raise ValueError(
            f"No value column for parameter {param_cd} found; "
            f"columns={list(df.columns)}"
        )
    # Prefer the exact param code, else the first suffixed variant.
    for c in value_cols:
        if str(c) == param_cd:
            return c
    return value_cols[0]


#: Kept so callers written against the turbidity-only API keep working.
select_turbidity_column = select_value_column


def filter_approved(
    df: pd.DataFrame, *, qualifier_col: str = "qualifier"
) -> pd.DataFrame:
    """Keep only USGS-approved rows (qualifier starting ``A``); drop the rest.

    Approved records have been through USGS record processing — fouling and
    calibration-drift corrections applied and prorated between field visits
    (TM 1-D3) — so an approved series is clean apart from gaps, which is why we
    inject synthetic anomalies straight into it (CLAUDE.md §9). Provisional
    (``P``), blank, and NaN qualifiers are dropped; downstream re-gridding turns
    the removed rows into missing rows (gaps), which is the honest label for
    them. Codes like ``A e`` (approved estimated) and ``A, >`` (approved,
    over-range) start with ``A`` and are kept — USGS approved them.

    If the frame has no qualifier column it is returned unchanged.
    """
    return filter_by_approval(df, "approved", qualifier_col=qualifier_col)


def filter_by_approval(
    df: pd.DataFrame, approval: str, *, qualifier_col: str = "qualifier"
) -> pd.DataFrame:
    """Keep the rows on one side of the USGS approval boundary.

    ``approved`` keeps qualifiers starting ``A``; ``provisional`` keeps exactly
    the complement — ``P``, blank and NaN — which is a real filter and not the
    absence of one. That distinction is the whole point of the provisional set:
    a series pulled from a recent window still carries approved rows at its
    head, and letting them through would put vetted data in
    ``<variable>/provisional/`` and quietly misrepresent what the file is.
    ``all`` keeps every row with its qualifier intact.

    A frame with no qualifier column is returned unchanged, because we cannot
    tell the two sides apart and dropping everything would be worse than
    passing it on for the caller to notice.
    """
    if approval not in APPROVAL_MODES:
        raise ValueError(f"approval must be one of {APPROVAL_MODES}, got {approval!r}")
    if approval == "all" or qualifier_col not in df.columns:
        return df
    q = df[qualifier_col].astype("string")
    is_approved = q.str.startswith(APPROVED_PREFIX).fillna(False)
    keep = is_approved if approval == "approved" else ~is_approved
    return df.loc[keep.to_numpy()].reset_index(drop=True)


def summarise_series(
    idx: pd.DatetimeIndex, values: pd.Series
) -> tuple[float, float, float, float, float]:
    """Compute (median_dt_min, span_days, completeness_pct, nan_pct, longest_gap_hr).

    ``completeness_pct`` = observed rows / rows expected at the median sampling
    step across the observed span. Gaps (missing rows) drive it below 100%.
    """
    if len(idx) < 2:
        return (float("nan"), 0.0, float("nan"), 0.0, float("nan"))
    diffs_min = idx.to_series().diff().dropna().dt.total_seconds() / 60.0
    median_dt = float(np.median(diffs_min))
    span_min = (idx.max() - idx.min()).total_seconds() / 60.0
    expected = span_min / median_dt if median_dt else float("nan")
    completeness = 100.0 * len(idx) / expected if expected else float("nan")
    nan_pct = 100.0 * float(values.isna().sum()) / len(values)
    longest_gap_hr = float(diffs_min.max() / 60.0)
    return (median_dt, span_min / 1440.0, completeness, nan_pct, longest_gap_hr)


def longest_unbroken_run_days(series: pd.Series, max_gap: pd.Timedelta) -> float:
    """Longest span (in days) with no internal gap larger than ``max_gap``.

    This is the primary data-quality gate for gauge selection (CLAUDE.md §9):
    we want a continuous span long enough to carve a clean multi-month injection
    segment from. Raw NWIS omits gap rows, so a "gap" is any step between
    consecutive *valid* observations that exceeds ``max_gap`` — NaN-valued rows
    are treated as missing. Allowing a small ``max_gap`` means a handful of
    scattered dropouts don't break an otherwise continuous stretch.

    Returns 0.0 for fewer than two valid observations. Unit-agnostic: correct
    whatever the underlying ``datetime64`` resolution.
    """
    valid = series.dropna()
    idx = pd.DatetimeIndex(valid.index)
    idx = idx[~idx.duplicated(keep="first")].sort_values()
    if len(idx) < 2:
        return 0.0
    times = idx.to_numpy()
    diffs_s = np.diff(times) / np.timedelta64(1, "s")            # gap sizes, seconds
    elapsed_s = (times - times[0]) / np.timedelta64(1, "s")      # seconds from start
    brk = np.where(diffs_s > max_gap.total_seconds())[0]        # break between i and i+1
    starts = np.concatenate(([0], brk + 1))
    ends = np.concatenate((brk, [len(times) - 1]))
    spans_days = (elapsed_s[ends] - elapsed_s[starts]) / 86400.0
    return float(spans_days.max())


def tidy_frame(df: pd.DataFrame, param_cd: str = TURBIDITY_PARAM) -> pd.DataFrame:
    """Convert a raw NWIS iv frame into the tidy ``datetime/value/qualifier`` schema.

    - Value column selected via :func:`select_turbidity_column`.
    - Index converted to UTC then made timezone-naive.
    - Qualifier column (``<param>_cd``) carried through if present.
    """
    val_col = select_value_column(df, param_cd)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out = pd.DataFrame(
        {
            "datetime": idx,
            "value": pd.to_numeric(df[val_col], errors="coerce").to_numpy(),
        }
    )
    cd_col = f"{val_col}_cd"
    out["qualifier"] = (
        df[cd_col].to_numpy() if cd_col in df.columns else pd.NA
    )
    out = out.sort_values("datetime").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Network layer.
# ---------------------------------------------------------------------------
def _fetch_iv_with_retry(
    site: str, param_cd: str, start: str, end: str,
    max_retries: int, retry_wait_s: float,
) -> pd.DataFrame:
    """Fetch instantaneous values with a simple retry on transient network errors."""
    from dataretrieval import nwis  # local import so tests need no network stack

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            df, _ = nwis.get_iv(
                sites=site, parameterCd=param_cd, start=start, end=end
            )
            return df
        except Exception as exc:  # noqa: BLE001 - retry any transient failure
            last_err = exc
            if attempt < max_retries:
                print(f"  [{site}] attempt {attempt} failed ({type(exc).__name__}); "
                      f"retrying in {retry_wait_s:.0f}s...")
                time.sleep(retry_wait_s)
    raise RuntimeError(f"Failed to fetch {site} after {max_retries} attempts: {last_err}")


def _station_name(site: str) -> str:
    """Best-effort station name lookup; returns '' on failure."""
    from dataretrieval import nwis

    try:
        info, _ = nwis.get_info(sites=site)
        if info is not None and not info.empty and "station_nm" in info.columns:
            return str(info["station_nm"].iloc[0])
    except Exception:  # noqa: BLE001
        pass
    return ""


def pull_site(site: str, cfg: PullConfig, write: bool = True) -> SiteResult:
    """Pull one site, optionally write its CSV, and return a :class:`SiteResult`."""
    raw = _fetch_iv_with_retry(
        site, cfg.param_cd, cfg.start, cfg.end, cfg.max_retries, cfg.retry_wait_s
    )
    if raw is None or raw.empty:
        raise RuntimeError(
            f"No {cfg.variable.name} data returned for {site} "
            f"in {cfg.start}..{cfg.end}"
        )

    tidy = tidy_frame(raw, cfg.param_cd)
    n_raw = len(tidy)
    tidy = filter_by_approval(tidy, cfg.approval)
    n_dropped = n_raw - len(tidy)
    if tidy.empty:
        # Which side came back empty says something different in each direction,
        # so the hint has to differ too: no approved rows means the window is too
        # recent, no provisional rows means it is too old.
        hint = (
            "Try an earlier window or --approval all."
            if cfg.approval == "approved"
            else "USGS approves a record from the past forward, so try a MORE "
                 "RECENT window — this one may be fully approved already."
        )
        raise RuntimeError(
            f"No {cfg.approval} {cfg.variable.name} rows for {site} in "
            f"{cfg.start}..{cfg.end} (dropped all {n_raw} rows). {hint}"
        )

    idx = pd.DatetimeIndex(tidy["datetime"])
    median_dt, span_days, completeness, nan_pct, longest_gap = summarise_series(
        idx, tidy["value"]
    )

    series = pd.Series(tidy["value"].to_numpy(), index=idx)
    longest_unbroken = longest_unbroken_run_days(series, pd.Timedelta(cfg.max_gap))
    meets = longest_unbroken >= cfg.min_unbroken_days

    # Filter: optionally skip writing sites that lack a qualifying unbroken span.
    out_path: Path | None = None
    if write and not (cfg.drop_unqualified and not meets):
        cfg.outdir.mkdir(parents=True, exist_ok=True)
        out_path = cfg.outdir / cfg.variable.filename(site, cfg.param_cd)
        tidy.to_csv(out_path, index=False)

    return SiteResult(
        site_no=site,
        station_nm=_station_name(site),
        n_obs=len(tidy),
        n_raw=n_raw,
        n_dropped_by_approval=n_dropped,
        median_dt_min=median_dt,
        span_days=span_days,
        completeness_pct=completeness,
        nan_pct=nan_pct,
        longest_gap_hr=longest_gap,
        value_min=float(tidy["value"].min()),
        value_median=float(tidy["value"].median()),
        value_max=float(tidy["value"].max()),
        longest_unbroken_days=longest_unbroken,
        max_gap=cfg.max_gap,
        meets_unbroken=meets,
        out_path=out_path,
    )


def pull_all(cfg: PullConfig, write: bool = True) -> list[SiteResult]:
    """Pull every site in ``cfg`` and return the list of results."""
    results: list[SiteResult] = []
    for site in cfg.sites:
        print(f"Pulling {site} ({cfg.param_cd}) {cfg.start} -> {cfg.end} ...")
        res = pull_site(site, cfg, write=write)
        print(
            f"  {res.n_obs:,} obs | {res.median_dt_min:.0f}-min | "
            f"{res.completeness_pct:.1f}% complete | "
            f"range {res.value_min:.1f}-{res.value_max:.1f} {cfg.variable.unit}"
        )
        if cfg.approval != "all":
            kept_pct = 100.0 * res.n_obs / res.n_raw if res.n_raw else float("nan")
            other = "non-approved (provisional/blank)" if cfg.approval == "approved" \
                else "already-approved"
            print(
                f"  {cfg.approval}: kept {res.n_obs:,}/{res.n_raw:,} rows "
                f"({kept_pct:.1f}%), dropped {res.n_dropped_by_approval:,} {other}"
            )
        gate = "PASS" if res.meets_unbroken else "FAIL"
        if res.out_path is not None:
            loc = f" -> {res.out_path}"
        elif cfg.drop_unqualified and not res.meets_unbroken:
            loc = " (not written: fails filter)"
        else:
            loc = ""
        print(
            f"  unbroken (gaps <= {res.max_gap}): {res.longest_unbroken_days:.0f} d "
            f"[{gate}, need >= {cfg.min_unbroken_days:.0f} d]{loc}"
        )
        if not res.meets_unbroken:
            print(
                f"  ⚠ {site} has no {cfg.min_unbroken_days:.0f}-day unbroken stretch "
                f"at gaps <= {res.max_gap}."
            )
        results.append(res)
    n_ok = sum(r.meets_unbroken for r in results)
    print(
        f"\n{n_ok}/{len(results)} site(s) meet the >= {cfg.min_unbroken_days:.0f}-day "
        f"unbroken filter (max_gap {cfg.max_gap})."
    )
    return results


def _validate(cfg: PullConfig) -> None:
    """Fail loudly on nonsensical configuration (CLAUDE.md §13)."""
    if not cfg.sites:
        raise ValueError("No sites specified.")
    for s in cfg.sites:
        if not (s.isdigit() and 8 <= len(s) <= 15):
            raise ValueError(f"Suspicious USGS site number: {s!r} (expected 8-15 digits).")
    start, end = pd.to_datetime(cfg.start), pd.to_datetime(cfg.end)
    if end <= start:
        raise ValueError(f"end ({cfg.end}) must be after start ({cfg.start}).")
    try:
        gap = pd.Timedelta(cfg.max_gap)
    except ValueError as exc:
        raise ValueError(f"Invalid max_gap {cfg.max_gap!r} (use e.g. '3h', '90min').") from exc
    if gap <= pd.Timedelta(0):
        raise ValueError(f"max_gap must be positive (got {cfg.max_gap!r}).")
    if cfg.min_unbroken_days < 0:
        raise ValueError(f"min_unbroken_days must be >= 0 (got {cfg.min_unbroken_days}).")
    if cfg.approval not in APPROVAL_MODES:
        raise ValueError(f"approval must be one of {APPROVAL_MODES}, got {cfg.approval!r}.")
    # An approved pull into provisional/ (or the reverse) would silently
    # mislabel what the file is, and inject.py globs on directory alone (§9).
    if cfg.approval == "approved" and cfg.outdir.name == "provisional":
        raise ValueError(
            f"approval='approved' writing into {cfg.outdir} — approved bases belong "
            f"in approved/. Pass --outdir explicitly if this is deliberate."
        )
    if cfg.approval != "approved" and cfg.outdir.name == "approved":
        raise ValueError(
            f"approval={cfg.approval!r} writing into {cfg.outdir} — only approved rows "
            f"may land in approved/, which src.datasets.inject globs as clean bases (§9)."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pull USGS NWIS continuous water-quality series "
                    "(turbidity or specific conductance)."
    )
    parser.add_argument("--variable", choices=sorted(VARIABLES), default=TURBIDITY.name,
                        help="Which measured quantity to pull. Sets the parameter code, "
                             "the default site list, the default window and the output "
                             f"directory (default: {TURBIDITY.name}).")
    parser.add_argument("--sites", nargs="+", default=None,
                        help="USGS site numbers (default: the variable's screened set "
                             "for the chosen approval mode).")
    parser.add_argument("--start", default=None, help="ISO start date.")
    parser.add_argument("--end", default=None, help="ISO end date.")
    parser.add_argument("--param", default=None, help="NWIS parameter code.")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Directory for output CSVs (default: the variable's "
                             "approved/ or provisional/ subdirectory).")
    parser.add_argument("--approval", choices=APPROVAL_MODES, default="approved",
                        help="Which side of the USGS approval boundary to keep. "
                             "'approved' (default) is the clean injection base; "
                             "'provisional' keeps ONLY unapproved rows; 'all' keeps "
                             "both with the qualifier intact. Anything but 'approved' "
                             "lands in provisional/, so unvetted data can never reach "
                             "the directory src.datasets.inject globs.")
    parser.add_argument("--max-gap", default=DEFAULT_MAX_GAP,
                        help=f"Largest gap that does NOT break an 'unbroken' stretch "
                             f"(pandas offset, e.g. '3h', '90min'; default: {DEFAULT_MAX_GAP}).")
    parser.add_argument("--min-unbroken-days", type=float, default=DEFAULT_MIN_UNBROKEN_DAYS,
                        help=f"Required longest unbroken stretch, in days "
                             f"(default: {DEFAULT_MIN_UNBROKEN_DAYS:.0f}).")
    parser.add_argument("--drop-unqualified", action="store_true",
                        help="Do not write CSVs for sites that fail the unbroken-stretch filter.")
    parser.add_argument("--keep-unapproved", action="store_true",
                        help="Deprecated alias for --approval all.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be pulled and exit without downloading.")
    args = parser.parse_args(argv)

    variable = VARIABLES[args.variable]
    approval = "all" if args.keep_unapproved else args.approval

    # Each of these falls back to the variable's own screened defaults, so
    # `--variable specific_conductance --approval provisional` needs no other
    # flags: it pulls the screened provisional gauges over the recent tail.
    provisional = approval != "approved"
    sites = tuple(args.sites) if args.sites else (
        variable.provisional_sites if provisional else variable.approved_sites
    )
    if not sites:
        raise SystemExit(
            f"No default sites recorded for {variable.name} / {approval}. Run the "
            f"screening script for this variable and record the result in "
            f"pull_usgs.VARIABLES, or pass --sites explicitly. A guessed site "
            f"list pulls the wrong river and the output still looks well-formed."
        )
    default_start = variable.provisional_start if provisional else variable.start
    default_end = variable.provisional_end if provisional else variable.end
    cfg = PullConfig(
        sites=sites,
        start=args.start or default_start or DEFAULT_START,
        end=args.end or default_end or DEFAULT_END,
        param_cd=args.param or variable.param_cd,
        # An explicit --outdir always wins; otherwise approval status picks the dir.
        outdir=args.outdir or variable.outdir_for(approval),
        approval=approval,
        max_gap=args.max_gap,
        min_unbroken_days=args.min_unbroken_days,
        drop_unqualified=args.drop_unqualified,
        variable=variable,
    )
    _validate(cfg)

    if args.dry_run:
        print("DRY RUN — nothing will be downloaded.")
        print(f"  variable: {variable.name} ({cfg.param_cd}, {variable.unit})")
        print(f"  window: {cfg.start} -> {cfg.end}")
        print(f"  outdir: {cfg.outdir}")
        print(f"  approval: {cfg.approval} " + {
            "approved": f"(keep only qualifiers starting '{APPROVED_PREFIX}')",
            "provisional": f"(keep only qualifiers NOT starting '{APPROVED_PREFIX}')",
            "all": "(keep every row, qualifier intact)",
        }[cfg.approval])
        print(f"  report: longest unbroken stretch at gaps <= {cfg.max_gap} "
              f"(>= {cfg.min_unbroken_days:.0f} d flagged PASS)"
              f"{' (drop unqualified)' if cfg.drop_unqualified else ''}")
        for site in cfg.sites:
            print(f"  site  : {site} -> "
                  f"{cfg.outdir / variable.filename(site, cfg.param_cd)}")
        return 0

    pull_all(cfg, write=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
