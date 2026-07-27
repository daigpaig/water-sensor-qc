"""Pull continuous, USGS-**approved** turbidity time series from NWIS.

Downloads instantaneous-value ("iv", sub-hourly / 15-min) turbidity for one or
more stream gauges via the ``dataretrieval`` package and writes one tidy CSV per
site to ``data/raw/`` (gitignored per CLAUDE.md §4).

**Approved data is our clean base (CLAUDE.md §9).** By default we keep only rows
whose USGS qualifier is *approved* (code starts with ``A``) and drop provisional
(``P``), blank, or NaN-qualified rows. Approved records have already been through
USGS record processing — fouling and calibration-drift corrections applied and
prorated between field visits (TM 1-D3) — so an approved series is clean apart
from gaps. That is what lets us inject synthetic anomalies straight into these
files (``src.inject`` reads ``data/raw`` directly): there is no separate
"clean-segment" carving or by-eye auditing step any more. Dropped rows simply
become missing rows, i.e. gaps, once the series is re-gridded downstream.

Turbidity parameter code
------------------------
``63680`` — *Turbidity, water, unfiltered, monochrome near infra-red LED light,
780-900 nm, detection angle 90 +-2.5 degrees, formazin nephelometric units
(FNU).* This is the standard **continuous optical-sensor** turbidity code. We
deliberately avoid ``00076`` (NTU), which is more often discrete / lab data.

Output CSV schema (one file per site, ``data/raw/<site>_turbidity_63680.csv``)
------------------------------------------------------------------------------
- ``datetime``  : ISO-8601, **UTC, timezone-naive** (converted from the site's
  local reporting zone so multiple gauges share one clock).
- ``value``     : turbidity in FNU (float; NaN where the sensor reported a NaN).
- ``qualifier`` : USGS approval/qualifier code for the reading, e.g. ``A``
  (approved), ``P`` (provisional), ``A e`` (approved estimated).

Note the raw file only contains rows that USGS actually reported; real gaps show
up as *missing rows*, not NaN. Down-stream inspection (Phase 1 ``inspect_data``)
re-indexes onto a regular grid to expose those gaps.

CLI
---
    # See what would be pulled, but download nothing:
    python -m src.pull_usgs --dry-run

    # Pull the default 3-gauge, 2-year set (each verified to hold a >=90-day
    # unbroken stretch at gaps <= 3h):
    python -m src.pull_usgs

    # Custom sites / window:
    python -m src.pull_usgs --sites 06818000 11501000 --start 2022-07-01 --end 2024-07-01

    # Screen candidates and only keep gauges with a >=120-day unbroken stretch:
    python -m src.pull_usgs --sites 06818000 11501000 --min-unbroken-days 120 --drop-unqualified

Usage from Python
-----------------
    from src.pull_usgs import PullConfig, pull_all
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
# Every default gauge over the default window is 100% USGS-APPROVED, samples at a
# consistent 15-min step, is >=95% complete, AND has a CALM approved baseline —
# very few real spikes in the base itself (all verified). "Approved" corrects
# fouling/drift and deletes clearly-bad data but KEEPS real storm spikes, so a
# flashy river's approved record is still spiky; unlabelled base spikes would score
# as false positives (§9.1), hence the calm-baseline requirement. Regime spread
# (moderate / high):
#   03447687  French Broad R nr Fletcher, NC  -> S. Appalachia; moderate regime
#                                                 (median ~8 FNU); ~95% complete;
#                                                 ~25 base spikes (0.04%).
#   02198840  Savannah R at I-95 nr Port Wentworth, GA -> tidal river; moderate-high
#                                                 (median ~13 FNU, ~9x range);
#                                                 ~99% complete; 0 base spikes.
#   08041770  LNVA Canal at Beaumont, TX       -> managed canal; high regime
#                                                 (median ~28 FNU, ~5x range);
#                                                 ~99% complete; ~1 base spike.
# Replaced 02203603 (South R, Atlanta) and 02198955 (Middle R, tidal): 100%
# approved, 15-min, dense, but too FLASHY — 120 (0.18%) and 637 (0.91%) real spikes
# in the base. Earlier retired: 12340500 (Blackfoot, 35% missing), 06818000
# (Missouri, 14% missing), 11501000 (Sprague, mostly provisional).
# ---------------------------------------------------------------------------
TURBIDITY_PARAM = "63680"
DEFAULT_SITES: tuple[str, ...] = ("03447687", "02198840", "08041770")
DEFAULT_START = "2023-07-01"
DEFAULT_END = "2025-07-01"
DEFAULT_OUTDIR = Path("data/raw")

# USGS qualifier codes starting with this prefix are "approved" (e.g. ``A``,
# ``A e``, ``A, >``); ``P`` (provisional), blank, and NaN are not. Approved rows
# are our clean base (CLAUDE.md §9); everything else is dropped by default.
APPROVED_PREFIX = "A"

# "Unbroken-stretch" reporting defaults. A stretch stays "unbroken" as long as no
# internal gap exceeds `max_gap`, so a few scattered 15-min dropouts don't break
# an otherwise continuous span. This is now an informational gauge-quality report
# (we inject into the whole approved series), not a carving step.
DEFAULT_MAX_GAP = "3h"
DEFAULT_MIN_UNBROKEN_DAYS = 90.0


@dataclass(frozen=True)
class PullConfig:
    """Configuration for a turbidity pull run."""

    sites: tuple[str, ...] = DEFAULT_SITES
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    param_cd: str = TURBIDITY_PARAM
    outdir: Path = DEFAULT_OUTDIR
    max_retries: int = 3
    retry_wait_s: float = 5.0
    # Keep only USGS-approved rows (qualifier starts with ``A``); drop the rest.
    approved_only: bool = True
    # Unbroken-stretch report (see `longest_unbroken_run_days`).
    max_gap: str = DEFAULT_MAX_GAP
    min_unbroken_days: float = DEFAULT_MIN_UNBROKEN_DAYS
    drop_unqualified: bool = False


@dataclass
class SiteResult:
    """Structured summary of one site's pulled series."""

    site_no: str
    station_nm: str
    n_obs: int
    n_raw: int
    n_dropped_unapproved: int
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
def select_turbidity_column(df: pd.DataFrame, param_cd: str = TURBIDITY_PARAM) -> str:
    """Return the name of the value column for ``param_cd`` in an NWIS frame.

    NWIS frames carry a value column (e.g. ``"63680"``) and a paired qualifier
    column (``"63680_cd"``). Some sites expose more than one sensor series
    (``"63680"`` plus ``"63680.1"`` / suffixed names); we take the first value
    column and leave a note for the caller to inspect if that happens.

    Raises ``ValueError`` if no turbidity value column is present.
    """
    value_cols = [
        c for c in df.columns
        if param_cd in str(c) and not str(c).endswith("_cd")
    ]
    if not value_cols:
        raise ValueError(
            f"No turbidity ({param_cd}) value column found; columns={list(df.columns)}"
        )
    # Prefer the exact param code, else the first suffixed variant.
    for c in value_cols:
        if str(c) == param_cd:
            return c
    return value_cols[0]


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
    if qualifier_col not in df.columns:
        return df
    q = df[qualifier_col].astype("string")
    keep = q.str.startswith(APPROVED_PREFIX).fillna(False)
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
    val_col = select_turbidity_column(df, param_cd)
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
        raise RuntimeError(f"No turbidity data returned for {site} in {cfg.start}..{cfg.end}")

    tidy = tidy_frame(raw, cfg.param_cd)
    n_raw = len(tidy)
    if cfg.approved_only:
        tidy = filter_approved(tidy)
    n_dropped = n_raw - len(tidy)
    if tidy.empty:
        raise RuntimeError(
            f"No approved turbidity rows for {site} in {cfg.start}..{cfg.end} "
            f"(dropped all {n_raw} rows as non-approved). Try an earlier window "
            f"or pass --keep-unapproved."
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
        out_path = cfg.outdir / f"{site}_turbidity_{cfg.param_cd}.csv"
        tidy.to_csv(out_path, index=False)

    return SiteResult(
        site_no=site,
        station_nm=_station_name(site),
        n_obs=len(tidy),
        n_raw=n_raw,
        n_dropped_unapproved=n_dropped,
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
            f"range {res.value_min:.1f}-{res.value_max:.1f} FNU"
        )
        if cfg.approved_only:
            approved_pct = 100.0 * res.n_obs / res.n_raw if res.n_raw else float("nan")
            print(
                f"  approved: kept {res.n_obs:,}/{res.n_raw:,} rows ({approved_pct:.1f}%), "
                f"dropped {res.n_dropped_unapproved:,} non-approved (provisional/blank)"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pull USGS NWIS continuous turbidity.")
    parser.add_argument("--sites", nargs="+", default=list(DEFAULT_SITES),
                        help="USGS site numbers (default: recommended 3-gauge set).")
    parser.add_argument("--start", default=DEFAULT_START, help="ISO start date.")
    parser.add_argument("--end", default=DEFAULT_END, help="ISO end date.")
    parser.add_argument("--param", default=TURBIDITY_PARAM, help="NWIS parameter code.")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), type=Path,
                        help="Directory for output CSVs.")
    parser.add_argument("--max-gap", default=DEFAULT_MAX_GAP,
                        help=f"Largest gap that does NOT break an 'unbroken' stretch "
                             f"(pandas offset, e.g. '3h', '90min'; default: {DEFAULT_MAX_GAP}).")
    parser.add_argument("--min-unbroken-days", type=float, default=DEFAULT_MIN_UNBROKEN_DAYS,
                        help=f"Required longest unbroken stretch, in days "
                             f"(default: {DEFAULT_MIN_UNBROKEN_DAYS:.0f}).")
    parser.add_argument("--drop-unqualified", action="store_true",
                        help="Do not write CSVs for sites that fail the unbroken-stretch filter.")
    parser.add_argument("--keep-unapproved", action="store_true",
                        help="Keep provisional/blank-qualified rows (default: approved-only, "
                             "qualifier starting 'A').")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be pulled and exit without downloading.")
    args = parser.parse_args(argv)

    cfg = PullConfig(
        sites=tuple(args.sites), start=args.start, end=args.end,
        param_cd=args.param, outdir=args.outdir,
        approved_only=not args.keep_unapproved,
        max_gap=args.max_gap, min_unbroken_days=args.min_unbroken_days,
        drop_unqualified=args.drop_unqualified,
    )
    _validate(cfg)

    if args.dry_run:
        print("DRY RUN — nothing will be downloaded.")
        print(f"  param : {cfg.param_cd} (turbidity, FNU)")
        print(f"  window: {cfg.start} -> {cfg.end}")
        print(f"  outdir: {cfg.outdir}")
        print(f"  approved-only: {cfg.approved_only} "
              f"(keep qualifier starting '{APPROVED_PREFIX}')")
        print(f"  report: longest unbroken stretch at gaps <= {cfg.max_gap} "
              f"(>= {cfg.min_unbroken_days:.0f} d flagged PASS)"
              f"{' (drop unqualified)' if cfg.drop_unqualified else ''}")
        for s in cfg.sites:
            print(f"  site  : {s} -> {cfg.outdir / f'{s}_turbidity_{cfg.param_cd}.csv'}")
        return 0

    pull_all(cfg, write=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
