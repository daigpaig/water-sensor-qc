"""Pull continuous turbidity time series from USGS NWIS.

Downloads instantaneous-value ("iv", sub-hourly / 15-min) turbidity for one or
more stream gauges via the ``dataretrieval`` package and writes one tidy CSV per
site to ``data/raw/`` (gitignored per CLAUDE.md §4).

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

    # Pull the default 3-gauge, 2-year set:
    python -m src.pull_usgs

    # Custom sites / window:
    python -m src.pull_usgs --sites 02336000 08181500 --start 2022-07-01 --end 2024-07-01

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
# Three distinct turbidity regimes, all >=92% complete at 15-min over the
# 2-year default window:
#   02336000  Chattahoochee River at Atlanta, GA  -> clean, well-maintained
#                                                     urban river (median ~6 FNU)
#   08181500  Medina Rv at San Antonio, TX        -> flashy, wide dynamic range,
#                                                     storm-driven spikes to ~950
#   05082500  Red River of the North, Grand Forks -> larger northern river,
#                                                     higher baseline, cold-season
#                                                     gaps (median ~21 FNU)
# ---------------------------------------------------------------------------
TURBIDITY_PARAM = "63680"
DEFAULT_SITES: tuple[str, ...] = ("02336000", "08181500", "05082500")
DEFAULT_START = "2023-07-01"
DEFAULT_END = "2025-07-01"
DEFAULT_OUTDIR = Path("data/raw")


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


@dataclass
class SiteResult:
    """Structured summary of one site's pulled series."""

    site_no: str
    station_nm: str
    n_obs: int
    median_dt_min: float
    span_days: float
    completeness_pct: float
    nan_pct: float
    longest_gap_hr: float
    value_min: float
    value_median: float
    value_max: float
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
    idx = pd.DatetimeIndex(tidy["datetime"])
    median_dt, span_days, completeness, nan_pct, longest_gap = summarise_series(
        idx, tidy["value"]
    )

    out_path: Path | None = None
    if write:
        cfg.outdir.mkdir(parents=True, exist_ok=True)
        out_path = cfg.outdir / f"{site}_turbidity_{cfg.param_cd}.csv"
        tidy.to_csv(out_path, index=False)

    return SiteResult(
        site_no=site,
        station_nm=_station_name(site),
        n_obs=len(tidy),
        median_dt_min=median_dt,
        span_days=span_days,
        completeness_pct=completeness,
        nan_pct=nan_pct,
        longest_gap_hr=longest_gap,
        value_min=float(tidy["value"].min()),
        value_median=float(tidy["value"].median()),
        value_max=float(tidy["value"].max()),
        out_path=out_path,
    )


def pull_all(cfg: PullConfig, write: bool = True) -> list[SiteResult]:
    """Pull every site in ``cfg`` and return the list of results."""
    results: list[SiteResult] = []
    for site in cfg.sites:
        print(f"Pulling {site} ({cfg.param_cd}) {cfg.start} -> {cfg.end} ...")
        res = pull_site(site, cfg, write=write)
        loc = f" -> {res.out_path}" if res.out_path else ""
        print(
            f"  {res.n_obs:,} obs | {res.median_dt_min:.0f}-min | "
            f"{res.completeness_pct:.1f}% complete | "
            f"range {res.value_min:.1f}-{res.value_max:.1f} FNU{loc}"
        )
        results.append(res)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pull USGS NWIS continuous turbidity.")
    parser.add_argument("--sites", nargs="+", default=list(DEFAULT_SITES),
                        help="USGS site numbers (default: recommended 3-gauge set).")
    parser.add_argument("--start", default=DEFAULT_START, help="ISO start date.")
    parser.add_argument("--end", default=DEFAULT_END, help="ISO end date.")
    parser.add_argument("--param", default=TURBIDITY_PARAM, help="NWIS parameter code.")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), type=Path,
                        help="Directory for output CSVs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be pulled and exit without downloading.")
    args = parser.parse_args(argv)

    cfg = PullConfig(
        sites=tuple(args.sites), start=args.start, end=args.end,
        param_cd=args.param, outdir=args.outdir,
    )
    _validate(cfg)

    if args.dry_run:
        print("DRY RUN — nothing will be downloaded.")
        print(f"  param : {cfg.param_cd} (turbidity, FNU)")
        print(f"  window: {cfg.start} -> {cfg.end}")
        print(f"  outdir: {cfg.outdir}")
        for s in cfg.sites:
            print(f"  site  : {s} -> {cfg.outdir / f'{s}_turbidity_{cfg.param_cd}.csv'}")
        return 0

    pull_all(cfg, write=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
