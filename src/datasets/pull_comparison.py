"""Pull the same turbidity sensor in BOTH approval states, into ``data/comparison/``.

Why this is a *split*, not a *pair*
-----------------------------------
NWIS serves the **current** state of each record. When a period is approved, the
provisional values it used to carry are overwritten in place — USGS publishes no
archive of superseded provisional values, and the instantaneous-values service
has no revision/as-of parameter. **So the same timestamps cannot be retrieved in
two states.** A true before/after pair would have to have been captured live,
before approval, and kept.

What a live record *does* give you is a moving **approval boundary**: everything
older than the boundary carries qualifier ``A`` (approved — fouling/calibration
corrections applied and clearly-bad data deleted per TM 1-D3), everything newer
carries ``P`` (provisional — as the sensor reported it, uncorrected). Both come
from the same site, the same sensor, and the same sampling grid. That is the
comparison this module assembles: not a paired diff, but two spans of one series
that differ in whether USGS record processing has been applied.

Read the two sides as *distributions*, not as row-for-row counterparts — they are
different calendar periods, so seasonality confounds a naive comparison. Each
manifest therefore records a ``season_matched_approved_window``: the same calendar
dates one year before the provisional span, which is the fairer slice of the
approved side to compare against.

Gauges (geography + turbidity-regime spread, 15-min, both spans dense)
----------------------------------------------------------------------
``03447687`` French Broad R nr Fletcher, NC — Southern Appalachian mountain river;
moderate regime (median ~8 FNU).
``02198840`` Savannah R at I-95 nr Port Wentworth, GA — Atlantic tidal coastal
plain; moderate-high (median ~13 FNU).
``06818000`` Missouri R at St Joseph, MO — Great Plains large river; high
(median ~24 FNU).

Selected by :mod:`scratchpad.screen_prov_vs_approved` from a 24-gauge pool, on:
>= 1 year approved, >= 3 months provisional, consistent 15-min step, and decent
completeness on *both* sides. The project's third injection base ``08041770``
(LNVA Canal, TX) cannot be used here — its record ends 2025-11-19 with zero
provisional rows.

Output layout (regenerable; gitignored like ``data/raw/``)
-----------------------------------------------------------
``data/comparison/<site>/``
  ``<site>_turbidity_63680_approved.csv``     datetime, value, qualifier
  ``<site>_turbidity_63680_provisional.csv``  datetime, value, qualifier
  ``<site>_comparison_manifest.json``         boundary, per-side stats, caveats

CLI
---
    python -m src.datasets.pull_comparison --dry-run
    python -m src.datasets.pull_comparison
    python -m src.datasets.pull_comparison --sites 02336000 --start 2022-01-01
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.pull_usgs import (
    APPROVED_PREFIX,
    TURBIDITY_PARAM,
    _fetch_iv_with_retry,
    _station_name,
    longest_unbroken_run_days,
    tidy_frame,
)

DEFAULT_OUTDIR = Path("data/comparison")
DEFAULT_START = "2023-07-01"
# Open-ended: the provisional tail is whatever has not been approved yet, so the
# end of the window must track "now" rather than a frozen date.
DEFAULT_END = pd.Timestamp.today().strftime("%Y-%m-%d")

# site -> short description of what it contributes to the spread.
COMPARISON_SITES: dict[str, str] = {
    "03447687": "S. Appalachian mountain river (NC) — moderate regime",
    "02198840": "Atlantic tidal coastal plain (GA) — moderate-high regime",
    "06818000": "Great Plains large river (MO) — high regime",
}


@dataclass(frozen=True)
class ComparisonConfig:
    """Configuration for a provisional-vs-approved pull."""

    sites: tuple[str, ...] = tuple(COMPARISON_SITES)
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    param_cd: str = TURBIDITY_PARAM
    outdir: Path = DEFAULT_OUTDIR
    max_retries: int = 3
    retry_wait_s: float = 5.0
    max_gap: str = "3h"


@dataclass
class SideStats:
    """Summary of one approval side (approved or provisional) of a series."""

    state: str
    n_rows: int
    n_nan: int
    start: str
    end: str
    span_days: float
    median_dt_min: float
    completeness_pct: float
    longest_gap_hr: float
    longest_unbroken_days: float
    value_min: float
    value_median: float
    value_p95: float
    value_max: float
    qualifier_counts: dict[str, int] = field(default_factory=dict)


def split_by_approval(tidy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a tidy frame into (approved, provisional) by qualifier prefix.

    Approved is any qualifier starting ``A`` — this deliberately keeps the
    modified codes (``A e`` estimated, ``A, >`` over-range, ``A, R`` revised),
    since USGS approved those readings. Everything else (``P``, blank, NaN) is
    treated as not-yet-approved.
    """
    if "qualifier" not in tidy.columns:
        raise ValueError("tidy frame has no 'qualifier' column; cannot split by approval")
    q = tidy["qualifier"].astype("string")
    is_appr = q.str.startswith(APPROVED_PREFIX).fillna(False).to_numpy()
    approved = tidy.loc[is_appr].reset_index(drop=True)
    provisional = tidy.loc[~is_appr].reset_index(drop=True)
    return approved, provisional


def summarise_side(df: pd.DataFrame, state: str, max_gap: str = "3h") -> SideStats:
    """Compute the per-side statistics recorded in the manifest."""
    idx = pd.DatetimeIndex(df["datetime"])
    vals = df["value"]
    if len(idx) < 2:
        return SideStats(state, len(df), int(vals.isna().sum()), "-", "-", 0.0,
                         float("nan"), float("nan"), float("nan"), 0.0,
                         float("nan"), float("nan"), float("nan"), float("nan"))
    steps_min = np.diff(idx.to_numpy()) / np.timedelta64(1, "m")
    dt = float(np.median(steps_min))
    span_min = (idx.max() - idx.min()).total_seconds() / 60.0
    series = pd.Series(vals.to_numpy(), index=idx)
    return SideStats(
        state=state,
        n_rows=len(df),
        n_nan=int(vals.isna().sum()),
        start=str(idx.min()),
        end=str(idx.max()),
        span_days=span_min / 1440.0,
        median_dt_min=dt,
        completeness_pct=100.0 * len(idx) / (span_min / dt) if dt and span_min else float("nan"),
        longest_gap_hr=float(steps_min.max() / 60.0),
        longest_unbroken_days=longest_unbroken_run_days(series, pd.Timedelta(max_gap)),
        value_min=float(vals.min()),
        value_median=float(vals.median()),
        value_p95=float(vals.quantile(0.95)),
        value_max=float(vals.max()),
        qualifier_counts={str(k): int(v) for k, v in
                          df["qualifier"].astype(str).value_counts().items()},
    )


def season_matched_window(
    prov: pd.DataFrame, appr: pd.DataFrame
) -> dict[str, str] | None:
    """The approved sub-window covering the provisional span's calendar dates.

    Shifts the provisional span back in whole years until it lands inside the
    approved span, so a like-for-like (same-season) comparison is possible.
    Returns ``None`` if no shift up to 3 years fits.
    """
    if prov.empty or appr.empty:
        return None
    p0, p1 = pd.Timestamp(prov["datetime"].min()), pd.Timestamp(prov["datetime"].max())
    a0, a1 = pd.Timestamp(appr["datetime"].min()), pd.Timestamp(appr["datetime"].max())
    for years in (1, 2, 3):
        s, e = p0 - pd.DateOffset(years=years), p1 - pd.DateOffset(years=years)
        if s >= a0 and e <= a1:
            n = int(((appr["datetime"] >= s) & (appr["datetime"] <= e)).sum())
            return {"start": str(s), "end": str(e), "years_back": str(years),
                    "n_approved_rows": str(n)}
    return None


def pull_comparison_site(site: str, cfg: ComparisonConfig, write: bool = True) -> dict:
    """Pull one site, split it by approval state, and write both sides + manifest."""
    raw = _fetch_iv_with_retry(
        site, cfg.param_cd, cfg.start, cfg.end, cfg.max_retries, cfg.retry_wait_s
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"No turbidity data for {site} in {cfg.start}..{cfg.end}")

    tidy = tidy_frame(raw, cfg.param_cd)
    approved, provisional = split_by_approval(tidy)
    if approved.empty or provisional.empty:
        raise RuntimeError(
            f"{site} has {len(approved):,} approved and {len(provisional):,} provisional "
            f"rows in {cfg.start}..{cfg.end}; a comparison needs both. Widen the window "
            f"or pick another gauge (see scratchpad/screen_prov_vs_approved.py)."
        )

    boundary = pd.Timestamp(provisional["datetime"].min())
    # The split is normally a clean cut at the boundary, but a mid-record period
    # can lag approval. Count the exceptions rather than assume there are none.
    n_appr_after = int((approved["datetime"] > boundary).sum())

    a_stats = summarise_side(approved, "approved", cfg.max_gap)
    p_stats = summarise_side(provisional, "provisional", cfg.max_gap)

    manifest = {
        "site_no": site,
        "station_nm": _station_name(site),
        "role": COMPARISON_SITES.get(site, ""),
        "param_cd": cfg.param_cd,
        "param_desc": "Turbidity, FNU (63680) — continuous optical sensor",
        "pull_window": {"start": cfg.start, "end": cfg.end},
        "pulled_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "approval_boundary": str(boundary),
        "n_approved_rows_after_boundary": n_appr_after,
        "approved": a_stats.__dict__,
        "provisional": p_stats.__dict__,
        "season_matched_approved_window": season_matched_window(provisional, approved),
        "caveat": (
            "These are two spans of ONE series either side of the approval boundary, "
            "NOT the same timestamps in two states. NWIS overwrites provisional values "
            "on approval and publishes no archive of them, so a row-for-row before/after "
            "diff is not retrievable. Compare distributions, and prefer the "
            "season_matched_approved_window to control for seasonality."
        ),
    }

    if write:
        site_dir = cfg.outdir / site
        site_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{site}_turbidity_{cfg.param_cd}"
        approved.to_csv(site_dir / f"{stem}_approved.csv", index=False)
        provisional.to_csv(site_dir / f"{stem}_provisional.csv", index=False)
        (site_dir / f"{site}_comparison_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        manifest["out_dir"] = str(site_dir)

    return manifest


def pull_all(cfg: ComparisonConfig, write: bool = True) -> list[dict]:
    """Pull every configured site and print a per-side summary."""
    out: list[dict] = []
    for site in cfg.sites:
        print(f"Pulling {site} ({cfg.param_cd}) {cfg.start} -> {cfg.end} ...")
        m = pull_comparison_site(site, cfg, write=write)
        print(f"  {m['station_nm']}")
        print(f"  approval boundary: {m['approval_boundary']}"
              + (f"  (+{m['n_approved_rows_after_boundary']:,} approved rows after it)"
                 if m["n_approved_rows_after_boundary"] else ""))
        for side in ("approved", "provisional"):
            s = m[side]
            print(f"  {side:<12} {s['n_rows']:>7,} rows  {s['span_days']:>6.0f} d  "
                  f"{s['median_dt_min']:>3.0f} min  {s['completeness_pct']:>5.1f}% complete  "
                  f"median {s['value_median']:>6.1f} / p95 {s['value_p95']:>7.1f} / "
                  f"max {s['value_max']:>8.1f} FNU")
        sm = m["season_matched_approved_window"]
        print(f"  season-matched approved window: "
              + (f"{sm['start'][:10]} -> {sm['end'][:10]} "
                 f"({int(sm['n_approved_rows']):,} rows, {sm['years_back']} yr back)"
                 if sm else "none available"))
        if write:
            print(f"  -> {m['out_dir']}")
        out.append(m)
    return out


def _validate(cfg: ComparisonConfig) -> None:
    """Fail loudly on nonsensical configuration (CLAUDE.md §13)."""
    if not cfg.sites:
        raise ValueError("No sites specified.")
    for s in cfg.sites:
        if not (s.isdigit() and 8 <= len(s) <= 15):
            raise ValueError(f"Suspicious USGS site number: {s!r} (expected 8-15 digits).")
    if pd.to_datetime(cfg.end) <= pd.to_datetime(cfg.start):
        raise ValueError(f"end ({cfg.end}) must be after start ({cfg.start}).")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Pull one turbidity series per gauge in BOTH approval states."
    )
    p.add_argument("--sites", nargs="+", default=list(COMPARISON_SITES))
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--param", default=TURBIDITY_PARAM)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cfg = ComparisonConfig(
        sites=tuple(args.sites), start=args.start, end=args.end,
        param_cd=args.param, outdir=args.outdir,
    )
    _validate(cfg)

    if args.dry_run:
        print("DRY RUN — nothing will be downloaded.")
        print(f"  param : {cfg.param_cd} (turbidity, FNU)")
        print(f"  window: {cfg.start} -> {cfg.end}")
        for s in cfg.sites:
            print(f"  site  : {s} -> {cfg.outdir / s}/  ({COMPARISON_SITES.get(s, '')})")
        return 0

    pull_all(cfg, write=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
