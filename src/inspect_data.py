"""Load and summarise a water-quality time series.

Reads a CSV with a ``datetime`` column + numeric column(s) and produces a
summary matching the contract in CLAUDE.md §5 / §7:

  rows, time range, inferred frequency, NaN count/%, per-column min/max/mean/std.

Also enforces the CSV column contracts used across Phase 1:

  - series  (raw / clean / injected): ``datetime``, ``value``  (+ optional extras)
  - labels  (injected ``*_labels.csv``): ``datetime``, ``is_anomaly``,
    ``anomaly_type``, ``true_value``

CLI
---
    python -m src.inspect_data summarise path/to.csv
    python -m src.inspect_data validate path/to.csv
    python -m src.inspect_data validate-labels path/to_labels.csv

Usage from Python
-----------------
    from src.inspect_data import load_series, summarise_series, validate_series_frame

    df = load_series("data/raw/02336000_turbidity_63680.csv")
    summary = summarise_series(df)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Column contracts (CLAUDE.md §5)
# ---------------------------------------------------------------------------
DATETIME_COL = "datetime"
VALUE_COL = "value"

SERIES_REQUIRED_COLS: tuple[str, ...] = (DATETIME_COL, VALUE_COL)
LABELS_REQUIRED_COLS: tuple[str, ...] = (
    DATETIME_COL,
    "is_anomaly",
    "anomaly_type",
    "true_value",
)
ANOMALY_TYPES: frozenset[str] = frozenset(
    {"spike", "plateau", "level_shift", "gap", "drift", ""}
)


class ContractError(ValueError):
    """Raised when a CSV / frame does not satisfy a Phase-1 data contract."""


# ---------------------------------------------------------------------------
# Summary return type
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ColumnStats:
    """Per-numeric-column summary stats."""

    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    n_nan: int
    pct_nan: float


@dataclass(frozen=True)
class SeriesSummary:
    """Structured summary of one loaded series (CLAUDE.md §7 ``inspect_dataset``)."""

    n_rows: int
    time_start: str | None
    time_end: str | None
    inferred_frequency: str | None
    median_dt_minutes: float | None
    n_nan: int
    pct_nan: float
    columns: dict[str, ColumnStats]
    value_column: str = VALUE_COL
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict (nested dataclasses flattened)."""
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# CSV contract helpers
# ---------------------------------------------------------------------------
def _require_columns(df: pd.DataFrame, required: tuple[str, ...], kind: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ContractError(
            f"{kind} CSV missing required column(s) {missing}; "
            f"got columns={list(df.columns)}"
        )


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Coerce common CSV bool encodings (True/False, 0/1, yes/no) to bool."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "t": True,
        "f": False,
    }
    if pd.api.types.is_numeric_dtype(series):
        as_num = pd.to_numeric(series, errors="coerce")
        if as_num.isna().any() or not as_num.isin([0, 1]).all():
            raise ContractError(
                "is_anomaly must be boolean (or 0/1); found non-binary numeric values."
            )
        return as_num.astype(bool)

    lowered = series.astype(str).str.strip().str.lower()
    unknown = sorted({v for v in lowered.unique() if v not in mapping})
    if unknown:
        raise ContractError(
            f"is_anomaly has unrecognised value(s) {unknown}; "
            "expected true/false, 0/1, or yes/no."
        )
    return lowered.map(mapping).astype(bool)


def _parse_datetime_column(series: pd.Series) -> pd.Series:
    """Parse ``datetime`` to timezone-naive timestamps; fail on all-NaT."""
    parsed = pd.to_datetime(series, errors="coerce", utc=False)
    # If values were tz-aware ISO strings, pandas may produce tz-aware; drop tz.
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    n_bad = int(parsed.isna().sum())
    if n_bad == len(parsed):
        raise ContractError("datetime column could not be parsed (all values invalid).")
    if n_bad > 0:
        raise ContractError(
            f"datetime column has {n_bad} unparseable value(s); fix or drop those rows."
        )
    return parsed


def validate_series_frame(df: pd.DataFrame, *, value_col: str = VALUE_COL) -> pd.DataFrame:
    """Validate / normalise a series frame to the ``datetime`` / ``value`` contract.

    - Requires ``datetime`` and ``value_col`` (default ``value``).
    - Extra columns (e.g. ``qualifier`` on raw USGS files) are kept.
    - ``datetime`` becomes timezone-naive ``datetime64[ns]``, sorted ascending.
    - ``value`` is coerced to float (invalid entries -> NaN).

    Returns a new DataFrame; does not mutate ``df``.
    Raises :class:`ContractError` on contract violations.
    """
    if df.empty:
        raise ContractError("series CSV has zero rows.")

    required = (DATETIME_COL, value_col)
    _require_columns(df, required, "series")

    out = df.copy()
    out[DATETIME_COL] = _parse_datetime_column(out[DATETIME_COL])
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")

    if not out[DATETIME_COL].is_monotonic_increasing:
        out = out.sort_values(DATETIME_COL)
    if out[DATETIME_COL].duplicated().any():
        n_dup = int(out[DATETIME_COL].duplicated().sum())
        raise ContractError(
            f"datetime column has {n_dup} duplicate timestamp(s); "
            "series contract expects unique timestamps."
        )

    return out.reset_index(drop=True)


def validate_labels_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate / normalise an injected labels frame (CLAUDE.md §5).

    Required columns: ``datetime``, ``is_anomaly``, ``anomaly_type``, ``true_value``.
    ``anomaly_type`` must be one of the five failure types, or empty.
    """
    if df.empty:
        raise ContractError("labels CSV has zero rows.")

    _require_columns(df, LABELS_REQUIRED_COLS, "labels")

    out = df.copy()
    out[DATETIME_COL] = _parse_datetime_column(out[DATETIME_COL])

    out["is_anomaly"] = _coerce_bool(out["is_anomaly"])

    types = out["anomaly_type"].fillna("").astype(str).str.strip()
    bad = sorted({t for t in types.unique() if t not in ANOMALY_TYPES})
    if bad:
        raise ContractError(
            f"anomaly_type has unknown value(s) {bad}; "
            f"allowed={sorted(ANOMALY_TYPES - {''})} or empty."
        )
    out["anomaly_type"] = types
    out["true_value"] = pd.to_numeric(out["true_value"], errors="coerce")

    if not out[DATETIME_COL].is_monotonic_increasing:
        out = out.sort_values(DATETIME_COL)

    return out.reset_index(drop=True)


def validate_series_csv(path: str | Path, *, value_col: str = VALUE_COL) -> pd.DataFrame:
    """Load a CSV from disk and validate it against the series contract."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"series CSV not found: {path}")
    raw = pd.read_csv(path)
    return validate_series_frame(raw, value_col=value_col)


def validate_labels_csv(path: str | Path) -> pd.DataFrame:
    """Load a labels CSV from disk and validate it against the labels contract."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"labels CSV not found: {path}")
    raw = pd.read_csv(path)
    return validate_labels_frame(raw)


# ---------------------------------------------------------------------------
# Load + summarise
# ---------------------------------------------------------------------------
def load_series(
    path: str | Path,
    *,
    value_col: str = VALUE_COL,
    reindex: bool = False,
) -> pd.DataFrame:
    """Load a series CSV, enforce the ``datetime``/``value`` contract, return a frame.

    If ``reindex`` is True, re-index onto a regular grid at the inferred median
    step so missing timestamps become explicit NaN rows (see ``pull_usgs`` notes:
    raw USGS files omit gap rows entirely).
    """
    df = validate_series_csv(path, value_col=value_col)
    if reindex:
        df = reindex_to_grid(df, value_col=value_col)
    return df


def reindex_to_grid(
    df: pd.DataFrame,
    *,
    value_col: str = VALUE_COL,
    freq: str | pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Re-index onto a regular time grid so gaps appear as NaN in ``value``.

    Extra columns are forward-filled only for the datetime index alignment; numeric
    value stays NaN in missing slots. ``freq`` defaults to the median timestep.
    """
    if len(df) < 2:
        return df.copy()

    indexed = df.set_index(DATETIME_COL)
    if freq is None:
        diffs = indexed.index.to_series().diff().dropna()
        if diffs.empty:
            return df.copy()
        # Prefer the most common positive step so a few gaps don't skew the grid.
        positive = diffs[diffs > pd.Timedelta(0)]
        if positive.empty:
            return df.copy()
        step = positive.value_counts().index[0]
        freq = step

    full_idx = pd.date_range(indexed.index.min(), indexed.index.max(), freq=freq)
    reindexed = indexed.reindex(full_idx)
    reindexed.index.name = DATETIME_COL
    out = reindexed.reset_index()
    # Ensure value column is float (reindex may introduce all-NaN object cols).
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    return out


def _infer_frequency(idx: pd.DatetimeIndex) -> tuple[str | None, float | None]:
    """Return (human-readable freq label, median step in minutes)."""
    if len(idx) < 2:
        return None, None
    diffs = idx.to_series().diff().dropna()
    if diffs.empty:
        return None, None
    median_td = diffs.median()
    median_min = float(median_td.total_seconds() / 60.0)

    # Prefer pandas' infer_freq when the series is regular enough.
    inferred = pd.infer_freq(idx)
    if inferred:
        return inferred, median_min

    # Fall back to a readable median label (e.g. "~15min").
    if median_min >= 1440 and abs(median_min / 1440 - round(median_min / 1440)) < 0.05:
        days = round(median_min / 1440)
        return f"~{days}D", median_min
    if median_min >= 60 and abs(median_min / 60 - round(median_min / 60)) < 0.05:
        hours = round(median_min / 60)
        return f"~{hours}h", median_min
    return f"~{median_min:.0f}min", median_min


def _column_stats(series: pd.Series) -> ColumnStats:
    numeric = pd.to_numeric(series, errors="coerce")
    n = len(numeric)
    n_nan = int(numeric.isna().sum())
    pct = 100.0 * n_nan / n if n else 0.0
    valid = numeric.dropna()
    if valid.empty:
        return ColumnStats(None, None, None, None, n_nan, pct)
    return ColumnStats(
        min=float(valid.min()),
        max=float(valid.max()),
        mean=float(valid.mean()),
        std=float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
        n_nan=n_nan,
        pct_nan=pct,
    )


def summarise_series(
    df: pd.DataFrame,
    *,
    value_col: str = VALUE_COL,
    path: str | Path | None = None,
) -> SeriesSummary:
    """Build a :class:`SeriesSummary` for a contract-valid series frame.

    Summarises every numeric column; NaN stats on the summary root refer to
    ``value_col`` (the column processed in a QC run).
    """
    if DATETIME_COL not in df.columns:
        raise ContractError(f"frame missing {DATETIME_COL!r}; load via load_series first.")
    if value_col not in df.columns:
        raise ContractError(f"frame missing value column {value_col!r}.")

    idx = pd.DatetimeIndex(df[DATETIME_COL])
    freq_label, median_min = _infer_frequency(idx)

    value_stats = _column_stats(df[value_col])
    col_stats: dict[str, ColumnStats] = {}
    for col in df.columns:
        if col == DATETIME_COL:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or col == value_col:
            col_stats[col] = _column_stats(df[col])

    time_start = idx.min().isoformat() if len(idx) else None
    time_end = idx.max().isoformat() if len(idx) else None

    return SeriesSummary(
        n_rows=len(df),
        time_start=time_start,
        time_end=time_end,
        inferred_frequency=freq_label,
        median_dt_minutes=median_min,
        n_nan=value_stats.n_nan,
        pct_nan=value_stats.pct_nan,
        columns=col_stats,
        value_column=value_col,
        path=str(path) if path is not None else None,
    )


def inspect_file(
    path: str | Path,
    *,
    value_col: str = VALUE_COL,
    reindex: bool = False,
) -> SeriesSummary:
    """Convenience: load + summarise a series CSV in one call."""
    path = Path(path)
    df = load_series(path, value_col=value_col, reindex=reindex)
    return summarise_series(df, value_col=value_col, path=path)


# ---------------------------------------------------------------------------
# Pretty printing + CLI
# ---------------------------------------------------------------------------
def format_summary(summary: SeriesSummary) -> str:
    """Human-readable multi-line summary for terminal inspection."""
    lines = [
        f"path              : {summary.path or '(in-memory)'}",
        f"n_rows            : {summary.n_rows:,}",
        f"time_start        : {summary.time_start}",
        f"time_end          : {summary.time_end}",
        f"inferred_frequency: {summary.inferred_frequency}",
        f"median_dt_minutes : {summary.median_dt_minutes}",
        f"n_nan ({summary.value_column}): {summary.n_nan:,}",
        f"pct_nan ({summary.value_column}): {summary.pct_nan:.2f}%",
        "columns:",
    ]
    for name, stats in summary.columns.items():
        lines.append(
            f"  {name}: min={stats.min} max={stats.max} "
            f"mean={stats.mean} std={stats.std} "
            f"nan={stats.n_nan} ({stats.pct_nan:.2f}%)"
        )
    return "\n".join(lines)


def _cmd_summarise(args: argparse.Namespace) -> int:
    summary = inspect_file(args.path, value_col=args.value_col, reindex=args.reindex)
    if args.json:
        print(json.dumps(summary.to_dict(), indent=2, default=str))
    else:
        print(format_summary(summary))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    df = validate_series_csv(args.path, value_col=args.value_col)
    extras = [c for c in df.columns if c not in {DATETIME_COL, args.value_col}]
    print(f"OK  series contract: {args.path}")
    print(f"    rows={len(df):,}  columns={list(df.columns)}")
    if extras:
        print(f"    extra columns kept: {extras}")
    return 0


def _cmd_validate_labels(args: argparse.Namespace) -> int:
    df = validate_labels_csv(args.path)
    n_anom = int(df["is_anomaly"].sum())
    print(f"OK  labels contract: {args.path}")
    print(f"    rows={len(df):,}  anomalies={n_anom:,}")
    type_counts = (
        df.loc[df["is_anomaly"], "anomaly_type"].value_counts().to_dict()
        if n_anom
        else {}
    )
    if type_counts:
        print(f"    by type: {type_counts}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect / validate water-quality series CSVs (datetime + value).",
    )
    parser.add_argument(
        "--value-col",
        default=VALUE_COL,
        help=f"Numeric value column name (default: {VALUE_COL}).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_sum = sub.add_parser("summarise", help="Load a series CSV and print its summary.")
    p_sum.add_argument("path", type=Path, help="Path to series CSV.")
    p_sum.add_argument(
        "--reindex",
        action="store_true",
        help="Re-index onto a regular grid so missing timestamps become NaN.",
    )
    p_sum.add_argument("--json", action="store_true", help="Emit summary as JSON.")
    p_sum.set_defaults(func=_cmd_summarise)

    p_val = sub.add_parser(
        "validate",
        help="Check that a series CSV satisfies the datetime/value contract.",
    )
    p_val.add_argument("path", type=Path, help="Path to series CSV.")
    p_val.set_defaults(func=_cmd_validate)

    p_lab = sub.add_parser(
        "validate-labels",
        help="Check that a labels CSV satisfies the §5 labels contract.",
    )
    p_lab.add_argument("path", type=Path, help="Path to labels CSV.")
    p_lab.set_defaults(func=_cmd_validate_labels)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ContractError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
