"""Tests for src/inspect_data.py (load, summary contract, CSV contracts)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.inspect_data import (
    ContractError,
    format_summary,
    inspect_file,
    load_series,
    main,
    reindex_to_grid,
    summarise_series,
    validate_labels_csv,
    validate_labels_frame,
    validate_series_csv,
    validate_series_frame,
)


@pytest.fixture
def series_csv(tmp_path: Path) -> Path:
    """Minimal valid datetime/value series (15-min, one NaN)."""
    idx = pd.date_range("2024-01-01", periods=8, freq="15min")
    values = [1.0, 2.0, 3.0, float("nan"), 5.0, 6.0, 7.0, 8.0]
    path = tmp_path / "series.csv"
    pd.DataFrame({"datetime": idx, "value": values}).to_csv(path, index=False)
    return path


@pytest.fixture
def raw_style_csv(tmp_path: Path) -> Path:
    """Series with an extra qualifier column (raw USGS shape)."""
    idx = pd.date_range("2024-01-01", periods=4, freq="15min")
    path = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "datetime": idx,
            "value": [1.0, 2.0, 3.0, 4.0],
            "qualifier": ["A", "A", "P", "A"],
        }
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def labels_csv(tmp_path: Path) -> Path:
    idx = pd.date_range("2024-01-01", periods=4, freq="15min")
    path = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "datetime": idx,
            "is_anomaly": [False, True, False, True],
            "anomaly_type": ["", "spike", "", "gap"],
            "true_value": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------
def test_validate_series_accepts_datetime_value(series_csv: Path) -> None:
    df = validate_series_csv(series_csv)
    assert list(df.columns) == ["datetime", "value"]
    assert len(df) == 8
    assert df["datetime"].is_monotonic_increasing
    assert df["value"].isna().sum() == 1


def test_validate_series_keeps_extra_columns(raw_style_csv: Path) -> None:
    df = validate_series_csv(raw_style_csv)
    assert "qualifier" in df.columns
    assert len(df) == 4


def test_validate_series_rejects_missing_value_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"datetime": ["2024-01-01"], "x": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ContractError, match="missing required column"):
        validate_series_csv(path)


def test_validate_series_rejects_duplicate_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "dups.csv"
    pd.DataFrame(
        {
            "datetime": ["2024-01-01T00:00:00", "2024-01-01T00:00:00"],
            "value": [1.0, 2.0],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ContractError, match="duplicate"):
        validate_series_csv(path)


def test_validate_series_frame_sorts_unsorted() -> None:
    df = pd.DataFrame(
        {
            "datetime": ["2024-01-01T00:15:00", "2024-01-01T00:00:00"],
            "value": [2.0, 1.0],
        }
    )
    out = validate_series_frame(df)
    assert list(out["value"]) == [1.0, 2.0]


def test_validate_labels_ok(labels_csv: Path) -> None:
    df = validate_labels_csv(labels_csv)
    assert df["is_anomaly"].dtype == bool
    assert int(df["is_anomaly"].sum()) == 2


def test_validate_labels_rejects_bad_anomaly_type(tmp_path: Path) -> None:
    path = tmp_path / "bad_labels.csv"
    pd.DataFrame(
        {
            "datetime": ["2024-01-01"],
            "is_anomaly": [True],
            "anomaly_type": ["weird"],
            "true_value": [1.0],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ContractError, match="anomaly_type"):
        validate_labels_csv(path)


# ---------------------------------------------------------------------------
# Load + summarise
# ---------------------------------------------------------------------------
def test_load_and_summarise(series_csv: Path) -> None:
    df = load_series(series_csv)
    summary = summarise_series(df, path=series_csv)
    assert summary.n_rows == 8
    assert summary.n_nan == 1
    assert abs(summary.pct_nan - 12.5) < 1e-9
    assert summary.inferred_frequency in {"15min", "15T", "~15min"}
    assert summary.median_dt_minutes == pytest.approx(15.0)
    assert summary.time_start is not None
    assert summary.time_end is not None
    assert "value" in summary.columns
    assert summary.columns["value"].min == pytest.approx(1.0)
    assert summary.columns["value"].max == pytest.approx(8.0)


def test_inspect_file_json_serialisable(series_csv: Path) -> None:
    summary = inspect_file(series_csv)
    d = summary.to_dict()
    assert d["n_rows"] == 8
    assert "columns" in d
    assert "value" in d["columns"]


def test_reindex_exposes_gap(tmp_path: Path) -> None:
    """Missing timestamps (not NaN rows) become NaN after reindex_to_grid."""
    # 00:00, 00:15, 00:45 — missing 00:30
    path = tmp_path / "gappy.csv"
    pd.DataFrame(
        {
            "datetime": [
                "2024-01-01T00:00:00",
                "2024-01-01T00:15:00",
                "2024-01-01T00:45:00",
            ],
            "value": [1.0, 2.0, 3.0],
        }
    ).to_csv(path, index=False)
    df = load_series(path, reindex=True)
    assert len(df) == 4
    assert df["value"].isna().sum() == 1


def test_format_summary_readable(series_csv: Path) -> None:
    text = format_summary(inspect_file(series_csv))
    assert "n_rows" in text
    assert "inferred_frequency" in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_summarise(series_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["summarise", str(series_csv)]) == 0
    out = capsys.readouterr().out
    assert "n_rows" in out
    assert "8" in out


def test_cli_summarise_json(series_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["summarise", str(series_csv), "--json"]) == 0
    out = capsys.readouterr().out
    assert '"n_rows": 8' in out


def test_cli_validate(series_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", str(series_csv)]) == 0
    assert "OK" in capsys.readouterr().out


def test_cli_validate_labels(labels_csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate-labels", str(labels_csv)]) == 0
    assert "OK" in capsys.readouterr().out


def test_cli_validate_fails_loud(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"datetime": ["2024-01-01"], "nope": [1]}).to_csv(path, index=False)
    assert main(["validate", str(path)]) == 1
    assert "ERROR" in capsys.readouterr().err
