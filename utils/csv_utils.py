"""Helpers for resilient CSV handling used across the project."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def _resolve_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _backup_path(path: Path, suffix: str = "bak") -> Path:
    """Return a non-conflicting backup path for *path* using *suffix*."""

    candidate = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{suffix}{counter}{path.suffix}")
        counter += 1
    return candidate


def ensure_csv_integrity(path: str | Path, expected_header: Sequence[str]) -> None:
    """Validate that *path* has consistent rows, repairing it if required."""

    path_obj = _resolve_path(path)
    if not path_obj.exists():
        return

    try:
        with path_obj.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
    except Exception as exc:  # pragma: no cover - best effort logging
        logging.warning("csv | %s | failed integrity read: %s", path_obj, exc)
        return

    if not rows:
        return

    expected_len = len(expected_header) if expected_header else len(rows[0])
    corrupted = False
    for idx, row in enumerate(rows[1:], start=2):
        if not row or all(field == "" for field in row):
            continue
        if len(row) != expected_len:
            logging.warning(
                "csv | %s | malformed row %d detected; rebuilding file", path_obj.name, idx
            )
            corrupted = True
            break

    if not corrupted:
        return

    backup = _backup_path(path_obj)
    try:
        path_obj.rename(backup)
    except OSError as exc:  # pragma: no cover - best effort logging
        logging.warning("csv | %s | failed to create backup: %s", path_obj.name, exc)
        return

    try:
        with path_obj.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(list(expected_header))
    except Exception as exc:  # pragma: no cover - best effort logging
        logging.error("csv | %s | failed to rebuild header: %s", path_obj.name, exc)
        try:
            if not path_obj.exists() and backup.exists():
                backup.rename(path_obj)
        except Exception:
            pass
        return

    logging.info(
        "csv | %s | corrupted log backed up to %s and reinitialised",
        path_obj.name,
        backup.name,
    )


def read_csv_safe(
    path: str | Path,
    *,
    expected_columns: Sequence[str] | None = None,
    on_empty_init: bool = False,
) -> pd.DataFrame:
    """Load *path* ignoring malformed rows and optionally enforce a schema."""

    path_obj = _resolve_path(path)
    columns: list[str] | None = list(expected_columns) if expected_columns else None

    if not path_obj.exists():
        return pd.DataFrame(columns=columns)

    try:
        df = pd.read_csv(path_obj, on_bad_lines="skip")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)
    except FileNotFoundError:
        return pd.DataFrame(columns=columns)
    except pd.errors.ParserError as exc:
        logging.warning("csv | %s | parser error, skipping bad lines: %s", path_obj.name, exc)
        try:
            df = pd.read_csv(path_obj, on_bad_lines="skip", engine="python")
        except Exception as inner_exc:  # pragma: no cover - best effort logging
            logging.error("csv | %s | failed to recover after parser error: %s", path_obj.name, inner_exc)
            if columns is not None:
                ensure_csv_integrity(path_obj, columns)
            return pd.DataFrame(columns=columns)
    except Exception as exc:  # pragma: no cover - best effort logging
        logging.warning("csv | %s | read failed: %s", path_obj.name, exc)
        return pd.DataFrame(columns=columns)

    if df.empty and columns is not None and on_empty_init:
        ensure_csv_integrity(path_obj, columns)
        return pd.DataFrame(columns=columns)

    if columns is not None:
        for col in columns:
            if col not in df.columns:
                df[col] = pd.Series(dtype="object")
        df = df[columns]

    return df


def read_multiple_csv_safe(paths: Iterable[str | Path], expected_columns: Sequence[str]) -> list[pd.DataFrame]:
    """Return list of DataFrames loaded with :func:`read_csv_safe`."""

    return [read_csv_safe(path, expected_columns=expected_columns) for path in paths]
