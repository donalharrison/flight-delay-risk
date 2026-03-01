from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


# -------------------------
# Configuration
# -------------------------

REQUIRED_COLUMNS = [
    "YEAR",
    "MONTH",
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "ARR_DELAY",
    "DISTANCE",
]

OPTIONAL_COLUMNS = [
    "FL_DATE",
    "OP_CARRIER_FL_NUM",
    "CANCELLED",
    "DIVERTED",
]

# Columns we will output (keep this stable for downstream feature engineering)
OUTPUT_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


DTYPES = {
    "YEAR": "Int16",
    "MONTH": "Int8",
    "DAY_OF_MONTH": "Int8",
    "DAY_OF_WEEK": "Int8",
    "OP_UNIQUE_CARRIER": "string",
    "ORIGIN": "string",
    "DEST": "string",
    "CRS_DEP_TIME": "Int32",
    "CRS_ARR_TIME": "Int32",
    "ARR_DELAY": "float32",
    "DISTANCE": "float32",
    "FL_DATE": "string",  # parse later, after concatenation
    "OP_CARRIER_FL_NUM": "string",
    "CANCELLED": "Int8",
    "DIVERTED": "Int8",
}


@dataclass(frozen=True)
class IngestPaths:
    raw_root: Path
    interim_root: Path


# -------------------------
# Helpers
# -------------------------

def find_csv_files(raw_root: Path) -> list[Path]:
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")
    files = sorted([p for p in raw_root.rglob("*.csv") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No CSV files found under: {raw_root}")
    return files


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize header naming and strip whitespace
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def ensure_columns(df: pd.DataFrame, source_file: Path) -> pd.DataFrame:
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Missing required columns in {source_file.name}: {missing_required}"
        )

    # Add optional columns if missing
    for c in OPTIONAL_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    # Keep only expected output columns (drop extras)
    extra = [c for c in df.columns if c not in OUTPUT_COLUMNS]
    if extra:
        df = df.drop(columns=extra)

    # Ensure consistent column ordering
    df = df[OUTPUT_COLUMNS]
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # First, coerce numeric columns safely; BTS may have blanks
    # Use to_numeric for safety before astype
    for col in ["YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "CRS_DEP_TIME", "CRS_ARR_TIME", "CANCELLED", "DIVERTED"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(DTYPES[col])

    for col in ["ARR_DELAY", "DISTANCE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(DTYPES[col])

    for col in ["OP_CARRIER", "ORIGIN", "DEST", "FL_DATE", "OP_CARRIER_FL_NUM"]:
        if col in df.columns:
            df[col] = df[col].astype(DTYPES[col]).str.strip()

    return df


def parse_flight_date(df: pd.DataFrame) -> pd.DataFrame:
    # If FL_DATE exists (string), parse to datetime64[ns]
    if "FL_DATE" in df.columns:
        # FL_DATE usually formatted YYYY-MM-DD
        df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    return df


def time_int_to_minutes(val: pd.Series) -> pd.Series:
    """
    BTS scheduled times come as local HHMM integers:
      5 -> 00:05
      945 -> 09:45
      2359 -> 23:59
    Convert to minutes since midnight.
    """
    v = pd.to_numeric(val, errors="coerce")
    # Preserve missing
    hhmm = v.astype("Int32")
    # Left pad via integer math
    hours = (hhmm // 100).astype("float32")
    mins = (hhmm % 100).astype("float32")
    out = hours * 60.0 + mins
    # If mins >= 60, data is malformed; set NaN
    out = out.where(mins < 60, other=pd.NA)
    return out.astype("float32")


def add_derived_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # Minutes since midnight from scheduled times (useful downstream, no leakage)
    df["CRS_DEP_MINUTES"] = time_int_to_minutes(df["CRS_DEP_TIME"])
    df["CRS_ARR_MINUTES"] = time_int_to_minutes(df["CRS_ARR_TIME"])

    # Hour-of-day derived for convenience
    df["CRS_DEP_HOUR"] = (df["CRS_DEP_MINUTES"] // 60).astype("Int8")
    df["CRS_ARR_HOUR"] = (df["CRS_ARR_MINUTES"] // 60).astype("Int8")
    return df


def read_one_csv(path: Path, usecols: list[str] | None, chunksize: int | None) -> pd.DataFrame:
    # Read with pandas; allow large files
    # If chunksize is set, we stream and concatenate (still can be large).
    if chunksize:
        chunks = []
        for ch in pd.read_csv(path, usecols=usecols, low_memory=False, chunksize=chunksize):
            chunks.append(ch)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(path, usecols=usecols, low_memory=False)

    df = normalize_columns(df)
    df = ensure_columns(df, path)
    df = coerce_types(df)

    # Helpful metadata
    df["SOURCE_FILE"] = path.name
    return df


def write_partitioned_parquet(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Partition by YEAR/MONTH for efficient training loads
    df.to_parquet(
        out_dir,
        index=False,
        engine="pyarrow",
        partition_cols=["YEAR", "MONTH"],
    )


def write_single_parquet(df: pd.DataFrame, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file, index=False, engine="pyarrow")


# -------------------------
# Main
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest BTS On-Time Performance monthly CSVs into a single dataset.")
    parser.add_argument("--raw-root", type=str, default="data/raw/bts_on_time", help="Root folder containing monthly CSVs.")
    parser.add_argument("--out-dir", type=str, default="data/interim/bts_on_time_parquet", help="Output directory for partitioned parquet dataset.")
    parser.add_argument("--out-file", type=str, default="", help="Optional: write a single combined parquet file to this path (can be huge).")
    parser.add_argument("--chunksize", type=int, default=0, help="Optional: read CSVs in chunks (e.g., 500000) to reduce memory spikes.")
    parser.add_argument("--no-derived-time", action="store_true", help="Do not create CRS_*_MINUTES and CRS_*_HOUR derived columns.")
    args = parser.parse_args()

    paths = IngestPaths(
        raw_root=Path(args.raw_root),
        interim_root=Path(args.out_dir),
    )

    files = find_csv_files(paths.raw_root)
    print(f"Found {len(files)} CSV files under {paths.raw_root}")

    # We read all columns listed in OUTPUT_COLUMNS (case-insensitive handled after read)
    # But usecols in read_csv is case-sensitive, so we skip it and filter after normalize.
    # If you want usecols optimization, keep local copies with consistent headers.
    chunksize = args.chunksize if args.chunksize > 0 else None

    dfs: list[pd.DataFrame] = []
    for i, f in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Reading {f}")
        df = read_one_csv(f, usecols=None, chunksize=chunksize)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Parse date if present, and add derived time features
    combined = parse_flight_date(combined)
    if not args.no_derived_time:
        combined = add_derived_time_features(combined)

    # Basic sanity stats
    print("Combined shape:", combined.shape)
    print("Date range (if FL_DATE present):", combined["FL_DATE"].min() if "FL_DATE" in combined.columns else "n/a",
          "to", combined["FL_DATE"].max() if "FL_DATE" in combined.columns else "n/a")

    # Write outputs
    write_partitioned_parquet(combined, paths.interim_root)
    print(f"Wrote partitioned parquet dataset to: {paths.interim_root}")

    if args.out_file:
        out_file = Path(args.out_file)
        write_single_parquet(combined, out_file)
        print(f"Wrote single parquet file to: {out_file}")


if __name__ == "__main__":
    main()