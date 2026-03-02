from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_parquet_dataset(parquet_dir: Path, columns: list[str] | None = None) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    dataset = ds.dataset(parquet_dir, format="parquet")
    table = dataset.to_table(columns=columns)

    # Decode any dictionary-encoded columns (fixes pandas NotImplementedError)
    for i, field in enumerate(table.schema):
        if pa.types.is_dictionary(field.type):
            decoded = pc.dictionary_decode(table.column(i))
            table = table.set_column(i, field.name, decoded)

    return table.to_pandas()


def ensure_flight_date(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer FL_DATE if present and parsed; otherwise construct from Y/M/D
    if "FL_DATE" in df.columns:
        # If FL_DATE is already datetime64[ns], to_datetime is cheap
        df["flight_date"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    else:
        df["flight_date"] = pd.NaT

    missing = df["flight_date"].isna()
    if missing.any():
        # Derive from YEAR/MONTH/DAY_OF_MONTH
        derived = pd.to_datetime(
            dict(
                year=pd.to_numeric(df.loc[missing, "YEAR"], errors="coerce"),
                month=pd.to_numeric(df.loc[missing, "MONTH"], errors="coerce"),
                day=pd.to_numeric(df.loc[missing, "DAY_OF_MONTH"], errors="coerce"),
            ),
            errors="coerce",
        )
        df.loc[missing, "flight_date"] = derived

    # Hard fail if still missing (can’t do time splits without a date)
    if df["flight_date"].isna().any():
        bad = df[df["flight_date"].isna()][["YEAR", "MONTH", "DAY_OF_MONTH"]].head(10)
        raise ValueError(f"Unable to construct flight_date for some rows. Examples:\n{bad}")

    return df


def build_labels(df: pd.DataFrame, exclude_cancelled_diverted: bool = True) -> pd.DataFrame:
    # Optional filters (recommended for v1): remove cancelled/diverted
    if exclude_cancelled_diverted:
        if "CANCELLED" in df.columns:
            df = df[df["CANCELLED"].fillna(0).astype(int) == 0]
        if "DIVERTED" in df.columns:
            df = df[df["DIVERTED"].fillna(0).astype(int) == 0]

    # ARR_DELAY can be NaN even when not cancelled (rare); drop for training label integrity
    df["arr_delay_minutes"] = pd.to_numeric(df["ARR_DELAY"], errors="coerce").astype("float32")
    df = df[~df["arr_delay_minutes"].isna()].copy()

    # Binary label: arrival delay >= 15 minutes
    df["delayed_15"] = (df["arr_delay_minutes"] >= 15.0).astype("int8")

    return df


def write_partitioned_parquet(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep partitions by YEAR/MONTH for consistency
    df.to_parquet(out_dir, index=False, engine="pyarrow", partition_cols=["YEAR", "MONTH"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build labels for BTS On-Time data.")
    parser.add_argument("--in-dir", type=str, default="data/interim/bts_on_time_parquet")
    parser.add_argument("--out-dir", type=str, default="data/interim/bts_on_time_labeled")
    parser.add_argument("--include-cancelled-diverted", action="store_true")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    # Load only what we need + a few useful columns
    cols = [
        "YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK",
        "OP_UNIQUE_CARRIER", "ORIGIN", "DEST",
        "CRS_DEP_TIME", "CRS_ARR_TIME",
        "ARR_DELAY", "DISTANCE",
        "FL_DATE", "CANCELLED", "DIVERTED",
        "CRS_DEP_MINUTES", "CRS_ARR_MINUTES", "CRS_DEP_HOUR", "CRS_ARR_HOUR",
        "SOURCE_FILE",
    ]
    # Some columns may not exist depending on your earlier ingest settings. We'll read broadly then trim.
    df = load_parquet_dataset(in_dir, columns=None)
    # Standardize carrier column name
    if "OP_CARRIER" not in df.columns and "OP_UNIQUE_CARRIER" in df.columns:
        df = df.rename(columns={"OP_UNIQUE_CARRIER": "OP_CARRIER"})

    df = ensure_flight_date(df)

    # Derive YEAR/MONTH for consistent partitioning (do NOT rely on partition keys)
    df["YEAR"] = df["flight_date"].dt.year.astype("Int16")
    df["MONTH"] = df["flight_date"].dt.month.astype("Int8")

    print(df[["flight_date","YEAR","MONTH"]].head())
    print(df[["YEAR","MONTH"]].drop_duplicates().head(10))

    exclude = not args.include_cancelled_diverted
    labeled = build_labels(df, exclude_cancelled_diverted=exclude)

    # Basic sanity prints
    print("Labeled shape:", labeled.shape)
    print("Date range:", labeled["flight_date"].min(), "to", labeled["flight_date"].max())
    print("Delay rate:", labeled["delayed_15"].mean())

    write_partitioned_parquet(labeled, out_dir)
    print(f"Wrote labeled dataset to: {out_dir}")


if __name__ == "__main__":
    main()