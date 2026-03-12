from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TOP10_AIRPORTS = ["ATL", "ORD", "DFW", "DEN", "LAX", "JFK", "CLT", "LAS", "PHX", "MCO"]
MAJOR4_CARRIERS = ["DL", "AA", "UA", "WN"]


def read_parquet_resilient(parquet_dir: Path, columns: list[str] | None = None) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    dataset = ds.dataset(parquet_dir, format="parquet")
    table = dataset.to_table(columns=columns)

    for i, field in enumerate(table.schema):
        if pa.types.is_dictionary(field.type):
            table = table.set_column(i, field.name, pc.dictionary_decode(table.column(i)))

    return table.to_pandas()


def write_partitioned_parquet(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        out_dir,
        index=False,
        engine="pyarrow",
        partition_cols=["YEAR", "MONTH"],
        use_dictionary=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter labeled BTS data to a reduced project scope.")
    parser.add_argument("--in-dir", type=str, default="data/interim/bts_on_time_labeled")
    parser.add_argument("--out-dir", type=str, default="data/interim/bts_on_time_labeled_reduced")
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--end-date", type=str, default="2025-11-30")
    parser.add_argument("--airports", type=str, default=",".join(TOP10_AIRPORTS))
    parser.add_argument("--carriers", type=str, default=",".join(MAJOR4_CARRIERS))
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    airports = [x.strip().upper() for x in args.airports.split(",") if x.strip()]
    carriers = [x.strip().upper() for x in args.carriers.split(",") if x.strip()]

    df = read_parquet_resilient(in_dir, columns=None)

    if "OP_CARRIER" not in df.columns and "OP_UNIQUE_CARRIER" in df.columns:
        df = df.rename(columns={"OP_UNIQUE_CARRIER": "OP_CARRIER"})

    if "flight_date" not in df.columns:
        raise ValueError("Expected 'flight_date' in labeled dataset.")

    df["flight_date"] = pd.to_datetime(df["flight_date"], errors="coerce")
    df = df.dropna(subset=["flight_date"]).copy()

    # Derive YEAR/MONTH if not materialized
    if "YEAR" not in df.columns:
        df["YEAR"] = df["flight_date"].dt.year.astype("Int16")
    if "MONTH" not in df.columns:
        df["MONTH"] = df["flight_date"].dt.month.astype("Int8")

    # Normalize keys
    for c in ["ORIGIN", "DEST", "OP_CARRIER"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip().str.upper()

    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)

    filtered = df[
        (df["flight_date"] >= start)
        & (df["flight_date"] <= end)
        & (df["OP_CARRIER"].isin(carriers))
        & (df["ORIGIN"].isin(airports))
        & (df["DEST"].isin(airports))
    ].copy()

    if filtered.empty:
        raise ValueError("Filtered dataset is empty. Check your filters.")

    print("Filtered rows:", len(filtered))
    print("Date range:", filtered["flight_date"].min(), "to", filtered["flight_date"].max())
    print("Carriers:", sorted(filtered["OP_CARRIER"].dropna().unique().tolist()))
    print("Origins:", sorted(filtered["ORIGIN"].dropna().unique().tolist()))
    print("Dests:", sorted(filtered["DEST"].dropna().unique().tolist()))
    print("Delay rate:", float(filtered["delayed_15"].mean()))

    write_partitioned_parquet(filtered, out_dir)
    print("Wrote reduced labeled dataset to:", out_dir)


if __name__ == "__main__":
    main()