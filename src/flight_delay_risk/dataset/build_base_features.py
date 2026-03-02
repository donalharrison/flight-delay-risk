from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


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
    df.to_parquet(out_dir, index=False, engine="pyarrow", partition_cols=["YEAR", "MONTH"], use_dictionary=False)


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize carrier name
    if "OP_CARRIER" not in df.columns and "OP_UNIQUE_CARRIER" in df.columns:
        df = df.rename(columns={"OP_UNIQUE_CARRIER": "OP_CARRIER"})

    # Required columns for v1 base features (do NOT require YEAR/MONTH)
    required = [
        "flight_date",
        "DAY_OF_WEEK",
        "ORIGIN",
        "DEST",
        "DISTANCE",
        "CRS_DEP_MINUTES",
        "CRS_ARR_MINUTES",
        "CRS_DEP_HOUR",
        "CRS_ARR_HOUR",
        "OP_CARRIER",
        "delayed_15",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for base features: {missing}")

    df = df.copy()
    df["flight_date"] = pd.to_datetime(df["flight_date"], errors="coerce")
    if df["flight_date"].isna().any():
        raise ValueError("Some flight_date values are NaT.")

    # Derive YEAR/MONTH for partitioning & downstream convenience
    if "YEAR" not in df.columns:
        df["YEAR"] = df["flight_date"].dt.year.astype("Int16")
    else:
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int16")

    if "MONTH" not in df.columns:
        df["MONTH"] = df["flight_date"].dt.month.astype("Int8")
    else:
        df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce").astype("Int8")

    # Calendar features (leakage-free)
    # BTS: DAY_OF_WEEK is 1=Mon ... 7=Sun
    df["is_weekend"] = df["DAY_OF_WEEK"].isin([6, 7]).astype("int8")
    df["week_of_year"] = df["flight_date"].dt.isocalendar().week.astype("int16")
    df["day_of_month"] = df["flight_date"].dt.day.astype("int8")

    # Route feature
    df["route"] = (df["ORIGIN"].astype("string") + "-" + df["DEST"].astype("string")).astype("string")

    # Daypart bucket (simple bins help models)
    dep_hour = pd.to_numeric(df["CRS_DEP_HOUR"], errors="coerce")
    df["dep_daypart"] = pd.cut(
        dep_hour,
        bins=[-1, 5, 11, 17, 21, 23],
        labels=["night", "morning", "afternoon", "evening", "late"],
    ).astype("string")

    # Select final columns
    feature_cols = [
        "YEAR", "MONTH",
        "DAY_OF_WEEK", "is_weekend", "week_of_year", "day_of_month",
        "OP_CARRIER", "ORIGIN", "DEST", "route", "dep_daypart",
        "DISTANCE",
        "CRS_DEP_MINUTES", "CRS_ARR_MINUTES", "CRS_DEP_HOUR", "CRS_ARR_HOUR",
    ]
    id_cols = ["flight_date"]
    label_cols = ["delayed_15"]

    out = df[id_cols + feature_cols + label_cols].copy()

    # Strong dtypes
    for c in ["OP_CARRIER", "ORIGIN", "DEST", "route", "dep_daypart"]:
        out[c] = out[c].astype("string").str.strip()

    out["DISTANCE"] = pd.to_numeric(out["DISTANCE"], errors="coerce").astype("float32")
    out["delayed_15"] = pd.to_numeric(out["delayed_15"], errors="coerce").astype("int8")
    out["YEAR"] = pd.to_numeric(out["YEAR"], errors="coerce").astype("Int16")
    out["MONTH"] = pd.to_numeric(out["MONTH"], errors="coerce").astype("Int8")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build base (no-aggregate) features for each split.")
    parser.add_argument("--splits-root", type=str, default="data/interim/splits")
    parser.add_argument("--out-root", type=str, default="data/features/base")
    parser.add_argument("--splits", type=str, default="train,val,test,holdout_2025",
                        help="Comma-separated split folder names under splits-root.")
    args = parser.parse_args()

    splits_root = Path(args.splits_root)
    out_root = Path(args.out_root)
    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in split_names:
        in_dir = splits_root / split
        if not in_dir.exists():
            print(f"Skipping missing split directory: {in_dir}")
            continue

        print(f"Reading split: {split} from {in_dir}")
        df = read_parquet_resilient(in_dir, columns=None)

        feats = build_base_features(df)
        out_dir = out_root / split
        write_partitioned_parquet(feats, out_dir)

        print(f"Wrote base features: {out_dir}")
        print(f"Rows: {len(feats)} | Delay rate: {feats['delayed_15'].mean():.4f}")

    print("Done.")


if __name__ == "__main__":
    main()