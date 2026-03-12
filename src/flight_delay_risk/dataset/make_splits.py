from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    train_end: str  # inclusive
    val_end: str    # inclusive
    test_end: str   # inclusive


def load_parquet_dataset_resilient(parquet_dir: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Read a partitioned parquet directory in a way that's resilient to pyarrow dictionary encoding
    issues across environments.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    dataset = ds.dataset(parquet_dir, format="parquet")
    table = dataset.to_table(columns=columns)

    for i, field in enumerate(table.schema):
        if pa.types.is_dictionary(field.type):
            table = table.set_column(i, field.name, pc.dictionary_decode(table.column(i)))

    return table.to_pandas()


def load_labeled(parquet_dir: Path) -> pd.DataFrame:
    df = load_parquet_dataset_resilient(parquet_dir, columns=None)

    if "flight_date" not in df.columns:
        raise ValueError("Expected 'flight_date' in labeled dataset. Run build_labels.py first.")

    df["flight_date"] = pd.to_datetime(df["flight_date"], errors="coerce")
    if df["flight_date"].isna().any():
        raise ValueError("Some flight_date values are NaT; cannot split.")

    # Derive YEAR/MONTH if partition columns are not materialized
    if "YEAR" not in df.columns:
        df["YEAR"] = df["flight_date"].dt.year.astype("Int16")
    else:
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int16")

    if "MONTH" not in df.columns:
        df["MONTH"] = df["flight_date"].dt.month.astype("Int8")
    else:
        df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce").astype("Int8")

    return df


def split_by_date(df: pd.DataFrame, cfg: SplitConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp(cfg.train_end)
    val_end = pd.Timestamp(cfg.val_end)
    test_end = pd.Timestamp(cfg.test_end)

    if not (train_end < val_end < test_end):
        raise ValueError("Require train_end < val_end < test_end")

    df = df.sort_values("flight_date").reset_index(drop=True)

    train = df[df["flight_date"] <= train_end].copy()
    val = df[(df["flight_date"] > train_end) & (df["flight_date"] <= val_end)].copy()
    test = df[(df["flight_date"] > val_end) & (df["flight_date"] <= test_end)].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"One of the splits is empty. Sizes: train={len(train)}, val={len(val)}, test={len(test)}. "
            f"Check your date boundaries."
        )

    return train, val, test


def write_split(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        out_dir,
        index=False,
        engine="pyarrow",
        partition_cols=["YEAR", "MONTH"],
        use_dictionary=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create time-based train/val/test splits for BTS labeled dataset.")
    parser.add_argument("--in-dir", type=str, default="data/interim/bts_on_time_labeled_reduced")
    parser.add_argument("--out-root", type=str, default="data/interim/splits_reduced")

    parser.add_argument("--train-end", type=str, default="2024-12-31")
    parser.add_argument("--val-end", type=str, default="2025-06-30")
    parser.add_argument("--test-end", type=str, default="2025-11-30")

    args = parser.parse_args()

    df = load_labeled(Path(args.in_dir))
    train, val, test = split_by_date(df, SplitConfig(args.train_end, args.val_end, args.test_end))

    out_root = Path(args.out_root)
    write_split(train, out_root / "train")
    write_split(val, out_root / "val")
    write_split(test, out_root / "test")

    print("Wrote splits:")
    print("  train:", len(train), "rows,", train["flight_date"].min(), "to", train["flight_date"].max())
    print("  val:  ", len(val), "rows,", val["flight_date"].min(), "to", val["flight_date"].max())
    print("  test: ", len(test), "rows,", test["flight_date"].min(), "to", test["flight_date"].max())
    print("Output root:", out_root)


if __name__ == "__main__":
    main()