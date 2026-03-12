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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight feature store for aggregate lookups.")
    parser.add_argument("--with-aggs-root", type=str, default="data/features/with_aggs")
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--out-path", type=str, default="data/feature_store/with_aggs_store.parquet")
    args = parser.parse_args()

    root = Path(args.with_aggs_root)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]

    frames = []
    for split in split_names:
        p = root / split
        if not p.exists():
            raise FileNotFoundError(f"Missing split directory: {p}")
        df = read_parquet_resilient(p, columns=None)
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)

    if "flight_date" not in df_all.columns:
        raise ValueError("Expected 'flight_date' in input feature datasets.")

    df_all["flight_date"] = pd.to_datetime(df_all["flight_date"], errors="coerce")
    df_all = df_all.dropna(subset=["flight_date"]).copy()

    agg_cols = [c for c in df_all.columns if ("_delay_rate_" in c or "_freq_" in c)]
    key_cols = ["flight_date", "ORIGIN", "DEST", "OP_CARRIER", "route", "CRS_DEP_HOUR"]
    keep = [c for c in key_cols if c in df_all.columns] + agg_cols

    store = df_all[keep].copy()

    # Normalize key columns for safer downstream lookups
    for c in ["ORIGIN", "DEST", "OP_CARRIER", "route"]:
        if c in store.columns:
            store[c] = store[c].astype("string").fillna("__MISSING__").str.strip()

    if "CRS_DEP_HOUR" in store.columns:
        store["CRS_DEP_HOUR"] = pd.to_numeric(store["CRS_DEP_HOUR"], errors="coerce").fillna(-1).astype("int16")

    store.to_parquet(out_path, index=False, engine="pyarrow", use_dictionary=False)

    print("Wrote feature store:", out_path)
    print("Rows:", len(store))
    print("Agg cols:", len(agg_cols))
    print("Date range:", store["flight_date"].min(), "to", store["flight_date"].max())


if __name__ == "__main__":
    main()