from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ----------------------------
# Parquet IO (resilient)
# ----------------------------

def read_parquet_resilient(parquet_dir: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Read a parquet dataset directory resiliently across pandas/pyarrow env quirks.
    Decodes dictionary-encoded columns before converting to pandas.
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


def write_partitioned_parquet(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        out_dir,
        index=False,
        engine="pyarrow",
        partition_cols=["YEAR", "MONTH"],
        use_dictionary=False,
    )


# ----------------------------
# Rolling aggregate machinery
# ----------------------------

@dataclass(frozen=True)
class AggSpec:
    name: str
    keys: list[str]


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _make_daily(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """
    Collapse to daily counts/sums for (keys + flight_date).
    """
    gcols = keys + ["flight_date"]
    daily = (
        df.groupby(gcols, dropna=False)["delayed_15"]
        .agg(n="size", s="sum")
        .reset_index()
        .sort_values("flight_date")
    )
    # Make sure types are friendly
    daily["n"] = pd.to_numeric(daily["n"], errors="coerce").fillna(0).astype("int32")
    daily["s"] = pd.to_numeric(daily["s"], errors="coerce").fillna(0).astype("int32")
    return daily


def _rolling_sums_by_group(
    daily: pd.DataFrame,
    keys: list[str],
    window_days: int,
) -> pd.DataFrame:
    """
    For each group (keys), compute rolling sums over the last N days, excluding current day:
      n_w = sum(n) over (t - window_days, t) with closed='left'
      s_w = sum(s) over same window
    Returns daily with added columns: n_{window}d, s_{window}d
    """
    wcol_n = f"n_{window_days}d"
    wcol_s = f"s_{window_days}d"

    def _apply(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("flight_date")
        g = g.set_index("flight_date")
        # Time-based rolling window; closed='left' prevents same-day leakage
        g[wcol_n] = g["n"].rolling(f"{window_days}D", closed="left").sum()
        g[wcol_s] = g["s"].rolling(f"{window_days}D", closed="left").sum()
        g = g.reset_index()
        return g

    rolled = daily.groupby(keys, group_keys=False, sort=False).apply(_apply).reset_index(drop=True)
    missing_keys = [k for k in keys if k not in rolled.columns]
    if missing_keys:
        raise ValueError(f"Rolling output missing group keys {missing_keys}. Columns: {rolled.columns.tolist()}")
    
    # Replace NaNs (no history) with 0 for sums
    rolled[wcol_n] = rolled[wcol_n].fillna(0).astype("float32")
    rolled[wcol_s] = rolled[wcol_s].fillna(0).astype("float32")
    return rolled


def _smoothed_rate(s: pd.Series, n: pd.Series, global_rate: float, alpha: float) -> pd.Series:
    """
    Empirical Bayes / Laplace smoothing toward global_rate.
    """
    return (s + alpha * global_rate) / (n + alpha)


def build_rolling_features_for_spec(
    df_all: pd.DataFrame,
    spec: AggSpec,
    windows: list[int],
    global_rate: float,
    alpha: float,
) -> pd.DataFrame:
    """
    Builds a DAILY table with rolling windows for the given spec,
    then returns a per-flight mergeable daily-asof feature frame:
      keys + flight_date + rate/freq columns
    """
    keys = spec.keys

    daily = _make_daily(df_all, keys)
    out = daily[keys + ["flight_date"]].copy().reset_index(drop=True)

    tmp = daily
    for w in windows:
        tmp = _rolling_sums_by_group(tmp, keys=keys, window_days=w).reset_index(drop=True)

        n_col = f"n_{w}d"
        s_col = f"s_{w}d"

        rate_col = f"{spec.name}_delay_rate_{w}d"
        freq_col = f"{spec.name}_freq_{w}d"

        # Now safe: indices align 0..n-1 with no duplicates
        out[freq_col] = tmp[n_col].astype("float32").to_numpy()
        out[rate_col] = _smoothed_rate(tmp[s_col], tmp[n_col], global_rate=global_rate, alpha=alpha).astype("float32").to_numpy()

    return out


def merge_rolling_features(
    df_split: pd.DataFrame,
    daily_feats: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    """
    Merge daily rolling features onto per-flight rows using (keys + flight_date).
    """
    merge_cols = keys + ["flight_date"]
    return df_split.merge(daily_feats, on=merge_cols, how="left")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe rolling historical aggregates (BTS-only v1) and merge into splits."
    )
    parser.add_argument("--base-root", type=str, default="data/features/base", help="Root of base feature splits.")
    parser.add_argument("--out-root", type=str, default="data/features/with_aggs", help="Output root for features with aggregates.")
    parser.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated split names to process.")
    parser.add_argument("--windows", type=str, default="30,90,365", help="Comma-separated rolling windows in days.")
    parser.add_argument("--alpha", type=float, default=50.0, help="Smoothing strength (higher = more shrinkage to global mean).")
    args = parser.parse_args()

    base_root = Path(args.base_root)
    out_root = Path(args.out_root)
    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]

    # Load base splits and concatenate for building history in chronological order
    base_frames = []
    for split in split_names:
        in_dir = base_root / split
        if not in_dir.exists():
            raise FileNotFoundError(f"Missing base split directory: {in_dir}")
        
        df = read_parquet_resilient(in_dir, columns=None)

        # Normalize column names defensively (handles accidental whitespace/casing)
        df.columns = [c.strip() for c in df.columns]

        # Optional: if someone saved origin/dest under different names, normalize here
        alias_map = {
            "Origin": "ORIGIN",
            "Dest": "DEST",
        }
        df = df.rename(columns=alias_map)

        df["split_name"] = split
        base_frames.append(df)

    df_all = pd.concat(base_frames, ignore_index=True)
    print("Loaded base features columns (sample):", sorted(df_all.columns)[:40])

    # Required columns in base features
    _ensure_columns(
        df_all,
        [
            "flight_date",
            "DAY_OF_WEEK",
            "OP_CARRIER",
            "ORIGIN",
            "DEST",
            "route",
            "CRS_DEP_HOUR",
            "delayed_15",
        ],
    )

    df_all["flight_date"] = pd.to_datetime(df_all["flight_date"], errors="coerce")
    # YEAR/MONTH are often partition keys and may not be materialized as columns;
    # derive them from flight_date to make output partitioning stable.
    if "YEAR" not in df_all.columns:
        df_all["YEAR"] = df_all["flight_date"].dt.year.astype("Int16")
    else:
        df_all["YEAR"] = pd.to_numeric(df_all["YEAR"], errors="coerce").astype("Int16")

    if "MONTH" not in df_all.columns:
        df_all["MONTH"] = df_all["flight_date"].dt.month.astype("Int8")
    else:
        df_all["MONTH"] = pd.to_numeric(df_all["MONTH"], errors="coerce").astype("Int8")

    # Standardize / type sanitize
    df_all["flight_date"] = pd.to_datetime(df_all["flight_date"], errors="coerce")
    if df_all["flight_date"].isna().any():
        raise ValueError("Some flight_date values are NaT in base features; cannot compute rolling aggregates.")

    # Ensure categorical key fields are strings (avoid weird merges)
    for c in ["OP_CARRIER", "ORIGIN", "DEST", "route"]:
        df_all[c] = df_all[c].astype("string").fillna("__MISSING__").str.strip()

    df_all["CRS_DEP_HOUR"] = pd.to_numeric(df_all["CRS_DEP_HOUR"], errors="coerce").fillna(-1).astype("int16")
    df_all["delayed_15"] = pd.to_numeric(df_all["delayed_15"], errors="coerce").astype("int8")

    # Global delay rate (used for smoothing)
    global_rate = float(df_all["delayed_15"].mean())
    print(f"Global delay rate (for smoothing): {global_rate:.4f}")
    print(f"Windows: {windows} | alpha: {args.alpha}")

    # Define aggregate specs (BTS-only v1)
    specs: list[AggSpec] = [
        AggSpec(name="origin", keys=["ORIGIN"]),
        AggSpec(name="dest", keys=["DEST"]),
        AggSpec(name="carrier", keys=["OP_CARRIER"]),
        AggSpec(name="route", keys=["route"]),
        AggSpec(name="carrier_origin", keys=["OP_CARRIER", "ORIGIN"]),
        AggSpec(name="origin_dep_hour", keys=["ORIGIN", "CRS_DEP_HOUR"]),
    ]

    # Build daily rolling features for each spec (computed from df_all)
    daily_feature_tables: dict[str, pd.DataFrame] = {}
    for spec in specs:
        print(f"Building rolling features for: {spec.name} (keys={spec.keys})")
        daily_feats = build_rolling_features_for_spec(
            df_all=df_all[[*spec.keys, "flight_date", "delayed_15"]].copy(),
            spec=spec,
            windows=windows,
            global_rate=global_rate,
            alpha=args.alpha,
        )
        daily_feature_tables[spec.name] = daily_feats

    # Now merge features back into each split and write
    for split in split_names:
        print(f"\nMerging rolling features into split: {split}")
        df_split = df_all[df_all["split_name"] == split].drop(columns=["split_name"]).copy()

        # Merge each spec’s daily features onto this split
        for spec in specs:
            df_split = merge_rolling_features(df_split, daily_feature_tables[spec.name], keys=spec.keys)

        # Optional: fill any remaining NaNs (should be rare) in new rate/freq columns
        new_cols = [c for c in df_split.columns if ("_delay_rate_" in c or "_freq_" in c)]
        for c in new_cols:
            if c.endswith("_freq_" + str(windows[0]) + "d") or "_freq_" in c:
                df_split[c] = pd.to_numeric(df_split[c], errors="coerce").fillna(0).astype("float32")
            if "_delay_rate_" in c:
                df_split[c] = pd.to_numeric(df_split[c], errors="coerce").fillna(global_rate).astype("float32")

        out_dir = out_root / split
        write_partitioned_parquet(df_split, out_dir)

        print(f"Wrote: {out_dir}")
        print(f"Rows: {len(df_split)}")

    print("\nDone.")


if __name__ == "__main__":
    main()