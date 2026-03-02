from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd

from src.flight_delay_risk.app.inference import load_feature_store  # uses resilient parquet


def build_index_maps(df: pd.DataFrame, agg_cols: list[str]):
    df = df.copy()
    df["flight_date"] = pd.to_datetime(df["flight_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["flight_date"]).copy()

    # Defaults
    rate_cols = [c for c in agg_cols if "_delay_rate_" in c]
    global_rate = 0.25
    if rate_cols:
        global_rate = float(pd.to_numeric(df[rate_cols[0]], errors="coerce").dropna().mean())

    def make_map(key_cols):
        tmp = df[key_cols + agg_cols].drop_duplicates(subset=key_cols, keep="first").copy()
        tmp = tmp.set_index(key_cols)
        out = {}
        for k, row in tmp.iterrows():
            if not isinstance(k, tuple):
                k = (k,)
            out[k] = {c: float(row[c]) if pd.notna(row[c]) else (0.0 if "_freq_" in c else global_rate) for c in agg_cols}
        return out

    return {
        "global_rate": global_rate,
        "max_date": pd.Timestamp(df["flight_date"].max()).normalize(),
        "agg_cols": agg_cols,
        "by_origin": make_map(["flight_date", "ORIGIN"]),
        "by_dest": make_map(["flight_date", "DEST"]),
        "by_carrier": make_map(["flight_date", "OP_CARRIER"]),
        "by_route": make_map(["flight_date", "route"]),
        "by_carrier_origin": make_map(["flight_date", "OP_CARRIER", "ORIGIN"]),
        "by_origin_dep_hour": make_map(["flight_date", "ORIGIN", "CRS_DEP_HOUR"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=str, default="data/feature_store/with_aggs_store_slim.parquet")
    parser.add_argument("--out", type=str, default="data/feature_store/with_aggs_index.joblib")
    args = parser.parse_args()

    store_path = Path(args.store)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fs = load_feature_store(store_path)
    df = fs.df

    agg_cols = [c for c in df.columns if ("_delay_rate_" in c or "_freq_" in c)]
    idx = build_index_maps(df, agg_cols)

    joblib.dump(idx, out_path, compress=3)
    print("Saved index:", out_path)


if __name__ == "__main__":
    main()