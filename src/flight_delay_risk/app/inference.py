from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from catboost import Pool


@dataclass(frozen=True)
class ModelArtifacts:
    model: CatBoostClassifier
    feature_cols: list[str]
    cat_cols: list[str]


def load_artifacts(artifacts_dir: Path) -> ModelArtifacts:
    artifacts_dir = Path(artifacts_dir)

    # Load feature column lists
    with open(artifacts_dir / "baseline_feature_cols.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    cat_cols = meta.get("cat_cols", [])

    # Load model
    model = CatBoostClassifier()
    model.load_model(str(artifacts_dir / "catboost_delayrisk_v1.cbm"))

    return ModelArtifacts(model=model, feature_cols=feature_cols, cat_cols=cat_cols)


def _risk_bucket(p: float) -> str:
    # Simple buckets for v1; we can calibrate later
    if p < 0.20:
        return "Low"
    if p < 0.40:
        return "Medium"
    return "High"


def _simple_reason_codes(row: pd.Series) -> list[str]:
    """
    Deterministic explanation scaffold.
    This is NOT SHAP. It's just a helpful v1 narrative.
    We'll upgrade to SHAP + LLM later.
    """
    reasons: list[str] = []

    # Heuristic examples
    dep_hour = row.get("CRS_DEP_HOUR")
    dist = row.get("DISTANCE")
    dow = row.get("DAY_OF_WEEK")
    weekend = row.get("is_weekend")

    try:
        if pd.notna(dep_hour) and int(dep_hour) >= 18:
            reasons.append("Evening departures tend to accumulate delays from earlier flights.")
        if pd.notna(dep_hour) and int(dep_hour) <= 7:
            reasons.append("Early flights can still be impacted by morning congestion at busy airports.")
    except Exception:
        pass

    try:
        if pd.notna(weekend) and int(weekend) == 1:
            reasons.append("Weekend travel often has different congestion patterns.")
    except Exception:
        pass

    try:
        if pd.notna(dist) and float(dist) >= 1500:
            reasons.append("Longer routes have more opportunities for upstream disruption.")
    except Exception:
        pass

    # Route / airports / carrier (placeholders for future aggregate insights)
    if "route" in row and pd.notna(row["route"]):
        reasons.append("Route-specific congestion patterns can affect delay risk.")

    return reasons[:4]


def sanitize_for_catboost(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    X = X.copy()

    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("string").fillna("__MISSING__")

    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    return X


def predict_one(
    artifacts: ModelArtifacts,
    features: dict[str, Any],
) -> dict[str, Any]:
    """
    features: dict keyed by model feature columns (or a superset).
    We'll build a single-row DataFrame with exactly feature_cols.
    """

    # Create a one-row frame with all features (missing -> NA)
    row = {c: features.get(c, pd.NA) for c in artifacts.feature_cols}
    X = pd.DataFrame([row])

    X = sanitize_for_catboost(X, artifacts.cat_cols)

    prob = float(artifacts.model.predict_proba(X)[:, 1][0])
    bucket = _risk_bucket(prob)

    # build a row series for simple reason codes
    reason_row = X.iloc[0].copy()
    reasons = _simple_reason_codes(reason_row)

    return {
        "probability": prob,
        "risk_bucket": bucket,
        "reasons": reasons,
        "features_used": row,  # handy for debugging; remove later if you want
    }

def load_background_sample(
    features_dir: Path,
    feature_cols: list[str],
    cat_cols: list[str],
    n_rows: int = 5000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Load a small background dataset for SHAP base value stability.
    This reads from your prepared features dataset (recommended: base/train or with_aggs/train).

    We keep it simple: sample up to n_rows uniformly.
    """
    df = read_parquet_resilient(features_dir, columns=None)

    # Keep only model features
    df = df[feature_cols].copy()

    # Sanitize
    df = sanitize_for_catboost(df, cat_cols)

    # Sample
    if len(df) > n_rows:
        df = df.sample(n=n_rows, random_state=random_seed)

    return df


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

@dataclass(frozen=True)
class FeatureStore:
    df: pd.DataFrame
    max_date: pd.Timestamp
    global_rate: float
    agg_cols: list[str]


def load_feature_store(store_path: Path) -> FeatureStore:
    df = read_parquet_resilient(store_path, columns=None)
    df["flight_date"] = pd.to_datetime(df["flight_date"], errors="coerce")
    df = df.dropna(subset=["flight_date"]).copy()

    agg_cols = [c for c in df.columns if ("_delay_rate_" in c or "_freq_" in c)]
    # global_rate used for filling missing rate columns
    # If you don’t store delayed_15 here, approximate with mean of one rate column if present, else 0.25
    global_rate = 0.25
    # Better: if you stored any “carrier_delay_rate_365d” etc, use overall mean
    rate_cols = [c for c in agg_cols if "_delay_rate_" in c]
    if rate_cols:
        global_rate = float(pd.to_numeric(df[rate_cols[0]], errors="coerce").dropna().mean())

    return FeatureStore(
        df=df,
        max_date=pd.Timestamp(df["flight_date"].max()),
        global_rate=global_rate,
        agg_cols=agg_cols,
    )


def apply_aggregate_features(
    fs: FeatureStore,
    features: dict,
) -> dict:
    """
    Add aggregate feature columns to `features` dict by looking up
    matching (flight_date, ORIGIN, OP_CARRIER, route, CRS_DEP_HOUR, etc.)

    If requested flight_date is beyond available history, fallback to fs.max_date.
    """
    # Determine lookup date
    req_date = pd.to_datetime(features["flight_date"])
    lookup_date = req_date
    if lookup_date > fs.max_date:
        lookup_date = fs.max_date

    # Build a 1-row frame for merge
    row = {
        "flight_date": lookup_date.normalize(),
        "ORIGIN": str(features.get("ORIGIN", "__MISSING__")),
        "DEST": str(features.get("DEST", "__MISSING__")),
        "OP_CARRIER": str(features.get("OP_CARRIER", "__MISSING__")),
        "route": str(features.get("route", "__MISSING__")),
        "CRS_DEP_HOUR": int(features.get("CRS_DEP_HOUR", -1)),
    }
    probe = pd.DataFrame([row])

    # Merge onto store; left join gives agg values if present
    merged = probe.merge(
        fs.df,
        on=[c for c in probe.columns if c in fs.df.columns],
        how="left",
    )

    # Fill missing agg values
    for c in fs.agg_cols:
        if c not in merged.columns:
            continue
        if "_freq_" in c:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)
        elif "_delay_rate_" in c:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(fs.global_rate)

        features[c] = float(merged.loc[0, c])

    return features

def shap_explain_one(
    artifacts: ModelArtifacts,
    X_row: pd.DataFrame,
    background: pd.DataFrame | None = None,
    top_k: int = 8,
) -> dict[str, object]:
    """
    Returns per-feature SHAP contributions for a single row using CatBoost native SHAP.

    Output:
      {
        "base_value": float,
        "probability": float,
        "top_positive": [(feature, value, shap), ...],
        "top_negative": [(feature, value, shap), ...],
        "all": [(feature, value, shap), ...]  # optional
      }
    """
    if len(X_row) != 1:
        raise ValueError("X_row must be a single-row DataFrame")

    # Ensure columns are in the correct order and sanitized
    X_row = X_row[artifacts.feature_cols].copy()
    X_row = sanitize_for_catboost(X_row, artifacts.cat_cols)

    # Build Pool for CatBoost
    row_pool = Pool(X_row, cat_features=artifacts.cat_cols)

    # CatBoost returns SHAP values in "raw margin" space (log-odds for binary)
    # shape: (n_rows, n_features + 1), where last column is expected_value (base)
    shap_vals = artifacts.model.get_feature_importance(row_pool, type="ShapValues")
    shap_vals = np.asarray(shap_vals)
    shap_row = shap_vals[0, :-1]
    base_value = float(shap_vals[0, -1])

    # Convert row prediction to probability (should match predict_one)
    prob = float(artifacts.model.predict_proba(row_pool)[:, 1][0])

    rows = []
    for feat, shap_v in zip(artifacts.feature_cols, shap_row):
        val = X_row.iloc[0][feat]
        # Convert numpy scalar
        shap_f = float(shap_v)
        # Keep value JSON-friendly
        if isinstance(val, (np.generic,)):
            val = val.item()
        rows.append((feat, val, shap_f))

    # Sort by contribution magnitude
    rows_sorted = sorted(rows, key=lambda t: abs(t[2]), reverse=True)

    # Separate pos/neg for narrative
    top_pos = [(f, v, s) for (f, v, s) in rows_sorted if s > 0][:top_k]
    top_neg = [(f, v, s) for (f, v, s) in rows_sorted if s < 0][:top_k]

    return {
        "base_value": base_value,
        "probability": prob,
        "top_positive": top_pos,
        "top_negative": top_neg,
        "all": rows_sorted[: max(top_k * 2, 12)],
    }