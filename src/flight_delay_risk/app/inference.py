from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


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