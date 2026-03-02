from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


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


def load_split(features_root: Path, split: str) -> pd.DataFrame:
    df = read_parquet_resilient(features_root / split, columns=None)
    if "delayed_15" not in df.columns:
        raise ValueError(f"Missing delayed_15 in {split}")
    return df

def sanitize_for_catboost(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """
    CatBoost doesn't like pandas NAType (pd.NA), especially in categorical columns.
    - Convert categorical cols to string and fill missing with sentinel
    - Convert numeric cols to float and replace pd.NA with np.nan (CatBoost handles np.nan)
    """
    X = X.copy()

    # Categorical: string + sentinel
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("string").fillna("__MISSING__")

    # Numeric: convert pd.NA to np.nan by forcing to numeric
    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")  # -> float w/ np.nan

    return X

def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "roc_auc": float(auc),
        "pr_auc": float(ap),
        "brier": float(brier),
        "threshold": float(threshold),
        "confusion_matrix": cm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline CatBoost model for delay risk (BTS-only v1).")
    # parser.add_argument("--features-root", type=str, default="data/features/base")
    parser.add_argument("--features-root", type=str, default="data/features/with_aggs")
    parser.add_argument("--artifacts-dir", type=str, default="models/artifacts")
    parser.add_argument("--model-name", type=str, default="catboost_delayrisk_v1.cbm")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    features_root = Path(args.features_root)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(features_root, "train")
    val_df = load_split(features_root, "val")
    test_df = load_split(features_root, "test")

    # Define label + drop non-feature columns
    label_col = "delayed_15"
    drop_cols = ["flight_date", label_col]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    # CatBoost categorical columns
    cat_cols = ["OP_CARRIER", "ORIGIN", "DEST", "route", "dep_daypart"]
    cat_cols = [c for c in cat_cols if c in feature_cols]

    X_train, y_train = train_df[feature_cols], train_df[label_col].astype(int).to_numpy()
    X_val, y_val = val_df[feature_cols], val_df[label_col].astype(int).to_numpy()
    X_test, y_test = test_df[feature_cols], test_df[label_col].astype(int).to_numpy()

    X_train = sanitize_for_catboost(X_train, cat_cols)
    X_val = sanitize_for_catboost(X_val, cat_cols)
    X_test = sanitize_for_catboost(X_test, cat_cols)

    train_pool = Pool(X_train, label=y_train, cat_features=cat_cols)
    val_pool = Pool(X_val, label=y_val, cat_features=cat_cols)
    test_pool = Pool(X_test, label=y_test, cat_features=cat_cols)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=5000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=args.random_seed,
        verbose=200,
        od_type="Iter",
        od_wait=200,
        auto_class_weights="Balanced",
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # Evaluate
    val_prob = model.predict_proba(val_pool)[:, 1]
    test_prob = model.predict_proba(test_pool)[:, 1]

    metrics = {
        "data": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_delay_rate": float(train_df[label_col].mean()),
            "val_delay_rate": float(val_df[label_col].mean()),
            "test_delay_rate": float(test_df[label_col].mean()),
        },
        "features": {
            "feature_cols": feature_cols,
            "cat_cols": cat_cols,
        },
        "metrics": {
            "val": evaluate(y_val, val_prob, threshold=0.5),
            "test": evaluate(y_test, test_prob, threshold=0.5),
        },
    }

    # Save artifacts
    model_path = artifacts_dir / args.model_name
    model.save_model(str(model_path))

    with open(artifacts_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(artifacts_dir / "baseline_feature_cols.json", "w", encoding="utf-8") as f:
        json.dump({"feature_cols": feature_cols, "cat_cols": cat_cols}, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {artifacts_dir / 'baseline_metrics.json'}")
    print("VAL  AUC:", metrics["metrics"]["val"]["roc_auc"], "PR AUC:", metrics["metrics"]["val"]["pr_auc"])
    print("TEST AUC:", metrics["metrics"]["test"]["roc_auc"], "PR AUC:", metrics["metrics"]["test"]["pr_auc"])


if __name__ == "__main__":
    main()