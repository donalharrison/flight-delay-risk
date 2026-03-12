from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
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


def get_dataset_columns(parquet_dir: Path) -> list[str]:
    import pyarrow.dataset as ds

    dataset = ds.dataset(parquet_dir, format="parquet")
    return list(dataset.schema.names)


def load_split(features_root: Path, split: str, columns: list[str] | None = None) -> pd.DataFrame:
    split_path = features_root / split
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split directory: {split_path}")

    df = read_parquet_resilient(split_path, columns=columns)
    if "delayed_15" not in df.columns:
        raise ValueError(f"Missing delayed_15 in {split}")
    return df


def maybe_sample_split(
    df: pd.DataFrame,
    max_rows: int | None,
    *,
    random_seed: int,
    label_col: str = "delayed_15",
) -> pd.DataFrame:
    """
    Sample down large splits to fit local memory.
    Tries to preserve class balance with per-class sampling.
    """
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)

    if label_col not in df.columns:
        return df.sample(n=max_rows, random_state=random_seed).reset_index(drop=True)

    frac = max_rows / len(df)

    parts = []
    for cls, g in df.groupby(label_col, dropna=False):
        n = max(1, int(round(len(g) * frac)))
        n = min(n, len(g))
        parts.append(g.sample(n=n, random_state=random_seed))

    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=random_seed)

    return sampled.reset_index(drop=True)


def sanitize_for_catboost(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """
    Return a compact, clean DataFrame for CatBoost.
    We explicitly make a single copy here after sampling, not before.
    """
    X = X.copy()

    for c in cat_cols:
        if c in X.columns:
            X.loc[:, c] = X[c].astype("string").fillna("__MISSING__")

    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in num_cols:
        s = pd.to_numeric(X[c], errors="coerce")
        if pd.api.types.is_float_dtype(s):
            X.loc[:, c] = s.astype("float32")
        else:
            X.loc[:, c] = s

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


def summarize_split(df: pd.DataFrame, label_col: str = "delayed_15") -> dict:
    out = {
        "rows": int(len(df)),
        "delay_rate": float(df[label_col].mean()),
    }
    if "flight_date" in df.columns:
        dates = pd.to_datetime(df["flight_date"], errors="coerce")
        out["date_min"] = str(dates.min())
        out["date_max"] = str(dates.max())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CatBoost delay-risk model (memory-aware).")
    parser.add_argument("--features-root", type=str, default="data/features/with_aggs")
    parser.add_argument("--artifacts-dir", type=str, default="models/artifacts")
    parser.add_argument("--model-name", type=str, default="catboost_delayrisk_v1.cbm")
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)

    # New: memory-safe sample caps
    parser.add_argument("--train-max-rows", type=int, default=2_000_000)
    parser.add_argument("--val-max-rows", type=int, default=500_000)
    parser.add_argument("--test-max-rows", type=int, default=500_000)

    args = parser.parse_args()

    features_root = Path(args.features_root)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    label_col = "delayed_15"
    drop_cols = ["flight_date", label_col]

    train_schema_cols = get_dataset_columns(features_root / "train")
    feature_cols = [c for c in train_schema_cols if c not in drop_cols]

    needed_cols = feature_cols + [label_col]
    if "flight_date" in train_schema_cols:
        needed_cols.append("flight_date")

    # Load full splits
    train_df = load_split(features_root, "train", columns=needed_cols)
    val_df = load_split(features_root, "val", columns=needed_cols)
    test_df = load_split(features_root, "test", columns=needed_cols)

    print("Loaded full splits:")
    print("Train summary:", summarize_split(train_df, label_col))
    print("Val summary:  ", summarize_split(val_df, label_col))
    print("Test summary: ", summarize_split(test_df, label_col))

    # Sample down for local training/evaluation
    train_df = maybe_sample_split(train_df, args.train_max_rows, random_seed=args.random_seed, label_col=label_col)
    val_df = maybe_sample_split(val_df, args.val_max_rows, random_seed=args.random_seed, label_col=label_col)
    test_df = maybe_sample_split(test_df, args.test_max_rows, random_seed=args.random_seed, label_col=label_col)

    print("\nUsing sampled splits for training/evaluation:")
    print("Train summary:", summarize_split(train_df, label_col))
    print("Val summary:  ", summarize_split(val_df, label_col))
    print("Test summary: ", summarize_split(test_df, label_col))

    cat_cols = ["OP_CARRIER", "ORIGIN", "DEST", "route", "dep_daypart"]
    cat_cols = [c for c in cat_cols if c in feature_cols]

    y_train = train_df[label_col].astype(int).to_numpy()
    y_val = val_df[label_col].astype(int).to_numpy()
    y_test = test_df[label_col].astype(int).to_numpy()

    X_train = sanitize_for_catboost(train_df[feature_cols], cat_cols)
    X_val = sanitize_for_catboost(val_df[feature_cols], cat_cols)
    X_test = sanitize_for_catboost(test_df[feature_cols], cat_cols)

    train_pool = Pool(X_train, label=y_train, cat_features=cat_cols)
    val_pool = Pool(X_val, label=y_val, cat_features=cat_cols)
    test_pool = Pool(X_test, label=y_test, cat_features=cat_cols)

    del train_df, val_df, test_df, X_train, X_val, X_test
    gc.collect()

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.random_seed,
        verbose=200,
        od_type="Iter",
        od_wait=200,
        auto_class_weights="Balanced",
    )

    print("\nTraining model...")
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_prob = model.predict_proba(val_pool)[:, 1]
    test_prob = model.predict_proba(test_pool)[:, 1]

    metrics = {
        "data_full": {
            "train_rows_loaded": int(len(y_train)) if False else None,  # placeholder for compatibility
        },
        "data_sampled": {
            "train_rows": int(len(y_train)),
            "val_rows": int(len(y_val)),
            "test_rows": int(len(y_test)),
            "train_delay_rate": float(y_train.mean()),
            "val_delay_rate": float(y_val.mean()),
            "test_delay_rate": float(y_test.mean()),
        },
        "features": {
            "feature_cols": feature_cols,
            "cat_cols": cat_cols,
        },
        "training": {
            "model_name": args.model_name,
            "features_root": str(features_root),
            "random_seed": args.random_seed,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "depth": args.depth,
            "l2_leaf_reg": args.l2_leaf_reg,
            "train_max_rows": args.train_max_rows,
            "val_max_rows": args.val_max_rows,
            "test_max_rows": args.test_max_rows,
        },
        "metrics": {
            "val": evaluate(y_val, val_prob, threshold=0.5),
            "test": evaluate(y_test, test_prob, threshold=0.5),
        },
    }

    model_path = artifacts_dir / args.model_name
    model.save_model(str(model_path))

    with open(artifacts_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(artifacts_dir / "baseline_feature_cols.json", "w", encoding="utf-8") as f:
        json.dump({"feature_cols": feature_cols, "cat_cols": cat_cols}, f, indent=2)

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {artifacts_dir / 'baseline_metrics.json'}")
    print("VAL  AUC:", metrics["metrics"]["val"]["roc_auc"], "PR AUC:", metrics["metrics"]["val"]["pr_auc"])
    print("TEST AUC:", metrics["metrics"]["test"]["roc_auc"], "PR AUC:", metrics["metrics"]["test"]["pr_auc"])


if __name__ == "__main__":
    main()