import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXCLUDE_FEATURES = ["loan_percent_income"]


def evaluate(y_true, pred, prob):
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
    }


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Train credit risk models and compare performance.")
    parser.add_argument("--input", default="data/processed/loan_clean.csv")
    parser.add_argument("--target", default="default_flag")
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Clean data not found: {args.input}")

    df = pd.read_csv(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Target column not found: {args.target}")

    y = df[args.target].astype(int)
    X = df.drop(columns=[args.target])
    drop_cols = [c for c in EXCLUDE_FEATURES if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)
        print(f"Dropped features: {drop_cols}")

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    models = {
        "logistic": LogisticRegression(
            max_iter=2500,
            class_weight=None,
            C=2.0,
            random_state=42,
            n_jobs=None,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=14,
            min_samples_leaf=3,
            class_weight=None,
            random_state=42,
            n_jobs=1,
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    metrics = {}
    best_name = None
    best_auc = -1

    for name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]

        m = evaluate(y_test, pred, prob)
        m["n_train"] = int(len(X_train))
        m["n_test"] = int(len(X_test))
        m["target_rate_train"] = float(y_train.mean())

        metrics[name] = m

        model_path = os.path.join(args.out_dir, f"{name}_pipeline.joblib")
        joblib.dump(pipe, model_path)

        if m["roc_auc"] > best_auc:
            best_auc = m["roc_auc"]
            best_name = name

    # Save best model alias for app use
    best_model_src = os.path.join(args.out_dir, f"{best_name}_pipeline.joblib")
    best_model_dst = os.path.join(args.out_dir, "best_model.joblib")
    joblib.dump(joblib.load(best_model_src), best_model_dst)

    metrics_summary = {
        "best_model": best_name,
        "threshold_default": 0.50,
        "dropped_columns_for_leakage_control": drop_cols,
        "models": metrics,
    }

    metrics_json = os.path.join(args.out_dir, "model_metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "model_name"})
    metrics_df.to_csv(os.path.join(args.out_dir, "model_comparison.csv"), index=False)

    print("Training done.")
    print(f"Best model: {best_name} (ROC-AUC={best_auc:.4f})")
    print(f"Metrics   : {metrics_json}")
    print(metrics_df)


if __name__ == "__main__":
    main()
