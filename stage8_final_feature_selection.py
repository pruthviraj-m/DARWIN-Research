"""
DARWIN Stage 8: Nested-CV final feature-set selection
====================================================
Compares four feature strategies:
  A. all 425 non-derived features (25 total_time columns removed)
  B. Extra Trees top 150
  C. Extra Trees top 100
  D. Extra Trees top 150 followed by training-fold-only Spearman pruning

Feature ranking and correlation pruning are repeated inside every inner-training
split and again on the complete outer-training split. The outer test fold is
never used for feature selection or configuration choice.

Run from the project root:
    python stage8_final_feature_selection.py

Expected dataset:
    data/data.csv

Outputs:
    outputs/stage8/stage8_outer_fold_results.csv
    outputs/stage8/stage8_inner_selection_results.csv
    outputs/stage8/stage8_summary.csv
    outputs/stage8/stage8_metadata.json
"""

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 3
CORRELATION_THRESHOLD = 0.85
TOP_COUNTS = (150, 100)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "data.csv"
OUTPUT_DIR = ROOT / "outputs" / "stage8"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_model(name):
    if name == "TabPFN":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device="cpu")
    if name == "Extra Trees":
        return ExtraTreesClassifier(
            n_estimators=400, max_features="sqrt",
            class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=-1
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=400, max_features="sqrt",
            class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=-1
        )
    if name == "SVM":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1.0, probability=True,
                        class_weight="balanced", random_state=RANDOM_STATE))
        ])
    raise ValueError(name)


def rank_features(X_train, y_train):
    selector = ExtraTreesClassifier(
        n_estimators=400, max_features="sqrt",
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1
    )
    selector.fit(X_train, y_train)
    order = np.argsort(selector.feature_importances_)[::-1]
    return list(X_train.columns[order])


def select_features(X_train, y_train, strategy):
    if strategy == "All 425":
        return list(X_train.columns)

    ranked = rank_features(X_train, y_train)

    if strategy == "Top 150":
        return ranked[:150]
    if strategy == "Top 100":
        return ranked[:100]
    if strategy == "Correlation-pruned":
        candidates = ranked[:150]
        corr = X_train[candidates].corr(method="spearman").abs()
        kept = []
        for feature in candidates:
            if not kept:
                kept.append(feature)
                continue
            if all(corr.loc[feature, prior] < CORRELATION_THRESHOLD
                   for prior in kept):
                kept.append(feature)
        return kept

    raise ValueError(strategy)


def positive_probability(model, X):
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    positive_index = classes.index(1)
    return probabilities[:, positive_index]


def evaluate(model_name, X_train, y_train, X_test, y_test):
    model = make_model(model_name)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = positive_probability(model, X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def main():
    start = time.time()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place data.csv inside the project's data folder."
        )

    df = pd.read_csv(DATA_PATH)
    required = {"ID", "class"}
    if not required.issubset(df.columns):
        raise ValueError("Expected columns named 'ID' and 'class'.")

    y_raw = df["class"].astype(str).str.strip()
    mapping = {"H": 0, "P": 1, "0": 0, "1": 1}
    y = y_raw.map(mapping)
    if y.isna().any():
        raise ValueError(f"Unexpected class labels: {sorted(y_raw.unique())}")

    # Remove identifier and target. total_time_n is exactly air_time_n + paper_time_n.
    feature_cols = [
        c for c in df.columns
        if c not in ("ID", "class") and not str(c).startswith("total_time")
    ]
    X = df[feature_cols].apply(pd.to_numeric, errors="raise")
    if X.shape[1] != 425:
        raise ValueError(
            f"Expected 425 non-derived features after removing total_time; got {X.shape[1]}."
        )

    strategies = ["All 425", "Top 150", "Top 100", "Correlation-pruned"]
    models = ["TabPFN", "Extra Trees", "Random Forest", "SVM"]

    outer_cv = StratifiedKFold(
        n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )

    outer_rows = []
    inner_rows = []
    fold_feature_counts = []

    print(f"Dataset: {len(df)} participants, {X.shape[1]} features")
    print(f"Outer CV: {OUTER_SPLITS}-fold; inner CV: {INNER_SPLITS}-fold")
    print("This is a nested evaluation; runtime may be substantial.\n")

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        print(f"\n========== OUTER FOLD {outer_fold}/{OUTER_SPLITS} ==========")
        X_outer_train = X.iloc[outer_train_idx].reset_index(drop=True)
        y_outer_train = y.iloc[outer_train_idx].reset_index(drop=True)
        X_outer_test = X.iloc[outer_test_idx].reset_index(drop=True)
        y_outer_test = y.iloc[outer_test_idx].reset_index(drop=True)

        inner_cv = StratifiedKFold(
            n_splits=INNER_SPLITS, shuffle=True,
            random_state=RANDOM_STATE + outer_fold
        )

        # Choose the feature strategy separately for each model using inner CV only.
        for model_name in models:
            strategy_scores = {strategy: [] for strategy in strategies}

            for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(
                inner_cv.split(X_outer_train, y_outer_train), start=1
            ):
                X_inner_train = X_outer_train.iloc[inner_train_idx]
                y_inner_train = y_outer_train.iloc[inner_train_idx]
                X_inner_valid = X_outer_train.iloc[inner_valid_idx]
                y_inner_valid = y_outer_train.iloc[inner_valid_idx]

                for strategy in strategies:
                    selected = select_features(
                        X_inner_train, y_inner_train, strategy
                    )
                    result = evaluate(
                        model_name,
                        X_inner_train[selected], y_inner_train,
                        X_inner_valid[selected], y_inner_valid
                    )
                    strategy_scores[strategy].append(result["balanced_accuracy"])

                    inner_rows.append({
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "inner_fold": inner_fold,
                        "strategy": strategy,
                        "selected_feature_count": len(selected),
                        "balanced_accuracy": result["balanced_accuracy"]
                    })

            mean_scores = {
                strategy: float(np.mean(scores))
                for strategy, scores in strategy_scores.items()
            }
            chosen_strategy = max(mean_scores, key=mean_scores.get)

            print(
                f"{model_name}: selected {chosen_strategy} "
                f"(inner balanced accuracy {mean_scores[chosen_strategy]*100:.2f}%)"
            )

            # Re-select features on the full outer-training set, never the test set.
            selected_outer = select_features(
                X_outer_train, y_outer_train, chosen_strategy
            )
            outer_result = evaluate(
                model_name,
                X_outer_train[selected_outer], y_outer_train,
                X_outer_test[selected_outer], y_outer_test
            )

            outer_rows.append({
                "outer_fold": outer_fold,
                "model": model_name,
                "chosen_strategy": chosen_strategy,
                "selected_feature_count": len(selected_outer),
                "inner_best_balanced_accuracy": mean_scores[chosen_strategy],
                **outer_result
            })
            fold_feature_counts.append({
                "outer_fold": outer_fold,
                "model": model_name,
                "strategy": chosen_strategy,
                "feature_count": len(selected_outer)
            })

            print(
                f"  Outer result: accuracy={outer_result['accuracy']*100:.2f}%, "
                f"balanced accuracy={outer_result['balanced_accuracy']*100:.2f}%, "
                f"features={len(selected_outer)}"
            )

    outer_df = pd.DataFrame(outer_rows)
    inner_df = pd.DataFrame(inner_rows)

    summary = (
        outer_df.groupby(["model", "chosen_strategy"])
        .agg(
            folds=("outer_fold", "count"),
            mean_accuracy=("accuracy", "mean"),
            sd_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            sd_balanced_accuracy=("balanced_accuracy", "std"),
            mean_f1=("f1", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_features=("selected_feature_count", "mean")
        )
        .reset_index()
    )

    outer_path = OUTPUT_DIR / "stage8_outer_fold_results.csv"
    inner_path = OUTPUT_DIR / "stage8_inner_selection_results.csv"
    summary_path = OUTPUT_DIR / "stage8_summary.csv"
    metadata_path = OUTPUT_DIR / "stage8_metadata.json"

    outer_df.to_csv(outer_path, index=False)
    inner_df.to_csv(inner_path, index=False)
    summary.to_csv(summary_path, index=False)

    metadata = {
        "stage": 8,
        "dataset": str(DATA_PATH),
        "participants": int(len(df)),
        "features_after_removing_total_time": int(X.shape[1]),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
        "outer_cv": f"{OUTER_SPLITS}-fold stratified",
        "inner_cv": f"{INNER_SPLITS}-fold stratified",
        "strategies": strategies,
        "models": models,
        "correlation_threshold": CORRELATION_THRESHOLD,
        "selection_metric": "inner balanced accuracy",
        "outer_test_used_for_feature_selection": False,
        "outer_test_used_for_strategy_selection": False,
        "runtime_seconds": round(time.time() - start, 2),
        "limitations": [
            "Nested cross-validation is internal validation, not external validation.",
            "The small dataset limits precision of performance estimates.",
            "The best strategy is selected separately for each model within each outer fold.",
            "Fold-level results are not independent observations; do not use a naive t-test across folds.",
            "No clinical diagnostic claim is justified by these results."
        ]
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n========== STAGE 8 SUMMARY ==========")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved:")
    print(outer_path)
    print(inner_path)
    print(summary_path)
    print(metadata_path)
    print(f"\nTotal runtime: {(time.time() - start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
