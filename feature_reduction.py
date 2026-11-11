import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tabpfn import TabPFNClassifier


# ============================================================
# DARWIN - REDUNDANCY-AWARE FEATURE REDUCTION
# ============================================================

DATA_PATH = Path("data/data.csv")
OUTPUT_DIR = Path("outputs/feature_reduction")

N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42

TOP_K = 150
CORRELATION_THRESHOLD = 0.85
SELECTOR_TREES = 500

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_models():
    return {
        "TabPFN": TabPFNClassifier(device="cpu"),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500, max_features="sqrt",
            class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_features="sqrt",
            class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf", C=1, probability=True,
                class_weight="balanced", random_state=RANDOM_STATE
            )
        ),
    }


def get_patient_probabilities(model, X_test):
    probs = model.predict_proba(X_test)
    return probs[:, list(model.classes_).index(1)]


def metrics_for_fold(y_true, pred, prob):
    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Sensitivity": recall_score(y_true, pred, zero_division=0),
        "Specificity": tn / (tn + fp) if tn + fp else 0.0,
        "F1 Score": f1_score(y_true, pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, prob),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def prune_correlated_features(X_train, ranked_features, threshold):
    """
    Greedy pruning in importance order.
    Correlations are calculated from this outer training fold only.
    The first/highest-ranked feature is retained from each correlated group.
    """
    candidates = ranked_features[:TOP_K]
    corr = X_train[candidates].corr(method="spearman").abs()

    kept = []
    removed = []

    for feature in candidates:
        conflict = None
        for chosen in kept:
            if corr.loc[feature, chosen] >= threshold:
                conflict = chosen
                break

        if conflict is None:
            kept.append(feature)
        else:
            removed.append({
                "Removed_Feature": feature,
                "Retained_Feature": conflict,
                "Absolute_Spearman_Rho": float(corr.loc[feature, conflict]),
            })

    return kept, removed


def main():
    print("\n" + "=" * 75)
    print(" DARWIN - REDUNDANCY-AWARE FEATURE REDUCTION")
    print("=" * 75)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)
    if not {"ID", "class"}.issubset(df.columns):
        raise ValueError("Dataset must contain ID and class columns.")

    y = df["class"].map({"H": 0, "P": 1})
    if y.isna().any():
        raise ValueError("Unexpected class labels in class column.")

    X = df.drop(columns=["ID", "class"]).apply(
        pd.to_numeric, errors="raise"
    ).astype(float)

    total_time_cols = [c for c in X.columns if c.startswith("total_time")]
    if len(total_time_cols) != 25:
        raise ValueError(f"Expected 25 total_time columns, found {len(total_time_cols)}")
    X = X.drop(columns=total_time_cols)

    if X.shape[1] != 425:
        raise ValueError(f"Expected 425 baseline features, found {X.shape[1]}")
    if X.isna().any().any():
        raise ValueError("Missing feature values found.")

    print(f"Participants: {len(X)} | Baseline features: {X.shape[1]}")
    print(f"CV: {N_SPLITS} folds x {N_REPEATS} repeats")
    print(f"Feature ranking: Extra Trees, top {TOP_K}")
    print(f"Redundancy threshold: |Spearman rho| >= {CORRELATION_THRESHOLD}")

    splits = list(RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS,
        random_state=RANDOM_STATE
    ).split(X, y))

    results = []
    selected_rows = []
    removed_rows = []
    started = time.time()
    total_fits = len(splits) * 3 * 4
    done = 0

    for split_idx, (train_idx, test_idx) in enumerate(splits):
        repeat = split_idx // N_SPLITS + 1
        fold = split_idx % N_SPLITS + 1

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Ranking is fitted only on the outer training fold.
        selector = ExtraTreesClassifier(
            n_estimators=SELECTOR_TREES,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        selector.fit(X_train, y_train)

        importance = pd.Series(
            selector.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False, kind="stable")
        ranked = importance.index.tolist()

        top_features = ranked[:TOP_K]
        pruned_features, removed = prune_correlated_features(
            X_train, ranked, CORRELATION_THRESHOLD
        )

        for rank, feature in enumerate(ranked, start=1):
            selected_rows.append({
                "Repeat": repeat, "Fold": fold,
                "Rank": rank, "Feature": feature,
                "Importance": float(importance.loc[feature]),
            })

        for item in removed:
            removed_rows.append({
                "Repeat": repeat, "Fold": fold, **item
            })

        configurations = {
            "Baseline_425": X_train.columns.tolist(),
            "Top_150": top_features,
            "Correlation_Pruned": pruned_features,
        }

        print(
            f"\nRepeat {repeat}/{N_REPEATS}, Fold {fold}/{N_SPLITS} | "
            f"Top-150: {len(top_features)} | "
            f"After pruning: {len(pruned_features)}"
        )

        for config_name, features in configurations.items():
            for model_name in create_models():
                model = create_models()[model_name]
                start = time.time()

                model.fit(X_train[features], y_train)
                pred = model.predict(X_test[features])
                prob = get_patient_probabilities(model, X_test[features])

                elapsed = time.time() - start
                fold_metrics = metrics_for_fold(y_test, pred, prob)

                results.append({
                    "Configuration": config_name,
                    "Feature Count": len(features),
                    "Model": model_name,
                    "Repeat": repeat,
                    "Fold": fold,
                    **fold_metrics,
                    "Training Time Seconds": elapsed,
                })

                done += 1
                print(
                    f"[{done}/{total_fits}] {config_name:20s} "
                    f"| {model_name:12s} | "
                    f"Acc={fold_metrics['Accuracy']:.4f} "
                    f"| {elapsed:.1f}s"
                )

    fold_df = pd.DataFrame(results)
    fold_df.to_csv(
        OUTPUT_DIR / "feature_reduction_fold_results.csv", index=False
    )

    summary = (
        fold_df.groupby(["Configuration", "Model"])
        .agg(
            Mean_Feature_Count=("Feature Count", "mean"),
            Accuracy_Mean=("Accuracy", "mean"),
            Accuracy_SD=("Accuracy", "std"),
            Balanced_Accuracy_Mean=("Balanced Accuracy", "mean"),
            Precision_Mean=("Precision", "mean"),
            Sensitivity_Mean=("Sensitivity", "mean"),
            Specificity_Mean=("Specificity", "mean"),
            F1_Mean=("F1 Score", "mean"),
            ROC_AUC_Mean=("ROC AUC", "mean"),
            Mean_Training_Time_Seconds=("Training Time Seconds", "mean"),
        )
        .reset_index()
        .sort_values(["Model", "Configuration"])
    )
    summary.to_csv(
        OUTPUT_DIR / "feature_reduction_summary.csv", index=False
    )

    pd.DataFrame(selected_rows).to_csv(
        OUTPUT_DIR / "feature_reduction_rankings_by_fold.csv", index=False
    )
    pd.DataFrame(removed_rows).to_csv(
        OUTPUT_DIR / "correlated_features_removed_by_fold.csv", index=False
    )

    metadata = {
        "experiment": "Fold-local redundancy-aware feature reduction",
        "participants": int(len(X)),
        "baseline_features": int(X.shape[1]),
        "removed_derived_features": total_time_cols,
        "cv": "RepeatedStratifiedKFold",
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "random_state": RANDOM_STATE,
        "ranking_method": "Extra Trees feature_importances_ fitted on outer training fold",
        "top_k": TOP_K,
        "correlation_method": "Spearman",
        "correlation_threshold": CORRELATION_THRESHOLD,
        "pruning_rule": "Greedy; retain higher-ranked feature from each correlated pair",
        "models": list(create_models().keys()),
        "classifier_fits": total_fits,
        "total_execution_time_seconds": time.time() - started,
        "caution": (
            "Feature selection and correlation pruning are fold-local. "
            "Repeated CV is internal validation, not external clinical validation."
        ),
    }
    with open(OUTPUT_DIR / "feature_reduction_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\n" + "=" * 75)
    print(" FEATURE REDUCTION SUMMARY")
    print("=" * 75)
    print(summary.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")
    print("EXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()
