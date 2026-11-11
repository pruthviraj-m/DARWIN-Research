import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tabpfn import TabPFNClassifier


# ============================================================
# DARWIN - LEAKAGE-SAFE FEATURE SELECTION EXPERIMENT
# ============================================================

DATA_PATH = Path("data/data.csv")
OUTPUT_DIR = Path("outputs/feature_selection")

N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42

# 425-feature baseline excludes the 25 total_time features,
# which were verified to equal air_time + paper_time exactly.
FEATURE_COUNTS = [425, 150, 100, 50]

# Extra Trees is used only to rank features within each outer
# training fold. The outer test fold is never used for selection.
SELECTOR_TREES = 500

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_models():
    """Create a fresh model instance for every fit."""
    return {
        "TabPFN": TabPFNClassifier(device="cpu"),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=1,
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    }


def get_patient_probabilities(model, X_test):
    """Return predicted probabilities for class 1 (patient)."""
    probabilities = model.predict_proba(X_test)
    classes = list(model.classes_)
    patient_index = classes.index(1)
    return probabilities[:, patient_index]


def calculate_metrics(y_true, predictions, patient_probabilities):
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced Accuracy": balanced_accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Sensitivity": recall_score(y_true, predictions, zero_division=0),
        "Specificity": specificity,
        "F1 Score": f1_score(y_true, predictions, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, patient_probabilities),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def main():
    print("\n" + "=" * 75)
    print(" DARWIN - LEAKAGE-SAFE FEATURE SELECTION")
    print("=" * 75)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}\n"
            "Run this script from the DARWIN-Research project root."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"ID", "class"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    y = df["class"].map({"H": 0, "P": 1})
    if y.isna().any():
        raise ValueError("Unexpected class labels found in 'class'.")

    X = df.drop(columns=["ID", "class"]).copy()
    X = X.apply(pd.to_numeric, errors="raise").astype(float)

    if X.isna().any().any():
        raise ValueError("Missing feature values found.")

    # Remove the 25 mathematically derived total_time features.
    total_time_columns = [
        column for column in X.columns
        if column.startswith("total_time")
    ]
    if len(total_time_columns) != 25:
        raise ValueError(
            f"Expected 25 total_time columns, found "
            f"{len(total_time_columns)}. Check the dataset headers."
        )

    X = X.drop(columns=total_time_columns)

    if X.shape[1] != 425:
        raise ValueError(
            f"Expected 425 features after removing total_time; "
            f"found {X.shape[1]}."
        )

    print(f"\nParticipants: {len(df)}")
    print(f"Features after removing total_time: {X.shape[1]}")
    print(f"Class counts:\n{df['class'].value_counts()}")
    print(f"CV: {N_SPLITS} folds x {N_REPEATS} repeats")
    print(f"Feature configurations: {FEATURE_COUNTS}")

    # Precompute the same deterministic outer splits for every model
    # and feature-count configuration.
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    splits = list(cv.split(X, y))

    fold_results = []
    selected_feature_records = []

    overall_start = time.time()
    total_runs = len(FEATURE_COUNTS) * len(create_models()) * len(splits)
    completed_runs = 0

    for split_number, (train_idx, test_idx) in enumerate(splits):
        repeat_number = split_number // N_SPLITS + 1
        fold_number = split_number % N_SPLITS + 1

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Fit selector using only this outer training fold.
        # No information from X_test or y_test is used here.
        selector = ExtraTreesClassifier(
            n_estimators=SELECTOR_TREES,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        selector.fit(X_train, y_train)

        importance = pd.Series(
            selector.feature_importances_,
            index=X_train.columns,
            name="Importance",
        ).sort_values(ascending=False, kind="stable")

        ranked_features = importance.index.tolist()

        for rank, feature_name in enumerate(ranked_features, start=1):
            selected_feature_records.append({
                "Repeat": repeat_number,
                "Fold": fold_number,
                "Rank": rank,
                "Feature": feature_name,
                "Importance": float(importance.loc[feature_name]),
            })

        for feature_count in FEATURE_COUNTS:
            if feature_count == 425:
                # Baseline uses all 425 non-derived features.
                selected_features = X.columns.tolist()
                configuration = "Baseline_425"
            else:
                selected_features = ranked_features[:feature_count]
                configuration = f"Selected_{feature_count}"

            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]

            for model_name, model_template in create_models().items():
                # Fresh estimator for each fold/configuration.
                # Recreate via create_models to avoid reusing fitted state.
                model = create_models()[model_name]

                start_time = time.time()
                model.fit(X_train_selected, y_train)

                predictions = model.predict(X_test_selected)
                patient_probabilities = get_patient_probabilities(
                    model, X_test_selected
                )
                elapsed = time.time() - start_time

                metrics = calculate_metrics(
                    y_test, predictions, patient_probabilities
                )

                fold_results.append({
                    "Configuration": configuration,
                    "Feature Count": feature_count,
                    "Model": model_name,
                    "Repeat": repeat_number,
                    "Fold": fold_number,
                    **metrics,
                    "Training Time Seconds": elapsed,
                })

                completed_runs += 1
                print(
                    f"[{completed_runs}/{total_runs}] "
                    f"{configuration:14s} | {model_name:12s} | "
                    f"Repeat {repeat_number}/{N_REPEATS}, "
                    f"Fold {fold_number}/{N_SPLITS} | "
                    f"Acc {metrics['Accuracy']:.4f} | "
                    f"BA {metrics['Balanced Accuracy']:.4f} | "
                    f"{elapsed:.1f}s"
                )

    # ========================================================
    # SAVE FOLD RESULTS AND SUMMARIES
    # ========================================================

    fold_df = pd.DataFrame(fold_results)
    fold_path = OUTPUT_DIR / "feature_selection_fold_results.csv"
    fold_df.to_csv(fold_path, index=False)

    summary = (
        fold_df.groupby(["Configuration", "Feature Count", "Model"])
        .agg(
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
        .sort_values(
            ["Model", "Feature Count"],
            ascending=[True, False],
        )
    )

    summary_path = OUTPUT_DIR / "feature_selection_summary.csv"
    summary.to_csv(summary_path, index=False)

    selected_df = pd.DataFrame(selected_feature_records)
    selected_path = OUTPUT_DIR / "feature_ranking_by_fold.csv"
    selected_df.to_csv(selected_path, index=False)

    # Count how often each feature appears in each selected top-k set.
    frequency_records = []
    for feature_count in FEATURE_COUNTS:
        if feature_count == 425:
            continue

        subset = selected_df[selected_df["Rank"] <= feature_count]
        frequencies = (
            subset.groupby("Feature")
            .size()
            .reindex(X.columns, fill_value=0)
            .sort_values(ascending=False)
        )

        for feature_name, count in frequencies.items():
            frequency_records.append({
                "Feature Count": feature_count,
                "Feature": feature_name,
                "Selected Folds": int(count),
                "Total Folds": len(splits),
                "Selection Frequency": count / len(splits),
            })

    frequency_df = pd.DataFrame(frequency_records)
    frequency_path = OUTPUT_DIR / "feature_selection_frequency.csv"
    frequency_df.to_csv(frequency_path, index=False)

    metadata = {
        "project": "DARWIN Research",
        "experiment": "Leakage-safe fold-specific Extra Trees feature selection",
        "dataset": str(DATA_PATH),
        "participants": int(len(df)),
        "original_features": int(df.drop(columns=["ID", "class"]).shape[1]),
        "removed_derived_features": total_time_columns,
        "baseline_features": int(X.shape[1]),
        "feature_counts": FEATURE_COUNTS,
        "selector": "ExtraTreesClassifier",
        "selector_n_estimators": SELECTOR_TREES,
        "outer_cv": "RepeatedStratifiedKFold",
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "random_state": RANDOM_STATE,
        "models": list(create_models().keys()),
        "device": "CPU",
        "target_mapping": {"H": 0, "P": 1},
        "selection_rule": (
            "Feature importance is fitted separately on each outer training "
            "fold. The corresponding outer test fold is never used for "
            "feature ranking or selection."
        ),
        "total_classifier_fits": int(total_runs),
        "total_execution_time_seconds": time.time() - overall_start,
    }

    metadata_path = OUTPUT_DIR / "feature_selection_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\n" + "=" * 85)
    print(" FEATURE SELECTION SUMMARY")
    print("=" * 85)
    display_columns = [
        "Configuration",
        "Feature Count",
        "Model",
        "Accuracy_Mean",
        "Accuracy_SD",
        "Balanced_Accuracy_Mean",
        "Sensitivity_Mean",
        "Specificity_Mean",
        "F1_Mean",
        "ROC_AUC_Mean",
    ]
    print(summary[display_columns].to_string(index=False))

    print("\nFiles saved in:")
    print(OUTPUT_DIR.resolve())
    print("\nEXPERIMENT COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
