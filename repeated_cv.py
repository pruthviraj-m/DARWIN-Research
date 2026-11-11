
import os
import time
import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from tabpfn import TabPFNClassifier


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/data.csv")
OUTPUT_DIR = Path("outputs/repeated_cv")

N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 65)
print(" DARWIN - REPEATED CROSS-VALIDATION")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

if "class" not in df.columns:
    raise ValueError("Target column 'class' not found.")

if "ID" not in df.columns:
    raise ValueError("Participant ID column 'ID' not found.")

X = df.drop(columns=["ID", "class"])
y = df["class"].map({"H": 0, "P": 1})

if y.isna().any():
    raise ValueError("Unexpected class labels found.")

if X.isna().any().any():
    raise ValueError("Missing feature values found.")

X = X.astype(float)

print(f"\nParticipants: {len(df)}")
print(f"Features: {X.shape[1]}")
print(f"Class distribution:\n{df['class'].value_counts()}")
print(f"Repeats: {N_REPEATS}")
print(f"Folds per repeat: {N_SPLITS}")


# ============================================================
# MODELS
# ============================================================

def create_models():
    return {
        "TabPFN": TabPFNClassifier(device="cpu"),

        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "SVM": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=1,
                probability=True,
                class_weight="balanced",
                random_state=42
            )
        )
    }


# ============================================================
# CROSS-VALIDATION
# ============================================================

cv = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE
)

fold_results = []
repeat_results = []

overall_start = time.time()

for model_name in create_models().keys():

    print("\n" + "=" * 65)
    print(f"MODEL: {model_name}")
    print("=" * 65)

    for repeat in range(N_REPEATS):

        repeat_start = time.time()
        repeat_accuracies = []

        print(f"\n--- Repeat {repeat + 1}/{N_REPEATS} ---")

        # Generate the folds for this repeat.
        # Recreate the same deterministic split sequence
        # for each model so comparisons remain paired.
        all_splits = list(
            RepeatedStratifiedKFold(
                n_splits=N_SPLITS,
                n_repeats=N_REPEATS,
                random_state=RANDOM_STATE
            ).split(X, y)
        )

        for fold in range(N_SPLITS):

            split_index = repeat * N_SPLITS + fold
            train_idx, test_idx = all_splits[split_index]

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model = create_models()[model_name]

            fold_start = time.time()

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            probabilities = model.predict_proba(X_test)

            # Identify the probability column for class 1 (Patient).
            classes = model.classes_
            positive_index = list(classes).index(1)
            patient_probabilities = probabilities[:, positive_index]

            accuracy = accuracy_score(y_test, predictions)

            fold_time = time.time() - fold_start

            repeat_accuracies.append(accuracy)

            fold_results.append({
                "Model": model_name,
                "Repeat": repeat + 1,
                "Fold": fold + 1,
                "Accuracy": accuracy,
                "Balanced Accuracy": balanced_accuracy_score(
                    y_test, predictions
                ),
                "Precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "Sensitivity": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "F1 Score": f1_score(
                    y_test, predictions, zero_division=0
                ),
                "ROC AUC": roc_auc_score(
                    y_test, patient_probabilities
                ),
                "Training Time Seconds": fold_time
            })

            print(
                f"Fold {fold + 1}/{N_SPLITS} | "
                f"Accuracy: {accuracy:.4f} | "
                f"Time: {fold_time:.2f}s"
            )

        repeat_accuracy = np.mean(repeat_accuracies)

        repeat_results.append({
            "Model": model_name,
            "Repeat": repeat + 1,
            "Mean Fold Accuracy": repeat_accuracy,
            "Fold Accuracy SD": np.std(
                repeat_accuracies, ddof=1
            ),
            "Execution Time Seconds": (
                time.time() - repeat_start
            )
        })

        print(
            f"Repeat {repeat + 1} mean accuracy: "
            f"{repeat_accuracy:.4f}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

fold_df = pd.DataFrame(fold_results)
repeat_df = pd.DataFrame(repeat_results)

fold_df.to_csv(
    OUTPUT_DIR / "repeated_cv_fold_results.csv",
    index=False
)

repeat_df.to_csv(
    OUTPUT_DIR / "repeated_cv_repeat_results.csv",
    index=False
)

summary = (
    fold_df.groupby("Model")
    .agg(
        Accuracy_Mean=("Accuracy", "mean"),
        Accuracy_SD=("Accuracy", "std"),
        Balanced_Accuracy_Mean=("Balanced Accuracy", "mean"),
        Precision_Mean=("Precision", "mean"),
        Sensitivity_Mean=("Sensitivity", "mean"),
        F1_Mean=("F1 Score", "mean"),
        ROC_AUC_Mean=("ROC AUC", "mean"),
        Mean_Fold_Time=("Training Time Seconds", "mean")
    )
    .reset_index()
    .sort_values("Accuracy_Mean", ascending=False)
)

summary.to_csv(
    OUTPUT_DIR / "repeated_cv_summary.csv",
    index=False
)

metadata = {
    "project": "DARWIN Research",
    "evaluation": "Repeated Stratified 5-Fold Cross-Validation",
    "number_of_repeats": N_REPEATS,
    "number_of_folds": N_SPLITS,
    "total_folds_per_model": N_REPEATS * N_SPLITS,
    "random_state": RANDOM_STATE,
    "participants": len(df),
    "features": X.shape[1],
    "device": "CPU",
    "target_mapping": {"H": 0, "P": 1},
    "total_execution_time_seconds": time.time() - overall_start
}

with open(
    OUTPUT_DIR / "repeated_cv_metadata.json",
    "w"
) as file:
    json.dump(metadata, file, indent=4)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 75)
print(" REPEATED CROSS-VALIDATION SUMMARY")
print("=" * 75)

print(summary.to_string(index=False))

print("\nResults saved in:")
print(OUTPUT_DIR.resolve())

print("\nEXPERIMENT COMPLETED SUCCESSFULLY")
