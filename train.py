
"""
DARWIN Research
TabPFN-based Alzheimer's Disease Classification

Evaluation:
- 5-fold Stratified Cross-Validation
- Out-of-fold predictions
- Participant-level evaluation
- Baseline model comparison

Target:
P = Patient
H = Healthy Control
"""

from pathlib import Path
import json
import platform
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
import tabpfn
import matplotlib.pyplot as plt
import seaborn as sns

from tabpfn import TabPFNClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

for folder in [METRICS_DIR, FIGURES_DIR, PREDICTIONS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

TARGET_COLUMN = "class"
ID_COLUMN = "ID"

# H = 0, P = 1
LABEL_MAP = {"H": 0, "P": 1}


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset():

    csv_files = list(DATA_DIR.glob("*.csv"))

    if len(csv_files) != 1:
        raise ValueError(
            f"Expected exactly one CSV in {DATA_DIR}. "
            f"Found {len(csv_files)}."
        )

    path = csv_files[0]
    df = pd.read_csv(path)

    print("\nDATASET INFORMATION")
    print("-" * 50)

    print("File:", path.name)
    print("Shape:", df.shape)

    if TARGET_COLUMN not in df.columns:
        raise ValueError("Target column not found.")

    if ID_COLUMN not in df.columns:
        raise ValueError("Participant ID column not found.")

    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate participant IDs detected.")

    if set(df[TARGET_COLUMN].dropna().unique()) != set(LABEL_MAP):
        raise ValueError("Unexpected target labels.")

    X = df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    y = df[TARGET_COLUMN].map(LABEL_MAP)

    if not X.select_dtypes(include=np.number).shape[1] == X.shape[1]:
        raise ValueError("Non-numeric features detected.")

    if X.isnull().any().any() or y.isnull().any():
        raise ValueError("Missing values detected.")

    if X.shape[1] != 450:
        raise ValueError(f"Expected 450 features, found {X.shape[1]}.")

    print("Participants:", len(df))
    print("Features:", X.shape[1])
    print("Class distribution:")
    print(df[TARGET_COLUMN].value_counts())

    return df, X, y


# ============================================================
# 3. MODEL DEFINITIONS
# ============================================================

def create_models():

    models = {

        "TabPFN": TabPFNClassifier(
            device="cpu"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "SVM": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=1.0,
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    }

    return models


# ============================================================
# 4. METRIC CALCULATION
# ============================================================

def calculate_metrics(y_true, predictions, probabilities):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0

    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced Accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Sensitivity": recall_score(
            y_true, predictions, zero_division=0
        ),
        "Specificity": specificity,
        "F1 Score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "ROC AUC": roc_auc_score(
            y_true, probabilities
        )
    }


# ============================================================
# 5. CROSS-VALIDATION
# ============================================================

def run_experiments(df, X, y):

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    models = create_models()

    all_results = []
    all_predictions = {}
    all_probabilities = {}

    for model_name, model in models.items():

        print("\n" + "=" * 60)
        print("MODEL:", model_name)
        print("=" * 60)

        oof_predictions = np.full(len(X), -1)
        oof_probabilities = np.full(len(X), np.nan)

        fold_scores = []

        for fold, (train_idx, test_idx) in enumerate(
            cv.split(X, y), start=1
        ):

            print(f"\nFold {fold}/{N_SPLITS}")

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            start_time = time.time()

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            probabilities = model.predict_proba(X_test)[:, 1]

            elapsed = time.time() - start_time

            oof_predictions[test_idx] = predictions
            oof_probabilities[test_idx] = probabilities

            fold_accuracy = accuracy_score(
                y_test, predictions
            )

            fold_scores.append(fold_accuracy)

            print(f"Accuracy: {fold_accuracy:.4%}")
            print(f"Time: {elapsed:.2f} seconds")

        if np.any(oof_predictions == -1):
            raise RuntimeError("Some participants lack predictions.")

        metrics = calculate_metrics(
            y.to_numpy(),
            oof_predictions,
            oof_probabilities
        )

        metrics["Model"] = model_name
        metrics["CV Accuracy Mean"] = np.mean(fold_scores)
        metrics["CV Accuracy SD"] = np.std(fold_scores, ddof=1)

        all_results.append(metrics)

        all_predictions[model_name] = oof_predictions
        all_probabilities[model_name] = oof_probabilities

        print("\nOUT-OF-FOLD RESULTS")
        for metric, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                print(f"{metric}: {value:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                y,
                oof_predictions,
                labels=[0, 1],
                target_names=["Healthy", "Patient"],
                zero_division=0
            )
        )

        save_confusion_matrix(
            y,
            oof_predictions,
            model_name
        )

    results_df = pd.DataFrame(all_results)

    results_df = results_df[
        [
            "Model",
            "Accuracy",
            "Balanced Accuracy",
            "Precision",
            "Sensitivity",
            "Specificity",
            "F1 Score",
            "ROC AUC",
            "CV Accuracy Mean",
            "CV Accuracy SD"
        ]
    ]

    results_df.to_csv(
        METRICS_DIR / "model_comparison.csv",
        index=False
    )

    # Save every participant's out-of-fold predictions
    prediction_df = pd.DataFrame({
        "ID": df[ID_COLUMN],
        "Actual": df[TARGET_COLUMN]
    })

    for name in all_predictions:

        safe_name = name.lower().replace(" ", "_")

        prediction_df[f"{safe_name}_prediction"] = [
            "P" if x == 1 else "H"
            for x in all_predictions[name]
        ]

        prediction_df[f"{safe_name}_probability_P"] = (
            all_probabilities[name]
        )

    prediction_df.to_csv(
        PREDICTIONS_DIR / "oof_predictions.csv",
        index=False
    )

    return results_df, all_probabilities, all_predictions


# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(y_true, predictions, model_name):

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Healthy", "Patient"],
        yticklabels=["Healthy", "Patient"]
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_")

    plt.savefig(
        FIGURES_DIR / f"{filename}_confusion_matrix.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 7. MODEL COMPARISON CHART
# ============================================================

def save_comparison_chart(results):

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=results,
        x="Model",
        y="Accuracy"
    )

    plt.ylim(0, 1)
    plt.ylabel("Out-of-fold Accuracy")
    plt.title("DARWIN Model Performance Comparison")

    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "model_accuracy_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 8. SAVE EXPERIMENT METADATA
# ============================================================

def save_metadata():

    metadata = {
        "project": "DARWIN Research",
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "tabpfn_version": tabpfn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "evaluation": "5-fold Stratified Cross-Validation",
        "random_state": RANDOM_STATE,
        "number_of_folds": N_SPLITS,
        "device": "CPU",
        "target_mapping": LABEL_MAP
    }

    with open(
        METRICS_DIR / "experiment_metadata.json",
        "w"
    ) as file:

        json.dump(metadata, file, indent=4)


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def main():

    print("\n")
    print("=" * 65)
    print(" DARWIN - ALZHEIMER'S CLASSIFICATION RESEARCH")
    print("=" * 65)

    start = time.time()

    df, X, y = load_dataset()

    results, probabilities, predictions = run_experiments(
        df, X, y
    )

    save_comparison_chart(results)
    save_metadata()

    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON")
    print("=" * 65)

    print(
        results.sort_values(
            "Accuracy",
            ascending=False
        ).to_string(index=False)
    )

    print("\nBest model:", results.iloc[
        results["Accuracy"].argmax()
    ]["Model"])

    print(
        "\nBest accuracy:",
        f"{results['Accuracy'].max():.4%}"
    )

    print(
        "\nTotal execution time:",
        f"{(time.time() - start) / 60:.2f} minutes"
    )

    print("\nResults saved in:", OUTPUT_DIR)

    print("\nEXPERIMENT COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
