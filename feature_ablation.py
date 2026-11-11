
import pandas as pd
import numpy as np
import re
import json
import time

from pathlib import Path

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from tabpfn import TabPFNClassifier


# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_PATH = Path("data/data.csv")
OUTPUT_DIR = Path("outputs/feature_ablation")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 5

TARGET_COL = "class"
ID_COL = "ID"


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 75)
print("DARWIN RESEARCH - FEATURE ABLATION EXPERIMENT")
print("=" * 75)

df = pd.read_csv(DATA_PATH)

if TARGET_COL not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COL}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

y = df[TARGET_COL].map({"H": 0, "P": 1})

if y.isna().any():
    raise ValueError("Unexpected or missing target labels.")

# Exclude participant ID and target from predictors.
excluded_columns = [TARGET_COL]

if ID_COL in df.columns:
    excluded_columns.append(ID_COL)

X_full = df.drop(columns=excluded_columns)

# Check all predictor columns are numeric.
non_numeric = X_full.select_dtypes(exclude=[np.number]).columns.tolist()

if non_numeric:
    raise ValueError(f"Non-numeric predictor columns found: {non_numeric}")

print(f"\nDataset shape: {df.shape}")
print(f"Participants: {len(df)}")
print(f"Original features: {X_full.shape[1]}")
print(f"Healthy controls: {(y == 0).sum()}")
print(f"Alzheimer's participants: {(y == 1).sum()}")


# ==========================================================
# IDENTIFY TASK-SPECIFIC TIME FEATURES
# ==========================================================

def get_task_features(columns, prefix):

    pattern = re.compile(rf"^{prefix}_?\d+$")

    return [
        col for col in columns
        if pattern.match(str(col))
    ]


total_time_features = get_task_features(
    X_full.columns, "total_time"
)

air_time_features = get_task_features(
    X_full.columns, "air_time"
)

paper_time_features = get_task_features(
    X_full.columns, "paper_time"
)

if len(total_time_features) != 25:
    raise ValueError(
        f"Expected 25 total_time features, found "
        f"{len(total_time_features)}: {total_time_features}"
    )

if len(air_time_features) != 25:
    raise ValueError(
        f"Expected 25 air_time features, found "
        f"{len(air_time_features)}: {air_time_features}"
    )

print("\nDetected timing features:")
print(f"total_time: {len(total_time_features)}")
print(f"air_time: {len(air_time_features)}")
print(f"paper_time: {len(paper_time_features)}")


# ==========================================================
# DEFINE FEATURE CONFIGURATIONS
# ==========================================================

feature_configs = {

    "A_Baseline_All_Features": X_full.columns.tolist(),

    "B_Remove_Total_Time": [
        col for col in X_full.columns
        if col not in total_time_features
    ],

    "C_Remove_Air_Time": [
        col for col in X_full.columns
        if col not in air_time_features
    ]

}

print("\nFeature configurations:")
print("-" * 75)

for name, columns in feature_configs.items():
    print(f"{name}: {len(columns)} features")


# ==========================================================
# MODEL DEFINITIONS
# Keep identical to the previous repeated CV experiment.
# ==========================================================

def create_models():

    return {

        "TabPFN": TabPFNClassifier(
            device="cpu"
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=1,
                probability=True,
                random_state=RANDOM_STATE
            ))
        ])

    }


# ==========================================================
# GENERATE IDENTICAL CV SPLITS FOR EVERY CONFIGURATION
# ==========================================================

cv = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE
)

splits = list(cv.split(X_full, y))

print(f"\nTotal CV folds: {len(splits)}")
print("Identical splits will be used for all configurations.")


# ==========================================================
# EVALUATION
# ==========================================================

fold_results = []

start_experiment = time.time()

total_fits = (
    len(feature_configs) *
    4 *
    len(splits)
)

completed_fits = 0


for config_name, selected_features in feature_configs.items():

    print("\n" + "=" * 75)
    print(f"CONFIGURATION: {config_name}")
    print(f"FEATURE COUNT: {len(selected_features)}")
    print("=" * 75)

    X = X_full[selected_features]

    for model_name, model_template in create_models().items():

        print(f"\nRunning model: {model_name}")

        for split_index, (train_idx, test_idx) in enumerate(splits):

            repeat_number = split_index // N_SPLITS + 1
            fold_number = split_index % N_SPLITS + 1

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # Fresh model instance for every fold.
            from sklearn.base import clone

            model = clone(model_template)

            fit_start = time.time()

            model.fit(X_train, y_train)

            training_time = time.time() - fit_start

            predictions = model.predict(X_test)

            probabilities = model.predict_proba(X_test)[:, 1]

            metrics = {
                "Configuration": config_name,
                "Model": model_name,
                "Repeat": repeat_number,
                "Fold": fold_number,
                "Feature Count": len(selected_features),

                "Accuracy": accuracy_score(
                    y_test, predictions
                ),

                "Balanced Accuracy": balanced_accuracy_score(
                    y_test, predictions
                ),

                "Precision": precision_score(
                    y_test, predictions,
                    zero_division=0
                ),

                "Sensitivity": recall_score(
                    y_test, predictions,
                    zero_division=0
                ),

                "F1 Score": f1_score(
                    y_test, predictions,
                    zero_division=0
                ),

                "ROC AUC": roc_auc_score(
                    y_test, probabilities
                ),

                "Training Time Seconds": training_time
            }

            fold_results.append(metrics)

            completed_fits += 1

            print(
                f"{completed_fits}/{total_fits} | "
                f"Repeat {repeat_number}, Fold {fold_number} | "
                f"Accuracy: {metrics['Accuracy']:.4f} | "
                f"Balanced Accuracy: "
                f"{metrics['Balanced Accuracy']:.4f} | "
                f"Time: {training_time:.2f}s"
            )


# ==========================================================
# SAVE FOLD RESULTS
# ==========================================================

fold_df = pd.DataFrame(fold_results)

fold_path = OUTPUT_DIR / "ablation_fold_results.csv"

fold_df.to_csv(fold_path, index=False)


# ==========================================================
# REPEAT-LEVEL RESULTS
# ==========================================================

metric_columns = [
    "Accuracy",
    "Balanced Accuracy",
    "Precision",
    "Sensitivity",
    "F1 Score",
    "ROC AUC",
    "Training Time Seconds"
]

repeat_df = (
    fold_df
    .groupby(["Configuration", "Model", "Repeat"])[metric_columns]
    .mean()
    .reset_index()
)

repeat_path = OUTPUT_DIR / "ablation_repeat_results.csv"

repeat_df.to_csv(repeat_path, index=False)


# ==========================================================
# SUMMARY RESULTS
# ==========================================================

summary_df = (
    fold_df
    .groupby(["Configuration", "Model"])[metric_columns]
    .agg(["mean", "std"])
)

summary_df.columns = [
    f"{metric}_{stat}"
    for metric, stat in summary_df.columns
]

summary_df = summary_df.reset_index()

summary_path = OUTPUT_DIR / "ablation_summary.csv"

summary_df.to_csv(summary_path, index=False)


# ==========================================================
# PAIRED FOLD DIFFERENCES
# ==========================================================

baseline = fold_df[
    fold_df["Configuration"] == "A_Baseline_All_Features"
].copy()

comparison_results = []

for config_name in [
    "B_Remove_Total_Time",
    "C_Remove_Air_Time"
]:

    alternative = fold_df[
        fold_df["Configuration"] == config_name
    ].copy()

    merged = baseline.merge(
        alternative,
        on=["Model", "Repeat", "Fold"],
        suffixes=("_baseline", "_alternative")
    )

    for model_name in fold_df["Model"].unique():

        model_data = merged[
            merged["Model"] == model_name
        ]

        for metric in [
            "Accuracy",
            "Balanced Accuracy",
            "F1 Score",
            "ROC AUC"
        ]:

            differences = (
                model_data[f"{metric}_alternative"] -
                model_data[f"{metric}_baseline"]
            )

            comparison_results.append({

                "Comparison": config_name,
                "Model": model_name,
                "Metric": metric,

                "Mean Difference": differences.mean(),

                "Difference SD": differences.std(),

                "Positive Folds": int((differences > 0).sum()),

                "Negative Folds": int((differences < 0).sum()),

                "Unchanged Folds": int((differences == 0).sum())

            })


comparison_df = pd.DataFrame(comparison_results)

comparison_path = OUTPUT_DIR / "ablation_paired_comparisons.csv"

comparison_df.to_csv(comparison_path, index=False)


# ==========================================================
# METADATA
# ==========================================================

metadata = {

    "experiment": "DARWIN Feature Ablation",

    "dataset": str(DATA_PATH),

    "participants": int(len(df)),

    "original_features": int(X_full.shape[1]),

    "configurations": {
        name: {
            "feature_count": len(columns),
            "features_removed": (
                total_time_features
                if name == "B_Remove_Total_Time"
                else air_time_features
                if name == "C_Remove_Air_Time"
                else []
            )
        }
        for name, columns in feature_configs.items()
    },

    "models": [
        "TabPFN",
        "Extra Trees",
        "Random Forest",
        "SVM"
    ],

    "cross_validation": {
        "method": "RepeatedStratifiedKFold",
        "splits": N_SPLITS,
        "repeats": N_REPEATS,
        "total_folds_per_model_configuration": len(splits),
        "random_state": RANDOM_STATE
    },

    "target_mapping": {
        "H": 0,
        "P": 1
    },

    "total_model_fits": total_fits,

    "execution_time_seconds": time.time() - start_experiment

}

metadata_path = OUTPUT_DIR / "ablation_metadata.json"

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)


# ==========================================================
# FINAL OUTPUT
# ==========================================================

print("\n" + "=" * 75)
print("FEATURE ABLATION EXPERIMENT COMPLETED")
print("=" * 75)

print("\nSUMMARY:")
print(summary_df.round(4).to_string(index=False))

print("\nPAIRED COMPARISONS:")
print(comparison_df.round(5).to_string(index=False))

print("\nFiles saved:")
print(fold_path)
print(repeat_path)
print(summary_path)
print(comparison_path)
print(metadata_path)

print(
    f"\nTotal execution time: "
    f"{(time.time() - start_experiment) / 60:.2f} minutes"
)
