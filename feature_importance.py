
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/data.csv")
OUTPUT_DIR = Path("outputs/feature_importance")

N_SPLITS = 5
N_PERMUTATIONS = 10
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HANDWRITING MEASUREMENTS
# ============================================================

BASE_FEATURES = [
    "air_time",
    "disp_index",
    "gmrt_in_air",
    "gmrt_on_paper",
    "max_x_extension",
    "max_y_extension",
    "mean_acc_in_air",
    "mean_acc_on_paper",
    "mean_gmrt",
    "mean_jerk_in_air",
    "mean_jerk_on_paper",
    "mean_speed_in_air",
    "mean_speed_on_paper",
    "num_of_pendown",
    "paper_time",
    "pressure_mean",
    "pressure_var",
    "total_time"
]


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 65)
print(" DARWIN - FEATURE GROUP IMPORTANCE")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["ID", "class"])
y = df["class"].map({"H": 0, "P": 1})

if y.isna().any():
    raise ValueError("Unexpected class labels.")

if X.isna().any().any():
    raise ValueError("Missing values detected.")

X = X.astype(float)


# ============================================================
# GROUP THE 450 FEATURES INTO 18 MEASUREMENTS
# ============================================================

feature_groups = {}

for base in BASE_FEATURES:

    pattern = re.compile(
        re.escape(base) + r"_?(?:[1-9]|1[0-9]|2[0-5])"
    )

    matching_columns = [
        column for column in X.columns
        if pattern.fullmatch(column)
    ]

    if len(matching_columns) != 25:
        raise ValueError(
            f"{base}: expected 25 columns, "
            f"found {len(matching_columns)}"
        )

    feature_groups[base] = matching_columns

total_grouped = sum(
    len(columns) for columns in feature_groups.values()
)

if total_grouped != X.shape[1]:
    raise ValueError(
        f"Feature grouping mismatch: "
        f"{total_grouped} grouped vs {X.shape[1]} total"
    )

print(f"\nParticipants: {len(X)}")
print(f"Total features: {X.shape[1]}")
print(f"Feature groups: {len(feature_groups)}")

for name, columns in feature_groups.items():
    print(f"{name}: {len(columns)} task features")


# ============================================================
# CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

model = ExtraTreesClassifier(
    n_estimators=500,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

fold_results = []

overall_start = time.time()

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y), start=1
):

    print("\n" + "-" * 60)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("-" * 60)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model.fit(X_train, y_train)

    baseline_predictions = model.predict(X_test)

    baseline_score = balanced_accuracy_score(
        y_test,
        baseline_predictions
    )

    print(f"Baseline balanced accuracy: {baseline_score:.4f}")

    # --------------------------------------------------------
    # GROUPED PERMUTATION IMPORTANCE
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_STATE + fold
    )

    for group_name, columns in feature_groups.items():

        importance_scores = []

        for permutation_number in range(N_PERMUTATIONS):

            # Permute participants jointly across all 25
            # columns belonging to this measurement.
            permutation_indices = rng.permutation(len(X_test))

            X_permuted = X_test.copy()

            X_permuted.loc[:, columns] = (
                X_test.iloc[permutation_indices][columns]
                .to_numpy()
            )

            predictions = model.predict(X_permuted)

            permuted_score = balanced_accuracy_score(
                y_test,
                predictions
            )

            importance = baseline_score - permuted_score

            importance_scores.append(importance)

        group_mean = np.mean(importance_scores)

        fold_results.append({
            "Fold": fold,
            "Feature Group": group_name,
            "Baseline Balanced Accuracy": baseline_score,
            "Importance Mean": group_mean,
            "Importance SD": np.std(
                importance_scores, ddof=1
            )
        })

        print(
            f"{group_name:<25} "
            f"Importance: {group_mean:+.5f}"
        )


# ============================================================
# SUMMARISE RESULTS
# ============================================================

results = pd.DataFrame(fold_results)

results.to_csv(
    OUTPUT_DIR / "feature_importance_by_fold.csv",
    index=False
)

summary = (
    results.groupby("Feature Group")
    .agg(
        Importance_Mean=("Importance Mean", "mean"),
        Importance_SD=("Importance Mean", "std"),
        Positive_Folds=(
            "Importance Mean",
            lambda values: int((values > 0).sum())
        )
    )
    .reset_index()
    .sort_values("Importance_Mean", ascending=False)
)

summary.to_csv(
    OUTPUT_DIR / "feature_group_importance.csv",
    index=False
)


# ============================================================
# VISUALISATION
# ============================================================

plot_data = summary.sort_values("Importance_Mean")

plt.figure(figsize=(11, 9))

plt.barh(
    plot_data["Feature Group"],
    plot_data["Importance_Mean"],
    xerr=plot_data["Importance_SD"],
    capsize=3
)

plt.axvline(0, linewidth=1)

plt.xlabel(
    "Decrease in balanced accuracy after permutation"
)

plt.ylabel("Handwriting measurement")

plt.title(
    "DARWIN - Grouped Permutation Feature Importance"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_group_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 75)
print(" FEATURE GROUP IMPORTANCE RANKING")
print("=" * 75)

print(summary.to_string(index=False))

print("\nInterpretation:")
print(
    "Higher positive importance indicates that disrupting "
    "the feature group reduced classification performance."
)

print(
    "Importance near zero indicates little measured "
    "contribution under this evaluation."
)

print(
    "Negative importance means permutation improved "
    "performance on the evaluated folds."
)

print("\nTotal execution time:")
print(f"{time.time() - overall_start:.2f} seconds")

print("\nResults saved to:")
print(OUTPUT_DIR.resolve())

print("\nFEATURE IMPORTANCE ANALYSIS COMPLETED")
