
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
OUTPUT_DIR = Path("outputs/task_feature_importance")

N_SPLITS = 5
N_PERMUTATIONS = 10
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TOP FIVE MEASUREMENT GROUPS
# ============================================================

SELECTED_GROUPS = [
    "total_time",
    "air_time",
    "paper_time",
    "pressure_var",
    "num_of_pendown"
]


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 65)
print(" DARWIN - TASK-LEVEL FEATURE IMPORTANCE")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["ID", "class"])
y = df["class"].map({"H": 0, "P": 1})

if y.isna().any():
    raise ValueError("Unexpected class labels.")

if X.isna().any().any():
    raise ValueError("Missing feature values detected.")

X = X.astype(float)


# ============================================================
# IDENTIFY THE 125 FEATURES
# ============================================================

selected_features = []

for group in SELECTED_GROUPS:

    pattern = re.compile(
        re.escape(group) + r"_?(?:[1-9]|1[0-9]|2[0-5])"
    )

    matching = [
        col for col in X.columns
        if pattern.fullmatch(col)
    ]

    if len(matching) != 25:
        raise ValueError(
            f"{group}: expected 25 features, found {len(matching)}"
        )

    selected_features.extend(matching)

if len(selected_features) != 125:
    raise ValueError(
        f"Expected 125 features, found {len(selected_features)}"
    )

print(f"\nParticipants: {len(X)}")
print(f"Selected groups: {len(SELECTED_GROUPS)}")
print(f"Selected individual features: {len(selected_features)}")


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

start_time = time.time()

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y), start=1
):

    print("\n" + "-" * 65)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("-" * 65)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model.fit(X_train, y_train)

    baseline_predictions = model.predict(X_test)

    baseline_score = balanced_accuracy_score(
        y_test, baseline_predictions
    )

    print(f"Baseline balanced accuracy: {baseline_score:.4f}")

    rng = np.random.default_rng(
        RANDOM_STATE + fold
    )

    for index, feature in enumerate(selected_features, start=1):

        importance_scores = []

        for _ in range(N_PERMUTATIONS):

            permutation_indices = rng.permutation(len(X_test))

            X_permuted = X_test.copy()

            X_permuted[feature] = (
                X_test[feature]
                .iloc[permutation_indices]
                .to_numpy()
            )

            predictions = model.predict(X_permuted)

            permuted_score = balanced_accuracy_score(
                y_test, predictions
            )

            importance_scores.append(
                baseline_score - permuted_score
            )

        fold_results.append({
            "Fold": fold,
            "Feature": feature,
            "Importance Mean": np.mean(importance_scores),
            "Importance SD": np.std(
                importance_scores, ddof=1
            )
        })

        if index % 25 == 0:
            print(
                f"Processed {index}/{len(selected_features)} features"
            )


# ============================================================
# SAVE FOLD-LEVEL RESULTS
# ============================================================

results = pd.DataFrame(fold_results)

results.to_csv(
    OUTPUT_DIR / "task_importance_by_fold.csv",
    index=False
)


# ============================================================
# SUMMARISE INDIVIDUAL FEATURES
# ============================================================

summary = (
    results.groupby("Feature")
    .agg(
        Importance_Mean=("Importance Mean", "mean"),
        Importance_SD=("Importance Mean", "std"),
        Positive_Folds=(
            "Importance Mean",
            lambda values: int((values > 0).sum())
        )
    )
    .reset_index()
    .sort_values(
        "Importance_Mean",
        ascending=False
    )
)

summary.to_csv(
    OUTPUT_DIR / "task_feature_importance.csv",
    index=False
)


# ============================================================
# ADD GROUP INFORMATION
# ============================================================

def identify_group(feature):

    for group in SELECTED_GROUPS:

        if feature.startswith(group):
            return group

    return "Unknown"


summary["Measurement Group"] = (
    summary["Feature"].apply(identify_group)
)

summary.to_csv(
    OUTPUT_DIR / "task_feature_importance.csv",
    index=False
)


# ============================================================
# TOP 25 FEATURE VISUALISATION
# ============================================================

top_features = (
    summary.head(25)
    .sort_values("Importance_Mean")
)

plt.figure(figsize=(12, 9))

plt.barh(
    top_features["Feature"],
    top_features["Importance_Mean"],
    xerr=top_features["Importance_SD"],
    capsize=3
)

plt.axvline(0, linewidth=1)

plt.xlabel(
    "Decrease in balanced accuracy after permutation"
)

plt.ylabel("Individual handwriting feature")

plt.title(
    "DARWIN - Top 25 Task-Level Features"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "top_25_task_features.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 75)
print(" TOP 25 TASK-LEVEL FEATURES")
print("=" * 75)

print(summary.head(25).to_string(index=False))

print("\nTotal execution time:")
print(f"{time.time() - start_time:.2f} seconds")

print("\nResults saved to:")
print(OUTPUT_DIR.resolve())

print("\nTASK-LEVEL ANALYSIS COMPLETED")
