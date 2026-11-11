import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier


# ============================================================
# DARWIN - CROSS-MODEL FEATURE AGREEMENT
# ============================================================
# For each outer CV fold:
# 1. Split outer-training participants into inner-train/inner-validation.
# 2. Fit each model on inner-train only.
# 3. Rank features using validation permutation importance.
# 4. Record top-k overlap/agreement across models.
#
# This stage studies agreement between algorithms. It does not use
# the outer test fold for feature ranking.
# ============================================================

DATA_PATH = Path("data/data.csv")
OUTPUT_DIR = Path("outputs/cross_model_agreement")

N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42

INNER_VALIDATION_SIZE = 0.25
PERMUTATION_REPEATS = 1  # Keep runtime manageable
TOP_K_VALUES = [25, 50, 100, 150]

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


def main():
    print("\n" + "=" * 75)
    print(" DARWIN - CROSS-MODEL FEATURE AGREEMENT")
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
        raise ValueError(f"Expected 25 total_time features; found {len(total_time_cols)}")
    X = X.drop(columns=total_time_cols)

    if X.shape[1] != 425:
        raise ValueError(f"Expected 425 features after removing total_time; got {X.shape[1]}")
    if X.isna().any().any():
        raise ValueError("Missing feature values found.")

    print(f"Participants: {len(X)}")
    print(f"Features ranked: {X.shape[1]}")
    print(f"Outer CV: {N_SPLITS} folds x {N_REPEATS} repeats")
    print(f"Permutation repeats: {PERMUTATION_REPEATS}")
    print("Ranking uses an inner validation split; outer test folds remain untouched.")

    outer_cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS,
        random_state=RANDOM_STATE
    )
    outer_splits = list(outer_cv.split(X, y))

    ranking_rows = []
    overlap_rows = []
    start_all = time.time()
    total_fits = len(outer_splits) * len(create_models())
    fit_counter = 0

    for split_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        repeat = split_idx // N_SPLITS + 1
        fold = split_idx % N_SPLITS + 1

        X_outer_train = X.iloc[outer_train_idx]
        y_outer_train = y.iloc[outer_train_idx]

        inner_train_idx, inner_val_idx = train_test_split(
            np.arange(len(X_outer_train)),
            test_size=INNER_VALIDATION_SIZE,
            random_state=RANDOM_STATE + split_idx,
            stratify=y_outer_train
        )

        X_inner_train = X_outer_train.iloc[inner_train_idx]
        y_inner_train = y_outer_train.iloc[inner_train_idx]
        X_inner_val = X_outer_train.iloc[inner_val_idx]
        y_inner_val = y_outer_train.iloc[inner_val_idx]

        rankings_this_fold = {}

        print(f"\nRepeat {repeat}/{N_REPEATS}, Fold {fold}/{N_SPLITS}")

        for model_name, model in create_models().items():
            fit_counter += 1
            fit_start = time.time()

            model.fit(X_inner_train, y_inner_train)

            # Permutation importance is calculated on inner validation data,
            # never on the outer test fold.
            perm = permutation_importance(
                model,
                X_inner_val,
                y_inner_val,
                scoring="balanced_accuracy",
                n_repeats=PERMUTATION_REPEATS,
                random_state=RANDOM_STATE + split_idx,
                n_jobs=1,
            )

            importance = pd.Series(
                perm.importances_mean,
                index=X.columns,
                name="Importance"
            ).sort_values(ascending=False, kind="stable")

            rankings_this_fold[model_name] = importance

            for rank, (feature, score) in enumerate(importance.items(), start=1):
                ranking_rows.append({
                    "Repeat": repeat,
                    "Fold": fold,
                    "Model": model_name,
                    "Rank": rank,
                    "Feature": feature,
                    "Permutation_Importance": float(score),
                })

            elapsed = time.time() - fit_start
            print(
                f"[{fit_counter}/{total_fits}] {model_name:12s} "
                f"| top feature: {importance.index[0]} "
                f"| importance: {importance.iloc[0]:.5f} "
                f"| {elapsed:.1f}s"
            )

        # Pairwise top-k Jaccard similarity for each fold.
        model_names = list(rankings_this_fold)
        for k in TOP_K_VALUES:
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model_a, model_b = model_names[i], model_names[j]
                    set_a = set(rankings_this_fold[model_a].head(k).index)
                    set_b = set(rankings_this_fold[model_b].head(k).index)
                    union = set_a | set_b
                    jaccard = len(set_a & set_b) / len(union) if union else 0.0

                    overlap_rows.append({
                        "Repeat": repeat,
                        "Fold": fold,
                        "Top_K": k,
                        "Model_A": model_a,
                        "Model_B": model_b,
                        "Shared_Features": len(set_a & set_b),
                        "Jaccard_Similarity": jaccard,
                    })

    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df.to_csv(
        OUTPUT_DIR / "cross_model_feature_rankings_by_fold.csv",
        index=False
    )

    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(
        OUTPUT_DIR / "cross_model_overlap_by_fold.csv",
        index=False
    )

    # Per-model feature stability across all 25 inner validation folds.
    stability_rows = []
    for model_name, group in ranking_df.groupby("Model"):
        for feature, feature_group in group.groupby("Feature"):
            row = {
                "Model": model_name,
                "Feature": feature,
                "Mean_Rank": feature_group["Rank"].mean(),
                "SD_Rank": feature_group["Rank"].std(),
                "Mean_Permutation_Importance": feature_group[
                    "Permutation_Importance"
                ].mean(),
                "SD_Permutation_Importance": feature_group[
                    "Permutation_Importance"
                ].std(),
            }
            for k in TOP_K_VALUES:
                row[f"Top_{k}_Folds"] = int((feature_group["Rank"] <= k).sum())
                row[f"Top_{k}_Frequency"] = row[f"Top_{k}_Folds"] / len(outer_splits)
            stability_rows.append(row)

    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv(
        OUTPUT_DIR / "cross_model_feature_stability.csv",
        index=False
    )

    overlap_summary = (
        overlap_df.groupby(["Top_K", "Model_A", "Model_B"])
        .agg(
            Mean_Shared_Features=("Shared_Features", "mean"),
            Mean_Jaccard=("Jaccard_Similarity", "mean"),
            SD_Jaccard=("Jaccard_Similarity", "std"),
        )
        .reset_index()
        .sort_values(["Top_K", "Mean_Jaccard"], ascending=[True, False])
    )
    overlap_summary.to_csv(
        OUTPUT_DIR / "cross_model_agreement_summary.csv",
        index=False
    )

    # Features that are repeatedly top-ranked across all four models.
    consensus_rows = []
    for k in TOP_K_VALUES:
        selected = ranking_df[ranking_df["Rank"] <= k]
        counts = (
            selected.groupby("Feature")["Model"].nunique()
            .rename("Models_Selected")
        )
        fold_counts = (
            selected.groupby("Feature").size()
            .rename("Total_TopK_Appearances")
        )
        consensus = pd.concat([counts, fold_counts], axis=1).fillna(0)
        consensus["Top_K"] = k
        consensus["Models_Selected"] = consensus["Models_Selected"].astype(int)
        consensus["Total_TopK_Appearances"] = (
            consensus["Total_TopK_Appearances"].astype(int)
        )
        consensus = consensus.reset_index()
        consensus_rows.append(consensus)

    pd.concat(consensus_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "cross_model_consensus_features.csv",
        index=False
    )

    metadata = {
        "experiment": "Cross-model feature agreement using inner-validation permutation importance",
        "participants": int(len(X)),
        "features": int(X.shape[1]),
        "removed_derived_features": total_time_cols,
        "outer_cv": "RepeatedStratifiedKFold",
        "outer_splits": N_SPLITS,
        "outer_repeats": N_REPEATS,
        "inner_validation_size": INNER_VALIDATION_SIZE,
        "permutation_repeats": PERMUTATION_REPEATS,
        "scoring": "balanced_accuracy",
        "models": list(create_models().keys()),
        "ranking_note": (
            "Each model is fitted on an inner training subset and ranked on "
            "its inner validation subset. Outer test folds are not used."
        ),
        "total_model_fits": total_fits,
        "total_execution_time_seconds": time.time() - start_all,
        "limitation": (
            "Permutation importance is model- and validation-sample-dependent. "
            "Correlated features can share or mask importance. This is an "
            "agreement analysis, not external clinical validation."
        ),
    }
    with open(OUTPUT_DIR / "cross_model_agreement_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\n" + "=" * 75)
    print(" CROSS-MODEL AGREEMENT SUMMARY")
    print("=" * 75)
    print(overlap_summary.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")
    print("EXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()
