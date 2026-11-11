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
from tabpfn import TabPFNClassifier


# ============================================================
# DARWIN - FAST CROSS-MODEL FEATURE AGREEMENT
# ============================================================
# This is a DESCRIPTIVE agreement experiment, not a new unbiased
# performance estimate. It uses the previously identified stable
# candidate pool and inner validation data only.
# ============================================================

DATA_PATH = Path("data/data.csv")
CANDIDATE_PATH = Path("outputs/feature_stability/stable_feature_candidates.csv")
OUTPUT_DIR = Path("outputs/cross_model_agreement_fast")

N_SPLITS = 5
N_REPEATS = 2
RANDOM_STATE = 42
CANDIDATE_LIMIT = 50
PERMUTATION_REPEATS = 1
INNER_VALIDATION_SIZE = 0.25

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_models():
    return {
        "TabPFN": TabPFNClassifier(device="cpu"),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, max_features="sqrt",
            class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_features="sqrt",
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
    print("\n" + "=" * 72)
    print(" DARWIN - FAST CROSS-MODEL FEATURE AGREEMENT")
    print("=" * 72)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")
    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(
            f"Stable candidate file not found: {CANDIDATE_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)
    y = df["class"].map({"H": 0, "P": 1})
    if y.isna().any():
        raise ValueError("Unexpected class labels.")

    X = df.drop(columns=["ID", "class"]).apply(
        pd.to_numeric, errors="raise"
    ).astype(float)

    total_cols = [c for c in X.columns if c.startswith("total_time")]
    if len(total_cols) != 25:
        raise ValueError(f"Expected 25 total_time columns, found {len(total_cols)}")
    X = X.drop(columns=total_cols)

    candidates_df = pd.read_csv(CANDIDATE_PATH)
    if "Feature" not in candidates_df.columns:
        raise ValueError("Candidate CSV must contain a 'Feature' column.")

    candidates = candidates_df["Feature"].drop_duplicates().tolist()
    candidates = [f for f in candidates if f in X.columns][:CANDIDATE_LIMIT]
    if len(candidates) < 2:
        raise ValueError("Not enough valid stable candidate features.")

    X = X[candidates]

    print(f"Participants: {len(X)}")
    print(f"Candidate features: {len(candidates)}")
    print(f"Outer CV: {N_SPLITS} folds x {N_REPEATS} repeats")
    print(f"Permutation repeats: {PERMUTATION_REPEATS}")
    print("Ranking uses inner validation only.")
    print("NOTE: candidate pool was identified in prior exploratory analysis.")

    splits = list(RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS,
        random_state=RANDOM_STATE
    ).split(X, y))

    ranking_rows = []
    overlap_rows = []
    total_fits = len(splits) * len(create_models())
    done = 0
    started = time.time()

    for split_idx, (train_idx, _outer_test_idx) in enumerate(splits):
        repeat = split_idx // N_SPLITS + 1
        fold = split_idx % N_SPLITS + 1

        X_outer_train = X.iloc[train_idx]
        y_outer_train = y.iloc[train_idx]

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

        fold_rankings = {}

        print(f"\nRepeat {repeat}/{N_REPEATS}, Fold {fold}/{N_SPLITS}")

        for model_name, model in create_models().items():
            done += 1
            t0 = time.time()
            model.fit(X_inner_train, y_inner_train)

            perm = permutation_importance(
                model,
                X_inner_val,
                y_inner_val,
                scoring="balanced_accuracy",
                n_repeats=PERMUTATION_REPEATS,
                random_state=RANDOM_STATE + split_idx,
                n_jobs=1
            )

            scores = pd.Series(
                perm.importances_mean,
                index=candidates
            ).sort_values(ascending=False, kind="stable")

            fold_rankings[model_name] = scores

            for rank, (feature, score) in enumerate(scores.items(), start=1):
                ranking_rows.append({
                    "Repeat": repeat,
                    "Fold": fold,
                    "Model": model_name,
                    "Rank": rank,
                    "Feature": feature,
                    "Permutation_Importance": float(score)
                })

            print(
                f"[{done}/{total_fits}] {model_name}: "
                f"top={scores.index[0]}, "
                f"importance={scores.iloc[0]:.5f}, "
                f"time={time.time()-t0:.1f}s"
            )

        names = list(fold_rankings)
        for k in [10, 20, 25, 50]:
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    sa = set(fold_rankings[a].head(k).index)
                    sb = set(fold_rankings[b].head(k).index)
                    union = sa | sb
                    jac = len(sa & sb) / len(union) if union else 0.0
                    overlap_rows.append({
                        "Repeat": repeat,
                        "Fold": fold,
                        "Top_K": k,
                        "Model_A": a,
                        "Model_B": b,
                        "Shared_Features": len(sa & sb),
                        "Jaccard_Similarity": jac
                    })

    rankings = pd.DataFrame(ranking_rows)
    rankings.to_csv(
        OUTPUT_DIR / "cross_model_feature_rankings_by_fold.csv",
        index=False
    )

    overlaps = pd.DataFrame(overlap_rows)
    overlaps.to_csv(
        OUTPUT_DIR / "cross_model_overlap_by_fold.csv",
        index=False
    )

    overlap_summary = (
        overlaps.groupby(["Top_K", "Model_A", "Model_B"])
        .agg(
            Mean_Shared_Features=("Shared_Features", "mean"),
            Mean_Jaccard=("Jaccard_Similarity", "mean"),
            SD_Jaccard=("Jaccard_Similarity", "std")
        )
        .reset_index()
        .sort_values(["Top_K", "Mean_Jaccard"], ascending=[True, False])
    )
    overlap_summary.to_csv(
        OUTPUT_DIR / "cross_model_agreement_summary.csv",
        index=False
    )

    # Feature-level ranking summaries for each model.
    stability_rows = []
    for model, group in rankings.groupby("Model"):
        for feature, fg in group.groupby("Feature"):
            row = {
                "Model": model,
                "Feature": feature,
                "Mean_Rank": fg["Rank"].mean(),
                "SD_Rank": fg["Rank"].std(),
                "Mean_Permutation_Importance": fg["Permutation_Importance"].mean(),
                "SD_Permutation_Importance": fg["Permutation_Importance"].std()
            }
            for k in [10, 20, 25]:
                row[f"Top_{k}_Folds"] = int((fg["Rank"] <= k).sum())
            stability_rows.append(row)

    pd.DataFrame(stability_rows).to_csv(
        OUTPUT_DIR / "cross_model_feature_stability.csv",
        index=False
    )

    # Consensus: number of models selecting each feature in top-k,
    # summed over model-fold observations.
    consensus_rows = []
    for k in [10, 20, 25]:
        selected = rankings[rankings["Rank"] <= k]
        counts = selected.groupby("Feature").agg(
            Model_Count=("Model", "nunique"),
            Model_Fold_Appearances=("Model", "size")
        ).reset_index()
        counts["Top_K"] = k
        consensus_rows.append(counts)

    pd.concat(consensus_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "cross_model_consensus_features.csv",
        index=False
    )

    metadata = {
        "experiment": "Fast descriptive cross-model feature agreement",
        "participants": int(len(X)),
        "candidate_pool_size": int(len(candidates)),
        "candidate_pool_source": str(CANDIDATE_PATH),
        "outer_cv": "RepeatedStratifiedKFold",
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "inner_validation_size": INNER_VALIDATION_SIZE,
        "permutation_repeats": PERMUTATION_REPEATS,
        "models": list(create_models().keys()),
        "total_model_fits": total_fits,
        "total_execution_time_seconds": time.time() - started,
        "limitation": (
            "Candidate pool comes from prior exploratory feature selection. "
            "This is descriptive agreement analysis, not an unbiased estimate "
            "of final model performance or independent external validation."
        )
    }
    with open(OUTPUT_DIR / "cross_model_agreement_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\n" + "=" * 72)
    print(" CROSS-MODEL AGREEMENT SUMMARY")
    print("=" * 72)
    print(overlap_summary.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_DIR.resolve()}")
    print("EXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()
