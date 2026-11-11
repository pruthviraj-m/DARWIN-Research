import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/data.csv")
RANKING_PATH = Path("outputs/feature_selection/feature_ranking_by_fold.csv")
OUTPUT_DIR = Path("outputs/feature_stability")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K_VALUES = [50, 100, 150]
CORRELATION_THRESHOLD = 0.85
STABLE_SELECTION_FREQUENCY = 0.80  # selected in at least 20 of 25 folds


def main():
    if not RANKING_PATH.exists():
        raise FileNotFoundError(f"Missing: {RANKING_PATH.resolve()}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing: {DATA_PATH.resolve()}")

    rankings = pd.read_csv(RANKING_PATH)
    required = {"Repeat", "Fold", "Rank", "Feature", "Importance"}
    if not required.issubset(rankings.columns):
        raise ValueError(f"Ranking file must contain: {sorted(required)}")

    fold_count = rankings[["Repeat", "Fold"]].drop_duplicates().shape[0]
    expected_features_per_fold = rankings.groupby(
        ["Repeat", "Fold"]
    ).size().unique()

    if fold_count != 25 or len(expected_features_per_fold) != 1:
        raise ValueError(
            "Expected 25 folds with a complete ranking of the same "
            "number of features in every fold."
        )

    # One row per feature, summarizing its ranking and importance over folds.
    stability = (
        rankings.groupby("Feature")
        .agg(
            Mean_Rank=("Rank", "mean"),
            SD_Rank=("Rank", "std"),
            Best_Rank=("Rank", "min"),
            Worst_Rank=("Rank", "max"),
            Mean_Importance=("Importance", "mean"),
            SD_Importance=("Importance", "std"),
        )
        .reset_index()
    )

    for k in TOP_K_VALUES:
        selected = rankings[rankings["Rank"] <= k]
        counts = selected.groupby("Feature").size()
        stability[f"Top_{k}_Folds"] = (
            stability["Feature"].map(counts).fillna(0).astype(int)
        )
        stability[f"Top_{k}_Frequency"] = (
            stability[f"Top_{k}_Folds"] / fold_count
        )

    stability = stability.sort_values(
        ["Top_50_Folds", "Mean_Rank"],
        ascending=[False, True],
    )
    stability.to_csv(
        OUTPUT_DIR / "feature_stability_summary.csv", index=False
    )

    # Stable candidates: in top 150 in >=80% of folds.
    stable = stability[
        stability["Top_150_Frequency"] >= STABLE_SELECTION_FREQUENCY
    ].copy()
    stable.to_csv(
        OUTPUT_DIR / "stable_feature_candidates.csv", index=False
    )

    # Calculate pairwise Spearman correlations for stable candidates
    # using the original participant-level data. This is descriptive
    # analysis, not model validation.
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["ID", "class"])
    X = X.apply(pd.to_numeric, errors="raise")
    stable_names = stable["Feature"].tolist()

    missing = [name for name in stable_names if name not in X.columns]
    if missing:
        raise ValueError(f"Features missing from dataset: {missing}")

    if len(stable_names) >= 2:
        corr = X[stable_names].corr(method="spearman")
        corr.to_csv(OUTPUT_DIR / "stable_feature_spearman_matrix.csv")

        pairs = []
        for i, feature_a in enumerate(stable_names):
            for feature_b in stable_names[i + 1:]:
                rho = corr.loc[feature_a, feature_b]
                if abs(rho) >= CORRELATION_THRESHOLD:
                    pairs.append({
                        "Feature_A": feature_a,
                        "Feature_B": feature_b,
                        "Spearman_Rho": rho,
                        "Absolute_Rho": abs(rho),
                    })

        pair_df = pd.DataFrame(
            pairs,
            columns=[
                "Feature_A", "Feature_B",
                "Spearman_Rho", "Absolute_Rho"
            ],
        )
        if not pair_df.empty:
            pair_df = pair_df.sort_values(
                "Absolute_Rho", ascending=False
            )
        pair_df.to_csv(
            OUTPUT_DIR / "stable_feature_high_correlations.csv",
            index=False,
        )

    print("\nFEATURE STABILITY ANALYSIS COMPLETE")
    print(f"Number of folds: {fold_count}")
    print(f"Stable candidates (top 150 in >=80% folds): {len(stable)}")
    print("\nTop 20 features by top-50 selection frequency:")
    columns = [
        "Feature", "Mean_Rank", "SD_Rank",
        "Mean_Importance", "Top_50_Folds",
        "Top_100_Folds", "Top_150_Folds"
    ]
    print(stability[columns].head(20).to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
