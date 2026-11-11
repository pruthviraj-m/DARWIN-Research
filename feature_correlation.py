
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/data.csv")

IMPORTANCE_PATH = Path(
    "outputs/task_feature_importance/task_feature_importance.csv"
)

OUTPUT_DIR = Path("outputs/feature_correlation")

CORRELATION_THRESHOLD = 0.85

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SELECTED MEASUREMENT GROUPS
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
print(" DARWIN - FEATURE CORRELATION AND REDUNDANCY")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
importance_df = pd.read_csv(IMPORTANCE_PATH)

X = df.drop(columns=["ID", "class"])

if X.isna().any().any():
    raise ValueError("Missing feature values detected.")

X = X.astype(float)


# ============================================================
# SELECT THE 125 FEATURES
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

X_selected = X[selected_features]

print(f"\nParticipants: {len(X_selected)}")
print(f"Selected features: {len(selected_features)}")


# ============================================================
# SPEARMAN CORRELATION
# ============================================================

correlation_matrix = X_selected.corr(method="spearman")

correlation_matrix.to_csv(
    OUTPUT_DIR / "correlation_matrix_125.csv"
)

print("\nSpearman correlation matrix generated.")


# ============================================================
# IDENTIFY HIGHLY CORRELATED PAIRS
# ============================================================

pairs = []

for i in range(len(selected_features)):

    for j in range(i + 1, len(selected_features)):

        feature_1 = selected_features[i]
        feature_2 = selected_features[j]

        correlation = correlation_matrix.loc[
            feature_1, feature_2
        ]

        if abs(correlation) >= CORRELATION_THRESHOLD:

            pairs.append({
                "Feature 1": feature_1,
                "Feature 2": feature_2,
                "Spearman Correlation": correlation,
                "Absolute Correlation": abs(correlation)
            })


pairs_df = pd.DataFrame(
    pairs,
    columns=[
        "Feature 1",
        "Feature 2",
        "Spearman Correlation",
        "Absolute Correlation"
    ]
)

if not pairs_df.empty:

    pairs_df = pairs_df.sort_values(
        "Absolute Correlation",
        ascending=False
    )

pairs_df.to_csv(
    OUTPUT_DIR / "high_correlation_pairs.csv",
    index=False
)


# ============================================================
# TOP 25 FEATURE CORRELATION
# ============================================================

top_features = (
    importance_df
    .sort_values("Importance_Mean", ascending=False)
    .head(25)["Feature"]
    .tolist()
)

missing = [
    feature for feature in top_features
    if feature not in X.columns
]

if missing:
    raise ValueError(
        f"Features from importance file not found: {missing}"
    )

top25_matrix = X[top_features].corr(method="spearman")

top25_matrix.to_csv(
    OUTPUT_DIR / "top25_correlation_matrix.csv"
)


# ============================================================
# TOP 25 HEATMAP
# ============================================================

fig, ax = plt.subplots(figsize=(16, 13))

image = ax.imshow(
    top25_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

ax.set_xticks(range(len(top_features)))
ax.set_yticks(range(len(top_features)))

ax.set_xticklabels(
    top_features,
    rotation=90,
    fontsize=8
)

ax.set_yticklabels(
    top_features,
    fontsize=8
)

ax.set_title(
    "DARWIN - Spearman Correlation of Top 25 Features",
    pad=18
)

colorbar = fig.colorbar(image, ax=ax)
colorbar.set_label("Spearman correlation")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "top25_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FEATURE REDUNDANCY SUMMARY
# ============================================================

if not pairs_df.empty:

    counts = pd.concat([
        pairs_df["Feature 1"],
        pairs_df["Feature 2"]
    ]).value_counts()

    redundancy = pd.DataFrame({
        "Feature": counts.index,
        "Highly Correlated Pair Count": counts.values
    })

else:

    redundancy = pd.DataFrame(
        columns=[
            "Feature",
            "Highly Correlated Pair Count"
        ]
    )

redundancy.to_csv(
    OUTPUT_DIR / "feature_redundancy_summary.csv",
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 75)
print(" CORRELATION ANALYSIS RESULTS")
print("=" * 75)

print(
    f"\nCorrelation threshold: "
    f"|Spearman correlation| >= {CORRELATION_THRESHOLD}"
)

print(f"Highly correlated pairs: {len(pairs_df)}")

if not pairs_df.empty:

    print("\nTop 20 highly correlated pairs:")

    print(
        pairs_df.head(20).to_string(index=False)
    )

else:

    print(
        "\nNo feature pairs crossed the selected correlation threshold."
    )

print("\nTop 25 features with the most correlated partners:")

print(redundancy.head(15).to_string(index=False))

print("\nFiles saved to:")
print(OUTPUT_DIR.resolve())

print("\nCORRELATION ANALYSIS COMPLETED")
