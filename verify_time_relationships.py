import pandas as pd
import numpy as np
import re
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATA_PATH = Path("data/data.csv")
OUTPUT_DIR = Path("outputs/time_relationships")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("DARWIN DATASET: TIME FEATURE RELATIONSHIP ANALYSIS")
print("=" * 65)

# --------------------------------------------------
# FIND FEATURE COLUMNS
# Supports names such as total_time1 or total_time_1
# --------------------------------------------------

def find_column(prefix, task):
    patterns = [
        f"{prefix}{task}",
        f"{prefix}_{task}"
    ]

    for col in df.columns:
        if col in patterns:
            return col

    return None


results = []

# --------------------------------------------------
# TEST ALL 25 TASKS
# --------------------------------------------------

for task in range(1, 26):

    total_col = find_column("total_time", task)
    air_col = find_column("air_time", task)
    paper_col = find_column("paper_time", task)

    if not all([total_col, air_col, paper_col]):
        print(f"Task {task}: Required column missing. Skipping.")
        continue

    total = pd.to_numeric(df[total_col], errors="coerce")
    air = pd.to_numeric(df[air_col], errors="coerce")
    paper = pd.to_numeric(df[paper_col], errors="coerce")

    calculated_total = air + paper

    residual = total - calculated_total
    absolute_error = residual.abs()

    exact_matches = np.isclose(
        total,
        calculated_total,
        rtol=0,
        atol=0,
        equal_nan=False
    )

    approximate_matches = np.isclose(
        total,
        calculated_total,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False
    )

    valid = total.notna() & air.notna() & paper.notna()

    valid_count = int(valid.sum())

    results.append({
        "task": task,
        "total_time_column": total_col,
        "air_time_column": air_col,
        "paper_time_column": paper_col,
        "valid_rows": valid_count,
        "exact_matches": int(exact_matches[valid].sum()),
        "exact_match_percentage": (
            exact_matches[valid].mean() * 100
            if valid_count else np.nan
        ),
        "approximate_matches": int(approximate_matches[valid].sum()),
        "approximate_match_percentage": (
            approximate_matches[valid].mean() * 100
            if valid_count else np.nan
        ),
        "mean_absolute_error": absolute_error[valid].mean(),
        "maximum_absolute_error": absolute_error[valid].max(),
        "mean_residual": residual[valid].mean(),
        "spearman_correlation": pd.DataFrame({
            "total": total,
            "calculated": calculated_total
        }).corr(method="spearman").iloc[0, 1]
    })


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

output_file = OUTPUT_DIR / "time_relationship_results.csv"
results_df.to_csv(output_file, index=False)

# --------------------------------------------------
# OVERALL SUMMARY
# --------------------------------------------------

if not results_df.empty:

    total_rows = results_df["valid_rows"].sum()
    exact_rows = results_df["exact_matches"].sum()
    approximate_rows = results_df["approximate_matches"].sum()

    print("\nOVERALL RESULTS")
    print("-" * 65)

    print(f"Tasks tested: {len(results_df)}")
    print(f"Total valid comparisons: {total_rows}")
    print(
        f"Exact matches: {exact_rows}/{total_rows} "
        f"({exact_rows / total_rows * 100:.2f}%)"
    )
    print(
        f"Approximate matches: {approximate_rows}/{total_rows} "
        f"({approximate_rows / total_rows * 100:.2f}%)"
    )

    print("\nTASK-WISE RESULTS")
    print("-" * 65)

    print(
        results_df[
            [
                "task",
                "exact_match_percentage",
                "approximate_match_percentage",
                "mean_absolute_error",
                "maximum_absolute_error",
                "spearman_correlation"
            ]
        ].round(6).to_string(index=False)
    )

    print(f"\nResults saved to: {output_file}")

else:
    print("No valid task comparisons found. Check the column names.")

print("\nAnalysis complete.")