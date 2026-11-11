
from pathlib import Path
import pandas as pd
import re

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

EXPECTED_FEATURES = [
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

def main():

    files = list(DATA_DIR.glob("*.csv"))

    if len(files) != 1:
        print(f"Expected exactly one CSV file. Found: {len(files)}")
        return

    df = pd.read_csv(files[0])

    print("\n========== DATA VALIDATION ==========")

    print("Dataset shape:", df.shape)

    # Check required columns
    print("\nTarget column exists:", "class" in df.columns)
    print("ID column exists:", "ID" in df.columns)

    # Class distribution
    print("\nClass distribution:")
    print(df["class"].value_counts(dropna=False))

    # Check IDs
    print("\nUnique participant IDs:", df["ID"].nunique())
    print("Duplicate IDs:", df["ID"].duplicated().sum())

    # Expected feature names
    expected = {
        f"{feature}{task}"
        for feature in EXPECTED_FEATURES
        for task in range(1, 26)
    }

    actual = set(df.columns) - {"ID", "class"}

    missing = expected - actual
    unexpected = actual - expected

    print("\nExpected features:", len(expected))
    print("Actual features:", len(actual))
    print("Missing expected features:", len(missing))
    print("Unexpected features:", len(unexpected))

    if missing:
        print("\nMissing feature names:")
        print(sorted(missing))

    if unexpected:
        print("\nUnexpected feature names:")
        print(sorted(unexpected))

    # Check numeric feature types
    numeric_check = df.drop(columns=["ID", "class"], errors="ignore")

    non_numeric = numeric_check.select_dtypes(
        exclude="number"
    ).columns.tolist()

    print("\nNon-numeric feature columns:", len(non_numeric))

    if non_numeric:
        print(non_numeric)

    # Check missing values
    print("\nTotal missing values:", df.isnull().sum().sum())

    # Check duplicate rows
    print("Duplicate rows:", df.duplicated().sum())

    # Check class labels
    print("\nClass labels:", sorted(df["class"].dropna().unique()))

    # Check malformed headers
    malformed = [
        col for col in df.columns
        if col not in ("ID", "class")
        and not re.fullmatch(
            r"(" + "|".join(EXPECTED_FEATURES) + r")([1-9]|1[0-9]|2[0-5])",
            str(col)
        )
    ]

    print("\nMalformed feature headers:", len(malformed))
    if malformed:
        print(malformed)

    # Save report
    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "features": len(actual),
        "unique_ids": df["ID"].nunique(),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "non_numeric_features": len(non_numeric),
        "missing_features": len(missing),
        "unexpected_features": len(unexpected),
        "malformed_headers": len(malformed)
    }

    pd.DataFrame([report]).to_csv(
        BASE_DIR / "outputs" / "metrics" / "data_validation.csv",
        index=False
    )

    print("\nValidation report saved.")
    print("\n========== VALIDATION COMPLETE ==========")


if __name__ == "__main__":
    main()
