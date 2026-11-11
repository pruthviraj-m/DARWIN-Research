
"""
DARWIN Stage 11B — TabPFN Explainability (Revised)

Run from project root:
    python stage11_tabpfn_efficient.py

Uses existing Stage 9 TabPFN model. No retraining.

Methods:
1. Subset permutation importance using ROC-AUC and log-loss.
2. Local median-replacement probability sensitivity.
3. Output validation to ensure all expected rows are saved.

Exploratory, in-sample analysis only.
"""

from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    log_loss
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SEED = 42
MAX_PARTICIPANTS = 40
MAX_FEATURES = 25
REPEATS = 3

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "data.csv"
S9 = ROOT / "outputs" / "stage9"
OUT = ROOT / "outputs" / "stage11_tabpfn_efficient"

OUT.mkdir(parents=True, exist_ok=True)


def get_probability(model, X):
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    return probabilities[:, classes.index(1)]


def main():

    start = time.time()

    # ------------------------------------------------
    # 1. Load and validate data
    # ------------------------------------------------

    required = [
        DATA,
        S9 / "selected_features.json",
        S9 / "tabpfn_model.joblib"
    ]

    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing files:\n" + "\n".join(missing)
        )

    df = pd.read_csv(DATA)

    y = (
        df["class"]
        .astype(str)
        .str.strip()
        .map({"H": 0, "P": 1, "0": 0, "1": 1})
    )

    if y.isna().any():
        raise ValueError("Unexpected class labels.")

    feature_info = json.loads(
        (S9 / "selected_features.json").read_text(
            encoding="utf-8"
        )
    )

    all_features = feature_info["selected_features"]

    missing_features = [
        f for f in all_features if f not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    # Preserve ALL 118 model features in original order.
    X_all = df[all_features].apply(
        pd.to_numeric,
        errors="raise"
    ).astype(float)

    if X_all.isna().any().any():
        raise ValueError("Missing feature values.")

    # ------------------------------------------------
    # 2. Select stratified participant subset
    # ------------------------------------------------

    if len(df) > MAX_PARTICIPANTS:

        _, subset_idx = train_test_split(
            np.arange(len(df)),
            test_size=MAX_PARTICIPANTS,
            stratify=y,
            random_state=SEED
        )

        subset_idx = np.sort(subset_idx)

    else:
        subset_idx = np.arange(len(df))

    X = (
        X_all.iloc[subset_idx]
        .reset_index(drop=True)
    )

    y_sub = (
        y.iloc[subset_idx]
        .reset_index(drop=True)
    )

    features_to_test = all_features[
        :min(MAX_FEATURES, len(all_features))
    ]

    print("Loading existing TabPFN model...")
    model = joblib.load(
        S9 / "tabpfn_model.joblib"
    )

    fitted_names = getattr(
        model, "feature_names_in_", None
    )

    if fitted_names is not None:
        if list(fitted_names) != list(all_features):
            raise ValueError(
                "Feature names/order do not match fitted model."
            )

    print(f"Participants: {len(X)}")
    print(f"Complete model features: {len(all_features)}")
    print(f"Features tested: {len(features_to_test)}")
    print(f"Permutation repeats: {REPEATS}")

    # ------------------------------------------------
    # 3. Baseline predictions and metrics
    # ------------------------------------------------

    print("\nCalculating baseline...")

    baseline_pred = model.predict(X).astype(int)
    baseline_prob = get_probability(model, X)

    baseline_balanced_accuracy = balanced_accuracy_score(
        y_sub, baseline_pred
    )

    baseline_auc = roc_auc_score(
        y_sub, baseline_prob
    )

    baseline_logloss = log_loss(
        y_sub, baseline_prob, labels=[0, 1]
    )

    print(
        f"Balanced accuracy: {baseline_balanced_accuracy:.4f}"
    )
    print(f"ROC-AUC: {baseline_auc:.4f}")
    print(f"Log-loss: {baseline_logloss:.6f}")

    # ------------------------------------------------
    # 4. Permutation importance
    # ------------------------------------------------

    print("\nStarting permutation importance...")

    rng = np.random.default_rng(SEED)
    importance_rows = []

    for number, feature in enumerate(
        features_to_test, start=1
    ):

        auc_drops = []
        logloss_increases = []
        accuracy_drops = []

        for repeat in range(REPEATS):

            X_perm = X.copy()

            X_perm[feature] = rng.permutation(
                X_perm[feature].to_numpy()
            )

            perm_prob = get_probability(
                model, X_perm
            )

            perm_pred = (perm_prob >= 0.5).astype(int)

            perm_auc = roc_auc_score(
                y_sub, perm_prob
            )

            perm_logloss = log_loss(
                y_sub,
                perm_prob,
                labels=[0, 1]
            )

            perm_accuracy = balanced_accuracy_score(
                y_sub, perm_pred
            )

            auc_drops.append(
                baseline_auc - perm_auc
            )

            logloss_increases.append(
                perm_logloss - baseline_logloss
            )

            accuracy_drops.append(
                baseline_balanced_accuracy - perm_accuracy
            )

        importance_rows.append({

            "feature": feature,

            "baseline_roc_auc": baseline_auc,

            "roc_auc_drop_mean": float(
                np.mean(auc_drops)
            ),

            "roc_auc_drop_std": float(
                np.std(auc_drops, ddof=1)
            ),

            "baseline_log_loss": baseline_logloss,

            "log_loss_increase_mean": float(
                np.mean(logloss_increases)
            ),

            "log_loss_increase_std": float(
                np.std(logloss_increases, ddof=1)
            ),

            "balanced_accuracy_drop_mean": float(
                np.mean(accuracy_drops)
            ),

            "repeats": REPEATS,

            "subset_n": len(X)
        })

        print(
            f"[{number}/{len(features_to_test)}] "
            f"{feature} | "
            f"AUC drop: {np.mean(auc_drops):.6f} | "
            f"Log-loss increase: "
            f"{np.mean(logloss_increases):.6f}"
        )

    importance_df = pd.DataFrame(
        importance_rows
    ).sort_values(
        "log_loss_increase_mean",
        ascending=False
    )

    importance_path = (
        OUT / "tabpfn_permutation_importance_subset.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False
    )

    # ------------------------------------------------
    # 5. Local median-replacement sensitivity
    # ------------------------------------------------

    print("\nStarting local sensitivity analysis...")

    local_rows = []

    for cls, label in [
        (0, "healthy"),
        (1, "patient")
    ]:

        positions = np.flatnonzero(
            y_sub.to_numpy() == cls
        )

        if len(positions) == 0:
            raise ValueError(
                f"No representative participant for {label}."
            )

        class_probs = baseline_prob[positions]

        # Select the participant closest to the class
        # median predicted probability.
        chosen_pos = int(
            positions[
                np.argmin(
                    np.abs(
                        class_probs - np.median(class_probs)
                    )
                )
            ]
        )

        original = X.iloc[[chosen_pos]].copy()

        original_probability = float(
            baseline_prob[chosen_pos]
        )

        # Create 25 independent perturbations.
        perturbed_rows = []

        for feature in features_to_test:

            row = original.copy()

            row[feature] = float(
                X[feature].median()
            )

            perturbed_rows.append(row)

        batch = pd.concat(
            perturbed_rows,
            ignore_index=True
        )

        # Predict all perturbations together.
        changed_probs = get_probability(
            model, batch
        )

        # Verify prediction count before pairing results.
        if len(changed_probs) != len(features_to_test):

            print(
                "Batch output mismatch. "
                "Retrying local predictions individually..."
            )

            changed_probs = []

            for row in perturbed_rows:

                probability = get_probability(
                    model, row
                )

                if len(probability) != 1:
                    raise RuntimeError(
                        "Unexpected local prediction shape."
                    )

                changed_probs.append(
                    float(probability[0])
                )

        if len(changed_probs) != len(features_to_test):
            raise RuntimeError(
                "Local explanation row count mismatch."
            )

        for feature, changed_probability in zip(
            features_to_test, changed_probs
        ):

            local_rows.append({

                "representative_class": label,

                "participant_row": int(
                    subset_idx[chosen_pos]
                ),

                "feature": feature,

                "original_patient_probability":
                    original_probability,

                "probability_after_median_replacement":
                    float(changed_probability),

                "probability_change":
                    float(
                        changed_probability
                        - original_probability
                    ),

                "replacement_value":
                    float(X[feature].median())
            })

        print(
            f"{label}: generated "
            f"{len(features_to_test)} local explanations."
        )

    local_df = pd.DataFrame(local_rows)

    expected_rows = (
        len(features_to_test) * 2
    )

    if len(local_df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} local rows, "
            f"but generated {len(local_df)}."
        )

    local_path = (
        OUT / "tabpfn_local_median_sensitivity.csv"
    )

    local_df.to_csv(
        local_path,
        index=False
    )

    # ------------------------------------------------
    # 6. Save metadata
    # ------------------------------------------------

    metadata = {

        "stage": "11B",

        "model": "TabPFN",

        "participants_in_dataset": int(len(df)),

        "subset_participants": int(len(X)),

        "model_input_features": int(len(all_features)),

        "features_tested": int(len(features_to_test)),

        "permutation_repeats": REPEATS,

        "baseline_balanced_accuracy":
            float(baseline_balanced_accuracy),

        "baseline_roc_auc":
            float(baseline_auc),

        "baseline_log_loss":
            float(baseline_logloss),

        "subset_sampling":
            "Stratified random subset",

        "permutation_metrics": [
            "ROC-AUC drop",
            "Log-loss increase",
            "Balanced accuracy drop"
        ],

        "local_explanation_rows": int(len(local_df)),

        "expected_local_explanation_rows":
            int(expected_rows),

        "limitations": [

            "Exploratory in-sample analysis.",

            "Final model was fitted on all 174 participants.",

            "Only 40 participants were used for this analysis.",

            "Only 25 of 118 features were tested.",

            "Correlated features can distort permutation importance.",

            "Median replacement is sensitivity analysis, not SHAP.",

            "No external validation or clinical validity established."
        ],

        "runtime_seconds": round(
            time.time() - start, 2
        )
    }

    (OUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    # ------------------------------------------------
    # 7. Final report
    # ------------------------------------------------

    print("\n======================================")
    print("STAGE 11B COMPLETE")
    print("======================================")

    print("\nTop 15 TabPFN features:")
    print(
        importance_df.head(15).to_string(
            index=False
        )
    )

    print(
        f"\nLocal explanation rows: "
        f"{len(local_df)} / {expected_rows}"
    )

    print(f"\nOutputs saved to: {OUT}")

    print(
        f"Runtime: "
        f"{(time.time() - start) / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()
