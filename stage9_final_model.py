"""DARWIN Stage 9: Final model development and artifact export
===========================================================
Run from the project root:
    python stage9_final_model.py

Fits TabPFN and Random Forest on all available participants after selecting
features with Extra Trees ranking (top 150) and training-data Spearman pruning.
This is model fitting, NOT an unbiased performance evaluation. Use Stage 8
nested-CV outer-fold results for internal validation; external validation
remains necessary.
"""

from pathlib import Path
import json
import time
import warnings
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
TOP_FEATURE_COUNT = 150
CORRELATION_THRESHOLD = 0.85

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "data.csv"
OUTPUT_DIR = ROOT / "outputs" / "stage9"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if not {"ID", "class"}.issubset(df.columns):
        raise ValueError("Dataset must contain ID and class columns.")

    labels = df["class"].astype(str).str.strip()
    y = labels.map({"H": 0, "P": 1, "0": 0, "1": 1})
    if y.isna().any():
        raise ValueError(f"Unexpected class labels: {sorted(labels.unique())}")

    cols = [
        c for c in df.columns
        if c not in ("ID", "class") and not str(c).startswith("total_time")
    ]
    X = df[cols].apply(pd.to_numeric, errors="raise")

    if X.isna().any().any():
        raise ValueError("Missing feature values detected.")
    if X.shape[1] != 425:
        raise ValueError(f"Expected 425 features after removing total_time; got {X.shape[1]}.")

    return X, y


def select_final_features(X, y):
    ranker = ExtraTreesClassifier(
        n_estimators=500, max_features="sqrt", class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    ranker.fit(X, y)
    importance = pd.Series(ranker.feature_importances_, index=X.columns)
    ranked = importance.sort_values(ascending=False)
    candidates = ranked.head(TOP_FEATURE_COUNT).index.tolist()

    corr = X[candidates].corr(method="spearman").abs()
    selected, removed = [], []
    for feature in candidates:
        conflict = next(
            (prior for prior in selected
             if corr.loc[feature, prior] >= CORRELATION_THRESHOLD),
            None
        )
        if conflict is None:
            selected.append(feature)
        else:
            removed.append({
                "feature": feature,
                "removed_because_of": conflict,
                "absolute_spearman_correlation": float(corr.loc[feature, conflict])
            })
    return selected, removed


def main():
    start = time.time()
    X, y = load_dataset()

    print(f"Participants: {len(X)} | Features before selection: {X.shape[1]}")
    print("Selecting final feature set...")
    features, removed = select_final_features(X, y)
    X_final = X[features]
    print(f"Selected {len(features)} features.")

    print("Training TabPFN...")
    from tabpfn import TabPFNClassifier
    tabpfn = TabPFNClassifier(device="cpu")
    tabpfn.fit(X_final, y)

    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=400, max_features="sqrt", class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_final, y)

    tabpfn_path = OUTPUT_DIR / "tabpfn_model.joblib"
    rf_path = OUTPUT_DIR / "random_forest_model.joblib"
    features_path = OUTPUT_DIR / "selected_features.json"
    metadata_path = OUTPUT_DIR / "model_metadata.json"
    predictor_path = OUTPUT_DIR / "predict_new_data.py"

    joblib.dump(tabpfn, tabpfn_path)
    joblib.dump(rf, rf_path)

    feature_data = {
        "selected_features": features,
        "candidate_pool_size": TOP_FEATURE_COUNT,
        "correlation_threshold": CORRELATION_THRESHOLD,
        "method": "Extra Trees ranking on full training data, top 150, greedy Spearman pruning",
        "removed_redundant_features": removed
    }
    features_path.write_text(json.dumps(feature_data, indent=2), encoding="utf-8")

    metadata = {
        "project": "DARWIN - Handwriting-Based Alzheimer's Disease Classification",
        "stage": 9,
        "participants_used_for_final_fit": int(len(X)),
        "class_mapping": {"H": 0, "P": 1},
        "input_features": int(X.shape[1]),
        "selected_features": int(len(features)),
        "feature_selection_method": feature_data["method"],
        "correlation_threshold": CORRELATION_THRESHOLD,
        "models": {
            "TabPFN": {
                "artifact": tabpfn_path.name,
                "parameters": {"device": "cpu"}
            },
            "Random Forest": {
                "artifact": rf_path.name,
                "parameters": {
                    "n_estimators": 400,
                    "max_features": "sqrt",
                    "class_weight": "balanced",
                    "random_state": RANDOM_STATE
                }
            }
        },
        "software": {
            "python": sys.version,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__
        },
        "runtime_seconds": round(time.time() - start, 2),
        "validation_note": (
            "Models were fitted on all available participants. This fit does not "
            "estimate generalisation. Use Stage 8 nested-CV outer-fold results "
            "for internal validation; independent external validation is still needed."
        )
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    predictor_code = r'''from pathlib import Path
import json
import sys
import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
FEATURES = json.loads((HERE / "selected_features.json").read_text())["selected_features"]

def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python predict_new_data.py path\\to\\input.csv")
    input_path = Path(sys.argv[1]).resolve()
    data = pd.read_csv(input_path)
    missing = [f for f in FEATURES if f not in data.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    X = data[FEATURES].apply(pd.to_numeric, errors="raise")

    choice = input("Choose model [1=TabPFN, 2=Random Forest]: ").strip()
    if choice == "1":
        model_path, model_name = HERE / "tabpfn_model.joblib", "TabPFN"
    elif choice == "2":
        model_path, model_name = HERE / "random_forest_model.joblib", "Random Forest"
    else:
        raise SystemExit("Invalid choice.")

    model = joblib.load(model_path)
    pred = model.predict(X).astype(int)
    prob = model.predict_proba(X)[:, list(model.classes_).index(1)]
    result = data[["ID"]].copy() if "ID" in data.columns else pd.DataFrame(index=data.index)
    result["predicted_class"] = pred
    result["predicted_label"] = ["P" if p == 1 else "H" for p in pred]
    result["patient_probability"] = prob
    output = input_path.with_name(
        input_path.stem + "_" + model_name.lower().replace(" ", "_") + "_predictions.csv"
    )
    result.to_csv(output, index=False)
    print(f"Saved: {output}")
    print("Research prototype only; this is not a clinical diagnosis.")

if __name__ == "__main__":
    main()
'''
    predictor_path.write_text(predictor_code, encoding="utf-8")

    print("\\nSTAGE 9 COMPLETE")
    print(f"Selected features: {len(features)}")
    print(f"Artifacts saved in: {OUTPUT_DIR}")
    print(f"Runtime: {(time.time() - start)/60:.1f} minutes")
    print("No new performance estimate was produced by this full-data fit.")


if __name__ == "__main__":
    main()