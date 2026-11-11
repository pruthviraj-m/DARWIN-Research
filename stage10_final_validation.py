"""
DARWIN Stage 10 — Nested-CV validation diagnostics
==================================================
Re-runs the Stage 8 nested-CV selection procedure and additionally saves
outer-fold predictions for confusion matrices, ROC-AUC, sensitivity,
specificity, precision, F1, Brier score and calibration diagnostics.

This is internal validation on the same 174-participant dataset, not
independent external validation. Because Stage 8 results informed the research
choices, treat Stage 10 as a reproducible diagnostic analysis, not a new
independent confirmation.

Run from project root:
    python stage10_final_validation.py

Input:
    data/data.csv

Outputs:
    outputs/stage10/oof_predictions.csv
    outputs/stage10/fold_metrics.csv
    outputs/stage10/summary_metrics.csv
    outputs/stage10/confusion_matrices.csv
    outputs/stage10/roc_curve_points.csv
    outputs/stage10/calibration_bins.csv
    outputs/stage10/metadata.json
"""

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    brier_score_loss, roc_curve
)

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
CORR_THRESHOLD = 0.85
STRATEGIES = ["All 425", "Top 150", "Top 100", "Correlation-pruned"]
MODELS = ["TabPFN", "Extra Trees", "Random Forest", "SVM"]

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "data.csv"
OUT = ROOT / "outputs" / "stage10"
OUT.mkdir(parents=True, exist_ok=True)


def make_model(name):
    if name == "TabPFN":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device="cpu")
    if name == "Extra Trees":
        return ExtraTreesClassifier(
            n_estimators=400, max_features="sqrt", class_weight="balanced",
            random_state=SEED, n_jobs=-1
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=400, max_features="sqrt", class_weight="balanced",
            random_state=SEED, n_jobs=-1
        )
    if name == "SVM":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(
                kernel="rbf", C=1.0, probability=True,
                class_weight="balanced", random_state=SEED
            ))
        ])
    raise ValueError(name)


def select_features(X_train, y_train, strategy):
    if strategy == "All 425":
        return list(X_train.columns)

    ranker = ExtraTreesClassifier(
        n_estimators=400, max_features="sqrt", class_weight="balanced",
        random_state=SEED, n_jobs=-1
    )
    ranker.fit(X_train, y_train)
    ranked = list(X_train.columns[np.argsort(ranker.feature_importances_)[::-1]])

    if strategy == "Top 150":
        return ranked[:150]
    if strategy == "Top 100":
        return ranked[:100]
    if strategy == "Correlation-pruned":
        candidates = ranked[:150]
        corr = X_train[candidates].corr(method="spearman").abs()
        kept = []
        for feature in candidates:
            if all(corr.loc[feature, prior] < CORR_THRESHOLD for prior in kept):
                kept.append(feature)
        return kept
    raise ValueError(strategy)


def positive_prob(model, X):
    probs = model.predict_proba(X)
    return probs[:, list(model.classes_).index(1)]


def main():
    start = time.time()
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    labels = df["class"].astype(str).str.strip()
    y = labels.map({"H": 0, "P": 1, "0": 0, "1": 1})
    if y.isna().any():
        raise ValueError(f"Unexpected labels: {sorted(labels.unique())}")

    feature_cols = [
        c for c in df.columns
        if c not in ("ID", "class") and not str(c).startswith("total_time")
    ]
    X = df[feature_cols].apply(pd.to_numeric, errors="raise")
    if X.shape[1] != 425:
        raise ValueError(f"Expected 425 features, found {X.shape[1]}")
    if X.isna().any().any():
        raise ValueError("Missing values detected.")

    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    pred_rows, fold_rows = [], []

    print(f"Participants: {len(df)} | features: {X.shape[1]}")
    print("Running nested CV and collecting outer-fold predictions.")
    print("Runtime may be substantial, especially for TabPFN on CPU.")

    for fold, (tr, te) in enumerate(outer.split(X, y), start=1):
        print(f"\nOUTER FOLD {fold}/{OUTER_FOLDS}")
        Xtr, ytr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
        Xte, yte = X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True)

        inner = StratifiedKFold(
            n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + fold
        )

        for model_name in MODELS:
            scores = {s: [] for s in STRATEGIES}
            for itr, iva in inner.split(Xtr, ytr):
                Xi, yi = Xtr.iloc[itr], ytr.iloc[itr]
                Xv, yv = Xtr.iloc[iva], ytr.iloc[iva]
                for strategy in STRATEGIES:
                    feats = select_features(Xi, yi, strategy)
                    model = make_model(model_name)
                    model.fit(Xi[feats], yi)
                    pred = model.predict(Xv[feats])
                    scores[strategy].append(balanced_accuracy_score(yv, pred))

            mean_scores = {s: float(np.mean(v)) for s, v in scores.items()}
            chosen = max(mean_scores, key=mean_scores.get)
            features = select_features(Xtr, ytr, chosen)

            final_model = make_model(model_name)
            final_model.fit(Xtr[features], ytr)
            predictions = final_model.predict(Xte[features]).astype(int)
            probabilities = positive_prob(final_model, Xte[features])

            tn, fp, fn, tp = confusion_matrix(yte, predictions, labels=[0, 1]).ravel()
            metrics = {
                "accuracy": accuracy_score(yte, predictions),
                "balanced_accuracy": balanced_accuracy_score(yte, predictions),
                "sensitivity": recall_score(yte, predictions, pos_label=1, zero_division=0),
                "specificity": recall_score(yte, predictions, pos_label=0, zero_division=0),
                "precision": precision_score(yte, predictions, pos_label=1, zero_division=0),
                "f1": f1_score(yte, predictions, pos_label=1, zero_division=0),
                "roc_auc": roc_auc_score(yte, probabilities),
                "brier_score": brier_score_loss(yte, probabilities),
            }
            fold_rows.append({
                "outer_fold": fold, "model": model_name,
                "chosen_strategy": chosen, "feature_count": len(features),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                **metrics
            })

            for local_i, original_i in enumerate(te):
                pred_rows.append({
                    "participant_row": int(original_i),
                    "participant_id": str(df.iloc[original_i]["ID"]),
                    "outer_fold": fold,
                    "model": model_name,
                    "chosen_strategy": chosen,
                    "true_class": int(y.iloc[original_i]),
                    "predicted_class": int(predictions[local_i]),
                    "patient_probability": float(probabilities[local_i])
                })

            print(
                f"{model_name}: {chosen}, {len(features)} features, "
                f"outer balanced accuracy {metrics['balanced_accuracy']*100:.2f}%"
            )

    pred_df = pd.DataFrame(pred_rows)
    fold_df = pd.DataFrame(fold_rows)

    # Fold-level mean and SD; do not treat folds as independent for significance tests.
    metric_cols = [
        "accuracy", "balanced_accuracy", "sensitivity", "specificity",
        "precision", "f1", "roc_auc", "brier_score"
    ]
    summary = fold_df.groupby("model")[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()

    # Aggregate confusion counts across the five outer folds for each model.
    confusion = fold_df.groupby("model")[["tn", "fp", "fn", "tp"]].sum().reset_index()

    # ROC points from pooled outer-fold predictions; useful descriptive curve.
    roc_rows = []
    calibration_rows = []
    for model_name in MODELS:
        part = pred_df[pred_df["model"] == model_name]
        fpr, tpr, thresholds = roc_curve(part["true_class"], part["patient_probability"])
        for a, b, threshold in zip(fpr, tpr, thresholds):
            roc_rows.append({
                "model": model_name, "false_positive_rate": float(a),
                "true_positive_rate": float(b), "threshold": float(threshold)
            })

        bins = pd.cut(
            part["patient_probability"], bins=np.linspace(0, 1, 11),
            include_lowest=True
        )
        cal = part.assign(probability_bin=bins).groupby(
            "probability_bin", observed=False
        ).agg(
            participant_count=("true_class", "size"),
            mean_predicted_probability=("patient_probability", "mean"),
            observed_patient_rate=("true_class", "mean")
        ).reset_index()
        cal.insert(0, "model", model_name)
        calibration_rows.extend(cal.to_dict("records"))

    pred_df.to_csv(OUT / "oof_predictions.csv", index=False)
    fold_df.to_csv(OUT / "fold_metrics.csv", index=False)
    summary.to_csv(OUT / "summary_metrics.csv", index=False)
    confusion.to_csv(OUT / "confusion_matrices.csv", index=False)
    pd.DataFrame(roc_rows).to_csv(OUT / "roc_curve_points.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(OUT / "calibration_bins.csv", index=False)

    metadata = {
        "stage": 10,
        "participants": int(len(df)),
        "input_features": int(X.shape[1]),
        "outer_cv": f"{OUTER_FOLDS}-fold stratified",
        "inner_cv": f"{INNER_FOLDS}-fold stratified",
        "models": MODELS,
        "strategies": STRATEGIES,
        "selection_metric": "inner balanced accuracy",
        "outer_predictions_saved": True,
        "validation_scope": "Internal nested cross-validation only",
        "important_limitations": [
            "No independent external dataset was available.",
            "Stage 10 reuses the same dataset and selection procedure as Stage 8; it is not independent confirmation.",
            "Pooled OOF predictions combine predictions from different fold-trained models.",
            "Fold-level metrics are not independent observations; avoid naive significance tests.",
            "Calibration bins are descriptive and small-sample estimates.",
            "This is research, not a clinical diagnostic validation."
        ],
        "runtime_seconds": round(time.time() - start, 2)
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nSTAGE 10 COMPLETE")
    print(summary.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
    print(f"Outputs: {OUT}")
    print(f"Runtime: {(time.time()-start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
