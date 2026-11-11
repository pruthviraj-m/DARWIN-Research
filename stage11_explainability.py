"""
DARWIN Stage 11 — Explainable AI
Run from project root: python stage11_explainability.py
If SHAP is missing: python -m pip install shap

Uses the final full-data Random Forest and TabPFN models. All explanations
are descriptive/in-sample, NOT independent validation or causal evidence.
Correlated features can split or distort attribution.
"""
from pathlib import Path
import json, time, warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
SEED = 42
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "data.csv"
S9 = ROOT / "outputs" / "stage9"
OUT = ROOT / "outputs" / "stage11"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    start = time.time()
    needed = [DATA, S9/"selected_features.json", S9/"random_forest_model.joblib", S9/"tabpfn_model.joblib"]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    df = pd.read_csv(DATA)
    y = df["class"].astype(str).str.strip().map({"H":0, "P":1, "0":0, "1":1})
    if y.isna().any(): raise ValueError("Unexpected class labels.")
    features = json.loads((S9/"selected_features.json").read_text(encoding="utf-8"))["selected_features"]
    missing_features = [f for f in features if f not in df.columns]
    if missing_features: raise ValueError(f"Missing selected features: {missing_features}")
    X = df[features].apply(pd.to_numeric, errors="raise")
    if X.isna().any().any(): raise ValueError("Missing values in selected features.")

    print(f"Participants: {len(X)} | selected features: {len(features)}")
    print("Loading final models...")
    rf = joblib.load(S9/"random_forest_model.joblib")
    tabpfn = joblib.load(S9/"tabpfn_model.joblib")

    # Random Forest SHAP
    try:
        import shap
    except ImportError:
        raise SystemExit("SHAP is not installed. Run: python -m pip install shap")

    print("\n[1/3] Random Forest SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(rf)
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        sv = np.asarray(raw[1])
    else:
        sv = np.asarray(raw)
        if sv.ndim == 3:
            sv = sv[:, :, 1] if sv.shape[-1] == 2 else sv[1]
    if sv.shape != X.shape:
        raise ValueError(f"Unexpected SHAP shape {sv.shape}; expected {X.shape}")

    shap_df = pd.DataFrame({
        "feature": features,
        "mean_absolute_shap": np.abs(sv).mean(axis=0),
        "mean_shap_value": sv.mean(axis=0)
    }).sort_values("mean_absolute_shap", ascending=False)
    shap_df.to_csv(OUT/"shap_global_importance.csv", index=False)

    plt.figure()
    shap.summary_plot(sv, X, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(OUT/"shap_summary_beeswarm.png", dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(sv, X, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(OUT/"shap_summary_bar.png", dpi=220, bbox_inches="tight")
    plt.close()

    for feature in shap_df.feature.head(5):
        plt.figure()
        shap.dependence_plot(feature, sv, X, interaction_index="auto", show=False)
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in feature)
        plt.tight_layout()
        plt.savefig(OUT/f"shap_dependence_{safe}.png", dpi=200, bbox_inches="tight")
        plt.close()

    # Illustrative local explanations for one record from each class.
    for cls, label in [(0, "healthy"), (1, "patient")]:
        idxs = np.where(y.to_numpy() == cls)[0]
        if len(idxs) == 0: continue
        idx = int(idxs[0])
        try:
            base_values = explainer.expected_value
            if isinstance(base_values, (list, np.ndarray)):
                base_values = np.asarray(base_values).reshape(-1)[-1]
            exp = shap.Explanation(
                values=sv[idx], base_values=float(base_values),
                data=X.iloc[idx].to_numpy(), feature_names=features
            )
            plt.figure()
            shap.plots.waterfall(exp, max_display=15, show=False)
            plt.tight_layout()
            plt.savefig(OUT/f"shap_local_{label}.png", dpi=220, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Local {label} plot skipped: {e}")

    # Model-agnostic permutation importance for both models.
    def perm(model, name):
        print(f"Permutation importance: {name}...")
        r = permutation_importance(
            model, X, y, scoring="balanced_accuracy",
            n_repeats=10, random_state=SEED, n_jobs=1
        )
        return pd.DataFrame({
            "model": name, "feature": features,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std
        }).sort_values("importance_mean", ascending=False)

    print("\n[2/3] Permutation importance (training data; descriptive only)...")
    rf_perm = perm(rf, "Random Forest")
    tab_perm = perm(tabpfn, "TabPFN")
    pd.concat([rf_perm, tab_perm]).to_csv(OUT/"permutation_importance.csv", index=False)

    meta = {
        "stage": 11, "participants": int(len(X)), "feature_count": len(features),
        "methods": {
            "Random Forest": ["SHAP TreeExplainer", "Permutation importance"],
            "TabPFN": ["Permutation importance"]
        },
        "permutation_repeats": 10,
        "top_10_random_forest_shap": shap_df.feature.head(10).tolist(),
        "top_10_random_forest_permutation": rf_perm.feature.head(10).tolist(),
        "top_10_tabpfn_permutation": tab_perm.feature.head(10).tolist(),
        "limitations": [
            "Explanations use final models fitted on all participants and are in-sample.",
            "Permutation importance on training data can be optimistic.",
            "Correlated features can split or distort attributions.",
            "SHAP values describe model behaviour, not causal effects or clinical biomarkers.",
            "No external validation or clinical validity is established."
        ],
        "runtime_seconds": round(time.time()-start, 2)
    }
    (OUT/"metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nTOP 15 RANDOM FOREST SHAP FEATURES")
    print(shap_df.head(15).to_string(index=False))
    print("\nTOP 10 RANDOM FOREST PERMUTATION FEATURES")
    print(rf_perm.head(10).to_string(index=False))
    print("\nTOP 10 TABPFN PERMUTATION FEATURES")
    print(tab_perm.head(10).to_string(index=False))
    print(f"\nSaved outputs: {OUT}")
    print(f"Runtime: {(time.time()-start)/60:.1f} minutes")

if __name__ == "__main__":
    main()
