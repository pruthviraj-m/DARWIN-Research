from pathlib import Path
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
