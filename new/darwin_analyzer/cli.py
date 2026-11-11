"""Command Line Interface for the DARWIN Handwriting Research Software Suite.

Usage:
    python -m darwin_analyzer extract -i raw_points.csv -o features.csv
    python -m darwin_analyzer validate -i raw_points.csv
    python -m darwin_analyzer predict -i 450_features.csv
    python -m darwin_analyzer report -i session_data.json -o report.html
"""

import argparse
import json
import sys
import pandas as pd
from .dataset_generator import ResearchDatasetGenerator
from .feature_extractor import HandwritingFeatureExtractor
from .model_runner import DarwinModelRunner
from .quality_validator import DataQualityValidator
from .report_generator import ResearchReportGenerator


def cmd_extract(args):
    print(f"Extracting features from '{args.input}'...")
    df = pd.read_csv(args.input)
    extractor = HandwritingFeatureExtractor(pixel_to_mm_ratio=args.calibration)
    result = extractor.extract_task_features(df, participant_id=args.participant, task_number=args.task)

    out_path = args.output or "extracted_features.json"
    if out_path.endswith(".csv"):
        # Flatten canonical features to 1-row CSV
        row = {"participant_id": result["participant_id"], "task_number": result["task_number"]}
        row.update(result["canonical_features"])
        pd.DataFrame([row]).to_csv(out_path, index=False)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
    print(f"Features saved to '{out_path}'.")


def cmd_validate(args):
    print(f"Validating recording quality for '{args.input}'...")
    df = pd.read_csv(args.input)
    validator = DataQualityValidator()
    report = validator.validate_raw_data(df, canvas_width=args.width, canvas_height=args.height)
    print("\n--- Quality Assessment Result ---")
    print(f"Status: {report['overall_status']}")
    print(f"Score:  {report['score']*100:.1f}%")
    print(f"Summary: {report['summary']}")
    if report["flags"]:
        print(f"\nFlags ({len(report['flags'])}):")
        for f in report["flags"]:
            print(f"  [{f['severity']}] {f['code']}: {f['message']}")


def cmd_predict(args):
    print(f"Running DARWIN ML model inference on '{args.input}'...")
    df = pd.read_csv(args.input)
    runner = DarwinModelRunner(classification_threshold=args.threshold)
    pred = runner.predict_sample(
        df,
        is_touchscreen_data=args.touchscreen,
        allow_touchscreen_bypass=args.bypass_touchscreen_guard,
    )

    if not pred["success"]:
        print(f"Inference Blocked/Failed: {pred.get('error')}")
        sys.exit(1)

    print("\n--- Model Prediction Result ---")
    print(f"Class:       {pred['class_label']} (Class {pred['predicted_class']})")
    print(f"Probability: {pred['probability_ad']*100:.1f}% AD vs {pred['probability_healthy']*100:.1f}% Healthy")
    print(f"Threshold:   {pred['threshold_used']}")
    print(f"Uncertainty: {pred['uncertainty_level']}")

    expl = pred.get("explanation", {})
    if expl.get("shap_available"):
        print("\nTop 5 Contributing Tasks (SHAP Attribution):")
        for t in expl["top_tasks"][:5]:
            print(f"  {t['task_label']}: {t['shap_sum']:+.4f}")


def main():
    parser = argparse.ArgumentParser(
        prog="darwin_analyzer",
        description="DARWIN Handwriting Research & Analysis Software Suite CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: extract
    p_ext = subparsers.add_parser("extract", help="Extract features from raw points CSV")
    p_ext.add_argument("-i", "--input", required=True, help="Path to raw points CSV")
    p_ext.add_argument("-o", "--output", help="Output path (.json or .csv)")
    p_ext.add_argument("-p", "--participant", help="Participant ID")
    p_ext.add_argument("-t", "--task", type=int, help="Task number (1-25)")
    p_ext.add_argument("-c", "--calibration", type=float, help="Pixels per mm calibration ratio")
    p_ext.set_defaults(func=cmd_extract)

    # Subcommand: validate
    p_val = subparsers.add_parser("validate", help="Run data-quality validation on raw points CSV")
    p_val.add_argument("-i", "--input", required=True, help="Path to raw points CSV")
    p_val.add_argument("--width", type=float, help="Canvas width in CSS px")
    p_val.add_argument("--height", type=float, help="Canvas height in CSS px")
    p_val.set_defaults(func=cmd_validate)

    # Subcommand: predict
    p_pred = subparsers.add_parser("predict", help="Run Extra Trees model on 450-feature CSV")
    p_pred.add_argument("-i", "--input", required=True, help="Path to 450-feature CSV")
    p_pred.add_argument("--threshold", type=float, default=0.50, help="Classification threshold")
    p_pred.add_argument("--touchscreen", action="store_true", help="Flag input as finger-touchscreen data")
    p_pred.add_argument("--bypass-touchscreen-guard", action="store_true", help="Explicitly bypass safety domain guard for research study")
    p_pred.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
