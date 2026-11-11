"""Unit tests for the DARWIN Research Analyzer package.
"""

import numpy as np
import pandas as pd
import pytest
from darwin_analyzer.feature_extractor import (
    CANONICAL_FEATURES,
    HandwritingFeatureExtractor,
    compute_local_polynomial_derivatives,
)
from darwin_analyzer.quality_validator import DataQualityValidator, QualityStatus
from darwin_analyzer.model_runner import DarwinModelRunner


def create_synthetic_raw_stroke(
    n_points: int = 50,
    stroke_id: int = 1,
    start_t: float = 100.0,
    dt_ms: float = 16.67,
    pointer_type: str = "touch",
) -> pd.DataFrame:
    """Generates a synthetic circular or wavy stroke for testing."""
    times = [start_t + i * dt_ms for i in range(n_points)]
    angles = np.linspace(0, 2 * np.pi, n_points)
    xs = 200.0 + 80.0 * np.cos(angles)
    ys = 200.0 + 80.0 * np.sin(angles)

    return pd.DataFrame({
        "participant_id": ["P_TEST"] * n_points,
        "task_number": [1] * n_points,
        "stroke_index": [stroke_id] * n_points,
        "point_index": list(range(1, n_points + 1)),
        "pointer_type": [pointer_type] * n_points,
        "x_px": xs,
        "y_px": ys,
        "timestamp_elapsed_ms": times,
    })


def test_polynomial_derivatives():
    # Test on a known quadratic trajectory x(t) = 5 * t^2, velocity = 10 * t, acc = 10
    t = np.linspace(0, 1, 21)
    x = 5.0 * (t**2)
    y = np.zeros_like(t)

    speed, acc, jerk = compute_local_polynomial_derivatives(t, x, y, degree=2, window_size=9)
    # Check middle points where window is full
    mid_idx = 10
    assert abs(speed[mid_idx] - 10.0 * t[mid_idx]) < 0.2
    assert abs(acc[mid_idx] - 10.0) < 0.5


def test_feature_extractor_synthetic():
    s1 = create_synthetic_raw_stroke(n_points=30, stroke_id=1, start_t=100.0)
    s2 = create_synthetic_raw_stroke(n_points=30, stroke_id=2, start_t=700.0)
    df = pd.concat([s1, s2], ignore_index=True)

    extractor = HandwritingFeatureExtractor()
    result = extractor.extract_task_features(df, participant_id="P_TEST", task_number=1)

    assert result["participant_id"] == "P_TEST"
    assert result["n_strokes"] == 2
    assert result["n_points"] == 60

    canon = result["canonical_features"]
    assert "total_time" in canon
    assert "paper_time" in canon
    assert "air_time" in canon
    assert canon["num_of_pendown"] == 2
    assert canon["max_x_extension"] > 0
    assert canon["max_y_extension"] > 0
    assert canon["mean_speed_on_paper"] > 0
    assert np.isnan(canon["gmrt_in_air"])  # Explicitly excluded on touchscreen


def test_quality_validator_valid():
    df = create_synthetic_raw_stroke(n_points=40, stroke_id=1)
    validator = DataQualityValidator(enforce_finger_only=True)
    report = validator.validate_raw_data(df, canvas_width=800, canvas_height=600)

    assert report["overall_status"] == QualityStatus.VALID.value
    assert report["score"] == 1.0
    assert len(report["flags"]) == 0


def test_quality_validator_flags_mouse_in_finger_mode():
    df = create_synthetic_raw_stroke(n_points=40, stroke_id=1, pointer_type="mouse")
    validator = DataQualityValidator(enforce_finger_only=True)
    report = validator.validate_raw_data(df)

    assert report["overall_status"] == QualityStatus.INVALID.value
    assert any(f["code"] == "NON_TOUCH_POINTER_DETECTED" for f in report["flags"])


def test_quality_validator_flags_non_monotonic_timestamps():
    df = create_synthetic_raw_stroke(n_points=30, stroke_id=1)
    # Inject backward timestamp
    df.loc[15, "timestamp_elapsed_ms"] = df.loc[14, "timestamp_elapsed_ms"] - 50.0

    validator = DataQualityValidator()
    report = validator.validate_raw_data(df)

    assert report["overall_status"] == QualityStatus.INVALID.value
    assert any(f["code"] == "NON_MONOTONIC_TIMESTAMPS" for f in report["flags"])


def test_model_runner_sample_rows():
    runner = DarwinModelRunner()
    # Test with sample row from app/sample_rows.csv
    sample_df = pd.read_csv("app/sample_rows.csv")
    row_match = sample_df.iloc[[0]]

    pred = runner.predict_sample(row_match, is_touchscreen_data=False)
    assert pred["success"] is True
    assert "probability_ad" in pred
    assert 0.0 <= pred["probability_ad"] <= 1.0
    assert "predicted_class" in pred
    assert pred["predicted_class"] in [0, 1]

    # Verify explainability
    expl = pred.get("explanation", {})
    if expl.get("shap_available"):
        assert len(expl["top_tasks"]) > 0
        assert len(expl["top_10_features"]) > 0


def test_model_runner_blocks_unauthorized_touchscreen():
    runner = DarwinModelRunner()
    sample_df = pd.read_csv("app/sample_rows.csv")
    row_match = sample_df.iloc[[0]]

    # Without bypass, touchscreen data should be blocked
    pred = runner.predict_sample(row_match, is_touchscreen_data=True, allow_touchscreen_bypass=False)
    assert pred["success"] is False
    assert pred.get("domain_mismatch_blocked") is True
