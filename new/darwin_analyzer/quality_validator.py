"""Data Quality and Acquisition Validation Engine for DARWIN.

Implements automated multi-tier integrity and sensor-quality checks on raw handwriting data.
DISCLAIMER: Academic research prototype. NOT a certified medical diagnostic device.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class QualityStatus(str, Enum):
    VALID = "VALID"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    INVALID = "INVALID"


class DataQualityValidator:
    """Evaluates raw touch event recording quality and flags protocol violations."""

    def __init__(
        self,
        enforce_finger_only: bool = True,
        min_task_points: int = 10,
        gap_warning_threshold_ms: float = 33.3,
        gap_severe_threshold_ms: float = 50.0,
    ):
        self.enforce_finger_only = enforce_finger_only
        self.min_task_points = min_task_points
        self.gap_warning_threshold_ms = gap_warning_threshold_ms
        self.gap_severe_threshold_ms = gap_severe_threshold_ms

    def validate_raw_data(
        self,
        df: pd.DataFrame,
        canvas_width: Optional[float] = None,
        canvas_height: Optional[float] = None,
        pixel_to_mm_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Runs the complete suite of quality checks on a raw points DataFrame.

        Returns a dictionary containing:
            - overall_status: QualityStatus ('VALID', 'REVIEW_REQUIRED', 'INVALID')
            - score: float between 0.0 and 1.0
            - flags: List of specific issues found with severity
            - checks: Dict of individual check results
            - summary: Human-readable diagnostic explanation
        """
        flags: List[Dict[str, Any]] = []
        checks: Dict[str, Any] = {}

        if df is None or len(df) == 0:
            return {
                "overall_status": QualityStatus.INVALID.value,
                "score": 0.0,
                "flags": [{
                    "code": "EMPTY_RECORDING",
                    "severity": "CRITICAL",
                    "message": "Raw recording contains 0 points.",
                }],
                "checks": {"empty_check": False},
                "summary": "Invalid: Recording is completely empty.",
            }

        n_points = len(df)

        # Standardize columns
        col_map = {
            "stroke": "stroke_index",
            "stroke_id": "stroke_index",
            "point": "point_index",
            "point_id": "point_index",
            "x": "x_px",
            "y": "y_px",
            "timestamp_ms": "timestamp_elapsed_ms",
            "t_ms": "timestamp_elapsed_ms",
            "device": "pointer_type",
            "type": "pointer_type",
        }
        df_clean = df.rename(columns={c: col_map[c.lower()] for c in df.columns if c.lower() in col_map}).copy()

        # 1. Missing / Non-numeric coordinates check
        has_coords = "x_px" in df_clean and "y_px" in df_clean
        if not has_coords:
            flags.append({
                "code": "MISSING_COORDINATE_COLUMNS",
                "severity": "CRITICAL",
                "message": "Required coordinate columns (x_px, y_px) are missing.",
            })
            nan_coords = n_points
        else:
            nan_coords = int(df_clean[["x_px", "y_px"]].isna().sum().sum())
            if nan_coords > 0:
                flags.append({
                    "code": "NAN_COORDINATES",
                    "severity": "CRITICAL",
                    "message": f"{nan_coords} coordinates contain NaN or non-numeric values.",
                })
        checks["nan_coordinates_count"] = nan_coords

        # 2. Missing / NaN timestamps check
        if "timestamp_elapsed_ms" not in df_clean:
            flags.append({
                "code": "MISSING_TIMESTAMP_COLUMN",
                "severity": "CRITICAL",
                "message": "Timestamp column 'timestamp_elapsed_ms' is missing.",
            })
            nan_timestamps = n_points
        else:
            nan_timestamps = int(df_clean["timestamp_elapsed_ms"].isna().sum())
            if nan_timestamps > 0:
                flags.append({
                    "code": "NAN_TIMESTAMPS",
                    "severity": "CRITICAL",
                    "message": f"{nan_timestamps} timestamps contain NaN values.",
                })
        checks["nan_timestamps_count"] = nan_timestamps

        # 3. Duplicate point keys (stroke_index + point_index)
        if "stroke_index" in df_clean and "point_index" in df_clean:
            keys = df_clean["stroke_index"].astype(str) + "_" + df_clean["point_index"].astype(str)
            duplicate_keys = int(keys.duplicated().sum())
            if duplicate_keys > 0:
                flags.append({
                    "code": "DUPLICATE_POINT_KEYS",
                    "severity": "WARNING",
                    "message": f"{duplicate_keys} duplicate (stroke_index, point_index) keys detected.",
                })
        else:
            duplicate_keys = 0
        checks["duplicate_point_keys"] = duplicate_keys

        # 4. Monotonic timestamps & within-stroke order
        non_monotonic_count = 0
        duplicate_timestamps_count = 0
        long_gaps_33ms = 0
        long_gaps_50ms = 0
        single_point_strokes = 0
        zero_duration_strokes = 0
        stroke_count = 0

        if "stroke_index" in df_clean and "timestamp_elapsed_ms" in df_clean:
            grouped = df_clean.groupby("stroke_index", sort=True)
            stroke_count = len(grouped)

            for _, s_df in grouped:
                pts_t = pd.to_numeric(s_df["timestamp_elapsed_ms"], errors="coerce").dropna().to_numpy()
                if len(pts_t) == 1:
                    single_point_strokes += 1
                if len(pts_t) >= 1 and (pts_t[-1] - pts_t[0]) <= 0.0:
                    zero_duration_strokes += 1

                if len(pts_t) >= 2:
                    dt = np.diff(pts_t)
                    non_monotonic = np.sum(dt < 0)
                    dup_t = np.sum(dt == 0)
                    non_monotonic_count += int(non_monotonic)
                    duplicate_timestamps_count += int(dup_t)

                    long_gaps_33ms += int(np.sum(dt > self.gap_warning_threshold_ms))
                    long_gaps_50ms += int(np.sum(dt > self.gap_severe_threshold_ms))

        if non_monotonic_count > 0:
            flags.append({
                "code": "NON_MONOTONIC_TIMESTAMPS",
                "severity": "CRITICAL",
                "message": f"{non_monotonic_count} intervals have backward/non-monotonic timestamps.",
            })
        if duplicate_timestamps_count > 0:
            flags.append({
                "code": "DUPLICATE_TIMESTAMPS",
                "severity": "WARNING",
                "message": f"{duplicate_timestamps_count} consecutive points have identical timestamps.",
            })
        if long_gaps_50ms > 0:
            flags.append({
                "code": "SEVERE_SAMPLING_GAP",
                "severity": "WARNING",
                "message": f"{long_gaps_50ms} within-stroke intervals exceed {self.gap_severe_threshold_ms}ms (sensor dropout risk).",
            })
        if single_point_strokes > 0:
            flags.append({
                "code": "SINGLE_POINT_STROKES",
                "severity": "INFO",
                "message": f"{single_point_strokes} isolated single-point taps recorded (no velocity measurable).",
            })

        checks["non_monotonic_timestamps"] = non_monotonic_count
        checks["duplicate_timestamps"] = duplicate_timestamps_count
        checks["long_gaps_33ms"] = long_gaps_33ms
        checks["long_gaps_50ms"] = long_gaps_50ms
        checks["single_point_strokes"] = single_point_strokes
        checks["zero_duration_strokes"] = zero_duration_strokes

        # 5. Out-of-bounds coordinates
        out_of_bounds_count = 0
        if has_coords and canvas_width and canvas_height:
            xs = pd.to_numeric(df_clean["x_px"], errors="coerce")
            ys = pd.to_numeric(df_clean["y_px"], errors="coerce")
            oob_mask = (xs < 0) | (xs > canvas_width) | (ys < 0) | (ys > canvas_height)
            out_of_bounds_count = int(oob_mask.sum())
            if out_of_bounds_count > 0:
                flags.append({
                    "code": "OUT_OF_BOUNDS_COORDINATES",
                    "severity": "WARNING",
                    "message": f"{out_of_bounds_count} points fall outside declared canvas dimensions ({canvas_width}x{canvas_height} px).",
                })
        checks["out_of_bounds_count"] = out_of_bounds_count

        # 6. Pointer type & mouse rejection
        pointer_types = []
        non_touch_count = 0
        if "pointer_type" in df_clean:
            pointer_types = [str(x) for x in df_clean["pointer_type"].dropna().unique().tolist()]
            if self.enforce_finger_only:
                non_touch_mask = df_clean["pointer_type"].astype(str).str.lower() != "touch"
                non_touch_count = int(non_touch_mask.sum())
                if non_touch_count > 0:
                    flags.append({
                        "code": "NON_TOUCH_POINTER_DETECTED",
                        "severity": "CRITICAL" if "mouse" in [t.lower() for t in pointer_types] else "WARNING",
                        "message": f"{non_touch_count} samples recorded from non-touch device (types: {pointer_types}) in finger-only session.",
                    })
        checks["pointer_types"] = pointer_types
        checks["non_touch_sample_count"] = non_touch_count

        # 7. Total points sufficiency
        if n_points < self.min_task_points:
            flags.append({
                "code": "INSUFFICIENT_POINTS",
                "severity": "CRITICAL",
                "message": f"Task has only {n_points} points (minimum required: {self.min_task_points}). Incomplete writing.",
            })
        checks["total_points"] = n_points

        # 8. Calibration validity
        if pixel_to_mm_ratio is not None:
            if pixel_to_mm_ratio <= 0 or pixel_to_mm_ratio > 100.0:
                flags.append({
                    "code": "INVALID_CALIBRATION_RATIO",
                    "severity": "WARNING",
                    "message": f"Pixel-to-mm ratio {pixel_to_mm_ratio} is outside plausible physical range (1-50 px/mm).",
                })

        # Calculate Overall Status & Quality Score
        critical_flags = [f for f in flags if f["severity"] == "CRITICAL"]
        warning_flags = [f for f in flags if f["severity"] == "WARNING"]

        if len(critical_flags) > 0:
            status = QualityStatus.INVALID
            score = max(0.0, 1.0 - 0.3 * len(critical_flags) - 0.1 * len(warning_flags))
        elif len(warning_flags) > 0:
            status = QualityStatus.REVIEW_REQUIRED
            score = max(0.5, 1.0 - 0.1 * len(warning_flags))
        else:
            status = QualityStatus.VALID
            score = 1.0

        # Construct concise summary message
        if status == QualityStatus.VALID:
            summary = f"Quality Valid (Score: {score*100:.0f}%): All data integrity checks passed across {n_points} points and {stroke_count} strokes."
        elif status == QualityStatus.REVIEW_REQUIRED:
            summary = f"Review Required (Score: {score*100:.0f}%): {len(warning_flags)} non-critical warnings detected ({', '.join([f['code'] for f in warning_flags[:3]])})."
        else:
            summary = f"Invalid (Score: {score*100:.0f}%): {len(critical_flags)} critical protocol violations ({', '.join([f['code'] for f in critical_flags[:3]])})."

        return {
            "overall_status": status.value,
            "score": round(score, 3),
            "flags": flags,
            "checks": checks,
            "summary": summary,
            "n_points": n_points,
            "n_strokes": stroke_count,
        }
