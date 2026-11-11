"""Handwriting Feature Extraction Engine for DARWIN.

Extracts the 18 canonical DARWIN handwriting features alongside touchscreen-adapted
kinematic, temporal, and spatial metrics using local polynomial differentiation.
DISCLAIMER: Academic research prototype. NOT a certified medical diagnostic device.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# 18 Canonical DARWIN Feature Names
CANONICAL_FEATURES = [
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
    "total_time",
]

BASE_FEATURES = CANONICAL_FEATURES

# Classification of touchscreen measurability
FEATURE_TOUCHSCREEN_STATUS = {
    "total_time": "DIRECTLY_MEASURABLE",
    "paper_time": "DIRECTLY_MEASURABLE",
    "air_time": "ADAPTED_INTERSTROKE_LATENCY",
    "num_of_pendown": "DIRECTLY_MEASURABLE",
    "mean_speed_on_paper": "DIRECTLY_MEASURABLE",
    "mean_acc_on_paper": "DIRECTLY_MEASURABLE_LOCAL_POLYNOMIAL",
    "mean_jerk_on_paper": "DIRECTLY_MEASURABLE_LOCAL_POLYNOMIAL",
    "max_x_extension": "DIRECTLY_MEASURABLE",
    "max_y_extension": "DIRECTLY_MEASURABLE",
    "disp_index": "DIRECTLY_MEASURABLE",
    "gmrt_on_paper": "ADAPTED_KINEMATIC_ROUGHNESS",
    "mean_gmrt": "ADAPTED_KINEMATIC_ROUGHNESS",
    "mean_speed_in_air": "NOT_MEASURABLE_TOUCHSCREEN",
    "mean_acc_in_air": "NOT_MEASURABLE_TOUCHSCREEN",
    "mean_jerk_in_air": "NOT_MEASURABLE_TOUCHSCREEN",
    "gmrt_in_air": "NOT_MEASURABLE_TOUCHSCREEN",
    "pressure_mean": "EXCLUDED_UNCALIBRATED",
    "pressure_var": "EXCLUDED_UNCALIBRATED",
}


def compute_local_polynomial_derivatives(
    timestamps_s: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    degree: int = 3,
    window_size: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes speed, acceleration, and jerk using a local least-squares polynomial fit.

    Fits x(t) and y(t) with a polynomial of given degree over a moving window of points.
    Returns:
        speed: magnitude of 1st derivative (velocity)
        acceleration: magnitude of 2nd derivative
        jerk: magnitude of 3rd derivative
    """
    n_pts = len(timestamps_s)
    speed = np.full(n_pts, np.nan, dtype=np.float64)
    acc = np.full(n_pts, np.nan, dtype=np.float64)
    jerk = np.full(n_pts, np.nan, dtype=np.float64)

    if n_pts < degree + 1:
        return speed, acc, jerk

    half_win = window_size // 2

    for i in range(n_pts):
        start_idx = max(0, min(i - half_win, n_pts - window_size))
        end_idx = min(n_pts, start_idx + window_size)

        win_t = timestamps_s[start_idx:end_idx]
        win_x = x_coords[start_idx:end_idx]
        win_y = y_coords[start_idx:end_idx]

        if len(win_t) < degree + 1:
            continue

        # Center and scale time window for numerical stability
        t0 = timestamps_s[i]
        dt = win_t - t0
        scale = np.max(np.abs(dt))
        if scale <= 1e-9:
            continue
        z = dt / scale

        # Vandermonde matrix for local polynomial
        try:
            poly_x = np.polyfit(z, win_x, deg=degree)
            poly_y = np.polyfit(z, win_y, deg=degree)

            # polyfit returns coefficients in descending order: [c3, c2, c1, c0]
            # derivatives evaluated at z = 0 (t = t0):
            # x'(0) = c1 / scale
            # x''(0) = 2 * c2 / (scale^2)
            # x'''(0) = 6 * c3 / (scale^3)
            c1_x = poly_x[-2] / scale
            c1_y = poly_y[-2] / scale
            speed[i] = float(np.hypot(c1_x, c1_y))

            if degree >= 2:
                c2_x = 2.0 * poly_x[-3] / (scale**2)
                c2_y = 2.0 * poly_y[-3] / (scale**2)
                acc[i] = float(np.hypot(c2_x, c2_y))

            if degree >= 3:
                c3_x = 6.0 * poly_x[-4] / (scale**3)
                c3_y = 6.0 * poly_y[-4] / (scale**3)
                jerk[i] = float(np.hypot(c3_x, c3_y))
        except (np.linalg.LinAlgError, ValueError):
            continue

    return speed, acc, jerk


class HandwritingFeatureExtractor:
    """Computes DARWIN and adapted touchscreen features for raw touch/pen recordings."""

    def __init__(
        self,
        pixel_to_mm_ratio: Optional[float] = None,
        derivative_window: int = 9,
        derivative_degree: int = 3,
    ):
        """Args:

        pixel_to_mm_ratio: Optional calibration factor (pixels per mm). If None,
        units are CSS pixels. derivative_window: Window size for local
        polynomial derivative fitting. derivative_degree: Polynomial degree
        (default 3 for cubic fit).
        """
        self.pixel_to_mm_ratio = pixel_to_mm_ratio
        self.derivative_window = derivative_window
        self.derivative_degree = derivative_degree

    def extract_task_features(
        self,
        raw_df: pd.DataFrame,
        participant_id: Optional[str] = None,
        task_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Extracts all task-level and stroke-level metrics from a raw points DataFrame.

        Required columns in raw_df:
            - 'stroke_index' (or 'stroke')
            - 'point_index' (or 'point')
            - 'x_px' (or 'x')
            - 'y_px' (or 'y')
            - 'timestamp_elapsed_ms' (or 'timestamp_ms', 't_ms')
        """
        df = self._standardize_dataframe(raw_df)
        if len(df) == 0:
            raise ValueError("Input DataFrame is empty.")

        pid = participant_id or str(df["participant_id"].iloc[0]) if "participant_id" in df else "P_UNKNOWN"
        task_num = int(task_number or df["task_number"].iloc[0]) if "task_number" in df else 1

        # Group by stroke_index and preserve temporal order
        stroke_groups = [group for _, group in df.groupby("stroke_index", sort=True)]
        n_strokes = len(stroke_groups)

        stroke_metrics: List[Dict[str, Any]] = []
        all_speeds: List[float] = []
        all_accs: List[float] = []
        all_jerks: List[float] = []
        all_intervals_ms: List[float] = []
        interstroke_gaps_ms: List[float] = []

        total_paper_time_ms = 0.0
        total_path_length_px = 0.0

        for s_idx, s_df in enumerate(stroke_groups):
            s_df = s_df.sort_values(by="point_index").reset_index(drop=True)
            pts_x = s_df["x_px"].to_numpy(dtype=np.float64)
            pts_y = s_df["y_px"].to_numpy(dtype=np.float64)
            pts_t_ms = s_df["timestamp_elapsed_ms"].to_numpy(dtype=np.float64)
            n_pts = len(s_df)

            stroke_id = int(s_df["stroke_index"].iloc[0])

            # Duration of contact
            s_duration_ms = float(pts_t_ms[-1] - pts_t_ms[0]) if n_pts > 0 else 0.0
            total_paper_time_ms += max(0.0, s_duration_ms)

            # Point-to-point steps
            if n_pts >= 2:
                dx = np.diff(pts_x)
                dy = np.diff(pts_y)
                step_dists = np.hypot(dx, dy)
                s_path_len = float(np.sum(step_dists))
                total_path_length_px += s_path_len

                dt_ms = np.diff(pts_t_ms)
                for dt in dt_ms:
                    if dt > 0:
                        all_intervals_ms.append(float(dt))

                # Step-based speeds
                positive_dt_mask = dt_ms > 0
                if np.any(positive_dt_mask):
                    step_speeds = step_dists[positive_dt_mask] / (dt_ms[positive_dt_mask] / 1000.0)
                    all_speeds.extend(step_speeds.tolist())

                # Local polynomial derivatives (velocity, acceleration, jerk)
                t_sec = pts_t_ms / 1000.0
                p_speed, p_acc, p_jerk = compute_local_polynomial_derivatives(
                    t_sec,
                    pts_x,
                    pts_y,
                    degree=self.derivative_degree,
                    window_size=self.derivative_window,
                )

                valid_acc = p_acc[~np.isnan(p_acc)]
                valid_jerk = p_jerk[~np.isnan(p_jerk)]
                all_accs.extend(valid_acc.tolist())
                all_jerks.extend(valid_jerk.tolist())
            else:
                s_path_len = 0.0
                step_speeds = np.array([])

            # Net displacement (straight-line distance between start and end)
            if n_pts >= 1:
                net_disp = float(np.hypot(pts_x[-1] - pts_x[0], pts_y[-1] - pts_y[0]))
            else:
                net_disp = 0.0

            straightness = float(net_disp / s_path_len) if s_path_len > 1e-6 else 0.0
            straightness = min(1.0, max(0.0, straightness))

            # Bounding box
            bbox_w = float(np.max(pts_x) - np.min(pts_x)) if n_pts > 0 else 0.0
            bbox_h = float(np.max(pts_y) - np.min(pts_y)) if n_pts > 0 else 0.0

            stroke_metrics.append({
                "stroke_index": stroke_id,
                "point_count": n_pts,
                "duration_ms": round(s_duration_ms, 2),
                "path_length_px": round(s_path_len, 4),
                "net_displacement_px": round(net_disp, 4),
                "straightness_ratio": round(straightness, 5),
                "bbox_width_px": round(bbox_w, 4),
                "bbox_height_px": round(bbox_h, 4),
                "mean_step_speed_px_s": round(float(np.mean(step_speeds)), 4) if len(step_speeds) else 0.0,
                "median_step_speed_px_s": round(float(np.median(step_speeds)), 4) if len(step_speeds) else 0.0,
                "single_point_stroke": bool(n_pts == 1),
            })

            # Inter-stroke gap calculation
            if s_idx > 0:
                prev_s = stroke_groups[s_idx - 1]
                gap = float(pts_t_ms[0] - prev_s["timestamp_elapsed_ms"].iloc[-1])
                interstroke_gaps_ms.append(max(0.0, gap))

        # Overall Task Metrics
        all_x = df["x_px"].to_numpy(dtype=np.float64)
        all_y = df["y_px"].to_numpy(dtype=np.float64)
        all_t = df["timestamp_elapsed_ms"].to_numpy(dtype=np.float64)

        overall_total_time_ms = float(all_t[-1] - all_t[0]) if len(all_t) > 0 else 0.0
        total_air_time_ms = float(np.sum(interstroke_gaps_ms)) if interstroke_gaps_ms else 0.0

        max_x_ext = float(np.max(all_x) - np.min(all_x)) if len(all_x) else 0.0
        max_y_ext = float(np.max(all_y) - np.min(all_y)) if len(all_y) else 0.0

        # Displacement index: spatial distribution of handwriting
        # Net stroke displacements divided by total path length
        sum_net_displacements = sum(s["net_displacement_px"] for s in stroke_metrics)
        disp_index = float(sum_net_displacements / total_path_length_px) if total_path_length_px > 1e-6 else 0.0
        disp_index = min(1.0, max(0.0, disp_index))

        # GMRT on paper (Geometric Measure of Relief Texture):
        # Kinematic variation index / speed fluctuation metric normalized by path length
        if len(all_speeds) > 1 and total_path_length_px > 1e-6:
            speed_arr = np.array(all_speeds)
            speed_variation = float(np.sum(np.abs(np.diff(speed_arr))))
            gmrt_paper = float(speed_variation / (total_path_length_px + 1e-6))
        else:
            gmrt_paper = 0.0

        # Aggregate Speed, Acc, Jerk
        mean_speed = float(np.mean(all_speeds)) if all_speeds else 0.0
        mean_acc = float(np.mean(all_accs)) if all_accs else 0.0
        mean_jerk = float(np.mean(all_jerks)) if all_jerks else 0.0

        # Construct the 18 Canonical Features Dictionary
        canonical_dict = {
            "air_time": round(total_air_time_ms, 2),
            "disp_index": round(disp_index, 5),
            "gmrt_in_air": np.nan,  # Touchscreen has no hover hardware; unmeasurable
            "gmrt_on_paper": round(gmrt_paper, 6),
            "max_x_extension": round(max_x_ext, 4),
            "max_y_extension": round(max_y_ext, 4),
            "mean_acc_in_air": np.nan,  # Unmeasurable
            "mean_acc_on_paper": round(mean_acc, 4),
            "mean_gmrt": round(gmrt_paper, 6),
            "mean_jerk_in_air": np.nan,  # Unmeasurable
            "mean_jerk_on_paper": round(mean_jerk, 4),
            "mean_speed_in_air": np.nan,  # Unmeasurable
            "mean_speed_on_paper": round(mean_speed, 4),
            "num_of_pendown": n_strokes,
            "paper_time": round(total_paper_time_ms, 2),
            "pressure_mean": np.nan,  # Uncalibrated browser metadata; excluded
            "pressure_var": np.nan,  # Excluded
            "total_time": round(overall_total_time_ms, 2),
        }

        # Extended touchscreen metrics
        sampling_intervals = np.array(all_intervals_ms) if all_intervals_ms else np.array([])
        median_dt = float(np.median(sampling_intervals)) if len(sampling_intervals) else np.nan
        sampling_hz = float(1000.0 / median_dt) if median_dt and median_dt > 0 else np.nan

        pointer_types = list(set(df["pointer_type"].dropna().astype(str).tolist())) if "pointer_type" in df else ["unknown"]

        task_features: Dict[str, Any] = {
            "participant_id": pid,
            "task_number": task_num,
            "n_points": len(df),
            "n_strokes": n_strokes,
            "input_pointer_types": "|".join(pointer_types),
            "canonical_features": canonical_dict,
            "extended_metrics": {
                "total_path_length_px": round(total_path_length_px, 4),
                "median_step_speed_px_s": round(float(np.median(all_speeds)), 4) if all_speeds else 0.0,
                "p95_step_speed_px_s": round(float(np.percentile(all_speeds, 95)), 4) if all_speeds else 0.0,
                "max_step_speed_px_s": round(float(np.max(all_speeds)), 4) if all_speeds else 0.0,
                "median_sampling_interval_ms": round(median_dt, 3) if not np.isnan(median_dt) else None,
                "estimated_sampling_hz": round(sampling_hz, 2) if not np.isnan(sampling_hz) else None,
                "intervals_over_33ms": int(np.sum(sampling_intervals > 33.0)) if len(sampling_intervals) else 0,
                "intervals_over_50ms": int(np.sum(sampling_intervals > 50.0)) if len(sampling_intervals) else 0,
                "single_point_stroke_count": sum(1 for s in stroke_metrics if s["single_point_stroke"]),
            },
            "stroke_metrics": stroke_metrics,
            "feature_status": FEATURE_TOUCHSCREEN_STATUS,
        }

        # If pixel-to-mm ratio is specified, also calculate calibrated physical millimeter features
        if self.pixel_to_mm_ratio and self.pixel_to_mm_ratio > 0:
            px_to_mm = 1.0 / self.pixel_to_mm_ratio
            task_features["calibrated_physical_mm"] = {
                "pixel_to_mm_ratio": self.pixel_to_mm_ratio,
                "total_path_length_mm": round(total_path_length_px * px_to_mm, 3),
                "max_x_extension_mm": round(max_x_ext * px_to_mm, 3),
                "max_y_extension_mm": round(max_y_ext * px_to_mm, 3),
                "mean_speed_on_paper_mm_s": round(mean_speed * px_to_mm, 3),
                "mean_acc_on_paper_mm_s2": round(mean_acc * px_to_mm, 3),
                "mean_jerk_on_paper_mm_s3": round(mean_jerk * px_to_mm, 3),
            }

        return task_features

    def _standardize_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes column names and types across various CSV formats."""
        df = raw_df.copy()

        # Map common aliases
        col_map = {
            "stroke": "stroke_index",
            "stroke_id": "stroke_index",
            "point": "point_index",
            "point_id": "point_index",
            "x": "x_px",
            "y": "y_px",
            "timestamp_ms": "timestamp_elapsed_ms",
            "t_ms": "timestamp_elapsed_ms",
            "time_ms": "timestamp_elapsed_ms",
            "device": "pointer_type",
            "type": "pointer_type",
            "task": "task_number",
            "task_id": "task_number",
            "participant": "participant_id",
            "subject_id": "participant_id",
        }

        rename_dict = {}
        for c in df.columns:
            c_lower = c.strip().lower()
            if c_lower in col_map and col_map[c_lower] not in df.columns:
                rename_dict[c] = col_map[c_lower]
        if rename_dict:
            df = df.rename(columns=rename_dict)

        required = ["stroke_index", "point_index", "x_px", "y_px", "timestamp_elapsed_ms"]
        for req in required:
            if req not in df.columns:
                raise ValueError(f"Missing required raw point column: '{req}'")
            df[req] = pd.to_numeric(df[req], errors="coerce")

        df = df.dropna(subset=required).sort_values(
            by=["stroke_index", "point_index"]
        ).reset_index(drop=True)
        return df


def extract_features_from_csv(
    csv_path: str,
    participant_id: Optional[str] = None,
    task_number: Optional[int] = None,
    pixel_to_mm_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """Convenience function to extract features directly from a CSV file."""
    df = pd.read_csv(csv_path)
    extractor = HandwritingFeatureExtractor(pixel_to_mm_ratio=pixel_to_mm_ratio)
    return extractor.extract_task_features(
        df, participant_id=participant_id, task_number=task_number
    )
