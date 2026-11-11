"""Research Dataset Generator for DARWIN.

Converts multi-task handwriting sessions into clean, structured research datasets:
- Raw touch-event level (CSV/JSON)
- Stroke level (CSV/JSON)
- Task level (CSV/JSON)
- Participant level (25 tasks x 18 features wide matrix)
- Quality audit and session metadata reports
DISCLAIMER: Academic research prototype. NOT a certified medical diagnostic device.
"""

import json
import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from .feature_extractor import BASE_FEATURES, HandwritingFeatureExtractor
from .quality_validator import DataQualityValidator


class ResearchDatasetGenerator:
    """Aggregates raw recordings from multi-task sessions into research-ready dataset exports."""

    def __init__(self, pixel_to_mm_ratio: Optional[float] = None):
        self.extractor = HandwritingFeatureExtractor(pixel_to_mm_ratio=pixel_to_mm_ratio)
        self.validator = DataQualityValidator()

    def compile_session_datasets(
        self,
        task_data_dict: Dict[int, pd.DataFrame],
        participant_id: str,
        session_id: str,
        session_metadata: Optional[Dict[str, Any]] = None,
        canvas_dims: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Processes a dict of {task_number: raw_dataframe} and produces all 6 research datasets.

        Returns:
            - raw_events_df: Combined point-level DataFrame
            - stroke_metrics_df: Combined stroke-level DataFrame
            - task_features_df: Long-format task features DataFrame
            - participant_wide_matrix_df: 450-column wide DataFrame
            - quality_reports: Dict of quality reports per task
            - session_metadata: Complete session metadata dict
        """
        raw_rows_all: List[pd.DataFrame] = []
        stroke_rows_all: List[Dict[str, Any]] = []
        task_features_list: List[Dict[str, Any]] = []
        quality_reports: Dict[int, Any] = {}

        # Wide matrix row container: 1 row with 450 feature columns
        wide_row: Dict[str, Any] = {
            "participant_id": participant_id,
            "session_id": session_id,
        }

        # Pre-populate all 450 columns with NaN
        for t in range(1, 26):
            for bf in BASE_FEATURES:
                wide_row[f"{bf}{t}"] = np.nan

        # Process each task in order
        for task_num in range(1, 26):
            if task_num in task_data_dict and len(task_data_dict[task_num]) > 0:
                raw_df = task_data_dict[task_num].copy()

                # Add participant, session, and task info if missing
                raw_df["participant_id"] = participant_id
                raw_df["session_id"] = session_id
                raw_df["task_number"] = task_num
                raw_rows_all.append(raw_df)

                # Quality Validation
                c_w = canvas_dims.get("width") if canvas_dims else None
                c_h = canvas_dims.get("height") if canvas_dims else None
                q_report = self.validator.validate_raw_data(raw_df, canvas_width=c_w, canvas_height=c_h)
                quality_reports[task_num] = q_report

                # Feature Extraction
                try:
                    feat_dict = self.extractor.extract_task_features(
                        raw_df, participant_id=participant_id, task_number=task_num
                    )

                    # Extract stroke metrics
                    for sm in feat_dict["stroke_metrics"]:
                        sm_copy = sm.copy()
                        sm_copy["participant_id"] = participant_id
                        sm_copy["session_id"] = session_id
                        sm_copy["task_number"] = task_num
                        stroke_rows_all.append(sm_copy)

                    # Long format task features
                    task_entry = {
                        "participant_id": participant_id,
                        "session_id": session_id,
                        "task_number": task_num,
                        "quality_status": q_report["overall_status"],
                        "quality_score": q_report["score"],
                        "n_points": feat_dict["n_points"],
                        "n_strokes": feat_dict["n_strokes"],
                    }
                    task_entry.update(feat_dict["canonical_features"])
                    task_entry.update(feat_dict["extended_metrics"])
                    task_features_list.append(task_entry)

                    # Update wide matrix row
                    for bf, val in feat_dict["canonical_features"].items():
                        col_key = f"{bf}{task_num}"
                        wide_row[col_key] = val

                except Exception as e:
                    quality_reports[task_num] = {
                        "overall_status": "PROCESSING_ERROR",
                        "error": str(e),
                    }
            else:
                quality_reports[task_num] = {
                    "overall_status": "MISSING_TASK",
                    "summary": f"Task {task_num} was not completed in this session.",
                }

        # Build DataFrames
        combined_raw_df = pd.concat(raw_rows_all, ignore_index=True) if raw_rows_all else pd.DataFrame()
        stroke_df = pd.DataFrame(stroke_rows_all) if stroke_rows_all else pd.DataFrame()
        task_df = pd.DataFrame(task_features_list) if task_features_list else pd.DataFrame()
        wide_df = pd.DataFrame([wide_row])

        # Session summary metadata
        meta = session_metadata or {}
        completed_tasks = [t for t in range(1, 26) if t in task_data_dict and len(task_data_dict[t]) > 0]
        meta.update({
            "participant_id": participant_id,
            "session_id": session_id,
            "n_tasks_completed": len(completed_tasks),
            "completed_tasks_list": completed_tasks,
            "total_points_captured": len(combined_raw_df),
            "total_strokes_captured": len(stroke_df),
            "overall_quality_status": "VALID" if all(
                quality_reports.get(t, {}).get("overall_status") == "VALID" for t in completed_tasks
            ) else "REVIEW_REQUIRED",
            "schema_version": "DARWIN_Protocol_2.0",
        })

        return {
            "raw_events_df": combined_raw_df,
            "stroke_metrics_df": stroke_df,
            "task_features_df": task_df,
            "participant_wide_matrix_df": wide_df,
            "quality_reports": quality_reports,
            "session_metadata": meta,
        }

    def export_all_to_directory(
        self,
        datasets: Dict[str, Any],
        output_dir: str,
        prefix: str = "darwin_session",
    ) -> Dict[str, str]:
        """Saves all CSV and JSON datasets to an output folder. Returns dict of saved file paths."""
        os.makedirs(output_dir, exist_ok=True)
        file_paths = {}

        # 1. Raw points CSV & JSON
        raw_csv = os.path.join(output_dir, f"{prefix}_raw_points.csv")
        datasets["raw_events_df"].to_csv(raw_csv, index=False)
        file_paths["raw_csv"] = raw_csv

        # 2. Stroke metrics CSV
        stroke_csv = os.path.join(output_dir, f"{prefix}_stroke_metrics.csv")
        datasets["stroke_metrics_df"].to_csv(stroke_csv, index=False)
        file_paths["stroke_csv"] = stroke_csv

        # 3. Task features CSV
        task_csv = os.path.join(output_dir, f"{prefix}_task_features.csv")
        datasets["task_features_df"].to_csv(task_csv, index=False)
        file_paths["task_csv"] = task_csv

        # 4. Participant-level wide 450-feature CSV
        wide_csv = os.path.join(output_dir, f"{prefix}_darwin_wide_450_matrix.csv")
        datasets["participant_wide_matrix_df"].to_csv(wide_csv, index=False)
        file_paths["wide_csv"] = wide_csv

        # 5. Quality report JSON
        q_json = os.path.join(output_dir, f"{prefix}_quality_report.json")
        with open(q_json, "w", encoding="utf-8") as f:
            json.dump(datasets["quality_reports"], f, indent=2, default=str)
        file_paths["quality_json"] = q_json

        # 6. Session metadata JSON
        meta_json = os.path.join(output_dir, f"{prefix}_session_metadata.json")
        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(datasets["session_metadata"], f, indent=2, default=str)
        file_paths["metadata_json"] = meta_json

        return file_paths
