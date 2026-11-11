"""Streamlit Web Research Demonstration for DARWIN Early Alzheimer's Detection Prototype.

DISCLAIMER: This is an academic research prototype, NOT a clinically validated diagnostic tool.
Uploaded data is strictly processed in-memory and NEVER stored or logged.
"""

import json
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from darwin_analyzer.feature_extractor import HandwritingFeatureExtractor, CANONICAL_FEATURES, FEATURE_TOUCHSCREEN_STATUS
from darwin_analyzer.quality_validator import DataQualityValidator
from src.data import BASE_FEATURES, FEATURE_GROUPS, get_base_feature_name, get_feature_group

DISCLAIMER_TEXT = "RESEARCH PROTOTYPE. NOT A CLINICALLY VALIDATED DIAGNOSTIC TOOL."

st.set_page_config(
    page_title="DARWIN Research & Analysis Suite v2.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 1. Persistent Disclaimer Banner at top of page
st.error(
    f"⚠️ **{DISCLAIMER_TEXT}**\n\n"
    "This system is an academic research prototype developed strictly for algorithmic study. "
    "It is **NOT** certified as medical device software and **MUST NOT** be used for clinical diagnosis, patient triage, or medical decision-making."
)


@st.cache_resource
def load_deployment_model():
    model_path = os.path.join(PROJECT_ROOT, "results", "final_model_extratrees.joblib")
    meta_path = os.path.join(PROJECT_ROOT, "results", "final_model_metadata.json")

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        st.error(f"Model files not found at {model_path}. Please run the training pipeline first.")
        st.stop()

    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Initialize TreeExplainer for real-time individual sample explanations
    explainer = shap.TreeExplainer(model)
    return model, metadata, explainer


model, metadata, explainer = load_deployment_model()
expected_features = metadata["feature_names"]

# 2. Sidebar Navigation
st.sidebar.title("🧠 DARWIN Research Suite")
st.sidebar.caption("v2.0 • Online Handwriting & Cognitive Dynamics")
st.sidebar.divider()

active_track = st.sidebar.radio(
    "Select Research Track:",
    (
        "Track A: 450-Feature Model Inference & SHAP",
        "Track B: Raw Touchscreen Feature Extractor & Quality Audit",
        "Track C: 25-Task Clinical Battery Reference",
    ),
)

# =========================================================================
# TRACK A: 450-FEATURE MODEL INFERENCE & SHAP
# =========================================================================
if active_track == "Track A: 450-Feature Model Inference & SHAP":
    st.sidebar.subheader("Model Configuration")
    st.sidebar.info(
        "**Deployed Model**: Extra Trees Classifier (500 estimators)\n\n"
        "**Selection Rationale**: Selected for slightly higher nested CV mean accuracy (89.46%) "
        "and lower variance (4.30%) vs Random Forest (88.31% ± 4.67%), **NOT** because the difference is statistically significant "
        "(p = 0.4050 under the Nadeau-Bengio corrected resampled t-test).\n\n"
        "**Training Cohort**: All 174 DARWIN participants (89 AD, 85 Healthy)."
    )

    st.sidebar.subheader("Input Selection")
    input_mode = st.sidebar.radio(
        "Select Input Method:",
        ("Use Anonymous Demo Participant", "Upload Custom 450-Feature CSV"),
    )

    input_df = None
    selected_case_name = ""

    if input_mode == "Use Anonymous Demo Participant":
        sample_path = os.path.join(PROJECT_ROOT, "app", "sample_rows.csv")
        if os.path.exists(sample_path):
            sample_df = pd.read_csv(sample_path)
            case_options = sample_df["sample_name"].tolist()
            selected_case_name = st.sidebar.selectbox("Choose Sample Case:", case_options)
            row_match = sample_df[sample_df["sample_name"] == selected_case_name].iloc[0]
            input_df = pd.DataFrame([row_match[expected_features].to_dict()])
        else:
            st.sidebar.error("Sample rows file not found at app/sample_rows.csv.")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV (must contain 450 feature columns):", type=["csv"]
        )
        st.sidebar.caption("🔒 Privacy Guarantee: Data is processed in-memory only and never written to disk or logged.")

        if uploaded_file is not None:
            try:
                raw_upload = pd.read_csv(uploaded_file)
                drop_candidates = ["id", "ID", "user_id", "participant_id", "class", "target", "label", "diagnosis"]
                clean_upload = raw_upload.drop(columns=[c for c in drop_candidates if c in raw_upload.columns])

                missing_cols = [c for c in expected_features if c not in clean_upload.columns]
                if len(missing_cols) > 0:
                    st.error(
                        f"❌ Upload validation failed: Missing {len(missing_cols)} expected feature columns. "
                        f"Example missing: {missing_cols[:5]}"
                    )
                    st.stop()

                if clean_upload[expected_features].isna().sum().sum() > 0:
                    st.error("❌ Upload validation failed: Input data contains NaN / missing values.")
                    st.stop()

                input_df = clean_upload[expected_features].iloc[[0]].copy()
                selected_case_name = f"Uploaded File ({uploaded_file.name}, Row 1)"
            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}")
                st.stop()

    st.title("Original DARWIN 450-Feature Machine Learning Inference")
    st.caption(
        "Evaluates fine-grained kinematic, temporal, pressure, and spatial handwriting indicators across all 25 standardized tasks."
    )

    if input_df is None:
        st.info("👈 Please select a demo participant from the sidebar or upload a CSV file to generate predictions.")
        st.stop()

    # Perform Model Inference
    pred_class = int(model.predict(input_df[expected_features])[0])
    pred_proba = float(model.predict_proba(input_df[expected_features])[0, 1])
    X_input = input_df[expected_features].to_numpy().astype(np.float32)

    # SHAP explanation
    shap_vals = explainer.shap_values(X_input)
    if isinstance(shap_vals, list):
        shap_sample = np.array(shap_vals[1])[0]
    elif len(shap_vals.shape) == 3:
        shap_sample = shap_vals[0, :, 1]
    else:
        shap_sample = shap_vals[0]

    st.subheader(f"Inference Results: {selected_case_name}")
    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])

    with col1:
        if pred_class == 1:
            st.markdown(
                """
                <div style="background-color: #ffebee; border: 2px solid #e53935; border-radius: 8px; padding: 16px; text-align: center;">
                    <h3 style="color: #c62828; margin: 0;">Potential Early AD</h3>
                    <p style="margin: 4px 0 0 0; color: #555;">Class 1 (Impaired Profile)</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background-color: #e8f5e9; border: 2px solid #43a047; border-radius: 8px; padding: 16px; text-align: center;">
                    <h3 style="color: #2e7d32; margin: 0;">Healthy Control</h3>
                    <p style="margin: 4px 0 0 0; color: #555;">Class 0 (Normal Profile)</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:
        st.metric(
            label="Predicted AD Probability Score",
            value=f"{pred_proba * 100:.1f}%",
            delta=f"{(pred_proba - 0.5) * 100:+.1f} pp vs 50% Threshold",
        )

    with col3:
        st.markdown("**Probability Gauge:**")
        st.progress(pred_proba)
        if pred_proba >= 0.70:
            st.caption("🔴 Elevated probability of psychomotor impairment pattern.")
        elif pred_proba <= 0.30:
            st.caption("🟢 Typical healthy motor control profile.")
        else:
            st.caption("🟡 Borderline range (0.30 - 0.70): elevated uncertainty.")

    st.divider()

    # Task Contribution Breakdown
    st.subheader("Task-Level Contribution Analysis")
    st.markdown(
        "Aggregates the individual feature contributions across the 25 handwriting tasks for this participant. "
        "*Positive values shift the prediction toward AD; negative values shift toward Healthy.*"
    )

    task_contributions = {t: 0.0 for t in range(1, 26)}
    group_contributions = {g: 0.0 for g in FEATURE_GROUPS}

    for f_i, col in enumerate(expected_features):
        c_val = float(shap_sample[f_i])
        _, t_num = get_base_feature_name(col)
        task_contributions[t_num] += c_val
        grp = get_feature_group(col)
        group_contributions[grp] += c_val

    task_chart_df = pd.DataFrame(
        [
            {"Task": f"Task {t:02d}", "Task_Num": t, "SHAP_Contribution": task_contributions[t]}
            for t in range(1, 26)
        ]
    ).sort_values(by="SHAP_Contribution", key=abs, ascending=False)

    col_task_left, col_task_right = st.columns([1.5, 1])

    with col_task_left:
        st.markdown("##### Top 8 Contributing Tasks for This Participant")
        top8 = task_chart_df.head(8).reset_index(drop=True)
        st.bar_chart(data=top8, x="Task", y="SHAP_Contribution", color="#4C72B0")
        st.caption("Note: Tasks 17 and 23 rank consistently among the top contributors across the cohort.")

    with col_task_right:
        st.markdown("##### Task Description Reference")
        st.markdown(
            """
            - **Task 17**: Complex geometric loop tracing (fine motor coordination).
            - **Task 23**: Complex sentence writing (cognitive-linguistic load).
            - **Task 07**: High-speed alternating pen-up/pen-down movements.
            - **Task 19**: Continuous cursive stroke chaining.
            - **Task 09**: Spatially constrained square tracing.
            """
        )

    st.divider()

    # Feature Group Breakdown
    st.subheader("Feature Group Contribution")
    group_df = pd.DataFrame(
        [
            {
                "Feature Group": g.replace("_", " ").title(),
                "Net SHAP Contribution": group_contributions[g],
                "Absolute Contribution": abs(group_contributions[g]),
            }
            for g in FEATURE_GROUPS
        ]
    ).sort_values(by="Absolute Contribution", ascending=False)

    col_g1, col_g2 = st.columns([1, 1.5])
    with col_g1:
        st.table(group_df[["Feature Group", "Net SHAP Contribution"]])

    with col_g2:
        st.markdown(
            """
            ##### Clinical Handwriting Modalities:
            - **Timing (air_time, paper_time, total_time)**: Latency in pen movement, hesitations in the air (37.1% cohort importance).
            - **GMRT & Kinematics (mean_gmrt, acceleration, speed, jerk)**: Tremors, stroke irregularities, and motor dysfluency (36.3% importance).
            - **Spatial (max_x/y extensions, disp_index)**: Micrographia and geometric boundary distortions (13.3% importance).
            - **Pressure (mean, variance)**: Pen force consistency and grip stability (13.3% importance).
            """
        )


# =========================================================================
# TRACK B: RAW TOUCHSCREEN FEATURE EXTRACTOR & QUALITY AUDIT
# =========================================================================
elif active_track == "Track B: Raw Touchscreen Feature Extractor & Quality Audit":
    st.title("Raw Touchscreen Feature Extractor & Data-Quality Audit")
    st.caption("Upload raw point streams (x, y, timestamp, stroke_index) captured from touchscreen devices to compute kinematic features and verify data integrity.")

    st.warning(
        "🛡 **Methodological Separation**: Finger touchscreen data lacks physical stylus pressure and continuous hover trajectory. "
        "Touchscreen features use operational adaptations (e.g. local cubic least-squares derivatives and inter-stroke latency) and are NOT directly compatible with the Wacom-trained 450-feature model without cross-device calibration."
    )

    uploaded_raw = st.file_uploader("Upload Raw Point Stream CSV:", type=["csv"])

    if uploaded_raw is not None:
        try:
            raw_df = pd.read_csv(uploaded_raw)
            st.success(f"Loaded raw recording: {len(raw_df)} points.")

            # Run Quality Validator
            validator = DataQualityValidator()
            q_report = validator.validate_raw_data(raw_df)

            col_q1, col_q2, col_q3 = st.columns(3)
            with col_q1:
                status_color = "green" if q_report["overall_status"] == "VALID" else ("orange" if q_report["overall_status"] == "REVIEW_REQUIRED" else "red")
                st.markdown(f"### Quality Status: :{status_color}[{q_report['overall_status']}]")
            with col_q2:
                st.metric("Integrity Score", f"{q_report['score']*100:.0f}%")
            with col_q3:
                st.metric("Detected Flags", len(q_report["flags"]))

            if q_report["flags"]:
                st.markdown("##### Detected Quality Warnings:")
                for f in q_report["flags"]:
                    st.warning(f"**[{f['severity']}] {f['code']}**: {f['message']}")

            # Run Feature Extractor
            extractor = HandwritingFeatureExtractor()
            feat_result = extractor.extract_task_features(raw_df)
            canon = feat_result["canonical_features"]

            st.divider()
            st.subheader("18 Canonical Feature Family Extraction Results")

            feat_rows = []
            for k in CANONICAL_FEATURES:
                val = canon.get(k)
                stat = FEATURE_TOUCHSCREEN_STATUS.get(k, "DIRECTLY_MEASURABLE")
                val_str = f"{val:.4f}" if val is not None and not np.isnan(val) else "Excluded (Uncalibrated / In-Air)"
                feat_rows.append({"Feature Name": k, "Extracted Value": val_str, "Touchscreen Status": stat})

            st.dataframe(pd.DataFrame(feat_rows), use_container_width=True)

            # Stroke Trajectory Visualization
            st.divider()
            st.subheader("Stroke Trajectory Visualization")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_facecolor("#f8fafc")
            for sid, grp in raw_df.groupby("stroke_index"):
                ax.plot(grp["x_px"], grp["y_px"], marker="o", markersize=2, label=f"Stroke {sid}")
            ax.invert_yaxis()  # Canvas coordinates have (0,0) at top-left
            ax.set_title("Visual Trajectory Plot (Canvas Pixels)")
            ax.set_xlabel("X (px)")
            ax.set_ylabel("Y (px)")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing raw CSV: {e}")
    else:
        st.info("💡 Upload a raw CSV file from `darwin_studio/sample_data/` to test feature extraction and quality checking.")


# =========================================================================
# TRACK C: 25-TASK CLINICAL BATTERY REFERENCE
# =========================================================================
elif active_track == "Track C: 25-Task Clinical Battery Reference":
    st.title("Standardized 25-Task DARWIN Handwriting Battery")
    st.caption("Standardized pen-tablet tasks organized into the 3 clinical psychomotor categories established in Cilia et al. (2018, 2022).")

    tasks_info = [
        ("Task 01", "Graphic", "Continuous cursive loops 'lelele...'", "Fine motor rhythm and continuous cursive chaining"),
        ("Task 02", "Graphic", "Continuous sharp zigzags", "Rapid direction reversals and angular velocity changes"),
        ("Task 03", "Graphic", "Archimedean spiral tracing", "Tremor analysis and continuous radius expansion smoothness"),
        ("Task 04", "Graphic", "Triangular peaks and valleys", "Spatial amplitude modulation and rhythm stability"),
        ("Task 05", "Graphic", "Geometric vertex connections", "Target aiming and ballistic motor control"),
        ("Task 06", "Graphic", "Rapid overlapping loops", "Cursive motor chaining and rapid wrist movements"),
        ("Task 07", "Graphic", "Alternating 7 pen-down/pen-up taps", "Rapid discrete tapping and in-air flight latency"),
        ("Task 08", "Graphic", "Sinusoidal smooth waves", "Continuous harmonic oscillation and speed variability"),
        ("Task 09", "Graphic", "Square boundary tracing", "Orthogonal transitions and spatial boundary constraints"),
        ("Task 10", "Graphic", "Figure-8 / infinity continuous loop", "Bimanual motor coordination and circular flow"),
        ("Task 11", "Copying", "Copy uppercase letters (A B C D E)", "Orthographic retrieval and discrete stroke assembly"),
        ("Task 12", "Copying", "Copy lowercase letters (a m r e l)", "Cursive letter construction"),
        ("Task 13", "Copying", "Copy simple high-frequency word ('casa')", "Automated linguistic motor execution"),
        ("Task 14", "Copying", "Copy phonetically rich word ('quaderno')", "Complex grapheme retrieval and motor sequencing"),
        ("Task 15", "Copying", "Copy short sentence ('La casa e bella')", "Inter-word spatial arrangement and sentence pacing"),
        ("Task 16", "Copying", "Copy DARWIN sentence ('Il gatto beve il latte...')", "Standardized syntactic copying benchmark"),
        ("Task 17", "Copying", "Complex geometric loop tracing", "Highest diagnostic weight (39.2 mSHAP): fine coordination"),
        ("Task 18", "Copying", "Copy pseudo-words ('morfina')", "Phonological reading and non-lexical graphemic writing"),
        ("Task 19", "Memory/Dictation", "Continuous cursive chaining ('ononon...')", "Continuous ligatures without letter segmentation"),
        ("Task 20", "Memory/Dictation", "Spontaneous free sentence composition", "Executive cognitive composition and linguistic planning"),
        ("Task 21", "Memory/Dictation", "Clock Drawing Test (CDT - 11:10)", "Visuospatial memory recall and executive planning"),
        ("Task 22", "Memory/Dictation", "Delayed 3-word memory recall writing", "Short-term episodic memory retrieval"),
        ("Task 23", "Memory/Dictation", "Complex sentence writing under load", "Top diagnostic weight (38.8 mSHAP): dual-task cognitive load"),
        ("Task 24", "Memory/Dictation", "Reverse number counting sequence ('20, 19...')", "Working memory manipulation and numerical motor patterns"),
        ("Task 25", "Memory/Dictation", "Personal signature / habitual motor mark", "Overlearned automated motor pattern fluidity"),
    ]

    df_tasks = pd.DataFrame(tasks_info, columns=["Task ID", "Category", "Prompt / Stimulus", "Clinical & Cognitive Rationale"])
    st.dataframe(df_tasks, use_container_width=True, height=650)
