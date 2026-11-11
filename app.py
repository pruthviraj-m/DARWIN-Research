from pathlib import Path
import json
import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# DARWIN — two input pathways
# 1) Standard DARWIN: exact 118-feature input
# 2) Experimental cross-dataset: explicit mapping + reference
#    median imputation. This does NOT establish feature or
#    model compatibility with another handwriting study.
# ============================================================

st.set_page_config(
    page_title="DARWIN | Handwriting Analysis",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
STAGE9 = ROOT / "outputs" / "stage9"
MODEL_PATH = STAGE9 / "random_forest_model.joblib"
FEATURES_PATH = STAGE9 / "selected_features.json"
METADATA_PATH = STAGE9 / "model_metadata.json"
REFERENCE_DATA_PATH = ROOT / "data" / "data.csv"

# ---------- Visual styling ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp {
        background:
          radial-gradient(circle at 85% 0%, rgba(197,164,100,.10), transparent 28%),
          linear-gradient(145deg, #090b0e 0%, #0d1116 55%, #10151a 100%);
        color: #edf0f2;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #11161c 0%, #0d1116 100%);
        border-right: 1px solid rgba(207,181,133,.15);
    }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -.035em; }
    h1 { font-weight: 800 !important; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(26,32,39,.96), rgba(17,22,28,.96));
        border: 1px solid rgba(207,181,133,.18);
        padding: 16px 18px;
        border-radius: 15px;
        box-shadow: 0 8px 28px rgba(0,0,0,.16);
    }
    [data-testid="stMetricLabel"] { color: #aeb8c2 !important; }
    [data-testid="stMetricValue"] { color: #f2e7d1 !important; }
    div[data-testid="stFileUploader"] {
        background: rgba(18,23,29,.72);
        border: 1px dashed rgba(207,181,133,.38);
        border-radius: 14px;
        padding: 12px;
    }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 10px;
        min-height: 2.8rem;
        font-weight: 700;
        border: 1px solid rgba(207,181,133,.38);
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #c6a66b, #e0c995);
        color: #17130d;
        border: 0;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(100deg, #d6b77d, #ead9ad);
        color: #17130d;
    }
    .hero {
        padding: 1.3rem 1.5rem;
        border: 1px solid rgba(207,181,133,.22);
        border-radius: 18px;
        background: linear-gradient(115deg, rgba(25,31,38,.96), rgba(17,22,28,.78));
        margin-bottom: 1.2rem;
    }
    .eyebrow {
        color: #c9ad78; text-transform: uppercase; letter-spacing: .18em;
        font-size: .72rem; font-weight: 700;
    }
    .hero-title {
        font-family: 'Manrope', sans-serif; font-size: 2.35rem;
        line-height: 1.08; font-weight: 800; color: #f3eee4;
        margin: .35rem 0 .5rem 0;
    }
    .hero-copy { color: #aeb8c2; font-size: .96rem; max-width: 850px; }
    .section-kicker {
        color: #c9ad78; font-size: .75rem; font-weight: 700;
        letter-spacing: .13em; text-transform: uppercase;
    }
    .subtle { color: #9da8b3; font-size: .88rem; }
    .panel {
        background: rgba(20,26,33,.78);
        border: 1px solid rgba(207,181,133,.16);
        border-radius: 14px; padding: 1rem 1.1rem;
    }
    .tag {
        display: inline-block; padding: .22rem .55rem; border-radius: 99px;
        font-size: .72rem; font-weight: 700; letter-spacing: .03em;
        background: rgba(198,166,107,.14); color: #dfc38e;
        border: 1px solid rgba(198,166,107,.28);
    }
    div[data-testid="stTabs"] button { color: #b8c0c8; }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #e2c68f !important;
        border-bottom-color: #c6a66b !important;
    }
    hr { border-color: rgba(207,181,133,.14) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    required = [MODEL_PATH, FEATURES_PATH, METADATA_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required Stage 9 files not found: " + ", ".join(missing)
        )

    payload = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    features = payload["selected_features"]
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if len(features) != 118:
        raise ValueError(f"Expected 118 selected features; found {len(features)}.")

    model = joblib.load(MODEL_PATH)
    return model, features, metadata


@st.cache_data
def load_reference_medians(path_string, mtime):
    path = Path(path_string)
    if not path.exists():
        raise FileNotFoundError(
            "The original training dataset was not found at data/data.csv. "
            "It is needed to calculate reference medians for the experimental mode."
        )
    reference = pd.read_csv(path)
    # The original source data may contain 450 features. Keep only the
    # selected 118, in the exact order used by the trained model.
    return reference


def normalize_name(value):
    return "".join(ch.lower() for ch in str(value).strip() if ch.isalnum())


def make_mapping_template(source_columns, target_features):
    return pd.DataFrame({
        "source_column": [""] * len(target_features),
        "darwin_feature": target_features,
    })


def make_prediction(model, X):
    predictions = model.predict(X).astype(int)
    class_values = list(model.classes_)
    if 1 not in class_values:
        raise ValueError(f"Expected model class 1 for patient label; found {class_values}.")
    positive_index = class_values.index(1)
    probabilities = model.predict_proba(X)[:, positive_index]
    return predictions, probabilities


def render_results(result, result_key, input_name, mode_label):
    st.markdown('<div class="section-kicker">Prediction output</div>', unsafe_allow_html=True)
    st.subheader("Results")

    total = len(result)
    healthy_count = int((result["predicted_label"] == "H").sum())
    patient_count = int((result["predicted_label"] == "P").sum())

    a, b, c = st.columns(3)
    a.metric("Records processed", f"{total:,}")
    b.metric("Predicted H", f"{healthy_count:,}")
    c.metric("Predicted P", f"{patient_count:,}")

    chart_col, table_col = st.columns([0.85, 1.65], gap="large")
    with chart_col:
        st.markdown("**Class distribution**")
        chart = pd.DataFrame(
            {"Records": [healthy_count, patient_count]},
            index=["H · Healthy label", "P · Patient label"],
        )
        st.bar_chart(chart, use_container_width=True)
    with table_col:
        st.markdown("**Participant-level output**")
        st.dataframe(result, use_container_width=True, hide_index=True, height=340)

    csv_data = result.to_csv(index=False).encode("utf-8-sig")
    safe_stem = Path(input_name).stem if input_name else "participants"
    st.download_button(
        "Download results as CSV",
        data=csv_data,
        file_name=f"{safe_stem}_darwin_{result_key}.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"download_{result_key}",
    )
    st.caption(
        f"Output generated using: {mode_label}. The patient probability is the model's "
        "predicted probability for class P, not a clinical probability."
    )


# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Handwriting dynamics · Machine learning</div>
      <div class="hero-title">DARWIN</div>
      <div class="hero-copy">
        A handwriting-based analysis platform. Select a data pathway, review the
        input checks, and generate participant-level model outputs.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## DARWIN")
    st.caption("Research workspace")
    st.divider()
    st.markdown("**Model**")
    st.write("Random Forest")
    st.write("118 selected features")
    st.divider()
    st.markdown("**Class labels**")
    st.write("H · Healthy label")
    st.write("P · Patient label")
    st.divider()
    st.caption("Stage 9 model artifact · No retraining in this app")

try:
    model, selected_features, metadata = load_assets()
except Exception as exc:
    st.error(f"Unable to load model assets: {exc}")
    st.info(
        "Keep app.py in the DARWIN project root. The folder outputs/stage9 must contain "
        "random_forest_model.joblib, selected_features.json and model_metadata.json."
    )
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Selected features", len(selected_features))
m2.metric("Training cohort", metadata.get("participants_used_for_final_fit", 174))
m3.metric("Prediction model", "Random Forest")

standard_tab, experimental_tab = st.tabs(
    ["  Standard DARWIN  ", "  Experimental cross-dataset  "]
)

# ============================================================
# STANDARD MODE — original behaviour, polished UI
# ============================================================
with standard_tab:
    st.markdown('<div class="section-kicker">01 / Standard pathway</div>', unsafe_allow_html=True)
    st.subheader("Upload DARWIN feature data")
    st.markdown(
        '<p class="subtle">Use a CSV containing the original DARWIN feature columns. '
        'The model requires all 118 selected inputs in their original form.</p>',
        unsafe_allow_html=True,
    )
    standard_file = st.file_uploader(
        "Choose DARWIN CSV",
        type=["csv"],
        key="standard_upload",
    )

    if standard_file is not None:
        try:
            raw = pd.read_csv(standard_file)
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
            st.stop()

        x1, x2, x3 = st.columns(3)
        x1.metric("Rows", f"{raw.shape[0]:,}")
        x2.metric("Columns", f"{raw.shape[1]:,}")
        x3.metric("Required inputs", len(selected_features))

        missing = [f for f in selected_features if f not in raw.columns]
        if missing:
            st.error(f"{len(missing)} of the 118 required DARWIN features are missing.")
            with st.expander("View missing feature names"):
                st.code(", ".join(missing))
        else:
            X = raw.loc[:, selected_features].copy()
            conversion_errors = []
            for col in selected_features:
                converted = pd.to_numeric(X[col], errors="coerce")
                bad = converted.isna() & X[col].notna()
                if bad.any():
                    conversion_errors.append((col, int(bad.sum())))
                X[col] = converted

            if conversion_errors:
                st.error("Some required feature columns contain non-numeric values.")
                st.dataframe(
                    pd.DataFrame(conversion_errors, columns=["Feature", "Invalid values"]),
                    use_container_width=True,
                )
            elif X.isna().any().any():
                st.error("Missing values were found in required features. Standard mode does not impute.")
                st.dataframe(
                    X.isna().sum().loc[lambda s: s > 0].rename("Missing values").reset_index(),
                    use_container_width=True,
                )
            elif not np.isfinite(X.to_numpy(dtype=float)).all():
                st.error("Infinite or non-finite values were found.")
            else:
                st.success("All 118 required features are present and numeric.")
                with st.expander("Preview model inputs"):
                    st.dataframe(X.head(8), use_container_width=True)

                if st.button("Run prediction", type="primary", use_container_width=True, key="run_standard"):
                    with st.spinner("Generating predictions..."):
                        try:
                            pred, prob = make_prediction(model, X)
                            result = pd.DataFrame(index=raw.index)
                            if "ID" in raw.columns:
                                result["ID"] = raw["ID"].values
                            result["predicted_class"] = pred
                            result["predicted_label"] = np.where(pred == 1, "P", "H")
                            result["patient_probability"] = prob
                            result["patient_probability_percent"] = np.round(prob * 100, 2)
                            st.session_state["darwin_standard_results"] = result
                            st.session_state["darwin_standard_name"] = standard_file.name
                        except Exception as exc:
                            st.error(f"Prediction failed: {exc}")

    if "darwin_standard_results" in st.session_state:
        render_results(
            st.session_state["darwin_standard_results"],
            "standard_predictions",
            st.session_state.get("darwin_standard_name", "participants.csv"),
            "Standard DARWIN 118-feature input",
        )

# ============================================================
# EXPERIMENTAL MODE — explicit user mapping + median imputation
# ============================================================
with experimental_tab:
    st.markdown('<div class="section-kicker">02 / Experimental pathway</div>', unsafe_allow_html=True)
    st.subheader("Cross-dataset feature mapping")
    st.markdown(
        '<p class="subtle">Use this pathway for datasets whose columns differ from the original '
        'DARWIN feature names. You provide an explicit mapping file; unmapped inputs are filled '
        'with medians calculated from the original DARWIN training data.</p>',
        unsafe_allow_html=True,
    )
    st.warning(
        "Experimental compatibility mode: mapping and median imputation make the input shape "
        "acceptable to the existing model, but do not make measurements equivalent. Outputs "
        "from this pathway are not validated cross-dataset predictions.",
        icon="⚠️",
    )

    exp_file = st.file_uploader(
        "Choose another handwriting dataset (CSV)",
        type=["csv"],
        key="experimental_upload",
    )

    if exp_file is not None:
        try:
            exp_raw = pd.read_csv(exp_file)
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
            st.stop()

        q1, q2, q3 = st.columns(3)
        q1.metric("Uploaded records", f"{len(exp_raw):,}")
        q2.metric("Available columns", f"{len(exp_raw.columns):,}")
        q3.metric("Model inputs", "118")

        st.markdown("**Step 1 — Inspect uploaded columns**")
        with st.expander("View uploaded dataset columns and preview", expanded=False):
            st.write(exp_raw.columns.tolist())
            st.dataframe(exp_raw.head(8), use_container_width=True)

        st.markdown("**Step 2 — Map source columns to DARWIN features**")
        st.caption(
            "Choose a source column and the DARWIN feature it is genuinely comparable to. "
            "The dropdowns help build the mapping inside the app; they do not establish "
            "scientific equivalence. Leave rows blank when no defensible match exists."
        )

        source_options = [""] + [str(c) for c in exp_raw.columns]
        target_options = [""] + list(selected_features)

        seed = pd.DataFrame({
            "source_column": [""] * 5,
            "darwin_feature": [""] * 5,
        })

        mapping_editor = st.data_editor(
            seed,
            key="experimental_mapping_editor",
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "source_column": st.column_config.SelectboxColumn(
                    "Dataset column",
                    help="Choose a numeric column from the uploaded dataset.",
                    options=source_options,
                    required=False,
                ),
                "darwin_feature": st.column_config.SelectboxColumn(
                    "DARWIN feature",
                    help="Choose the original model feature that has a defensible matching definition.",
                    options=target_options,
                    required=False,
                ),
            },
        )

        mapping = mapping_editor[["source_column", "darwin_feature"]].copy()
        mapping = mapping.fillna("")
        mapping["source_column"] = mapping["source_column"].astype(str).str.strip()
        mapping["darwin_feature"] = mapping["darwin_feature"].astype(str).str.strip()
        mapping = mapping[
            (mapping["source_column"] != "") & (mapping["darwin_feature"] != "")
        ]

        if mapping.empty:
            st.info(
                "No mappings selected yet. Add rows and choose both columns to create a mapping. "
                "You can still inspect the dataset above without running experimental predictions."
            )
        else:
            unknown_targets = sorted(set(mapping["darwin_feature"]) - set(selected_features))
            unknown_sources = sorted(set(mapping["source_column"]) - set(exp_raw.columns))
            duplicate_targets = mapping["darwin_feature"].duplicated(keep=False)
            duplicate_sources = mapping["source_column"].duplicated(keep=False)

            if unknown_targets:
                st.error("Mapping contains unknown DARWIN target features.")
                st.write(unknown_targets)
            elif unknown_sources:
                st.error("Mapping contains source columns that are not in the uploaded dataset.")
                st.write(unknown_sources)
            elif duplicate_targets.any() or duplicate_sources.any():
                st.error(
                    "Each source column and each DARWIN feature can be mapped only once. "
                    "Remove duplicate selections."
                )
            else:
                st.markdown("**Mapping preview**")
                st.dataframe(mapping, use_container_width=True, hide_index=True)

                try:
                    reference = load_reference_medians(
                        str(REFERENCE_DATA_PATH),
                        REFERENCE_DATA_PATH.stat().st_mtime if REFERENCE_DATA_PATH.exists() else 0,
                    )
                except Exception as exc:
                    st.error(str(exc))
                    st.stop()

                missing_reference = [f for f in selected_features if f not in reference.columns]
                if missing_reference:
                    st.error(
                        "The reference dataset does not contain all selected DARWIN features. "
                        "Cannot calculate reference medians."
                    )
                    st.write(missing_reference)
                    st.stop()

                medians = reference.loc[:, selected_features].apply(
                    pd.to_numeric, errors="coerce"
                ).median()
                if medians.isna().any():
                    st.error("Some reference medians could not be calculated.")
                    st.stop()

                # Initialise all 118 inputs with original-DARWIN reference medians.
                # Explicit mappings overwrite only the selected target columns.
                X_exp = pd.DataFrame(
                    np.tile(medians.to_numpy(dtype=float), (len(exp_raw), 1)),
                    columns=selected_features,
                    index=exp_raw.index,
                )
                conversion_errors = []
                for row in mapping.itertuples(index=False):
                    source_col = row.source_column
                    target_col = row.darwin_feature
                    converted = pd.to_numeric(exp_raw[source_col], errors="coerce")
                    invalid = converted.isna() & exp_raw[source_col].notna()
                    if invalid.any():
                        conversion_errors.append((source_col, int(invalid.sum())))
                    X_exp[target_col] = converted

                if conversion_errors:
                    st.error("Mapped source columns contain non-numeric values.")
                    st.dataframe(
                        pd.DataFrame(
                            conversion_errors,
                            columns=["Source column", "Invalid values"],
                        ),
                        use_container_width=True,
                    )
                else:
                    # Missing values in mapped source columns are also imputed
                    # with original-DARWIN medians for their selected targets.
                    for row in mapping.itertuples(index=False):
                        target_col = row.darwin_feature
                        X_exp[target_col] = X_exp[target_col].fillna(medians[target_col])

                    mapped_targets = set(mapping["darwin_feature"])
                    imputed_features = [f for f in selected_features if f not in mapped_targets]

                    st.markdown("**Step 3 — Review prepared model inputs**")
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Mapped features", len(mapped_targets))
                    r2.metric("Median-imputed features", len(imputed_features))
                    r3.metric("Mapping coverage", f"{len(mapped_targets) / 118 * 100:.1f}%")

                    st.warning(
                        f"Only {len(mapped_targets)} of 118 model inputs are populated from "
                        "the uploaded dataset. The remaining inputs use original-DARWIN "
                        "training-set medians. Predictions are experimental and may be "
                        "dominated by imputed values.",
                        icon="⚠️",
                    )

                    with st.expander("View prepared 118-feature input"):
                        st.dataframe(X_exp.head(8), use_container_width=True)
                    with st.expander("View imputed feature names"):
                        st.write(imputed_features)

                    if st.button(
                        "Run prediction",
                        type="primary",
                        use_container_width=True,
                        key="run_experimental",
                    ):
                        with st.spinner("Generating experimental outputs..."):
                            try:
                                pred, prob = make_prediction(model, X_exp)
                                result = pd.DataFrame(index=exp_raw.index)
                                id_col = next(
                                    (
                                        c for c in exp_raw.columns
                                        if normalize_name(c) in {
                                            "id", "subject", "participantid", "participant"
                                        }
                                    ),
                                    None,
                                )
                                if id_col:
                                    result["source_ID"] = exp_raw[id_col].values
                                result["predicted_class"] = pred
                                result["predicted_label"] = np.where(pred == 1, "P", "H")
                                result["patient_probability"] = prob
                                result["patient_probability_percent"] = np.round(prob * 100, 2)
                                result["mapped_features"] = len(mapped_targets)
                                result["imputed_features"] = len(imputed_features)
                                result["input_mode"] = "Experimental cross-dataset"

                                st.session_state["darwin_experimental_results"] = result
                                st.session_state["darwin_experimental_name"] = exp_file.name
                                st.session_state["darwin_experimental_mapped"] = len(mapped_targets)
                                st.session_state["darwin_experimental_imputed"] = len(imputed_features)
                            except Exception as exc:
                                st.error(f"Experimental prediction failed: {exc}")

    if "darwin_experimental_results" in st.session_state:
        st.markdown("---")
        st.markdown("**Most recent experimental output**")
        st.caption(
            f"Feature mapping: {st.session_state.get('darwin_experimental_mapped', 0)}/118; "
            f"median-imputed: {st.session_state.get('darwin_experimental_imputed', 118)}/118."
        )
        render_results(
            st.session_state["darwin_experimental_results"],
            "experimental_predictions",
            st.session_state.get("darwin_experimental_name", "external_dataset.csv"),
            "Experimental cross-dataset mapping and median imputation",
        )

st.markdown("---")
st.markdown(
    '<div class="subtle">DARWIN · Stage 9 Random Forest · Standard and experimental input pathways</div>',
    unsafe_allow_html=True,
)
