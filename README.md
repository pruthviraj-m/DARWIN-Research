# DARWIN — Handwriting-Based Alzheimer's Disease Classification

DARWIN is a research prototype that investigates whether handwriting-derived
kinematic and dynamic features can distinguish participants labelled healthy
(`H`) from participants labelled as patients (`P`) in the DARWIN dataset.

The repository contains a machine-learning research pipeline and a Streamlit
application for processing compatible DARWIN feature CSV files.

> **Research-only notice:** DARWIN is not a medical device and must not be used
> to diagnose, screen, rule out, or make treatment decisions about Alzheimer's
> disease. Model outputs are research predictions, not clinical probabilities.
> Independent external validation and clinical evaluation have not been
> established.

## 1. Dataset

The current experiment uses `data/data.csv`.

| Property | Value |
|---|---:|
| Participants | 174 |
| Healthy-labelled (`H`) | 85 |
| Patient-labelled (`P`) | 89 |
| Original feature columns | 450 |
| Derived `total_time_*` columns removed | 25 |
| Candidate features after removal | 425 |
| Final full-data selected features | 118 |

The dataset contains 25 handwriting tasks and 18 feature groups. The 25
`total_time_*` columns were verified to equal the corresponding
`air_time_* + paper_time_*` values and were removed as derived/redundant
features before modelling.

Class encoding used in the scripts:

- `H` → `0`
- `P` → `1`

Keep the original dataset in `data/data.csv`. Do not commit participant-level
data to a public repository unless its licence, consent conditions, and
sharing permissions explicitly allow it.

## 2. Research methodology

### Internal validation: nested cross-validation

Stage 8 and Stage 10 use the same nested cross-validation design:

- Outer loop: 5-fold stratified cross-validation.
- Inner loop: 3-fold stratified cross-validation.
- Random seed: 42 for the outer split; inner split seed is `42 + outer_fold`.
- Feature-selection strategy is selected using inner-loop balanced accuracy.
- The selected strategy is refitted on the outer training partition and
  evaluated on the untouched outer test partition.

The candidate feature strategies are:

1. All 425 candidate features.
2. Top 150 features ranked by Extra Trees.
3. Top 100 features ranked by Extra Trees.
4. Correlation-pruned features: Extra Trees top 150 followed by greedy
   absolute Spearman-correlation pruning at a threshold of 0.85.

The compared models are TabPFN, Extra Trees, Random Forest, and an RBF-kernel
SVM. The SVM is placed in a pipeline with standard scaling.

Stage 10 additionally records outer-fold predictions and diagnostics,
including accuracy, balanced accuracy, sensitivity, specificity, precision,
F1-score, ROC-AUC, Brier score, confusion matrices, ROC points, and
calibration bins.

**Interpretation:** Stage 8 and Stage 10 use the same 174 participants and
the same nested-CV procedure. Stage 10 adds diagnostic outputs; it is not an
independent replication or external validation. Pooled out-of-fold predictions
are useful for paired internal model comparisons, but they do not replace
validation on a new cohort.

### Final inference artifacts: Stage 9

Stage 9 fits the final TabPFN and Random Forest models on all 174 participants.
Its feature selection is performed on the full dataset:

1. Extra Trees ranking with 500 estimators.
2. Retain the top 150 candidate features.
3. Greedily remove features with absolute Spearman correlation at or above
   0.85 with an already retained feature.
4. Fit TabPFN (CPU) and Random Forest (400 estimators) on the resulting
   selected feature set.

The saved full-data feature set contains 118 features.

These full-data models are for inference and software integration. Their fit
on all participants is **not** a performance estimate. Use the nested-CV
outer-fold results for internal validation.

### Explainability

- Stage 11: Random Forest SHAP TreeExplainer and exploratory permutation
  importance.
- Stage 11B: bounded TabPFN permutation-importance and local
  median-replacement sensitivity analysis.

Explainability outputs are descriptive and in-sample because the final models
were fitted on all participants. Correlated features can share or distort
attributions. These analyses do not establish causal effects or clinical
biomarkers. Median-replacement sensitivity is not SHAP.

## 3. Main project layout

The project inventory includes the following main files and output folders:

```text
DARWIN-Research/
├── app.py
├── data/
│   └── data.csv
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── feature_selection/
│   ├── feature_reduction/
│   ├── feature_ablation/
│   ├── feature_correlation/
│   ├── feature_stability/
│   ├── cross_model_agreement_fast/
│   ├── repeated_cv/
│   ├── stage8/
│   ├── stage9/
│   │   ├── random_forest_model.joblib
│   │   ├── tabpfn_model.joblib
│   │   ├── selected_features.json
│   │   ├── model_metadata.json
│   │   └── predict_new_data.py
│   ├── stage10/
│   │   ├── oof_predictions.csv
│   │   ├── fold_metrics.csv
│   │   ├── summary_metrics.csv
│   │   ├── confusion_matrices.csv
│   │   ├── roc_curve_points.csv
│   │   ├── calibration_bins.csv
│   │   └── metadata.json
│   ├── stage11/
│   └── stage11_tabpfn_efficient/
├── stage8_final_feature_selection.py
├── stage9_final_model.py
├── stage10_final_validation.py
├── stage11_explainability.py
├── stage11_tabpfn_efficient.py
├── inspect_dataset.py
├── verify_time_relationships.py
├── feature_selection.py
├── feature_reduction.py
├── feature_ablation.py
├── feature_correlation.py
├── feature_stability.py
├── cross_model_agreement.py
├── cross_model_agreement_fast.py
├── repeated_cv.py
├── feature_importance.py
├── task_feature_importance.py
├── train.py
├── config.py
├── requirements.txt
└── README.md
```

This is a functional overview, not a guarantee that every generated output is
present in every checkout. The `outputs/` directory contains experiment
artifacts and may be large.

## 4. Environment

The recorded development environment was:

- Windows
- Python 3.13.1
- NumPy 2.5.4
- pandas 3.1.1
- scikit-learn 1.9.1
- Streamlit 1.61.1
- CPU execution for TabPFN

These are the versions recorded during development, not a guarantee that every
future package release will remain compatible. TabPFN may install additional
dependencies, including PyTorch. For a reproducible environment, record the
complete installed package set after confirming that all project scripts run.

## 5. Setup (Windows PowerShell)

Run commands from the project root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place the permitted dataset at:

```text
data\data.csv
```

## 6. Running the application

The current Streamlit application uses the saved **Random Forest** artifact
and the 118-feature list from Stage 9.

Required files:

```text
outputs/stage9/random_forest_model.joblib
outputs/stage9/selected_features.json
outputs/stage9/model_metadata.json
```

Start the application:

```powershell
streamlit run app.py
```

### Standard DARWIN pathway

Upload a CSV containing all 118 selected DARWIN features, with numeric,
finite, non-missing values. The app checks the required columns and generates
predicted class labels and the model's class-P probability output.

The probability is a model score, not a clinically calibrated risk estimate.

### Experimental cross-dataset pathway

The app also contains an experimental column-mapping and reference-median
imputation pathway. Mapping columns by name or shape does not establish that
measurements from another study are scientifically equivalent to DARWIN
features. Outputs from this pathway are not validated cross-dataset predictions
and must not be presented as such.

## 7. Re-running the research stages

Run commands from the project root after installing dependencies and placing
the dataset in `data/data.csv`.

```powershell
python inspect_dataset.py
python verify_time_relationships.py
python stage8_final_feature_selection.py
python stage9_final_model.py
python stage10_final_validation.py
python stage11_explainability.py
python stage11_tabpfn_efficient.py
```

Some stages can be computationally expensive, especially TabPFN on CPU.
Stage 9 refits final models on all participants. Stage 10 reruns nested
cross-validation. Stage 11 and Stage 11B perform explainability analyses and
do not provide independent validation.

The additional feature-selection, feature-reduction, ablation, stability,
correlation, repeated-CV, task-importance, and cross-model-agreement scripts
generate supporting research outputs. Check each script's header and output
directory before running it.

## 8. Current model comparison

The Stage 10 pooled out-of-fold results are the primary internal comparison.
The current reported pattern is:

- TabPFN: strongest balanced-accuracy / specificity / Brier-score profile.
- Random Forest: highly competitive; strong F1 and sensitivity.
- Extra Trees: strongest ROC-AUC and sensitivity in the reported comparison.
- SVM: lower headline performance than the other three models.

Paired comparisons did not establish statistically significant differences
in classification error rates between model pairs after correction for
multiple comparisons. Small-sample uncertainty and the single-cohort design
must be acknowledged.

Do not describe Stage 8 and Stage 10 as two independent validation studies.
Do not claim that the results demonstrate clinical readiness.

## 9. Limitations and responsible use

1. The dataset contains 174 participants; this is a small cohort for a
   high-stakes clinical application.
2. No independent external validation cohort is currently available.
3. Nested cross-validation estimates internal performance on this dataset;
   it does not establish generalisation to other populations, devices, or
   acquisition protocols.
4. Full-data model fitting and explainability are descriptive/in-sample.
5. Correlated handwriting features can affect feature selection and
   explainability attribution.
6. Cross-dataset feature mapping and median imputation do not establish
   measurement equivalence.
7. Model outputs are not diagnoses and must not guide medical decisions.

## 10. Reproducibility checklist

Before publishing or handing over the project:

- [ ] Add the dataset's official citation, version, licence, and access link.
- [ ] Confirm the Python and package versions on the machine used for the
      final run; save `python -m pip freeze` to an environment lock file.
- [ ] Keep the exact source CSV and document its checksum/version where
      permitted.
- [ ] Preserve Stage 8/10 metadata, OOF predictions, and fold-level metrics.
- [ ] Preserve Stage 9 selected-feature JSON and model metadata alongside
      model artifacts.
- [ ] Document the exact commit or source-code version used to generate
      each reported result.
- [ ] Do not describe internal nested CV as external validation.
- [ ] Keep participant data and model artifacts out of public repositories
      unless their sharing terms explicitly permit distribution.
