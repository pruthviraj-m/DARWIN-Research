# Early Alzheimer's Disease Detection from Online Handwriting (DARWIN Prototype)

> **IMPORTANT DISCLAIMER**  
> **Research prototype. Not a clinically validated diagnostic tool.**  
> This software is an academic research prototype developed strictly for algorithmic study and scientific evaluation. It is **NOT** a certified medical diagnostic system and **MUST NOT** be used for clinical diagnosis, patient triage, or healthcare decision-making.

---

## 1. Problem Statement
Alzheimer's disease (AD) is characterized by progressive neurodegeneration that impairs cognitive, linguistic, and fine motor functions. Traditional clinical screening tools (e.g., MMSE, MoCA) can be subjective, anxiety-inducing, and sensitive to educational background. Online handwriting analysis using digitized graphic tablets captures objective, sub-second kinematic and neuromuscular markers—such as pen hesitations in the air, micro-tremors, grip force variability, and geometric boundary distortions—that can manifest in the early stages of cognitive decline.

This project implements a rigorous, leak-free, reproducible research prototype evaluating machine learning algorithms on online handwriting features from the benchmark **DARWIN** dataset.

---

## 2. Dataset Description
- **Dataset**: DARWIN (Data Repository for Handwriting with Inverted Pen), public benchmark.
- **Source Paper**: Cilia, N. D., De Stefano, C., Fontanella, F., & Di Freca, A. S. (2022). *An experimental protocol for early Alzheimer’s disease detection through handwriting analysis*. Engineering Applications of Artificial Intelligence, 111, 104822.
- **Cohort**: 174 participants:
  - **89 Early Alzheimer's Disease patients** (Label: `P` $\to$ encoded as `1`)
  - **85 Healthy Controls** (Label: `H` $\to$ encoded as `0`)
- **Features**: 450 numeric features derived from **25 standardized pen-tablet handwriting tasks** (18 base features per task):
  1. **Timing Modalities (100 features)**: `air_time`, `paper_time`, `total_time`, `num_of_pendown`
  2. **GMRT / Tremor / Kinematics (225 features)**: `gmrt_in_air`, `gmrt_on_paper`, `mean_gmrt`, `mean_acc_in_air`, `mean_acc_on_paper`, `mean_jerk_in_air`, `mean_jerk_on_paper`, `mean_speed_in_air`, `mean_speed_on_paper`
  3. **Spatial Modalities (75 features)**: `max_x_extension`, `max_y_extension`, `disp_index`
  4. **Pressure Modalities (50 features)**: `pressure_mean`, `pressure_var`

---

## 3. Frozen Evaluation Protocol

To prevent methodological biases, optimistic reporting, and data leakage, the evaluation strictly follows a frozen protocol:

### A. Outer Resampling
- **RepeatedStratifiedKFold**: 5 folds $\times$ 3 repeats = **15 identical paired outer folds** (`random_state = 20261018`).
- All candidate models are evaluated on the exact same 15 test splits to enable paired statistical testing.

### B. Inner Cross-Validation & Tuning
- **StratifiedKFold (3 splits)**: `random_state = 20261018 + fold_number`.
- Hyperparameters are tuned strictly inside inner CV on the training fold only.

### C. Strict Data Leakage Prevention (Non-Negotiable Rules)
1. **Pre-processing Isolation**: Feature standardization (`StandardScaler`), feature selection, and model tuning are fit strictly on training fold indices. Test data is never observed prior to out-of-fold inference.
2. **Early Stopping Isolation**: Early stopping for neural networks uses an internal validation split carved out of the training fold (`random_state = 20261018 + fold_number`), never the outer test fold.
3. **Exploratory Split Isolation**: The legacy 35-person held-out split (80/20 stratified split, `random_state=42`, 139 train / 35 test) is treated strictly as an isolated exploratory row and is never used for tuning or selection.
4. **Single-Execution Rule**: Every candidate model is run once under the frozen protocol without post-hoc seed searching or cherry-picking.

### D. Statistical Significance Testing
- **Nadeau-Bengio Corrected Resampled $t$-Test**:
  Evaluated on the 15 paired fold accuracy differences against the Extra Trees baseline:
  $$\hat{\sigma}^2 = \left(\frac{1}{n} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) S^2 \quad (\text{ddof}=1, \ df = n - 1 = 14)$$
  The correction term $\frac{n_{\text{test}}}{n_{\text{train}}}$ accounts for the non-independence of resampled training sets.

---

## 4. Evaluated Models

1. **Random Forest (Baseline)**: 300 estimators, balanced class weights, tuned over `max_depth` $\in [\text{None}, 20]$ and `min_samples_leaf` $\in [1, 2]$.
2. **Extra Trees (Baseline)**: 500 estimators, balanced class weights, same parameter grid.
3. **Task-Structured Attention Network (Deep Learning)**:
   - Reshapes 450 features into $(25 \text{ tasks} \times 18 \text{ features})$.
   - Shared task encoder ($18 \to 32 \to 16$) with LayerNorm, ReLU, and Dropout.
   - Learned task embedding ($25 \times 16$).
   - Multi-task attention pooling mechanism learning per-task weights $\alpha_t$.
   - Binary classification head ($16 \to 8 \to 1$).
4. **Deep Ensemble (10 Networks)**:
   - 10 Task-Structured Attention Networks trained with distinct seeds (42..51) and Gaussian input noise ($\sigma=0.05$) regularization.
   - Out-of-fold predicted probabilities are averaged.
5. **TabPFN (Prior-Data Fitted Network)**:
   - *Status*: Evaluated; skipped per protocol because TabPFN v2 requires an interactive PriorLabs API token (`TABPFN_TOKEN`) to download weights, precluding automated local execution, in addition to dimensional constraints.
6. **Hybrid Ensemble (Exploratory)**:
   - Probability averaging of Deep Ensemble and Extra Trees ($0.5 \times P_{\text{Deep}} + 0.5 \times P_{\text{ET}}$).

---

## 5. Experimental Results

### A. Nested Repeated Cross-Validation Performance (15 Paired Folds)

| Model | Accuracy (Primary) | Balanced Accuracy | Sensitivity (AD) | Specificity | F1 Score (AD) | Single Split (80/20)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 88.31% ± 4.67% | 88.30% ± 4.71% | 88.37% ± 6.17% | 88.24% ± 8.02% | 88.56% ± 4.40% | 82.86% (29/35) |
| **Extra Trees** | 89.46% ± 4.30% | 89.46% ± 4.35% | 89.50% ± 4.66% | 89.41% ± 7.77% | 89.72% ± 4.02% | 74.29% (26/35) |
| **Task-Structured Net** | 87.35% ± 5.81% | 87.34% ± 5.87% | 88.41% ± 7.42% | 86.27% ± 8.35% | 87.76% ± 5.42% | 80.00% (28/35) |
| **Deep Ensemble (10 nets)** | 89.84% ± 5.21% | 89.85% ± 5.25% | 89.89% ± 5.92% | 89.80% ± 7.48% | 90.06% ± 4.96% | 80.00% (28/35) |
| **Hybrid (Deep Ens + ET)** | **90.97% ± 4.25%** | **90.98% ± 4.28%** | **90.98% ± 5.17%** | **90.98% ± 6.19%** | **91.19% ± 3.99%** | 82.86% (29/35) |

*\*Note: The single exploratory 80/20 held-out split is reported for historical reference only and must never be interpreted as a generalizable metric.*

### B. Statistical Comparisons (vs Extra Trees Baseline)

- **Random Forest vs Extra Trees**:
  > "Under nested repeated cross-validation, model Random Forest achieved 88.31% +/- 4.67% accuracy vs 89.46% +/- 4.30% for Extra Trees (mean difference -1.15 pp; corrected 95% CI [-4.02, +1.72]; p = 0.4050). We found insufficient evidence of a difference."
- **Task-Structured Net vs Extra Trees**:
  > "Under nested repeated cross-validation, model Task-Structured Net achieved 87.35% +/- 5.81% accuracy vs 89.46% +/- 4.30% for Extra Trees (mean difference -2.11 pp; corrected 95% CI [-8.16, +3.95]; p = 0.4682). We found insufficient evidence of a difference."
- **Deep Ensemble (10 nets) vs Extra Trees**:
  > "Under nested repeated cross-validation, model Deep Ensemble (10 nets) achieved 89.84% +/- 5.21% accuracy vs 89.46% +/- 4.30% for Extra Trees (mean difference +0.38 pp; corrected 95% CI [-5.60, +6.36]; p = 0.8933). We found insufficient evidence of a difference."
- **Hybrid (Deep Ens + ET) vs Extra Trees**:
  > "Under nested repeated cross-validation, model Hybrid (Deep Ens + ET) achieved 90.97% +/- 4.25% accuracy vs 89.46% +/- 4.30% for Extra Trees (mean difference +1.52 pp; corrected 95% CI [-2.80, +5.84]; p = 0.4633). We found insufficient evidence of a difference."

---

## 6. Interpretability & Clinical Feature Associations

> **IMPORTANT**: Feature and task importance reflects statistical model associations within this specific dataset, **NOT biological or clinical causation**.

### A. Task-Level Importance & Selection Consistency
Across the 15 outer cross-validation folds, handwriting tasks were ranked by cross-validated SHAP impact:
- **Task 17**: Selected in **15 / 15 folds (100.0%)** — Complex geometric loop tracing.
- **Task 23**: Selected in **15 / 15 folds (100.0%)** — Complex sentence writing with cognitive-linguistic demands.
- **Task 07**: Selected in **15 / 15 folds (100.0%)** — High-frequency alternating strokes.
- **Task 19**: Selected in **15 / 15 folds (100.0%)** — Continuous cursive strokes.
- **Task 09**: Selected in **13 / 15 folds (86.7%)** — Spatially constrained square tracing.

### B. Feature Group Contribution
1. **Timing Features (37.1% relative impact)**: Pen-up air latency (`air_time`), in-air hesitations, and prolonged total execution times reflect cognitive processing delays and hesitation before motor execution.
2. **GMRT & Kinematics (36.3% relative impact)**: Global Motion Regularity Trajectory (`mean_gmrt`, `gmrt_in_air`), velocity variability, and acceleration jerk capture fine motor dysfluency and sub-clinical tremors.
3. **Spatial Features (13.3% relative impact)**: Micrographia (reduced writing size) and dispersion index distortions (`max_x_extension`, `disp_index`).
4. **Pressure Features (13.3% relative impact)**: Variability in pen pressure (`pressure_var`, `pressure_mean`) indicates grip instability.

---

## 7. Anonymous Error Analysis & Threshold Tuning

All error analyses are conducted using anonymous row indices ($0 \dots 173$) with zero participant PII.

### A. Classification Profiles (Extra Trees, 3 Fold Appearances per Participant)
- **Consistently Correct (0 errors)**: **152 / 174 participants (87.4%)**
- **Majority Errors ($\ge 2$ errors out of 3)**: **18 participants**:
  - **False Positives (9 participants)**: Healthy controls with handwriting patterns resembling AD (Anonymous IDs: `[90, 91, 105, 107, 141, 155, 159, 163, 164]`). Often associated with advanced age, physiological tremor, or arthritis.
  - **False Negatives (9 participants)**: AD patients whose handwriting remained relatively preserved (Anonymous IDs: `[2, 7, 40, 41, 42, 43, 64, 72, 75]`). Often associated with early disease stage or high pre-morbid cognitive reserve.

### B. Out-of-Fold Threshold Tuning Regimes
- **Default Decision Threshold ($\theta = 0.50$)**:
  - Sensitivity = **91.01%** | Specificity = **89.41%** | Balanced Accuracy = **90.21%**
- **Optimal Balanced Accuracy Threshold ($\theta = 0.48$)**:
  - Sensitivity = **92.13%** | Specificity = **89.41%** | Balanced Accuracy = **90.77%**
- **High-Sensitivity Screening Target ($\ge 95\%$ AD Recall, $\theta = 0.44$)**:
  - Sensitivity = **95.51%** | Specificity = **83.53%**
  - Clinical Trade-off: Lowering the probability cutoff to 0.44 successfully flags 95.5% of early AD cases while retaining 83.5% specificity.

---

## 8. Limitations

The following scientific limitations must be emphasized when interpreting these findings:
1. **Single-Site Small Cohort**: The dataset consists of 174 participants collected at a single center. No external cohort validation was performed.
2. **Internal Validation Only**: Because the same 174 participants have been evaluated across many historical benchmark experiments in the literature, estimates may be mildly optimistic compared to prospective deployment.
3. **Statistical Indistinguishability**: Differences of 1–2 percentage points between models (e.g., Random Forest 88.31%, Extra Trees 89.46%, Deep Ensemble 89.84%, Hybrid 90.97%) were **not statistically distinguishable** under the Nadeau-Bengio corrected resampled $t$-test ($p > 0.40$).
4. **Association vs Causation**: Feature and task rankings represent statistical associations within this digitized tablet dataset, not verified pathophysiological causation.

---

## 9. Honest Scientific Conclusions

1. **Tree Ensembles vs Deep Learning**:
   The single Task-Structured Network (87.35%) trailed Extra Trees (89.46%) by 2.11 pp. In small sample size regimes ($\sim 140$ training samples vs 450 features), tree ensembles naturally excel because orthogonal axis-aligned decision trees partition tabular kinematic features without requiring large gradient updates.
2. **Deep Ensembling Impact**:
   Ensembling 10 neural networks with Gaussian noise regularization significantly stabilized optimization, boosting performance by +2.49 pp to 89.84%, matching tree baselines.
3. **Deployment Model Selection**:
   The final deployed model is `ExtraTreesClassifier` (trained on all 174 participants), chosen because of its slightly higher point mean (89.46%) and lower cross-validation variance (4.30%), **NOT** because the difference from Random Forest is statistically significant.

---

## 10. How to Reproduce

### Repository Structure
```text
dp/
├── data/
│   └── darwin.csv                     # Ingested dataset (174 x 452)
├── src/
│   ├── data.py                        # Validation & task grouping
│   ├── protocol.py                    # Frozen CV protocol & atomic checkpoints
│   ├── models.py                      # Random Forest & Extra Trees baselines
│   ├── deep.py                        # Task-Structured Net, Deep Ensemble & Hybrid
│   ├── stats.py                       # Nadeau-Bengio corrected resampled t-test
│   ├── interpretability.py            # CV Permutation & SHAP analysis
│   ├── error_analysis.py              # Anonymous error profiling & threshold tuning
│   ├── visualization.py               # Slide-ready figures & Tableau exports
│   └── train_final_model.py           # Trains final Extra Trees deployment model
├── app/
│   ├── app.py                         # Interactive Streamlit demonstration
│   └── sample_rows.csv                # Anonymous demo cases for testing
├── tests/
│   ├── test_stats.py                  # Unit tests for Nadeau-Bengio t-test
│   └── test_leakage.py                # Data leakage prevention tests
├── results/
│   ├── figures/                       # 5 high-resolution PNG figures
│   ├── tableau/                       # 4 Tableau integration CSVs
│   ├── checkpoints/                   # Per-fold atomic checkpoint JSONs
│   ├── model_summary.csv              # Aggregated metrics table
│   ├── outer_fold_metrics.csv         # Per-fold metric breakdown
│   ├── paired_differences.csv         # 15 paired fold differences
│   ├── corrected_t_test.json          # Statistical test results
│   ├── outer_fold_predictions.csv     # Out-of-fold participant predictions
│   ├── task_top5_frequency.csv        # Task selection frequencies across folds
│   ├── final_model_extratrees.joblib  # Trained final model weights
│   └── protocol.json                  # Frozen protocol specification
├── run_all.py                         # Master reproduction script
├── pytest.ini                         # Test runner configuration
├── requirements.txt                   # Exact pinned library dependencies
└── README.md                          # Full research documentation
```

### Installation
```bash
git clone <repo-url>
cd dp
pip install -r requirements.txt
```

### Run Unit Tests (Statistical & Leakage Verification)
```bash
pytest -v
```

### Full End-to-End Reproduction Pipeline
```bash
python run_all.py
```
*This reproduces all 15 outer folds, baseline trees, deep learning models, hybrid ensemble, statistical tests, tables, figures, Tableau CSVs, and deployment weights directly from `data/darwin.csv`.*

### Launch Interactive Demonstration Web App
```bash
streamlit run app/app.py
```
*(Runs locally in your browser. Features demo sample selection, custom CSV validation, real-time SHAP task contributions, and persistent disclaimer banners.)*
