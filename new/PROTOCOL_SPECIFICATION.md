# DARWIN Research Software Suite v2.0 — Protocol Specification

## 1. Executive Summary & Purpose
The DARWIN Handwriting Research Software Suite provides a rigorous, end-to-end framework for acquiring, validating, extracting, and interpreting online handwriting dynamics in cognitive research.

Unlike simplistic digital drawing pads that immediately produce unverified clinical predictions, this platform:
1. **Preserves pristine raw sensor streams** (touch coordinates, sub-millisecond timestamps, stroke contact episodes).
2. **Performs automated multi-point data quality and sensor integrity audits** before accepting any recording.
3. **Calculates 18 canonical DARWIN feature families** using local cubic polynomial least-squares derivatives.
4. **Enforces a strict dual-pipeline architecture** distinguishing digitizer tablet pen handwriting (Wacom protocol) from capacitive touchscreen finger handwriting.
5. **Provides real-time explainability (SHAP attribution)** and task-level psychomotor decomposition.

---

## 2. Standardized 25-Task Battery Taxonomy
The 25 standardized handwriting tasks are organized into three clinical psychomotor categories as defined in Cilia et al. (2018, 2022) and Kobayashi et al. (2022):

### Category 1: Graphic Tasks (Tasks 1 – 10)
| Task ID | Task Name | Description & Prompt | Primary Psychomotor Function |
|---|---|---|---|
| **Task 01** | Continuous Cursive Loops | Write continuous loops 'lelele...' without lifting finger | Fine motor rhythm, continuous cursive chaining |
| **Task 02** | Continuous Zigzags | Draw continuous sharp zigzags across canvas | Rapid direction reversals, angular velocity control |
| **Task 03** | Archimedean Spiral | Draw continuous expanding spiral from center outward | Tremor detection, continuous radius modulation |
| **Task 04** | Triangular Peaks & Valleys | Alternating high and low triangular peaks | Spatial amplitude modulation, rhythm stability |
| **Task 05** | Geometric Dot Connections | Connect geometric vertices with straight lines | Ballistic motor control, spatial aiming |
| **Task 06** | Rapid Overlapping Loops | Draw rapid overlapping loops continuously | Cursive motor fluency, wrist movement coordination |
| **Task 07** | Alternating Pen Taps | Make 7 discrete, rapid deliberate taps | Inter-tap latency, in-air flight timing |
| **Task 08** | Sinusoidal Waves | Draw a smooth continuous sine wave | Harmonic oscillation, velocity regulation |
| **Task 09** | Square Boundary Tracing | Trace a square boundary with 4 orthogonal strokes | Orthogonal stroke transitions, spatial constraints |
| **Task 10** | Figure-8 (Infinity Loop) | Trace infinity loops 3 times without stopping | Continuous circular flow, bimanual coordination |

### Category 2: Copying Tasks (Tasks 11 – 18)
| Task ID | Task Name | Description & Prompt | Primary Psychomotor Function |
|---|---|---|---|
| **Task 11** | Copy Capital Letters | Copy uppercase letters: A B C D E | Orthographic retrieval, discrete stroke assembly |
| **Task 12** | Copy Lowercase Letters | Copy lowercase cursive letters: a m r e l | Cursive letter construction, fine ligature timing |
| **Task 13** | Copy Simple Word | Copy high-frequency word: 'casa' | Automated lexical-motor execution |
| **Task 14** | Copy Complex Word | Copy multi-syllable word: 'quaderno' | Phonological retrieval, multi-letter sequencing |
| **Task 15** | Copy Short Sentence | Copy short sentence: 'La casa e bella' | Inter-word spatial arrangement, sentence pacing |
| **Task 16** | Copy DARWIN Sentence | Standard: 'Il gatto beve il latte nella tazza' | Standardized syntactic copying benchmark |
| **Task 17** | Complex Loop Tracing | Complex geometric loop pattern tracing | **Highest Diagnostic Weight (39.2 mSHAP)**: fine motor coordination |
| **Task 18** | Copy Pseudo-Word | Copy non-word: 'morfina' / 'palfido' | Phonological decoding, non-lexical motor execution |

### Category 3: Memory and Dictation Tasks (Tasks 19 – 25)
| Task ID | Task Name | Description & Prompt | Primary Psychomotor Function |
|---|---|---|---|
| **Task 19** | Cursive Chaining | Continuous cursive chaining ('ononon...') | Continuous stroke ligatures without segmentation |
| **Task 20** | Free Sentence Writing | Write one sentence about today's weather from memory | Spontaneous cognitive composition, executive planning |
| **Task 21** | Clock Drawing Test (CDT) | Draw clock face, numbers 1-12, set time to 11:10 | Visuospatial memory recall, executive function |
| **Task 22** | Delayed Word Recall | Recall and write 3 target words presented earlier | Short-term episodic memory retrieval |
| **Task 23** | Sentence Under Load | Complex sentence written from dictation | **Top Diagnostic Weight (38.8 mSHAP)**: dual-task cognitive load |
| **Task 24** | Reverse Counting | Write numbers counting down from 20 to 10 | Working memory manipulation, numerical motor patterns |
| **Task 25** | Habitual Signature | Sign natural signature 3 times | Overlearned automated motor pattern fluidity |

---

## 3. The 18 Canonical Feature Families & Touchscreen Adaptation Status

| No. | Feature Name | Description | Touchscreen Status | Mathematical Formulation |
|---|---|---|---|---|
| **1** | `total_time` | Total task duration | Directly Measurable | $T_{\text{last}} - T_{\text{first}}$ (ms) |
| **2** | `air_time` | Pen-up / inter-stroke latency | Adapted Definition | $\sum \Delta t_{\text{interstroke}}$ (ms) |
| **3** | `paper_time` | Active surface contact time | Directly Measurable | $\sum (t_{\text{end}, i} - t_{\text{start}, i})$ (ms) |
| **4** | `mean_speed_on_paper` | Contact trajectory speed | Directly Measurable | $\frac{1}{N}\sum \frac{\sqrt{\Delta x^2 + \Delta y^2}}{\Delta t}$ (px/s or mm/s) |
| **5** | `mean_speed_in_air` | Lifted pen flight velocity | **Excluded (Touchscreen)** | Unmeasurable on capacitive touchscreen |
| **6** | `mean_acc_on_paper` | Contact acceleration magnitude | Directly Measurable | Local 9-point cubic least-squares 2nd derivative $\sqrt{x''^2 + y''^2}$ |
| **7** | `mean_acc_in_air` | Lifted pen flight acceleration | **Excluded (Touchscreen)** | Unmeasurable |
| **8** | `mean_jerk_on_paper` | Contact jerk magnitude | Directly Measurable | Local 9-point cubic least-squares 3rd derivative $\sqrt{x'''^2 + y'''^2}$ |
| **9** | `mean_jerk_in_air` | Lifted pen flight jerk | **Excluded (Touchscreen)** | Unmeasurable |
| **10** | `pressure_mean` | Average physical pen force | **Excluded (Uncalibrated)** | Touchscreen browser pressure is uncalibrated metadata |
| **11** | `pressure_var` | Variance in pen force | **Excluded (Uncalibrated)** | Excluded from primary feature vector |
| **12** | `gmrt_on_paper` | Geometric Measure of Relief Texture | Adapted Definition | Speed irregularity index: $\frac{\sum \|\Delta v\|}{L_{\text{path}}}$ |
| **13** | `gmrt_in_air` | In-air GMRT roughness | **Excluded (Touchscreen)** | Unmeasurable |
| **14** | `mean_gmrt` | Combined kinematic smoothness | Adapted Definition | On-paper speed fluctuation metric |
| **15** | `num_of_pendown` | Number of contact episodes | Directly Measurable | Total stroke count |
| **16** | `max_x_extension` | Horizontal handwriting span | Directly Measurable | $\max(X) - \min(X)$ |
| **17** | `max_y_extension` | Vertical handwriting span | Directly Measurable | $\max(Y) - \min(Y)$ |
| **18** | `disp_index` | Spatial distribution index | Directly Measurable | $\frac{\sum \text{NetDisplacement}_i}{L_{\text{total}}}$ |

---

## 4. Automated 14-Point Data Quality Audit Rules
Every recorded task undergoes automated verification:
1. **Q01**: Non-empty recording ($\ge 10$ points).
2. **Q02**: Coordinate finiteness (no NaNs or non-numeric values).
3. **Q03**: Strict timestamp monotonicity ($\Delta t \ge 0$).
4. **Q04**: Input modality compliance (rejection of mouse clicks in finger-only protocol).
5. **Q05**: Canvas boundary compliance ($0 \le x \le W, 0 \le y \le H$).
6. **Q06**: Sampling gap dropouts ($\Delta t > 33.3\text{ ms}$ and $> 50\text{ ms}$).
7. **Q07**: Duplicate point keys ($(stroke, point)$ uniqueness).
8. **Q08**: Single-point stroke detection (flagged as non-velocity points).
9. **Q09**: Zero-duration stroke detection.
10. **Q10**: Multi-touch contact detection (accidental palm touches).
11. **Q11**: Touch cancellation event tracking (`pointercancel`).
12. **Q12**: Physical coordinate calibration verification (plausible px/mm range).
13. **Q13**: Cross-device domain mismatch guard (Wacom vs Touchscreen).
14. **Q14**: IRB informed consent metadata completeness.

---

## 5. Dual-Pipeline Architecture & Clinical Guardrails
```
┌─────────────────────────────────────────────────────────────┐
│                 DARWIN INPUT DATA STREAM                    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                Is input from Wacom Stylus Tablet
                   with physical pressure & hover?
                               │
               ┌───────────────┴───────────────┐
              YES                              NO
               │                               │
┌──────────────▼──────────────┐ ┌──────────────▼──────────────┐
│         PIPELINE A:         │ │         PIPELINE B:         │
│     Original DARWIN UCI     │ │    New Touchscreen Study    │
│    450-Feature Protocol     │ │   Operational Adaptations   │
├─────────────────────────────┤ ├─────────────────────────────┤
│ • Full 18 features x 25     │ │ • Contact kinematics only   │
│ • Physical tip pressure     │ │ • Local polynomial deriva-  │
│ • True in-air hover flight  │ │   tives (acc, jerk, speed)  │
│ • Extra Trees Deployment    │ │ • Inter-stroke gap latency  │
│   (89.46% Nested CV Acc)    │ │ • Pressure & In-Air EXCLUDED│
│ • SHAP Feature Importance   │ │ • REQUIRES CROSS-DEVICE     │
│   (Tasks 17, 23 prominence) │ │   CALIBRATION & VALIDATION  │
└─────────────────────────────┘ └─────────────────────────────┘
```

> [!WARNING]
> Under no circumstances should uncalibrated finger-touchscreen data be silently fed into Pipeline A without explicit researcher authorization and prominent domain-shift warnings.
