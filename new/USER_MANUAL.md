# DARWIN Research Software Suite v2.0 — User Manual & Operations Guide

## 1. Quick Start Guide

### Option 1: Standalone Web Studio (Zero Installation Required)
1. Double-click `start_web_studio.bat` or open `darwin_studio/index.html` directly in **Google Chrome**, **Microsoft Edge**, **Mozilla Firefox**, or **Apple Safari**.
2. No internet connection or local web server is required. All processing, calculations, interactive replay, and file exports occur 100% locally in your browser.

### Option 2: Python Research Demonstration App (Streamlit)
1. Ensure dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```
2. Double-click `start_streamlit_app.bat` or run:
   ```bash
   streamlit run app/app.py
   ```
3. Open `http://localhost:8501` in your browser.

### Option 3: Command-Line Interface (CLI)
1. Double-click `run_cli_demo.bat` to see an automated end-to-end extraction, quality validation, and model inference run.
2. Or invoke individual commands:
   ```bash
   # Validate raw touch recording
   python -m darwin_analyzer validate -i darwin_studio/sample_data/sample_task01_loops_raw.csv --width 800 --height 600

   # Extract all 18 feature families from raw touch CSV
   python -m darwin_analyzer extract -i darwin_studio/sample_data/sample_task01_loops_raw.csv -o results/task1_features.json -p P_DEMO_01 -t 1

   # Run Extra Trees model inference and SHAP explainability
   python -m darwin_analyzer predict -i app/sample_rows.csv --threshold 0.50
   ```

---

## 2. Navigating the Web Studio

### Tab 1: ✍ 25 Handwriting Tasks & Acquisition
1. **Configure Session**: Enter pseudonymous Participant ID (e.g. `P_2026_A01`) and ensure informed consent is checked.
2. **Select Task**: Choose any task from the top horizontal strip (Tasks 1 to 25 organized into Graphic, Copying, and Memory/Dictation).
3. **Calibrate Screen**: Click **⚙ Screen Calibration** in the header. Use a standard credit card (85.6 mm width) to set exact pixels per millimetre.
4. **Recording**:
   - Click **▶ Start Task**.
   - Write or trace the stimulus on the canvas using your finger (or stylus).
   - Accidental mouse clicks are automatically rejected if *Finger-Touch Only* mode is active.
   - Click **✓ Finish & Compute Features** when done.
   - Telemetry cards update immediately with total points, strokes, active contact time, sampling Hz, and quality status.

### Tab 2: 🔍 Stroke Review & Replay
1. Select any completed task from the dropdown.
2. View the segmented strokes (each stroke rendered in an individual high-contrast color).
3. **Interactive Replay Controls**:
   - Click **▶ Play** to watch the writing process re-enacted at real-time speed.
   - Adjust speed: `0.25x`, `0.5x`, `1.0x`, `2.0x`, `5.0x`.
   - Use the time scrubber to navigate to any millisecond.
4. **Stroke Inspection Table**: Click any row in the stroke table to highlight that stroke on the canvas and view its duration, path length, net displacement, straightness ratio, and mean speed.
5. **Researcher Observations**: Type qualitative notes (e.g., motor hesitations, tremor, pen lifts) and click **Save Annotations to Session**.

### Tab 3: 📊 Extracted Features (18 Canonical Families)
1. Select any task to inspect its full mathematical feature table.
2. Every feature displays its operational description, touchscreen measurability status (`Directly Measurable`, `Adapted Definition`, or `Excluded`), extracted value, and units (px, ms, px/s, px/s², px/s³).

### Tab 4: 🛡 Quality & Sensor Integrity Audit
1. Displays the composite session quality score (0% to 100%) and validation state (`VALID`, `REVIEW_REQUIRED`, `INVALID`).
2. The 14-point checklist audits coordinate finiteness, timestamp monotonicity, pointer type compliance, sampling gaps, and bounds enforcement.

### Tab 5: 🧠 Machine Learning & SHAP Explainability
1. Load sample patient profiles (**Demo AD Case A**, **Demo AD Case B**, or **Demo Healthy Case C**).
2. Inspect the predicted Early AD probability gauge and adjust the decision threshold slider (default 50%).
3. Inspect the top contributing tasks via SHAP attribution (Tasks 17, 23, 7, 19, 9 prominence).
4. Review the clinical modality breakdown: Timing (37.1%), Kinematics & GMRT (36.3%), Spatial (13.3%), Pressure (13.3%).

### Tab 6: 📋 Researcher Dashboard
1. Review study-level KPIs: Enrolled Participants, Total Sessions, Tasks Completed, Quality Pass Rate.
2. Inspect the 5x5 Task Matrix displaying the completion state of all 25 tasks.
3. Search and filter participant session logs.

### Tab 7: 💾 Export Research Datasets
1. **Raw Touch-Events Dataset** (`raw_points.csv`): Point-by-point coordinates, timestamps, pointer ID, and pressure metadata.
2. **Stroke-Level Metrics** (`stroke_metrics.csv`): Metrics for each stroke contact episode.
3. **Task-Level Features** (`task_features.csv`): Long-format feature table across completed tasks.
4. **Participant 450-Feature Wide Matrix** (`darwin_450_matrix.csv`): Standard wide matrix (18 features x 25 tasks) directly compatible with machine learning models.
5. **Data Quality Audit JSON** (`quality_audit.json`): Machine-readable audit report.
6. **Import & Analyze CSV**: Drag and drop any raw touch CSV or 450-feature CSV to analyze immediately.

### Tab 8: 📄 Printable Research Report
1. View a publication-grade HTML research summary.
2. Click **🖨 Open & Print PDF Report** to save a reproducible PDF via the browser's print dialog.

---

## 3. Data Privacy & Ethical Compliance
- **Zero Cloud Transmission**: All calculations, audio/visual rendering, and data storage execute strictly within your local machine.
- **Anonymization**: Do not record patient names, birthdates, or direct identifiers in the Participant ID field. Use pseudonymous alphanumeric codes (e.g., `P_2026_001`).
- **IRB Consent Check**: The software enforces an informed consent confirmation flag on every session record.
- **Audit Traceability**: Every export includes hardware platform metadata, screen resolution, timestamps, and protocol versions.
