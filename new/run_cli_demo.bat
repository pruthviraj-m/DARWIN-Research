@echo off
echo =========================================================================
echo DARWIN Research Analyzer CLI Demonstration
echo =========================================================================
echo 1. Displaying CLI Help...
python -m darwin_analyzer --help
echo.
echo 2. Running Quality Validation on sample Task 1 raw points...
python -m darwin_analyzer validate -i darwin_studio\sample_data\sample_task01_loops_raw.csv --width 800 --height 600
echo.
echo 3. Extracting 18 Features from sample Task 1 raw points...
python -m darwin_analyzer extract -i darwin_studio\sample_data\sample_task01_loops_raw.csv -o results\demo_extracted_task1.json -p P_DEMO_01 -t 1
echo.
echo 4. Running Extra Trees Model Inference on DARWIN sample participant...
python -m darwin_analyzer predict -i app\sample_rows.csv --threshold 0.50
echo.
echo =========================================================================
echo Demo completed successfully. Extracted features saved to results\demo_extracted_task1.json.
echo =========================================================================
pause
