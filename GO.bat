@echo off
REM Quick startup batch file for Windows
REM Usage: Double-click or run in terminal

echo.
echo =========================================
echo Sign Language Classification - STARTUP
echo =========================================
echo.

REM Check if venv activated
if not defined VIRTUAL_ENV (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
)

echo.
echo Choose action:
echo 1. Collect data (hello)
echo 2. Train model
echo 3. Run app (Streamlit)
echo.

set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    echo Starting data collection for 'hello'...
    python 01_Training_Lab/data_collection/1_collect_seq.py --action_name="hello" --num_samples=30
) else if "%choice%"=="2" (
    echo Starting model training...
    python 01_Training_Lab/model_training/2_train_hybrid.py
) else if "%choice%"=="3" (
    echo Starting Streamlit app...
    streamlit run main.py
) else (
    echo Invalid choice!
)

pause
