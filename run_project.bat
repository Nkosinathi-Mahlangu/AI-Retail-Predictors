@echo off
echo ============================================================
echo   SmartFuel AI - Intelligent Filling Station System
echo ============================================================
echo.

REM Step 1: Install dependencies
echo [1/3] Installing dependencies...
pip install -r requirements.txt --quiet
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Dependency installation failed.
    pause
    exit /b 1
)
echo       Done.
echo.

REM Step 2: Train the model
echo [2/3] Training demand forecasting models...
python train_model.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Training pipeline failed. Check the output above.
    pause
    exit /b 1
)
echo.

REM Step 3: Launch dashboard
echo [3/3] Launching SmartFuel AI Dashboard...
echo       Open your browser at: http://localhost:8501
echo.
streamlit run app.py

pause
