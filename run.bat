@echo off
echo ========================================
echo 🦴 FractureDetect - Streamlit App
echo ========================================
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing requirements...
    pip install -r requirements.txt
)

echo.
echo Starting Streamlit app...
echo.
streamlit run streamlit_app.py
pause
