@echo off
REM Quick start script for downloading GPT-2 weights

echo.
echo ============================================
echo   GPT-2 Weight Downloader
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo Step 1: Installing required Python packages...
echo (This may take a few minutes)
echo.

python -m pip install --upgrade pip
python -m pip install transformers torch numpy

echo.
echo Step 2: Downloading GPT-2 from HuggingFace...
echo.

python download_weights.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================
    echo   SUCCESS! Weights downloaded
    echo ============================================
    echo.
    echo Next steps:
    echo   1. Run: build.bat
    echo   2. See WEIGHT_LOADING.md for details
    echo.
) else (
    echo Download failed!
    pause
    exit /b 1
)
