@echo off
title 🌌 Anti-Gravity Virtual Aquarium Engine — All-Time Best Edition
color 0B
cls
echo.
echo  ================================================================
echo    🌌  Anti-Gravity Virtual Aquarium Engine
echo    Version : 2.0 — All-Time Best Edition
echo    Author  : Sharjeel Safdar  ^|  CG Final Project
echo    Engine  : PyOpenGL + GLSL 3.30 + Pygame + NumPy
echo  ================================================================
echo.
echo  [1/3] Checking Python...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo  ERROR: Python not found. Please install Python 3.9+.
    pause
    exit /b 1
)
echo.
echo  [2/3] Installing / verifying dependencies...
pip install -r requirements.txt -q --disable-pip-version-check
if %ERRORLEVEL% NEQ 0 (
    echo  WARNING: Some packages may not have installed correctly.
)
echo.
echo  [3/3] Launching simulation...
echo.
cd /d "%~dp0"
python main.py
echo.
echo  ================================================================
echo    Simulation ended. Thank you for watching!
echo  ================================================================
echo.
pause
