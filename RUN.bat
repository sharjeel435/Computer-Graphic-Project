@echo off
title Time-Reversal Physics Simulator
echo ============================================================
echo   TIME-REVERSAL PHYSICS SIMULATOR
echo ============================================================
echo.
echo   Checking dependencies...
python -m pip install -r requirements.txt --quiet
echo   Dependencies verified.
echo.
echo   Starting Launcher...
python main.py
pause
