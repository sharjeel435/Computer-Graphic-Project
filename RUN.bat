@echo off
cd /d "%~dp0"
title Advanced OpenGL Solar System
echo ============================================================
echo   ADVANCED OPENGL 3.3 SOLAR SYSTEM
echo ============================================================
echo.
echo   Checking dependencies...
python -m pip install -r requirements.txt --quiet
echo   Starting simulation...
echo.
python main.py
pause
