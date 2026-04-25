@echo off
setlocal
cd /d "%~dp0"
title Advanced OpenGL Solar System

echo ============================================================
echo   ADVANCED OPENGL 3.3 SOLAR SYSTEM
echo ============================================================
echo.

set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if not defined PYTHON_EXE (
    where py >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=py -3"
)
if not defined PYTHON_EXE (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    echo [error] Python was not found.
    echo Install Python 3.9+ and try again.
    pause
    exit /b 1
)

echo   Using Python: %PYTHON_EXE%
echo   Checking dependencies...
call %PYTHON_EXE% -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [error] Dependency installation failed.
    pause
    exit /b 1
)

echo.
echo   Running startup check...
call %PYTHON_EXE% -c "import main; main.require_assets(); print('Assets OK'); print('OpenGL init...'); version = main.init_window(); print('OpenGL', version); import pygame; pygame.quit()"
if errorlevel 1 (
    echo.
    echo [error] Startup check failed.
    echo This usually means OpenGL 3.3 is unavailable or a dependency is broken.
    pause
    exit /b 1
)

echo.
echo   Starting simulation...
echo.
call %PYTHON_EXE% main.py

if errorlevel 1 (
    echo.
    echo [error] The program exited with an error.
)

pause
