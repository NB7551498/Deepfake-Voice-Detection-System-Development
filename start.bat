@echo off
title Deepfake Voice Detector 2.0
echo ============================================================
echo   Deepfake Voice Detector 2.0 - Starting Production Engine
echo ============================================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set PY_CMD=python
) else (
    if exist "C:\ProgramData\anaconda3\python.exe" (
        set PY_CMD=C:\ProgramData\anaconda3\python.exe
    ) else (
        echo [ERROR] Python not found. Please install Python or Anaconda.
        pause
        exit /b 1
    )
)

echo [1/2] Launching Web Dashboard in your browser...
start "" http://127.0.0.1:8001/

echo [2/2] Starting Deepfake Voice Detector 2.0 API Server on port 8001...
echo Dashboard URL: http://127.0.0.1:8001/
echo API Docs:      http://127.0.0.1:8001/docs
echo Press Ctrl+C to stop the server.
echo.
%PY_CMD% src/api/real_api.py
pause
