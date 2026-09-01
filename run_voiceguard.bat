@echo off
title SIH26104 - VoiceGuard AI Platform Launcher
color 0B

:: Ensure terminal working directory is the script directory
cd /d "%~dp0"

echo ======================================================================
echo           VOICEGUARD AI - NEURAL VOICE DEEPFAKE FORENSIC SYSTEM
echo                      SIH Hackathon 2024 / 2025 / 2026
echo ======================================================================
echo.
echo [*] Project Location: %CD%
echo.

:: Check if models exist
if not exist "models\baseline_rf.pkl" (
    echo [!] Pre-trained models not found. Initiating Dataset Generation ^& Training Pipeline...
    python scripts\train_models.py
    if errorlevel 1 (
        echo [X] Training failed. Please check Python dependencies.
        pause
        exit /b 1
    )
) else (
    echo [OK] Pre-trained AI models verified.
)

echo.
echo [*] Starting VoiceGuard AI Server on http://127.0.0.1:8000 ...
echo [*] Open http://127.0.0.1:8000 in your browser.
echo.

python main.py

pause
