@echo off
setlocal
cd /d "%~dp0"
title VoiceGuard AI - Voice Forensics
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" main.py
) else (
  python main.py
)
if errorlevel 1 (
  echo.
  echo VoiceGuard could not start. Install Python 3.12 and follow README.md.
  echo Setup: python -m venv .venv
  echo Then: .venv\Scripts\python -m pip install -r requirements.txt
)
pause
