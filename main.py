r"""
VoiceGuard AI - Main Entry Point
SIH26104: Neural Voice Deepfake & Clone Forensic System

Can be run from anywhere:
    python main.py
or:
    python C:/Users/SUPRIYA/.gemini/antigravity/scratch/sih26104-voiceguard-ai/main.py
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Automatically resolve and set project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Change working directory to project root
os.chdir(PROJECT_ROOT)

import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print(">> SIH26104 - VoiceGuard AI Platform")
    print(f"[*] Project Root: {PROJECT_ROOT}")
    print("[*] Starting Web Server: http://127.0.0.1:8000")
    print("[*] Press CTRL+C to stop the server.")
    print("=" * 70)
    uvicorn.run("api.server:app", host="127.0.0.1", port=8000, reload=True)
