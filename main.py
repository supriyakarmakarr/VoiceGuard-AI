r"""
VoiceGuard AI - Main Entry Point
SIH26104: Neural Voice Deepfake & Clone Forensic System

[LATEST UPDATES]:
- core/forensic_pipeline.py: Fixed "Insufficient evidence" on live voice & added multilingual detection
- api/server.py: Added language parameter support to analyze and job endpoints
- script.js: Added dynamic audio sample rate header & language forwarding
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
# Keep audio JIT caches inside the project, including restricted desktop sessions.
os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(PROJECT_ROOT, '.cache', 'numba'))
os.makedirs(os.environ['NUMBA_CACHE_DIR'], exist_ok=True)

import uvicorn

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0" if os.getenv("PORT") else "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print("=" * 70)
    print(">> SIH26104 - VoiceGuard AI Platform")
    print(f"[*] Project Root: {PROJECT_ROOT}")
    print(f"[*] Starting Web Server: http://{host}:{port}")
    print("[*] Press CTRL+C to stop the server.")
    print("=" * 70)
    uvicorn.run("api.server:app", host=host, port=port, workers=1, proxy_headers=True, forwarded_allow_ips=os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1"), reload=os.getenv("VOICEGUARD_DEV_RELOAD", "false").lower() == "true")
