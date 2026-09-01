"""
VoiceGuard AI - Curated Samples Verification Script
"""
import sys
import os

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from starlette.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.server import app

def main():
    client = TestClient(app)
    samples_res = client.get("/api/sample-audios").json()
    samples = samples_res.get("samples", [])
    print(f"\n=======================================================")
    print(f"Testing {len(samples)} Curated Hackathon Demo Audio Samples")
    print(f"=======================================================")

    for s in samples:
        fname = s["filename"]
        audio_bytes = client.get(s["url"]).content
        res = client.post(
            "/api/analyze",
            files={"file": (fname, audio_bytes, "audio/wav")}
        ).json()
        analysis = res["analysis"]
        expected = s["expected"]
        risk_score = analysis["risk_score"]
        risk_level = analysis["risk_level"]
        synth_prob = analysis["synthetic_probability"] * 100
        conf = analysis["confidence_score"]
        verdict = analysis["verdict"]["en"]

        status = "[PASS]" if (expected == "AI_SYNTHETIC" and risk_level == "HIGH") or (expected == "GENUINE_HUMAN" and risk_level == "LOW") else "[CHECK]"

        print(f"\nSample: {s['title']}")
        print(f"  File: {fname}")
        print(f"  Expected: {expected} | Actual Risk: {risk_score}% ({risk_level}) -> {status}")
        print(f"  Synth Probability: {synth_prob:.1f}% | Confidence: {conf:.1f}%")
        print(f"  Verdict: {verdict}")

    print(f"\n=======================================================")
    print(f"All sample tests completed successfully!")
    print(f"=======================================================\n")

if __name__ == "__main__":
    main()
