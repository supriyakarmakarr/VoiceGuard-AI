"""
VoiceGuard AI - FastAPI Forensic Server & Enterprise Production Engine
Provides REST, Chunk-Streaming, and WebSocket APIs for:
1. Audio Deepfake Forensic Detection & Explainability
2. Probability Calibration (Platt Scaling & Logistic Meta-Learner)
3. Biometric Speaker Verification & Voiceprint Enrollment (CEO Fraud Defense)
4. Out-of-Distribution Telephony Codec Simulation (G.711 μ-law, AMR-NB, Noise)
5. Live Carrier VoIP / SIP Media Stream Hook (Twilio & Asterisk WebSocket)
6. Enterprise API Security & Sliding-Window Rate Limiting
"""

import os
import sys
import io
import time
import base64
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, WebSocket, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from typing import Optional

# Ensure root is in Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.audio_preprocessor import AudioPreprocessor
from core.feature_extractor import FeatureExtractor
from core.baseline_model import BaselineVoiceClassifier
from core.deep_learning_model import DeepLearningVoiceClassifier
from core.risk_engine import RiskEngine
from core.calibrator import LearnedRiskCalibrator
from core.speaker_verifier import SpeakerVerifier
from core.telephony_degradation import TelephonyDegradationPipeline
from api.security import enforce_security_and_rate_limit
from api.telephony import handle_telephony_websocket

# Paths
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
CURATED_DIR = os.path.join(DATASET_DIR, "curated_samples")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "web") if os.path.isdir(os.path.join(PROJECT_ROOT, "web")) else PROJECT_ROOT

# Global instances
preprocessor = AudioPreprocessor()
feature_extractor = FeatureExtractor()
telephony_degrader = TelephonyDegradationPipeline()
speaker_verifier = SpeakerVerifier()
calibrator = None
risk_engine = RiskEngine()

baseline_model = None
deep_model = None
model_metadata = {}


def load_models():
    """Initializes and loads trained ML, Deep Learning, and Calibration models."""
    global baseline_model, deep_model, calibrator, risk_engine, model_metadata

    baseline_path = os.path.join(MODELS_DIR, "baseline_rf.pkl")
    deep_path = os.path.join(MODELS_DIR, "deep_cnn.pt")
    deep_quantized_path = os.path.join(MODELS_DIR, "deep_cnn_quantized.pt")
    calibrator_path = os.path.join(MODELS_DIR, "risk_calibrator.pkl")
    meta_path = os.path.join(MODELS_DIR, "model_meta.json")

    # 1. Load baseline ensemble
    if os.path.exists(baseline_path):
        try:
            baseline_model = BaselineVoiceClassifier(baseline_path)
            print(f"[API] Loaded Baseline ML model from {baseline_path}")
        except Exception as e:
            print(f"[API] Warning loading baseline model: {e}")

    # 2. Load deep learning model (prefer quantized INT8 on CPU if available)
    load_path = deep_quantized_path if os.path.exists(deep_quantized_path) else deep_path
    if os.path.exists(load_path):
        try:
            deep_model = DeepLearningVoiceClassifier(load_path)
            print(f"[API] Loaded Deep CNN model from {load_path} (Quantized: {os.path.exists(deep_quantized_path)})")
        except Exception as e:
            print(f"[API] Warning loading deep learning model: {e}")

    # 3. Load learned risk calibrator
    if os.path.exists(calibrator_path):
        try:
            calibrator = LearnedRiskCalibrator(calibrator_path)
            risk_engine = RiskEngine(calibrator=calibrator)
            print(f"[API] Loaded Learned Risk Calibrator from {calibrator_path}")
        except Exception as e:
            print(f"[API] Warning loading calibrator: {e}")

    # 4. Load metadata
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                model_metadata = json.load(f)
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    speaker_verifier.load_all_profiles()
    yield


app = FastAPI(
    title="VoiceGuard AI - Enterprise Deepfake & Voiceprint Forensic API",
    description="Production-Ready Audio Deepfake Detection, Biometric Speaker Verification & Carrier Telephony Hook",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_spectrogram_base64(audio: np.ndarray, sr: int = 16000) -> str:
    """Generates a high-contrast forensic Mel-Spectrogram image encoded as base64 PNG."""
    try:
        fig, ax = plt.subplots(figsize=(6, 2.5), dpi=100)
        fig.patch.set_facecolor("#0b0f19")
        ax.set_facecolor("#0b0f19")

        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        ax.imshow(
            mel_db,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[0, len(audio) / sr, 0, sr // 2],
        )
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        b64_str = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"
    except Exception as e:
        print(f"Spectrogram render error: {e}")
        return ""


# ==========================================
# CORE FORENSIC REST ENDPOINTS
# ==========================================

@app.get("/api/health")
def health_check():
    """Health status, model load verification, and system readiness."""
    return {
        "status": "healthy",
        "service": "SIH26104 - VoiceGuard AI Production Engine",
        "version": "2.0.0",
        "baseline_model_loaded": baseline_model is not None,
        "deep_model_loaded": deep_model is not None,
        "calibrator_loaded": calibrator is not None and getattr(calibrator, "is_fitted", False),
        "quantized_cpu_active": os.path.exists(os.path.join(MODELS_DIR, "deep_cnn_quantized.pt")),
        "enrolled_speakers_count": len(speaker_verifier.enrolled_profiles),
        "device": str(deep_model.device) if deep_model else "cpu",
        "timestamp": time.time(),
    }


@app.get("/api/models")
def get_models_info():
    """Returns model benchmark accuracy, metrics, architecture, and calibration details."""
    lat_report_path = os.path.join(MODELS_DIR, "benchmark_latency.json")
    stress_report_path = os.path.join(MODELS_DIR, "stress_test_report.json")

    latency_bench = {}
    stress_bench = {}
    if os.path.exists(lat_report_path):
        try:
            with open(lat_report_path, "r", encoding="utf-8") as f:
                latency_bench = json.load(f)
        except Exception:
            pass

    if os.path.exists(stress_report_path):
        try:
            with open(stress_report_path, "r", encoding="utf-8") as f:
                stress_bench = json.load(f)
        except Exception:
            pass

    return {
        "metadata": model_metadata,
        "baseline_metrics": baseline_model.metrics if baseline_model else {},
        "deep_metrics": deep_model.metrics if deep_model else {},
        "calibration_metrics": calibrator.metrics if calibrator else {},
        "latency_benchmark": latency_bench,
        "stress_test_summary": stress_bench,
        "enrolled_profiles": speaker_verifier.list_profiles(),
        "feature_count": len(baseline_model.feature_names) if baseline_model else 0,
    }


@app.get("/api/sample-audios")
def list_sample_audios():
    """Returns curated demo audio files for 1-click hackathon and enterprise testing."""
    os.makedirs(CURATED_DIR, exist_ok=True)
    files = [f for f in os.listdir(CURATED_DIR) if f.lower().endswith(".wav")]
    samples = []

    descriptions = {
        "sample_1_real_human_voice.wav": {
            "title": "Natural Human Speech 1 (CEO / Vikram)",
            "expected": "GENUINE_HUMAN",
            "category": "Authentic",
            "badge": "Enrolled CEO Voice",
            "speaker_id": "ceo_vikram",
        },
        "sample_2_real_female_voice.wav": {
            "title": "Natural Human Speech 2 (CFO / Ananya)",
            "expected": "GENUINE_HUMAN",
            "category": "Authentic",
            "badge": "Enrolled CFO Voice",
            "speaker_id": "cfo_ananya",
        },
        "sample_3_ai_cloned_voice.wav": {
            "title": "AI Voice Clone (ElevenLabs style)",
            "expected": "AI_SYNTHETIC",
            "category": "Voice Clone",
            "badge": "High Risk Clone",
            "speaker_id": "ceo_vikram",  # targeting CEO!
        },
        "sample_4_neural_tts_deepfake.wav": {
            "title": "Neural TTS Deepfake (Tacotron vocoder)",
            "expected": "AI_SYNTHETIC",
            "category": "Synthetic TTS",
            "badge": "High Risk Synth",
        },
        "sample_5_robotic_voice_scam.wav": {
            "title": "Voice Scam Impersonator (Robotic artifacts)",
            "expected": "AI_SYNTHETIC",
            "category": "Voice Scam",
            "badge": "High Risk Scam",
        },
    }

    for f in files:
        meta = descriptions.get(f, {
            "title": f.replace("_", " ").replace(".wav", "").title(),
            "expected": "UNKNOWN",
            "category": "Demo Sample",
            "badge": "Sample",
        })
        samples.append({
            "filename": f,
            "url": f"/api/sample-audio/{f}",
            **meta,
        })
    return {"samples": samples}


@app.get("/api/sample-audio/{filename}")
def get_sample_audio_file(filename: str):
    """Serves the WAV audio file for playback."""
    file_path = os.path.join(CURATED_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Sample audio file not found.")
    return FileResponse(file_path, media_type="audio/wav")


@app.post("/api/analyze")
async def analyze_audio(
    request: Request,
    file: UploadFile = File(...),
    threshold_low: float = Form(30.0),
    threshold_high: float = Form(70.0),
    claimed_speaker_id: Optional[str] = Form(None),
    simulate_codec: Optional[str] = Form(None),  # 'none', 'g711_mulaw', 'amr_nb', 'babble_noise'
    _auth = Depends(enforce_security_and_rate_limit),
):
    """
    Main multi-factor forensic analysis endpoint:
    Preprocessing -> [Optional Telephony Codec Degradation] -> Feature Extraction
    -> Dual Models (Spectrogram CNN + Baseline RF/GB) -> Calibrated Risk Engine
    -> [Optional Speaker Biometric Verification for CEO-Fraud Defense].
    """
    if baseline_model is None or deep_model is None:
        load_models()

    start_time = time.time()
    try:
        audio_bytes = await file.read()
        if len(audio_bytes) < 100:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty or corrupted.")

        # Preprocessing
        proc = preprocessor.process(audio_bytes)
        audio = proc["audio"]
        raw_trimmed = proc["raw_trimmed"]
        duration = proc["duration_sec"]

        # Apply Telephony Simulation if requested
        if simulate_codec and simulate_codec != "none":
            if simulate_codec == "g711_mulaw":
                audio = telephony_degrader.encode_decode_g711_mulaw(audio)
                raw_trimmed = telephony_degrader.encode_decode_g711_mulaw(raw_trimmed)
            elif simulate_codec == "amr_nb":
                audio = telephony_degrader.apply_amr_narrowband_filter(audio)
                raw_trimmed = telephony_degrader.apply_amr_narrowband_filter(raw_trimmed)
            elif simulate_codec == "babble_noise":
                audio = telephony_degrader.inject_environmental_noise(audio, snr_db=12.0)
                raw_trimmed = telephony_degrader.inject_environmental_noise(raw_trimmed, snr_db=12.0)
            elif simulate_codec == "full_phone_call":
                audio = telephony_degrader.apply_realistic_phone_call_pipeline(audio)
                raw_trimmed = telephony_degrader.apply_realistic_phone_call_pipeline(raw_trimmed)

        # Run Baseline Model Inference
        if baseline_model:
            base_result = baseline_model.predict(audio)
        else:
            base_result = {"synthetic_probability": 0.5, "genuine_probability": 0.5, "confidence": 0.5}

        # Run Deep Learning Spectrogram CNN Inference
        if deep_model:
            deep_result = deep_model.predict(audio)
        else:
            deep_result = {"synthetic_probability": 0.5, "genuine_probability": 0.5, "confidence": 0.5}

        # Compute Granular Forensic Acoustic Signals
        forensic_signals = feature_extractor.compute_forensic_signals(raw_trimmed)

        # Risk Engine Multi-Factor Assessment (incorporating Learned Platt Calibrator)
        custom_risk_engine = RiskEngine(
            low_threshold=threshold_low,
            high_threshold=threshold_high,
            calibrator=calibrator,
        )
        risk_result = custom_risk_engine.evaluate(
            deep_result=deep_result,
            baseline_result=base_result,
            forensic_signals=forensic_signals,
            audio_duration=duration,
        )

        # Optional Biometric Speaker Verification (CEO-Fraud Check)
        dual_factor_result = None
        if claimed_speaker_id and claimed_speaker_id.strip():
            dual_factor_result = speaker_verifier.evaluate_dual_factor_transaction(
                deepfake_risk_result=risk_result,
                claimed_speaker_id=claimed_speaker_id.strip(),
                audio_input=raw_trimmed,
            )

        # Generate Forensic Spectrogram Base64 Image
        spectrogram_b64 = generate_spectrogram_base64(raw_trimmed, sr=preprocessor.target_sr)

        # Waveform preview points (for fast rendering on frontend canvas)
        num_preview_pts = 80
        step = max(1, len(raw_trimmed) // num_preview_pts)
        waveform_pts = [round(float(np.max(np.abs(raw_trimmed[i : i + step]))), 3) for i in range(0, len(raw_trimmed), step)][:num_preview_pts]

        latency_ms = round((time.time() - start_time) * 1000, 1)

        response_payload = {
            "success": True,
            "filename": file.filename,
            "duration_seconds": round(duration, 2),
            "sample_rate_hz": preprocessor.target_sr,
            "latency_ms": latency_ms,
            "codec_simulation_applied": simulate_codec or "none",
            "analysis": risk_result,
            "deepfake_analysis": risk_result,
            "speaker_verification": dual_factor_result,
            "models_raw": {
                "deep_cnn": deep_result,
                "baseline_rf": base_result,
            },
            "spectrogram_image": spectrogram_b64,
            "waveform_preview": waveform_pts,
            "timestamp": time.time(),
        }

        # Inject rate limit headers
        headers = {
            "X-RateLimit-Limit": str(getattr(request.state, "rate_limit_limit", 60)),
            "X-RateLimit-Remaining": str(getattr(request.state, "rate_limit_remaining", 59)),
        }
        return JSONResponse(content=response_payload, headers=headers)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/analyze-chunk")
async def analyze_stream_chunk(
    request: Request,
    file: UploadFile = File(...),
    chunk_index: int = Form(0),
    _auth = Depends(enforce_security_and_rate_limit),
):
    """
    Lightweight streaming chunk endpoint for real-time microphone monitoring.
    Processes small 2-3 second audio chunks with low-latency response.
    """
    start_time = time.time()
    if baseline_model is None or deep_model is None:
        load_models()

    try:
        audio_bytes = await file.read()
        proc = preprocessor.process(audio_bytes, target_duration=3.0)
        audio = proc["audio"]
        raw_trimmed = proc["raw_trimmed"]
        duration = proc["duration_sec"]

        base_result = baseline_model.predict(audio) if baseline_model else {"synthetic_probability": 0.5}
        deep_result = deep_model.predict(audio) if deep_model else {"synthetic_probability": 0.5}
        forensic_signals = feature_extractor.compute_forensic_signals(raw_trimmed)

        risk_result = risk_engine.evaluate(deep_result, base_result, forensic_signals, audio_duration=duration)
        latency_ms = round((time.time() - start_time) * 1000, 1)

        return {
            "chunk_index": chunk_index,
            "risk_score": risk_result["risk_score"],
            "risk_level": risk_result["risk_level"],
            "risk_color": risk_result["risk_color"],
            "confidence": risk_result["confidence_score"],
            "synthetic_probability": risk_result["synthetic_probability"],
            "calibration_mode": risk_result["calibration"]["mode"],
            "latency_ms": latency_ms,
            "alert": risk_result["risk_level"] == "HIGH",
            "verdict": risk_result["verdict"],
            "advisory": risk_result["advisory"],
            "advisory_title": risk_result["advisory"]["title"],
            "indicators": risk_result["indicators"],
            "analysis": risk_result,
            "deepfake_analysis": risk_result,
            "timestamp": time.time(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunk analysis error: {str(e)}")


# ==========================================
# BIOMETRIC SPEAKER VERIFICATION ENDPOINTS
# ==========================================

@app.get("/api/voiceprint/profiles")
def list_voiceprint_profiles():
    """Lists all enrolled biometric speaker profiles (e.g. CEO, CFO, Family members)."""
    return {
        "profiles": speaker_verifier.list_profiles(),
        "threshold": speaker_verifier.threshold,
    }


@app.post("/api/voiceprint/enroll")
async def enroll_voiceprint(
    file: UploadFile = File(...),
    speaker_id: str = Form(...),
    name: str = Form(...),
    role: str = Form("Executive"),
    department: str = Form("Corporate Treasury"),
    _auth = Depends(enforce_security_and_rate_limit),
):
    """Enrolls a new trusted voiceprint biometric profile."""
    try:
        audio_bytes = await file.read()
        res = speaker_verifier.enroll_speaker(
            speaker_id=speaker_id,
            name=name,
            audio_input=audio_bytes,
            role=role,
            department=department,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")


@app.post("/api/verify-and-detect")
async def verify_speaker_and_detect(
    file: UploadFile = File(...),
    claimed_speaker_id: str = Form(...),
    _auth = Depends(enforce_security_and_rate_limit),
):
    """
    Dual-Factor Verification endpoint for wire transfer & executive authorization:
    1. Is this voice synthetic / cloned? (Deepfake Engine)
    2. Does this voice match the enrolled voiceprint of the claimed executive? (Biometric Engine)
    """
    if baseline_model is None or deep_model is None:
        load_models()

    try:
        audio_bytes = await file.read()
        proc = preprocessor.process(audio_bytes)
        audio = proc["audio"]
        raw_trimmed = proc["raw_trimmed"]

        base_res = baseline_model.predict(audio) if baseline_model else {"synthetic_probability": 0.5}
        deep_res = deep_model.predict(audio) if deep_model else {"synthetic_probability": 0.5}
        sig = feature_extractor.compute_forensic_signals(raw_trimmed)
        risk = risk_engine.evaluate(deep_res, base_res, sig, audio_duration=proc["duration_sec"])

        dual_decision = speaker_verifier.evaluate_dual_factor_transaction(
            deepfake_risk_result=risk,
            claimed_speaker_id=claimed_speaker_id,
            audio_input=raw_trimmed,
        )

        return {
            "success": True,
            "decision": dual_decision,
            "deepfake_analysis": risk,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dual-factor verification failed: {str(e)}")


# ==========================================
# TELEPHONY CARRIER WEBSOCKET HOOK
# ==========================================

@app.websocket("/api/telephony/ws")
async def telephony_carrier_websocket(websocket: WebSocket):
    """
    Live carrier telecom WebSocket hook (Twilio Media Streams & Asterisk AudioSocket compatible).
    Receives base64 μ-law 8kHz audio packets and streams back instant fraud intercept warnings.
    """
    if baseline_model is None or deep_model is None:
        load_models()
    await handle_telephony_websocket(websocket, baseline_model, deep_model, risk_engine)


# ==========================================
# STATIC WEB FRONTEND
# ==========================================

if os.path.exists(FRONTEND_DIR):
    if os.path.isdir(os.path.join(FRONTEND_DIR, "static")):
        app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, "static")), name="static")

    @app.get("/styles.css")
    def serve_styles():
        return FileResponse(os.path.join(FRONTEND_DIR, "styles.css"), media_type="text/css")

    @app.get("/tokens.css")
    def serve_tokens():
        return FileResponse(os.path.join(FRONTEND_DIR, "tokens.css"), media_type="text/css")

    @app.get("/app.js")
    def serve_app_js():
        return FileResponse(os.path.join(FRONTEND_DIR, "app.js"), media_type="application/javascript")

    @app.get("/script.js")
    def serve_script_js():
        return FileResponse(os.path.join(FRONTEND_DIR, "script.js"), media_type="application/javascript")

    @app.get("/", response_class=HTMLResponse)
    def serve_index():
        index_file = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse("<h2>VoiceGuard AI Web UI initializing...</h2>")
