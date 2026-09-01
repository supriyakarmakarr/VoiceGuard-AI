"""
VoiceGuard AI - FastAPI Forensic Server & Real-Time Stream Engine
Provides REST and Chunk-Streaming APIs for Voice Deepfake Detection,
Forensic Explainability, and Live Cyber Risk Dashboard.
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
import soundfile as sf
import librosa
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse

# Ensure root is in Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.audio_preprocessor import AudioPreprocessor
from core.feature_extractor import FeatureExtractor
from core.baseline_model import BaselineVoiceClassifier
from core.deep_learning_model import DeepLearningVoiceClassifier
from core.risk_engine import RiskEngine

app = FastAPI(
    title="VoiceGuard AI - Voice Deepfake & Clone Detection API",
    description="Real-Time Audio Deepfake Forensic Detection & Risk Engine for SIH26104",
    version="1.0.0",
)

# Enable CORS for cross-origin frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
CURATED_DIR = os.path.join(DATASET_DIR, "curated_samples")
WEB_DIR = os.path.join(PROJECT_ROOT, "web")

# Global instances
preprocessor = AudioPreprocessor()
feature_extractor = FeatureExtractor()
risk_engine = RiskEngine()

baseline_model = None
deep_model = None
model_metadata = {}


def load_models():
    """Initializes and loads trained ML and Deep Learning models."""
    global baseline_model, deep_model, model_metadata

    baseline_path = os.path.join(MODELS_DIR, "baseline_rf.pkl")
    deep_path = os.path.join(MODELS_DIR, "deep_cnn.pt")
    meta_path = os.path.join(MODELS_DIR, "model_meta.json")

    # Load baseline model
    if os.path.exists(baseline_path):
        try:
            baseline_model = BaselineVoiceClassifier(baseline_path)
            print(f"[API] Loaded Baseline ML model from {baseline_path}")
        except Exception as e:
            print(f"[API] Warning loading baseline model: {e}")
    else:
        print("[API] Baseline model not found on disk yet.")

    # Load deep learning model
    if os.path.exists(deep_path):
        try:
            deep_model = DeepLearningVoiceClassifier(deep_path)
            print(f"[API] Loaded Deep CNN model from {deep_path}")
        except Exception as e:
            print(f"[API] Warning loading deep learning model: {e}")
    else:
        print("[API] Deep CNN model not found on disk yet.")

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                model_metadata = json.load(f)
        except Exception:
            pass


@app.on_event("startup")
def startup_event():
    load_models()


def generate_spectrogram_base64(audio: np.ndarray, sr: int = 16000) -> str:
    """Generates a high-contrast forensic Mel-Spectrogram image encoded as base64 PNG."""
    try:
        fig, ax = plt.subplots(figsize=(6, 2.5), dpi=100)
        fig.patch.set_facecolor("#0b0f19")
        ax.set_facecolor("#0b0f19")

        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        img = ax.imshow(
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


@app.get("/api/health")
def health_check():
    """Health status and model load verification."""
    return {
        "status": "healthy",
        "service": "SIH26104 - VoiceGuard AI Engine",
        "baseline_model_loaded": baseline_model is not None,
        "deep_model_loaded": deep_model is not None,
        "device": str(deep_model.device) if deep_model else "cpu",
        "timestamp": time.time(),
    }


@app.get("/api/models")
def get_models_info():
    """Returns model benchmark accuracy, metrics, and architecture details."""
    return {
        "metadata": model_metadata,
        "baseline_metrics": baseline_model.metrics if baseline_model else {},
        "deep_metrics": deep_model.metrics if deep_model else {},
        "feature_count": len(baseline_model.feature_names) if baseline_model else 0,
    }


@app.get("/api/sample-audios")
def list_sample_audios():
    """Returns curated demo audio files for 1-click hackathon testing."""
    os.makedirs(CURATED_DIR, exist_ok=True)
    files = [f for f in os.listdir(CURATED_DIR) if f.lower().endswith(".wav")]
    samples = []

    descriptions = {
        "sample_1_real_human_voice.wav": {
            "title": "Natural Human Speech 1 (Male)",
            "expected": "GENUINE_HUMAN",
            "category": "Authentic",
            "badge": "Real Voice",
        },
        "sample_2_real_female_voice.wav": {
            "title": "Natural Human Speech 2 (Female)",
            "expected": "GENUINE_HUMAN",
            "category": "Authentic",
            "badge": "Real Voice",
        },
        "sample_3_ai_cloned_voice.wav": {
            "title": "AI Voice Clone (ElevenLabs style)",
            "expected": "AI_SYNTHETIC",
            "category": "Voice Clone",
            "badge": "High Risk Clone",
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
    """Serves the WAV audio file for demo playback."""
    file_path = os.path.join(CURATED_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Sample audio file not found.")
    return FileResponse(file_path, media_type="audio/wav")


@app.post("/api/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    threshold_low: float = Form(30.0),
    threshold_high: float = Form(70.0),
):
    """
    Main forensic analysis endpoint:
    Processes uploaded audio -> Feature Extraction -> Dual Models -> Risk & Explainability Engine.
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

        # Risk Engine Multi-Factor Assessment
        custom_risk_engine = RiskEngine(low_threshold=threshold_low, high_threshold=threshold_high)
        risk_result = custom_risk_engine.evaluate(
            deep_result=deep_result,
            baseline_result=base_result,
            forensic_signals=forensic_signals,
            audio_duration=duration,
        )

        # Generate Forensic Spectrogram Base64 Image
        spectrogram_b64 = generate_spectrogram_base64(raw_trimmed, sr=preprocessor.target_sr)

        # Waveform preview points (for fast rendering on frontend canvas)
        num_preview_pts = 80
        step = max(1, len(raw_trimmed) // num_preview_pts)
        waveform_pts = [round(float(np.max(np.abs(raw_trimmed[i : i + step]))), 3) for i in range(0, len(raw_trimmed), step)][:num_preview_pts]

        latency_ms = round((time.time() - start_time) * 1000, 1)

        return {
            "success": True,
            "filename": file.filename,
            "duration_seconds": round(duration, 2),
            "sample_rate_hz": preprocessor.target_sr,
            "latency_ms": latency_ms,
            "analysis": risk_result,
            "models_raw": {
                "deep_cnn": deep_result,
                "baseline_rf": base_result,
            },
            "spectrogram_image": spectrogram_b64,
            "waveform_preview": waveform_pts,
            "timestamp": time.time(),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/analyze-chunk")
async def analyze_stream_chunk(
    file: UploadFile = File(...),
    chunk_index: int = Form(0),
):
    """
    Lightweight streaming chunk endpoint for real-time microphone monitoring.
    Processes small 2-4 second audio chunks and returns immediate risk level and delta.
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

        # Run models
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
            "latency_ms": latency_ms,
            "alert": risk_result["risk_level"] == "HIGH",
            "advisory_title": risk_result["advisory"]["title"],
            "timestamp": time.time(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunk analysis error: {str(e)}")


# Serve static web frontend
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def serve_index():
        index_file = os.path.join(WEB_DIR, "index.html")
        if os.path.exists(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse("<h2>VoiceGuard AI Web UI initializing...</h2>")
