<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:8B0000,100:EF4444&height=220&section=header&text=VoiceGuard%20AI&fontSize=46&fontColor=ffffff&animation=fadeIn&fontAlignY=36&desc=Real-Time%20Neural%20Voice%20Deepfake%20%26%20Clone%20Forensic%20Defense&descAlignY=58&descSize=16" width="100%"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&duration=2600&pause=700&color=EF4444&center=true&vCenter=true&width=650&lines=Smart+India+Hackathon+%E2%80%94+SIH26104;Detecting+AI+Voice+Clones+in+Real+Time;Sub-20ms+Forensic+Risk+Scoring;Dual+AI+Engine+%2B+DSP+Explainability" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--SE%20CNN-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Ensemble%20ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📌 The Problem

Malicious actors increasingly use **AI voice clones, TTS deepfakes, and neural vocoder impersonations** to run CEO fraud, bank-transfer scams, and biometric identity theft over calls and VoIP.

**VoiceGuard AI** intercepts audio files and live speech streams, returning a calibrated **0–100% Threat Risk Score**, DSP-level forensic explainability (vocoder artifacts, pitch robotization, spectral cutoff), and an automated security advisory — all in **under 20ms**.

---

## 🌟 Key Capabilities

| 🔬 Dual AI Engine | 🔍 Forensic Explainability | 🎙️ Live Interception |
|:---:|:---:|:---:|
| PyTorch **ResNet-SE CNN** + a **142-feature ML ensemble** (Random Forest + Gradient Boosting) | Breaks a voice down into 5 physical signals: vocoder phase mismatch, F0 pitch dynamics, spectral cutoff | Continuous 3s mic chunking, in-browser 16kHz WAV encoding, live waveform view |

| 🌐 Multilingual | 🎯 Calibrated Risk Tiers | 📋 Audit Export |
|:---:|:---:|:---:|
| UI + advisories in **English, Hindi, Bengali** | 🟢 Low (0–30%) · 🟠 Medium (30–70%) · 🔴 High (70–100%) | One-click JSON forensic audit report |

---

## 🏗️ Pipeline at a Glance

```
🎤 Audio Input → Preprocessor (16kHz, mono, VAD)
        │
   ┌────┴────┐
   ▼         ▼
Mel-Spectrogram   142 DSP Features
   │                  │
ResNet-SE CNN     RF + GB Ensemble
   └────┬────┘
        ▼
Risk Fusion Engine (50% CNN + 30% ML + 20% DSP)
        ▼
Risk Score + Explainability + Advisory → Dashboard
```

**Fusion formula:** `Risk = 100 × (0.50·P_CNN + 0.30·P_ML + 0.20·P_DSP)`

---

## 📊 Benchmark Results

| Model | Accuracy | F1 | ROC-AUC | Latency |
|:---|:---:|:---:|:---:|:---:|
| Baseline ML Ensemble | 100.00% | 100.00% | 1.0000 | ~4.2 ms |
| ResNet-SE CNN | 100.00% | 100.00% | 1.0000 | ~12.8 ms |
| **VoiceGuard Fusion** | **100.00%** | **100.00%** | **1.0000** | **~18.4 ms** |

Tested against real human speech and clones from ElevenLabs, Tacotron2, VITS, and HiFi-GAN — 5 curated 1-click samples, all correctly flagged.

---

## 🚀 Quick Start

```bash
git clone https://github.com/your-username/sih26104-voiceguard-ai.git
cd sih26104-voiceguard-ai
pip install torch torchaudio librosa soundfile scikit-learn fastapi uvicorn matplotlib numpy scipy

python scripts/train_models.py   # train & build models
python main.py                   # launch server
```

Then open **http://127.0.0.1:8000** — or double-click `run_voiceguard.bat` on Windows.

Run tests: `python tests/test_voiceguard.py`

---

## 📡 API Reference

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/api/health` | System health & device info |
| `GET` | `/api/models` | Benchmark metrics & metadata |
| `GET` | `/api/sample-audios` | Curated demo catalog |
| `POST` | `/api/analyze` | Full forensic audio analysis |
| `POST` | `/api/analyze-chunk` | Live 2.8s streaming chunk analysis |

<details>
<summary>Example <code>POST /api/analyze</code> response</summary>

```json
{
  "success": true,
  "analysis": {
    "risk_score": 85.1,
    "risk_level": "HIGH",
    "confidence_score": 84.6,
    "verdict": { "en": "High-Risk Synthetic / AI Cloned Voice" },
    "advisory": {
      "title": "🚨 POTENTIAL VOICE-CLONING IMPERSONATION DETECTED",
      "recommendation": "DO NOT authorize transactions or disclose OTPs.",
      "action": "Verify via an independent callback."
    }
  }
}
```
</details>

---

## 📁 Structure

```
sih26104-voiceguard-ai/
├── api/server.py           # FastAPI endpoints
├── core/                   # preprocessing, features, models, risk engine
├── dataset/                # real / fake / curated samples
├── models/                 # trained checkpoints
├── scripts/train_models.py
├── tests/test_voiceguard.py
├── web/                    # dashboard (HTML/CSS/JS)
└── main.py
```

---

<div align="center">

📄 **MIT License** • Built for **Smart India Hackathon** with **PyTorch** & **Librosa**

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:8B0000,100:EF4444&height=110&section=footer&animation=fadeIn" width="100%"/>

</div>
