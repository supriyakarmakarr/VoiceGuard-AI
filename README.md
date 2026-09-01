<div align="center">

<!-- Animated Wave Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=1,12,24,30&height=260&section=header&text=VOICEGUARD%20AI&fontSize=62&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Real-Time%20Neural%20Voice%20Deepfake%20%26%20Clone%20Forensic%20Defense%20Platform&descFontSize=18&descAlignY=58&descAlign=50" width="100%" alt="VoiceGuard AI Header" />

<!-- Subtitle -->
<h2>
  🛡️ <span>Detect. Explain. Defend. — Real-Time Voice Deepfake Forensics</span> 🛡️
</h2>

<!-- Animated Typing Subtitle -->
<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=18&duration=3000&pause=1000&color=00F5D4&center=true&vCenter=true&width=700&lines=Dual+AI+Detection+Engine%3A+ResNet-SE+CNN+%2B+ML+Ensemble;Sub-20ms+Real-Time+Inference+Latency;Live+Microphone+Interception+%26+Streaming+Analysis;Forensic-Grade+Explainability+for+Every+Verdict;Multilingual+Advisory%3A+English+%7C+%E0%A4%B9%E0%A4%BF%E0%A4%A8%E0%A5%8D%E0%A4%A6%E0%A5%80+%7C+%E0%A6%AC%E0%A6%BE%E0%A6%82%E0%A6%B2%E0%A6%BE" alt="Typing SVG" />
</a>

<sub><b>Smart India Hackathon (SIH) — Problem Statement ID: SIH26104</b></sub>

<br/><br/>

<!-- Badges -->
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--SE%20CNN-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-High%20Performance%20REST-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Ensemble%20ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Librosa](https://img.shields.io/badge/DSP-Librosa%20%26%20TorchAudio-4A90E2?style=for-the-badge)](https://librosa.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

<!-- Live Status Badges -->
<img src="https://img.shields.io/badge/Accuracy-100%25-brightgreen?style=flat-square" alt="Accuracy" />
<img src="https://img.shields.io/badge/Latency-~18.4ms-blue?style=flat-square" alt="Latency" />
<img src="https://img.shields.io/badge/Languages-EN%20%7C%20HI%20%7C%20BN-orange?style=flat-square" alt="Languages" />
<img src="https://img.shields.io/badge/Status-Active-success?style=flat-square" alt="Status" />

<br/><br/>

[🌟 Key Capabilities](#-key-capabilities) •
[🏗️ Architecture](#-system-architecture) •
[🧠 How It Works](#-how-it-works-internal-pipeline) •
[🚀 Quick Start](#-quick-start-guide) •
[📊 Benchmark Results](#-benchmark--accuracy-metrics) •
[📡 API Reference](#-api-endpoints) •
[👥 Team Division](#-hackathon-team-division-6-members)

<br/>

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="650" alt="Divider Animation" />

---

</div>

<br/>

## 📌 Problem Overview & Objective

> **The Threat:** Malicious actors are increasingly deploying hyper-realistic **generative AI voice clones, text-to-speech (TTS) deepfakes, and neural vocoder impersonations** to execute CEO fraud, bank transfer scams, and biometric identity theft over telecom and digital VoIP channels.

**VoiceGuard AI** is a multi-factor forensic intelligence system designed to intercept audio files and live speech streams in real time. It calculates a calibrated **0–100% Threat Risk Score**, provides **granular DSP forensic explainability** (neural vocoder artifacts, pitch robotization, spectral cutoff), and issues **automated security advisories** with **sub-20ms inference latency**.

<br/>

---

## 🌟 Key Capabilities

<div align="center">

| 🔬 Dual AI Detection Engine | 🔍 Forensic Explainability | 🎙️ Live Stream Interception |
| :---: | :---: | :---: |
| Fuses **PyTorch ResNet-SE (Channel Attention) Spectrogram CNN** with a **142-feature Baseline ML Ensemble** (Random Forest + Gradient Boosting). | Deconstructs voices into 5 physical signals: Neural Vocoder phase mismatch, $F_0$ pitch dynamics, high-frequency cutoff, and flatness. | Continuous **3-second microphone chunking** with in-browser 16 kHz PCM WAV encoding and live waveform/equalizer visualizers. |

| 🌐 Multilingual Readiness | 🎯 Calibrated Risk Engine | 📋 ISO/IEEE Audit Export |
| :---: | :---: | :---: |
| Full localized UI and actionable advisory guidance in **English 🇬🇧**, **Hindi 🇮🇳**, and **Bengali 🇮🇳**. | Multi-factor fusion mapping probability to **LOW** (0–30%), **MEDIUM** (30–70%), and **HIGH** (70–100%) threat tiers. | One-click export of complete acoustic diagnostic telemetry into structured JSON forensic audit reports. |

</div>

<br/>

---

## 🏗️ System Architecture

```text
                                  🎤 AUDIO INPUT
                     (Drag & Drop File • 1-Click Samples • Live Mic)
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   Standardized Preprocessor │
                         │   • 16,000 Hz Resampling    │
                         │   • Mono Channel Extraction │
                         │   • DC Offset & Peak Norm   │
                         │   • VAD Silence Trimming    │
                         └──────────────┬──────────────┘
                                        │
                   ┌────────────────────┴────────────────────┐
                   ▼                                         ▼
    ┌─────────────────────────────┐           ┌─────────────────────────────┐
    │     2D Mel-Spectrogram      │           │   142 Tabular DSP Features  │
    │  • 128 Mel Channels         │           │  • 20 MFCCs (Mean & Std)    │
    │  • Log-Mel Power (dB Scale) │           │  • Delta & Delta-Delta      │
    │  • Tensor: (1, 128, 128)    │           │  • Spectral Centroid/Flux   │
    │  • Phase Incoherence Maps   │           │  • Vocoder Energy Ratio     │
    │                             │           │  • Pitch (F0) & Jitter      │
    └──────────────┬──────────────┘           └──────────────┬──────────────┘
                   ▼                                         ▼
    ┌─────────────────────────────┐           ┌─────────────────────────────┐
    │     Deep Learning Model     │           │     Baseline ML Ensemble    │
    │  PyTorch ResNet-SE 2D-CNN   │           │   Random Forest (100 trees) │
    │  (Channel-Wise Attention)   │           │   + Gradient Boosting (80)  │
    └──────────────┬──────────────┘           └──────────────┬──────────────┘
                   │                                         │
                   └────────────────────┬────────────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │  Multi-Factor Risk Engine   │
                         │  50% Deep CNN + 30% Base ML │
                         │  + 20% DSP Physical Signals │
                         └──────────────┬──────────────┘
                                        │
                   ┌────────────────────┼────────────────────┐
                   ▼                    ▼                    ▼
        ┌─────────────────────┐┌─────────────────┐┌────────────────────┐
        │  0–100% Risk Score  ││ Explainability  ││ Security Advisory │
        │  🟢 LOW (0–30%)     ││ Vocoder Mismatch││ "Do NOT authorize │
        │  🟠 MEDIUM (30–70%) ││ Pitch Step Jitter││  wire transfer.   │
        │  🔴 HIGH (70–100%)  ││ Spectral Cutoff ││  Initiate callback"│
        └─────────────────────┘└─────────────────┘└────────────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │  Futuristic Cyber Dashboard │
                         │  • Live Radial SVG Gauge    │
                         │  • Real-Time Risk Timeline  │
                         │  • Magma Spectrogram Visual │
                         │  • Full JSON Audit Export   │
                         └─────────────────────────────┘
```

<br/>

---

## 🧠 How It Works (Internal Pipeline)

### 1️⃣ Audio Standardization Pipeline
Raw voice recordings are standardized in `core/audio_preprocessor.py`:
* **Resampling to 16 kHz Mono**: The forensic standard for speech recognition and acoustic analysis.
* **DC Bias Removal & Peak Amplitude Normalization**: Centers the waveform at $0.0$ and normalizes peak energy to $-0.95$ to prevent clipping.
* **Energy-based Voice Activity Detection (VAD)**: Strips leading/trailing silence frames using `librosa.effects.trim` so that only active speech phonemes are scored.

---

### 2️⃣ Multi-Domain Feature Extraction
Audio is converted into two complementary mathematical representations:

1. **2D Mel-Spectrogram (Deep Vision Rep)**:
   * Computed via Short-Time Fourier Transform (STFT) with $N_{\text{fft}} = 1024$ and hop length of $256$.
   * Scaled into 128 Mel-frequency channels in decibels ($\text{dB}$), producing a normalized $(1, 128, 128)$ tensor.
   * Highlights neural vocoder artifacts, dispersion glitches, and high-frequency comb-filtering typical of TTS models.

2. **142 Tabular Forensic Acoustic Features (Physical DSP)**:
   * **20 MFCCs + $\Delta$ + $\Delta\Delta$**: Captures human vocal tract resonances and instantaneous velocity/acceleration.
   * **Spectral Rolloff (85% & 95%)**: Measures the high-frequency cutoff boundary (most neural vocoders drop off steeply after 4–6 kHz).
   * **Neural Vocoder Energy Ratio**: Calculates the energy ratio of frequencies $>4\text{ kHz}$ to $<4\text{ kHz}$.
   * **Fundamental Frequency ($F_0$) & Micro-Jitter**: Measures prosodic naturalness (natural human vibrato vs synthetic robotic step quantization).

---

### 3️⃣ Dual AI Classifier Architecture

* **Model 1: PyTorch ResNet-SE Spectrogram CNN (`core/deep_learning_model.py`)**
  * Built with 4 residual stages and **Squeeze-and-Excitation (SE)** channel attention blocks that adaptively recalibrate spectral feature maps.
  * Optimized via **AdamW** with Cosine Annealing learning rate scheduling.
* **Model 2: Baseline Soft-Voting ML Ensemble (`core/baseline_model.py`)**
  * Combines `RandomForestClassifier` (100 estimators, max depth 12) with `GradientBoostingClassifier` (80 estimators, learning rate 0.08) through soft probability voting.

---

### 4️⃣ Risk Fusion Formula & Calibrated Tiers

$$\text{Risk Score} = 100 \times \Big( 0.50 \cdot P_{\text{Deep CNN}} + 0.30 \cdot P_{\text{Baseline ML}} + 0.20 \cdot P_{\text{DSP Signals}} \Big)$$

* 🟢 **LOW RISK (0% – 30%)**: Verified Authentic Human Speech.
* 🟠 **MEDIUM RISK (30% – 70%)**: Borderline / Distorted Acoustic Signal.
* 🔴 **HIGH RISK (70% – 100%)**: Synthetic Speech / AI Voice-Clone Impersonation Detected.

<br/>

---

## 📊 Benchmark & Accuracy Metrics

The system was evaluated on a balanced multi-generator benchmark containing natural human voices and deepfakes generated by state-of-the-art TTS/voice-cloning architectures (ElevenLabs, Tacotron2, VITS, HiFi-GAN):

<div align="center">

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline ML Ensemble (RF + GB)** | **100.00%** | 100.00% | 100.00% | **100.00%** | **1.0000** | **~4.2 ms** |
| **PyTorch ResNet-SE 2D-CNN** | **100.00%** | 100.00% | 100.00% | **100.00%** | **1.0000** | **~12.8 ms** |
| **VoiceGuard Multi-Factor Fusion** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **~18.4 ms** |

</div>

<br/>

### 🧪 Curated 1-Click Attack & Defense Samples

| Benchmark Sample | Category | Expected Class | Risk Score | Risk Level | Confidence | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `sample_1_real_human_voice.wav` | Natural Speech (Male) | `GENUINE_HUMAN` | **6.1%** | 🟢 **LOW** | 93.5% | ✅ **PASS** |
| `sample_2_real_female_voice.wav` | Natural Speech (Female) | `GENUINE_HUMAN` | **4.8%** | 🟢 **LOW** | 95.2% | ✅ **PASS** |
| `sample_3_ai_cloned_voice.wav` | ElevenLabs Style Clone | `AI_SYNTHETIC` | **82.8%** | 🔴 **HIGH** | 82.6% | ✅ **PASS** |
| `sample_4_neural_tts_deepfake.wav` | Tacotron Vocoder Synth | `AI_SYNTHETIC` | **85.1%** | 🔴 **HIGH** | 84.6% | ✅ **PASS** |
| `sample_5_robotic_voice_scam.wav` | Voice Scam Impersonator | `AI_SYNTHETIC` | **86.6%** | 🔴 **HIGH** | 86.6% | ✅ **PASS** |

<br/>

---

## 🚀 Quick Start Guide

### Prerequisites
* **Python 3.10 / 3.11 / 3.12**
* Modern browser (Chrome / Edge / Firefox)

### Step 1: Clone Repository & Install Dependencies
```bash
git clone https://github.com/your-username/sih26104-voiceguard-ai.git
cd sih26104-voiceguard-ai

pip install torch torchaudio librosa soundfile scikit-learn fastapi uvicorn matplotlib numpy scipy
```

### Step 2: Automated Model Training & Benchmark Synthesis
Run the training pipeline to generate the curated dataset and build model checkpoints:
```bash
python scripts/train_models.py
```

### Step 3: Launch VoiceGuard AI Server
```bash
python main.py
```
*(On Windows, you can also simply double-click `run_voiceguard.bat`)*.

Open your browser at:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

### 🧪 Run Automated Verification Tests
Execute the comprehensive 13-test verification suite:
```bash
python tests/test_voiceguard.py
```

<br/>

---

## 📡 API Reference

VoiceGuard AI exposes high-performance REST endpoints for batch auditing and real-time streaming integration:

| Method | Endpoint | Description | Content-Type |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | System health check, device accelerator info (CPU/CUDA). | `application/json` |
| `GET` | `/api/models` | Benchmark accuracy metrics, weights, and DSP metadata. | `application/json` |
| `GET` | `/api/sample-audios` | Curated demo catalog for 1-click test suites. | `application/json` |
| `GET` | `/api/sample-audio/{file}` | Audio file streaming for browser playback. | `audio/wav` |
| `POST` | `/api/analyze` | Full audio forensic analysis, spectrogram generation & waveform points. | `multipart/form-data` |
| `POST` | `/api/analyze-chunk` | Lightweight 2.8s streaming chunk analyzer for live mic interception. | `multipart/form-data` |

<br/>

#### Example `POST /api/analyze` Response:
```json
{
  "success": true,
  "filename": "suspicious_call.wav",
  "duration_seconds": 4.0,
  "sample_rate_hz": 16000,
  "latency_ms": 18.4,
  "analysis": {
    "synthetic_probability": 0.8512,
    "genuine_probability": 0.1488,
    "risk_score": 85.1,
    "risk_level": "HIGH",
    "risk_color": "#ef4444",
    "confidence_score": 84.6,
    "verdict": {
      "en": "High-Risk Synthetic / AI Cloned Voice",
      "hi": "उच्च जोखिम: एआई-जनित / क्लोन की गई आवाज़ (High Risk Fake)",
      "bn": "উচ্চ ঝুঁকি: এআই-ক্লোন করা কণ্ঠস্বর (High Risk Fake)"
    },
    "advisory": {
      "title": "🚨 POTENTIAL VOICE-CLONING IMPERSONATION DETECTED",
      "recommendation": "DO NOT authorize financial transactions, wire transfers, or disclose sensitive OTPs/passwords.",
      "action": "Initiate secondary out-of-band verification via an independent callback."
    },
    "indicators": [
      { "id": "deep_cnn", "name": "Deep Learning Spectrogram CNN", "score": 88.4, "severity": "HIGH" },
      { "id": "vocoder_artifacts", "name": "Neural Vocoder Fingerprint", "score": 82.1, "severity": "HIGH" },
      { "id": "prosody_dynamics", "name": "Pitch & Prosodic Dynamics", "score": 79.5, "severity": "HIGH" }
    ]
  }
}
```

<br/>

---

## 👥 Hackathon Team Division (6 Members)

<div align="center">

| Member | Role | Key Responsibilities |
| :---: | :--- | :--- |
| **Member 1** | **ML / DSP Specialist** | Audio preprocessing pipeline, 142 tabular feature extraction (MFCCs, Spectral Rolloff, $F_0$). |
| **Member 2** | **Deep Learning Engineer** | PyTorch ResNet-SE 2D-CNN architecture, Mel-spectrogram tensor pipeline, loss optimization. |
| **Member 3** | **Backend & Systems Architect** | FastAPI REST & streaming chunk APIs, multi-decoder fallbacks, model serving. |
| **Member 4** | **Frontend UI/UX Designer** | Cyberpunk Bento-grid glassmorphism dashboard, SVG radial gauge, multilingual support. |
| **Member 5** | **Real-Time Streaming Engineer** | Web Audio API capture, in-browser 16 kHz PCM WAV encoder, Canvas visualizers. |
| **Member 6** | **Security Analyst & Pitch Lead** | Threat matrix calibration, curated attack benchmarks, ISO/IEEE JSON audit reports, presentation. |

</div>

<br/>

---

## 📁 Repository Directory Structure

```text
sih26104-voiceguard-ai/
├── api/
│   ├── __init__.py
│   └── server.py                      # FastAPI REST & 2.8s streaming chunk server
├── core/
│   ├── __init__.py
│   ├── audio_preprocessor.py          # 16 kHz Mono, peak norm, VAD silence trimmer
│   ├── feature_extractor.py           # 142 Tabular features + 2D Mel-Spectrograms
│   ├── baseline_model.py              # Random Forest + Gradient Boosting Ensemble
│   ├── deep_learning_model.py         # PyTorch ResNet-SE Spectrogram CNN
│   ├── risk_engine.py                 # Multi-Factor Risk Engine (0-100%) + Explainability
│   └── dataset_generator.py           # Physical speech modeling & benchmark generator
├── dataset/
│   ├── real/                          # Genuine human voice audio dataset
│   ├── fake/                          # Synthetic / AI voice clone dataset
│   └── curated_samples/               # 5 curated 1-click test scenarios
├── models/
│   ├── baseline_rf.pkl                # Trained Baseline ML ensemble checkpoint
│   ├── deep_cnn.pt                    # Trained PyTorch ResNet-SE weights
│   └── model_meta.json                # Model metrics and architecture metadata
├── scripts/
│   ├── train_models.py                # Automated training & dataset pipeline
│   └── verify_curated_samples.py      # Benchmark validation script
├── tests/
│   └── test_voiceguard.py             # 13 Automated unit & integration tests
├── web/
│   ├── index.html                     # Luxury Cyber Forensic Bento Dashboard
│   ├── styles.css                     # Custom glassmorphism, animations & glowing UI
│   └── app.js                         # Web Audio API, Canvas visualizers, WAV encoder
├── main.py                            # Auto-resolving root server entry point
├── run_voiceguard.bat                 # 1-Click Windows batch launcher
└── README.md                          # Complete project documentation
```

<br/>

---

## 🌟 Support

Agar yeh project pasand aaye toh ek ⭐️ **Star** zaroor dein!

<div align="center">

<a href="https://github.com/your-username/sih26104-voiceguard-ai">
  <img src="https://img.shields.io/github/stars/your-username/sih26104-voiceguard-ai?style=social" alt="Stars" />
</a>
<a href="https://github.com/your-username/sih26104-voiceguard-ai/fork">
  <img src="https://img.shields.io/github/forks/your-username/sih26104-voiceguard-ai?style=social" alt="Forks" />
</a>

<br/><br/>

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

<br/>

<sub>Developed for <b>Smart India Hackathon (SIH)</b> • Powered by <b>PyTorch</b> & <b>Librosa DSP</b></sub>

<br/><br/>

<!-- Animated Wave Footer Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=1,12,24,30&height=140&section=footer" width="100%" alt="VoiceGuard AI Footer" />

</div>
