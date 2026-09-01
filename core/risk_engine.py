"""
VoiceGuard AI - Forensic Risk & Explainability Engine
Fuses Deep Learning CNN, Baseline Ensemble ML, and Multi-Domain Acoustic Forensic Signals
to generate calibrated Risk Scores (0-100%), Threat Levels, and Explainable Forensic Indicators.
"""

import numpy as np


class RiskEngine:
    """
    Multi-Factor Forensic Risk Engine for Voice Deepfake & Clone Analysis.
    """

    def __init__(
        self,
        weight_deep: float = 0.50,
        weight_baseline: float = 0.30,
        weight_forensic: float = 0.20,
        low_threshold: float = 30.0,
        high_threshold: float = 70.0,
    ):
        self.weight_deep = weight_deep
        self.weight_baseline = weight_baseline
        self.weight_forensic = weight_forensic
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def evaluate(
        self,
        deep_result: dict,
        baseline_result: dict,
        forensic_signals: dict,
        audio_duration: float = 4.0,
    ) -> dict:
        """
        Fuses all analytical signals into a unified risk assessment.
        """
        p_deep = float(deep_result.get("synthetic_probability", 0.5))
        p_base = float(baseline_result.get("synthetic_probability", 0.5))

        # Forensic signal composite score
        hf_anomaly = float(forensic_signals.get("hf_anomaly_score", 0.0))
        prosody_anomaly = float(forensic_signals.get("prosody_anomaly_score", 0.0))
        cutoff_anomaly = float(forensic_signals.get("spectral_cutoff_score", 0.0))
        flatness_anomaly = float(forensic_signals.get("spectral_flatness_score", 0.0))

        p_forensic = float(
            0.35 * hf_anomaly + 0.30 * prosody_anomaly + 0.20 * cutoff_anomaly + 0.15 * flatness_anomaly
        )

        # Composite Synthetic Probability
        composite_prob = float(
            self.weight_deep * p_deep + self.weight_baseline * p_base + self.weight_forensic * p_forensic
        )
        composite_prob = float(np.clip(composite_prob, 0.01, 0.99))

        # Scaled Risk Score (0 - 100)
        risk_score = round(composite_prob * 100.0, 1)

        # Determine Risk Level
        if risk_score < self.low_threshold:
            risk_level = "LOW"
            risk_color = "#10b981"  # Emerald Green
            verdict_en = "Likely Genuine Human Voice"
            verdict_hi = "संभवतः वास्तविक मानव आवाज़ (Genuine)"
            verdict_bn = "সম্ভবত আসল মানুষের কণ্ঠ (Genuine)"
        elif risk_score < self.high_threshold:
            risk_level = "MEDIUM"
            risk_color = "#f59e0b"  # Amber Orange
            verdict_en = "Suspicious / Inconclusive Voice Signals"
            verdict_hi = "संदिग्ध / अनिश्चित आवाज़ संकेत (Suspicious)"
            verdict_bn = "সন্দেহজনক / অনিশ্চিত কণ্ঠস্বর (Suspicious)"
        else:
            risk_level = "HIGH"
            risk_color = "#ef4444"  # Crimson Red
            verdict_en = "High-Risk Synthetic / AI Cloned Voice"
            verdict_hi = "उच्च जोखिम: एआई-जनित / क्लोन की गई आवाज़ (High Risk Fake)"
            verdict_bn = "উচ্চ ঝুঁকি: এআই-ক্লোন করা কণ্ঠস্বর (High Risk Fake)"

        # Model Agreement & Confidence Calculation
        model_diff = abs(p_deep - p_base)
        base_confidence = max(composite_prob, 1.0 - composite_prob)
        # Penalize confidence slightly if models strongly disagree or audio duration is too short (<1.5s)
        duration_factor = min(1.0, audio_duration / 2.5)
        agreement_factor = 1.0 - (model_diff * 0.4)
        final_confidence = round(float(base_confidence * agreement_factor * duration_factor * 100.0), 1)
        final_confidence = max(55.0, min(99.5, final_confidence))

        # Forensic Indicators Breakdown
        indicators = [
            {
                "id": "deep_cnn",
                "name": "Deep Learning Spectrogram CNN",
                "score": round(p_deep * 100, 1),
                "severity": "HIGH" if p_deep >= 0.7 else ("MEDIUM" if p_deep >= 0.35 else "LOW"),
                "detected": p_deep >= 0.5,
                "description": "2D Mel-Spectrogram convolutional residual network response.",
            },
            {
                "id": "vocoder_artifacts",
                "name": "Neural Vocoder Fingerprint",
                "score": round(hf_anomaly * 100, 1),
                "severity": "HIGH" if hf_anomaly >= 0.65 else ("MEDIUM" if hf_anomaly >= 0.35 else "LOW"),
                "detected": hf_anomaly >= 0.5,
                "description": "High-frequency phase mismatch & vocoder harmonic patterns (HiFi-GAN / Bark / VITS).",
            },
            {
                "id": "prosody_dynamics",
                "name": "Pitch & Prosodic Dynamics",
                "score": round(prosody_anomaly * 100, 1),
                "severity": "HIGH" if prosody_anomaly >= 0.65 else ("MEDIUM" if prosody_anomaly >= 0.35 else "LOW"),
                "detected": prosody_anomaly >= 0.5,
                "description": "Unnatural fundamental frequency (F0) monotonicity or robotic step transitions.",
            },
            {
                "id": "spectral_cutoff",
                "name": "High-Frequency Spectral Cutoff",
                "score": round(cutoff_anomaly * 100, 1),
                "severity": "HIGH" if cutoff_anomaly >= 0.65 else ("MEDIUM" if cutoff_anomaly >= 0.35 else "LOW"),
                "detected": cutoff_anomaly >= 0.5,
                "description": "Artificial bandwidth shelf or steep roll-off typical of downsampled TTS generators.",
            },
            {
                "id": "baseline_rf",
                "name": "Acoustic Feature Ensemble (RF/GB)",
                "score": round(p_base * 100, 1),
                "severity": "HIGH" if p_base >= 0.7 else ("MEDIUM" if p_base >= 0.35 else "LOW"),
                "detected": p_base >= 0.5,
                "description": "Statistical tabular classifier over MFCCs, Spectral Flux, and ZCR dynamics.",
            },
        ]

        # Actionable Advisory Guidance
        if risk_level == "HIGH":
            advisory = {
                "title": "🚨 POTENTIAL VOICE-CLONING IMPERSONATION DETECTED",
                "recommendation": "DO NOT authorize financial transactions, wire transfers, or disclose sensitive OTPs/passwords.",
                "action": "Initiate secondary out-of-band verification via an independent callback to the registered phone number.",
                "title_hi": "🚨 संभावित वॉइस-क्लोनिंग धोखाधड़ी पाई गई",
                "recommendation_hi": "लेन-देन को अधिकृत न करें और न ही ओटीपी/पासवर्ड साझा करें।",
                "action_hi": "पंजीकृत नंबर पर स्वतंत्र कॉल-बैक के माध्यम से द्वितीयक सत्यापन करें।",
            }
        elif risk_level == "MEDIUM":
            advisory = {
                "title": "⚠️ SUSPICIOUS ACOUSTIC SIGNALS DETECTED",
                "recommendation": "Acoustic markers show borderline synthetic patterns or elevated noise distortion.",
                "action": "Prompt the caller with a dynamic challenge question or request a longer clear speech sample.",
                "title_hi": "⚠️ संदिग्ध ध्वनि संकेत पाए गए",
                "recommendation_hi": "ध्वनि में कृत्रिम लक्षण पाए गए हैं।",
                "action_hi": "कॉलर से एक गतिशील सुरक्षा प्रश्न पूछें या दोबारा बोलने का अनुरोध करें।",
            }
        else:
            advisory = {
                "title": "✅ VOICE INTEGRITY VERIFIED (AUTHENTIC)",
                "recommendation": "Acoustic harmonics, pitch contour dynamics, and phase coherence match natural human speech.",
                "action": "Standard security verification procedures can proceed normally.",
                "title_hi": "✅ आवाज़ की प्रामाणिकता सत्यापित (Authentic)",
                "recommendation_hi": "ध्वनि की गतिशीलता और हार्मोनिक्स प्राकृतिक मानव आवाज़ से मेल खाते हैं।",
                "action_hi": "सामान्य सुरक्षा प्रक्रिया जारी रखी जा सकती है।",
            }

        return {
            "synthetic_probability": round(composite_prob, 4),
            "genuine_probability": round(1.0 - composite_prob, 4),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "confidence_score": final_confidence,
            "verdict": {
                "en": verdict_en,
                "hi": verdict_hi,
                "bn": verdict_bn,
            },
            "advisory": advisory,
            "indicators": indicators,
            "forensic_metrics": {
                "deep_prob": round(p_deep, 4),
                "baseline_prob": round(p_base, 4),
                "f0_mean_hz": round(float(forensic_signals.get("f0_mean_hz", 150.0)), 1),
                "f0_std_hz": round(float(forensic_signals.get("f0_std_hz", 20.0)), 1),
                "vocoder_ratio": round(float(forensic_signals.get("hf_vocoder_ratio", 0.05)), 4),
                "voiced_ratio_pct": round(float(forensic_signals.get("voiced_ratio", 0.5)) * 100, 1),
                "audio_duration_sec": round(float(audio_duration), 2),
            },
        }
