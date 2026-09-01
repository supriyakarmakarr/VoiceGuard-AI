"""
VoiceGuard AI - Comprehensive Verification Test Suite
SIH26104: Real-Time Audio Deepfake Forensic Detection & Risk Engine
"""

import os
import sys
import io
import unittest
import numpy as np
import soundfile as sf
import torch
from starlette.testclient import TestClient

# Ensure root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.audio_preprocessor import AudioPreprocessor
from core.feature_extractor import FeatureExtractor
from core.dataset_generator import generate_human_speech_sample, generate_synthetic_speech_sample
from core.baseline_model import BaselineVoiceClassifier
from core.deep_learning_model import DeepLearningVoiceClassifier, VoiceGuardSpectrogramCNN
from core.risk_engine import RiskEngine
from api.server import app


class TestAudioPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = AudioPreprocessor(target_sr=16000, target_duration=4.0)

    def test_normalize_audio(self):
        audio = np.array([0.1, -0.5, 0.8, -0.2], dtype=np.float32)
        norm = self.preprocessor.normalize_audio(audio, target_peak=0.95)
        self.assertAlmostEqual(float(np.max(np.abs(norm))), 0.95, places=2)
        self.assertAlmostEqual(float(np.mean(norm)), 0.0, places=2)

    def test_pad_or_truncate(self):
        short_audio = np.ones(1000, dtype=np.float32)
        padded = self.preprocessor.pad_or_truncate(short_audio, target_length=64000)
        self.assertEqual(len(padded), 64000)

        long_audio = np.ones(100000, dtype=np.float32)
        truncated = self.preprocessor.pad_or_truncate(long_audio, target_length=64000)
        self.assertEqual(len(truncated), 64000)

    def test_process_bytes(self):
        sample = generate_human_speech_sample(duration=2.0)
        buf = io.BytesIO()
        sf.write(buf, sample, 16000, format="WAV")
        buf.seek(0)
        proc = self.preprocessor.process(buf.read())
        self.assertEqual(len(proc["audio"]), 64000)
        self.assertEqual(proc["sr"], 16000)


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor(sr=16000)
        self.audio = generate_human_speech_sample(duration=4.0)

    def test_extract_tabular_features(self):
        feats, names = self.extractor.extract_tabular_features(self.audio)
        self.assertIsInstance(feats, np.ndarray)
        self.assertFalse(np.isnan(feats).any(), "Tabular features must not contain NaNs")
        self.assertFalse(np.isinf(feats).any(), "Tabular features must not contain Infs")
        self.assertEqual(len(feats), len(names))
        self.assertGreater(len(feats), 50)

    def test_extract_mel_spectrogram(self):
        spec = self.extractor.extract_mel_spectrogram(self.audio, target_time_frames=128)
        self.assertEqual(spec.shape, (1, 128, 128))
        self.assertTrue(np.all(spec >= -1.0) and np.all(spec <= 1.0))

    def test_compute_forensic_signals(self):
        signals = self.extractor.compute_forensic_signals(self.audio)
        required_keys = [
            "hf_vocoder_ratio", "hf_anomaly_score", "prosody_anomaly_score",
            "spectral_cutoff_score", "spectral_flatness_score", "f0_mean_hz"
        ]
        for k in required_keys:
            self.assertIn(k, signals)


class TestRiskEngine(unittest.TestCase):
    def setUp(self):
        self.risk_engine = RiskEngine(low_threshold=30.0, high_threshold=70.0)

    def test_low_risk_genuine_evaluation(self):
        deep_res = {"synthetic_probability": 0.08, "genuine_probability": 0.92}
        base_res = {"synthetic_probability": 0.12, "genuine_probability": 0.88}
        forensic_sig = {
            "hf_anomaly_score": 0.05,
            "prosody_anomaly_score": 0.10,
            "spectral_cutoff_score": 0.05,
            "spectral_flatness_score": 0.05,
        }
        res = self.risk_engine.evaluate(deep_res, base_res, forensic_sig, audio_duration=4.0)
        self.assertEqual(res["risk_level"], "LOW")
        self.assertLess(res["risk_score"], 30.0)
        self.assertIn("en", res["verdict"])
        self.assertIn("hi", res["verdict"])
        self.assertIn("bn", res["verdict"])

    def test_high_risk_synthetic_evaluation(self):
        deep_res = {"synthetic_probability": 0.95, "genuine_probability": 0.05}
        base_res = {"synthetic_probability": 0.89, "genuine_probability": 0.11}
        forensic_sig = {
            "hf_anomaly_score": 0.85,
            "prosody_anomaly_score": 0.80,
            "spectral_cutoff_score": 0.75,
            "spectral_flatness_score": 0.70,
        }
        res = self.risk_engine.evaluate(deep_res, base_res, forensic_sig, audio_duration=4.0)
        self.assertEqual(res["risk_level"], "HIGH")
        self.assertGreaterEqual(res["risk_score"], 70.0)
        self.assertIn("DO NOT authorize", res["advisory"]["recommendation"])


class TestDeepSpectrogramCNN(unittest.TestCase):
    def test_cnn_architecture(self):
        model = VoiceGuardSpectrogramCNN()
        dummy_in = torch.randn(2, 1, 128, 128)
        out = model(dummy_in)
        self.assertEqual(out.shape, (2, 1))


class TestFastAPIServer(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_sample_audios_endpoint(self):
        response = self.client.get("/api/sample-audios")
        self.assertEqual(response.status_code, 200)
        self.assertIn("samples", response.json())

    def test_analyze_endpoint_with_wav(self):
        # Generate valid audio bytes
        audio = generate_human_speech_sample(duration=3.0)
        buf = io.BytesIO()
        sf.write(buf, audio, 16000, format="WAV")
        buf.seek(0)

        response = self.client.post(
            "/api/analyze",
            files={"file": ("test_speech.wav", buf.read(), "audio/wav")},
            data={"threshold_low": "30.0", "threshold_high": "70.0"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("analysis", data)
        self.assertIn("risk_score", data["analysis"])
        self.assertIn("spectrogram_image", data)

    def test_analyze_chunk_endpoint(self):
        audio = generate_synthetic_speech_sample(duration=3.0)
        buf = io.BytesIO()
        sf.write(buf, audio, 16000, format="WAV")
        buf.seek(0)

        response = self.client.post(
            "/api/analyze-chunk",
            files={"file": ("chunk_01.wav", buf.read(), "audio/wav")},
            data={"chunk_index": "1"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("risk_score", data)
        self.assertIn("risk_level", data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
