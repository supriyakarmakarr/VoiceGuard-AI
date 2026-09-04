"""
VoiceGuard AI - Comprehensive Verification Test Suite
SIH26104: Production Verification for Deepfake Detection, Probability Calibration,
Biometric Speaker Verification, Telephony Degradation & Carrier Security.
"""

import os
import sys
import io
import unittest
import numpy as np
import soundfile as sf
import torch
import tempfile
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
from core.calibrator import LearnedRiskCalibrator
from core.speaker_verifier import SpeakerVerifier
from core.telephony_degradation import TelephonyDegradationPipeline
from api.security import rate_limiter
from api.server import app


class TestAudioPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = AudioPreprocessor(target_sr=16000, target_duration=4.0)

    def test_normalize_audio(self):
        audio = np.array([0.1, -0.5, 0.8, -0.2], dtype=np.float32)
        norm = self.preprocessor.normalize_audio(audio, target_peak=0.95)
        self.assertAlmostEqual(float(np.max(np.abs(norm))), 0.95, places=2)
        self.assertAlmostEqual(float(np.mean(norm)), 0.0, places=2)

    def test_pad_or_truncate_with_crossfade(self):
        # Test short 1.0s clip repeat
        short_audio = np.ones(16000, dtype=np.float32) * 0.5
        padded = self.preprocessor.pad_or_truncate(short_audio, target_length=64000)
        self.assertEqual(len(padded), 64000)

        # Test long clip truncation
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


class TestLearnedCalibrator(unittest.TestCase):
    def setUp(self):
        self.calibrator = LearnedRiskCalibrator(calibration_method="sigmoid")

    def test_fit_and_calibrate(self):
        # Create synthetic validation meta-features
        np.random.seed(42)
        n_samples = 40
        y = np.random.randint(0, 2, n_samples)
        # Fake features correlate with label
        X = np.zeros((n_samples, 7), dtype=np.float32)
        for i in range(n_samples):
            p_val = 0.85 if y[i] == 1 else 0.15
            X[i] = [p_val, p_val, p_val * 0.8, p_val * 0.7, p_val * 0.6, p_val * 0.5, 0.8]

        metrics = self.calibrator.fit(X, y)
        self.assertIn("brier_score", metrics)
        self.assertIn("calibrated_ece", metrics)
        self.assertTrue(self.calibrator.is_fitted)

        prob, conf = self.calibrator.predict_calibrated_probability(
            deep_prob=0.9, baseline_prob=0.85, forensic_signals={"hf_anomaly_score": 0.8}, duration=3.0
        )
        self.assertGreaterEqual(prob, 0.5)
        self.assertGreaterEqual(conf, 0.5)


class TestSpeakerVerifier(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.verifier = SpeakerVerifier(profiles_dir=self.temp_dir.name)
        self.speech_a = generate_human_speech_sample(duration=3.0, accent_seed=42)
        self.speech_b = generate_synthetic_speech_sample(duration=3.0, accent_seed=99)
        self.verifier.enroll_speaker(
            speaker_id="test_exec",
            name="Test Executive",
            audio_input=self.speech_a,
            role="CEO",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extract_voiceprint_embedding(self):
        emb = self.verifier.extract_voiceprint_embedding(self.speech_a)
        self.assertEqual(len(emb), 128)
        # Verify L2 unit normalization
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_enrollment_and_verification(self):
        self.assertIn("test_exec", self.verifier.enrolled_profiles)

        # Same audio verified against enrolled profile -> MATCH
        match_res = self.verifier.verify_claimed_identity(self.speech_a, "test_exec")
        self.assertTrue(match_res["verified"])
        self.assertGreaterEqual(match_res["similarity_score"], 0.78)

        # Different audio -> LOWER similarity
        mismatch_res = self.verifier.verify_claimed_identity(self.speech_b, "test_exec")
        self.assertLess(mismatch_res["similarity_score"], match_res["similarity_score"])

    def test_dual_factor_matrix(self):
        # Case 1: Genuine + Matched -> Authorized
        deepfake_safe = {"synthetic_probability": 0.08, "risk_score": 8.0}
        res_auth = self.verifier.evaluate_dual_factor_transaction(
            deepfake_risk_result=deepfake_safe,
            claimed_speaker_id="test_exec",
            audio_input=self.speech_a,
        )
        self.assertEqual(res_auth["transaction_decision"], "AUTHORIZED_DUAL_FACTOR")

        # Case 2: Synthetic clone targeting CEO -> Blocked
        deepfake_danger = {"synthetic_probability": 0.92, "risk_score": 92.0}
        res_blocked = self.verifier.evaluate_dual_factor_transaction(
            deepfake_risk_result=deepfake_danger,
            claimed_speaker_id="test_exec",
            audio_input=self.speech_a,
        )
        self.assertEqual(res_blocked["transaction_decision"], "BLOCKED_AI_IMPERSONATION")


class TestTelephonyDegradation(unittest.TestCase):
    def setUp(self):
        self.degrader = TelephonyDegradationPipeline()
        self.audio = generate_human_speech_sample(duration=2.0)

    def test_g711_mulaw(self):
        degraded = self.degrader.encode_decode_g711_mulaw(self.audio, sr=16000)
        self.assertEqual(len(degraded), len(self.audio))
        self.assertFalse(np.isnan(degraded).any())

    def test_amr_narrowband(self):
        degraded = self.degrader.apply_amr_narrowband_filter(self.audio, sr=16000)
        self.assertEqual(len(degraded), len(self.audio))
        self.assertFalse(np.isnan(degraded).any())

    def test_packet_loss(self):
        degraded = self.degrader.apply_voip_packet_loss(self.audio, sr=16000, packet_loss_rate=0.1)
        self.assertEqual(len(degraded), len(self.audio))


class TestFastAPIServer(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["version"], "2.0.0")

    def test_models_endpoint(self):
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("enrolled_profiles", data)

    def test_sample_audios_endpoint(self):
        response = self.client.get("/api/sample-audios")
        self.assertEqual(response.status_code, 200)
        self.assertIn("samples", response.json())

    def test_voiceprint_profiles_endpoint(self):
        response = self.client.get("/api/voiceprint/profiles")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("profiles", data)

    def test_analyze_with_speaker_and_codec_simulation(self):
        audio = generate_human_speech_sample(duration=3.0)
        buf = io.BytesIO()
        sf.write(buf, audio, 16000, format="WAV")
        buf.seek(0)

        response = self.client.post(
            "/api/analyze",
            headers={"X-API-Key": "voiceguard-enterprise-demo-key-2026"},
            files={"file": ("test_exec_speech.wav", buf.read(), "audio/wav")},
            data={
                "threshold_low": "30.0",
                "threshold_high": "70.0",
                "claimed_speaker_id": "ceo_vikram",
                "simulate_codec": "g711_mulaw",
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("speaker_verification", data)
        self.assertEqual(data["codec_simulation_applied"], "g711_mulaw")
        self.assertIn("calibration", data["analysis"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
