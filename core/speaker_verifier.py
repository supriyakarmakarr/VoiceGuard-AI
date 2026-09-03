"""
VoiceGuard AI - Biometric Speaker Verification & Voiceprint Engine
Solves the CEO-Fraud / Bank-Transfer Identity Gap:
1. Extracts 128-dimensional acoustic biometric voiceprint embeddings (MFCCs, Formants, F0 dynamics)
2. Enrolls and stores trusted speaker voiceprints (e.g. CEO, CFO, Family members)
3. Computes Cosine Similarity against claimed speaker identity
4. Evaluates Dual-Factor Decision Matrix:
   - Genuine Voice + Matched Voiceprint   -> AUTHORIZED
   - Genuine Voice + Mismatched Voiceprint -> REJECTED (Human Impersonator)
   - Synthetic Voice (Any)                -> BLOCKED (Deepfake Attack)
"""

import os
import json
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional

from .audio_preprocessor import AudioPreprocessor


class SpeakerVerifier:
    """
    Biometric voiceprint extraction and verification engine.
    """

    DEFAULT_SIMILARITY_THRESHOLD = 0.78  # Cosine similarity threshold for identity match

    def __init__(self, profiles_dir: str = None, threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.preprocessor = AudioPreprocessor()
        self.threshold = threshold
        self.profiles_dir = profiles_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "voiceprints"
        )
        os.makedirs(self.profiles_dir, exist_ok=True)
        self.enrolled_profiles: Dict[str, dict] = {}
        self.load_all_profiles()

    def extract_voiceprint_embedding(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extracts a 128-dimensional biometric voiceprint embedding vector from speech audio.
        Combines acoustic timbre, vocal tract resonance (formants), and pitch dynamics.
        """
        if len(audio) < 1600:
            audio = np.pad(audio, (0, 1600 - len(audio)))

        embedding = []

        # 1. 24 MFCC Means, Standard Deviations, and Skewness (72 dims)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=24, n_fft=1024, hop_length=256)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        # Numerical skewness
        diff = mfcc - mfcc_mean[:, np.newaxis]
        mfcc_skew = np.mean(diff ** 3, axis=1) / (mfcc_std ** 3 + 1e-6)

        embedding.extend(mfcc_mean[:20])
        embedding.extend(mfcc_std[:20])
        embedding.extend(mfcc_skew[:16])  # 56 dims

        # 2. Spectral Formants & Timbre Centroids (24 dims)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)  # 7 bands x T

        embedding.append(float(np.mean(centroid)))
        embedding.append(float(np.std(centroid)))
        embedding.append(float(np.mean(bandwidth)))
        embedding.append(float(np.std(bandwidth)))
        embedding.extend(np.mean(contrast, axis=1))  # 7 dims
        embedding.extend(np.std(contrast, axis=1))   # 7 dims
        embedding.extend([float(np.percentile(centroid, p)) for p in [25, 50, 75, 90]])  # 4 dims
        embedding.extend([float(np.percentile(bandwidth, p)) for p in [25, 75]])         # 2 dims (Total: 24)

        # 3. Fundamental Pitch (F0) & Prosodic Distribution (24 dims)
        try:
            f0, _, _ = librosa.pyin(audio, fmin=65, fmax=400, sr=sr, frame_length=1024)
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 5:
                p_stats = [
                    float(np.mean(valid_f0)),
                    float(np.std(valid_f0)),
                    float(np.median(valid_f0)),
                    float(np.percentile(valid_f0, 10)),
                    float(np.percentile(valid_f0, 90)),
                    float(np.max(valid_f0) - np.min(valid_f0)),
                ]
            else:
                p_stats = [150.0, 20.0, 150.0, 120.0, 180.0, 60.0]
        except Exception:
            p_stats = [150.0, 20.0, 150.0, 120.0, 180.0, 60.0]

        embedding.extend(p_stats)  # 6 dims

        # Chroma energy distribution across 12 pitch classes (12 dims)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=1024, hop_length=256)
        chroma_mean = np.mean(chroma, axis=1)  # 12 dims
        embedding.extend(chroma_mean)

        # Spectral flatness & rolloff percentiles (6 dims)
        flatness = librosa.feature.spectral_flatness(y=audio)[0]
        embedding.extend([
            float(np.mean(flatness)), float(np.std(flatness)), float(np.percentile(flatness, 90)),
            float(librosa.feature.zero_crossing_rate(audio)[0].mean()),
            float(librosa.feature.rms(y=audio)[0].mean()),
            float(librosa.feature.rms(y=audio)[0].std()),
        ])  # 6 dims

        # 4. Fill or truncate to exact 128 dimensions
        emb_arr = np.nan_to_num(np.array(embedding, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        if len(emb_arr) < 128:
            emb_arr = np.pad(emb_arr, (0, 128 - len(emb_arr)))
        else:
            emb_arr = emb_arr[:128]

        # L2-normalize to unit hypersphere
        norm = np.linalg.norm(emb_arr) + 1e-8
        emb_normalized = emb_arr / norm
        return emb_normalized.astype(np.float32)

    def enroll_speaker(
        self,
        speaker_id: str,
        name: str,
        audio_input,
        role: str = "EXECUTIVE",
        department: str = "Corporate Treasury",
    ) -> dict:
        """
        Enrolls a new trusted speaker voiceprint profile.
        """
        proc = self.preprocessor.process(audio_input)
        embedding = self.extract_voiceprint_embedding(proc["audio"])

        profile = {
            "speaker_id": speaker_id,
            "name": name,
            "role": role,
            "department": department,
            "sample_duration_sec": proc["duration_sec"],
            "embedding": embedding.tolist(),
            "enrolled_at": json.dumps(os.path.getmtime(self.profiles_dir)) if os.path.exists(self.profiles_dir) else "",
        }

        # Save profile JSON
        profile_path = os.path.join(self.profiles_dir, f"{speaker_id}.json")
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)

        self.enrolled_profiles[speaker_id] = profile
        print(f"[SpeakerVerifier] Enrolled voiceprint for '{name}' ({role}) -> ID: {speaker_id}")
        return {
            "success": True,
            "speaker_id": speaker_id,
            "name": name,
            "role": role,
            "profile_path": profile_path,
        }

    def verify_claimed_identity(
        self,
        audio_input,
        claimed_speaker_id: str,
    ) -> dict:
        """
        Computes Cosine Similarity between incoming audio and claimed speaker profile.
        """
        if claimed_speaker_id not in self.enrolled_profiles:
            return {
                "verified": False,
                "similarity_score": 0.0,
                "threshold": self.threshold,
                "claimed_speaker": claimed_speaker_id,
                "error": f"Speaker ID '{claimed_speaker_id}' not enrolled in biometric vault.",
            }

        proc = self.preprocessor.process(audio_input)
        test_emb = self.extract_voiceprint_embedding(proc["audio"])

        ref_profile = self.enrolled_profiles[claimed_speaker_id]
        ref_emb = np.array(ref_profile["embedding"], dtype=np.float32)

        # Cosine similarity: (u . v) / (|u| * |v|)
        dot_product = float(np.dot(test_emb, ref_emb))
        similarity = round(float(np.clip(dot_product, -1.0, 1.0)), 4)
        is_match = similarity >= self.threshold

        return {
            "verified": is_match,
            "similarity_score": similarity,
            "threshold": self.threshold,
            "claimed_speaker_id": claimed_speaker_id,
            "claimed_name": ref_profile.get("name", claimed_speaker_id),
            "claimed_role": ref_profile.get("role", "Executive"),
            "confidence_pct": round(max(0.0, min(100.0, (similarity / self.threshold) * 85.0)), 1),
        }

    def evaluate_dual_factor_transaction(
        self,
        deepfake_risk_result: dict,
        claimed_speaker_id: Optional[str] = None,
        audio_input = None,
    ) -> dict:
        """
        Dual-Factor Security Decision:
        Combines AI Synthetic Deepfake Risk + Biometric Voiceprint Verification.
        """
        synth_prob = deepfake_risk_result.get("synthetic_probability", 0.5)
        is_synthetic = synth_prob >= 0.50

        if not claimed_speaker_id:
            # Only synthetic detection requested
            return {
                "transaction_decision": "BLOCKED" if is_synthetic else "APPROVED_WITHOUT_SPEAKER_CHECK",
                "reason": "Synthetic voice clone detected" if is_synthetic else "Voice appears authentic (no claimed speaker)",
                "speaker_check_performed": False,
            }

        # Run speaker verification
        verification = self.verify_claimed_identity(audio_input, claimed_speaker_id)
        is_matched = verification.get("verified", False)
        similarity = verification.get("similarity_score", 0.0)

        if is_synthetic:
            decision = "BLOCKED_AI_IMPERSONATION"
            reason = f"CRITICAL: High-risk synthetic voice clone mimicking {verification.get('claimed_name')}! Transaction blocked."
            status_color = "#ef4444"
        elif not is_matched:
            decision = "BLOCKED_VOICE_MISMATCH"
            reason = f"REJECTED: Natural voice detected, but biometric voiceprint does NOT match claimed identity ({verification.get('claimed_name')}). Possible human imposter."
            status_color = "#f59e0b"
        else:
            decision = "AUTHORIZED_DUAL_FACTOR"
            reason = f"AUTHORIZED: Both authentic voice dynamics and biometric voiceprint confirmed for {verification.get('claimed_name')}."
            status_color = "#10b981"

        return {
            "transaction_decision": decision,
            "authorized": decision == "AUTHORIZED_DUAL_FACTOR",
            "reason": reason,
            "status_color": status_color,
            "speaker_check_performed": True,
            "verification_details": verification,
            "deepfake_risk_score": deepfake_risk_result.get("risk_score", 50.0),
        }

    def load_all_profiles(self):
        """Loads all enrolled JSON profiles from disk."""
        self.enrolled_profiles.clear()
        if not os.path.exists(self.profiles_dir):
            return

        for fname in os.listdir(self.profiles_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(self.profiles_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        prof = json.load(f)
                        self.enrolled_profiles[prof["speaker_id"]] = prof
                except Exception as e:
                    print(f"Error loading voiceprint {fname}: {e}")

    def list_profiles(self) -> List[dict]:
        """Returns metadata of all enrolled profiles (omitting raw embeddings for brevity)."""
        profiles = []
        for pid, p in self.enrolled_profiles.items():
            profiles.append({
                "speaker_id": pid,
                "name": p.get("name", pid),
                "role": p.get("role", "User"),
                "department": p.get("department", "Default"),
                "sample_duration_sec": p.get("sample_duration_sec", 4.0),
            })
        return profiles
