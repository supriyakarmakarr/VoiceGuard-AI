"""
VoiceGuard AI - Forensic Feature Extractor
Extracts multi-domain acoustic, spectral, prosodic, and vocoder forensic features:
1. Tabular features for Baseline ML (MFCCs, Deltas, Spectral Rolloff, Flux, ZCR, F0, HNR, Vocoder Ratio)
2. 2D Mel-Spectrogram tensors for Deep Learning CNN
"""

import numpy as np
import librosa
import scipy.signal
import scipy.stats


class FeatureExtractor:
    """
    Forensic Audio Feature Extractor for synthetic and cloned voice detection.
    """

    def __init__(
        self,
        sr: int = 16000,
        n_mfcc: int = 20,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 256,
    ):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract_tabular_features(self, audio: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """
        Extracts 1D vector of statistical forensic features for Baseline ML models (Random Forest, XGBoost).
        Returns (feature_vector: np.ndarray, feature_names: list[str]).
        """
        features = []
        feature_names = []

        if len(audio) < 512:
            # Pad if too short
            audio = np.pad(audio, (0, 512 - len(audio)))

        # 1. MFCCs + Deltas
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length
        )
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        for i in range(self.n_mfcc):
            features.append(float(np.mean(mfcc[i])))
            feature_names.append(f"mfcc_{i+1}_mean")
            features.append(float(np.std(mfcc[i])))
            feature_names.append(f"mfcc_{i+1}_std")

            features.append(float(np.mean(mfcc_delta[i])))
            feature_names.append(f"mfcc_delta_{i+1}_mean")
            features.append(float(np.std(mfcc_delta[i])))
            feature_names.append(f"mfcc_delta_{i+1}_std")

            features.append(float(np.mean(mfcc_delta2[i])))
            feature_names.append(f"mfcc_delta2_{i+1}_mean")
            features.append(float(np.std(mfcc_delta2[i])))
            feature_names.append(f"mfcc_delta2_{i+1}_std")

        # 2. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features.extend([float(np.mean(centroid)), float(np.std(centroid))])
        feature_names.extend(["spectral_centroid_mean", "spectral_centroid_std"])

        # 3. Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        features.extend([float(np.mean(bandwidth)), float(np.std(bandwidth))])
        feature_names.extend(["spectral_bandwidth_mean", "spectral_bandwidth_std"])

        # 4. Spectral Rolloff (85% & 95% - vocoder cutoff detector)
        rolloff_85 = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, roll_percent=0.85
        )
        rolloff_95 = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, roll_percent=0.95
        )
        features.extend([
            float(np.mean(rolloff_85)),
            float(np.std(rolloff_85)),
            float(np.mean(rolloff_95)),
            float(np.std(rolloff_95)),
        ])
        feature_names.extend([
            "rolloff_85_mean", "rolloff_85_std",
            "rolloff_95_mean", "rolloff_95_std"
        ])

        # 5. Spectral Flatness & Flux
        flatness = librosa.feature.spectral_flatness(y=audio, n_fft=self.n_fft, hop_length=self.hop_length)
        features.extend([float(np.mean(flatness)), float(np.std(flatness))])
        feature_names.extend(["spectral_flatness_mean", "spectral_flatness_std"])

        # Spectral Flux (onset strength)
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        features.extend([float(np.mean(onset_env)), float(np.std(onset_env))])
        feature_names.extend(["spectral_flux_mean", "spectral_flux_std"])

        # 6. Zero Crossing Rate (ZCR) & RMS Energy
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=self.n_fft, hop_length=self.hop_length)
        rms = librosa.feature.rms(y=audio, frame_length=self.n_fft, hop_length=self.hop_length)
        features.extend([
            float(np.mean(zcr)), float(np.std(zcr)),
            float(np.mean(rms)), float(np.std(rms)),
            float(np.max(rms) - np.min(rms))  # Dynamic range
        ])
        feature_names.extend(["zcr_mean", "zcr_std", "rms_mean", "rms_std", "rms_dynamic_range"])

        # 7. High-Frequency Vocoder Energy Ratio (>4000 Hz / <4000 Hz)
        stft = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        idx_4k = np.searchsorted(freqs, 4000)

        low_freq_energy = np.sum(stft[:idx_4k, :]) + 1e-8
        high_freq_energy = np.sum(stft[idx_4k:, :]) + 1e-8
        vocoder_ratio = float(high_freq_energy / low_freq_energy)
        features.append(vocoder_ratio)
        feature_names.append("vocoder_hf_energy_ratio")

        # 8. Pitch (F0) & Prosodic Variations
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=self.sr, frame_length=self.n_fft
            )
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 2:
                f0_mean = float(np.mean(valid_f0))
                f0_std = float(np.std(valid_f0))
                # Jitter approximation: average absolute difference between consecutive pitches / mean pitch
                pitch_diffs = np.abs(np.diff(valid_f0))
                jitter = float(np.mean(pitch_diffs) / (f0_mean + 1e-6))
                voiced_ratio = float(np.sum(voiced_flag) / len(voiced_flag))
            else:
                f0_mean, f0_std, jitter, voiced_ratio = 150.0, 20.0, 0.02, 0.5
        except Exception:
            f0_mean, f0_std, jitter, voiced_ratio = 150.0, 20.0, 0.02, 0.5

        features.extend([f0_mean, f0_std, jitter, voiced_ratio])
        feature_names.extend(["f0_mean", "f0_std", "f0_jitter_ratio", "voiced_frame_ratio"])

        # Clean NaN/Inf if any
        feat_arr = np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        return feat_arr, feature_names

    def extract_mel_spectrogram(
        self, audio: np.ndarray, target_time_frames: int = 128
    ) -> np.ndarray:
        """
        Extracts 2D Mel Spectrogram (dB scaled and normalized) for Deep Learning CNN.
        Output shape: (1, n_mels, target_time_frames) -> (1, 128, 128)
        """
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)))

        # Compute Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )

        # Convert to log-mel (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [-1, 1] range
        min_db = -80.0
        mel_norm = np.clip((mel_spec_db - min_db) / (-min_db), 0.0, 1.0) * 2.0 - 1.0

        # Adjust time dimension to fixed target_time_frames
        curr_frames = mel_norm.shape[1]
        if curr_frames == target_time_frames:
            spec_fixed = mel_norm
        elif curr_frames > target_time_frames:
            start = (curr_frames - target_time_frames) // 2
            spec_fixed = mel_norm[:, start : start + target_time_frames]
        else:
            pad_width = target_time_frames - curr_frames
            spec_fixed = np.pad(mel_norm, ((0, 0), (0, pad_width)), mode="constant", constant_values=-1.0)

        # Add channel dimension -> (1, 128, 128)
        return spec_fixed[np.newaxis, :, :].astype(np.float32)

    def compute_forensic_signals(self, audio: np.ndarray) -> dict:
        """
        Computes granular forensic diagnostic metrics for the Explainability Engine.
        """
        tabular_feats, names = self.extract_tabular_features(audio)
        feat_dict = dict(zip(names, tabular_feats))

        # 1. High-frequency energy anomaly
        hf_ratio = float(feat_dict.get("vocoder_hf_energy_ratio", 0.05))
        # Neural vocoders often display an unnatural high frequency drop-off or unnatural high peak
        hf_anomaly_score = float(np.clip(abs(hf_ratio - 0.08) * 10.0, 0.0, 1.0))

        # 2. Prosodic Monotonicity / Jitter Anomaly
        f0_std = float(feat_dict.get("f0_std", 25.0))
        jitter = float(feat_dict.get("f0_jitter_ratio", 0.03))
        # Extreme flatness (std < 10) or extreme robotic jitter (jitter > 0.08)
        prosody_anomaly = float(np.clip((15.0 - min(f0_std, 15.0)) / 15.0 + max(0.0, jitter - 0.04) * 5.0, 0.0, 1.0))

        # 3. Spectral Rolloff anomaly
        rolloff_95 = float(feat_dict.get("rolloff_95_mean", 6000.0))
        # TTS models typically cut off at 7.5kHz or 4kHz
        spectral_cutoff_score = float(1.0 - np.clip(rolloff_95 / 7500.0, 0.0, 1.0))

        # 4. Spectral Flatness (measure of synthetic hiss or over-smoothing)
        flatness = float(feat_dict.get("spectral_flatness_mean", 0.001))
        flatness_score = float(np.clip(flatness * 200.0, 0.0, 1.0))

        return {
            "hf_vocoder_ratio": hf_ratio,
            "hf_anomaly_score": hf_anomaly_score,
            "prosody_anomaly_score": prosody_anomaly,
            "spectral_cutoff_score": spectral_cutoff_score,
            "spectral_flatness_score": flatness_score,
            "f0_mean_hz": float(feat_dict.get("f0_mean", 150.0)),
            "f0_std_hz": f0_std,
            "zcr_mean": float(feat_dict.get("zcr_mean", 0.05)),
            "voiced_ratio": float(feat_dict.get("voiced_frame_ratio", 0.5)),
        }
