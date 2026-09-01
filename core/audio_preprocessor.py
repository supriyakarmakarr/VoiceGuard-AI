"""
VoiceGuard AI - Audio Preprocessor
Standardizes audio files/streams to 16kHz Mono WAV, applies normalization,
silence trimming, and segment framing for ML and Deep Learning feature extraction.
"""

import io
import os
import wave
import numpy as np
import soundfile as sf
import librosa

TARGET_SAMPLE_RATE = 16000
DEFAULT_DURATION_SEC = 4.0  # standard fixed-window for deep learning (64000 samples)
TARGET_SAMPLE_LENGTH = int(TARGET_SAMPLE_RATE * DEFAULT_DURATION_SEC)


class AudioPreprocessor:
    """
    Standardized audio preprocessing pipeline for VoiceGuard AI.
    Handles multi-format audio loading, mono conversion, 16kHz resampling,
    normalization, silence removal, and framing.
    """

    def __init__(self, target_sr: int = TARGET_SAMPLE_RATE, target_duration: float = DEFAULT_DURATION_SEC):
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_samples = int(target_sr * target_duration)

    def load_audio(self, audio_source, sr: int = None) -> tuple[np.ndarray, int]:
        """
        Loads audio from a file path, file-like object, or bytes buffer.
        Returns (audio_data: np.ndarray, sample_rate: int) in float32.
        """
        if sr is None:
            sr = self.target_sr

        if isinstance(audio_source, (str, os.PathLike)):
            try:
                y, orig_sr = librosa.load(audio_source, sr=sr, mono=True)
                return y.astype(np.float32), sr
            except Exception:
                data, orig_sr = sf.read(audio_source, dtype="float32")
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if orig_sr != sr:
                    data = librosa.resample(data, orig_sr=orig_sr, target_sr=sr)
                return data.astype(np.float32), sr

        elif isinstance(audio_source, bytes):
            byte_io = io.BytesIO(audio_source)
            # 1. Try SoundFile
            try:
                data, orig_sr = sf.read(byte_io, dtype="float32")
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if orig_sr != sr:
                    data = librosa.resample(data, orig_sr=orig_sr, target_sr=sr)
                return data.astype(np.float32), sr
            except Exception:
                pass

            # 2. Try scipy.io.wavfile
            try:
                byte_io.seek(0)
                import scipy.io.wavfile as wavfile
                orig_sr, data = wavfile.read(byte_io)
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                elif data.dtype != np.float32:
                    data = data.astype(np.float32)
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if orig_sr != sr:
                    data = librosa.resample(data, orig_sr=orig_sr, target_sr=sr)
                return data.astype(np.float32), sr
            except Exception:
                pass

            # 3. Try torchaudio
            try:
                byte_io.seek(0)
                import torchaudio
                tensor, orig_sr = torchaudio.load(byte_io)
                data = tensor.mean(dim=0).numpy().astype(np.float32)
                if orig_sr != sr:
                    data = librosa.resample(data, orig_sr=orig_sr, target_sr=sr)
                return data, sr
            except Exception:
                pass

            # 4. Try librosa
            try:
                byte_io.seek(0)
                y, orig_sr = librosa.load(byte_io, sr=sr, mono=True)
                return y.astype(np.float32), sr
            except Exception:
                pass

            # 5. Raw PCM int16 fallback
            try:
                data = np.frombuffer(audio_source, dtype=np.int16).astype(np.float32) / 32768.0
                if len(data) > 100:
                    return data, sr
            except Exception:
                pass

            raise ValueError("Unable to decode audio bytes with available decoders.")

        elif isinstance(audio_source, np.ndarray):
            y = audio_source.astype(np.float32)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            return y, sr

        else:
            raise ValueError(f"Unsupported audio source type: {type(audio_source)}")

    def normalize_audio(self, audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """
        Removes DC offset and normalizes peak amplitude to target_peak.
        """
        if len(audio) == 0:
            return audio

        # Remove DC offset
        audio = audio - np.mean(audio)

        # Peak normalization
        max_val = np.max(np.abs(audio))
        if max_val > 1e-6:
            audio = audio * (target_peak / max_val)

        return audio.astype(np.float32)

    def trim_silence(self, audio: np.ndarray, top_db: int = 25) -> np.ndarray:
        """
        Trims leading and trailing silence using energy threshold.
        Preserves non-silent speech content.
        """
        if len(audio) < 1600:
            return audio
        try:
            trimmed, _ = librosa.effects.trim(audio, top_db=top_db, frame_length=512, hop_length=128)
            if len(trimmed) < 1600:
                return audio
            return trimmed
        except Exception:
            return audio

    def pad_or_truncate(self, audio: np.ndarray, target_length: int = None) -> np.ndarray:
        """
        Pads with repeat/reflection or truncates audio to exact target length in samples.
        """
        if target_length is None:
            target_length = self.target_samples

        curr_len = len(audio)
        if curr_len == target_length:
            return audio
        elif curr_len > target_length:
            start = (curr_len - target_length) // 2
            return audio[start : start + target_length]
        else:
            if curr_len > 0:
                reps = int(np.ceil(target_length / curr_len))
                repeated = np.tile(audio, reps)
                return repeated[:target_length]
            else:
                return np.zeros(target_length, dtype=np.float32)

    def apply_preemphasis(self, audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
        """
        Applies pre-emphasis filter to boost high frequencies for spectral forensics.
        y[t] = x[t] - coeff * x[t-1]
        """
        if len(audio) <= 1:
            return audio
        return np.append(audio[0], audio[1:] - coeff * audio[:-1]).astype(np.float32)

    def process(self, audio_source, target_duration: float = None) -> dict:
        """
        Full standardized pipeline:
        1. Load & convert to Mono 16kHz Float32
        2. DC removal & Peak Normalization
        3. Silence Trimming
        4. Fixed-duration windowing
        Returns dictionary with processed audio, raw trimmed audio, duration, and sample rate.
        """
        target_len = int(self.target_sr * (target_duration or self.target_duration))
        raw_audio, sr = self.load_audio(audio_source, sr=self.target_sr)
        norm_audio = self.normalize_audio(raw_audio)
        trimmed_audio = self.trim_silence(norm_audio)
        fixed_audio = self.pad_or_truncate(trimmed_audio, target_length=target_len)

        return {
            "audio": fixed_audio,
            "raw_trimmed": trimmed_audio,
            "sr": self.target_sr,
            "duration_sec": float(len(trimmed_audio) / self.target_sr),
            "num_samples": len(fixed_audio),
        }

    def save_wav(self, audio: np.ndarray, output_path: str, sr: int = None):
        """
        Saves standard 16kHz mono WAV file.
        """
        if sr is None:
            sr = self.target_sr
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sf.write(output_path, audio, sr, subtype="PCM_16")
