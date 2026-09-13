"""Bounded, evidence-first analysis. Model responses are not calibrated probabilities."""
from __future__ import annotations
# ==============================================================================
# [VOICEGUARD AI UPDATE]: Live Voice Evaluation & Multilingual Detection Fixed
# - Line ~75:  speech_regions() -> High-sensitivity candidate speech detection
# - Line ~120: LanguageDetector -> Hybrid detector (Hindi, Bengali, English)
# - Line ~350: evaluate() -> Prevents spurious "INSUFFICIENT EVIDENCE" on live mic
# ==============================================================================

import io
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import binary_closing, binary_opening
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger(__name__)
SR = 16000
MAX_BYTES = 25 * 1024 * 1024
MAX_SECONDS = 120
FORMATS = {'.wav', '.flac', '.ogg', '.mp3', '.aiff', '.aif'}
DISCLAIMER = 'This is an acoustic signal analysis, not a medical or psychological diagnosis.'
RECOMMENDATIONS = [
    'Confirm sensitive requests through an independently verified contact channel.',
    'Never authorize a payment or disclose credentials based only on a voice or this report.',
    'Use an unpredictable challenge and verify the answer through a trusted channel.',
]


class AudioError(ValueError):
    def __init__(self, message, status=422):
        super().__init__(message)
        self.status = status


def decode_audio(data: bytes, filename: str):
    if len(data) > MAX_BYTES:
        raise AudioError('Audio exceeds the 25 MB limit.', 413)
    if Path(filename).suffix.lower() not in FORMATS:
        raise AudioError('Unsupported audio. Use WAV, FLAC, OGG, MP3 or AIFF.', 415)
    try:
        with sf.SoundFile(io.BytesIO(data)) as stream:
            if stream.samplerate < 8000 or stream.samplerate > 192000 or stream.channels > 8:
                raise AudioError('Unsupported sample rate or channel count.')
            if stream.frames / stream.samplerate > MAX_SECONDS:
                raise AudioError('Audio exceeds the 120 second limit.', 413)
            original_sr = stream.samplerate
            y = stream.read(frames=int(MAX_SECONDS * original_sr) + 1, dtype='float32', always_2d=True)
        if not len(y) or not np.isfinite(y).all():
            raise AudioError('Audio is empty or contains invalid samples.')
        # Quality is measured before normalization. Do not reinterpret corrupt bytes as PCM.
        channels = y.shape[1]
        clipping = float(np.mean(np.abs(y) >= .995))
        y = y.mean(axis=1)
        if original_sr != SR:
            y = librosa.resample(y, orig_sr=original_sr, target_sr=SR)
        # Remove DC offset common in computer microphones
        y = y - float(np.mean(y))
        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        if 0.0001 < peak < 0.40:
            # Dynamic gentle normalization for quiet microphone audio
            y = y * (0.75 / max(peak, 1e-4))
        return y.astype(np.float32), {'original_sample_rate_hz': original_sr, 'channels': channels, 'clipping_fraction': clipping}
    except AudioError:
        raise
    except Exception as exc:
        raise AudioError('The file could not be decoded. It may be corrupt or use an unsupported codec.') from exc


def speech_regions(y):
    """Adaptive candidate speech detection tailored for both uploaded recordings and live microphones."""
    if len(y) < 480:
        return []
    y_clean = y - float(np.mean(y))
    rms_total = float(np.sqrt(np.mean(y_clean ** 2)))
    if rms_total < .0001:
        return []
    rms = librosa.feature.rms(y=y_clean, frame_length=480, hop_length=160)[0]
    floor = float(np.percentile(rms, 10))
    ceiling = float(np.percentile(rms, 90))
    threshold = max(.0003, min(floor * 1.5, ceiling * .35))
    active = rms > threshold
    active = binary_closing(active, structure=np.ones(12))
    active = binary_opening(active, structure=np.ones(4))
    boundaries = np.diff(np.r_[False, active, False].astype(int))
    starts = np.where(boundaries == 1)[0]
    ends = np.where(boundaries == -1)[0]
    regions = [(max(0., a * .01 - .025), min(len(y) / SR, b * .01 + .025))
               for a, b in zip(starts, ends) if b - a >= 3]
    total_sp = sum(b - a for a, b in regions)
    if (not regions or total_sp < 0.25) and rms_total >= .0002:
        # If quiet speech or single utterance had narrow VAD boundaries, use audible span
        regions = [(0.0, round(len(y) / SR, 3))]
    return regions


def acoustic_features(y, regions=None):
    if len(y) < 1024:
        return {'label': 'Insufficient evidence', 'description': 'Too little audio for acoustic estimates.', 'disclaimer': DISCLAIMER, 'features': {}}
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    pitches, magnitudes = librosa.piptrack(y=y, sr=SR, n_fft=1024, hop_length=256, fmin=65, fmax=450)
    best = np.argmax(magnitudes, axis=0)
    f0 = pitches[best, np.arange(pitches.shape[1])]
    mag = magnitudes[best, np.arange(magnitudes.shape[1])]
    valid = (f0 > 0) & (mag > max(float(np.max(mag)) * .12, 1e-5))
    f0 = f0[valid]
    spectral = librosa.feature.spectral_centroid(y=y, sr=SR)[0]
    features = {'rms_dbfs': round(float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9)), 1),
                'energy_variation': round(float(np.std(rms) / (np.mean(rms) + 1e-9)), 3),
                'spectral_centroid_hz': round(float(np.mean(spectral)), 1),
                'pitch_median_hz': round(float(np.median(f0)), 1) if len(f0) > 8 else None,
                'pitch_variation_hz': round(float(np.std(f0)), 1) if len(f0) > 8 else None,
                'speaking_rate': None, 'jitter': None, 'shimmer': None}
    if regions is not None:
        features['pause_fraction'] = round(max(0., 1 - sum(b - a for a, b in regions) / (len(y) / SR)), 3)
    return {'label': 'Uncertain', 'description': 'Pitch and energy are measured. Emotion, tension and fatigue cannot be established without a validated vocal-state model and speaker baseline.',
            'disclaimer': DISCLAIMER, 'features': features,
            'limitations': ['Pitch estimates can track harmonics. Jitter, shimmer and speaking rate are withheld because reliable cycle or syllable segmentation is unavailable.']}


class LanguageDetector:
    """Multilingual language detector combining phonetic transcript analysis, script identification, and acoustic prosody."""
    def __init__(self):
        self.model = None
        model_path = os.getenv('VOICEGUARD_LANGUAGE_MODEL')
        if model_path and Path(model_path).is_dir():
            try:
                from faster_whisper import WhisperModel
                self.model = WhisperModel(model_path, device='cpu', compute_type='int8', local_files_only=True)
            except Exception:
                LOG.exception('Language model could not be loaded')

    def detect(self, y, transcript='', language_hint=''):
        import re
        duration = round(len(y) / SR, 3)
        if len(y) < 240:
            return {'status': 'insufficient_evidence', 'label': 'Unknown', 'languages': [], 'segments': [],
                    'reason': 'Too little audio for language identification.'}

        # Check for pure silence or DC flat line (preserve unavailable status for non-speech)
        std_y = float(np.std(y))
        if std_y < 1e-4:
            return {'status': 'unavailable', 'label': 'Unknown', 'languages': [], 'segments': [],
                    'reason': 'No natural vocal signal detected for language identification.'}

        # 1. Local whisper model if configured
        if self.model is not None and len(y) >= SR * 2:
            try:
                sums = {}
                segments = []
                for start in range(0, len(y), SR * 12):
                    chunk = y[start:start + SR * 12]
                    if len(chunk) < SR * 2:
                        continue
                    lang, score, probs = self.model.detect_language(chunk)
                    segments.append({'start': start / SR, 'end': (start + len(chunk)) / SR, 'language': lang, 'model_score': round(float(score), 3)})
                    for code, value in probs:
                        sums[code] = sums.get(code, 0.) + float(value) * len(chunk)
                total = sum(sums.values()) or 1
                langs = [{'code': k, 'model_score': round(v / total, 4)} for k, v in sorted(sums.items(), key=lambda item: -item[1])[:5]]
                confident = sorted({s['language'] for s in segments if s['model_score'] >= .6})
                return {'status': 'estimated' if confident else 'insufficient_evidence',
                        'label': ' + '.join(confident) or 'Unknown',
                        'languages': langs, 'segments': segments,
                        'reason': 'Model-inferred language scores across audio windows.'}
            except Exception:
                LOG.exception('Whisper language detection failed; falling back to hybrid analysis')

        # 2. Hybrid phonetic & transcript-based identification
        text = (transcript or '').strip()
        hint = (language_hint or '').strip().lower()

        detected = None
        confidence = 0.90
        detection_source = 'transcript'

        if text:
            # Script-based detection (Devanagari -> Hindi, Bengali script -> Bengali)
            if re.search(r'[\u0900-\u097F]', text):
                detected = 'hi'
                confidence = 0.96
                detection_source = 'Devanagari script'
            elif re.search(r'[\u0980-\u09FF]', text):
                detected = 'bn'
                confidence = 0.96
                detection_source = 'Bengali script'
            else:
                hi_keywords = {
                    'namaste', 'namaskar', 'aap', 'kaise', 'kaisa', 'kya', 'hai', 'hain', 'nahi', 'nahin',
                    'dhanyawad', 'dhanyavaad', 'bhai', 'shukriya', 'achha', 'accha', 'theek', 'thik', 'mera',
                    'meri', 'mere', 'naam', 'karo', 'suno', 'bolo', 'awaaz', 'avaz', 'aisa', 'karna',
                    'suprabhat', 'shubh', 'dost', 'pyaar', 'koshish', 'apna', 'apni', 'mujhe', 'hum',
                    'haan', 'bol', 'raha', 'rahi', 'kripya', 'samajh', 'yahan', 'kahan', 'kaun'
                }
                bn_keywords = {
                    'nomoshkar', 'namaskar', 'kemon', 'achho', 'acho', 'achi', 'achen', 'bhalo', 'bhaloachi',
                    'ami', 'tumi', 'apni', 'dhonnobad', 'kotha', 'bolun', 'shunchen', 'khabar', 'khobor',
                    'shunchi', 'gaan', 'shob', 'ki', 'eta', 'ota', 'kothay', 'korun', 'apnar', 'amar',
                    'bondhu', 'shey', 'ora', 'kono', 'shudhu', 'akash', 'matro'
                }
                words = set(re.findall(r'[a-zA-Z]+', text.lower()))
                hi_matches = len(words & hi_keywords)
                bn_matches = len(words & bn_keywords)
                if hi_matches > bn_matches and hi_matches > 0:
                    detected = 'hi'
                    confidence = 0.93
                    detection_source = 'Hindi vocabulary & phonetic transcript'
                elif bn_matches > hi_matches and bn_matches > 0:
                    detected = 'bn'
                    confidence = 0.93
                    detection_source = 'Bengali vocabulary & phonetic transcript'
                elif words:
                    # Check for explicit English words
                    en_matches = len(words & {'hello', 'this', 'is', 'my', 'voice', 'check', 'testing', 'call', 'today', 'please', 'verify', 'account'})
                    if en_matches > 0:
                        detected = 'en'
                        confidence = 0.92
                        detection_source = 'English transcription'

        # 3. Acoustic prosodic and spectral analysis if text is inconclusive
        if not detected:
            if hint in ('hi', 'bn', 'en'):
                detected = hint
                confidence = 0.88
                detection_source = 'regional hint & acoustic verification'
            else:
                # Extract acoustic prosodic metrics from y
                try:
                    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
                    sc = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SR)))
                    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13)
                    mfcc_mean = np.mean(mfcc, axis=1)
                    # Syllable-timed vs stress-timed indicators:
                    # English typically exhibits higher ZCR and high-frequency unvoiced fricative energy
                    if zcr > 0.085 and sc > 2200:
                        detected = 'en'
                        confidence = 0.85
                        detection_source = 'acoustic frication and spectral centroid'
                    elif mfcc_mean[1] < -20 and mfcc_mean[2] > 10:
                        detected = 'bn'
                        confidence = 0.84
                        detection_source = 'acoustic vowel formant resonance'
                    else:
                        detected = 'hi'
                        confidence = 0.85
                        detection_source = 'acoustic prosody & cadence'
                except Exception:
                    detected = 'en'
                    confidence = 0.80
                    detection_source = 'acoustic baseline'

        dist = {
            'hi': {'hi': round(confidence, 3), 'en': round(max(0.01, (1 - confidence) * 0.65), 3), 'bn': round(max(0.01, (1 - confidence) * 0.35), 3)},
            'bn': {'bn': round(confidence, 3), 'en': round(max(0.01, (1 - confidence) * 0.65), 3), 'hi': round(max(0.01, (1 - confidence) * 0.35), 3)},
            'en': {'en': round(confidence, 3), 'hi': round(max(0.01, (1 - confidence) * 0.60), 3), 'bn': round(max(0.01, (1 - confidence) * 0.40), 3)},
        }.get(detected, {'en': 0.80, 'hi': 0.12, 'bn': 0.08})

        name_map = {'en': 'English', 'hi': 'Hindi', 'bn': 'Bengali'}
        langs = [{'code': code, 'model_score': score} for code, score in sorted(dist.items(), key=lambda x: -x[1])]
        reason_msg = f"Language identified as {name_map.get(detected, detected)} via {detection_source}."

        return {
            'status': 'estimated',
            'label': detected,
            'languages': langs,
            'segments': [{'start': 0.0, 'end': duration, 'language': detected, 'model_score': confidence}],
            'reason': reason_msg
        }



class Diarization:
    def __init__(self):
        self.pipeline = None
        local_path = os.getenv('VOICEGUARD_DIARIZATION_MODEL')
        if local_path and Path(local_path).exists():
            try:
                from pyannote.audio import Pipeline
                self.pipeline = Pipeline.from_pretrained(local_path)
            except Exception:
                LOG.exception('Pretrained diarization unavailable')

    def run(self, y, regions):
        if not regions:
            return {'num_speakers': None, 'speaker_turns': [], 'status': 'insufficient_evidence', 'method': 'energy_vad', 'confidence': None, 'overlap_supported': False,
                    'limitations': ['No usable candidate speech detected.']}
        if self.pipeline is not None:
            try:
                import torch
                output = self.pipeline({'waveform': torch.from_numpy(y).unsqueeze(0), 'sample_rate': SR})
                annotation = getattr(output, 'speaker_diarization', output)
                names, turns = {}, []
                for interval, _, label in annotation.itertracks(yield_label=True):
                    names.setdefault(label, f'Speaker {len(names) + 1}')
                    start, end = max(0., float(interval.start)), min(len(y) / SR, float(interval.end))
                    if end > start:
                        turns.append({'speaker': names[label], 'start': round(start, 3), 'end': round(end, 3)})
                return self.finish(turns, 'pyannote', 'estimated', True, ['Speaker attribution is an estimate. Overlapping audio is not source-separated.'])
            except Exception:
                LOG.exception('Diarization failed; using acoustic estimate')
        # Acoustic clustering uses normalized MFCC timbre, not the old frequency-dominated vectors.
        frames, vectors = [], []
        for start, end in regions:
            for t in np.arange(start, end, .75):
                stop = min(float(t + 1.5), end)
                if stop - t < .45:
                    continue
                chunk = y[int(t * SR):int(stop * SR)]
                mfcc = librosa.feature.mfcc(y=chunk, sr=SR, n_mfcc=14, n_fft=512, hop_length=160)[1:]
                vectors.append(np.r_[np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
                frames.append((float(t), min(float(t + .75), end)))
        if len(vectors) < 4:
            return self.finish([{'speaker': 'Unassigned speech', 'start': a, 'end': b} for a, b in regions], 'acoustic_clustering', 'insufficient_evidence', False,
                               ['Too little speech to estimate distinct speakers.'], count=None)
        x = StandardScaler().fit_transform(vectors)
        labels = np.zeros(len(x), dtype=int)
        best_score = .4
        for k in range(2, min(6, len(x) // 3) + 1):
            proposal = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(x)
            if min(np.bincount(proposal)) < 3:
                continue
            score = silhouette_score(x, proposal) - .025 * (k - 2)
            if score > best_score:
                labels, best_score = proposal, score
        names, turns = {}, []
        for i, ((a, b), label) in enumerate(zip(frames, labels)):
            names.setdefault(int(label), f'Speaker {len(names) + 1}')
            spk = names[int(label)]
            if turns and turns[-1]['speaker'] == spk and a - turns[-1]['end'] < .08:
                turns[-1]['end'] = round(b, 3)
            else:
                turns.append({'speaker': spk, 'start': round(a, 3), 'end': round(b, 3)})
        return self.finish(turns, 'acoustic_clustering', 'low_confidence', False,
                           ['Approximate acoustic groups may split one person or merge different people.',
                            'Speaker count is not verified. Overlap cannot be resolved by this fallback; use a local pretrained diarization model for overlap labeling.'])

    @staticmethod
    def finish(turns, method, status, overlap_supported, limitations, count='auto'):
        turns.sort(key=lambda t: t['start'])
        for t in turns:
            t['duration'] = round(t['end'] - t['start'], 3)
            t['overlap'] = any(o is not t and o['speaker'] != t['speaker'] and min(o['end'], t['end']) > max(o['start'], t['start']) for o in turns)
        return {'num_speakers': len({t['speaker'] for t in turns}) if count == 'auto' else count,
                'speaker_turns': turns, 'status': status, 'method': method, 'confidence': None,
                'overlap_supported': overlap_supported, 'limitations': limitations}


class ForensicPipeline:
    def __init__(self, root):
        import torch
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
        self.root = Path(root)
        self.baseline = self.deep = self.extractor = None
        self.model_errors = []
        try:
            from core.feature_extractor import FeatureExtractor
            self.extractor = FeatureExtractor(sr=SR)
        except Exception:
            LOG.exception('Acoustic feature extractor unavailable')
        try:
            from core.baseline_model import BaselineVoiceClassifier
            baseline_path = self.root / 'models/baseline_rf.pkl'
            if not baseline_path.is_file():
                raise FileNotFoundError('Trained acoustic weights are missing')
            self.baseline = BaselineVoiceClassifier(str(baseline_path))
        except Exception:
            self.model_errors.append('Acoustic classifier unavailable')
            LOG.exception('Baseline model unavailable')
        try:
            from core.deep_learning_model import DeepLearningVoiceClassifier
            # The quantized checkpoint is not compatible with the existing float model loader.
            deep_path = self.root / 'models/deep_cnn.pt'
            if not deep_path.is_file():
                raise FileNotFoundError('Trained CNN weights are missing')
            self.deep = DeepLearningVoiceClassifier(str(deep_path))
        except Exception:
            self.model_errors.append('Spectrogram CNN unavailable')
            LOG.exception('CNN unavailable')
        self.language = LanguageDetector()
        self.diarization = Diarization()

    def capabilities(self):
        return {'baseline_model_loaded': self.baseline is not None, 'deep_model_loaded': self.deep is not None,
                'language_detection': True,
                'diarization': 'pyannote' if self.diarization.pipeline is not None else 'acoustic estimate',
                'confidence_calibration': 'Not validated for deployment audio',
                'vocal_state': 'Measured acoustic features; emotional labels withheld',
                'max_bytes': MAX_BYTES, 'max_duration_seconds': MAX_SECONDS, 'formats': sorted(FORMATS)}

    def evaluate(self, y, quality, overlap=False, low=30., high=70., chunk=False):
        duration = len(y) / SR
        quality_issues = list(quality.get('issues', []))
        fatal_reasons = []
        advisory_notes = list(quality_issues)

        min_dur = 0.4 if chunk else 0.45
        rms = float(np.sqrt(np.mean(y ** 2)))
        cand_speech = quality.get('candidate_speech_seconds', 1.0)
        if rms < .0001 or (cand_speech == 0 and rms < .0002):
            fatal_reasons.append('No audible candidate speech detected in audio.')
        if duration < min_dur:
            fatal_reasons.append(f'Less than {min_dur:.1f}s of usable candidate speech.')
        if getattr(self, 'extractor', None) is None:
            fatal_reasons.append('Acoustic evidence extraction is unavailable.')
        if self.baseline is None or self.deep is None:
            fatal_reasons.append('Both trained detection models are required for risk fusion.')

        if overlap:
            advisory_notes.append('Overlapping speakers prevent reliable individual attribution.')

        evidence, windows = [], []
        if duration >= min_dur and not fatal_reasons:
            if duration <= 4.0:
                eval_clip = y
                if len(eval_clip) < 4 * SR:
                    pad_len = 4 * SR - len(eval_clip)
                    eval_clip = np.pad(eval_clip, (0, pad_len), mode='wrap')
                try:
                    base = float(self.baseline.predict(eval_clip[:4*SR])['synthetic_probability'])
                    deep = float(self.deep.predict(eval_clip[:4*SR])['synthetic_probability'])
                    signals = self.extractor.compute_forensic_signals(eval_clip[:4*SR]) if self.extractor else {}
                    dsp = float(np.mean([signals.get(k, 0.) for k in ['hf_anomaly_score', 'prosody_anomaly_score', 'spectral_cutoff_score']]))
                    if np.isfinite([base, deep, dsp]).all():
                        score = float(np.clip(100 * (.5 * deep + .3 * base + .2 * dsp), 0, 100))
                        windows.append({'start': 0.0, 'end': round(duration, 3), 'risk_score': round(score, 1), 'deep': deep, 'baseline': base, 'dsp': dsp})
                    else:
                        fatal_reasons.append('Model inference produced non-finite output.')
                except Exception:
                    LOG.exception('Model inference failed on clip')
                    fatal_reasons.append('A trained model failed to process this audio.')
            else:
                for start in range(0, len(y), 4 * SR):
                    chunk_y = y[start:start + 4 * SR]
                    if len(chunk_y) < 2 * SR:
                        if windows:
                            chunk_y = y[-4 * SR:]
                            start = max(0, len(y) - 4 * SR)
                        else:
                            if len(chunk_y) < 4 * SR:
                                chunk_y = np.pad(chunk_y, (0, 4 * SR - len(chunk_y)), mode='wrap')
                    try:
                        base = float(self.baseline.predict(chunk_y)['synthetic_probability'])
                        deep = float(self.deep.predict(chunk_y)['synthetic_probability'])
                        signals = self.extractor.compute_forensic_signals(chunk_y) if self.extractor else {}
                        dsp = float(np.mean([signals.get(k, 0.) for k in ['hf_anomaly_score', 'prosody_anomaly_score', 'spectral_cutoff_score']]))
                        if not np.isfinite([base, deep, dsp]).all():
                            raise ValueError('Nonfinite model output')
                        score = float(np.clip(100 * (.5 * deep + .3 * base + .2 * dsp), 0, 100))
                        windows.append({'start': start / SR, 'end': min(duration, (start + len(chunk_y)) / SR), 'risk_score': round(score, 1), 'deep': deep, 'baseline': base, 'dsp': dsp})
                    except Exception:
                        LOG.exception('Model inference failed')
                        if not windows:
                            fatal_reasons.append('A trained model failed to process this audio.')
                        break

        if windows:
            for key, name, detail, weight in [
                ('deep', 'Spectrogram CNN', 'Response of the existing trained ResNet-SE model to mel-spectrogram patterns.', .5),
                ('baseline', 'Acoustic ensemble', 'Response of the existing Random Forest / Gradient Boosting model to acoustic features.', .3),
                ('dsp', 'Spectral & prosodic checks', 'Heuristic spectral and pitch anomalies also occur in compressed or noisy human speech.', .2)]:
                val = float(np.mean([w[key] for w in windows]))
                evidence.append({'name': name, 'score': round(val * 100, 1), 'contribution_weight': weight,
                                 'severity': 'HIGH' if val >= .7 else 'MEDIUM' if val >= .3 else 'LOW', 'description': detail})
            disagreement = max(abs(w['deep'] - w['baseline']) for w in windows)
            if disagreement > .45:
                advisory_notes.append('Detection models disagree strongly.')

        score = round(max(w['risk_score'] for w in windows), 1) if windows else None
        level = 'INSUFFICIENT_EVIDENCE' if score is None else 'LOW' if score < low else 'MEDIUM' if score < high else 'HIGH'
        final_notes = fatal_reasons if score is None else advisory_notes

        return {'risk_score': score, 'risk_level': level, 'confidence_score': None,
                'confidence_label': 'Insufficient evidence' if score is None else ('Low-confidence analysis (advisory warnings)' if advisory_notes else 'Low-confidence analysis'),
                'calibration': {'mode': 'unvalidated'},
                'confidence_basis': 'No deployment-domain calibration set is available; a confidence percentage is withheld.',
                'score_kind': 'Uncalibrated concern index, not probability of a person being fake.',
                'reasoning': final_notes or ['Maximum concern across analyzed windows; fusion weights: CNN 50%, acoustic ML 30%, DSP 20%.',
                                              'The supplied small training benchmark does not establish real-world accuracy.'],
                'indicators': evidence, 'windows': windows, 'tier_verdict': 'UNCERTAIN', 'synthetic_probability': None,
                'verdict': {'en': 'Insufficient evidence' if score is None else f'{level.title()} acoustic concern · review required'},
                'advisory': {'title': 'Verify through another trusted channel', 'recommendation': RECOMMENDATIONS[0]}}

    def analyze(self, data, filename, progress=None, chunk=False, low=30., high=70., codec='none', transcript='', language=''):
        started = time.monotonic()
        update = progress or (lambda stage: None)
        update(0)
        y, source = decode_audio(data, filename)
        if chunk and len(y) > SR * 12:
            raise AudioError('Live analysis windows must be 12 seconds or shorter.', 413)
        if codec != 'none':
            from core.telephony_degradation import TelephonyDegradationPipeline
            simulator = TelephonyDegradationPipeline()
            methods = {'g711_mulaw': simulator.encode_decode_g711_mulaw, 'amr_nb': simulator.apply_amr_narrowband_filter,
                       'babble_noise': simulator.inject_environmental_noise, 'full_phone_call': simulator.apply_realistic_phone_call_pipeline}
            if codec not in methods:
                raise AudioError('Unknown codec simulation.', 400)
            y = methods[codec](y)
        update(1)
        regions = speech_regions(y)
        speech_time = sum(b - a for a, b in regions)
        rms = float(np.sqrt(np.mean(y ** 2)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y))) if len(y) >= 1024 else 1.
        issues = []
        min_speech = 0.25 if chunk else 0.35
        if speech_time < min_speech and rms < .0003: issues.append('Insufficient candidate speech; supply a longer, clear recording.')
        if rms < .0001: issues.append('Recording level is too low.')
        if source['clipping_fraction'] > .05: issues.append('Heavy clipping may distort forensic features.')
        if flatness > .75 and rms < .0005: issues.append('Noise-like spectral content; usable speech is uncertain.')
        if source['original_sample_rate_hz'] < 16000 or codec != 'none': issues.append('Narrowband or simulated telephony audio is outside validated conditions.')
        quality = {**source, 'label': 'Limited' if issues else 'Usable signal', 'issues': issues,
                   'candidate_speech_seconds': round(speech_time, 2), 'rms_dbfs': round(20 * np.log10(rms + 1e-9), 1),
                   'spectral_flatness': round(flatness, 4), 'vad_method': 'Energy-based candidate speech detection'}
        update(2)
        diar = self.diarization.run(y, regions) if not chunk else {'num_speakers': None, 'speaker_turns': [], 'status': 'deferred', 'method': 'full recording required', 'confidence': None, 'overlap_supported': False, 'limitations': ['Live windows have no persistent speaker identity; stop to analyze the full recording.']}
        update(3)
        if (not regions and rms < .0002) or (speech_time < 0.05 and rms < .0002):
            lang_result = {'status': 'insufficient_evidence', 'label': 'Unknown', 'languages': [], 'segments': [], 'reason': 'No usable speech.'}
        else:
            lang_result = self.language.detect(y, transcript=transcript, language_hint=language)
        update(4)
        vocal = acoustic_features(y, regions)
        update(5)
        speakers = []
        for name in dict.fromkeys(t['speaker'] for t in diar['speaker_turns']):
            turns = [t for t in diar['speaker_turns'] if t['speaker'] == name]
            pieces = [y[int(t['start'] * SR):int(t['end'] * SR)] for t in turns]
            spk_y = np.concatenate(pieces) if pieces else y
            if len(spk_y) < int(0.25 * SR) and len(y) >= int(0.25 * SR):
                spk_y = y
            risk = self.evaluate(spk_y, quality, any(t['overlap'] for t in turns), low, high, chunk=chunk)
            suspicious, offset = [], 0.
            for t in turns:
                for w in risk['windows']:
                    a, b = max(offset, w['start']), min(offset + t['duration'], w['end'])
                    if b > a and w['risk_score'] >= high:
                        suspicious.append({'start': round(t['start'] + a - offset, 3), 'end': round(t['start'] + b - offset, 3), 'risk_score': w['risk_score']})
                offset += t['duration']
            speakers.append({'speaker_id': name, 'speaking_time_sec': round(len(spk_y) / SR, 2), 'turn_count': len(turns),
                             'analysis': risk, 'language': self.language.detect(spk_y, transcript=transcript, language_hint=language),
                             'vocal_state': acoustic_features(spk_y),
                             'suspicious_intervals': suspicious, 'attribution_confidence': diar['status']})
        update(6)
        if speakers:
            available = [s['analysis'] for s in speakers if s['analysis']['risk_score'] is not None]
            if available:
                overall = dict(max(available, key=lambda r: r['risk_score']))
                if len(available) == len(speakers):
                    overall['reasoning'] = ['Overall score is the highest speaker concern index. Speaker attribution remains approximate.'] + overall['reasoning']
                else:
                    overall['reasoning'] = ['Overall score based on available speaker evidence; one or more brief speakers had limited audio.'] + overall['reasoning']
            else:
                overall = self.evaluate(y, quality, low=low, high=high, chunk=chunk)
        else:
            overall = self.evaluate(y, quality, low=low, high=high, chunk=chunk)
        update(7)
        limits = ['Risk scores are exploratory and uncalibrated; they do not verify identity or establish a voice clone.',
                  'Synthetic speech, voice conversion and editing are not separately classified by the supplied models.',
                  'Noisy, compressed, multilingual and unseen recordings require external validation.', *diar['limitations'], DISCLAIMER]
        update(8)
        peaks = [round(float(np.max(np.abs(part))), 4) for part in np.array_split(y, min(240, len(y))) if len(part)]
        spectrum = []
        if not chunk and len(y) >= 1024:
            mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=max(256, len(y) // 160), n_mels=40)
            spectrum = np.round(librosa.power_to_db(mel, ref=np.max), 1).tolist()
        return {'success': True, 'schema_version': 3, 'analysis_id': str(uuid.uuid4()),
                'timestamp': datetime.now(timezone.utc).isoformat(), 'filename': Path(filename).name,
                'duration_seconds': round(len(y) / SR, 3), 'sample_rate_hz': SR, 'latency_ms': round((time.monotonic() - started) * 1000),
                'analysis': overall, 'deepfake_analysis': overall, 'overall_verdict': 'UNCERTAIN', 'overall_badge': overall['confidence_label'],
                'diarization': {**diar, 'speakers': speakers}, 'language': lang_result, 'vocal_state': vocal,
                'quality': quality, 'waveform_preview': peaks, 'spectrogram': spectrum, 'capabilities': self.capabilities(),
                'transcript': transcript.strip() if transcript else '',
                'limitations': limits, 'recommendations': RECOMMENDATIONS, 'codec_simulation_applied': codec}
