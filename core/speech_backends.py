"""Local pretrained speech models. Missing evidence never becomes a guessed label."""
from pathlib import Path
import logging
import os
import site
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / '.runtime'
if RUNTIME.is_dir():
    # Keep the existing NumPy/PyTorch stack ahead of optional project-local packages.
    site.addsitedir(str(RUNTIME))
MODEL_DIR = ROOT / 'models' / 'pretrained'
LOG = logging.getLogger(__name__)
SR = 16000


class LanguageDetector:
    def __init__(self):
        self.model = None
        path = Path(os.getenv('VOICEGUARD_LANGUAGE_MODEL', str(MODEL_DIR / 'whisper-small')))
        if path.is_dir() and (path / 'model.bin').is_file():
            try:
                from faster_whisper import WhisperModel
                self.model = WhisperModel(str(path), device='cpu', compute_type='int8',
                                          cpu_threads=4, local_files_only=True)
            except Exception:
                LOG.exception('Local language model unavailable')

    def detect(self, y, transcript='', language_hint=''):
        empty = {'status': 'unavailable', 'label': 'Unknown', 'languages': [], 'segments': [],
                 'source': 'audio_model', 'reason': 'Local language model unavailable. Interface language and transcript are not audio-language evidence.'}
        if len(y) < SR * 2 or float(np.std(y)) < .0001:
            return {**empty, 'status': 'insufficient_evidence', 'reason': 'At least two seconds of clear speech are needed for language identification.'}
        if self.model is None:
            return empty
        try:
            sums, segments = {}, []
            for start in range(0, len(y), SR * 12):
                chunk = np.asarray(y[start:start + SR * 12], dtype=np.float32)
                if len(chunk) < SR * 2 or float(np.std(chunk)) < .0001:
                    continue
                lang, score, probs = self.model.detect_language(chunk)
                segments.append({'start': start / SR, 'end': (start + len(chunk)) / SR,
                                 'language': lang, 'model_score': round(float(score), 4)})
                for code, value in probs:
                    sums[code] = sums.get(code, 0.) + float(value) * len(chunk)
            total = sum(sums.values()) or 1
            langs = [{'code': code, 'model_score': round(value / total, 4)}
                     for code, value in sorted(sums.items(), key=lambda item: -item[1])[:5]]
            supported = list(dict.fromkeys(s['language'] for s in segments if s['model_score'] >= .60))
            return {'status': 'estimated' if supported else 'insufficient_evidence',
                    'label': ' + '.join(supported) or 'Unknown', 'languages': langs,
                    'segments': segments, 'source': 'whisper_audio',
                    'reason': 'Local multilingual Whisper inference on the recording. Scores are model responses, not calibrated accuracy.' if supported
                              else 'The audio model could not identify a language confidently. Use a longer, clearer recording.'}
        except Exception:
            LOG.exception('Language inference failed')
            return {**empty, 'reason': 'Audio-language inference failed. No fallback language was invented.'}


class NeuralDiarizer:
    """Pretrained segmentation, speaker embeddings, and automatic clustering on CPU."""
    def __init__(self):
        self.pipeline = None
        segmentation = Path(os.getenv('VOICEGUARD_SEGMENTATION_MODEL', str(MODEL_DIR / 'speaker-segmentation.onnx')))
        embedding = Path(os.getenv('VOICEGUARD_EMBEDDING_MODEL', str(ROOT / 'speaker-embedding-multilingual.onnx')))
        if not segmentation.is_file() or not embedding.is_file():
            return
        try:
            import sherpa_onnx
            config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=str(segmentation), window_shift_ratio=.1), num_threads=4),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(embedding), num_threads=4),
                clustering=sherpa_onnx.FastClusteringConfig(
                    num_clusters=-1, threshold=float(os.getenv('VOICEGUARD_SPEAKER_THRESHOLD', '0.65'))),
                min_duration_on=.25, min_duration_off=.25)
            if not config.validate():
                raise ValueError('Invalid diarization model configuration')
            self.pipeline = sherpa_onnx.OfflineSpeakerDiarization(config)
        except Exception:
            LOG.exception('Pretrained ONNX diarization unavailable')

    def turns(self, y):
        if self.pipeline is None:
            return None
        result = self.pipeline.process(np.asarray(y, dtype=np.float32)).sort_by_start_time()
        names, turns = {}, []
        for segment in result:
            names.setdefault(segment.speaker, f'Speaker {len(names) + 1}')
            start, end = max(0., float(segment.start)), min(len(y) / SR, float(segment.end))
            if end > start:
                turns.append({'speaker': names[segment.speaker], 'start': round(start, 3), 'end': round(end, 3)})
        return turns


def exclusive_turns(turns, all_turns):
    """Subtract other speakers' active intervals before assigning individual risk."""
    output = []
    for turn in turns:
        spans = [(turn['start'], turn['end'])]
        for other in all_turns:
            if other['speaker'] == turn['speaker']:
                continue
            next_spans = []
            for start, end in spans:
                if other['end'] <= start or other['start'] >= end:
                    next_spans.append((start, end))
                else:
                    if start < other['start']: next_spans.append((start, other['start']))
                    if other['end'] < end: next_spans.append((other['end'], end))
            spans = next_spans
        output.extend({'speaker': turn['speaker'], 'start': a, 'end': b, 'duration': b-a}
                      for a, b in spans if b-a >= .08)
    return output
