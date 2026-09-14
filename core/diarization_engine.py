"""Compatibility adapter to the same estimated diarization used by the API.

No independent energy-only counter: silence and missing models must never become
one invented speaker. Learned segmentation preserves pauses and overlap.
"""
import librosa
import numpy as np
from .forensic_pipeline import Diarization, SR, speech_regions
from .speech_backends import exclusive_turns


class SpeakerDiarizer:
    def __init__(self, sr=16000, frame_window_sec=1.2, frame_hop_sec=.4,
                 max_speakers=6, similarity_threshold=.72):
        # Keep the old constructor callable; clustering belongs to the shared backend.
        self.sr = sr
        self.backend = Diarization()

    def diarize(self, audio, sr=16000, num_speakers=None):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if audio.ndim != 1 or not np.isfinite(audio).all() or sr <= 0:
            raise ValueError('Expected finite mono audio and a positive sample rate')
        if sr != SR and len(audio):
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        result = self.backend.run(audio, speech_regions(audio))
        if num_speakers is not None:
            result['limitations'].append('Legacy requested count is ignored; counts are estimated from audio.')
        speaker_audio, stats = {}, {}
        duration = len(audio) / SR
        for name in dict.fromkeys(t['speaker'] for t in result['speaker_turns']):
            turns = [t for t in result['speaker_turns'] if t['speaker'] == name]
            clean = exclusive_turns(turns, result['speaker_turns'])
            pieces = [audio[round(t['start']*SR):round(t['end']*SR)] for t in clean]
            speaker_audio[name] = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)
            total = sum(t['duration'] for t in turns)
            stats[name] = {'total_time_sec': round(total, 3),
                           'percentage': round(100*total/max(duration, .001), 1),
                           'turn_count': len(turns)}
        count = result['num_speakers']
        return {**result, 'speaker_audio': speaker_audio, 'speaker_stats': stats,
                'sample_rate_hz': SR,
                'timeline_summary': 'Speaker count unavailable.' if count is None else
                    f'Estimated {count} speakers across {len(result["speaker_turns"])} turns.'}
