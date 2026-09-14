"""Context retention and learned backend deployment regressions."""
import numpy as np
import pytest
from core.speech_backends import NeuralDiarizer, LiveSpeakerTracker
from core.forensic_pipeline import Diarization, SR


def test_installed_speaker_backend_loads():
    assert NeuralDiarizer().pipeline is not None


def test_legacy_entry_point_does_not_invent_a_speaker_for_silence():
    from core.diarization_engine import SpeakerDiarizer
    result = SpeakerDiarizer().diarize(np.zeros(SR, dtype=np.float32))
    assert result['num_speakers'] is None
    assert result['speaker_turns'] == []
    assert result['speaker_audio'] == {}


def test_empty_neural_segmentation_is_insufficient_evidence():
    result = Diarization.finish([], 'pretrained_onnx', 'estimated', True, [])
    assert result['num_speakers'] is None
    assert result['status'] == 'insufficient_evidence'


class RecordedBackend:
    """Explicit context/timestamp fixture, not an accuracy benchmark."""
    def __init__(self):
        self.seen = []

    def run(self, y, regions):
        self.seen.append(y.copy())
        turns = [{'speaker': 'Speaker 1', 'start': 0., 'end': 2.}]
        if len(y) >= 4*SR:
            turns.append({'speaker': 'Speaker 2', 'start': 2., 'end': 4.})
        if len(y) >= 6*SR:
            turns.append({'speaker': 'Speaker 1', 'start': 4., 'end': 6.})
        return Diarization.finish(turns, 'pretrained_onnx', 'estimated', True, [])


def test_returning_speech_is_clustered_with_prior_audio():
    tracker, backend = LiveSpeakerTracker(), RecordedBackend()
    for i in range(3):
        output = tracker.update(np.ones(2*SR, dtype=np.float32)*(i+1), backend, i*2.)
        assert len(backend.seen[-1]) == (i+1)*2*SR
    assert output['num_speakers'] == 2
    assert output['speaker_turns'][0]['speaker'] == 'Speaker 1'
    assert output['speaker_turns'][0]['start'] == 0.
    assert output['session_turns'][-1]['start'] == 4.
    np.testing.assert_array_equal(backend.seen[-1][:2*SR], np.ones(2*SR))


def test_session_audio_is_isolated():
    first, second, backend = LiveSpeakerTracker(), LiveSpeakerTracker(), RecordedBackend()
    first.update(np.ones(2*SR), backend, 0.)
    assert len(second.audio) == 0


def test_out_of_order_or_overlong_audio_is_rejected():
    tracker, backend = LiveSpeakerTracker(), RecordedBackend()
    with pytest.raises(ValueError, match='contiguous'):
        tracker.update(np.ones(SR), backend, 3.)
    with pytest.raises(ValueError, match='120 seconds'):
        tracker.update(np.ones(121*SR), backend, 0.)
    assert not backend.seen


def test_failed_diarization_retains_context_for_next_chunk():
    class Broken:
        def run(self, y, regions):
            raise RuntimeError('Model failed')
    tracker = LiveSpeakerTracker()
    with pytest.raises(RuntimeError):
        tracker.update(np.ones(SR), Broken(), 0.)
    assert len(tracker.audio) == SR
    output = tracker.update(np.ones(SR), RecordedBackend(), 1.)
    assert output['num_speakers'] == 1
