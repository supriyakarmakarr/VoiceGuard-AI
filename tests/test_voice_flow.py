"""Test suite for voice input improvements, speech-to-text metadata, and short clip evaluation."""
import io
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from core.dataset_generator import generate_human_speech_sample
from core.forensic_pipeline import SR
from api import server, security


def make_wav(y, sr=SR):
    b = io.BytesIO()
    sf.write(b, y, sr, format='WAV')
    return b.getvalue()


@pytest.fixture(scope='module')
def client():
    security.rate_limiter.limit = 10000
    with TestClient(server.app) as c:
        yield c


def test_short_natural_speech_produces_valid_result(client):
    """Natural single sentence (1.5s speech) should produce a valid risk score."""
    speech = generate_human_speech_sample(duration=1.5)
    wav_bytes = make_wav(speech)

    r = client.post('/api/analyze', files={'file': ('short_speech.wav', wav_bytes, 'audio/wav')})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['analysis']['risk_score'] is not None
    assert d['analysis']['risk_level'] in ('LOW', 'MEDIUM', 'HIGH')


def test_transcript_metadata_passed_through(client):
    """Transcripts provided from speech recognition should be returned in the analysis."""
    speech = generate_human_speech_sample(duration=2.0)
    wav_bytes = make_wav(speech)

    test_transcript = "Hello this is a voice authenticity check"
    r = client.post(
        '/api/analyze',
        data={'transcript': test_transcript},
        files={'file': ('speech_with_transcript.wav', wav_bytes, 'audio/wav')}
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d.get('transcript') == test_transcript


def test_live_chunk_with_brief_speech(client):
    """Live streaming chunks with brief speech (1.2s) should return evaluated risk."""
    speech = generate_human_speech_sample(duration=1.2)
    wav_bytes = make_wav(speech)

    r = client.post(
        '/api/analyze-chunk',
        data={'chunk_index': 1, 'transcript': 'testing live speech'},
        files={'file': ('live_chunk.wav', wav_bytes, 'audio/wav')}
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['chunk_index'] == 1
    assert d['risk_score'] is not None
    assert d.get('transcript') == 'testing live speech'


def test_quiet_microphone_normalization(client):
    """Quiet microphone input (low RMS) should be gently normalized and evaluated."""
    speech = generate_human_speech_sample(duration=2.0)
    quiet_sig = speech * 0.08  # Very quiet mic input
    wav_bytes = make_wav(quiet_sig)

    r = client.post('/api/analyze', files={'file': ('quiet_mic.wav', wav_bytes, 'audio/wav')})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['analysis']['risk_score'] is not None



@pytest.mark.parametrize('transcript,hint', [
    ('नमस्ते मेरा नाम सुप्रियो है', 'hi'),
    ('নমস্কার কেমন আছেন', 'bn'),
    ('namaste aap kaise ho', 'hi'),
    ('nomoshkar kemon achhen', 'bn'),
    ('Hello this is my voice', 'en'),
    ('', ''),
])
def test_audio_language_is_independent_of_transcript_and_ui(client, monkeypatch, transcript, hint):
    # Generated test signals are not real Hindi/Bengali speech. The audio-model
    # response is mocked explicitly to test metadata isolation at the API boundary.
    class AudioModel:
        def detect_language(self, audio):
            return 'en', .96, [('en', .96), ('hi', .02), ('bn', .02)]
    monkeypatch.setattr(server.pipeline.language, 'model', AudioModel())
    speech = generate_human_speech_sample(duration=3.0)
    response = client.post('/api/analyze', data={'transcript': transcript, 'language': hint},
                           files={'file': ('metadata-isolation.wav', make_wav(speech), 'audio/wav')})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['language']['label'] == 'en'
    assert result['language']['source'] == 'whisper_audio'
    assert result['transcript'] == transcript
    assert result['analysis']['risk_score'] is not None
