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


def test_hindi_live_voice_detection(client):
    """Hindi live speech should produce evaluated risk and Hindi language label."""
    speech = generate_human_speech_sample(duration=2.0)
    wav_bytes = make_wav(speech)

    r = client.post(
        '/api/analyze',
        data={'transcript': 'नमस्ते मेरा नाम सुप्रियो है और यह मेरी आवाज़ है', 'language': 'hi'},
        files={'file': ('live_hindi.wav', wav_bytes, 'audio/wav')}
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['analysis']['risk_score'] is not None
    assert d['analysis']['risk_level'] in ('LOW', 'MEDIUM', 'HIGH')
    assert d['language']['label'] == 'hi'
    assert d['language']['status'] == 'estimated'


def test_bengali_live_voice_detection(client):
    """Bengali live speech should produce evaluated risk and Bengali language label."""
    speech = generate_human_speech_sample(duration=2.0)
    wav_bytes = make_wav(speech)

    r = client.post(
        '/api/analyze',
        data={'transcript': 'নমস্কার কেমন আছেন আপনারা', 'language': 'bn'},
        files={'file': ('live_bengali.wav', wav_bytes, 'audio/wav')}
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['analysis']['risk_score'] is not None
    assert d['language']['label'] == 'bn'
    assert d['language']['status'] == 'estimated'


def test_live_mic_with_ambient_noise_not_insufficient_evidence(client):
    """Real microphone input with background noise should be evaluated without returning INSUFFICIENT_EVIDENCE."""
    t = np.linspace(0, 2.5, int(2.5 * SR))
    speech = np.zeros_like(t)
    speech[int(0.4*SR):int(1.9*SR)] = 0.25 * np.sin(2 * np.pi * 180 * np.linspace(0, 1.5, int(1.5*SR)))
    noise = 0.02 * np.random.randn(len(t))
    y = (speech + noise).astype(np.float32)
    wav_bytes = make_wav(y)

    r = client.post(
        '/api/analyze',
        data={'transcript': 'Testing live speech on microphone with room noise', 'language': 'en'},
        files={'file': ('live_mic_ambient.wav', wav_bytes, 'audio/wav')}
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['analysis']['risk_score'] is not None
    assert d['analysis']['risk_level'] in ('LOW', 'MEDIUM', 'HIGH')
    assert d['analysis']['risk_level'] != 'INSUFFICIENT_EVIDENCE'
    assert d['language']['label'] == 'en'


def test_hindi_transliterated_and_acoustic_detection(client):
    """Hinglish transliterated speech should detect Hindi."""
    speech = generate_human_speech_sample(duration=2.0)
    wav_bytes = make_wav(speech)

    r = client.post(
        '/api/analyze',
        data={'transcript': 'namaste aap kaise ho mera naam VoiceGuard hai'},
        files={'file': ('hinglish_voice.wav', wav_bytes, 'audio/wav')}
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['analysis']['risk_score'] is not None
    assert d['language']['label'] == 'hi'
    assert d['language']['status'] == 'estimated'


def test_bengali_transliterated_and_acoustic_detection(client):
    """Banglish transliterated speech should detect Bengali."""
    speech = generate_human_speech_sample(duration=2.0)
    wav_bytes = make_wav(speech)

    r = client.post(
        '/api/analyze',
        data={'transcript': 'nomoshkar kemon achhen ami bhalo achi'},
        files={'file': ('banglish_voice.wav', wav_bytes, 'audio/wav')}
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['analysis']['risk_score'] is not None
    assert d['language']['label'] == 'bn'
    assert d['language']['status'] == 'estimated'


def test_live_voice_without_transcript_not_insufficient_evidence(client):
    """Live voice recorded without any transcript or language hint must evaluate risk and language."""
    speech = generate_human_speech_sample(duration=2.0)
    wav_bytes = make_wav(speech)

    r = client.post(
        '/api/analyze',
        files={'file': ('no_transcript_voice.wav', wav_bytes, 'audio/wav')}
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['analysis']['risk_score'] is not None
    assert d['analysis']['risk_level'] in ('LOW', 'MEDIUM', 'HIGH')
    assert d['analysis']['risk_level'] != 'INSUFFICIENT_EVIDENCE'
    assert d['language']['label'] in ('hi', 'bn', 'en')
    assert d['language']['status'] == 'estimated'


