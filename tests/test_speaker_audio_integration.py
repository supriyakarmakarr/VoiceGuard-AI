"""Real recorded speech integration suite. Fetch fixtures with check_speaker_audio.py --download."""
import io
import json
import time
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from api import server, security
from core.forensic_pipeline import SR, Diarization, speech_regions
from scripts.check_speaker_audio import recordings


@pytest.fixture(scope='module')
def samples():
    data = recordings()
    assert len(data) == 4, 'Download the public speech fixtures with scripts/check_speaker_audio.py --download'
    return data


def wav(y):
    buf = io.BytesIO()
    sf.write(buf, y, SR, format='WAV')
    return buf.getvalue()


def conversation(samples, names):
    seen = set()
    parts = []
    for name in names:
        start = 6*SR if name in seen else 0
        parts.extend([samples[name][start:start+6*SR], np.zeros(SR//2)])
        seen.add(name)
    return np.concatenate(parts).astype(np.float32)


@pytest.fixture(scope='module')
def client():
    security.rate_limiter.limit = 10000
    with TestClient(server.app) as client:
        yield client


@pytest.mark.parametrize('names', ['A', 'AB', 'ABC', 'ABCD', 'ABA'])
def test_full_uploaded_recordings(client, samples, names):
    audio = conversation(samples, names)
    response = client.post('/api/analyze', files={'file': (names+'.wav', wav(audio))})
    assert response.status_code == 200, response.text
    result = response.json()
    diar = result['diarization']
    assert diar['method'] == 'pretrained_onnx'
    assert diar['num_speakers'] == len(set(names)), diar
    assert len(diar['speakers']) == len(set(names))
    assert result['analysis']['risk_score'] is not None
    for speaker in diar['speakers']:
        assert [x['contribution_weight'] for x in speaker['analysis']['indicators']] == [.5, .3, .2]
    if names == 'ABA':
        assert diar['speaker_turns'][0]['speaker'] == diar['speaker_turns'][-1]['speaker']
    for turn in diar['speaker_turns']:
        assert 0 <= turn['start'] < turn['end'] <= len(audio)/SR + .001
    folder = server.ROOT / '.cache/speaker-validation'
    (folder / (names+'-report.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    sf.write(folder / (names+'.wav'), audio, SR)


def test_live_returning_speaker_and_retry(client, samples):
    session = client.post('/api/live-sessions').json()['session_id']
    counts, labels = [], []
    for index, (name, offset) in enumerate([('A', 0), ('B', 0), ('A', 6)]):
        audio = samples[name][offset*SR:(offset+4)*SR]
        payload = wav(audio)
        data = {'session_id': session, 'chunk_index': index, 'chunk_start_sec': index*4}
        response = client.post('/api/analyze-chunk', data=data, files={'file': ('live.wav', payload)})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result['analysis']['risk_score'] is not None
        diar = result['diarization']
        counts.append(diar['num_speakers'])
        # Boundaries are estimates: a small tail from the previous turn can
        # cross the chunk edge. Check the dominant voice, not that tail.
        durations = {}
        for turn in diar['speaker_turns']:
            durations[turn['speaker']] = durations.get(turn['speaker'], 0.) + turn['duration']
        labels.append(max(durations, key=durations.get))
        retry = client.post('/api/analyze-chunk', data=data, files={'file': ('live.wav', payload)})
        assert retry.json() == result
    assert counts == [1, 2, 2], (counts, labels)
    assert labels == ['Speaker 1', 'Speaker 2', 'Speaker 1']
    assert diar['session_turns'][-1]['start'] >= 8.
    assert client.delete('/api/live-sessions/'+session).status_code == 200
    assert session not in server.live_sessions


def test_live_3500ms_windows_across_turn_boundaries(client, samples):
    audio = conversation(samples, 'ABCA')
    session = client.post('/api/live-sessions').json()['session_id']
    reports = []
    for index, start in enumerate(range(0, len(audio), int(3.5*SR))):
        response = client.post('/api/analyze-chunk',
            data={'session_id': session, 'chunk_index': index, 'chunk_start_sec': start/SR},
            files={'file': ('window.wav', wav(audio[start:start+int(3.5*SR)]))})
        assert response.status_code == 200, response.text
        reports.append(response.json())
    diar = reports[-1]['diarization']
    assert diar['num_speakers'] == 3, [(r['diarization']['num_speakers'], r['diarization']['speaker_turns']) for r in reports]
    assigned = [t for t in diar['session_turns'] if t['speaker'].startswith('Speaker')]
    assert assigned[0]['speaker'] == assigned[-1]['speaker']
    (server.ROOT / '.cache/speaker-validation/live-report.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    client.delete('/api/live-sessions/'+session)


def test_session_validation_and_expiry(client, samples):
    session = client.post('/api/live-sessions').json()['session_id']
    files = {'file': ('a.wav', wav(samples['A'][:2*SR]))}
    assert client.post('/api/analyze-chunk', data={'session_id': session, 'chunk_index': 2}, files=files).status_code == 409
    assert client.post('/api/analyze-chunk', data={'session_id': session, 'chunk_start_sec': 'nan'}, files=files).status_code == 422
    server.live_sessions[session]['updated'] = time.monotonic() - server.LIVE_TTL - 1
    server.expire_live_sessions()
    assert session not in server.live_sessions
    assert client.post('/api/analyze-chunk', data={'session_id': session}, files=files).status_code == 404


def test_noise_silence_and_overlap(samples):
    diarizer = Diarization()
    for audio in [np.zeros(SR*4, dtype=np.float32), np.random.default_rng(7).normal(0, .02, SR*4).astype(np.float32)]:
        result = diarizer.run(audio, speech_regions(audio))
        assert result['num_speakers'] is None, result
    audio = conversation(samples, 'ABC')
    noise = np.random.default_rng(4).normal(0, np.std(audio)/10, len(audio))
    result = diarizer.run((audio+noise).astype(np.float32), speech_regions(audio))
    assert result['num_speakers'] == 3, result
    # Deliberate overlap is a capability probe, not a claim of perfect separation.
    overlap = audio.copy()
    overlap[2*SR:5*SR] += samples['B'][:3*SR]
    result = diarizer.run(overlap, speech_regions(overlap))
    assert result['overlap_supported']
    assert any(t['overlap'] for t in result['speaker_turns']), result


def test_legacy_adapter_uses_real_speaker_evidence(samples):
    from core.diarization_engine import SpeakerDiarizer
    import librosa
    audio = conversation(samples, 'AB')
    audio = librosa.resample(audio, orig_sr=SR, target_sr=22050)
    result = SpeakerDiarizer().diarize(audio, sr=22050, num_speakers=4)
    assert result['num_speakers'] == 2
    assert result['sample_rate_hz'] == SR
    assert len(result['speaker_audio']) == 2


def test_consecutive_voices_without_inserted_pauses(samples):
    audio = np.concatenate([samples[n][:6*SR] for n in 'ABCD']).astype(np.float32)
    result = Diarization().run(audio, speech_regions(audio))
    assert result['num_speakers'] == 4, result


@pytest.mark.parametrize('amplitude', [0., .0001, .02])
def test_uploaded_noise_does_not_become_speakers(client, amplitude):
    audio = np.random.default_rng(7).normal(0, amplitude, SR*4).astype(np.float32)
    response = client.post('/api/analyze', files={'file': ('noise.wav', wav(audio))})
    assert response.status_code == 200
    result = response.json()
    assert result['diarization']['num_speakers'] is None, result['diarization']
    assert result['diarization']['speaker_turns'] == []
