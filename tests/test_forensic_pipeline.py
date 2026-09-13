"""Regression checks for uncertainty, real model processing, upload safety and timelines."""
import io
import json
import time
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from core.forensic_pipeline import ForensicPipeline, AudioError, decode_audio, speech_regions, Diarization, LanguageDetector, SR
from api import server, security


def wav(y, sr=SR):
    b=io.BytesIO();sf.write(b,y,sr,format='WAV');return b.getvalue()

@pytest.fixture(scope='module')
def client():
    security.rate_limiter.limit=10000
    with TestClient(server.app) as c:
        yield c


def test_silence_never_invents_speakers_or_verdict(client):
    r=client.post('/api/analyze',files={'file':('silence.wav',wav(np.zeros(48000)),'audio/wav')})
    assert r.status_code==200,r.text
    d=r.json()
    assert d['analysis']['risk_score'] is None
    assert d['analysis']['confidence_score'] is None
    assert d['diarization']['num_speakers'] is None
    assert d['diarization']['speaker_turns']==[]
    assert d['language']['label']=='Unknown'

@pytest.mark.parametrize('name,content,status',[('bad.wav',b'this is not a wave'*100,422),('bad.exe',b'abc'*100,415),('empty.wav',b'',422)])
def test_invalid_audio_has_precise_error(client,name,content,status):
    r=client.post('/api/analyze',files={'file':(name,content)})
    assert r.status_code==status,r.text
    assert 'traceback' not in r.text.lower()


def test_oversized_body(client):
    r=client.post('/api/analyze',content=b'empty',headers={'Content-Length':str(26*1024*1024)})
    assert r.status_code==413


def test_duration_limit_before_decode():
    with pytest.raises(AudioError) as e:decode_audio(wav(np.zeros(SR*121)),'long.wav')
    assert e.value.status==413


def test_nonfinite_rejected():
    b=io.BytesIO();sf.write(b,np.array([np.nan]*SR),SR,format='WAV',subtype='FLOAT')
    with pytest.raises(AudioError):decode_audio(b.getvalue(),'nan.wav')


def test_short_clip_does_not_gain_confidence_from_padding(client):
    y=.1*np.sin(np.arange(4000)*2*np.pi*200/SR)
    r=client.post('/api/analyze-chunk',files={'file':('short.wav',wav(y))}).json()
    assert r['risk_score'] is None
    assert r['confidence'] is None


def test_threshold_validation(client):
    assert client.post('/api/analyze',files={'file':('x.wav',wav(np.zeros(SR)))},data={'threshold_low':80,'threshold_high':30}).status_code==422


def test_real_models_and_timeline(client):
    sample=next((server.ROOT/'dataset/curated_samples').glob('*.wav'))
    r=client.post('/api/analyze',files={'file':(sample.name,sample.read_bytes())})
    assert r.status_code==200,r.text
    d=r.json();assert d['capabilities']['baseline_model_loaded'] and d['capabilities']['deep_model_loaded']
    assert d['analysis']['confidence_score'] is None
    assert d['analysis']['synthetic_probability'] is None
    assert d['waveform_preview'] and d['spectrogram']
    for turn in d['diarization']['speaker_turns']:
        assert 0<=turn['start']<turn['end']<=d['duration_seconds']+.001
    for s in d['diarization']['speakers']:
        assert s['analysis']['risk_score'] is None or 0<=s['analysis']['risk_score']<=100
        assert all(i['name'] not in ['WavLM','AASIST','RawNet'] for i in s['analysis']['indicators'])
    json.dumps(d,allow_nan=False)


def test_missing_models_never_yield_genuine():
    p=object.__new__(ForensicPipeline);p.baseline=p.deep=None
    r=p.evaluate(np.ones(SR*3,dtype=np.float32),{'issues':[]})
    assert r['risk_score'] is None and r['tier_verdict']=='UNCERTAIN'


def test_language_unavailable_is_not_guessed(monkeypatch):
    monkeypatch.delenv('VOICEGUARD_LANGUAGE_MODEL',raising=False)
    d=LanguageDetector().detect(np.ones(SR*4,dtype=np.float32))
    assert d['status']=='unavailable' and d['languages']==[]


def test_language_backend_mixed_windows_without_transcription():
    d=object.__new__(LanguageDetector)
    class Model:
        def detect_language(self,y):
            code='hi' if y[0]==1 else 'bn'
            return code,.8,[(code,.8),('en',.2)]
    d.model=Model();r=d.detect(np.r_[np.ones(SR*12),np.zeros(SR*12)].astype(np.float32))
    assert r['label']=='bn + hi'
    assert [s['start'] for s in r['segments']]==[0,12]


def test_overlap_retained_and_flagged():
    turns=[{'speaker':'Speaker 1','start':0.,'end':2.},{'speaker':'Speaker 2','start':1.,'end':3.}]
    r=Diarization.finish(turns,'test','estimated',True,[])
    assert r['num_speakers']==2 and all(t['overlap'] for t in r['speaker_turns'])


def test_silence_gaps_not_filled_by_acoustic_clusters(monkeypatch):
    monkeypatch.delenv('VOICEGUARD_DIARIZATION_MODEL',raising=False)
    y=np.zeros(SR*9,dtype=np.float32)
    y[:SR*3]=.2*np.sin(np.arange(SR*3)*2*np.pi*180/SR)
    y[SR*6:]=.2*np.sin(np.arange(SR*3)*2*np.pi*280/SR)
    regions=speech_regions(y)
    d=Diarization().run(y,regions)
    assert not any(t['start']<4<t['end'] for t in d['speaker_turns'])


def test_jobs_report_lifecycle(client):
    r=client.post('/api/jobs',files={'file':('silence.wav',wav(np.zeros(SR*3)))})
    assert r.status_code==202,r.text
    job=r.json()['job_id'];status={}
    for _ in range(150):
        status=client.get('/api/jobs/'+job).json()
        if status['status'] in ('complete','failed'):break
        time.sleep(.05)
    assert status['status']=='complete',status
    assert client.get('/api/forensic-report/'+job).status_code==200
    assert client.delete('/api/jobs/'+job).status_code==200
    assert client.get('/api/jobs/'+job).status_code==404


def test_authentication_and_spoofed_ip(client,monkeypatch):
    monkeypatch.setattr(security,'CONFIGURED_API_KEY','test-secret')
    r=client.get('/api/voiceprint/profiles',headers={'X-Forwarded-For':'127.0.0.1'})
    assert r.status_code==401
    assert client.get('/api/voiceprint/profiles',headers={'X-API-Key':'test-secret'}).status_code==200


def test_no_source_or_model_files_exposed(client):
    for path in ['/main.py','/models/baseline_rf.pkl','/.gitignore','/api/sample-audio/..%5C..%5Cmain.py']:
        assert client.get(path).status_code==404
    assert client.post('/api/jobs',headers={'Origin':'https://untrusted.example'},files={'file':('x.wav',wav(np.zeros(SR)))}).status_code==403


def test_enrollment_traversal_denied(client):
    r=client.post('/api/voiceprint/enroll',data={'speaker_id':'../../escape','name':'Test'},files={'file':('x.wav',wav(np.zeros(SR)))})
    assert r.status_code==422


def test_g711_known_silence_codes():
    from api.telephony import decode_mulaw_byte_chunk
    assert np.allclose(decode_mulaw_byte_chunk(bytes([255,127])),0)
    assert decode_mulaw_byte_chunk(bytes([0]))[0]<-.9
    assert decode_mulaw_byte_chunk(bytes([128]))[0]>.9

def test_chunked_upload_is_bounded_without_content_length(client):
    boundary='voiceguard-test-boundary'
    def body():
        yield f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="large.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
        for _ in range(27): yield b'x'*(1024*1024)
        yield f'\r\n--{boundary}--\r\n'.encode()
    r=client.post('/api/analyze',content=body(),headers={'Content-Type':f'multipart/form-data; boundary={boundary}'})
    assert r.status_code==413,r.text


def test_expired_report_is_removed(client):
    with server.jobs_lock:
        server.jobs['expiry-test']={'created':time.monotonic()-server.JOB_TTL-1,'status':'complete','result':{'private':'test'}}
    assert client.get('/api/jobs/expiry-test').status_code==404
    assert 'expiry-test' not in server.jobs

def test_missing_checkpoint_paths_never_create_random_detectors(tmp_path):
    p=ForensicPipeline(tmp_path)
    assert p.baseline is None and p.deep is None
    assert not p.capabilities()['deep_model_loaded']


def test_untrained_cnn_refuses_inference():
    from core.deep_learning_model import DeepLearningVoiceClassifier
    model=DeepLearningVoiceClassifier()
    with pytest.raises(RuntimeError,match='Trained weights'):
        model.predict(np.ones(SR*3,dtype=np.float32))
