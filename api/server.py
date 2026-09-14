"""VoiceGuard API: in-memory jobs, bounded audio processing, honest capabilities."""
import asyncio
import copy
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.forensic_pipeline import ForensicPipeline, AudioError, MAX_BYTES, decode_audio, acoustic_features, speech_regions
from api.security import enforce_security_and_rate_limit
from core.speech_backends import LiveSpeakerTracker

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
pipeline = None
jobs = {}
jobs_lock = threading.Lock()
processing_lock = threading.Lock()
JOB_TTL = 900
live_sessions = {}
LIVE_TTL = 180
MAX_LIVE_SESSIONS = 32
STAGES = ['Reading audio', 'Detecting speech', 'Separating speakers', 'Identifying language', 'Extracting acoustic features', 'Running trained models', 'Cross-checking forensic signals', 'Assessing confidence', 'Generating forensic report']


@asynccontextmanager
async def lifespan(app):
    global pipeline
    pipeline = await run_in_threadpool(ForensicPipeline, ROOT)
    if os.getenv('VOICEGUARD_REQUIRE_MODELS', 'false').lower() == 'true':
        capabilities = pipeline.capabilities()
        if not all(capabilities.get(k) for k in ('baseline_model_loaded', 'deep_model_loaded', 'language_detection', 'live_speaker_tracking')):
            raise RuntimeError('Required analysis models failed to load. Check build logs and run scripts/setup_speech_models.py.')
    async def expire_reports():
        while True:
            await asyncio.sleep(15)
            with jobs_lock:
                cleanup_jobs()
            await run_in_threadpool(expire_live_sessions)
    expiry_task = asyncio.create_task(expire_reports())
    yield
    expiry_task.cancel()
    try:
        await expiry_task
    except asyncio.CancelledError:
        pass
    pending = list(getattr(app.state, 'tasks', set()))
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    jobs.clear()
    live_sessions.clear()


app = FastAPI(title='VoiceGuard AI · Voice Forensics', version='3.0.0', lifespan=lifespan)
class BodySizeLimit:
    """Bound chunked multipart bodies before the parser can spool unbounded data."""
    def __init__(self, app):
        self.app = app
    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        total = 0
        async def bounded_receive():
            nonlocal total
            message = await receive()
            total += len(message.get('body', b''))
            if total > MAX_BYTES + 65536:
                raise StarletteHTTPException(413, 'Upload exceeds 25 MB.')
            return message
        await self.app(scope, bounded_receive, send)

app.add_middleware(BodySizeLimit)
# One exact origin allowlist for preflight and write requests. Paths are not origins.
default_origins = ['https://supriyakarmakarr.github.io'] + [
    f'http://{host}:{port}' for host in ('localhost', '127.0.0.1', '[::1]')
    for port in (8000, 5500, 3000, 5173)]
configured_origins = [s.strip().rstrip('/') for s in os.getenv('VOICEGUARD_ALLOWED_ORIGINS', '').split(',') if s.strip()]
origins = configured_origins or default_origins


@app.middleware('http')
async def security_headers(request: Request, call_next):
    origin = request.headers.get('origin')
    is_local_origin = not origin or origin == str(request.base_url).rstrip('/') or origin in origins
    if request.method in ('POST', 'DELETE') and origin and not is_local_origin:
        return JSONResponse({'detail': 'Origin is not allowed.'}, status_code=403)
    try:
        if int(request.headers.get('content-length', '0')) > MAX_BYTES + 65536:
            return JSONResponse({'detail': 'Upload exceeds 25 MB.'}, status_code=413)
    except ValueError:
        return JSONResponse({'detail': 'Invalid content length.'}, status_code=400)
    try:
        response = await call_next(request)
    except Exception:
        LOG.exception('API request failed')
        response = JSONResponse({'detail': 'Analysis service failed to process the request. Check backend logs or retry.'}, status_code=500)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'same-origin'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Permissions-Policy'] = 'camera=(), geolocation=(), microphone=(self)'
    response.headers['Content-Security-Policy'] = "default-src 'self' http://127.0.0.1:* http://localhost:*; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self' http://127.0.0.1:* http://localhost:* ws://127.0.0.1:* ws://localhost:*; worker-src 'self' blob:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    if request.url.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    return response


# Added last so CORS also wraps errors produced by the security/body middleware.
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_credentials=False, allow_methods=['GET', 'POST', 'DELETE', 'OPTIONS'],
                   allow_headers=['Content-Type', 'X-API-Key', 'Authorization', 'Accept'],
                   expose_headers=['Retry-After', 'Content-Disposition'])


@app.exception_handler(AudioError)
async def audio_error_handler(request, exc):
    return JSONResponse({'detail': str(exc)}, status_code=exc.status)


async def read_upload(file):
    try:
        data = await file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise AudioError('Upload exceeds 25 MB.', 413)
        return data
    finally:
        await file.close()


def require_pipeline():
    if pipeline is None:
        raise HTTPException(503, 'The analysis service is starting. Please retry.')
    return pipeline


def analyze_locked(data, filename, **kwargs):
    engine = require_pipeline()
    if not processing_lock.acquire(blocking=True, timeout=8.0):
        raise HTTPException(429, 'The analysis engine is busy. Please retry shortly.', headers={'Retry-After': '3'})
    try:
        session_id = kwargs.pop('live_session_id', '')
        chunk_index = kwargs.pop('live_chunk_index', 0)
        start_sec = kwargs.get('chunk_start_sec', 0.)
        if not session_id:
            return engine.analyze(data, filename, **kwargs)
        session = live_sessions.get(session_id)
        if session is None or time.monotonic() - session['updated'] > LIVE_TTL:
            live_sessions.pop(session_id, None)
            raise HTTPException(404, 'Live speaker session expired. Start a new recording.')
        digest = hashlib.sha256(data + repr(start_sec).encode()).hexdigest()
        if chunk_index == session['next_index'] - 1 and digest == session.get('digest'):
            return session['result']
        if chunk_index != session['next_index']:
            raise HTTPException(409, 'Live chunks must be submitted in sequence.')
        y, _ = decode_audio(data, filename)
        end_sec = start_sec + len(y)/16000
        if abs(start_sec - session['end_sec']) > .002 or end_sec > 120.002:
            raise HTTPException(422, 'Live timestamps must be contiguous and within the 120 second session limit.')
        tracker = copy.deepcopy(session['tracker'])
        result = engine.analyze(data, filename, speaker_tracker=tracker, **kwargs)
        session.update(tracker=tracker, next_index=chunk_index+1, end_sec=end_sec,
                       updated=time.monotonic(), digest=digest, result=result)
        return result
    finally:
        processing_lock.release()


@app.get('/api/health')
def health_check():
    return {'status': 'ready' if pipeline else 'starting', 'version': '3.0.0', **(pipeline.capabilities() if pipeline else {})}


@app.get('/api/models')
def model_info():
    return {'capabilities': require_pipeline().capabilities(),
            'limitations': 'Supplied models have a small development benchmark. No real-world accuracy or deployment confidence is claimed.'}


@app.post('/api/analyze')
@app.post('/api/analyze-multispeaker')
async def analyze_audio(file: UploadFile = File(...), threshold_low: float = Form(30), threshold_high: float = Form(70),
                        claimed_speaker_id: str = Form(''), simulate_codec: str = Form('none'), transcript: str = Form(''),
                        language: str = Form(''), _auth=Depends(enforce_security_and_rate_limit)):
    if not 0 <= threshold_low < threshold_high <= 100:
        raise HTTPException(422, 'Thresholds must satisfy 0 ≤ low < high ≤ 100.')
    name = file.filename or 'audio.wav'
    data = await read_upload(file)
    result = await run_in_threadpool(analyze_locked, data, name, low=threshold_low, high=threshold_high, codec=simulate_codec or 'none', transcript=transcript, language=language)
    if claimed_speaker_id:
        result['speaker_verification'] = await run_in_threadpool(verify_identity, data, name, claimed_speaker_id)
    return result


@app.post('/api/analyze-chunk')
async def analyze_chunk(file: UploadFile = File(...), chunk_index: int = Form(0), transcript: str = Form(''),
                        language: str = Form(''), session_id: str = Form(''), chunk_start_sec: float = Form(0., ge=0., le=120.),
                        _auth=Depends(enforce_security_and_rate_limit)):
    name = file.filename or 'chunk.wav'
    data = await read_upload(file)
    result = await run_in_threadpool(analyze_locked, data, name, chunk=True, transcript=transcript, language=language,
                                    live_session_id=session_id, live_chunk_index=chunk_index, chunk_start_sec=chunk_start_sec)
    risk = result['analysis']
    return {**result, 'chunk_index': chunk_index, 'risk_score': risk['risk_score'], 'risk_level': risk['risk_level'],
            'confidence': None, 'alert': risk['risk_level'] == 'HIGH', 'calibration_mode': 'unvalidated'}


def expire_live_sessions():
    with processing_lock:
        for key in list(live_sessions):
            if time.monotonic() - live_sessions[key]['updated'] > LIVE_TTL:
                del live_sessions[key]


@app.post('/api/live-sessions', status_code=201)
def create_live_session(_auth=Depends(enforce_security_and_rate_limit)):
    require_pipeline()
    expire_live_sessions()
    with processing_lock:
        if len(live_sessions) >= MAX_LIVE_SESSIONS:
            raise HTTPException(429, 'Live speaker session capacity reached. Retry shortly.')
        session_id = secrets.token_urlsafe(24)
        live_sessions[session_id] = {'tracker': LiveSpeakerTracker(), 'next_index': 0,
                                     'end_sec': 0., 'updated': time.monotonic()}
    return {'session_id': session_id, 'expires_in_seconds': LIVE_TTL}


@app.delete('/api/live-sessions/{session_id}')
def delete_live_session(session_id: str, _auth=Depends(enforce_security_and_rate_limit)):
    with processing_lock:
        live_sessions.pop(session_id, None)
    return {'deleted': True}


def cleanup_jobs():
    now = time.monotonic()
    for key in list(jobs):
        if now - jobs[key]['created'] > JOB_TTL and jobs[key]['status'] not in ('queued', 'running'):
            del jobs[key]


def job_view(job):
    return {k: job[k] for k in ('status', 'stage', 'stage_label', 'error', 'result') if k in job}


def run_job(job_id, data, filename, transcript='', language=''):
    with jobs_lock:
        jobs[job_id]['status'] = 'running'
    def progress(stage):
        with jobs_lock:
            jobs[job_id].update(stage=stage, stage_label=STAGES[stage])
    try:
        result = analyze_locked(data, filename, progress=progress, transcript=transcript, language=language)
        with jobs_lock:
            jobs[job_id].update(status='complete', result=result)
    except (AudioError, HTTPException) as exc:
        with jobs_lock:
            jobs[job_id].update(status='failed', error=str(exc) if isinstance(exc, AudioError) else exc.detail)
    except Exception:
        LOG.exception('Analysis job failed')
        with jobs_lock:
            jobs[job_id].update(status='failed', error='Audio processing failed. Try a shorter, clear WAV recording.')


@app.post('/api/jobs', status_code=202)
async def create_job(file: UploadFile = File(...), transcript: str = Form(''), language: str = Form(''), _auth=Depends(enforce_security_and_rate_limit)):
    require_pipeline()
    filename = file.filename or 'audio.wav'
    data = await read_upload(file)
    # Decode before accepting; malformed audio gets a precise error instead of a fake stage.
    await run_in_threadpool(decode_audio, data, filename)
    with jobs_lock:
        cleanup_jobs()
        if len(jobs) >= 30 or any(j['status'] in ('queued', 'running') for j in jobs.values()):
            raise HTTPException(429, 'The analysis queue is busy. Please retry shortly.', headers={'Retry-After': '5'})
        job_id = secrets.token_urlsafe(24)
        jobs[job_id] = {'created': time.monotonic(), 'status': 'queued', 'stage': 0, 'stage_label': STAGES[0]}
    # Retain the task and consume its result; audio only lives in this bounded worker.
    task = asyncio.create_task(run_in_threadpool(run_job, job_id, data, filename, transcript, language))
    app.state.tasks = getattr(app.state, 'tasks', set())
    app.state.tasks.add(task)
    task.add_done_callback(app.state.tasks.discard)
    return {'job_id': job_id, 'expires_in_seconds': JOB_TTL}


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str, _auth=Depends(enforce_security_and_rate_limit)):
    with jobs_lock:
        cleanup_jobs()
        if job_id not in jobs:
            raise HTTPException(404, 'Analysis expired or was removed.')
        return job_view(jobs[job_id])


@app.delete('/api/jobs/{job_id}')
def delete_job(job_id: str, _auth=Depends(enforce_security_and_rate_limit)):
    with jobs_lock:
        if job_id in jobs and jobs[job_id]['status'] in ('running', 'queued'):
            raise HTTPException(409, 'Analysis is running. Wait until it completes to remove its report.')
        jobs.pop(job_id, None)
    return {'deleted': True}


@app.get('/api/forensic-report/{job_id}')
def forensic_report(job_id: str, _auth=Depends(enforce_security_and_rate_limit)):
    job = get_job(job_id)
    if job['status'] != 'complete':
        raise HTTPException(409, 'The report is not ready.')
    return JSONResponse(job['result'], headers={'Content-Disposition': 'attachment; filename="voiceguard-forensic-report.json"'})


@app.post('/api/speaker-diarization')
@app.post('/api/language-detection')
@app.post('/api/vocal-state')
async def component_analysis(request: Request, file: UploadFile = File(...), transcript: str = Form(''),
                             language: str = Form(''), _auth=Depends(enforce_security_and_rate_limit)):
    name = file.filename or 'audio.wav'
    data = await read_upload(file)
    result = await run_in_threadpool(analyze_locked, data, name, transcript=transcript, language=language)
    key = {'speaker-diarization': 'diarization', 'language-detection': 'language', 'vocal-state': 'vocal_state'}[request.url.path.rsplit('/', 1)[-1]]
    return {key: result[key], 'limitations': result['limitations']}


@app.get('/api/sample-audios')
def samples():
    folder = ROOT / 'dataset/curated_samples'
    return {'samples': [{'filename': p.name, 'title': p.stem.replace('_', ' '), 'url': '/api/sample-audio/' + p.name,
                         'category': 'Development sample · not ground truth'} for p in sorted(folder.glob('*.wav'))[:12]]}


@app.get('/api/sample-audio/{filename}')
def sample_audio(filename: str):
    folder = (ROOT / 'dataset/curated_samples').resolve()
    candidate = (folder / filename).resolve()
    if candidate.parent != folder or candidate.suffix.lower() != '.wav' or not candidate.is_file():
        raise HTTPException(404, 'Sample not found.')
    return FileResponse(candidate, media_type='audio/wav')


_verifier = None

def get_verifier():
    global _verifier
    if _verifier is None:
        from core.speaker_verifier import SpeakerVerifier
        _verifier = SpeakerVerifier(profiles_dir=str(ROOT / 'models/speaker_profiles'))
        _verifier.load_all_profiles()
    return _verifier


def verify_identity(data, filename, claimed):
    y, _ = decode_audio(data, filename)
    match = get_verifier().verify_claimed_identity(y, claimed)
    match.pop('confidence_pct', None)
    match['verified'] = False
    return {'transaction_decision': 'SECONDARY_VERIFICATION_REQUIRED', 'authorized': False, 'verification_details': match,
            'reason': 'Acoustic similarity is exploratory and cannot authorize a transaction or verify identity.'}


@app.get('/api/voiceprint/profiles')
def profiles(_auth=Depends(enforce_security_and_rate_limit)):
    return {'profiles': get_verifier().list_profiles(), 'threshold': get_verifier().threshold}


@app.post('/api/voiceprint/enroll')
async def enroll(file: UploadFile = File(...), speaker_id: str = Form(...), name: str = Form(...), role: str = Form(''), department: str = Form(''),
                 _auth=Depends(enforce_security_and_rate_limit)):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', speaker_id):
        raise HTTPException(422, 'Speaker ID must contain 1–64 letters, digits, underscores or hyphens.')
    filename = file.filename or 'audio.wav'
    data = await read_upload(file)
    y, _ = await run_in_threadpool(decode_audio, data, filename)
    if len(y) < 3 * 16000 or not speech_regions(y):
        raise HTTPException(422, 'Enrollment requires at least three seconds of clear speech.')
    result = await run_in_threadpool(get_verifier().enroll_speaker, speaker_id, name[:120], y, role[:120], department[:120])
    result.pop('profile_path', None)
    return {**result, 'notice': 'Enrollment intentionally persists an acoustic embedding locally. No raw audio is saved.'}


@app.post('/api/verify-and-detect')
async def verify(file: UploadFile = File(...), claimed_speaker_id: str = Form(...), _auth=Depends(enforce_security_and_rate_limit)):
    name = file.filename or 'audio.wav'
    data = await read_upload(file)
    result = await run_in_threadpool(analyze_locked, data, name)
    return {'success': True, 'decision': await run_in_threadpool(verify_identity, data, name, claimed_speaker_id), 'deepfake_analysis': result['analysis']}


@app.websocket('/api/telephony/ws')
async def telephony(websocket: WebSocket):
    # Carrier integration retained behind an explicit deployment switch and server-side key.
    key = os.getenv('VOICEGUARD_API_KEY', '')
    supplied = websocket.headers.get('x-api-key', '')
    if os.getenv('VOICEGUARD_ENABLE_TELEPHONY') != 'true' or not key or not hmac.compare_digest(key, supplied):
        await websocket.close(code=1008, reason='Carrier streaming requires configuration and authentication.')
        return
    from api.telephony import handle_telephony_websocket
    from core.risk_engine import RiskEngine
    engine = require_pipeline()
    await handle_telephony_websocket(websocket, engine.baseline, engine.deep, RiskEngine())


@app.get('/')
def index():
    return FileResponse(ROOT / 'index.html')


@app.get('/api-config.js')
def frontend_config(request: Request):
    # When FastAPI serves the UI, use its actual origin, including custom PORT.
    return Response('window.VOICEGUARD_CONFIG = {API_URL: window.location.origin};',
                    media_type='application/javascript', headers={'Cache-Control': 'no-store'})


@app.get('/{asset}')
def static_asset(asset: str):
    if asset not in {'tokens.css', 'styles.css', 'script.js', 'app.js', 'core-visual.js', 'recorder-worklet.js', 'api-config.js'}:
        raise HTTPException(404, 'Not found.')
    return FileResponse(ROOT / asset, media_type='text/css' if asset.endswith('.css') else 'application/javascript')
