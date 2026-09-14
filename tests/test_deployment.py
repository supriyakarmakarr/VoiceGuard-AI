"""Cross-origin regression tests exercise actual middleware and multipart routes."""
import io
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from api import server

ORIGIN = 'https://supriyakarmakarr.github.io'

def test_pages_origin_preflight_and_upload_errors():
    # No model startup needed to demonstrate routing and browser visibility.
    with TestClient(server.app) as client:
        response = client.options('/api/analyze', headers={'Origin': ORIGIN,
            'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'content-type,x-api-key'})
        assert response.status_code == 200
        assert response.headers['access-control-allow-origin'] == ORIGIN
        for path in ['/api/analyze', '/api/analyze-chunk', '/api/jobs']:
            response = client.post(path, headers={'Origin': ORIGIN}, files={'file': ('bad.wav', b'bad')})
            assert response.status_code == 422, response.text
            assert response.headers['access-control-allow-origin'] == ORIGIN
        response = client.post('/api/jobs', headers={'Origin': ORIGIN, 'Content-Length': str(26*1024*1024)})
        assert response.status_code == 413
        assert response.headers['access-control-allow-origin'] == ORIGIN
        response = client.post('/api/jobs', headers={'Origin': 'http://localhost.evil.example'})
        assert response.status_code == 403


def test_tls_proxy_same_origin_and_runtime_config():
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
    proxied = ProxyHeadersMiddleware(server.app, trusted_hosts=['10.0.0.1'])
    with TestClient(proxied, base_url='http://voiceguard-test.onrender.com', client=('10.0.0.1', 443)) as client:
        headers = {'Origin': 'https://voiceguard-test.onrender.com', 'X-Forwarded-Proto': 'https'}
        response = client.get('/api-config.js', headers=headers)
        assert response.status_code == 200
        assert 'window.location.origin' in response.text
        response = client.post('/api/jobs', headers=headers)
        # Missing auth or multipart fields are allowed here; origin rejection is not.
        assert response.status_code in (401, 422), response.text


def test_inference_failure_is_json_with_cors(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('Internal model diagnostic')
    with TestClient(server.app) as client:
        monkeypatch.setattr(server.pipeline, 'analyze', fail)
        response = client.post('/api/analyze', headers={'Origin': ORIGIN}, files={'file': ('audio.wav', b'audio')})
        assert response.status_code == 500
        assert response.headers['access-control-allow-origin'] == ORIGIN
        assert 'detail' in response.json()
        assert 'Internal model diagnostic' not in response.text
