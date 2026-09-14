"""Build the static frontend using one public setting: API_URL."""
import ipaddress
import json
import os
from pathlib import Path
import shutil
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ('index.html', 'script.js', 'app.js', 'core-visual.js', 'recorder-worklet.js',
          'styles.css', 'tokens.css', '.nojekyll')


def build():
    url = os.getenv('API_URL', '').strip().rstrip('/')
    parsed = urlsplit(url)
    host = parsed.hostname or ''
    local = host.lower() == 'localhost'
    try:
        local = local or ipaddress.ip_address(host).is_loopback
    except ValueError:
        pass
    if parsed.scheme != 'https' or not host or local or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Set API_URL to the public HTTPS FastAPI base URL (no credentials, query or fragment).')
    if parsed.path.rstrip('/').endswith('/api'):
        raise ValueError('API_URL must omit the /api suffix; frontend requests already include /api.')
    destination = ROOT / 'dist'
    destination.mkdir(exist_ok=True)
    for name in ASSETS:
        shutil.copyfile(ROOT / name, destination / name)
    (destination / 'api-config.js').write_text(
        'window.VOICEGUARD_CONFIG = ' + json.dumps({'API_URL': url}) + ';\n', encoding='utf-8')
    print('Static frontend built in dist using API_URL:', url)


if __name__ == '__main__':
    build()
