"""Explicit one-time setup; normal analysis never downloads models or uploads audio."""
from pathlib import Path
import argparse
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'models' / 'pretrained'

def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        print('Already installed:', destination.name, flush=True)
        return
    temporary = destination.with_suffix(destination.suffix + '.part')
    for attempt in range(5):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'})
            with urllib.request.urlopen(request, timeout=90) as response:
                resumed = response.status == 206
                if resumed and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Server returned an unexpected resume range')
                expected = int(response.headers.get('Content-Length', '0'))
                received = 0
                with temporary.open('ab' if resumed else 'wb') as target:
                    while block := response.read(1024 * 1024):
                        target.write(block)
                        received += len(block)
                if expected and received != expected:
                    raise IOError('Incomplete download')
            temporary.replace(destination)
            print('Installed:', destination.name, flush=True)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install-runtime', action='store_true', help='Install pinned Python speech dependencies into project .runtime')
    args = parser.parse_args()
    if args.install_runtime:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(ROOT / '.runtime'),
                        '-r', str(ROOT / 'requirements-speech.txt')], check=True)
    for name in ['model.bin', 'config.json', 'tokenizer.json', 'vocabulary.txt']:
        download('https://huggingface.co/Systran/faster-whisper-small/resolve/main/' + name,
                 MODELS / 'whisper-small' / name)
    archive = MODELS / 'segmentation.tar.bz2'
    download('https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2', archive)
    with tarfile.open(archive) as source:
        # Only extract the named model bytes; never extract arbitrary archive paths.
        member = next(m for m in source.getmembers() if m.isfile() and m.name.endswith('/model.onnx'))
        (MODELS / 'speaker-segmentation.onnx').write_bytes(source.extractfile(member).read())
    download('https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx', ROOT / 'speaker-embedding-multilingual.onnx')
    print('Setup complete. Restart VoiceGuard and check connection capabilities.')

if __name__ == '__main__':
    main()
