"""Developer diagnostic using real public speech, never fabricated waveforms as people."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.speaker-runtime'))
import argparse
import urllib.request
import numpy as np
import librosa
from core.forensic_pipeline import Diarization, speech_regions, SR

SOURCES = {
    'A': 'https://librosa.org/data/audio/5703-47212-0000.ogg',
    'B': 'https://librosa.org/data/audio/3436-172162-0000.ogg',
    'C': 'https://librosa.org/data/audio/198-209-0000.ogg',
    'D': 'https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac',
}

def recordings(download=False):
    folder = ROOT / '.cache' / 'speaker-validation'
    folder.mkdir(parents=True, exist_ok=True)
    output = {}
    for name, url in SOURCES.items():
        path = folder / (name + Path(url).suffix)
        if download and not path.exists():
            urllib.request.urlretrieve(url, path)
        if path.exists():
            output[name] = librosa.load(path, sr=SR)[0]
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    samples = recordings(args.download)
    diarizer = Diarization()
    print('neural loaded', diarizer.neural.pipeline is not None, flush=True)
    for names in ['A', 'AB', 'ABC', 'ABCD', 'ABA']:
        if not set(names) <= samples.keys():
            continue
        y = np.concatenate([np.r_[samples[n][:6*SR], np.zeros(SR//2)] for n in names]).astype(np.float32)
        result = diarizer.run(y, speech_regions(y))
        print(names, result['num_speakers'], result['method'], result['speaker_turns'], flush=True)
