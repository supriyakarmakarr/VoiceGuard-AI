# VoiceGuard AI — Voice Forensics Lab

A light, interactive audio-forensics workspace built on the existing FastAPI, ResNet-SE CNN, acoustic classifier and DSP code. Upload a recording or capture a microphone session, inspect speaker activity, review model evidence and download a printable forensic report.

## Run locally

Use **Python 3.12**, then run these commands from the project directory:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py
```

Open **http://127.0.0.1:8000**. On Windows, `run_voiceguard.bat` uses the project `.venv` when present. The launcher never generates training data or overwrites model checkpoints. If `python` opens the Microsoft Store, install Python 3.12 and enable its PATH option first.

For CPU-only PyTorch, install its official CPU wheel before the requirements:

```powershell
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

No frontend build is required. Opening `index.html` directly or publishing only static files gives an interface but **cannot run the Python analysis**. Serve the interface and API together. AudioWorklet microphone capture requires localhost or HTTPS and a supported browser.

## What is implemented

- Responsive light interface, sticky navigation, reduced-motion support, keyboard controls and interactive perspective-projected 3D acoustic geometry. The idle visualization is explicitly illustrative; its waveform switches to measured audio after analysis or during microphone capture.
- Real API-backed file analysis. The old random browser verdict generator is removed.
- Strict decoding, byte/duration/channel/sample-rate limits, original-timeline candidate speech regions, quality checks, per-speaker inference, explainable model fusion and conservative uncertainty states.
- The existing trained float CNN and RF/GB ensemble are loaded from `models/`. Untrained WavLM/AASIST/RawNet placeholders no longer contribute or masquerade as trained models.
- Approximate acoustic clustering with separate speaker lanes, speaking durations, original-recording segment playback, overlap indicators when a pretrained backend supplies them, per-speaker evidence and elevated-concern intervals.
- Microphone PCM capture through AudioWorklet, actual waveform/spectrum/amplitude, bounded short-window API analysis and full-session analysis after stopping. Window results do not claim persistent live speaker identity.
- Measured pitch, energy, spectral centroid and pause information. Emotion, fatigue, tension, jitter, shimmer and syllable rate are withheld where reliable evidence is absent.
- Session history in tab memory, temporary server reports, printable HTML report downloads and JSON report API. Reports include identifiers, time, quality, languages, speakers, model contributions, measured features, limitations and verification recommendations. HTML reports contain no audio; open one and use **Print → Save as PDF**.
- Primary controls in English, Hindi and Bengali. Analytical explanations currently remain English; interface language is independent of spoken-language detection.
- Prevention center, explainable architecture, real server progress, useful offline/corrupt/unsupported/short/noisy/uncertain states.

## Scientific scope and limitations

**This is an exploratory analysis tool, not a validated fraud prevention system.** No field accuracy or calibrated confidence percentage is claimed. The supplied model metadata describes a small development dataset. Its scores cannot establish real-world accuracy across people, languages, codecs and unseen generators.

The concern index is 0–100: **LOW <30**, **MEDIUM 30–<70**, **HIGH ≥70**, or **INSUFFICIENT EVIDENCE** with a null score. Within each analysis window, the CNN contributes 50%, acoustic ML 30%, and DSP 20%. The highest window concern is used for a speaker, and the highest speaker concern is used only if all attributed speakers have usable evidence. Strong disagreement, short speech, heavy noise/clipping, detected overlap or missing models withhold the overall index. A low score is not proof of human origin. `synthetic_probability` remains null in public results; individual detector responses are explicitly uncalibrated indices.

The default diarizer clusters measured MFCC timbre features. It is a **low-confidence acoustic estimate**, not a trained biometric speaker counter. It can merge different people or split one person, and does not separate simultaneous voices. Silent gaps are preserved. Very short audio is left unassigned rather than inventing a speaker count. Use the optional pretrained backend for stronger attribution and overlap labels.

The existing models do not separately classify voice cloning, conversion and other editing. They flag acoustic concern only. Voiceprint similarity is retained as an exploratory API feature, and **never authorizes transactions**. Reports always recommend independent verification.

Vocal features are signal measurements: **This is an acoustic signal analysis, not a medical or psychological diagnosis.** The application does not infer someone’s mental state from their voice.

## Optional pretrained language and diarization backends

Adapters are implemented, but **weights are not included**. No audio is sent to a third-party API, no model weights download during a request, and unavailable capabilities are reported honestly.

### Language identification

Install `faster-whisper` and supply a local **multilingual** CTranslate2 Whisper model directory. Do not use an English-only `.en` model for Hindi/Bengali.

```powershell
.venv\Scripts\python -m pip install "faster-whisper>=1.1,<2"
$env:VOICEGUARD_LANGUAGE_MODEL = 'C:\path\to\local-multilingual-whisper-model'
.venv\Scripts\python main.py
```

The adapter attempts language identification in independent 12-second windows, aggregates language model responses, and retains per-window evidence to surface possible code switching. It also analyzes each speaker. Short or ambiguous windows return unknown. Window-level language response percentages are not calibrated confidence. No transcript is retained. Language failure does not change deepfake scores.

Reference: [faster-whisper source and language detection API](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py).

### Pretrained speaker diarization

Use a local pyannote Community-1 pipeline after obtaining its weights under the model publisher’s terms. This may require accepting the model’s access conditions in your own Hugging Face account.

```powershell
.venv\Scripts\python -m pip install "pyannote.audio>=4,<5"
$env:VOICEGUARD_DIARIZATION_MODEL = 'C:\path\to\local-pyannote-pipeline'
.venv\Scripts\python main.py
```

Pyannote 4 dependency requirements may require a newer compatible NumPy/numba stack than the default pinned runtime. Use a separate environment for this optional configuration, allow pip to resolve its declared requirements, then rerun the test suite. The adapter supports the Community-1 `speaker_diarization` output and preserves overlapping turns. Overlapping waveforms are **not source-separated**, so per-speaker risk is withheld on those turns. No diarization accuracy is claimed without a labeled multi-speaker benchmark.

Reference: [pyannote Community-1 setup and local use](https://huggingface.co/pyannote/speaker-diarization-community-1).

## API

| Route | Purpose |
|---|---|
| `GET /api/health` | Loaded capabilities and processing limits |
| `GET /api/models` | Model availability and validation limitations |
| `POST /api/analyze` | Existing multipart analysis route (`file`, optional thresholds, claimed speaker and codec simulation) |
| `POST /api/analyze-multispeaker` | Alias of the full pipeline |
| `POST /api/analyze-chunk` | Bounded live audio window; no cross-window speaker identity |
| `POST /api/jobs` | Start an in-memory analysis; returns a random job token |
| `GET /api/jobs/{token}` | Actual server stage and completed report |
| `DELETE /api/jobs/{token}` | Remove a completed report |
| `GET /api/forensic-report/{token}` | Download JSON report |
| `POST /api/speaker-diarization` | Full analysis with diarization component response |
| `POST /api/language-detection` | Full analysis with language component response |
| `POST /api/vocal-state` | Full analysis with acoustic component response |
| `GET /api/sample-audios` | Supplied development sample list |
| `GET /api/sample-audio/{name}` | Safe WAV sample retrieval |
| `GET /api/voiceprint/profiles` | Existing acoustic profile metadata |
| `POST /api/voiceprint/enroll` | Explicitly persist an acoustic profile locally |
| `POST /api/verify-and-detect` | Exploratory acoustic match plus concern evidence; independent verification required |
| `WS /api/telephony/ws` | Existing carrier adapter, disabled until explicitly configured |

Public analysis JSON is schema version 3. Route names remain compatible, but old clients must handle nullable scores/confidence and the nested `speakers[].analysis` object. Fake-probability fields are intentionally nullable. The source training utilities and checkpoints remain; the public API does not use their unvalidated calibration claims.

## Privacy and deployment

- Supported encoded inputs: WAV, MP3, FLAC, OGG and AIFF as supported by the installed libsndfile decoder. Arbitrary bytes are never treated as raw PCM. Live browser recordings are encoded to PCM WAV.
- Limits: 25 MB, 120 seconds, up to 8 channels, original sample rates 8–192 kHz; live windows are limited to 12 seconds. Mono analysis resamples to 16 kHz. Clipping/quality is assessed before normalization.
- One analysis worker runs at a time; busy callers receive 429. Use a **single uvicorn worker** with the in-memory job implementation. Reports expire after approximately 15 minutes and are periodically purged. Closing the tab discards local history; its server report expires automatically.
- Multipart parsers may spool large uploads to the operating system’s temporary directory. The upload handle is always closed; no permanent audio file is created by analysis. Explicit voiceprint enrollment persists only an embedding and profile metadata under `models/speaker_profiles/`.
- Localhost works without a key. Remote access requires `VOICEGUARD_API_KEY`. Enter the key in the UI’s connection settings; it remains only in tab memory. Never embed a deployment key in frontend source.
- For an intentionally public demo only, set `API_DEMO_MODE=true`. This makes uploaded audio and job access available without account authentication; do not use it for sensitive organizational recordings.
- Same-origin by default. Optional allowed origins are configured with `VOICEGUARD_ALLOWED_ORIGINS` (comma separated). A browser response CSP, content-type protection and origin checks are installed. Rate limits use connection IP, not untrusted forwarded headers. Behind a proxy, configure trusted proxy behavior at the deployment layer.
- Job tokens are bearer capabilities: possession grants access to that report. The current implementation is not a multi-tenant authenticated report vault. Add user-scoped authorization and an external queue before enterprise use.
- Carrier streaming additionally requires `VOICEGUARD_ENABLE_TELEPHONY=true` and the server API key in `X-API-Key`. It supplies advisory evidence only, never an instruction to automatically terminate a call or approve a transaction.
- `render.yaml` installs dependencies without retraining. Check RAM/CPU requirements before deploying. No website is published by this upgrade.

## Verification

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest tests -q
node --check script.js
node --check core-visual.js
node --check recorder-worklet.js
```

Regression coverage includes silence, short clips, corrupt/unsupported/non-finite/oversized/overlong audio, model availability, language adapter behavior, timeline gaps, overlap labels, report lifecycle, authentication, origin rejection, path containment and known G.711 decode values. Browser validation covers real sample analysis, evidence expansion, speaker playback, report download, and responsive layouts. See `VALIDATION.md` for the actual run results and remaining limits.

The project’s existing MIT license is retained.


## September 14: audio-language and pretrained speaker update

Current detection behavior and verification supersede older hybrid-language and acoustic-only descriptions above. See `DETECTION-VALIDATION.md`.

Multilingual Whisper now detects audio language independently of UI/transcription language. Pretrained sherpa-onnx segmentation and multilingual speaker embeddings provide estimated voice counts and overlap-aware timelines. Individual concern indices use isolated speech. Unknown count is never replaced with one person.

For a fresh copy, run `python scripts/setup_speech_models.py --install-runtime` after installing the main requirements, then restart `python main.py`. Models are already installed in this working copy. The acoustic core rotates continuously, slows on hover and respects reduced motion.
