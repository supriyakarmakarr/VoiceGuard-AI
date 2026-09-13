# VoiceGuard: language, speakers and motion fixes

Completed September 14, 2026 in the original project folder.

## Language

Removed the spectral/pitch fallback that usually selected Hindi and assigned invented confidence values. Language now comes from local multilingual Whisper audio inference. Browser transcription and UI language are independent controls; neither can override detected language. Missing models, too little speech and uncertain responses return Unknown rather than a forced language. The connection panel reports actual model availability.

Installed faster-whisper small locally. Normal analysis never downloads weights or sends audio to these model publishers. Optional browser Web Speech transcription retains its separate browser-provider behavior and explicit language selector.

## Speakers and individual concern

Installed pretrained pyannote segmentation through sherpa-onnx, with multilingual 3D-Speaker CAMPPlus embeddings. Automatic clustering replaces the acoustic fallback when these models are available. The fallback no longer reports unknown speaker count as one person.

Every attributed speaker appears in a selectable overview card with a concern index, active duration and isolated duration. Timeline lanes retain overlap labels. Individual scores use only non-overlapping intervals; the same mixed waveform is not attributed to multiple people. Insufficient isolated speech produces an unavailable individual score. Downloads include isolated and excluded-overlap durations.

Scores remain **uncalibrated concern indices out of 100**, not verified percentages of fraud or identity. No model can guarantee the exact number of people in every recording. In particular, three fully simultaneous voices, very short contributions, similar voices and heavy background noise remain difficult. Counting anonymous voices does not identify a person's name.

## Acoustic core

Continuous, gentle rotation while visible. Hover smoothly reduces angular speed and follows the cursor; leaving restores rotation. Hidden/offscreen rendering pauses. Reduced-motion preference disables decorative rotation.

## Final verification

- **58 Python tests passed**, one library warning, in 52.43 seconds.
- JavaScript syntax checks passed.
- Core motion tests passed: idle rotation, hover slowdown, pointer following, leave recovery, reduced motion.
- Report builder and AudioWorklet tests passed.
- Real browser/API check: Hindi UI plus English recorded speech produced English.
- Controlled overlap fixture using two real voices: **2 speakers**, overlap marked, distinct individual risk cards using isolated audio.
- Derived three-voice fixture: **3 speakers**, three individual risk cards.
- Speaker selection and 1440/1280/1024/768/480/390/360 layouts passed without horizontal overflow or page exceptions.
- Final pretrained embedding comparison: single-voice excerpt returned 1; two supplied English conversations returned 2; supplied four-speaker recording returned 4; controlled overlap returned 2 at selected threshold 0.65.
- A separate harder file named `3-two-speakers-en.wav` returned **3 groups rather than 2**. This known overcount remains; the system is not claimed to have perfect diarization accuracy. Development sample checks and threshold selection are not an independent validation benchmark.

The previous language tests attached Hindi/Bengali text to generated tones and expected those text labels to be treated as detected speech. They were replaced with explicit mocked-audio-model API tests proving metadata cannot override the model. Mixed-window tests now use non-flat signals. New regressions cover missing model, incorrect hints, uncertain language response, interval subtraction and fully overlapping speaker evidence.

## Run

Models and project-local speech dependencies are already installed on this computer. Run the existing `run_voiceguard.bat`, or run `python main.py` in the project folder. The current local preview server is at http://127.0.0.1:8000. Refresh the page to load the updated files.

For another copy or Python environment, install the main requirements first, then:

```powershell
python scripts/setup_speech_models.py --install-runtime
python main.py
```

The explicit setup command installs speech dependencies under `.runtime` and downloads approximately 520 MB of model weights. No setup/download runs inside an analysis request. The setup script resumes interrupted downloads. The project-local wheels must match the Python interpreter used to launch the server.

Default model locations:

- `models/pretrained/whisper-small/`
- `models/pretrained/speaker-segmentation.onnx`
- `speaker-embedding-multilingual.onnx`

Optional overrides: `VOICEGUARD_LANGUAGE_MODEL`, `VOICEGUARD_SEGMENTATION_MODEL`, `VOICEGUARD_EMBEDDING_MODEL`, `VOICEGUARD_SPEAKER_THRESHOLD`. The 0.65 clustering threshold is tied to the selected CAMPPlus embedding model; changing models requires rechecking the threshold. Existing optional pyannote pipeline configuration remains supported.

## Sources and evidence

- [faster-whisper implementation](https://github.com/SYSTRAN/faster-whisper)
- [Whisper small converted weights](https://huggingface.co/Systran/faster-whisper-small)
- [sherpa-onnx diarization example](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-speaker-diarization.py)
- [Segmentation model and public test recordings](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models)
- [Speaker embedding models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models)

`detection-browser-checks.json` and `three-speaker-check.json` contain the observed test outputs. The overlap fixture is a deliberate mixture of the two voices in the first public English recording; the three-voice fixture is assembled from three attributed regions of the public four-voice recording. These are development fixtures, not user recordings.

`voiceguard-detection-fixes.zip` contains the current source files, tests and setup script. The original folder is already updated. The archive excludes installed packages and model binaries; another copy needs the setup command above. No production deployment was performed.
