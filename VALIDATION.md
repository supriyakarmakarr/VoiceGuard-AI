# VoiceGuard AI upgrade — validation record

Validated locally on Windows with Python 3.12, NumPy 2.2.6, SciPy 1.15.3, librosa 0.11.0, numba 0.61.2 and scikit-learn 1.9.0. Both supplied trained checkpoints load. Optional pretrained language and diarization weights are not included.

## Automated checks

- Python regression suite: **42 passed**, 2 library deprecation warnings, in 48.82 seconds. Command: `python -m pytest tests -q --disable-warnings --tb=short`.
- JavaScript syntax checks: `script.js`, `core-visual.js`, `recorder-worklet.js` passed.
- AudioWorklet transport test passed: stereo-to-mono averaging, 2048-sample PCM blocks, buffer reset, empty input.
- Report builder test passed: download trigger, identifiers, speaker findings, original intervals, model evidence, limitations, recommendations, print styling and hostile-filename HTML escaping.
- Actual model inference is exercised using a supplied development sample; this is functional verification, not an accuracy benchmark.
- Upload regression tests cover corrupt, empty, unsupported, short, oversized, overlong and non-finite audio, including chunked uploads without Content-Length.
- Uncertainty checks cover silence, missing weights, untrained CNN inference refusal, unavailable languages, no invented confidence, speaker overlap and silent timeline gaps.
- API checks cover report creation/retrieval/deletion/expiry, authentication, spoofed forwarding headers, origin rejection, sample path traversal and safe voiceprint identifiers.
- Legacy preprocessing, features, calibration utility, exploratory speaker matching, codec simulation and API tests remain covered. Expectations were updated where the old tests required unsupported identity authorization.

## Browser checks

- Desktop, 768 px tablet and 390 px phone layouts inspected. Tablet/phone had no horizontal overflow. Mobile navigation expanded correctly.
- A supplied four-second development recording went through the real job API. The browser displayed its completed low concern index, explicit low-confidence label, model evidence, one approximate speaker group and measured acoustic values.
- Evidence disclosure expanded correctly. Clicking a speaker segment started audio playback; the player returned to its stopped state after the segment.
- The completed readout and timeline rendered correctly, without captured console errors.
- The report download button executed without a console error. The in-app browser did not expose a download event; saved-file delivery in a normal browser remains a manual integration check. The actual HTML builder and download trigger were separately tested.
- The microphone capture pipeline was tested with deterministic PCM input. A physical microphone and browser permission prompts were not exercised; no private microphone audio was collected during development.

## Not established by these checks

No real-world detector accuracy, speaker-count accuracy, calibrated confidence, emotional-state inference, persistent live speaker identity, or generalization to English/Hindi/Bengali is claimed. Optional pretrained adapters need weights and an integration benchmark in their configured environment. No production deployment was performed.

Initial tests with unpinned newest audio dependencies failed because Windows application control blocked a library. The final pinned stable audio stack loaded successfully; no operating-system protection was disabled.

