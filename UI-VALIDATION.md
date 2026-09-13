# VoiceGuard UI/UX upgrade — September 13, 2026

Implemented directly in the original project:
`C:\Users\SUPRIYA\.gemini\antigravity\scratch\sih26104-voiceguard-ai`

## What changed

- Warm ivory page, indigo primary actions, lavender readouts, sage supporting surfaces, peach prevention/about accents. No neon treatment.
- Larger sentence-case hero, framed acoustic core, clearer type hierarchy, stronger upload/readout composition, and quieter prevention rows.
- Shared tokens in `tokens.css`; existing component styles edited in place and formatted for readability.
- Finite entrance choreography, grouped scroll reveals, tactile controls, loading-button feedback, server-driven progress, completed-progress transition, animated evidence bars and result entry.
- Idle acoustic geometry settles instead of rotating continuously; pointer movement is small and reduced-motion aware. Live canvas buffers are reused, canvas resizing happens only when dimensions change, and rendering is capped near 30 fps (lower with reduced motion).
- Mobile CTAs stack with large touch targets; pipeline and prevention items become intentional single-column rows. Mobile navigation closes with Escape and restores focus.
- Existing uploads, audio preview, microphone flow, APIs, scoring pipeline, timeline playback, reports, languages and session history retained.
- Clarified browser speech-recognition privacy wording. Added a brand favicon.

## Files

Modified: `index.html`, `styles.css`, `script.js`, `core-visual.js`, `api/server.py`, `tests/test_report.cjs`.
Added: `tokens.css`, `UI-VALIDATION.md`.
The only server change is adding `tokens.css` to its explicit static-asset allowlist. The report test now tolerates existing source whitespace. Model/checkpoint files, backend scoring logic, dependencies and deployment configuration were not changed.

## Verification

- Python regression suite: **52 passed, 1 warning** in 31.67 seconds.
- JavaScript syntax: `script.js`, `core-visual.js`, `recorder-worklet.js` passed.
- Existing report-builder test passed: actual download trigger, report content, print stylesheet and hostile-filename escaping.
- Existing AudioWorklet test passed: stereo averaging, PCM block length, buffer reset, empty input.
- Headless Microsoft Edge browser journeys passed with **zero page exceptions**: real development-sample upload and analysis, server progress, evidence disclosure, speaker segment playback, actual HTML report download, history, English/Hindi/Bengali controls, settings, invalid-file rejection, mobile menu, keyboard tabs, reduced motion and offline state.
- Synthetic browser microphone capture, stop-to-analysis, and stream/context cleanup passed. No personal microphone audio was collected.
- No document or non-positioned content overflow at **1440, 1280, 1024, 768, 480, 390, 360**, plus 320, 375 and 414 pixels.
- Rendered desktop, mobile, empty, processing and real result states inspected; a second pass corrected metadata spacing and scoped live transitions.

## Baseline issues and environment

The initial report test failed because its source extractor expected older minified formatting; this is fixed. Initial Python execution encountered a sandbox cache-write error. Setting `NUMBA_CACHE_DIR` to a writable task folder and pytest `--basetemp` to a writable test folder resolved it. No OS protection was disabled and no packages were installed.

Commands from the project folder:
```powershell
$env:NUMBA_CACHE_DIR = "$env:TEMP\voiceguard-numba-cache"
python -m pytest tests -q --disable-warnings --tb=line -p no:cacheprovider --basetemp="$env:TEMP\voiceguard-pytest-check"
node --check script.js
node --check core-visual.js
node --check recorder-worklet.js
node tests/test_report.cjs
node tests/test_audio_worklet.cjs
python main.py
```

Open http://127.0.0.1:8000. The local server was left running for review. No production deployment was performed.

## Deliverables and limits

`voiceguard-upgrade.zip` contains the changed/new files with project-relative paths, for applying to another copy of this same project. The original folder already has the changes; the ZIP is not a standalone application and intentionally omits unchanged models and dependencies.

Desktop/mobile screenshots and a real-analysis screenshot accompany the browser-check record. Browser automation validates functionality, not detector accuracy. Physical microphone permission prompts and the optional external speech-recognition service were not validated; synthetic microphone capture was. Optional pretrained backends still require their existing configured weights. No new accuracy or calibration claim is made.
