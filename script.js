/* ══════════════════════════════════════════════════════════════
   VoiceGuard AI — Audio Forensics Workbench
   Premium frontend demo — vanilla JS, no frameworks.
   All analysis is simulated for demonstration purposes.
   ══════════════════════════════════════════════════════════════ */

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

/* ── DOM References ── */
const input = $('#audioInput');
const dropZone = $('#dropZone');
const filePrompt = $('#filePrompt');
const fileMeta = $('#fileMeta');
const analyzeButton = $('#analyzeButton');
const emptyReadout = $('#emptyReadout');
const loadingReadout = $('#loadingReadout');
const resultReadout = $('#resultReadout');
const readoutStatus = $('#readoutStatus');
const analysisStage = $('#analysisStage');
const progressLine = $('#progressLine');
const reportButton = $('#reportButton');
const audioPreview = $('#audioPreview');
const recordButton = $('#recordButton');
const recordLabelText = $('#recordLabelText');
const recordDetail = $('#recordDetail');
const recordTimer = $('#recordTimer');
const liveWaveCanvas = $('#liveWave');
const clearHistoryBtn = $('#clearHistory');
const heroSampleBtn = $('#heroSample');

/* ── State ── */
let selectedAudio = null;
let currentResult = null;
let recording = false;
let recorder = null;
let stream = null;
let audioContext = null;
let analyser = null;
let animationFrame = null;
let startedAt = 0;
let timerInterval = null;
let chunks = [];
let dragCounter = 0;
let previousAudioUrl = null;

/* ── Preset Data ── */
const presets = {
  clone: {
    name: 'sample_01_ai_cloned_voice.wav',
    duration: '00:18.42',
    risk: 78,
    verdict: 'LIKELY SYNTHETIC',
    tag: 'ELEVATED RISK',
    confidence: '94%',
    agreement: '87%',
    rate: '48 kHz',
    range: '0–24 kHz',
    severity: 'high',
    evidence: [
      ['Spectral discontinuity', '4.2 kHz', 'HIGH'],
      ['Unnatural pitch jitter', '± 31 Hz', 'MEDIUM'],
      ['Phase inconsistency', '3.8 kHz', 'HIGH'],
      ['Formant instability', 'F2 shift', 'MEDIUM'],
      ['Background-noise mismatch', '−18 dB', 'LOW']
    ]
  },
  human: {
    name: 'sample_02_natural_human_speech.wav',
    duration: '00:14.06',
    risk: 18,
    verdict: 'LIKELY REAL',
    tag: 'LOW RISK',
    confidence: '91%',
    agreement: '89%',
    rate: '44.1 kHz',
    range: '0–22 kHz',
    severity: 'low',
    evidence: [
      ['Natural micro-prosody', 'detected', 'LOW'],
      ['Consistent breath noise', '−32 dB', 'LOW'],
      ['Pitch variation', '± 14 Hz', 'LOW'],
      ['Formant continuity', 'stable', 'LOW']
    ]
  },
  processed: {
    name: 'sample_03_processed_audio.m4a',
    duration: '00:22.73',
    risk: 51,
    verdict: 'INCONCLUSIVE',
    tag: 'REVIEW ADVISED',
    confidence: '68%',
    agreement: '62%',
    rate: '48 kHz',
    range: '0–24 kHz',
    severity: 'medium',
    evidence: [
      ['Compression artifacts', '128 kbps', 'MEDIUM'],
      ['Phase irregularity', '2.9 kHz', 'MEDIUM'],
      ['Noise-floor variation', '−21 dB', 'MEDIUM'],
      ['Pitch jitter', 'within range', 'LOW']
    ]
  }
};

/* ══════════════════════════════════════════════════════════════
   HERO ANIMATION
   ══════════════════════════════════════════════════════════════ */
function initHeroAnimation() {
  const instrument = $('.hero-instrument');
  if (instrument) {
    // Small delay so the user sees the beginning
    setTimeout(() => {
      instrument.classList.add('animate');

      // Show frequency markers staggered
      const markers = $$('.freq-marker');
      markers.forEach((m, i) => {
        setTimeout(() => m.classList.add('visible'), 1200 + i * 500);
      });

      // Show signal notes staggered
      const notes = $$('.signal-note');
      notes.forEach((n, i) => {
        setTimeout(() => n.classList.add('visible'), 1800 + i * 400);
      });

      // Smoothly transition to continuous ambient radar loop after initial scan completes
      setTimeout(() => {
        instrument.classList.add('looping');
      }, 3200);
    }, 300);
  }

  // Subtle periodic cyber glitch burst on hero title
  const glitchEl = $('.glitch-text');
  if (glitchEl) {
    setInterval(() => {
      glitchEl.classList.add('glitch-active');
      setTimeout(() => glitchEl.classList.remove('glitch-active'), 500);
    }, 4500);
  }
}

/* ══════════════════════════════════════════════════════════════
   NAVIGATION — Active Link Tracking
   ══════════════════════════════════════════════════════════════ */
function initNavTracking() {
  const sections = ['analyze', 'live-detect', 'history', 'about'];
  const navLinks = $$('.main-nav a');

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navLinks.forEach(l => l.classList.remove('is-active'));
        const link = $(`.main-nav a[href="#${entry.target.id}"]`);
        if (link) link.classList.add('is-active');
      }
    });
  }, { threshold: 0.3, rootMargin: '-80px 0px -40% 0px' });

  sections.forEach(id => {
    const el = document.getElementById(id);
    if (el) observer.observe(el);
  });
}

/* ══════════════════════════════════════════════════════════════
   FILE UPLOAD — Drag & Drop + Click
   ══════════════════════════════════════════════════════════════ */
const ACCEPTED = ['audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/x-m4a', 'audio/mp4',
                  'audio/flac', 'audio/ogg', 'audio/webm', 'audio/aac'];

function isAudioFile(file) {
  if (ACCEPTED.includes(file.type)) return true;
  // Fallback: check extension
  const ext = file.name.split('.').pop().toLowerCase();
  return ['mp3', 'wav', 'flac', 'ogg', 'm4a', 'webm', 'aac'].includes(ext);
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function handleFile(file) {
  if (!isAudioFile(file)) {
    filePrompt.textContent = 'Unsupported format';
    setTimeout(() => { filePrompt.textContent = 'Drop an audio file here'; }, 2000);
    return;
  }

  selectedAudio = { file, name: file.name, size: file.size, type: file.type };

  // Update UI
  dropZone.classList.add('has-file');
  dropZone.classList.remove('drag-over');
  filePrompt.textContent = file.name;

  fileMeta.innerHTML = `<span>${file.name}</span><span>${formatFileSize(file.size)}</span>`;

  // Show audio preview
  if (previousAudioUrl) URL.revokeObjectURL(previousAudioUrl);
  previousAudioUrl = URL.createObjectURL(file);
  audioPreview.src = previousAudioUrl;
  audioPreview.hidden = false;

  // Show analyze button
  showAnalyzeButton();

  // Deselect samples
  $$('.sample-card').forEach(c => c.setAttribute('aria-pressed', 'false'));
}

function showAnalyzeButton() {
  analyzeButton.hidden = false;
  analyzeButton.style.display = '';
}

dropZone.addEventListener('click', () => input.click());
dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
});

input.addEventListener('change', () => {
  if (input.files.length) handleFile(input.files[0]);
});

// Drag events
dropZone.addEventListener('dragenter', (e) => {
  e.preventDefault();
  dragCounter++;
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
});
dropZone.addEventListener('dragleave', (e) => {
  e.preventDefault();
  dragCounter--;
  if (dragCounter <= 0) { dragCounter = 0; dropZone.classList.remove('drag-over'); }
});
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dragCounter = 0;
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

/* ══════════════════════════════════════════════════════════════
   SAMPLE CARDS
   ══════════════════════════════════════════════════════════════ */
$$('.sample-card').forEach(card => {
  card.addEventListener('click', () => {
    const key = card.dataset.sample;
    const preset = presets[key];
    if (!preset) return;

    // Toggle selection
    $$('.sample-card').forEach(c => c.setAttribute('aria-pressed', 'false'));
    card.setAttribute('aria-pressed', 'true');

    selectedAudio = { name: preset.name, size: 0, type: 'sample', presetKey: key };

    // Update file meta
    dropZone.classList.remove('has-file', 'drag-over');
    filePrompt.textContent = preset.name;
    fileMeta.innerHTML = `<span>${preset.name}</span><span>DEMO SAMPLE</span>`;
    audioPreview.hidden = true;

    showAnalyzeButton();
  });
});

// Hero "Try a Sample" button
if (heroSampleBtn) {
  heroSampleBtn.addEventListener('click', () => {
    const analyzeSection = document.getElementById('analyze');
    if (analyzeSection) analyzeSection.scrollIntoView({ behavior: 'smooth' });
    // Auto-select first sample after scroll
    setTimeout(() => {
      const first = $('.sample-card[data-sample="clone"]');
      if (first) first.click();
    }, 500);
  });
}

/* ══════════════════════════════════════════════════════════════
   MICROPHONE RECORDING
   ══════════════════════════════════════════════════════════════ */
function formatTime(ms) {
  const totalSec = ms / 1000;
  const min = Math.floor(totalSec / 60).toString().padStart(2, '0');
  const sec = Math.floor(totalSec % 60).toString().padStart(2, '0');
  const cs = Math.floor((totalSec % 1) * 100).toString().padStart(2, '0');
  return `${min}:${sec}.${cs}`;
}

function drawLiveWave() {
  if (!analyser) return;
  const ctx = liveWaveCanvas.getContext('2d');
  const w = liveWaveCanvas.width;
  const h = liveWaveCanvas.height;
  const bufLen = analyser.frequencyBinCount;
  const data = new Uint8Array(bufLen);

  function draw() {
    if (!recording) return;
    animationFrame = requestAnimationFrame(draw);
    analyser.getByteTimeDomainData(data);
    ctx.clearRect(0, 0, w, h);

    // Draw amplitude bars
    const barCount = 28;
    const barW = (w / barCount) - 2;
    analyser.getByteFrequencyData(data);
    for (let i = 0; i < barCount; i++) {
      const idx = Math.floor(i * bufLen / barCount);
      const val = data[idx] / 255;
      const barH = Math.max(2, val * h * 0.85);
      const x = i * (barW + 2);
      const y = (h - barH) / 2;

      ctx.fillStyle = `rgba(232, 133, 12, ${0.4 + val * 0.6})`;
      ctx.fillRect(x, y, barW, barH);
    }
  }
  draw();
}

async function startRecording() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    recordDetail.textContent = 'Microphone access was unavailable.';
    return;
  }

  recording = true;
  recorder = new MediaRecorder(stream);
  chunks = [];
  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  recorder.onstop = () => {
    const blob = new Blob(chunks, { type: 'audio/webm' });
    if (previousAudioUrl) URL.revokeObjectURL(previousAudioUrl);
    previousAudioUrl = URL.createObjectURL(blob);
    audioPreview.src = previousAudioUrl;
    audioPreview.hidden = false;

    selectedAudio = {
      name: `recording_${new Date().toISOString().slice(11,19).replace(/:/g,'')}.webm`,
      size: blob.size,
      type: 'recording'
    };

    showAnalyzeButton();

    recordLabelText.textContent = 'Recording captured';
    recordDetail.textContent = 'Preview is ready. Analyze when you are set.';
  };

  recorder.start();
  startedAt = performance.now();

  // Audio analysis for visualization
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 256;
  const source = audioContext.createMediaStreamSource(stream);
  source.connect(analyser);

  drawLiveWave();

  timerInterval = setInterval(() => {
    recordTimer.textContent = formatTime(performance.now() - startedAt);
  }, 50);

  recordButton.setAttribute('aria-pressed', 'true');
  recordButton.setAttribute('aria-label', 'Stop recording');
  recordLabelText.textContent = 'Recording in progress';
  recordDetail.textContent = 'Tap the microphone again to finish.';
}

function stopRecording() {
  recording = false;
  if (recorder && recorder.state !== 'inactive') recorder.stop();
  if (stream) stream.getTracks().forEach(t => t.stop());
  if (audioContext) audioContext.close();
  if (animationFrame) cancelAnimationFrame(animationFrame);
  if (timerInterval) clearInterval(timerInterval);

  recordButton.setAttribute('aria-pressed', 'false');
  recordButton.setAttribute('aria-label', 'Start recording');
  audioContext = null;
  analyser = null;
  stream = null;
}

recordButton.addEventListener('click', () => {
  if (recording) stopRecording();
  else startRecording();
});

/* ══════════════════════════════════════════════════════════════
   ANALYSIS PIPELINE (Simulated)
   ══════════════════════════════════════════════════════════════ */
const STAGES = [
  'Upload received…',
  'Extracting spectral features…',
  'Mapping acoustic fingerprints…',
  'Running ensemble model…',
  'Comparing vocal signatures…',
  'Generating forensic report…'
];

function generateRandomResult(filename) {
  const risk = Math.floor(Math.random() * 80) + 10;
  let verdict, tag, severity;
  if (risk >= 65) {
    verdict = 'LIKELY SYNTHETIC'; tag = 'ELEVATED RISK'; severity = 'high';
  } else if (risk <= 35) {
    verdict = 'LIKELY REAL'; tag = 'LOW RISK'; severity = 'low';
  } else {
    verdict = 'INCONCLUSIVE'; tag = 'REVIEW ADVISED'; severity = 'medium';
  }
  return {
    name: filename,
    duration: `00:${(Math.random() * 30 + 5).toFixed(2).padStart(5, '0')}`,
    risk,
    verdict,
    tag,
    confidence: `${Math.floor(Math.random() * 20 + 70)}%`,
    agreement: `${Math.floor(Math.random() * 20 + 65)}%`,
    rate: ['16 kHz', '44.1 kHz', '48 kHz'][Math.floor(Math.random() * 3)],
    range: '0–24 kHz',
    severity,
    evidence: [
      ['Spectral analysis', `${(Math.random() * 6 + 1).toFixed(1)} kHz`, risk > 50 ? 'HIGH' : 'LOW'],
      ['Pitch jitter', `± ${Math.floor(Math.random() * 30 + 5)} Hz`, risk > 40 ? 'MEDIUM' : 'LOW'],
      ['Phase coherence', `${(Math.random() * 5 + 1).toFixed(1)} kHz`, risk > 55 ? 'HIGH' : 'MEDIUM'],
      ['Formant stability', risk > 60 ? 'F2 drift' : 'stable', risk > 50 ? 'MEDIUM' : 'LOW'],
      ['Background noise', `−${Math.floor(Math.random() * 20 + 12)} dB`, 'LOW']
    ]
  };
}

function showLoading() {
  emptyReadout.hidden = true;
  resultReadout.hidden = true;
  loadingReadout.hidden = false;
  readoutStatus.textContent = 'ANALYZING';
  progressLine.style.width = '0%';
}

function renderResult(result) {
  currentResult = result;

  loadingReadout.hidden = true;
  resultReadout.hidden = false;
  readoutStatus.textContent = 'REPORT READY';

  // Verdict
  const verdictText = $('#verdictText');
  const verdictTag = $('#verdictTag');
  verdictText.textContent = result.verdict;
  verdictTag.textContent = result.tag;

  // Set severity class on result container
  resultReadout.className = 'result-readout';
  resultReadout.classList.add(`verdict-${result.severity}`);

  // Gauge
  const gaugeValue = $('#gaugeValue');
  const gaugeScore = $('#gaugeScore');
  const circumference = 2 * Math.PI * 63; // ~395.84
  const offset = circumference - (result.risk / 100) * circumference;

  // Color the gauge
  if (result.severity === 'high') gaugeValue.style.stroke = 'var(--danger)';
  else if (result.severity === 'low') gaugeValue.style.stroke = 'var(--safe)';
  else gaugeValue.style.stroke = 'var(--warn)';

  // Animate gauge
  gaugeValue.style.strokeDashoffset = circumference;
  gaugeScore.textContent = '0';
  requestAnimationFrame(() => {
    gaugeValue.style.strokeDashoffset = offset;
  });

  // Animate score number
  animateNumber(gaugeScore, 0, result.risk, 1200);

  // Meta
  $('#confidence').textContent = result.confidence;
  $('#agreement').textContent = result.agreement;
  $('#duration').textContent = result.duration;
  $('#rate').textContent = result.rate;
  $('#range').textContent = result.range;

  // Evidence rows
  const evidenceContainer = $('#evidenceRows');
  evidenceContainer.innerHTML = '';
  result.evidence.forEach(([name, detail, sev], idx) => {
    const row = document.createElement('div');
    row.className = 'evidence-row';
    row.style.animationDelay = `${idx * 0.08}s`;
    row.innerHTML = `
      <span class="evidence-name">${name}</span>
      <span class="evidence-detail">${detail}</span>
      <span class="evidence-severity severity-${sev}">${sev}</span>
    `;
    evidenceContainer.appendChild(row);
  });

  // Enable report
  reportButton.disabled = false;

  // Save to history
  saveToHistory(result);
}

function animateNumber(el, from, to, durationMs) {
  const start = performance.now();
  function tick(now) {
    const t = Math.min((now - start) / durationMs, 1);
    const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
    el.textContent = Math.round(from + (to - from) * eased);
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

async function runAnalysis() {
  if (!selectedAudio) return;

  showLoading();
  analyzeButton.hidden = true;
  analyzeButton.style.display = 'none';

  for (let i = 0; i < STAGES.length; i++) {
    analysisStage.textContent = STAGES[i];
    progressLine.style.width = `${((i + 1) / STAGES.length) * 100}%`;
    await sleep(600 + Math.random() * 400);
  }

  // Determine result
  let result;
  if (selectedAudio.presetKey && presets[selectedAudio.presetKey]) {
    result = { ...presets[selectedAudio.presetKey] };
  } else {
    result = generateRandomResult(selectedAudio.name);
  }

  renderResult(result);
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

analyzeButton.addEventListener('click', runAnalysis);

/* ══════════════════════════════════════════════════════════════
   HISTORY — localStorage
   ══════════════════════════════════════════════════════════════ */
const HISTORY_KEY = 'voiceguard_history';

function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
  } catch { return []; }
}

function saveToHistory(result) {
  const history = getHistory();
  const entry = {
    id: Date.now(),
    name: result.name,
    timestamp: new Date().toLocaleTimeString('en-GB'),
    verdict: result.verdict,
    risk: result.risk,
    severity: result.severity,
    result: result
  };
  history.unshift(entry);
  if (history.length > 20) history.pop();
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  renderHistory();
}

function renderHistory() {
  const list = $('#historyList');
  const history = getHistory();

  if (!history.length) {
    list.innerHTML = '<p class="history-empty">Your completed analyses will appear here.</p>';
    return;
  }

  list.innerHTML = '';
  history.forEach(entry => {
    const item = document.createElement('div');
    item.className = 'history-item';
    item.setAttribute('role', 'button');
    item.setAttribute('tabindex', '0');
    item.setAttribute('aria-label', `Restore ${entry.name} analysis`);

    // Score color
    let scoreColor;
    if (entry.severity === 'high') scoreColor = 'var(--danger)';
    else if (entry.severity === 'low') scoreColor = 'var(--safe)';
    else scoreColor = 'var(--warn)';

    // Verdict color
    let verdictColor;
    if (entry.severity === 'high') verdictColor = 'var(--danger)';
    else if (entry.severity === 'low') verdictColor = 'var(--safe)';
    else verdictColor = 'var(--warn)';

    item.innerHTML = `
      <span class="h-name">${entry.name}</span>
      <span class="h-time">${entry.timestamp}</span>
      <span class="h-verdict" style="color:${verdictColor}">${entry.verdict}</span>
      <span class="h-score" style="border-color:${scoreColor}; color:${scoreColor}">${entry.risk}</span>
    `;

    item.addEventListener('click', () => restoreFromHistory(entry));
    item.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); restoreFromHistory(entry); }
    });

    list.appendChild(item);
  });
}

function restoreFromHistory(entry) {
  // Scroll to workspace
  const analyzeSection = document.getElementById('analyze');
  if (analyzeSection) analyzeSection.scrollIntoView({ behavior: 'smooth' });

  // Set file meta
  selectedAudio = { name: entry.name };
  filePrompt.textContent = entry.name;
  fileMeta.innerHTML = `<span>${entry.name}</span><span>FROM HISTORY</span>`;

  // Render the stored result
  setTimeout(() => renderResult(entry.result), 400);
}

clearHistoryBtn.addEventListener('click', () => {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
});

/* ══════════════════════════════════════════════════════════════
   REPORT DOWNLOAD
   ══════════════════════════════════════════════════════════════ */
reportButton.addEventListener('click', () => {
  if (!currentResult) return;

  const report = {
    tool: 'VoiceGuard AI — Audio Forensics',
    disclaimer: 'DEMO SIMULATION — This report was generated by a frontend demonstration. No actual ML model was executed.',
    generated: new Date().toISOString(),
    filename: currentResult.name,
    verdict: currentResult.verdict,
    riskScore: currentResult.risk,
    confidence: currentResult.confidence,
    modelAgreement: currentResult.agreement,
    audioDuration: currentResult.duration,
    sampleRate: currentResult.rate,
    frequencyRange: currentResult.range,
    forensicEvidence: currentResult.evidence.map(([name, detail, sev]) => ({
      indicator: name,
      detail,
      severity: sev
    }))
  };

  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `voiceguard_report_${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

/* ══════════════════════════════════════════════════════════════
   LANGUAGE SWITCHER (demo labels)
/* ══════════════════════════════════════════════════════════════
   LANGUAGE SWITCHER & TRANSLATIONS
   ══════════════════════════════════════════════════════════════ */
const translations = {
  en: {
    heroEyebrow: 'SYNTHETIC SPEECH INVESTIGATION',
    heroMark: 'is real.',
    heroText: 'VoiceGuard AI analyzes acoustic patterns, spectral behavior, and vocal characteristics to identify signs of synthetic speech.',
    analyzeBtn: 'Analyze Audio',
    trySampleBtn: 'Try a Sample',
    statSignals: 'Signals modeled',
    statWindow: 'Analysis window',
    statFormats: 'Formats read',
    workspaceEyebrow: 'WORKSPACE / 01',
    workspaceTitle: 'Bring a voice into focus.',
    workspaceDesc: 'Load a recording, select a controlled sample, or capture a short clip live.',
    dropPrompt: 'Drop an audio file here',
    dropSub: 'or browse from your device',
    sampleLabel: 'DEMO SAMPLE',
    orLabel: 'OR',
    recordPrompt: 'Record a sample',
    recordDesc: 'Use your microphone for a short clip.',
    analyzeVoiceBtn: 'Analyze Voice',
    readoutTitle: 'FORENSIC READOUT',
    statusAwaiting: 'AWAITING INPUT',
    emptyPrompt: 'Load an audio signal to initialize a forensic readout.',
    uploadReceived: 'Upload received…',
    demoDisclaimer: 'DEMO SIMULATION — NO LIVE MODEL IS RUNNING',
    verdictLabel: 'VERDICT',
    riskScoreLabel: 'RISK SCORE',
    dtConfidence: 'Confidence',
    dtAgreement: 'Model agreement',
    dtDuration: 'Audio duration',
    dtRate: 'Sample rate',
    dtRange: 'Frequency range',
    evidenceLabel: 'Forensic Evidence',
    downloadReportBtn: 'Download Report',
    methodEyebrow: 'METHOD / 02',
    methodTitle: 'Two signals. One\nforensic decision.',
    specModelTitle: 'Spectrogram Model',
    specModelDesc: 'The system studies frequency patterns and spectral structures that can reveal artifacts commonly associated with synthetic speech.',
    acousticModelTitle: 'Acoustic Feature Ensemble',
    acousticModelDesc: 'The system examines pitch behavior, timing, formants, jitter, energy distribution, and other acoustic characteristics.',
    combinedSignalLabel: 'COMBINED SIGNAL',
    forensicResultLabel: 'FORENSIC\nRESULT',
    sessionLogEyebrow: 'SESSION LOG',
    historyTitle: 'Previous investigations.',
    clearHistoryBtn: 'Clear history',
    historyEmpty: 'Your completed analyses will appear here.',
    trustTitle: 'Read this before you decide.',
    trustText: 'VoiceGuard AI is a probabilistic detection tool, not proof of authenticity. Results can be affected by compression, background noise, recording quality, voice characteristics, and evolving synthesis techniques. It should not be the sole basis for high-stakes decisions.',
    footerSubtitle: 'Audio Forensics',
    footerHackathon: 'Built for SIH Hackathon',
    footerDocs: 'Documentation'
  },
  hi: {
    heroEyebrow: 'सिंथेटिक भाषण जांच',
    heroMark: 'असली है।',
    heroText: 'VoiceGuard AI सिंथेटिक आवाज के संकेतों की पहचान करने के लिए ध्वनिक पैटर्न, स्पेक्ट्रल व्यवहार और स्वर विशेषताओं का विश्लेषण करता है।',
    analyzeBtn: 'ऑडियो विश्लेषण करें',
    trySampleBtn: 'नमूना आज़माएँ',
    statSignals: 'मॉडल किए गए संकेत',
    statWindow: 'विश्लेषण विंडो',
    statFormats: 'समर्थित प्रारूप',
    workspaceEyebrow: 'कार्यक्षेत्र / 01',
    workspaceTitle: 'आवाज को स्पष्टता में लाएं।',
    workspaceDesc: 'रिकॉर्डिंग लोड करें, नमूना चुनें, या लाइव रिकॉर्ड करें।',
    dropPrompt: 'ऑडियो फ़ाइल यहाँ छोड़ें',
    dropSub: 'या अपने डिवाइस से चुनें',
    sampleLabel: 'डेमो नमूना',
    orLabel: 'या',
    recordPrompt: 'आवाज रिकॉर्ड करें',
    recordDesc: 'एक छोटी क्लिप के लिए माइक्रोफ़ोन का उपयोग करें।',
    analyzeVoiceBtn: 'आवाज़ का विश्लेषण करें',
    readoutTitle: 'फोरेंसिक रीडआउट',
    statusAwaiting: 'प्रतीक्षा में',
    emptyPrompt: 'फोरेंसिक रीडआउट शुरू करने के लिए एक ऑडियो सिग्नल लोड करें।',
    uploadReceived: 'अपलोड प्राप्त हुआ…',
    demoDisclaimer: 'डेमो सिमुलेशन — कोई लाइव मॉडल नहीं चल रहा है',
    verdictLabel: 'निर्णय',
    riskScoreLabel: 'जोखिम स्कोर',
    dtConfidence: 'विश्वास स्तर',
    dtAgreement: 'मॉडल सहमति',
    dtDuration: 'ऑडियो अवधि',
    dtRate: 'सैंपल दर',
    dtRange: 'आवृत्ति सीमा',
    evidenceLabel: 'फोरेंसिक साक्ष्य',
    downloadReportBtn: 'रिपोर्ट डाउनलोड करें',
    methodEyebrow: 'विधि / 02',
    methodTitle: 'दो संकेत। एक\nफोरेंसिक निर्णय।',
    specModelTitle: 'स्पेक्ट्रोग्राम मॉडल',
    specModelDesc: 'सिस्टम आवृत्ति पैटर्न और स्पेक्ट्रल संरचनाओं का अध्ययन करता है जो कृत्रिम भाषण के संकेतों को प्रकट करते हैं।',
    acousticModelTitle: 'ध्वनिक विशेषता पहनावा',
    acousticModelDesc: 'सिस्टम पिच व्यवहार, समय, फॉर्मैंट्स, घबराहट और ऊर्जा वितरण की जांच करता है।',
    combinedSignalLabel: 'संयुक्त संकेत',
    forensicResultLabel: 'फोरेंसिक\nपरिणाम',
    sessionLogEyebrow: 'सत्र लॉग',
    historyTitle: 'पिछली जांचें।',
    clearHistoryBtn: 'इतिहास साफ़ करें',
    historyEmpty: 'आपके पूर्ण किए गए विश्लेषण यहां दिखाई देंगे।',
    trustTitle: 'निर्णय लेने से पहले यह पढ़ें।',
    trustText: 'VoiceGuard AI एक संभाव्यता-आधारित पहचान उपकरण है, प्रामाणिकता का अंतिम प्रमाण नहीं। परिणाम संपीड़न, पृष्ठभूमि शोर और विकसित संश्लेषण तकनीकों से प्रभावित हो सकते हैं।',
    footerSubtitle: 'ऑडियो फोरेंसिक्स',
    footerHackathon: 'स्मार्ट इंडिया हैकाथॉन के लिए निर्मित',
    footerDocs: 'दस्तावेज़ीकरण'
  },
  bn: {
    heroEyebrow: 'সিন্থেটিক স্পিচ তদন্ত',
    heroMark: 'আসল কিনা।',
    heroText: 'VoiceGuard AI সিন্থেটিক ভয়েসের লক্ষণ শনাক্ত করতে শাব্দিক প্যাটার্ন, বর্ণালী আচরণ এবং কণ্ঠস্বরের বৈশিষ্ট্য বিশ্লেষণ করে।',
    analyzeBtn: 'অডিও বিশ্লেষণ করুন',
    trySampleBtn: 'নমুনা চেষ্টা করুন',
    statSignals: 'মডেল করা সংকেত',
    statWindow: 'বিশ্লেষণ উইন্ডো',
    statFormats: 'সমর্থিত ফরম্যাট',
    workspaceEyebrow: 'ওয়ার্কস্পেস / ০১',
    workspaceTitle: 'কণ্ঠস্বর স্পষ্ট করে তুলুন।',
    workspaceDesc: 'একটি রেকর্ডিং লোড করুন, একটি নমুনা নির্বাচন করুন, বা লাইভ রেকর্ড করুন।',
    dropPrompt: 'এখানে অডিও ফাইল ফেলুন',
    dropSub: 'অথবা আপনার ডিভাইস থেকে ব্রাউজ করুন',
    sampleLabel: 'ডেমো নমুনা',
    orLabel: 'বা',
    recordPrompt: 'একটি নমুনা রেকর্ড করুন',
    recordDesc: 'সংক্ষিপ্ত ক্লিপের জন্য মাইক্রোফোন ব্যবহার করুন।',
    analyzeVoiceBtn: 'ভয়েস বিশ্লেষণ করুন',
    readoutTitle: 'ফরেনসিক রিডআউট',
    statusAwaiting: 'ইনপুটের অপেক্ষায়',
    emptyPrompt: 'ফরেনসিক রিডআউট শুরু করতে একটি অডিও সংকেত লোড করুন।',
    uploadReceived: 'আপলোড গৃহীত হয়েছে…',
    demoDisclaimer: 'ডেমো সিমুলেশন — কোনো লাইভ মডেল চলছে না',
    verdictLabel: 'রায়',
    riskScoreLabel: 'ঝুঁকি স্কোর',
    dtConfidence: 'আত্মবিশ্বাস',
    dtAgreement: 'মডেল সম্মতি',
    dtDuration: 'অডিওর সময়কাল',
    dtRate: 'নমুনা হার',
    dtRange: 'ফ্রিকোয়েন্সি পরিসীমা',
    evidenceLabel: 'ফরেনসিক প্রমাণ',
    downloadReportBtn: 'রিপোর্ট ডাউনলোড করুন',
    methodEyebrow: 'পদ্ধতি / ০২',
    methodTitle: 'দুটি সংকেত। একটি\nফরেনসিক সিদ্ধান্ত।',
    specModelTitle: 'স্পেকট্রোগ্রাম মডেল',
    specModelDesc: 'সিস্টেমটি ফ্রিকোয়েন্সি প্যাটার্ন এবং বর্ণালী কাঠামো অধ্যয়ন করে যা কৃত্রিম বক্তৃতার ত্রুটি উন্মোচন করে।',
    acousticModelTitle: 'অ্যাকোস্টিক বৈশিষ্ট্য সংগ্রহ',
    acousticModelDesc: 'সিস্টেম পিচ আচরণ, সময়, ফরম্যান্ট, জিটার এবং শক্তি বণ্টন পরীক্ষা করে।',
    combinedSignalLabel: 'সম্মিলিত সংকেত',
    forensicResultLabel: 'ফরেনসিক\nফলাফল',
    sessionLogEyebrow: 'সেশন লগ',
    historyTitle: 'পূর্ববর্তী তদন্ত।',
    clearHistoryBtn: 'ইতিহাস মুছুন',
    historyEmpty: 'আপনার সম্পন্ন করা বিশ্লেষণগুলি এখানে প্রদর্শিত হবে।',
    trustTitle: 'সিদ্ধান্ত নেওয়ার আগে এটি পড়ুন।',
    trustText: 'VoiceGuard AI একটি সম্ভাব্যতা-ভিত্তিক শনাক্তকরণ সরঞ্জাম, প্রামাণিকতার নিশ্চিত প্রমাণ নয়। ফলাফল কম্প্রেশন, ব্যাকগ্রাউন্ড শব্দ এবং উন্নত সংশ্লেষণ কৌশলের দ্বারা প্রভাবিত হতে পারে।',
    footerSubtitle: 'অডিও ফরেনসিক্স',
    footerHackathon: 'এসআইএইচ হ্যাকাথনের জন্য নির্মিত',
    footerDocs: 'নথিপত্র'
  }
};

function setLanguage(lang) {
  const t = translations[lang];
  if (!t) return;

  // Update button active state
  $$('.languages button, .lang-switcher button').forEach(b => {
    const isTarget = b.dataset.lang === lang;
    b.classList.toggle('is-selected', isTarget);
    b.setAttribute('aria-pressed', isTarget ? 'true' : 'false');
  });

  // Update all translatable elements with data-key
  $$('.translatable[data-key]').forEach(el => {
    const key = el.dataset.key;
    if (t[key] !== undefined) {
      if (t[key].includes('\n')) {
        el.innerHTML = t[key].replace(/\n/g, '<br />');
      } else {
        el.textContent = t[key];
      }
    }
  });

  // Keep dynamic file drop prompt in sync if no file is currently selected
  if (!dropZone.classList.contains('has-file')) {
    filePrompt.textContent = t.dropPrompt;
  }
}

$$('.languages button, .lang-switcher button').forEach(btn => {
  btn.addEventListener('click', () => {
    const lang = btn.dataset.lang;
    if (lang) setLanguage(lang);
  });
});

/* ══════════════════════════════════════════════════════════════
   INIT
   ══════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  initHeroAnimation();
  initNavTracking();
  renderHistory();

  // Ensure analyze button hidden initially
  analyzeButton.hidden = true;
  analyzeButton.style.display = 'none';
});
