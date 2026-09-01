/**
 * VoiceGuard AI - Ultra-Premium Cyber Forensic Dashboard Logic
 * SIH26104 - Neural Voice Deepfake & Clone Forensic System
 */

// Global Application State
const state = {
  currentFile: null,
  isRecording: false,
  audioContext: null,
  analyser: null,
  micStream: null,
  scriptProcessor: null,
  pcmBuffer: [],
  targetSampleRate: 16000,
  chunkIntervalId: null,
  streamChunkIndex: 0,
  riskTrajectory: [],
  language: 'en',
  thresholdLow: 30,
  thresholdHigh: 70,
  lastAnalysis: null,
  demoSamples: [],
};

// Comprehensive Localization Dictionary
const i18n = {
  en: {
    appTitle: "VOICEGUARD AI",
    tagline: "Real-Time Neural Voice Deepfake & Clone Forensic System",
    tabUpload: "File Upload",
    tabSamples: "1-Click Benchmark",
    tabLive: "Live Interception",
    btnAnalyze: "RUN FORENSIC SCAN",
    btnAnalyzing: "ANALYZING AUDIO SPECTRUM...",
    riskScoreTitle: "COMPOSITE THREAT RISK",
    syntheticProb: "Estimated Synthetic Probability:",
    confidenceLabel: "Forensic Confidence:",
    indicatorsTitle: "FORENSIC SIGNALS & ACOUSTIC INDICATORS",
    spectrogramTitle: "MEL-SPECTROGRAM FORENSIC HEATMAP",
    metricsTitle: "ACOUSTIC METRICS",
    liveStatusIdle: "Ready to monitor. Click 'Start Live Interception' to analyze incoming speech stream.",
    liveStatusRecording: "STREAMING & ANALYZING LIVE AUDIO (16 kHz PCM WAV CHUNKS)...",
    btnStartLive: "START LIVE INTERCEPTION",
    btnStopLive: "STOP MONITORING",
    downloadReport: "EXPORT FORENSIC AUDIT REPORT (JSON)",
  },
  hi: {
    appTitle: "वॉइसगार्ड एआई",
    tagline: "आर्टिफिशियल इंटेलिजेंस आवाज़ क्लोनिंग एवं डीपफेक पहचान प्रणाली",
    tabUpload: "ऑडियो अपलोड",
    tabSamples: "1-क्लिक बेंचमार्क",
    tabLive: "लाइव इंटरसेप्शन",
    btnAnalyze: "फोरेंसिक स्कैन चलाएं",
    btnAnalyzing: "ऑडियो स्पेक्ट्रम का विश्लेषण जारी है...",
    riskScoreTitle: "समग्र जोखिम स्कोर (RISK SCORE)",
    syntheticProb: "अनुमानित सिंथेटिक/क्लोन संभावना:",
    confidenceLabel: "जाँच विश्वसनीयता:",
    indicatorsTitle: "फोरेंसिक संकेत एवं ध्वनि विश्लेषण",
    spectrogramTitle: "मेल-स्पेक्ट्रोग्राम फोरेंसिक हीटमैप",
    metricsTitle: "ध्वनिक भौतिक पैरामीटर्स",
    liveStatusIdle: "निगरानी के लिए तैयार। लाइव ऑडियो स्ट्रीम शुरू करने के लिए क्लिक करें।",
    liveStatusRecording: "लाइव ऑडियो स्ट्रीम का विश्लेषण जारी है (16 kHz PCM)...",
    btnStartLive: "लाइव निगरानी शुरू करें",
    btnStopLive: "निगरानी बंद करें",
    downloadReport: "फोरेंसिक रिपोर्ट डाउनलोड करें (JSON)",
  },
  bn: {
    appTitle: "ভয়েসগার্ড এআই",
    tagline: "কৃত্রিম ভয়েস ক্লোন ও ডিপফেক সনাক্তকরণ সিস্টেম",
    tabUpload: "অডিও আপলোড",
    tabSamples: "১-ক্লিক বেঞ্চমার্ক",
    tabLive: "লাইভ ইন্টারসেপশন",
    btnAnalyze: "ফরেনসিক স্ক্যান শুরু করুন",
    btnAnalyzing: "অডিও স্পেকট্রাম বিশ্লেষণ করা হচ্ছে...",
    riskScoreTitle: "সম্মিলিত ঝুঁকি স্কোর (RISK SCORE)",
    syntheticProb: "আনুমানিক সিন্থেটিক ভয়েস সম্ভাবনা:",
    confidenceLabel: "ফরেনসিক আত্মবিশ্বাস:",
    indicatorsTitle: "ফরেনসিক সংকেত ও ধ্বনি বিশ্লেষণ",
    spectrogramTitle: "মেল-স্পেকট্রোগ্রাম হিটম্যাপ",
    metricsTitle: "অ্যাকোস্টিক শারীরিক প্যারামিটার",
    liveStatusIdle: "লাইভ অডিও মনিটর করার জন্য প্রস্তুত।",
    liveStatusRecording: "লাইভ অডিও বিশ্লেষণ চলছে...",
    btnStartLive: "লাইভ ট্র্যাকিং শুরু করুন",
    btnStopLive: "ট্র্যাকিং বন্ধ করুন",
    downloadReport: "ফরেনসিক রিপোর্ট ডাউনলোড করুন (JSON)",
  }
};

// On Page Load Initialization
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initDropZone();
  initSampleAudios();
  initThresholdModal();
  initLanguageSwitcher();
  initExportReport();
  checkBackendHealth();
  drawEmptyRiskChart();
});

// Check Server Engine Health & Latency
async function checkBackendHealth() {
  const statusPill = document.getElementById('systemStatusPill');
  const latencyVal = document.getElementById('headerLatencyVal');
  const t0 = performance.now();
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    const roundTrip = Math.round(performance.now() - t0);
    if (data.status === 'healthy') {
      const dev = (data.device || 'CPU').toUpperCase();
      statusPill.innerHTML = `<span class="inline-block w-2 h-2 rounded-full bg-emerald-400 mr-2 animate-ping"></span> CORE ACTIVE (${dev}) • DUAL-ENGINE READY`;
      statusPill.className = 'px-3.5 py-1.5 bg-emerald-950/60 border border-emerald-500/40 text-emerald-300 text-xs font-mono rounded-full flex items-center shadow-lg shadow-emerald-950/50';
      if (latencyVal) latencyVal.textContent = `${roundTrip}ms Latency`;
    }
  } catch (err) {
    statusPill.innerHTML = `<span class="inline-block w-2 h-2 rounded-full bg-amber-400 mr-2"></span> INITIALIZING ENGINE...`;
    statusPill.className = 'px-3.5 py-1.5 bg-amber-950/60 border border-amber-500/40 text-amber-300 text-xs font-mono rounded-full flex items-center';
  }
}

// Tab Switching
function initTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const target = tab.dataset.tab;
      document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.add('hidden');
      });
      const activeContent = document.getElementById(target);
      if (activeContent) activeContent.classList.remove('hidden');
    });
  });
}

// Drag & Drop / File Input Handling
function initDropZone() {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('audioFileInput');
  const analyzeBtn = document.getElementById('btnAnalyze');

  // Direct click anywhere on dropzone opens native file picker
  dropZone.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  analyzeBtn.addEventListener('click', () => {
    if (state.currentFile) {
      runForensicAnalysis(state.currentFile);
    }
  });
}

function handleFileSelected(file) {
  state.currentFile = file;
  const fileNameDisplay = document.getElementById('selectedFileName');
  const fileSizeDisplay = document.getElementById('selectedFileSize');
  const filePreview = document.getElementById('filePreviewCard');
  const audioPlayer = document.getElementById('sourceAudioPlayer');
  const analyzeBtn = document.getElementById('btnAnalyze');

  fileNameDisplay.textContent = file.name;
  fileSizeDisplay.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB • ${file.type || 'audio/wav'}`;
  audioPlayer.src = URL.createObjectURL(file);
  filePreview.classList.remove('hidden');
  
  analyzeBtn.disabled = false;
  analyzeBtn.classList.remove('opacity-50', 'cursor-not-allowed');
}

// 1-Click Curated Benchmark Samples
async function initSampleAudios() {
  const container = document.getElementById('samplesGrid');
  try {
    const res = await fetch('/api/sample-audios');
    const data = await res.json();
    state.demoSamples = data.samples || [];

    container.innerHTML = '';
    state.demoSamples.forEach(sample => {
      const isFake = sample.expected === 'AI_SYNTHETIC';
      const card = document.createElement('div');
      card.className = `p-5 rounded-2xl border ${isFake ? 'border-red-500/30 bg-red-950/20 hover:border-red-400' : 'border-emerald-500/30 bg-emerald-950/20 hover:border-emerald-400'} transition-all cursor-pointer flex flex-col justify-between group shadow-lg`;
      card.innerHTML = `
        <div>
          <div class="flex items-center justify-between mb-3">
            <span class="px-2.5 py-0.5 text-xs font-mono font-semibold rounded-md ${isFake ? 'bg-red-500/20 text-red-300 border border-red-500/40' : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'}">${sample.badge}</span>
            <span class="text-xs text-slate-400 font-mono">${sample.category}</span>
          </div>
          <h4 class="font-bold text-slate-100 group-hover:text-cyan-300 text-sm mb-1.5 font-syne">${sample.title}</h4>
          <p class="text-xs text-slate-400 mb-4">${sample.expected === 'AI_SYNTHETIC' ? '🚨 Expected: Deepfake / Voice-Clone' : '✅ Expected: Authentic Human Voice'}</p>
        </div>
        <button class="w-full py-2.5 px-3.5 rounded-xl bg-slate-900/90 hover:bg-cyan-600 text-slate-200 hover:text-white text-xs font-semibold transition flex items-center justify-center gap-2 border border-slate-700 shadow-sm">
          <svg class="w-4 h-4 text-cyan-400 group-hover:text-white" fill="currentColor" viewBox="0 0 20 20"><path d="M6.3 2.841A1.5 1.5 0 004 4.11v11.78a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z"></path></svg>
          Load & Test Sample
        </button>
      `;

      card.querySelector('button').addEventListener('click', async (e) => {
        e.stopPropagation();
        await loadAndTestSample(sample);
      });
      container.appendChild(card);
    });
  } catch (err) {
    container.innerHTML = `<p class="text-slate-500 text-xs col-span-full">Sample audios are being generated...</p>`;
  }
}

async function loadAndTestSample(sample) {
  try {
    const res = await fetch(sample.url);
    const blob = await res.blob();
    const file = new File([blob], sample.filename, { type: 'audio/wav' });
    handleFileSelected(file);
    
    // Switch to upload tab and run scan immediately
    const uploadTabBtn = document.querySelector('.tab-btn[data-tab="tab-upload"]');
    if (uploadTabBtn) uploadTabBtn.click();
    runForensicAnalysis(file);
  } catch (err) {
    alert("Could not load sample audio: " + err.message);
  }
}

// Run Full Forensic Analysis
async function runForensicAnalysis(fileOrBlob) {
  const analyzeBtn = document.getElementById('btnAnalyze');
  const originalText = analyzeBtn.innerHTML;
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = `
    <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
    </svg>
    ${i18n[state.language].btnAnalyzing}
  `;

  const formData = new FormData();
  formData.append('file', fileOrBlob, fileOrBlob.name || 'uploaded_sample.wav');
  formData.append('threshold_low', state.thresholdLow);
  formData.append('threshold_high', state.thresholdHigh);

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const errDetail = await res.text();
      throw new Error(`Server returned status ${res.status}: ${errDetail}`);
    }
    const result = await res.json();
    state.lastAnalysis = result;
    renderAnalysisResults(result);
  } catch (err) {
    alert("Forensic Analysis Error: " + err.message);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = originalText;
  }
}

// Render Results to UI
function renderAnalysisResults(data) {
  const analysis = data.analysis;
  const resultsCard = document.getElementById('resultsCard');
  resultsCard.classList.remove('hidden');

  // Update Latency Badge
  const latencyBadge = document.getElementById('inferenceLatencyBadge');
  if (latencyBadge && data.latency_ms) {
    latencyBadge.textContent = `⚡ ${data.latency_ms}ms`;
  }

  // 1. Update Gauge & Risk Score
  const riskScore = analysis.risk_score;
  const riskScoreEl = document.getElementById('riskScoreValue');
  const riskProbEl = document.getElementById('syntheticProbValue');
  const confidenceEl = document.getElementById('confidenceValue');
  const riskBadgeEl = document.getElementById('riskLevelBadge');
  const verdictTextEl = document.getElementById('verdictText');

  riskScoreEl.textContent = `${riskScore}%`;
  riskProbEl.textContent = `${(analysis.synthetic_probability * 100).toFixed(1)}%`;
  confidenceEl.textContent = `${analysis.confidence_score}%`;

  verdictTextEl.textContent = analysis.verdict[state.language] || analysis.verdict.en;
  riskBadgeEl.textContent = `${analysis.risk_level} RISK`;

  const circle = document.getElementById('gaugeCircleProgress');
  const radius = circle.r.baseVal.value;
  const circumference = 2 * Math.PI * radius;
  circle.style.strokeDasharray = `${circumference} ${circumference}`;
  const offset = circumference - (riskScore / 100) * circumference;
  circle.style.strokeDashoffset = offset;
  circle.style.stroke = analysis.risk_color;

  if (analysis.risk_level === 'HIGH') {
    riskBadgeEl.className = 'px-4 py-1.5 rounded-full text-xs font-bold tracking-wider uppercase bg-red-500/20 text-red-400 border border-red-500/50 pulse-danger';
    riskScoreEl.className = 'text-5xl font-extrabold mono-font text-red-500 glow-text-crimson';
  } else if (analysis.risk_level === 'MEDIUM') {
    riskBadgeEl.className = 'px-4 py-1.5 rounded-full text-xs font-bold tracking-wider uppercase bg-amber-500/20 text-amber-400 border border-amber-500/50';
    riskScoreEl.className = 'text-5xl font-extrabold mono-font text-amber-400';
  } else {
    riskBadgeEl.className = 'px-4 py-1.5 rounded-full text-xs font-bold tracking-wider uppercase bg-emerald-500/20 text-emerald-400 border border-emerald-500/50 pulse-safe';
    riskScoreEl.className = 'text-5xl font-extrabold mono-font text-emerald-400 glow-text-emerald';
  }

  // 2. Actionable Advisory Banner
  const advisoryContainer = document.getElementById('advisoryBanner');
  const adv = analysis.advisory;
  const advTitle = state.language === 'hi' && adv.title_hi ? adv.title_hi : adv.title;
  const advRec = state.language === 'hi' && adv.recommendation_hi ? adv.recommendation_hi : adv.recommendation;
  const advAction = state.language === 'hi' && adv.action_hi ? adv.action_hi : adv.action;

  advisoryContainer.className = `p-5 rounded-2xl border mb-6 shadow-xl ${analysis.risk_level === 'HIGH' ? 'bg-red-950/50 border-red-500/60 text-red-100' : (analysis.risk_level === 'MEDIUM' ? 'bg-amber-950/50 border-amber-500/60 text-amber-100' : 'bg-emerald-950/50 border-emerald-500/60 text-emerald-100')}`;
  advisoryContainer.innerHTML = `
    <div class="flex items-start gap-3.5">
      <div class="p-2.5 rounded-xl ${analysis.risk_level === 'HIGH' ? 'bg-red-500/20 text-red-400 border border-red-500/40' : (analysis.risk_level === 'MEDIUM' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40' : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40')}">
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
      </div>
      <div>
        <h4 class="font-bold text-base mb-1 tracking-wide font-syne">${advTitle}</h4>
        <p class="text-xs mb-1.5 opacity-90 leading-relaxed">${advRec}</p>
        <p class="text-xs font-mono font-semibold underline opacity-100">Recommended Action: ${advAction}</p>
      </div>
    </div>
  `;

  // 3. Render Forensic Indicators
  const indicatorsContainer = document.getElementById('indicatorsList');
  indicatorsContainer.innerHTML = '';
  analysis.indicators.forEach(ind => {
    const isHigh = ind.severity === 'HIGH';
    const isMed = ind.severity === 'MEDIUM';
    const colorClass = isHigh ? 'text-red-400 bg-red-500/15 border-red-500/40' : (isMed ? 'text-amber-400 bg-amber-500/15 border-amber-500/40' : 'text-emerald-400 bg-emerald-500/15 border-emerald-500/40');
    const barBg = isHigh ? 'bg-gradient-to-r from-red-600 to-rose-500' : (isMed ? 'bg-gradient-to-r from-amber-600 to-yellow-500' : 'bg-gradient-to-r from-emerald-600 to-teal-500');

    const row = document.createElement('div');
    row.className = 'p-3.5 bg-slate-950/70 rounded-xl border border-slate-800/90 flex flex-col gap-1.5';
    row.innerHTML = `
      <div class="flex items-center justify-between text-xs">
        <div class="flex items-center gap-2">
          <span class="font-bold text-slate-100 font-syne">${ind.name}</span>
          <span class="px-2 py-0.5 rounded text-[10px] font-mono font-semibold border ${colorClass}">${ind.severity} RISK</span>
        </div>
        <span class="font-mono font-bold text-slate-200">${ind.score}%</span>
      </div>
      <div class="w-full bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-800">
        <div class="${barBg} h-full rounded-full transition-all duration-700" style="width: ${ind.score}%"></div>
      </div>
      <p class="text-[11px] text-slate-400 leading-snug">${ind.description}</p>
    `;
    indicatorsContainer.appendChild(row);
  });

  // 4. Mel-Spectrogram Heatmap Display
  const specImg = document.getElementById('spectrogramImg');
  const specPlaceholder = document.getElementById('spectrogramPlaceholder');
  if (data.spectrogram_image) {
    specImg.src = data.spectrogram_image;
    specImg.classList.remove('hidden');
    if (specPlaceholder) specPlaceholder.classList.add('hidden');
  }

  // 5. Acoustic Physical Metrics
  const metrics = analysis.forensic_metrics;
  document.getElementById('metricF0Mean').textContent = `${metrics.f0_mean_hz} Hz`;
  document.getElementById('metricF0Std').textContent = `± ${metrics.f0_std_hz} Hz`;
  document.getElementById('metricVocoderRatio').textContent = `${metrics.vocoder_ratio}`;
  document.getElementById('metricVoicedRatio').textContent = `${metrics.voiced_ratio_pct}%`;
  document.getElementById('metricDuration').textContent = `${metrics.audio_duration_sec}s`;

  resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Pure In-Browser 16kHz PCM WAV Encoder
function encodeFloatToWav(samples, sampleRate = 16000) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM Format
  view.setUint16(22, 1, true); // 1 Mono Channel
  view.setUint32(24, sampleRate, true); // 16,000 Hz
  view.setUint32(28, sampleRate * 2, true); // Byte rate
  view.setUint16(32, 2, true); // Block align
  view.setUint16(34, 16, true); // 16 bits per sample
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  // Write 16-bit signed PCM audio samples
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([view], { type: 'audio/wav' });
}

// Live Microphone Streaming Engine
const btnToggleLive = document.getElementById('btnToggleLive');
btnToggleLive.addEventListener('click', toggleLiveStreaming);

async function toggleLiveStreaming() {
  if (!state.isRecording) {
    await startLiveStreaming();
  } else {
    stopLiveStreaming();
  }
}

async function startLiveStreaming() {
  try {
    state.micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
      video: false,
    });

    state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = state.audioContext.createMediaStreamSource(state.micStream);

    // Audio Visualizer Analyser
    state.analyser = state.audioContext.createAnalyser();
    state.analyser.fftSize = 64;
    source.connect(state.analyser);

    // Buffer for collecting 16kHz audio samples
    state.pcmBuffer = [];
    state.isRecording = true;
    state.streamChunkIndex = 0;
    state.riskTrajectory = [];

    // Script Processor for raw audio capture
    state.scriptProcessor = state.audioContext.createScriptProcessor(4096, 1, 1);
    source.connect(state.scriptProcessor);
    state.scriptProcessor.connect(state.audioContext.destination);

    const inputSampleRate = state.audioContext.sampleRate;
    const downsampleRatio = inputSampleRate / state.targetSampleRate;

    state.scriptProcessor.onaudioprocess = (e) => {
      if (!state.isRecording) return;
      const inputData = e.inputBuffer.getChannelData(0);
      
      // Resample to 16kHz
      for (let i = 0; i < inputData.length; i += downsampleRatio) {
        state.pcmBuffer.push(inputData[Math.floor(i)]);
      }

      // Keep max 5 seconds buffer
      if (state.pcmBuffer.length > state.targetSampleRate * 5) {
        state.pcmBuffer = state.pcmBuffer.slice(-state.targetSampleRate * 5);
      }
    };

    // UI Updates
    btnToggleLive.innerHTML = `
      <span class="relative flex h-3 w-3 mr-2">
        <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
        <span class="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
      </span>
      ${i18n[state.language].btnStopLive}
    `;
    btnToggleLive.className = 'w-full max-w-sm mx-auto py-4 px-6 rounded-2xl font-bold bg-red-600 hover:bg-red-700 text-white flex items-center justify-center transition shadow-xl shadow-red-950/60';
    document.getElementById('liveStatusText').textContent = i18n[state.language].liveStatusRecording;
    document.getElementById('liveVisualizerContainer').classList.remove('hidden');

    startVisualizer();

    // Trigger chunk analysis every 2.8 seconds
    state.chunkIntervalId = setInterval(processCurrentLiveChunk, 2800);

  } catch (err) {
    alert("Microphone permission or access error: " + err.message);
  }
}

function stopLiveStreaming() {
  state.isRecording = false;
  if (state.chunkIntervalId) clearInterval(state.chunkIntervalId);
  if (state.scriptProcessor) {
    state.scriptProcessor.disconnect();
    state.scriptProcessor = null;
  }
  if (state.micStream) {
    state.micStream.getTracks().forEach(t => t.stop());
    state.micStream = null;
  }
  if (state.audioContext) {
    state.audioContext.close();
    state.audioContext = null;
  }

  btnToggleLive.innerHTML = `
    <svg class="w-5 h-5 mr-2 text-red-400" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z" clip-rule="evenodd"></path></svg>
    ${i18n[state.language].btnStartLive}
  `;
  btnToggleLive.className = 'btn-glow-cyan w-full max-w-sm mx-auto py-4 px-6 rounded-2xl font-bold flex items-center justify-center gap-2';
  document.getElementById('liveStatusText').textContent = i18n[state.language].liveStatusIdle;
}

async function processCurrentLiveChunk() {
  if (!state.isRecording || state.pcmBuffer.length < state.targetSampleRate * 1.5) return;

  const currentSamples = new Float32Array(state.pcmBuffer.slice(-state.targetSampleRate * 3));
  const wavBlob = encodeFloatToWav(currentSamples, state.targetSampleRate);

  state.streamChunkIndex++;
  const formData = new FormData();
  formData.append('file', wavBlob, `mic_chunk_${state.streamChunkIndex}.wav`);
  formData.append('chunk_index', state.streamChunkIndex);

  try {
    const res = await fetch('/api/analyze-chunk', {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) return;
    const data = await res.json();

    const timestampStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    state.riskTrajectory.push({
      time: timestampStr,
      score: data.risk_score,
      level: data.risk_level,
      color: data.risk_color,
      alert: data.alert,
    });

    if (state.riskTrajectory.length > 15) {
      state.riskTrajectory.shift();
    }

    updateStreamRiskUI(data);
    drawTrajectoryChart();
  } catch (err) {
    console.warn("Live chunk stream error:", err);
  }
}

function updateStreamRiskUI(data) {
  const streamRiskBadge = document.getElementById('streamCurrentRiskBadge');
  const streamRiskScore = document.getElementById('streamCurrentRiskScore');
  const streamAlertBanner = document.getElementById('streamLiveAlert');

  streamRiskScore.textContent = `${data.risk_score}%`;
  streamRiskBadge.textContent = `${data.risk_level} RISK`;
  streamRiskBadge.style.backgroundColor = `${data.risk_color}25`;
  streamRiskBadge.style.color = data.risk_color;
  streamRiskBadge.style.borderColor = `${data.risk_color}70`;

  if (data.alert) {
    streamAlertBanner.classList.remove('hidden');
    streamAlertBanner.innerHTML = `
      <div class="flex items-center gap-2.5">
        <span class="animate-ping h-3 w-3 rounded-full bg-red-400"></span>
        <span class="font-bold text-red-100 text-xs tracking-wider uppercase font-syne">${data.advisory_title}</span>
      </div>
    `;
  } else {
    streamAlertBanner.classList.add('hidden');
  }
}

// Live Audio Spectrum Canvas Visualizer
function startVisualizer() {
  const canvas = document.getElementById('liveAudioCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const bufferLength = state.analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function draw() {
    if (!state.isRecording) return;
    requestAnimationFrame(draw);
    state.analyser.getByteFrequencyData(dataArray);

    ctx.fillStyle = '#030712';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const barWidth = (canvas.width / bufferLength) * 1.8;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * canvas.height;
      const gradient = ctx.createLinearGradient(0, canvas.height, 0, 0);
      gradient.addColorStop(0, '#06b6d4');
      gradient.addColorStop(0.65, '#a855f7');
      gradient.addColorStop(1, '#ef4444');

      ctx.fillStyle = gradient;
      ctx.fillRect(x, canvas.height - barHeight, barWidth - 2.5, barHeight);
      x += barWidth;
    }
  }
  draw();
}

// Draw Real-Time Trajectory Chart on Canvas
function drawTrajectoryChart() {
  const canvas = document.getElementById('streamRiskCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Background Grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
  ctx.lineWidth = 1;
  for (let y = 0; y <= h; y += h / 4) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // 70% Danger Line
  const y70 = h - (70 / 100) * (h - 24) - 12;
  ctx.strokeStyle = 'rgba(239, 68, 68, 0.45)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, y70);
  ctx.lineTo(w, y70);
  ctx.stroke();
  ctx.setLineDash([]);

  if (state.riskTrajectory.length < 2) return;

  const points = state.riskTrajectory.map((pt, idx) => {
    const x = (idx / (state.riskTrajectory.length - 1)) * (w - 40) + 20;
    const y = h - (pt.score / 100) * (h - 30) - 15;
    return { x, y, score: pt.score, color: pt.color };
  });

  // Gradient Area Fill
  const gradient = ctx.createLinearGradient(0, 0, 0, h);
  gradient.addColorStop(0, 'rgba(6, 182, 212, 0.3)');
  gradient.addColorStop(1, 'rgba(6, 182, 212, 0.0)');
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.moveTo(points[0].x, h);
  points.forEach(p => ctx.lineTo(p.x, p.y));
  ctx.lineTo(points[points.length - 1].x, h);
  ctx.closePath();
  ctx.fill();

  // Line Curve
  ctx.strokeStyle = '#06b6d4';
  ctx.lineWidth = 2.8;
  ctx.beginPath();
  points.forEach((p, i) => {
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.stroke();

  // Node Points
  points.forEach(p => {
    ctx.fillStyle = p.color || '#06b6d4';
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.2;
    ctx.stroke();
  });
}

function drawEmptyRiskChart() {
  const canvas = document.getElementById('streamRiskCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#030712';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Threshold Adjustment Modal
function initThresholdModal() {
  const modal = document.getElementById('thresholdModal');
  const openBtn = document.getElementById('btnOpenThresholds');
  const closeBtn = document.getElementById('btnCloseThresholds');
  const saveBtn = document.getElementById('btnSaveThresholds');
  const sliderLow = document.getElementById('sliderLowThreshold');
  const sliderHigh = document.getElementById('sliderHighThreshold');
  const valLow = document.getElementById('valLowThreshold');
  const valHigh = document.getElementById('valHighThreshold');

  openBtn.addEventListener('click', () => {
    sliderLow.value = state.thresholdLow;
    sliderHigh.value = state.thresholdHigh;
    valLow.textContent = `${state.thresholdLow}%`;
    valHigh.textContent = `${state.thresholdHigh}%`;
    modal.classList.remove('hidden');
  });

  closeBtn.addEventListener('click', () => modal.classList.add('hidden'));

  sliderLow.addEventListener('input', (e) => {
    valLow.textContent = `${e.target.value}%`;
  });
  sliderHigh.addEventListener('input', (e) => {
    valHigh.textContent = `${e.target.value}%`;
  });

  saveBtn.addEventListener('click', () => {
    state.thresholdLow = parseFloat(sliderLow.value);
    state.thresholdHigh = parseFloat(sliderHigh.value);
    modal.classList.add('hidden');
    if (state.currentFile) {
      runForensicAnalysis(state.currentFile);
    }
  });
}

// Language Switcher
function initLanguageSwitcher() {
  const langSelect = document.getElementById('languageSelector');
  langSelect.addEventListener('change', (e) => {
    state.language = e.target.value;
    updateLanguageTexts();
    if (state.lastAnalysis) {
      renderAnalysisResults(state.lastAnalysis);
    }
  });
}

function updateLanguageTexts() {
  const dict = i18n[state.language] || i18n.en;
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.dataset.i18n;
    if (dict[key]) el.textContent = dict[key];
  });
}

// Export Audit Report
function initExportReport() {
  const btnExport = document.getElementById('btnExportReport');
  btnExport.addEventListener('click', () => {
    if (!state.lastAnalysis) {
      alert("Please run an audio analysis first before exporting the audit report.");
      return;
    }
    const reportData = {
      project: "SIH26104 - VoiceGuard AI Forensic Audit",
      generated_at: new Date().toISOString(),
      audio_filename: state.lastAnalysis.filename,
      audio_duration_seconds: state.lastAnalysis.duration_seconds,
      forensic_results: state.lastAnalysis.analysis,
      models_diagnostics: state.lastAnalysis.models_raw,
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `VoiceGuard_Forensic_Report_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });
}
