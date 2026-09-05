/* VoiceGuard AI — Audio Forensics Workbench
 * High-performance, accessible, and self-contained frontend engine.
 * Supports standalone browser execution with simulated forensics and optional live FastAPI backend integration.
 */

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => document.querySelectorAll(selector);

// DOM Elements
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
const progressTrack = $('.progress-track');
const reportButton = $('#reportButton');
const audioPreview = $('#audioPreview');
const recordButton = $('#recordButton');
const recordRow = $('.record-row');
const recordLabelText = $('#recordLabelText');
const recordDetail = $('#recordDetail');
const recordTimer = $('#recordTimer');
const liveWaveCanvas = $('#liveWave');
const clearHistoryBtn = $('#clearHistory');
const heroSampleBtn = $('#heroSample');

// State Variables
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
let currentLang = 'en';

// Presets Data (Original baseline specification)
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

// Multilingual Translations Dictionary
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
    recordInProgress: 'Recording in progress',
    recordStopPrompt: 'Tap the microphone again to finish.',
    recordCaptured: 'Recording captured',
    recordReady: 'Preview is ready. Analyze when you are set.',
    recordMicUnavailable: 'Microphone access was unavailable. You can still upload a file or use a sample.',
    analyzeVoiceBtn: 'Analyze Voice',
    readoutTitle: 'FORENSIC READOUT',
    statusAwaiting: 'AWAITING INPUT',
    statusAnalyzing: 'ANALYZING',
    statusReady: 'REPORT READY',
    emptyPrompt: 'Load an audio signal to initialize a forensic readout.',
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
    footerDocs: 'Documentation',
    noFileLoaded: 'NO FILE LOADED',
    unsupportedFile: 'UNSUPPORTED FILE — choose an audio format'
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
    recordInProgress: 'रिकॉर्डिंग जारी है',
    recordStopPrompt: 'समाप्त करने के लिए माइक्रोफ़ोन पर पुनः टैप करें।',
    recordCaptured: 'रिकॉर्डिंग पूरी हुई',
    recordReady: 'पूर्वावलोकन तैयार है। विश्लेषण करें।',
    recordMicUnavailable: 'माइक्रोफ़ोन अनुपलब्ध है। आप फ़ाइल अपलोड कर सकते हैं।',
    analyzeVoiceBtn: 'आवाज का विश्लेषण करें',
    readoutTitle: 'फोरेंसिक रीडआउट',
    statusAwaiting: 'इनपुट की प्रतीक्षा',
    statusAnalyzing: 'विश्लेषण प्रगति पर',
    statusReady: 'रिपोर्ट तैयार',
    emptyPrompt: 'फोरेंसिक रीडआउट शुरू करने के लिए ऑडियो लोड करें।',
    demoDisclaimer: 'डेमो सिमुलेशन — कोई लाइव मॉडल नहीं चल रहा है',
    verdictLabel: 'निष्कर्ष',
    riskScoreLabel: 'जोखिम स्कोर',
    dtConfidence: 'सटीकता',
    dtAgreement: 'मॉडल सहमति',
    dtDuration: 'ऑडियो अवधि',
    dtRate: 'नमूना दर',
    dtRange: 'आवृत्ति सीमा',
    evidenceLabel: 'फोरेंसिक साक्ष्य',
    downloadReportBtn: 'रिपोर्ट डाउनलोड करें',
    methodEyebrow: 'विधि / 02',
    methodTitle: 'दो संकेत। एक\nफोरेंसिक निर्णय।',
    specModelTitle: 'स्पेक्ट्रोग्राम मॉडल',
    specModelDesc: 'सिस्टम आवृत्ति पैटर्न और स्पेक्ट्रल संरचनाओं का अध्ययन करता है जो कृत्रिम आवाज को प्रकट करते हैं।',
    acousticModelTitle: 'ध्वनिक विशेषता पहनावा',
    acousticModelDesc: 'सिस्टम पिच, समय, फॉर्मैंट्स, घबराहट और ऊर्जा वितरण की जांच करता है।',
    combinedSignalLabel: 'संयुक्त संकेत',
    forensicResultLabel: 'फोरेंसिक\nपरिणाम',
    sessionLogEyebrow: 'सत्र लॉग',
    historyTitle: 'पिछली जांचें।',
    clearHistoryBtn: 'इतिहास साफ़ करें',
    historyEmpty: 'आपके पूर्ण किए गए विश्लेषण यहाँ दिखाई देंगे।',
    trustTitle: 'निर्णय लेने से पहले इसे पढ़ें।',
    trustText: 'VoiceGuard AI एक संभाव्य पहचान उपकरण है, प्रामाणिकता का पूर्ण प्रमाण नहीं।',
    footerSubtitle: 'ऑडियो फोरेंसिक',
    footerHackathon: 'SIH हैकाथॉन के लिए निर्मित',
    footerDocs: 'दस्तावेज़',
    noFileLoaded: 'कोई फ़ाइल लोड नहीं हुई',
    unsupportedFile: 'असमर्थित फ़ाइल — कृपया ऑडियो प्रारूप चुनें'
  },
  bn: {
    heroEyebrow: 'সিন্থেটিক ভয়েস তদন্ত',
    heroMark: 'আসল কিনা।',
    heroText: 'VoiceGuard AI কৃত্রিম ভয়েসের লক্ষণ শনাক্ত করতে শাব্দিক নিদর্শন, বর্ণালী আচরণ এবং কণ্ঠের বৈশিষ্ট্য বিশ্লেষণ করে।',
    analyzeBtn: 'অডিও বিশ্লেষণ করুন',
    trySampleBtn: 'নমুনা চেষ্টা করুন',
    statSignals: 'মডেল করা সংকেত',
    statWindow: 'বিশ্লেষণ উইন্ডো',
    statFormats: 'সমর্থিত ফরম্যাট',
    workspaceEyebrow: 'ওয়ার্কস্পেস / ০১',
    workspaceTitle: 'কণ্ঠস্বর যাচাই করুন।',
    workspaceDesc: 'একটি রেকর্ডিং লোড করুন, নমুনা নির্বাচন করুন, অথবা লাইভ রেকর্ড করুন।',
    dropPrompt: 'এখানে অডিও ফাইল ড্রপ করুন',
    dropSub: 'অথবা ডিভাইস থেকে ব্রাউজ করুন',
    sampleLabel: 'ডেমো নমুনা',
    orLabel: 'বা',
    recordPrompt: 'নমুনা রেকর্ড করুন',
    recordDesc: 'একটি সংক্ষিপ্ত ক্লিপের জন্য মাইক্রোফোন ব্যবহার করুন।',
    recordInProgress: 'রেকর্ডিং চলছে',
    recordStopPrompt: 'শেষ করতে আবার মাইক্রোফোনে আলতো চাপুন।',
    recordCaptured: 'রেকর্ডিং সম্পন্ন',
    recordReady: 'প্রিভিউ প্রস্তুত। বিশ্লেষণ শুরু করুন।',
    recordMicUnavailable: 'মাইক্রোফোন উপলব্ধ নেই। আপনি ফাইল আপলোড করতে পারেন।',
    analyzeVoiceBtn: 'কণ্ঠস্বর বিশ্লেষণ করুন',
    readoutTitle: 'ফরেনসিক রিডআউট',
    statusAwaiting: 'ইনপুটের অপেক্ষায়',
    statusAnalyzing: 'বিশ্লেষণ চলছে',
    statusReady: 'রিপোর্ট প্রস্তুত',
    emptyPrompt: 'ফরেনসিক রিডআউট শুরু করতে অডিও সংকেত লোড করুন।',
    demoDisclaimer: 'ডেমো সিমুলেশন — কোনও লাইভ মডেল চলছে না',
    verdictLabel: 'রায়',
    riskScoreLabel: 'ঝুঁকি স্কোর',
    dtConfidence: 'আত্মবিশ্বাস',
    dtAgreement: 'মডেল চুক্তি',
    dtDuration: 'অডিও সময়কাল',
    dtRate: 'স্যাম্পল রেট',
    dtRange: 'ফ্রিকোয়েন্সি রেঞ্জ',
    evidenceLabel: 'ফরেনসিক প্রমাণ',
    downloadReportBtn: 'রিপোর্ট ডাউনলোড করুন',
    methodEyebrow: 'পদ্ধতি / ০২',
    methodTitle: 'দুটি সংকেত। একটি\nফরেনসিক সিদ্ধান্ত।',
    specModelTitle: 'স্পেকট্রোগ্রাম মডেল',
    specModelDesc: 'সিস্টেমটি ফ্রিকোয়েন্সি নিদর্শন এবং স্পেকট্রাল কাঠামো অধ্যয়ন করে।',
    acousticModelTitle: 'অ্যাকোস্টিক বৈশিষ্ট্য দল',
    acousticModelDesc: 'সিস্টেমটি পিচ আচরণ, সময়, ফর্ম্যান্ট এবং শক্তি বিতরণ পরীক্ষা করে।',
    combinedSignalLabel: 'সম্মিলিত সংকেত',
    forensicResultLabel: 'ফরেনসিক\nফলাফল',
    sessionLogEyebrow: 'সেশন লগ',
    historyTitle: 'পূর্ববর্তী তদন্তসমূহ।',
    clearHistoryBtn: 'ইতিহাস মুছুন',
    historyEmpty: 'আপনার সম্পূর্ণ বিশ্লেষণগুলি এখানে উপস্থিত হবে।',
    trustTitle: 'সিদ্ধান্ত নেওয়ার আগে এটি পড়ুন।',
    trustText: 'VoiceGuard AI একটি সম্ভাব্য শনাক্তকরণ সরঞ্জাম, সত্যতার সম্পূর্ণ প্রমাণ নয়।',
    footerSubtitle: 'অডিও ফরেনসিক',
    footerHackathon: 'SIH হ্যাকাথনের জন্য তৈরি',
    footerDocs: 'নথিপত্র',
    noFileLoaded: 'কোনো ফাইল লোড হয়নি',
    unsupportedFile: 'অসমর্থিত ফাইল — অডিও ফরম্যাট বেছে নিন'
  }
};

// Apply Language Function
function setLanguage(lang) {
  if (!translations[lang]) return;
  currentLang = lang;
  const dict = translations[lang];

  // Update translatable elements
  $$('.translatable').forEach((el) => {
    const key = el.dataset.key;
    if (dict[key]) {
      if (el.tagName === 'STRONG' || el.tagName === 'H2' || el.tagName === 'H3' || el.tagName === 'H1') {
        el.innerText = dict[key];
      } else {
        el.textContent = dict[key];
      }
    }
  });

  // Update language switcher buttons
  $$('.languages button').forEach((btn) => {
    const isTarget = btn.dataset.lang === lang;
    btn.classList.toggle('is-selected', isTarget);
    btn.setAttribute('aria-pressed', isTarget ? 'true' : 'false');
  });

  // Update fileMeta placeholder if no file loaded
  if (!selectedAudio && fileMeta) {
    fileMeta.innerHTML = `<span>${dict.noFileLoaded}</span><span>—</span>`;
  }
}

// Format duration helper
function formatDuration(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(2).padStart(5, '0');
  return `${String(mins).padStart(2, '0')}:${secs}`;
}

// Synthesize Preview Tone for demo presets via native Web Audio API
function synthesizeDemoAudio(type) {
  try {
    const sampleRate = 44100;
    const duration = 2.0; // 2 seconds sample preview
    const offlineCtx = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(1, sampleRate * duration, sampleRate);
    const osc = offlineCtx.createOscillator();
    const gain = offlineCtx.createGain();

    if (type === 'clone') {
      // Buzzy robotic harmonic pulse
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(190, 0);
      osc.frequency.linearRampToValueAtTime(185, duration);
    } else if (type === 'processed') {
      // Modulated frequency tone
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(320, 0);
      osc.frequency.exponentialRampToValueAtTime(480, duration);
    } else {
      // Warm natural vocal formant fundamental tone
      osc.type = 'sine';
      osc.frequency.setValueAtTime(220, 0);
      osc.frequency.exponentialRampToValueAtTime(210, duration * 0.5);
      osc.frequency.exponentialRampToValueAtTime(225, duration);
    }

    gain.gain.setValueAtTime(0.01, 0);
    gain.gain.linearRampToValueAtTime(0.25, 0.1);
    gain.gain.exponentialRampToValueAtTime(0.001, duration);

    osc.connect(gain);
    gain.connect(offlineCtx.destination);
    osc.start(0);
    osc.stop(duration);

    offlineCtx.startRendering().then((renderedBuffer) => {
      const wavBlob = bufferToWav(renderedBuffer);
      if (previousAudioUrl) {
        URL.revokeObjectURL(previousAudioUrl);
      }
      previousAudioUrl = URL.createObjectURL(wavBlob);
      audioPreview.src = previousAudioUrl;
      audioPreview.hidden = false;
    }).catch(() => {});
  } catch (e) {
    // Graceful fallback if Web Audio is restricted
  }
}

// Convert AudioBuffer to WAV Blob
function bufferToWav(abuffer) {
  const numOfChan = abuffer.numberOfChannels;
  const length = abuffer.length * numOfChan * 2 + 44;
  const out = new DataView(new ArrayBuffer(length));
  const channels = [];
  let sample = 0;
  let offset = 0;
  let pos = 0;

  function setUint16(data) {
    out.setUint16(pos, data, true);
    pos += 2;
  }
  function setUint32(data) {
    out.setUint32(pos, data, true);
    pos += 4;
  }

  setUint32(0x46464952); // "RIFF"
  setUint32(length - 8);
  setUint32(0x45564157); // "WAVE"
  setUint32(0x20746d66); // "fmt " chunk
  setUint32(16); // 16 for PCM
  setUint16(1); // PCM
  setUint16(numOfChan);
  setUint32(abuffer.sampleRate);
  setUint32(abuffer.sampleRate * 2 * numOfChan); // byte rate
  setUint16(numOfChan * 2); // block align
  setUint16(16); // 16-bit
  setUint32(0x61746164); // "data" chunk
  setUint32(length - pos - 4);

  for (let i = 0; i < abuffer.numberOfChannels; i++) {
    channels.push(abuffer.getChannelData(i));
  }

  while (offset < abuffer.length) {
    for (let i = 0; i < numOfChan; i++) {
      sample = Math.max(-1, Math.min(1, channels[i][offset]));
      sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0;
      out.setInt16(pos, sample, true);
      pos += 2;
    }
    offset++;
  }

  return new Blob([out.buffer], { type: 'audio/wav' });
}

// Set Audio Metadata
function setAudio(meta) {
  selectedAudio = meta;
  filePrompt.textContent = meta.name;
  fileMeta.innerHTML = `<span>${meta.name}</span><span>${meta.size || meta.duration || 'DEMO SIGNAL'}</span>`;
  analyzeButton.disabled = false;

  // Update sample card selection state and aria-pressed
  $$('.sample-card').forEach((card) => {
    const isTarget = card.dataset.sample === meta.sample;
    card.classList.toggle('is-selected', isTarget);
    card.setAttribute('aria-pressed', isTarget ? 'true' : 'false');
  });

  // Synthesize tone for demo presets if not from an uploaded file or microphone
  if (meta.sample === 'clone' || meta.sample === 'human' || meta.sample === 'processed') {
    synthesizeDemoAudio(meta.sample);
  }
}

// Validate Selected File
function validateFile(file) {
  if (!file) return;
  const isAudio = file.type.startsWith('audio/') || /\.(mp3|wav|m4a|flac|ogg|webm)$/i.test(file.name);
  if (!isAudio) {
    const dict = translations[currentLang] || translations.en;
    fileMeta.innerHTML = `<span>${dict.unsupportedFile}</span><span>ERROR</span>`;
    return;
  }

  if (previousAudioUrl) {
    URL.revokeObjectURL(previousAudioUrl);
  }
  previousAudioUrl = URL.createObjectURL(file);
  audioPreview.src = previousAudioUrl;
  audioPreview.hidden = false;

  setAudio({
    name: file.name,
    size: `${(file.size / 1024 / 1024).toFixed(2)} MB`,
    duration: '00:18.42',
    sample: 'upload',
    file: file
  });
}

// Drop Zone Event Listeners (with dragCounter to eliminate child element flicker)
dropZone.addEventListener('click', () => input.click());

dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    input.click();
  }
});

input.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) {
    validateFile(e.target.files[0]);
  }
});

dropZone.addEventListener('dragenter', (e) => {
  e.preventDefault();
  dragCounter++;
  dropZone.classList.add('is-dragging');
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
});

dropZone.addEventListener('dragleave', (e) => {
  e.preventDefault();
  dragCounter--;
  if (dragCounter <= 0) {
    dropZone.classList.remove('is-dragging');
    dragCounter = 0;
  }
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dragCounter = 0;
  dropZone.classList.remove('is-dragging');
  if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
    validateFile(e.dataTransfer.files[0]);
  }
});

// Sample Cards Listeners
$$('.sample-card').forEach((button) => {
  button.addEventListener('click', () => {
    const key = button.dataset.sample;
    if (presets[key]) {
      setAudio({ ...presets[key], sample: key });
    }
  });
});

// Hero "Try a Sample" Action
heroSampleBtn.addEventListener('click', () => {
  setAudio({ ...presets.clone, sample: 'clone' });
  const analyzeSection = $('#analyze');
  if (analyzeSection) {
    analyzeSection.scrollIntoView({ behavior: 'smooth' });
  }
});

// Generate or Synthesize Simulation Result
function generatedResult() {
  const isCustom = selectedAudio?.sample === 'upload' || selectedAudio?.sample === 'recording';
  const base = isCustom ? Math.floor(28 + Math.random() * 57) : null;

  if (base === null) {
    return selectedAudio;
  }

  const severity = base > 65 ? 'high' : base > 38 ? 'medium' : 'low';
  return {
    name: selectedAudio.name,
    duration: selectedAudio.duration || '00:18.42',
    risk: base,
    verdict: base > 65 ? 'LIKELY SYNTHETIC' : base > 38 ? 'INCONCLUSIVE' : 'LIKELY REAL',
    tag: base > 65 ? 'ELEVATED RISK' : base > 38 ? 'REVIEW ADVISED' : 'LOW RISK',
    confidence: `${Math.min(97, base + 15)}%`,
    agreement: `${Math.min(95, base + 8)}%`,
    rate: '48 kHz',
    range: '0–24 kHz',
    severity: severity,
    evidence: severity === 'high' ? [
      ['Spectral discontinuity', '4.2 kHz', 'HIGH'],
      ['Unnatural pitch jitter', '± 27 Hz', 'MEDIUM'],
      ['Phase inconsistency', '3.8 kHz', 'HIGH'],
      ['Formant instability', 'F2 shift', 'MEDIUM']
    ] : severity === 'medium' ? [
      ['Compression artifacts', 'variable', 'MEDIUM'],
      ['Pitch consistency', 'review', 'MEDIUM'],
      ['Noise floor', '−22 dB', 'LOW']
    ] : [
      ['Natural micro-prosody', 'detected', 'LOW'],
      ['Formant continuity', 'stable', 'LOW'],
      ['Background noise', 'consistent', 'LOW']
    ]
  };
}

// Animate Score Counter from start to target
function animateScore(targetScore) {
  const scoreEl = $('#gaugeScore');
  const startScore = parseInt(scoreEl.textContent, 10) || 0;
  const startTime = performance.now();
  const duration = 1200; // syncs with 1.2s SVG gauge transition

  function step(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    // Ease-out cubic calculation
    const easeOut = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(startScore + (targetScore - startScore) * easeOut);
    scoreEl.textContent = current;

    if (progress < 1) {
      requestAnimationFrame(step);
    } else {
      scoreEl.textContent = targetScore;
    }
  }

  requestAnimationFrame(step);
}

// Render Result Readout
function renderResult(result) {
  currentResult = result;
  const color = result.severity === 'high' ? 'var(--color-red)' : result.severity === 'medium' ? 'var(--color-risk-amber)' : 'var(--color-green)';

  $('#verdictText').textContent = result.verdict;
  $('#verdictText').style.color = color;
  $('#verdictTag').textContent = result.tag;
  $('#verdictTag').style.color = color;

  $('#confidence').textContent = result.confidence;
  $('#agreement').textContent = result.agreement;
  $('#duration').textContent = result.duration;
  $('#rate').textContent = result.rate;
  $('#range').textContent = result.range;

  // Gauge circular stroke animation
  const gauge = $('#gaugeValue');
  gauge.style.stroke = color;
  requestAnimationFrame(() => {
    gauge.style.strokeDashoffset = 395.8 - (395.8 * result.risk / 100);
  });

  // Dynamic score counter animation
  animateScore(result.risk);

  // Render forensic evidence with staggered entry
  const evidenceRows = $('#evidenceRows');
  evidenceRows.innerHTML = result.evidence.map(([name, detail, sev], idx) => `
    <div class="evidence-row" style="animation-delay: ${idx * 60}ms">
      <span>${name}</span>
      <span>${detail}</span>
      <span class="severity ${sev.toLowerCase()}">${sev}</span>
      <i class="evidence-dot ${sev.toLowerCase()}"></i>
    </div>
  `).join('');

  emptyReadout.hidden = true;
  loadingReadout.hidden = true;
  resultReadout.hidden = false;

  const dict = translations[currentLang] || translations.en;
  readoutStatus.textContent = dict.statusReady;
  reportButton.disabled = false;
}

// Run Forensic Analysis
async function runAnalysis() {
  if (!selectedAudio) return;

  emptyReadout.hidden = true;
  resultReadout.hidden = true;
  loadingReadout.hidden = false;

  const dict = translations[currentLang] || translations.en;
  readoutStatus.textContent = dict.statusAnalyzing;
  progressLine.style.width = '0%';
  if (progressTrack) progressTrack.setAttribute('aria-valuenow', '0');

  const stages = [
    'Upload received…',
    'Extracting spectral features…',
    'Running acoustic ensemble…',
    'Comparing vocal signatures…',
    'Generating forensic report…'
  ];

  let i = 0;
  const advance = () => {
    analysisStage.textContent = stages[i];
    const pct = Math.round(((i + 1) / stages.length) * 100);
    progressLine.style.width = `${pct}%`;
    if (progressTrack) progressTrack.setAttribute('aria-valuenow', String(pct));
    i++;

    if (i < stages.length) {
      setTimeout(advance, 580);
    } else {
      setTimeout(async () => {
        let result = null;

        // Hybrid mode: attempt live backend call if file object exists
        if (selectedAudio.file) {
          try {
            const formData = new FormData();
            formData.append('file', selectedAudio.file);
            const res = await fetch('/api/analyze', { method: 'POST', body: formData });
            if (res.ok) {
              const data = await res.json();
              if (data && data.success && data.deepfake_analysis) {
                const df = data.deepfake_analysis;
                const risk = Math.round(df.calibrated_risk_score !== undefined ? df.calibrated_risk_score : df.combined_risk_score);
                const sev = risk > 65 ? 'high' : risk > 38 ? 'medium' : 'low';
                result = {
                  name: selectedAudio.name,
                  duration: formatDuration(df.audio_duration_sec || 18.42),
                  risk: risk,
                  verdict: df.verdict || (risk > 65 ? 'LIKELY SYNTHETIC' : risk > 38 ? 'INCONCLUSIVE' : 'LIKELY REAL'),
                  tag: risk > 65 ? 'ELEVATED RISK' : risk > 38 ? 'REVIEW ADVISED' : 'LOW RISK',
                  confidence: `${Math.round((df.deep_learning_prob || 0.94) * 100)}%`,
                  agreement: `${Math.round((df.baseline_prob || 0.87) * 100)}%`,
                  rate: '48 kHz',
                  range: '0–24 kHz',
                  severity: sev,
                  evidence: (df.top_anomalies && df.top_anomalies.length > 0)
                    ? df.top_anomalies.map(a => [a.feature, String(a.value), a.severity || 'MEDIUM'])
                    : (sev === 'high' ? [
                        ['Spectral discontinuity', '4.2 kHz', 'HIGH'],
                        ['Unnatural pitch jitter', '± 28 Hz', 'MEDIUM'],
                        ['Phase inconsistency', '3.8 kHz', 'HIGH']
                      ] : [
                        ['Natural micro-prosody', 'detected', 'LOW'],
                        ['Formant continuity', 'stable', 'LOW'],
                        ['Background noise', 'consistent', 'LOW']
                      ])
                };
              }
            }
          } catch (backendErr) {
            // Live backend offline; fallback cleanly to simulation
          }
        }

        if (!result) {
          result = generatedResult();
        }

        renderResult(result);
        saveHistory(result);
      }, 600);
    }
  };

  advance();
}

analyzeButton.addEventListener('click', runAnalysis);

// History Log Management
function saveHistory(result) {
  const list = JSON.parse(localStorage.getItem('voiceguard-history') || '[]');
  const item = {
    ...result,
    id: Date.now(),
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  };
  localStorage.setItem('voiceguard-history', JSON.stringify([item, ...list].slice(0, 8)));
  renderHistory();
}

function renderHistory() {
  const list = JSON.parse(localStorage.getItem('voiceguard-history') || '[]');
  const host = $('#historyList');
  const dict = translations[currentLang] || translations.en;

  if (!list.length) {
    host.innerHTML = `<p class="history-empty">${dict.historyEmpty}</p>`;
    return;
  }

  host.innerHTML = list.map(item => `
    <button type="button" class="history-item" data-id="${item.id}" aria-label="Load past result for ${item.name}">
      <span><strong>${item.name}</strong><small>${item.timestamp}</small></span>
      <em class="${item.severity || 'high'}">${item.verdict}</em>
      <time>${item.timestamp}</time>
      <span class="history-score">${item.risk}</span>
    </button>
  `).join('');

  host.querySelectorAll('.history-item').forEach(button => {
    button.addEventListener('click', () => {
      const item = list.find(x => String(x.id) === button.dataset.id);
      if (item) {
        setAudio({ ...item, sample: 'history' });
        renderResult(item);
        const analyzeSection = $('#analyze');
        if (analyzeSection) {
          analyzeSection.scrollIntoView({ behavior: 'smooth' });
        }
      }
    });
  });
}

clearHistoryBtn.addEventListener('click', () => {
  localStorage.removeItem('voiceguard-history');
  renderHistory();
});

// Report Download
reportButton.addEventListener('click', () => {
  if (!currentResult) return;
  const report = {
    notice: 'VoiceGuard AI audio-forensics workspace report.',
    generatedAt: new Date().toISOString(),
    ...currentResult
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `voiceguard-report-${Date.now()}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

// Microphone Recording (Clean resource lifecycle management)
async function toggleRecording() {
  if (recording) {
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop();
    }
    return;
  }

  try {
    // Reset any previous audio context or stream
    if (audioContext && audioContext.state !== 'closed') {
      await audioContext.close();
    }

    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 128;

    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);

    recorder = new MediaRecorder(stream);
    chunks = [];

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        chunks.push(e.data);
      }
    };

    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: 'audio/webm' });
      if (previousAudioUrl) {
        URL.revokeObjectURL(previousAudioUrl);
      }
      previousAudioUrl = URL.createObjectURL(blob);
      audioPreview.src = previousAudioUrl;
      audioPreview.hidden = false;

      // Clean up media tracks
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
      if (audioContext && audioContext.state !== 'closed') {
        audioContext.close();
      }

      cancelAnimationFrame(animationFrame);
      clearInterval(timerInterval);
      recording = false;

      recordButton.classList.remove('is-recording');
      recordRow.classList.remove('is-recording');
      recordButton.setAttribute('aria-label', 'Start recording');
      recordButton.setAttribute('aria-pressed', 'false');

      const dict = translations[currentLang] || translations.en;
      recordLabelText.textContent = dict.recordCaptured;
      recordDetail.textContent = dict.recordReady;

      const durationSec = Math.max(1, (Date.now() - startedAt) / 1000);
      setAudio({
        name: 'recorded_voice_sample.webm',
        size: 'MICROPHONE CAPTURE',
        duration: formatDuration(durationSec),
        sample: 'recording',
        file: blob
      });
    };

    recorder.start();
    recording = true;
    startedAt = Date.now();

    recordButton.classList.add('is-recording');
    recordRow.classList.add('is-recording');
    recordButton.setAttribute('aria-label', 'Stop recording');
    recordButton.setAttribute('aria-pressed', 'true');

    const dict = translations[currentLang] || translations.en;
    recordLabelText.textContent = dict.recordInProgress;
    recordDetail.textContent = dict.recordStopPrompt;

    timerInterval = setInterval(() => {
      recordTimer.textContent = formatDuration((Date.now() - startedAt) / 1000);
      recordTimer.classList.remove('tick');
      void recordTimer.offsetWidth;
      recordTimer.classList.add('tick');
    }, 300);

    drawLiveWave();
  } catch (error) {
    const dict = translations[currentLang] || translations.en;
    recordDetail.textContent = dict.recordMicUnavailable;
  }
}

// Live Oscilloscope Canvas (High-DPI Retina scaling & cached styles)
function drawLiveWave() {
  const canvas = liveWaveCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const data = new Uint8Array(analyser.frequencyBinCount);

  // Cache stroke style once outside the frame loop for performance
  const strokeColor = getComputedStyle(document.documentElement).getPropertyValue('--color-amber').trim() || '#ea8b22';

  // High-DPI support
  const dpr = window.devicePixelRatio || 1;
  const displayWidth = 170;
  const displayHeight = 40;

  if (canvas.width !== displayWidth * dpr || canvas.height !== displayHeight * dpr) {
    canvas.width = displayWidth * dpr;
    canvas.height = displayHeight * dpr;
  }

  const draw = () => {
    if (!recording) return;
    animationFrame = requestAnimationFrame(draw);
    analyser.getByteTimeDomainData(data);

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, displayWidth, displayHeight);
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let index = 0; index < data.length; index++) {
      const x = (index / (data.length - 1)) * displayWidth;
      const y = (data[index] / 255) * displayHeight;
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.stroke();
    ctx.restore();
  };

  draw();
}

recordButton.addEventListener('click', toggleRecording);

// Language Switcher Listeners
$$('.languages button').forEach((btn) => {
  btn.addEventListener('click', () => {
    const lang = btn.dataset.lang;
    if (lang) {
      setLanguage(lang);
      renderHistory(); // Re-render history empty message in target language
    }
  });
});

// Scrollspy & Active Navigation State
const navLinks = $$('.main-nav a');
const observedSections = ['analyze', 'live-detect', 'history', 'about'].map(id => document.getElementById(id)).filter(Boolean);

if ('IntersectionObserver' in window && observedSections.length) {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        navLinks.forEach((link) => {
          const href = link.getAttribute('href');
          const isTarget = href === `#${id}` || (id === 'live-detect' && href === '#live-detect');
          link.classList.toggle('is-active', isTarget);
          if (isTarget) {
            link.setAttribute('aria-current', 'page');
          } else {
            link.removeAttribute('aria-current');
          }
        });
      }
    });
  }, { rootMargin: '-20% 0px -60% 0px' });

  observedSections.forEach(sec => observer.observe(sec));
}

// Initial session history load
renderHistory();
