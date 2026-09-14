"use strict";
const $ = (id) => document.getElementById(id),
  reduced = matchMedia("(prefers-reduced-motion: reduce)");
const API_BASE = (window.VOICEGUARD_CONFIG?.API_URL || "").trim().replace(/\/+$/, "") ||
  ((location.protocol === "file:" ||
    (["localhost", "127.0.0.1", "[::1]"].includes(location.hostname) && location.port !== "8000"))
    ? "http://127.0.0.1:8000" : "");
function apiUrl(path) {
  const localPage = location.protocol === "file:" || ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
  if (!API_BASE && location.hostname.endsWith(".github.io"))
    throw Error("Backend URL is not configured. Set the API_URL repository variable and redeploy the frontend.");
  const base = new URL(API_BASE || location.origin);
  if (!localPage && (base.protocol !== "https:" || ["localhost", "127.0.0.1", "[::1]"].includes(base.hostname)))
    throw Error("Production API_URL must be the HTTPS address of the deployed backend.");
  if (!["http:", "https:"].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw Error("API_URL must be an HTTP(S) backend address without credentials, query or fragment.");
  return base.href.replace(/\/+$/, "") + path;
}
const stages = [
  "Reading audio",
  "Detecting speech",
  "Separating speakers",
  "Identifying language",
  "Extracting acoustic features",
  "Running trained models",
  "Cross-checking forensic signals",
  "Assessing confidence",
  "Generating forensic report",
];
const state = {
  file: null,
  url: null,
  result: null,
  job: null,
  busy: false,
  key: "",
  history: [],
  recording: false,
  stream: null,
  context: null,
  analyser: null,
  full: [],
  pending: [],
  samples: 0,
  liveBusy: false,
  recordToken: 0,
  segmentEnd: null,
  sessionJobs: new Set(),
  transcript: "",
  interimTranscript: "",
  speechRec: null,
  currentLang: "",
  speechLanguage: "en",
  activeVoiceFrames: 0,
};
const palette = [
    "#9fbadd",
    "#a7c6c2",
    "#bbb5d5",
    "#d2bc97",
    "#b3c5a5",
    "#caaeb6",
  ],
  languageNames = { en: "English", hi: "Hindi", bn: "Bengali" };

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}
function fmt(s) {
  s = Math.max(0, Number(s) || 0);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}
function riskText(r) {
  return r?.risk_score == null
    ? "Insufficient evidence"
    : `${Math.round(r.risk_score)} / 100 · ${r.risk_level}`;
}
function langText(l) {
  return (l?.label || "Unknown")
    .split(" + ")
    .map((c) => languageNames[c] || c)
    .join(" + ");
}
function notify(message) {
  const t = $("toast");
  if (!t) return;
  t.textContent = message;
  t.hidden = false;
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => (t.hidden = true), 4500);
}
function fail(message) {
  const err = $("inputError");
  if (!err) return;
  err.textContent = message;
  err.hidden = false;
}
function clearError() {
  const err = $("inputError");
  if (err) err.hidden = true;
}

async function api(path, options = {}) {
  const url = apiUrl(path);
  const controller = new AbortController();
  const signal = options.signal || controller.signal;
  const timeout = setTimeout(() => controller.abort(), 90000);
  let response;
  try {
    response = await fetch(url, {
      ...options, signal,
      headers: { ...(state.key ? { "X-API-Key": state.key } : {}), ...options.headers },
    });
    let data;
    try { data = await response.json(); }
    catch {
      throw Error(`Backend returned a non-JSON response (HTTP ${response.status}). Check API_URL points to FastAPI and that the backend deployment is healthy.`);
    }
    if (!response.ok)
      throw Error(typeof data.detail === "string" ? data.detail : `Backend request failed (HTTP ${response.status}).`);
    if (!data || typeof data !== "object" || Array.isArray(data))
      throw Error("Backend returned an invalid JSON payload. Check the deployed API version.");
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw error;
    if (!response)
      throw Error(`Could not reach the analysis backend at ${new URL(url).origin}. Check API_URL, backend availability and allowed frontend origins (CORS).`);
    throw error;
  } finally { clearTimeout(timeout); }
}

async function health() {
  try {
    const d = await api("/api/health");
    state.capabilities = d;
    $("connectionText").textContent =
      d.status === "ready"
        ? "Analysis engine connected"
        : "Analysis engine starting";
    $("connectionDot").style.background =
      d.status === "ready" ? "#3d796c" : "#eab308";
    $("capabilityDetails").textContent =
      `CNN: ${d.deep_model_loaded ? "loaded" : "unavailable"} · Acoustic ML: ${d.baseline_model_loaded ? "loaded" : "unavailable"} · Diarization: ${d.diarization || "unavailable"} · Language: ${d.language_detection ? "loaded" : "not configured"}. Confidence percentages require deployment validation.`;
  } catch (error) {
    $("connectionText").textContent = "Analysis engine offline";
    $("connectionDot").style.background = "#a84d53";
    $("capabilityDetails").textContent =
      error.message || "The analysis backend is unavailable.";
  }
}

$("connectionButton").onclick = () => $("settingsDialog").showModal();
$("saveSettings").onclick = () => {
  state.key = $("apiKey").value.trim();
  health();
};
$("menuToggle").onclick = () => {
  const open = $("menuToggle").getAttribute("aria-expanded") === "true";
  $("menuToggle").setAttribute("aria-expanded", String(!open));
  $("navigation").classList.toggle("open", !open);
};
$("navigation").onclick = (e) => {
  if (e.target.closest("a")) {
    $("navigation").classList.remove("open");
    $("menuToggle").setAttribute("aria-expanded", "false");
  }
};

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && $("navigation").classList.contains("open")) {
    $("navigation").classList.remove("open");
    $("menuToggle").setAttribute("aria-expanded", "false");
    $("menuToggle").focus();
  }
});

function tab(name) {
  if (state.recording && name === "upload") {
    notify("Stop the microphone before changing audio source.");
    return;
  }
  for (const item of ["upload", "live"]) {
    const active = item === name;
    $(item + "Tab").setAttribute("aria-selected", String(active));
    $(item + "Tab").tabIndex = active ? 0 : -1;
    $(item + "Panel").hidden = !active;
  }
}
for (const name of ["upload", "live"]) {
  $(name + "Tab").onclick = () => tab(name);
  $(name + "Tab").onkeydown = (e) => {
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) {
      e.preventDefault();
      const next =
        e.key === "Home"
          ? "upload"
          : e.key === "End"
            ? "live"
            : name === "upload"
              ? "live"
              : "upload";
      tab(next);
      $(next + "Tab").focus();
    }
  };
}
$("heroLive").onclick = () => {
  tab("live");
  $("analyze").scrollIntoView({
    behavior: reduced.matches ? "instant" : "smooth",
  });
  $("recordButton").focus({ preventScroll: true });
  if (!state.recording) startRecording();
};

function resetResult() {
  state.result = null;
  state.segmentEnd = null;
  $("audioPreview").pause();
  $("resultReadout").hidden = true;
  $("emptyReadout").hidden = false;
  $("timelineEmpty").hidden = false;
  $("timelineResult").hidden = true;
  $("signalPanels").hidden = true;
  $("readoutStatus").textContent = "AWAITING INPUT";
  $("timelineBadge").textContent = "AWAITING RECORDING";
  const tr = $("resultTranscriptBox");
  if (tr) tr.hidden = true;
  window.voiceCore?.setSignal(null);
}

function chooseFile(file, transcript = "") {
  if (!file) return;
  if (state.busy || state.recording) {
    notify("Finish the current analysis or recording first.");
    return;
  }
  clearError();
  if (file.size > 25 * 1024 * 1024) {
    fail("This file exceeds 25 MB. Choose a shorter recording.");
    return;
  }
  if (!/\.(wav|mp3|flac|ogg|aiff|aif)$/i.test(file.name)) {
    fail("Unsupported audio. Choose WAV, MP3, FLAC, OGG or AIFF.");
    return;
  }
  if (!file.size) {
    fail("The file is empty.");
    return;
  }
  resetResult();
  if (state.url) URL.revokeObjectURL(state.url);
  state.file = file;
  state.url = URL.createObjectURL(file);
  state.job = null;
  state.transcript = transcript || "";
  state.interimTranscript = "";
  $("filePrompt").textContent = "Your recording is ready";
  $("fileDetail").textContent =
    `${(file.size / 1024 / 1024).toFixed(2)} MB · ready for inspection`;
  $("fileName").textContent = file.name;
  $("fileMeta").hidden = false;
  $("audioPreview").src = state.url;
  $("audioPreview").hidden = false;
  $("analyzeButton").disabled = false;
  tab("upload");
}

$("dropZone").onclick = () => $("audioInput").click();
$("audioInput").onchange = (e) => chooseFile(e.target.files[0]);
for (const type of ["dragenter", "dragover"])
  $("dropZone").addEventListener(type, (e) => {
    e.preventDefault();
    $("dropZone").classList.add("dragging");
  });
for (const type of ["dragleave", "drop"])
  $("dropZone").addEventListener(type, (e) => {
    e.preventDefault();
    $("dropZone").classList.remove("dragging");
    if (type === "drop") chooseFile(e.dataTransfer.files[0]);
  });
$("removeFile").onclick = () => {
  if (state.busy) return;
  resetResult();
  if (state.url) URL.revokeObjectURL(state.url);
  state.file = state.url = null;
  state.transcript = "";
  $("audioInput").value = "";
  $("audioPreview").removeAttribute("src");
  $("audioPreview").hidden = true;
  $("fileMeta").hidden = true;
  $("analyzeButton").disabled = true;
  $("filePrompt").textContent = "Drop a voice into the lab";
  $("fileDetail").textContent = "or click to browse your files";
};
$("audioPreview").onloadedmetadata = () => {
  if ($("audioPreview").duration > 120) {
    fail("This recording exceeds 2 minutes. Trim it before analysis.");
    $("analyzeButton").disabled = true;
  }
};
$("audioPreview").ontimeupdate = () => {
  if (
    state.segmentEnd !== null &&
    $("audioPreview").currentTime >= state.segmentEnd
  ) {
    $("audioPreview").pause();
    state.segmentEnd = null;
  }
};

function playSegment(start, end) {
  if (!state.url) {
    notify("Audio is not attached to this saved report.");
    return;
  }
  state.segmentEnd = end;
  $("audioPreview").currentTime = start;
  $("audioPreview")
    .play()
    .catch(() =>
      notify("Playback could not start. Use the recording controls."),
    );
}

async function samples() {
  try {
    const { samples } = await api("/api/sample-audios");
    $("sampleList").replaceChildren();
    for (const item of samples.slice(0, 5)) {
      const b = el("button", "", item.title.replace(/^sample_\d+ /, ""));
      b.type = "button";
      b.onclick = async () => {
        if (state.busy || state.recording) return;
        b.disabled = true;
        try {
          const r = await fetch(apiUrl(item.url));
          if (!r.ok) throw Error("Sample could not be loaded.");
          chooseFile(
            new File([await r.blob()], item.filename, { type: "audio/wav" }),
          );
        } catch (e) {
          fail(e.message);
        } finally {
          b.disabled = false;
        }
      };
      $("sampleList").append(b);
    }
    if (!samples.length)
      $("sampleList").textContent = "No development samples installed.";
  } catch {
    $("sampleList").textContent = "Samples require the analysis server.";
  }
}

function busy(value) {
  state.busy = value;
  $("analyzeButton").setAttribute("aria-busy", String(value));
  $("analyzeButton").disabled = value || !state.file;
  $("removeFile").disabled = value;
  $("dropZone").disabled = value;
  $("recordButton").disabled = value;
  $("readout").setAttribute("aria-busy", String(value));
  window.voiceCore?.setActive(value);
}

function stage(index) {
  $("analysisStage").textContent = stages[index] || stages[0];
  const progress = Math.round(((index + 1) / stages.length) * 95);
  $("progressLine").style.transform = `scaleX(${progress / 100})`;
  $("progressLine").parentElement.setAttribute("aria-valuenow", String(progress));
  $("stageList").replaceChildren(
    ...stages.map((s, i) => {
      const li = el("li", i < index ? "done" : i === index ? "active" : "");
      li.append(
        el("span", "", i < index ? "✓" : String(i + 1).padStart(2, "0")),
        document.createTextNode(s),
      );
      return li;
    }),
  );
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function runAnalysis() {
  if (!state.file || state.busy || state.recording) return;
  clearError();
  resetResult();
  busy(true);
  $("emptyReadout").hidden = true;
  $("loadingReadout").hidden = false;
  $("readoutStatus").textContent = "PROCESSING";
  stage(0);
  try {
    const form = new FormData();
    form.append("file", state.file);
    if (state.transcript) form.append("transcript", state.transcript);

    const created = await api("/api/jobs", { method: "POST", body: form });
    state.job = created.job_id;
    state.sessionJobs.add(state.job);
    const deadline = Date.now() + 12 * 60 * 1000;
    let result;
    while (Date.now() < deadline) {
      const job = await api(`/api/jobs/${encodeURIComponent(state.job)}`);
      stage(job.stage);
      if (job.status === "failed") throw Error(job.error);
      if (job.status === "complete") {
        result = job.result;
        break;
      }
      await sleep(1500);
    }
    if (!result)
      throw Error(
        "Analysis is taking longer than expected. The server may still be processing.",
      );
    if (state.transcript && !result.transcript)
      result.transcript = state.transcript;
    $("progressLine").style.transform = "scaleX(1)";
    $("progressLine").parentElement.setAttribute("aria-valuenow", "100");
    $("analysisStage").textContent = "Evidence ready";
    if (!reduced.matches) await sleep(220);
    renderResult(result);
    saveHistory(result);
    notify("Analysis complete. Review the forensic evidence and findings.");
  } catch (e) {
    fail(e.message || "Could not reach the server.");
    $("readoutStatus").textContent = "ANALYSIS UNAVAILABLE";
    $("emptyReadout").hidden = false;
  } finally {
    busy(false);
    $("loadingReadout").hidden = true;
  }
}
$("analyzeButton").onclick = runAnalysis;

function evidenceRows(container, indicators) {
  container.replaceChildren();
  for (const item of indicators) {
    const detail = el("details"),
      summary = el("summary"),
      bar = el("span", "evidence-bar"),
      fill = el("i");
    fill.style.width = `${Math.min(100, Math.max(0, item.score))}%`;
    bar.append(fill);
    summary.append(
      el("span", "", item.name),
      bar,
      el("span", "evidence-score", String(item.score)),
    );
    detail.append(
      summary,
      el(
        "p",
        "",
        `${item.description} Fusion weight: ${Math.round(item.contribution_weight * 100)}%. Uncalibrated model response / concern index.`,
      ),
    );
    container.append(detail);
  }
  if (!indicators.length)
    container.append(
      el(
        "p",
        "subtle",
        "Model evidence is withheld when the signal is unusable or required models are unavailable.",
      ),
    );
}

function renderResult(result) {
  state.result = result;
  $("loadingReadout").hidden = true;
  $("emptyReadout").hidden = true;
  $("resultReadout").hidden = false;
  $("readoutStatus").textContent = "ANALYSIS COMPLETE";
  const r = result.analysis;
  const color =
    r.risk_level === "HIGH"
      ? "#a84d53"
      : r.risk_level === "MEDIUM"
        ? "#d97706"
        : r.risk_level === "LOW"
          ? "#3d796c"
          : "#7991ad";
  $("riskGauge").style.setProperty("--risk-color", color);
  $("gaugeValue").style.strokeDashoffset = "477.52";
  requestAnimationFrame(() => requestAnimationFrame(() => {
    if (state.result === result) $("gaugeValue").style.strokeDashoffset = String(
      477.52 * (1 - (r.risk_score ?? 0) / 100));
  }));
  $("gaugeScore").textContent =
    r.risk_score == null ? "—" : String(Math.round(r.risk_score));
  if (r.risk_score != null && !reduced.matches) {
    const start = performance.now();
    const tick = (now) => {
      if (state.result !== result) return;
      const x = Math.min(1, (now - start) / 900);
      $("gaugeScore").textContent = String(
        Math.round(r.risk_score * (1 - (1 - x) ** 3)),
      );
      if (x < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  $("verdictText").textContent =
    r.risk_score == null
      ? "Insufficient evidence"
      : `${r.risk_level[0] + r.risk_level.slice(1).toLowerCase()} acoustic concern`;
  $("verdictTag").textContent = (
    r.confidence_label || "EVALUATED"
  ).toUpperCase();
  $("confidence").textContent = "Acoustic signal evaluation complete";
  $("summarySpeakers").textContent =
    result.diarization.num_speakers == null
      ? "Not determined"
      : `${result.diarization.num_speakers} estimated`;
  $("summaryLanguage").textContent = langText(result.language);
  $("languageEvidence").textContent = result.language.reason || "Language evidence unavailable.";
  $("summaryQuality").textContent = result.quality.label;
  $("summaryDuration").textContent = fmt(result.duration_seconds);
  $("resultReason").textContent =
    (r.reasoning || []).join(" ") + " " + (r.score_kind || "");
  const trBox = $("resultTranscriptBox");
  const trText = $("resultTranscriptText");
  const transcriptContent = result.transcript || state.transcript;
  if (trBox && trText) {
    if (transcriptContent && transcriptContent.trim()) {
      trBox.hidden = false;
      trText.textContent = `“${transcriptContent.trim()}”`;
    } else {
      trBox.hidden = true;
    }
  }
  evidenceRows($("evidenceRows"), r.indicators || []);
  renderTimeline(result);
  renderVocal(result.vocal_state);
  $("signalPanels").hidden = false;
  drawSpectrogram(result.spectrogram);
  window.voiceCore?.setSignal(result.waveform_preview);
  $("coreCaption").textContent = "Analyzed waveform · geometry is illustrative";
  $("readout").scrollIntoView({
    behavior: reduced.matches ? "instant" : "smooth",
  });
}

function renderTimeline(result) {
  const d = result.diarization;
  $("timelineBadge").textContent =
    ["pyannote", "pretrained_onnx"].includes(d.method) ? "PRETRAINED SPEAKER ESTIMATE" : "COUNT NOT VERIFIED";
  renderSpeakerOverview(d);
  $("timelineEmpty").hidden = true;
  $("timelineResult").hidden = false;
  $("diarizationNote").textContent = (d.limitations || []).join(" ");
  $("timelineEnd").textContent = fmt(result.duration_seconds);
  $("speakerLanes").replaceChildren();
  for (const [index, s] of (d.speakers || []).entries()) {
    const lane = el("div", "speaker-lane");
    lane.dataset.speaker = s.speaker_id;
    lane.style.setProperty("--lane-color", palette[index % palette.length]);
    const name = el("button", "speaker-name", s.speaker_id);
    name.onclick = () => selectSpeaker(s);
    const track = el("div", "lane-track");
    for (const t of (d.speaker_turns || []).filter(
      (t) => t.speaker === s.speaker_id,
    )) {
      const b = el("button", "segment" + (t.overlap ? " overlap" : ""));
      b.style.left = `${(100 * t.start) / result.duration_seconds}%`;
      b.style.width = `${Math.max(2, (100 * (t.end - t.start)) / result.duration_seconds)}%`;
      b.setAttribute(
        "aria-label",
        `${s.speaker_id}, ${t.start.toFixed(1)} to ${t.end.toFixed(1)} seconds. Play segment.`,
      );
      b.title = b.getAttribute("aria-label");
      b.onclick = () => {
        selectSpeaker(s);
        playSegment(t.start, t.end);
      };
      track.append(b);
    }
    lane.append(name, track);
    $("speakerLanes").append(lane);
  }
  if (d.speakers?.length) selectSpeaker(d.speakers[0]);
  else
    $("speakerDetail").replaceChildren(
      el("p", "subtle", "No individual speaker could be attributed. Review the overall audio findings."),
    );
}


// Browser transcription language is explicitly separate from UI and detected language.
$("speechLanguage").onchange = () => {
  state.speechLanguage = $("speechLanguage").value;
  if (state.recording) {
    stopSpeechRecognition();
    startSpeechRecognition();
  }
};

function renderSpeakerOverview(d) {
  const overview = $("speakerOverview");
  overview.replaceChildren();
  $("speakerCountTitle").textContent = d.num_speakers == null
    ? "Speaker count could not be determined"
    : `${d.num_speakers} distinct voice${d.num_speakers === 1 ? "" : "s"} estimated`;
  for (const speaker of d.speakers || []) {
    const button = el("button", "speaker-overview-card");
    button.setAttribute("aria-label", `Inspect ${speaker.speaker_id}`);
    const score = speaker.analysis?.risk_score;
    button.append(el("span", "speaker-overview-name", speaker.speaker_id),
      el("strong", "", score == null ? "Not enough evidence" : `${Number(score).toFixed(1)} / 100`),
      el("span", "subtle", score == null ? "Individual risk unavailable" : `${speaker.analysis.risk_level} concern · uncalibrated`),
      el("small", "", `${speaker.speaking_time_sec}s active · ${speaker.isolated_speech_seconds ?? "—"}s isolated`));
    button.onclick = () => selectSpeaker(speaker);
    overview.append(button);
  }
}

function selectSpeaker(s) {
  for (const button of $("speakerOverview").children) {
    button.setAttribute("aria-pressed", String(button.querySelector(".speaker-overview-name").textContent === s.speaker_id));
  }
  for (const lane of $("speakerLanes").children) {
    const selected = lane.dataset.speaker === s.speaker_id;
    lane.classList.toggle("selected", selected);
    lane
      .querySelector(".speaker-name")
      .setAttribute("aria-pressed", String(selected));
  }
  const intro = el("div");
  intro.append(
    el("div", "eyebrow", "SELECTED VOICE"),
    el("h3", "", s.speaker_id),
    el("p", "", riskText(s.analysis)),
  );
  const t = state.result?.diarization?.speaker_turns?.find(
    (t) => t.speaker === s.speaker_id,
  );
  if (t) {
    const play = el("button", "button secondary small", "Play segment ↗");
    play.onclick = () => playSegment(t.start, t.end);
    intro.append(play);
  }
  const content = el("div"),
    summary = el("div", "summary-grid");
  for (const [label, value] of [
    ["SPEAKING TIME", `${s.speaking_time_sec?.toFixed(1) || "—"} sec`],
    ["ISOLATED AUDIO", `${s.isolated_speech_seconds ?? "—"} sec`],
    ["OVERLAPPING", `${s.overlap_seconds ?? "—"} sec`],
    ["LANGUAGE", langText(s.language)],
    ["CONFIDENCE", s.analysis?.confidence_label || "Evaluated"],
    ["VOCAL STATE", s.vocal_state?.label || "Measured"],
  ]) {
    const cell = el("div");
    cell.append(el("small", "", label), el("strong", "", value));
    summary.append(cell);
  }
  content.append(
    summary,
    el(
      "p",
      "notice",
      `${s.vocal_state?.description || ""} ${(s.analysis?.reasoning || []).join(" ")}`,
    ),
  );
  const details = el("details", "sample-details");
  details.append(
    el(
      "summary",
      "",
      `Speaker evidence · ${s.suspicious_intervals?.length || 0} elevated-concern intervals`,
    ),
  );
  const list = el("div", "evidence-list");
  evidenceRows(list, s.analysis?.indicators || []);
  details.append(list);
  for (const interval of s.suspicious_intervals || []) {
    const b = el(
      "button",
      "text-button",
      `${interval.start.toFixed(1)}–${interval.end.toFixed(1)} s · ${interval.risk_score}/100 ↗`,
    );
    b.onclick = () => playSegment(interval.start, interval.end);
    details.append(b);
  }
  content.append(details);
  $("speakerDetail").replaceChildren(intro, content);
  if (s.vocal_state) renderVocal(s.vocal_state);
}

function renderVocal(v) {
  if (!v) return;
  const box = $("vocalDetails");
  box.replaceChildren(
    el("div", "acoustic-label", v.label),
    el("p", "", v.description),
  );
  const grid = el("div", "acoustic-list"),
    f = v.features || {};
  for (const [label, value, unit] of [
    ["Median pitch", f.pitch_median_hz, " Hz"],
    ["Pitch variation", f.pitch_variation_hz, " Hz"],
    ["Signal level", f.rms_dbfs, " dBFS"],
    ["Energy variation", f.energy_variation, ""],
  ]) {
    const item = el("div");
    item.append(
      el("small", "", label),
      el("strong", "", value == null ? "Unavailable" : `${value}${unit}`),
    );
    grid.append(item);
  }
  box.append(grid, el("p", "subtle", v.disclaimer || ""));
}

function drawSpectrogram(matrix) {
  const c = $("spectrogram"),
    ctx = c.getContext("2d");
  c.width = Math.max(300, c.clientWidth * devicePixelRatio);
  c.height = 220 * devicePixelRatio;
  ctx.fillStyle = "#f3f7fd";
  ctx.fillRect(0, 0, c.width, c.height);
  if (!matrix?.length) {
    ctx.fillStyle = "#627389";
    ctx.font = "13px Segoe UI";
    ctx.fillText("Spectrogram generated after speech processing.", 20, 40);
    return;
  }
  const rows = matrix.length,
    cols = matrix[0].length;
  for (let r = 0; r < rows; r++)
    for (let i = 0; i < cols; i++) {
      const v = Math.max(0, Math.min(1, (matrix[r][i] + 80) / 80));
      ctx.fillStyle = `rgb(${Math.round(246 - v * 200)},${Math.round(249 - v * 145)},${Math.round(254 - v * 80)})`;
      ctx.fillRect(
        (i * c.width) / cols,
        c.height - ((r + 1) * c.height) / rows,
        c.width / cols + 1,
        c.height / rows + 1,
      );
    }
}

function saveHistory(result) {
  state.history.unshift({ result, job: state.job });
  state.history = state.history.slice(0, 8);
  renderHistory();
}

function renderHistory() {
  $("historyList").replaceChildren();
  if (!state.history.length) {
    $("historyList").append(
      el(
        "p",
        "subtle",
        "No investigations yet. Results stay in this tab until you close or refresh it.",
      ),
    );
    return;
  }
  for (const entry of state.history) {
    const r = entry.result,
      row = el("div", "history-entry");
    row.append(
      el("strong", "", r.filename),
      el("small", "", new Date(r.timestamp).toLocaleTimeString()),
      el("span", "history-risk", riskText(r.analysis)),
    );
    const view = el("button", "", "Open report ↗");
    view.onclick = () => {
      if (state.busy || state.recording) return;
      state.job = entry.job;
      if (state.result?.analysis_id !== r.analysis_id) {
        if (state.url) URL.revokeObjectURL(state.url);
        state.url = state.file = null;
        $("audioPreview").pause();
        $("audioPreview").removeAttribute("src");
        $("audioPreview").hidden = true;
        $("fileMeta").hidden = true;
        $("analyzeButton").disabled = true;
        $("filePrompt").textContent = "Attach a recording to begin";
      }
      renderResult(r);
    };
    row.append(view);
    $("historyList").append(row);
  }
}

$("clearHistory").onclick = async () => {
  if (state.busy) {
    notify("Wait for the current analysis before clearing reports.");
    return;
  }
  let failed = 0;
  for (const job of state.sessionJobs) {
    try {
      await api(`/api/jobs/${encodeURIComponent(job)}`, { method: "DELETE" });
    } catch {
      failed++;
    }
  }
  state.sessionJobs.clear();
  state.history = [];
  renderHistory();
  notify(
    failed
      ? "Local history cleared."
      : "Session history and temporary server reports cleared.",
  );
};

const escapeHTML = (value) =>
  String(value ?? "Unavailable").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );

function downloadReport() {
  const r = state.result;
  if (!r) return;
  const esc = escapeHTML,
    pairs = (items) =>
      `<dl>${items.map(([a, b]) => `<div><dt>${esc(a)}</dt><dd>${esc(b)}</dd></div>`).join("")}</dl>`,
    bullets = (arr) =>
      `<ul>${arr.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>`;
  const evidence = (items) =>
      `<table><thead><tr><th>Evidence source</th><th>Response index</th><th>Weight</th><th>Interpretation</th></tr></thead><tbody>${items.map((i) => `<tr><td>${esc(i.name)}</td><td>${esc(i.score)} / 100</td><td>${esc(i.contribution_weight * 100)}%</td><td>${esc(i.description)}</td></tr>`).join("")}</tbody></table>`,
    vocal = (v) =>
      pairs(
        Object.entries(v?.features || {}).map(([k, val]) => [
          k.replaceAll("_", " "),
          val,
        ]),
      );
  const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VoiceGuard report ${esc(r.analysis_id)}</title><style>body{font:14px/1.7 Arial,sans-serif;color:#18324f;max-width:980px;margin:45px auto;padding:0 30px}h1{font-size:36px;letter-spacing:-1px;line-height:1.2}h2{margin-top:38px;font-size:23px}h3{margin-top:27px}p,li{color:#50657b}header{border-bottom:2px solid #3568ac;padding-bottom:25px}.meta{font:11px monospace;color:#71869f}.risk{font-size:30px;color:#3568ac}dl{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}dt{font-size:11px;color:#6b7e92}dd{margin:3px 0;font-weight:600;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:11px}td,th{text-align:left;border-bottom:1px solid #dfe7f1;padding:10px;vertical-align:top}th{background:#eef4fb}.note{background:#eff5fc;padding:15px;border-left:3px solid #789bc5}.transcript-quote{background:#fffbeb;border-left:3px solid #f59e0b;padding:14px;font-style:italic;margin:16px 0;border-radius:4px;}article{border-top:1px solid #dbe5f0;margin-top:22px;padding-top:5px}footer{margin-top:40px;border-top:1px solid #dbe5f0;font-size:11px;color:#6b7e92}@media(max-width:600px){dl{grid-template-columns:1fr 1fr}body{padding:15px;margin:0}table{font-size:9px}td,th{padding:5px}}@media print{@page{size:A4;margin:16mm}body{margin:0;padding:0;font-size:11px}h2,h3{break-after:avoid}tr,dl,.note{break-inside:avoid}thead{display:table-header-group}}</style></head><body><header><p class="meta">VOICEGUARD AI / FORENSIC INVESTIGATION</p><h1>Voice analysis report</h1><p>${esc(r.filename)}</p>${pairs(
    [
      ["Analysis ID", r.analysis_id],
      ["Timestamp UTC", r.timestamp],
      ["Duration", `${r.duration_seconds} seconds`],
    ],
  )}</header><h2>Overall findings</h2><p class="risk">${esc(riskText(r.analysis))}</p><p class="note">${esc(r.analysis.score_kind || "")} ${esc(r.analysis.confidence_basis || "")}</p>${r.transcript ? `<h2>Transcribed Speech (STT)</h2><div class="transcript-quote">“${esc(r.transcript)}”</div>` : ""}${pairs(
    [
      ["Confidence", r.analysis.confidence_label],
      ["Estimated speakers", r.diarization.num_speakers],
      ["Language", langText(r.language)],
      ["Audio quality", r.quality.label],
      ["Original sample rate", `${r.quality.original_sample_rate_hz} Hz`],
      ["Diarization method", r.diarization.method],
    ],
  )}${bullets(r.analysis.reasoning || [])}<h2>Detection evidence & model contribution</h2>${evidence(r.analysis.indicators || [])}<h2>Speaker-level findings</h2>${
    r.diarization.speakers?.length
      ? r.diarization.speakers
          .map(
            (s) =>
              `<article><h3>${esc(s.speaker_id)}</h3>${pairs([
                ["Concern index", riskText(s.analysis)],
                ["Speaking duration", `${s.speaking_time_sec} seconds`],
                ["Isolated speech used for risk", `${s.isolated_speech_seconds ?? "Unavailable"} seconds`],
                ["Overlap excluded from risk", `${s.overlap_seconds ?? "Unavailable"} seconds`],
                ["Confidence", s.analysis.confidence_label],
                ["Language", langText(s.language)],
                ["Vocal state", s.vocal_state?.label || "Measured"],
                ["Speaker attribution", s.attribution_confidence],
              ])}<p>${esc(s.vocal_state?.description || "")}</p>${vocal(s.vocal_state)}${evidence(s.analysis.indicators || [])}<h4>Original recording intervals</h4>${bullets((r.diarization.speaker_turns || []).filter((t) => t.speaker === s.speaker_id).map((t) => `${t.start.toFixed(2)}–${t.end.toFixed(2)} seconds${t.overlap ? " · overlap" : ""}`))}<h4>Elevated-concern intervals</h4>${bullets((s.suspicious_intervals || []).map((t) => `${t.start.toFixed(2)}–${t.end.toFixed(2)} seconds · ${t.risk_score}/100`))}${bullets(s.analysis.reasoning || [])}</article>`,
          )
          .join("")
      : "<p>Single speaker evaluated.</p>"
  }<h2>Language evidence</h2><p>${esc(r.language.reason)}</p>${bullets((r.language.languages || []).map((l) => `${languageNames[l.code] || l.code}: ${(l.model_score * 100).toFixed(1)}% model response`))}<h2>Recording quality & vocal characteristics</h2>${pairs(
    [
      ["Candidate speech", `${r.quality.candidate_speech_seconds} seconds`],
      ["Signal level", `${r.quality.rms_dbfs} dBFS`],
      ["Clipping", `${(r.quality.clipping_fraction * 100).toFixed(2)}%`],
    ],
  )}${bullets(r.quality.issues || [])}${vocal(r.vocal_state)}<h2>Limitations</h2>${bullets(r.limitations || [])}<h2>Security recommendations</h2>${bullets(r.recommendations || [])}<footer>Exploratory evidence. This report does not authenticate a person or authorize a transaction. Use your browser’s Print command to save as PDF. No audio is embedded.</footer></body></html>`;
  const url = URL.createObjectURL(
      new Blob([html], { type: "text/html;charset=utf-8" }),
    ),
    a = el("a");
  a.href = url;
  a.download = `VoiceGuard-${r.analysis_id.slice(0, 8)}.html`;
  a.hidden = true;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}
$("reportButton").onclick = downloadReport;

async function pcmBlob(parts, sampleRate) {
  const total = parts.reduce((n, p) => n + p.length, 0);
  if (!total) throw Error("No audio samples were captured.");
  const data = new Float32Array(total);
  let offset = 0;
  for (const p of parts) {
    data.set(p, offset);
    offset += p.length;
  }
  let pcm = data;
  let outRate = sampleRate;
  if (sampleRate !== 16000) {
    try {
      const offline = new OfflineAudioContext(
        1,
        Math.max(1, Math.round((total * 16000) / sampleRate)),
        16000,
      );
      const buffer = offline.createBuffer(1, total, sampleRate);
      buffer.copyToChannel(data, 0);
      const source = offline.createBufferSource();
      source.buffer = buffer;
      source.connect(offline.destination);
      source.start();
      pcm = (await offline.startRendering()).getChannelData(0);
      outRate = 16000;
    } catch (err) {
      console.warn("Offline resampling fallback to native rate:", err);
      pcm = data;
      outRate = sampleRate;
    }
  } else {
    outRate = 16000;
  }
  const output = new ArrayBuffer(44 + pcm.length * 2),
    v = new DataView(output),
    str = (at, s) => {
      for (let i = 0; i < s.length; i++) v.setUint8(at + i, s.charCodeAt(i));
    };
  str(0, "RIFF");
  v.setUint32(4, 36 + pcm.length * 2, true);
  str(8, "WAVE");
  str(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, outRate, true);
  v.setUint32(28, outRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, "data");
  v.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i++) {
    const x = Math.max(-1, Math.min(1, pcm[i]));
    v.setInt16(44 + i * 2, x < 0 ? x * 32768 : x * 32767, true);
  }
  return new Blob([output], { type: "audio/wav" });
}

function startSpeechRecognition() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {
    const p = $("liveTranscriptText");
    if (p)
      p.textContent =
        "Speech recognition requires Chrome, Edge, or Safari with Web Speech support. Voice analysis continues normally.";
    return;
  }
  try {
    const rec = new SpeechRec();
    rec.continuous = true;
    rec.interimResults = true;
    rec.maxAlternatives = 1;
    rec.lang =
      state.speechLanguage === "hi"
        ? "hi-IN"
        : state.speechLanguage === "bn"
          ? "bn-IN"
          : state.speechLanguage === "en"
            ? "en-US"
            : (navigator.language || "en-US");
    rec.onresult = (e) => {
      let interim = "";
      for (let i = e.resultIndex; i < e.results.length; ++i) {
        if (e.results[i].isFinal) {
          state.transcript +=
            (state.transcript ? " " : "") + e.results[i][0].transcript.trim();
        } else {
          interim += e.results[i][0].transcript;
        }
      }
      state.interimTranscript = interim;
      const display = (
        state.transcript + (interim ? " " + interim : "")
      ).trim();
      const p = $("liveTranscriptText");
      if (p) {
        p.textContent = display ? `“${display}”` : "Listening for speech…";
        p.classList.toggle("transcript-placeholder", !display);
      }
      $("liveTranscriptBox")?.classList.toggle("active", !!display);
    };
    rec.onerror = (e) => {
      if (e.error !== "no-speech")
        console.warn("Speech recognition notice:", e.error);
    };
    rec.onend = () => {
      if (state.recording) {
        try {
          rec.start();
        } catch {}
      }
    };
    rec.start();
    state.speechRec = rec;
  } catch (e) {
    console.warn("Speech recognition start failed:", e);
  }
}

function stopSpeechRecognition() {
  if (state.speechRec) {
    try {
      state.speechRec.onend = null;
      state.speechRec.stop();
    } catch {}
    state.speechRec = null;
  }
}

async function startRecording() {
  if (state.busy || state.recording || state.starting) return;
  clearError();
  if (!navigator.mediaDevices?.getUserMedia) {
    fail(
      "Live capture requires a supported browser on HTTPS or localhost. You can still upload audio.",
    );
    return;
  }
  state.starting = true;
  $("recordButton").disabled = true;
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
      video: false,
    });
    state.context = new (window.AudioContext || window.webkitAudioContext)();
    await state.context.resume();
    try {
      await state.context.audioWorklet.addModule("recorder-worklet.js");
      state.worklet = new AudioWorkletNode(
        state.context,
        "voiceguard-recorder",
      );
    } catch {
      // Fallback in case of relative path issue or local port
      try {
        await state.context.audioWorklet.addModule("recorder-worklet.js");
        state.worklet = new AudioWorkletNode(
          state.context,
          "voiceguard-recorder",
        );
      } catch (err) {
        console.warn(
          "Worklet load failed, falling back to script processor:",
          err,
        );
      }
    }
    state.source = state.context.createMediaStreamSource(state.stream);
    state.analyser = state.context.createAnalyser();
    state.analyser.fftSize = 1024;
    state.source.connect(state.analyser);

    state.full = [];
    state.pending = [];
    state.samples = 0;
    const session = await api('/api/live-sessions', { method: 'POST' });
    state.liveSession = session.session_id;
    state.liveChunkIndex = 0;
    state.liveSentSamples = 0;
    state.livePacket = null;
    state.liveBusy = false;
    state.recording = true;
    state.recordToken++;
    state.recordStart = Date.now();
    state.transcript = "";
    state.interimTranscript = "";

    if (state.worklet) {
      const silent = state.context.createGain();
      silent.gain.value = 0;
      state.analyser.connect(state.worklet);
      state.worklet.connect(silent).connect(state.context.destination);
      state.worklet.port.onmessage = (e) => {
        if (!state.recording) return;
        const part = new Float32Array(e.data);
        const remaining =
          Math.floor(120 * state.context.sampleRate) - state.samples;
        if (remaining <= 0) {
          stopRecording();
          return;
        }
        const trimmed = part.subarray(0, remaining);
        state.full.push(trimmed);
        state.pending.push(trimmed);
        state.samples += trimmed.length;
      };
    } else {
      // ScriptProcessor fallback
      const sp = state.context.createScriptProcessor(2048, 1, 1);
      state.source.connect(sp);
      sp.connect(state.context.destination);
      sp.onaudioprocess = (e) => {
        if (!state.recording) return;
        const input = e.inputBuffer.getChannelData(0);
        const part = new Float32Array(input);
        state.full.push(part);
        state.pending.push(part);
        state.samples += part.length;
      };
      state.scriptProcessor = sp;
    }

    $("recordButton").textContent = "Stop & Analyze Voice ■";
    $("recordButton").classList.add("recording");
    $("liveStatus").textContent = "● LISTENING";
    $("liveStatus").classList.add("active-status");
    $("liveRisk").textContent = "Gathering first window…";
    $("livePanel")?.classList.add("recording-active");
    const p = $("liveTranscriptText");
    if (p) {
      p.textContent =
        "Spoken words will appear here in real time as you speak…";
      p.classList.add("transcript-placeholder");
    }

    startSpeechRecognition();

    state.liveTimer = setInterval(streamWindow, 3500);
    state.recordTimer = setInterval(() => {
      const seconds = (Date.now() - state.recordStart) / 1000;
      const el = $("recordTimer");
      el.textContent = fmt(seconds);
      el.classList.add("ticking");
      if (seconds >= 120) stopRecording();
    }, 250);
    window.voiceCore?.setActive(true);
    drawLive();
  } catch (e) {
    await cleanupRecording();
    fail(
      e.name === "NotAllowedError"
        ? "Microphone permission was denied. Allow it in browser settings, or upload a recording."
        : `Microphone could not start: ${e.message}`,
    );
  } finally {
    state.starting = false;
    $("recordButton").disabled = false;
  }
}

async function streamWindow() {
  if (!state.recording || state.liveBusy || (!state.pending.length && !state.livePacket)) return;
  const token = state.recordToken;
  state.liveBusy = true;
  const sampleRate = state.context.sampleRate;
  if (!state.livePacket) {
    const parts = [];
    let remaining = 8 * sampleRate;
    while (remaining > 0 && state.pending.length) {
      const part = state.pending.shift();
      const count = Math.min(remaining, part.length);
      parts.push(part.subarray(0, count));
      if (count < part.length) state.pending.unshift(part.subarray(count));
      remaining -= count;
    }
    state.livePacket = { parts, samples: 8 * sampleRate - remaining,
      start: state.liveSentSamples / sampleRate, index: state.liveChunkIndex,
      session: state.liveSession, transcript: state.transcript };
  }
  const packet = state.livePacket;
  const controller = new AbortController();
  state.liveAbort = controller;
  const timeout = setTimeout(() => controller.abort(), 25000);
  try {
    const blob = await pcmBlob(packet.parts, sampleRate),
      form = new FormData();
    form.append("file", blob, "live-window.wav");
    form.append('session_id', packet.session);
    form.append('chunk_index', packet.index);
    form.append('chunk_start_sec', packet.start);
    if (packet.transcript) form.append("transcript", packet.transcript);

    const r = await api("/api/analyze-chunk", {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    if (!state.recording || token !== state.recordToken) return;
    state.liveSentSamples += packet.samples;
    state.liveChunkIndex++;
    state.livePacket = null;
    if (r.analysis?.risk_score != null) {
      $("liveRisk").textContent = `${Math.round(r.analysis.risk_score)} / 100 · ${r.analysis.risk_level}`;
    } else {
      $("liveRisk").textContent = "Listening for speech…";
    }
    $("liveLanguage").textContent = langText(r.language);
    $("liveVocal").textContent = r.vocal_state?.label || "Measured";
    const speakerCount = r.diarization?.num_speakers;
    $("liveStatus").textContent = speakerCount == null
      ? '● STREAMING · SPEAKERS UNCERTAIN'
      : `● STREAMING · ${speakerCount} SPEAKER${speakerCount === 1 ? '' : 'S'} ESTIMATED`;
  } catch (e) {
    if (e.name !== "AbortError") {
      console.warn("[VoiceGuard] Live window analysis error:", e.message || e);
    }
    if (state.recording && token === state.recordToken) {
      $("liveRisk").textContent = e.name === "AbortError" ? "Analysis timed out; retrying the same audio window…" : e.message;
      $("liveStatus").textContent = "● RECORDING · ANALYSIS RETRYING";
    }
  } finally {
    clearTimeout(timeout);
    if (token === state.recordToken) state.liveBusy = false;
  }
}

async function cleanupRecording() {
  state.recording = false;
  state.recordToken++;
  clearInterval(state.liveTimer);
  clearInterval(state.recordTimer);
  cancelAnimationFrame(state.liveFrame);
  state.liveAbort?.abort();
  const sessionId = state.liveSession;
  state.liveSession = null;
  state.livePacket = null;
  state.liveBusy = false;
  if (sessionId) api(`/api/live-sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' }).catch(() => {});
  stopSpeechRecognition();
  state.stream?.getTracks().forEach((t) => t.stop());
  state.worklet?.disconnect();
  state.source?.disconnect();
  if (state.scriptProcessor) {
    state.scriptProcessor.disconnect();
    state.scriptProcessor = null;
  }
  if (state.context && state.context.state !== "closed") {
    try {
      await state.context.close();
    } catch {}
  }
  state.stream =
    state.context =
    state.analyser =
    state.worklet =
    state.source =
      null;
  window.voiceCore?.setActive(false);
  $("recordButton").textContent = "Enable microphone ◉";
  $("recordButton").classList.remove("recording");
  $("liveStatus").textContent = "MICROPHONE OFF";
  $("liveStatus").classList.remove("active-status");
  $("livePanel")?.classList.remove("recording-active");
  const timerEl = $("recordTimer");
  if (timerEl) timerEl.classList.remove("ticking");
  $("voiceActivityDot")?.classList.remove("active");
  $("liveVoiceBar")?.classList.remove("speaking");
  const txt = $("voiceActivityText");
  if (txt) txt.textContent = "Awaiting speech input…";
}

async function stopRecording() {
  if (!state.recording || state.stopping) return;
  state.stopping = true;
  const sampleRate = state.context?.sampleRate || 16000;
  const parts = state.full.slice();
  const capturedTranscript = (
    state.transcript +
    (state.interimTranscript ? " " + state.interimTranscript : "")
  ).trim();
  await cleanupRecording();
  state.full = [];
  state.pending = [];
  try {
    if (parts.length) {
      const blob = await pcmBlob(parts, sampleRate);
      const audioFile = new File(
        [blob],
        `VoiceGuard-voice-${new Date().toISOString().replaceAll(":", "-")}.wav`,
        { type: "audio/wav" },
      );
      chooseFile(audioFile, capturedTranscript);
      notify("Voice input captured. Running full forensic analysis…");
      // Automatically initiate analysis seamlessly
      await runAnalysis();
    } else {
      fail("No audio samples captured. Check your microphone settings.");
    }
  } catch (e) {
    fail(e.message);
  } finally {
    state.stopping = false;
  }
}
$("recordButton").onclick = () =>
  state.recording ? stopRecording() : startRecording();

function drawLive(now = performance.now()) {
  if (!state.recording || !state.analyser) return;
  if (document.hidden || now - (state.lastLiveDraw || 0) < (reduced.matches ? 150 : 33)) {
    state.liveFrame = requestAnimationFrame(drawLive);
    return;
  }
  state.lastLiveDraw = now;
  const wave = state.waveBuffer || (state.waveBuffer = new Uint8Array(state.analyser.fftSize)),
    freq = state.frequencyBuffer || (state.frequencyBuffer = new Uint8Array(state.analyser.frequencyBinCount));
  state.analyser.getByteTimeDomainData(wave);
  state.analyser.getByteFrequencyData(freq);

  // Energy & amplitude
  let energy = 0;
  for (const x of wave) energy += ((x - 128) / 128) ** 2;
  const db = 20 * Math.log10(Math.sqrt(energy / wave.length) + 1e-9);
  $("liveAmplitude").textContent = `${Math.max(-96, db).toFixed(1)} dBFS`;

  // Voice activity detection
  const isSpeaking = db > -46;
  const dot = $("voiceActivityDot"),
    bar = $("liveVoiceBar"),
    txt = $("voiceActivityText");
  if (isSpeaking) {
    dot?.classList.add("active");
    bar?.classList.add("speaking");
    if (txt) txt.textContent = "● VOICE DETECTED — CAPTURING SPEECH";
  } else {
    dot?.classList.remove("active");
    bar?.classList.remove("speaking");
    if (txt) txt.textContent = "Listening for voice input…";
  }

  // Feed 3D core with sampled waveform
  window.voiceCore?.setSignal(
    Array.from(
      wave.filter((_, i) => i % 8 === 0),
      (v) => (v - 128) / 128,
    ),
  );

  // ── Waveform canvas ────────────────────────────────────────────────────────
  const wc = $("liveWave"),
    wctx = wc.getContext("2d");
  const ratio = Math.min(2, devicePixelRatio || 1);
  const waveWidth = Math.max(1, Math.round(wc.clientWidth * ratio));
  if (wc.width !== waveWidth) wc.width = waveWidth;
  if (wc.height !== 115 * ratio) wc.height = 115 * ratio;
  wctx.clearRect(0, 0, wc.width, wc.height);

  const midY = wc.height / 2;

  // Subtle centre reference line
  wctx.beginPath();
  wctx.moveTo(0, midY);
  wctx.lineTo(wc.width, midY);
  wctx.strokeStyle = isSpeaking
    ? "rgba(251,191,36,0.22)"
    : "rgba(140,170,210,0.18)";
  wctx.lineWidth = 1 * devicePixelRatio;
  wctx.stroke();

  // Filled area
  wctx.beginPath();
  for (let i = 0; i < wave.length; i++) {
    const x = (i / wave.length) * wc.width,
      y = midY + ((wave[i] - 128) / 128) * (wc.height * 0.42);
    i ? wctx.lineTo(x, y) : wctx.moveTo(x, y);
  }
  wctx.lineTo(wc.width, midY);
  wctx.lineTo(0, midY);
  wctx.closePath();
  wctx.fillStyle = isSpeaking
    ? "rgba(245,158,11,0.09)"
    : "rgba(100,140,200,0.06)";
  wctx.fill();

  // Waveform stroke with horizontal gradient
  const wg = wctx.createLinearGradient(0, 0, wc.width, 0);
  if (isSpeaking) {
    wg.addColorStop(0, "rgba(217,119,6,0.65)");
    wg.addColorStop(0.5, "rgba(245,158,11,1)");
    wg.addColorStop(1, "rgba(217,119,6,0.65)");
  } else {
    wg.addColorStop(0, "rgba(100,140,190,0.45)");
    wg.addColorStop(0.5, "rgba(120,160,210,0.75)");
    wg.addColorStop(1, "rgba(100,140,190,0.45)");
  }
  wctx.beginPath();
  for (let i = 0; i < wave.length; i++) {
    const x = (i / wave.length) * wc.width,
      y = midY + ((wave[i] - 128) / 128) * (wc.height * 0.42);
    i ? wctx.lineTo(x, y) : wctx.moveTo(x, y);
  }
  wctx.strokeStyle = wg;
  wctx.lineWidth = 1.8 * devicePixelRatio;
  wctx.lineJoin = "round";
  wctx.stroke();

  // ── Spectrum canvas ────────────────────────────────────────────────────────
  const sc = $("liveSpectrum"),
    sctx = sc.getContext("2d");
  const spectrumWidth = Math.max(1, Math.round(sc.clientWidth * ratio));
  if (sc.width !== spectrumWidth) sc.width = spectrumWidth;
  if (sc.height !== 60 * ratio) sc.height = 60 * ratio;
  sctx.clearRect(0, 0, sc.width, sc.height);

  // Shared vertical gradient for all bars
  const sg = sctx.createLinearGradient(0, 0, 0, sc.height);
  if (isSpeaking) {
    sg.addColorStop(0, "rgba(251,191,36,0.88)");
    sg.addColorStop(0.55, "rgba(245,158,11,0.92)");
    sg.addColorStop(1, "rgba(180,83,9,0.98)");
  } else {
    sg.addColorStop(0, "rgba(160,200,235,0.55)");
    sg.addColorStop(1, "rgba(90,130,180,0.78)");
  }
  sctx.fillStyle = sg;

  const nbars = 48,
    bw = sc.width / nbars;
  for (let i = 0; i < nbars; i++) {
    const v = freq[Math.floor((i * freq.length) / nbars)] / 255;
    if (v < 0.02) continue;
    const bh = v * sc.height;
    sctx.fillRect(i * bw + 1, sc.height - bh, Math.max(1, bw - 2), bh);
  }

  state.liveFrame = requestAnimationFrame(drawLive);
}

window.addEventListener("pagehide", () => {
  cleanupRecording();
  clearTimeout(state.toastTimer);
  clearTimeout(resizeTimer);
  if (state.url) URL.revokeObjectURL(state.url);
});

const copy = {
  en: {
    analyzeVoice: "Analyze Voice ↗",
    startLive: "Start Live Analysis ◉",
    workspaceTitle: "A closer listen. A clearer picture.",
    upload: "Upload audio",
    live: "Live microphone",
    dropPrompt: "Drop a voice into the lab",
    analyzeRecording: "Analyze recording ↗",
    enableMic: "Enable microphone ◉",
  },
  hi: {
    analyzeVoice: "आवाज़ का विश्लेषण ↗",
    startLive: "लाइव विश्लेषण ◉",
    workspaceTitle: "ध्यान से सुनें। स्पष्ट समझें।",
    upload: "ऑडियो अपलोड करें",
    live: "लाइव माइक्रोफ़ोन",
    dropPrompt: "यहाँ ऑडियो फ़ाइल जोड़ें",
    analyzeRecording: "रिकॉर्डिंग का विश्लेषण ↗",
    enableMic: "माइक्रोफ़ोन शुरू करें ◉",
  },
  bn: {
    analyzeVoice: "কণ্ঠ বিশ্লেষণ ↗",
    startLive: "লাইভ বিশ্লেষণ ◉",
    workspaceTitle: "মন দিয়ে শুনুন। স্পষ্টভাবে বুঝুন।",
    upload: "অডিও আপলোড করুন",
    live: "লাইভ মাইক্রোফোন",
    dropPrompt: "এখানে অডিও ফাইল যোগ করুন",
    analyzeRecording: "রেকর্ডিং বিশ্লেষণ ↗",
    enableMic: "মাইক্রোফোন চালু করুন ◉",
  },
};

for (const b of document.querySelectorAll("[data-lang]"))
  b.onclick = () => {
    const lang = b.dataset.lang;
    state.currentLang = lang;
    document.documentElement.lang = lang;
    for (const n of document.querySelectorAll("[data-i18n]")) {
      if (
        (n.id === "recordButton" && state.recording) ||
        (n.id === "filePrompt" && state.file)
      )
        continue;
      n.textContent = copy[lang][n.dataset.i18n] || copy.en[n.dataset.i18n];
    }
    for (const n of document.querySelectorAll("[data-lang]"))
      n.setAttribute("aria-pressed", String(n === b));

    notify(
      lang === "en"
        ? "Primary controls set to English."
        : lang === "hi"
          ? "मुख्य नियंत्रण हिंदी में हैं। विश्लेषण का विवरण अंग्रेज़ी में है।"
          : "মূল नियंत्रण বাংলায়। বিশ্লেষণের বিবরণ ইংরেজিতে আছে।",
    );
  };

if ("IntersectionObserver" in window) {
  document.body.classList.add("motion-ready");
  const observer = new IntersectionObserver(
    (entries) => {
      for (const e of entries)
        if (e.isIntersecting) {
          e.target.classList.add("visible");
          observer.unobserve(e.target);
        }
    },
    { threshold: 0.08 },
  );
  for (const group of document.querySelectorAll(".workspace-grid,.pipeline-grid,.prevention-grid")) {
    [...group.children].forEach((n, i) => n.style.setProperty("--reveal-delay", `${Math.min(i % 3, 2) * 70}ms`));
  }
  document.querySelectorAll(".reveal").forEach((n) => observer.observe(n));
  const nav = new IntersectionObserver(
    (entries) => {
      for (const e of entries)
        if (e.isIntersecting)
          for (const a of $("navigation").querySelectorAll("a")) {
            const active = a.hash === "#" + e.target.id;
            a.classList.toggle("active", active);
            if (active) a.setAttribute("aria-current", "location");
            else a.removeAttribute("aria-current");
          }
    },
    { rootMargin: "-15% 0px -60% 0px" },
  );
  document.querySelectorAll("section[id]").forEach((n) => nav.observe(n));
}

window.addEventListener(
  "scroll",
  () =>
    document
      .querySelector(".site-header")
      .classList.toggle("scrolled", scrollY > 15),
  { passive: true },
);
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (state.result) drawSpectrogram(state.result.spectrogram);
  }, 180);
});

health();
samples();
