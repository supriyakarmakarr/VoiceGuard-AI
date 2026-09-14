// Exercise the existing UI functions with actual API reports plus live retry behavior.
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const source = fs.readFileSync('script.js', 'utf8');
const elements = new Map();
function element(tag = 'div', cls = '', text = '') {
  return { tag, className: cls, textContent: text, children: [], dataset: {}, attrs: {},
    style: { setProperty() {} }, append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = items; },
    setAttribute(k, v) { this.attrs[k] = v; }, getAttribute(k) { return this.attrs[k]; } };
}
const $ = id => { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); };
const context = { $, el: element, palette: ['red', 'green', 'blue', 'orange'], fmt: String,
  selectSpeaker() {}, playSegment() {} };
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf('function renderTimeline('), source.indexOf('// Browser transcription language'))+
  source.slice(source.indexOf('function renderSpeakerOverview('), source.indexOf('function selectSpeaker(')), context);
for (const names of ['A', 'AB', 'ABC', 'ABCD', 'ABA']) {
  const result = JSON.parse(fs.readFileSync(`.cache/speaker-validation/${names}-report.json`, 'utf8'));
  context.renderTimeline(result);
  const expected = new Set(names).size;
  assert.equal($('speakerLanes').children.length, expected);
  assert.equal($('speakerOverview').children.length, expected);
  assert($('speakerCountTitle').textContent.startsWith(`${expected} distinct voice`));
  for (const lane of $('speakerLanes').children) assert(lane.children[1].children.length > 0);
}

async function live() {
  const calls = [];
  let fail = true;
  Object.assign(context, { Blob, FormData, AbortController, setTimeout, clearTimeout, console,
    state: { recording: true, recordToken: 1, context: { sampleRate: 16000 }, pending: [new Float32Array(16000*10)],
      liveSession: 'session', liveChunkIndex: 0, liveSentSamples: 0, transcript: '' },
    pcmBlob: async parts => new Blob([new Uint8Array(parts.reduce((n,p) => n+p.length, 0)*2)]),
    langText: () => 'Unknown',
    api: async (url, options) => {
      calls.push(options.body);
      if (fail) { fail = false; throw Error('Simulated response loss'); }
      return { analysis: { risk_score: 35, risk_level: 'MEDIUM' }, diarization: { num_speakers: 2 } };
    } });
  vm.runInContext(source.slice(source.indexOf('async function streamWindow('), source.indexOf('async function cleanupRecording(')), context);
  await context.streamWindow();
  assert.equal(context.state.pending[0].length, 32000); // excess queued, not discarded
  await context.streamWindow();
  assert.equal(calls[0].get('file').size, calls[1].get('file').size);
  assert.equal(calls[0].get('chunk_index'), calls[1].get('chunk_index'));
  assert.equal(calls[1].get('session_id'), 'session');
  assert.equal(context.state.liveChunkIndex, 1);
  assert.equal($('liveStatus').textContent, '● STREAMING · 2 SPEAKERS ESTIMATED');
  await context.streamWindow();
  assert.equal(calls[2].get('chunk_start_sec'), '8');
  assert.equal(context.state.pending.length, 0);
  assert.equal(context.state.liveSentSamples, 160000);
  console.log('PASS: Actual 1/2/3/4/returning speaker reports render counts, cards and lanes; live queue, retries and count display.');
}
live().catch(error => { console.error(error); process.exitCode = 1; });
