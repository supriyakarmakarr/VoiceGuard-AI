// Deterministic DSP transport test. This fixture is never included in product results.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
let Processor;const messages=[];
const context={AudioWorkletProcessor:class{constructor(){this.port={postMessage:(buffer)=>messages.push(new Float32Array(buffer).slice())};}},Float32Array,registerProcessor:(name,klass)=>{assert.equal(name,'voiceguard-recorder');Processor=klass;}};
vm.runInNewContext(fs.readFileSync('recorder-worklet.js','utf8'),context);
const p=new Processor();for(let i=0;i<16;i++)assert.equal(p.process([[new Float32Array(128).fill(.6),new Float32Array(128).fill(.2)]]),true);
assert.equal(messages.length,1);assert.equal(messages[0].length,2048);assert(Math.abs(messages[0][40]-.4)<1e-6);assert.equal(p.index,0);
assert.equal(p.process([]),true);assert.equal(messages.length,1);
console.log('PASS: AudioWorklet stereo-to-mono capture, PCM block length, buffer reset and empty input.');
