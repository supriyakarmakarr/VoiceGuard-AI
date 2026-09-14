const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const source = fs.readFileSync('script.js', 'utf8');
function resolve(href, configured) {
  const context = {location: new URL(href), window: {VOICEGUARD_CONFIG: {API_URL: configured}}, URL};
  vm.createContext(context);
  vm.runInContext(source.slice(source.indexOf('const API_BASE'), source.indexOf('const stages')), context);
  return vm.runInContext('API_BASE', context);
}
assert.equal(resolve('https://supriyakarmakarr.github.io/VoiceGuard-AI/', 'https://api.example.org/'), 'https://api.example.org');
assert.equal(resolve('https://frontend.example.org:8443/', 'https://api.example.org'), 'https://api.example.org');
assert.equal(resolve('http://localhost:5500/', ''), 'http://127.0.0.1:8000');
assert.equal(resolve('http://localhost:8000/', ''), '');
assert.equal(resolve('https://api.example.org/', ''), '');
console.log('PASS production configuration and local/same-origin routing');
// Exercise request URL validation and error handling; responses here are transport fixtures.
const source2 = fs.readFileSync('script.js', 'utf8');
async function requests() {
  let calls = [];
  const context = {location: new URL('https://supriyakarmakarr.github.io/VoiceGuard-AI/'),
    window: {VOICEGUARD_CONFIG: {API_URL: 'https://api.example.org'}}, URL,
    AbortController, setTimeout, clearTimeout, state: {key: 'test-key'},
    fetch: async (url, options) => { calls.push([url, options]); return new Response('{"status":"ready"}', {headers: {'Content-Type': 'application/json'}}); }, Response};
  vm.createContext(context);
  vm.runInContext(source2.slice(source2.indexOf('const API_BASE'), source2.indexOf('const stages')) +
    source2.slice(source2.indexOf('async function api('), source2.indexOf('async function health(')), context);
  assert.equal((await context.api('/api/health')).status, 'ready');
  assert.equal(calls[0][0], 'https://api.example.org/api/health');
  assert.equal(calls[0][1].headers['X-API-Key'], 'test-key');
  context.fetch = async () => new Response('<html>404</html>', {status:404});
  await assert.rejects(context.api('/api/analyze'), /non-JSON.*404/);
  context.fetch = async () => new Response('{"detail":"Enter a valid API key"}', {status:401});
  await assert.rejects(context.api('/api/jobs'), /valid API key/);
  context.fetch = async () => {throw new TypeError('Network error');};
  await assert.rejects(context.api('/api/jobs'), /CORS/);
  for (const value of ['', 'http://localhost:8000', 'http://api.example.org']) {
    const ctx={location:context.location, window:{VOICEGUARD_CONFIG:{API_URL:value}}, URL};
    vm.createContext(ctx);
    vm.runInContext(source2.slice(source2.indexOf('const API_BASE'), source2.indexOf('const stages')),ctx);
    assert.throws(()=>ctx.apiUrl('/api/health'), /API_URL/);
  }
  console.log('PASS request routing, auth, non-JSON, network and production configuration errors');
}
requests().catch(error => {console.error(error);process.exitCode=1;});
