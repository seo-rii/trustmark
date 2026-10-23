// Model files: missing models with downloads disabled, digests, size limits and atomic
// downloads from a local HTTP server. Uses the fake ONNX Runtime; no models or internet needed.
// Run from js/node with: npm test

import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import { createRequire } from 'node:module';
import path from 'node:path';
import test from 'node:test';

import { fakeModelDir, fakeOrt, rgbaImage, tempDir } from './helpers.mjs';

const require = createRequire(import.meta.url);
const { createTrustmark } = require('..');
const { encodeImage } = require('../lib/image.js');

const PNG_INPUT = encodeImage(rgbaImage(32, 32), { format: 'png' });
const MODEL_BODIES = { 'resizer.onnx': 'resizer', 'encoder_Q.onnx': 'encoder', 'decoder_Q.onnx': 'decoder' };
const sha256 = (text) => crypto.createHash('sha256').update(text).digest('hex');

/** Local model server. `behaviour(file)` may return { status, body, delayMs, chunks }. */
async function modelServer(t, behaviour = () => ({})) {
  const requests = [];
  const server = http.createServer(async (req, res) => {
    const file = decodeURIComponent(req.url.split('/').pop());
    requests.push(file);
    const plan = { status: MODEL_BODIES[file] ? 200 : 404, body: MODEL_BODIES[file] || 'missing', ...behaviour(file) };
    res.writeHead(plan.status, { 'content-type': 'application/octet-stream' });
    for (const chunk of plan.chunks || [plan.body]) {
      if (plan.delayMs) await new Promise((resolve) => setTimeout(resolve, plan.delayMs));
      if (res.destroyed) return;
      res.write(chunk);
    }
    res.end();
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise((resolve) => { server.closeAllConnections(); server.close(resolve); }));
  return { url: `http://127.0.0.1:${server.address().port}/models/`, requests };
}

function spyFetch() {
  const calls = [];
  const original = globalThis.fetch;
  const spy = (...args) => { calls.push(String(args[0])); return original(...args); };
  return { calls, spy };
}

test('missing models with downloads disabled fail with MODEL_NOT_FOUND and never touch the network', async (t) => {
  const modelDir = tempDir(t);
  const { calls, spy } = spyFetch();
  const original = globalThis.fetch;
  globalThis.fetch = spy;
  t.after(() => { globalThis.fetch = original; });
  const injected = [];
  const { ort } = fakeOrt();
  const tm = createTrustmark({ modelDir, ort, fetch: (url) => { injected.push(url); throw new Error('no network'); } });
  t.after(() => tm.close());

  await assert.rejects(tm.encode(PNG_INPUT, '1'), (error) => {
    assert.equal(error.code, 'MODEL_NOT_FOUND');
    assert.match(error.message, /allowDownload: false/);
    assert.match(error.message, /cai-watermark\.adobe\.net/, 'says where to get the model');
    return true;
  });
  const decoded = await tm.decode(PNG_INPUT);
  assert.equal(decoded.status, 'error', 'a missing decoder is not "no watermark"');
  assert.equal(decoded.error.code, 'MODEL_NOT_FOUND');
  await assert.rejects(tm.ready(), { code: 'MODEL_NOT_FOUND' });
  assert.deepEqual(injected, []);
  assert.deepEqual(calls, []);
  assert.deepEqual(fs.readdirSync(modelDir), [], 'nothing was written');
});

test('local models are checked against expected digests', async (t) => {
  const modelDir = fakeModelDir(t);
  const digests = Object.fromEntries(Object.entries(MODEL_BODIES).map(([file, body]) => [file, sha256(body)]));
  const good = createTrustmark({ modelDir, ort: fakeOrt().ort, expectedDigests: digests, requireDigests: true });
  t.after(() => good.close());
  const status = await good.ready();
  assert.equal(status.models.encoder.verified, true);
  assert.equal(status.models.encoder.sha256, digests['encoder_Q.onnx']);

  const bad = createTrustmark({ modelDir, ort: fakeOrt().ort, expectedDigests: { ...digests, 'decoder_Q.onnx': sha256('tampered') } });
  t.after(() => bad.close());
  const decoded = await bad.decode(PNG_INPUT);
  assert.equal(decoded.status, 'error');
  assert.equal(decoded.error.code, 'MODEL_INTEGRITY');
  assert.match(decoded.error.message, /expected/);
  assert.ok(await bad.encode(PNG_INPUT, '1'), 'models with matching digests still load');

  const strict = createTrustmark({ modelDir, ort: fakeOrt().ort, requireDigests: true, expectedDigests: { 'resizer.onnx': digests['resizer.onnx'] } });
  t.after(() => strict.close());
  await assert.rejects(strict.encode(PNG_INPUT, '1'), { code: 'MODEL_INTEGRITY', message: /requireDigests/ });
});

test('local models larger than maxModelBytes are refused', async (t) => {
  const tm = createTrustmark({ modelDir: fakeModelDir(t), ort: fakeOrt().ort, maxModelBytes: 6 });
  t.after(() => tm.close());
  await assert.rejects(tm.encode(PNG_INPUT, '1'), { code: 'MODEL_INTEGRITY', message: /maxModelBytes/ });
});

test('downloads missing models once, atomically, and verifies them', async (t) => {
  const server = await modelServer(t);
  const modelDir = path.join(tempDir(t), 'nested', 'models');
  const digests = Object.fromEntries(Object.entries(MODEL_BODIES).map(([file, body]) => [file, sha256(body)]));
  const tm = createTrustmark({ modelDir, ort: fakeOrt().ort, allowDownload: true, modelBaseUrl: server.url, expectedDigests: digests, maxConcurrency: 3 });
  t.after(() => tm.close());
  await Promise.all([tm.encode(PNG_INPUT, '1'), tm.encode(PNG_INPUT, '0'), tm.decode(PNG_INPUT)]);
  assert.deepEqual(server.requests.sort(), ['decoder_Q.onnx', 'encoder_Q.onnx', 'resizer.onnx'], 'one request per model');
  assert.deepEqual(fs.readdirSync(modelDir).sort(), ['decoder_Q.onnx', 'encoder_Q.onnx', 'resizer.onnx'], 'no temporary files left');
  assert.equal(fs.readFileSync(path.join(modelDir, 'encoder_Q.onnx'), 'utf8'), 'encoder');
  const status = tm.status().models.encoder;
  assert.equal(status.downloaded, true);
  assert.equal(status.verified, true);

  // A second instance finds the files and does not download again
  const again = createTrustmark({ modelDir, ort: fakeOrt().ort, allowDownload: true, modelBaseUrl: server.url });
  t.after(() => again.close());
  await again.ready();
  assert.equal(server.requests.length, 3);
});

test('a download that fails its digest, size limit or HTTP status leaves no file behind', async (t) => {
  const cases = [
    { name: 'digest', behaviour: (file) => (file === 'encoder_Q.onnx' ? { body: 'evil' } : {}), options: { expectedDigests: { 'encoder_Q.onnx': sha256('encoder') } }, code: 'MODEL_INTEGRITY' },
    { name: 'size', behaviour: (file) => (file === 'encoder_Q.onnx' ? { chunks: ['enc', 'oder', 'x'.repeat(64)] } : {}), options: { maxModelBytes: 32 }, code: 'MODEL_INTEGRITY' },
    { name: 'status', behaviour: (file) => (file === 'encoder_Q.onnx' ? { status: 500, body: 'oops' } : {}), options: {}, code: 'MODEL_DOWNLOAD_FAILED' },
  ];
  for (const c of cases) {
    const server = await modelServer(t, c.behaviour);
    const modelDir = tempDir(t);
    const tm = createTrustmark({ modelDir, ort: fakeOrt().ort, allowDownload: true, modelBaseUrl: server.url, ...c.options });
    await assert.rejects(tm.encode(PNG_INPUT, '1'), { code: c.code }, c.name);
    await tm.close();
    assert.deepEqual(fs.readdirSync(modelDir).filter((f) => f.includes('encoder') || f.endsWith('.part')), [], `${c.name}: no encoder file or temporary file`);
  }
});

test('a failed download is retried by the next call', async (t) => {
  let failures = 1;
  const server = await modelServer(t, (file) => (file === 'decoder_Q.onnx' && failures-- > 0 ? { status: 503 } : {}));
  const tm = createTrustmark({ modelDir: tempDir(t), ort: fakeOrt().ort, allowDownload: true, modelBaseUrl: server.url });
  t.after(() => tm.close());
  assert.equal((await tm.decode(PNG_INPUT)).error.code, 'MODEL_DOWNLOAD_FAILED');
  assert.equal((await tm.decode(PNG_INPUT)).status, 'not_detected');
});

test('close() stops a download in progress and removes the temporary file', async (t) => {
  const server = await modelServer(t, (file) => (file === 'encoder_Q.onnx' ? { chunks: ['en', 'co', 'der'], delayMs: 300 } : {}));
  const modelDir = tempDir(t);
  const tm = createTrustmark({ modelDir, ort: fakeOrt().ort, allowDownload: true, modelBaseUrl: server.url });
  const job = tm.encode(PNG_INPUT, '1');
  for (let i = 0; i < 100 && !server.requests.includes('encoder_Q.onnx'); i++) await new Promise((r) => setTimeout(r, 10));
  const rejected = assert.rejects(job, { code: 'CLOSED' });
  await tm.close();
  await rejected;
  assert.ok(!fs.readdirSync(modelDir).some((f) => f.includes('encoder')), 'no partial encoder file');
});

test('an aborted caller does not cancel a download shared with others', async (t) => {
  const server = await modelServer(t, (file) => (file === 'decoder_Q.onnx' ? { chunks: ['deco', 'der'], delayMs: 100 } : {}));
  const tm = createTrustmark({ modelDir: tempDir(t), ort: fakeOrt().ort, allowDownload: true, modelBaseUrl: server.url, maxConcurrency: 2 });
  t.after(() => tm.close());
  const controller = new AbortController();
  const aborted = tm.decode(PNG_INPUT, { signal: controller.signal });
  const other = tm.decode(PNG_INPUT);
  setTimeout(() => controller.abort(), 50);
  await assert.rejects(aborted, { name: 'AbortError' });
  assert.equal((await other).status, 'not_detected');
});
