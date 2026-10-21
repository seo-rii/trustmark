// createTrustmark with a fake ONNX Runtime: options, payloads, results, errors, the job
// queue, AbortSignal and close(). No models or network needed. Run from js/node with: npm test

import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

import { PACKET, PAYLOAD, deferred, fakeModelDir, fakeOrt, rgbaImage, tick } from './helpers.mjs';

const require = createRequire(import.meta.url);
const { createTrustmark, capabilities, TrustmarkError } = require('..');
const { decodePacket: decodePacketRaw } = require('../lib/core.js');
const { encodeImage } = require('../lib/image.js');

function setup(t, { fake = {}, ...options } = {}) {
  const { ort, log } = fakeOrt(fake);
  const tm = createTrustmark({ modelDir: fakeModelDir(t, options.variant), ort, ...options });
  t.after(() => tm.close());
  return { tm, log };
}

const PNG_INPUT = encodeImage(rgbaImage(64, 48), { format: 'png' });

test('capabilities come from the JavaScript core', () => {
  const caps = capabilities();
  for (const variant of ['Q', 'P', 'B', 'C']) assert.deepEqual(caps.variants[variant], { encode: true, decode: true }, variant);
  assert.deepEqual(caps.schemas, [
    { name: 'BCH_SUPER', capacity: 40 },
    { name: 'BCH_5', capacity: 61 },
    { name: 'BCH_4', capacity: 68 },
    { name: 'BCH_3', capacity: 75 },
  ]);
});

test('validates the options', (t) => {
  assert.throws(() => createTrustmark({}), { code: 'INVALID_ARGUMENT', message: /modelDir/ });
  const modelDir = fakeModelDir(t);
  assert.throws(() => createTrustmark({ modelDir, variant: 'X' }), { code: 'UNSUPPORTED' });
  assert.throws(() => createTrustmark({ modelDir, schema: 'BCH_9' }), { code: 'UNSUPPORTED' });
  assert.throws(() => createTrustmark({ modelDir, executionProviders: [] }), { code: 'INVALID_ARGUMENT' });
  assert.throws(() => createTrustmark({ modelDir, maxConcurrency: 0 }), { code: 'INVALID_ARGUMENT' });
  assert.throws(() => createTrustmark({ modelDir, modelBaseUrl: 'http://models.example.com/' }), { code: 'INVALID_ARGUMENT', message: /https/ });
  assert.throws(() => createTrustmark({ modelDir, expectedDigests: { 'encoder_Q.onnx': 'abc' } }), { code: 'INVALID_ARGUMENT' });
  assert.ok(createTrustmark({ modelDir, modelBaseUrl: 'http://127.0.0.1:9/' }));
  const error = (() => { try { createTrustmark({}); } catch (e) { return e; } })();
  assert.ok(error instanceof TrustmarkError && error instanceof Error);
});

test('encode builds the Python packet, returns a PNG and the padded payload', async (t) => {
  const { tm, log } = setup(t);
  const result = await tm.encode(PNG_INPUT, PAYLOAD);
  assert.equal(log.packets[0], PACKET);
  assert.equal(result.format, 'png');
  assert.equal(result.mimeType, 'image/png');
  assert.ok(Buffer.isBuffer(result.bytes));
  assert.deepEqual([result.width, result.height, result.variant, result.schema, result.backend, result.strength], [64, 48, 'Q', 'BCH_5', 'cpu', 1]);
  assert.equal(result.payload, PAYLOAD);

  const short = await tm.encode(PNG_INPUT, '101');
  assert.equal(short.payload, '101'.padEnd(61, '0'));
});

test('payloads must be bit strings within the schema capacity', async (t) => {
  const { tm, log } = setup(t, { schema: 'BCH_SUPER' });
  assert.equal(tm.capacity, 40);
  await assert.rejects(tm.encode(PNG_INPUT, 12345), { code: 'INVALID_ARGUMENT', message: /string/ });
  await assert.rejects(tm.encode(PNG_INPUT, 2n ** 60n), { code: 'INVALID_ARGUMENT' });
  await assert.rejects(tm.encode(PNG_INPUT, ''), { code: 'INVALID_ARGUMENT' });
  await assert.rejects(tm.encode(PNG_INPUT, '10201'), { code: 'INVALID_ARGUMENT' });
  await assert.rejects(tm.encode(PNG_INPUT, '1'.repeat(41)), { code: 'INVALID_ARGUMENT', message: /40/ });
  assert.equal(log.runs.length, 0, 'nothing ran for invalid payloads');
  assert.equal((await tm.encode(PNG_INPUT, '1'.repeat(40))).payload, '1'.repeat(40));
});

test('validates the encode options', async (t) => {
  const { tm } = setup(t);
  await assert.rejects(tm.encode(PNG_INPUT, '1', { format: 'webp' }), { code: 'UNSUPPORTED' });
  await assert.rejects(tm.encode(PNG_INPUT, '1', { format: 'jpeg', quality: 0 }), { code: 'INVALID_ARGUMENT' });
  await assert.rejects(tm.encode(PNG_INPUT, '1', { quality: 80 }), { code: 'INVALID_ARGUMENT', message: /JPEG/ });
  await assert.rejects(tm.encode(PNG_INPUT, '1', { strength: 0 }), { code: 'INVALID_ARGUMENT' });
  await assert.rejects(tm.encode(PNG_INPUT, '1', { strength: NaN }), { code: 'INVALID_ARGUMENT' });
  await assert.rejects(tm.encode(PNG_INPUT, '1', { background: [0, 0] }), { code: 'INVALID_ARGUMENT' });
  await assert.rejects(tm.encode(PNG_INPUT, '1', { signal: {} }), { code: 'INVALID_ARGUMENT' });
});

test('JPEG output of a transparent image needs a background', async (t) => {
  const { tm } = setup(t);
  const transparent = rgbaImage(32, 32, { alpha: (x) => (x < 16 ? 0 : 255) });
  await assert.rejects(tm.encode(transparent, '1', { format: 'jpeg' }), { code: 'UNSUPPORTED', message: /background/ });
  const result = await tm.encode(transparent, '1', { format: 'jpeg', quality: 80, background: [255, 255, 255] });
  assert.equal(result.mimeType, 'image/jpeg');
  assert.equal(result.quality, 80);
});

for (const variant of ['P', 'B', 'C']) {
  test(`variant ${variant} embeds and decodes with its own models`, async (t) => {
    const { tm, log } = setup(t, { variant, schema: 'BCH_4' });
    const encoded = await tm.encode(PNG_INPUT, '1101');
    assert.deepEqual([encoded.variant, encoded.schema, encoded.payload], [variant, 'BCH_4', '1101'.padEnd(68, '0')]);
    const decoded = await tm.decode(encoded.bytes);
    assert.deepEqual([decoded.status, decoded.variant, decoded.payload], ['detected', variant, encoded.payload]);
    assert.equal(decoded.softBinding['c2pa.soft-binding'].alg, `com.adobe.trustmark.${variant}`);
    assert.deepEqual(Object.values(tm.status().models).map((m) => m.file), ['resizer.onnx', `encoder_${variant}.onnx`, `decoder_${variant}.onnx`]);
    assert.equal(log.created.length, 3);
  });
}

test('decode reports the payload, schema, backend and soft binding', async (t) => {
  const { tm } = setup(t, { fake: { decoderBits: () => PACKET } });
  const result = await tm.decode(PNG_INPUT);
  assert.equal(result.status, 'detected');
  assert.equal(result.payload, PAYLOAD);
  assert.equal(result.schema, 'BCH_5');
  assert.equal(result.backend, 'cpu');
  assert.equal(result.bitFlips, 0);
  assert.equal(result.error, null);
  assert.deepEqual(result.softBinding, { 'c2pa.soft-binding': { alg: 'com.adobe.trustmark.Q', blocks: [{ scope: {}, value: `1*${PAYLOAD}` }] } });
});

// Raw bits of the variant B decoder for the unwatermarked images/ghost.png: the version bits
// signal BCH_4, which fails, but BCH_3 happens to accept the bits.
const GHOST_BITS = '0110010000011101001111011000101101110010101000100101110100111100010110000010001100010000001011010110';

test('the schema policy decides which schemas count as a detection', () => {
  const decodePacket = (bits, variant, policy) => decodePacketRaw(Array.from(bits, (c) => c === '1'), variant, policy);
  assert.equal(decodePacket(GHOST_BITS, 'B').valid, false, 'signalled schema only, as in Python');
  assert.equal(decodePacket(GHOST_BITS, 'B', 'any').schema, 'BCH_3', 'any schema: a false positive');
  assert.equal(decodePacket(GHOST_BITS, 'B', 'BCH_5').valid, false);

  // A BCH_5 packet whose version bits were corrupted to signal BCH_3
  const corrupted = PACKET.slice(0, 98) + '11';
  assert.equal(decodePacket(corrupted, 'Q').valid, false);
  assert.equal(decodePacket(corrupted, 'Q', 'any').payload, PAYLOAD);
  assert.equal(decodePacket(corrupted, 'Q', 'BCH_5').payload, PAYLOAD);
  assert.equal(decodePacket(PACKET, 'Q', 'BCH_4').valid, false);
  assert.equal(decodePacket(PACKET, 'Q').payload, PAYLOAD);
});

test('decode passes the schema policy through', async (t) => {
  const { tm } = setup(t, { fake: { decoderBits: () => PACKET.slice(0, 98) + '11' } });
  assert.equal((await tm.decode(PNG_INPUT)).status, 'not_detected');
  assert.equal((await tm.decode(PNG_INPUT, { schemaPolicy: 'BCH_5' })).payload, PAYLOAD);
  await assert.rejects(tm.decode(PNG_INPUT, { schemaPolicy: 'BCH_7' }), { code: 'UNSUPPORTED' });
});

test('decode reports errors as status error, never as not_detected', async (t) => {
  const failing = setup(t, { fake: { hook: (role) => { if (role === 'decoder') throw new Error('device lost'); } } });
  const failed = await failing.tm.decode(PNG_INPUT);
  assert.equal(failed.status, 'error');
  assert.equal(failed.error.code, 'INFERENCE_FAILED');
  assert.match(failed.error.message, /device lost/);

  const { tm } = setup(t);
  const corrupt = await tm.decode(Buffer.from('not an image'));
  assert.equal(corrupt.status, 'error');
  assert.equal(corrupt.error.code, 'INVALID_IMAGE');
  await assert.rejects(tm.decode(PNG_INPUT, { signal: 'nope' }), { code: 'INVALID_ARGUMENT' });
});

test('encode rejects instead of returning an image when inference fails', async (t) => {
  const { tm } = setup(t, { fake: { hook: (role) => { if (role === 'encoder') throw new Error('out of memory'); } } });
  await assert.rejects(tm.encode(PNG_INPUT, PAYLOAD), { code: 'INFERENCE_FAILED', message: /out of memory/ });
});

test('falls back to the next execution provider and reports the one in use', async (t) => {
  const { tm, log } = setup(t, { executionProviders: ['cuda', 'cpu'], fake: { failProviders: ['cuda'] } });
  const result = await tm.encode(PNG_INPUT, PAYLOAD);
  assert.equal(result.backend, 'cpu');
  const status = tm.status().models;
  assert.deepEqual(status.encoder.requestedProviders, ['cuda', 'cpu']);
  assert.equal(status.encoder.backend, 'cpu');
  assert.equal(status.encoder.fallbacks[0].provider, 'cuda');
  assert.match(status.encoder.fallbacks[0].error, /not available/);
  assert.deepEqual(status.resizer.requestedProviders, ['cpu'], 'the resizer always runs on the CPU');
  assert.deepEqual(log.created.map((c) => `${c.role}:${c.provider}`), ['encoder:cpu'], 'embedding does not need the resizer');
});

test('a provider that is not available fails clearly when there is no fallback', async (t) => {
  const { tm } = setup(t, { executionProviders: [{ name: 'cuda', deviceId: 0 }], fake: { failProviders: ['cuda'] } });
  await assert.rejects(tm.encode(PNG_INPUT, PAYLOAD), { code: 'MODEL_LOAD_FAILED', message: /cuda/ });
  const decoded = await tm.decode(PNG_INPUT);
  assert.equal(decoded.status, 'error');
  assert.equal(decoded.error.code, 'MODEL_LOAD_FAILED');
  assert.equal(tm.status().models.decoder.status, 'error');
});

test('sessions are created once and shared by concurrent calls', async (t) => {
  const { tm, log } = setup(t, { maxConcurrency: 4 });
  await Promise.all([tm.encode(PNG_INPUT, '1'), tm.encode(PNG_INPUT, '0'), tm.decode(PNG_INPUT), tm.decode(PNG_INPUT)]);
  assert.deepEqual(log.created.map((c) => c.role).sort(), ['decoder', 'encoder', 'resizer']);
  const status = await tm.ready();
  assert.equal(status.models.encoder.status, 'ready');
  assert.equal(log.created.length, 3);
});

test('runs at most maxConcurrency jobs, in FIFO order', async (t) => {
  const gates = [];
  const started = [];
  const hook = async (role, feeds) => {
    if (role !== 'encoder') return;
    const packet = Array.from(feeds['onnx::Gemm_1'].data).join('');
    started.push(packet.slice(0, 3));
    const gate = deferred();
    gates.push(gate);
    await gate.promise;
  };
  const { tm, log } = setup(t, { maxConcurrency: 2, fake: { hook } });
  await tm.ready();
  const jobs = ['100', '010', '001', '110', '011'].map((p) => tm.encode(PNG_INPUT, p));
  for (let released = 0; released < jobs.length; released++) {
    for (let i = 0; i < 200 && gates.length < Math.min(jobs.length, released + 2); i++) await tick();
    await tick();
    assert.equal(gates.length - released, Math.min(2, jobs.length - released), 'two jobs run while more are waiting');
    gates[released].resolve();
  }
  await Promise.all(jobs);
  assert.deepEqual(started, ['100', '010', '001', '110', '011']);
  assert.ok(log.maxActive <= 2, `at most 2 inferences at once (saw ${log.maxActive})`);
});

test('rejects new jobs when maxQueue jobs are waiting', async (t) => {
  const gate = deferred();
  const { tm } = setup(t, { maxConcurrency: 1, maxQueue: 1, fake: { hook: () => gate.promise } });
  const first = tm.encode(PNG_INPUT, '1');
  const second = tm.encode(PNG_INPUT, '1');
  await assert.rejects(tm.encode(PNG_INPUT, '1'), { code: 'QUEUE_FULL' });
  await assert.rejects(tm.decode(PNG_INPUT), { code: 'QUEUE_FULL' }, 'decode rejects too: the image was not checked');
  gate.resolve();
  await Promise.all([first, second]);
});

test('a job aborted while waiting never runs', async (t) => {
  const gate = deferred();
  const { tm, log } = setup(t, { fake: { hook: () => gate.promise } });
  await tm.ready();
  const first = tm.encode(PNG_INPUT, '1');
  const controller = new AbortController();
  const second = tm.encode(PNG_INPUT, '0', { signal: controller.signal });
  const third = tm.decode(PNG_INPUT, { signal: controller.signal });
  controller.abort();
  await assert.rejects(second, { name: 'AbortError' });
  await assert.rejects(third, { name: 'AbortError' });
  gate.resolve();
  await first;
  assert.deepEqual(log.packets.map((p) => p[0]), ['1'], 'only the first job reached the encoder');
});

test('a running job stops at the next step after an abort', async (t) => {
  const gate = deferred();
  const { tm, log } = setup(t, { fake: { hook: (role) => (role === 'resizer' ? gate.promise : undefined) } });
  const controller = new AbortController();
  const job = tm.decode(PNG_INPUT, { signal: controller.signal });
  for (let i = 0; i < 50 && !log.runs.includes('resizer'); i++) await tick();
  controller.abort(new Error('caller gave up'));
  gate.resolve();
  await assert.rejects(job, { message: 'caller gave up' });
  assert.ok(!log.runs.includes('decoder'), 'the decoder never ran');

  const encodeGate = deferred();
  const encoding = setup(t, { fake: { hook: (role) => (role === 'encoder' ? encodeGate.promise : undefined) } });
  const stopped = new AbortController();
  const encodeJob = encoding.tm.encode(PNG_INPUT, '1', { signal: stopped.signal });
  for (let i = 0; i < 50 && !encoding.log.runs.includes('encoder'); i++) await tick();
  stopped.abort(new Error('caller gave up'));
  encodeGate.resolve();
  await assert.rejects(encodeJob, { message: 'caller gave up' }, 'the result of an inference that ends after the abort is dropped');
});

test('an already aborted signal and AbortSignal.timeout reject', async (t) => {
  const { tm } = setup(t, { fake: { hook: () => new Promise((resolve) => setTimeout(resolve, 200)) } });
  await assert.rejects(tm.encode(PNG_INPUT, '1', { signal: AbortSignal.abort() }), { name: 'AbortError' });
  await assert.rejects(tm.decode(PNG_INPUT, { signal: AbortSignal.timeout(20) }), { name: 'TimeoutError' });
});

test('close() rejects waiting jobs, lets running ones finish and releases the sessions', async (t) => {
  const gate = deferred();
  const { tm, log } = setup(t, { fake: { hook: (role) => (role === 'encoder' ? gate.promise : undefined) } });
  await tm.ready();
  const running = tm.encode(PNG_INPUT, '1');
  for (let i = 0; i < 50 && !log.runs.includes('encoder'); i++) await tick();
  const waiting = tm.encode(PNG_INPUT, '0');
  const closed = tm.close();
  await assert.rejects(waiting, { code: 'CLOSED' });
  assert.deepEqual(log.released, [], 'sessions stay alive while a job runs');
  gate.resolve();
  assert.equal((await running).payload[0], '1');
  await closed;
  assert.deepEqual(log.released.sort(), ['decoder', 'encoder', 'resizer']);
  assert.equal(tm.status().closed, true);
  await assert.rejects(tm.encode(PNG_INPUT, '1'), { code: 'CLOSED' });
  await assert.rejects(tm.decode(PNG_INPUT), { code: 'CLOSED' });
  await assert.rejects(tm.ready(), { code: 'CLOSED' });
  assert.equal(tm.close(), closed, 'close() is idempotent');
});
