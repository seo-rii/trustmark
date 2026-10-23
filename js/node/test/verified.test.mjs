// encodeVerified: success, verification failures (mismatch, not detected, wrong schema,
// decoder error), bounded retries, abort and timeout, with the fake ONNX Runtime; and, when
// TRUSTMARK_MODEL_DIR is set, the real models with PNG, JPEG and a JPEG quality that destroys
// the watermark. Run from js/node with: npm test

import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import test from 'node:test';

import { IMAGES_DIR, MODEL_DIR, NO_WATERMARK_BITS, PACKET, PAYLOAD, fakeModelDir, fakeOrt, modelSkipReason, rgbaImage } from './helpers.mjs';

const require = createRequire(import.meta.url);
const { createTrustmark, encodeVerified, TrustmarkError, WatermarkVerificationError } = require('..');
const { buildPacket, schemaInfo } = require('../lib/core.js');
const { encodeImage } = require('../lib/image.js');

const INPUT = encodeImage(rgbaImage(48, 40), { format: 'png' });
const OTHER_PAYLOAD = '1'.repeat(61);

function setup(t, fake = {}, options = {}) {
  const { ort, log } = fakeOrt(fake);
  const tm = createTrustmark({ modelDir: fakeModelDir(t), ort, ...options });
  t.after(() => tm.close());
  return { tm, log };
}

const packetOf = (payload, schema) => buildPacket(payload, schemaInfo(schema)).packet.join('');

test('returns the verified bytes with their digest and the verification result', async (t) => {
  const { tm } = setup(t);
  const result = await encodeVerified(tm, INPUT, PAYLOAD);
  assert.equal(result.payload, PAYLOAD);
  assert.equal(result.sha256, crypto.createHash('sha256').update(result.bytes).digest('hex'));
  assert.deepEqual(result.verification, {
    payload: PAYLOAD, schema: 'BCH_5', bitFlips: 0, backend: 'cpu',
    attempts: [{ strength: 1, status: 'detected', payload: PAYLOAD, schema: 'BCH_5', bitFlips: 0, error: null, reason: null }],
  });
});

test('a decoder that reads a different payload fails verification', async (t) => {
  const { tm } = setup(t, { decoderBits: () => packetOf(OTHER_PAYLOAD, 'BCH_5') });
  await assert.rejects(encodeVerified(tm, INPUT, PAYLOAD), (error) => {
    assert.ok(error instanceof WatermarkVerificationError && error instanceof TrustmarkError);
    assert.equal(error.code, 'VERIFICATION_FAILED');
    assert.equal(error.reason, 'payload_mismatch');
    assert.equal(error.attempts[0].payload, OTHER_PAYLOAD);
    assert.ok(!('bytes' in error) && !JSON.stringify(error.details).includes('bytes'), 'the error carries no image bytes');
    return true;
  });
});

test('an injected verifier that reads a different payload fails verification', async (t) => {
  const { tm, log } = setup(t);
  const verifier = { variant: 'Q', decode: async () => ({ status: 'detected', payload: OTHER_PAYLOAD, schema: 'BCH_5', bitFlips: 0, backend: 'test' }) };
  await assert.rejects(encodeVerified(tm, INPUT, PAYLOAD, { verifier }), { code: 'VERIFICATION_FAILED', reason: 'payload_mismatch' });
  assert.equal(log.runs.filter((r) => r === 'decoder').length, 0, 'the injected verifier was used');
});

test('a watermark that is not found, or found with another schema, fails verification', async (t) => {
  const missing = setup(t, { decoderBits: () => NO_WATERMARK_BITS });
  await assert.rejects(encodeVerified(missing.tm, INPUT, PAYLOAD), { reason: 'not_detected' });

  // The same bits read back as a BCH_4 packet instead of BCH_5
  const other = setup(t, { decoderBits: () => packetOf(PAYLOAD, 'BCH_4') });
  await assert.rejects(encodeVerified(other.tm, INPUT, PAYLOAD), { reason: 'schema_mismatch' });

  // Corrupted version bits: 'any' would recover the payload, but other decoders would not
  const corrupted = setup(t, { decoderBits: () => PACKET.slice(0, 98) + '11' });
  await assert.rejects(encodeVerified(corrupted.tm, INPUT, PAYLOAD), { reason: 'not_detected' });
});

test('a decoder error fails verification without retrying', async (t) => {
  const { tm, log } = setup(t, { hook: (role) => { if (role === 'decoder') throw new Error('decoder crashed'); } });
  await assert.rejects(encodeVerified(tm, INPUT, PAYLOAD, { retryStrengths: [1.5, 2] }), (error) => {
    assert.equal(error.reason, 'decode_error');
    assert.equal(error.attempts.length, 1);
    assert.equal(error.attempts[0].error.code, 'INFERENCE_FAILED');
    assert.equal(error.cause.code, 'INFERENCE_FAILED');
    return true;
  });
  assert.equal(log.packets.length, 1, 'no retry');
});

test('a missing decoder model is a verification error, not a release', async (t) => {
  const { ort } = fakeOrt();
  const modelDir = fakeModelDir(t);
  fs.rmSync(path.join(modelDir, 'decoder_Q.onnx'));
  const tm = createTrustmark({ modelDir, ort });
  t.after(() => tm.close());
  await assert.rejects(encodeVerified(tm, INPUT, PAYLOAD), (error) => error.reason === 'decode_error' && error.attempts[0].error.code === 'MODEL_NOT_FOUND');
});

test('encode errors are passed through', async (t) => {
  const { tm } = setup(t, { hook: (role) => { if (role === 'encoder') throw new Error('no memory'); } });
  await assert.rejects(encodeVerified(tm, INPUT, PAYLOAD), { code: 'INFERENCE_FAILED' });
  await assert.rejects(encodeVerified(tm, INPUT, 42), { code: 'INVALID_ARGUMENT' });
});

test('retries at the given higher strengths and reports every attempt', async (t) => {
  let decodes = 0;
  const { tm, log } = setup(t, { decoderBits: () => (++decodes < 3 ? NO_WATERMARK_BITS : log.packets[log.packets.length - 1]) });
  const result = await encodeVerified(tm, INPUT, PAYLOAD, { strength: 1.2, retryStrengths: [1.5, 2] });
  assert.equal(result.strength, 2);
  assert.deepEqual(result.verification.attempts.map((a) => [a.strength, a.reason]), [[1.2, 'not_detected'], [1.5, 'not_detected'], [2, null]]);
});

test('retries are bounded by the caller', async (t) => {
  const { tm, log } = setup(t, { decoderBits: () => NO_WATERMARK_BITS });
  await assert.rejects(encodeVerified(tm, INPUT, PAYLOAD), (error) => error.attempts.length === 1);
  await assert.rejects(encodeVerified(tm, INPUT, PAYLOAD, { retryStrengths: [1.5, 2] }), (error) => {
    assert.deepEqual(error.attempts.map((a) => a.strength), [1, 1.5, 2]);
    assert.match(error.message, /strengths tried: 1, 1\.5, 2/);
    return true;
  });
  assert.equal(log.packets.length, 4);
});

test('validates the options', async (t) => {
  const { tm } = setup(t);
  await assert.rejects(encodeVerified(tm, INPUT, PAYLOAD, { retryStrengths: [0.5] }), { code: 'INVALID_ARGUMENT' });
  await assert.rejects(encodeVerified(tm, INPUT, PAYLOAD, { retryStrengths: [2, 1.5] }), { code: 'INVALID_ARGUMENT' });
  await assert.rejects(encodeVerified(tm, INPUT, PAYLOAD, { retryStrengths: 2 }), { code: 'INVALID_ARGUMENT' });
  await assert.rejects(encodeVerified(tm, INPUT, PAYLOAD, { strength: -1 }), { code: 'INVALID_ARGUMENT' });
  await assert.rejects(encodeVerified(tm, INPUT, PAYLOAD, { verifier: {} }), { code: 'INVALID_ARGUMENT' });
  await assert.rejects(encodeVerified(tm, INPUT, PAYLOAD, { verifier: { variant: 'P', decode() {} } }), { code: 'INVALID_ARGUMENT', message: /variant P/ });
  await assert.rejects(encodeVerified({}, INPUT, PAYLOAD), { code: 'INVALID_ARGUMENT' });
});

test('an abort or timeout during verification rejects without bytes', async (t) => {
  const slowDecoder = (role) => (role === 'decoder' ? new Promise((resolve) => setTimeout(resolve, 300)) : undefined);
  const { tm } = setup(t, { hook: slowDecoder });
  await assert.rejects(encodeVerified(tm, INPUT, PAYLOAD, { signal: AbortSignal.timeout(100) }), { name: 'TimeoutError' });
  const controller = new AbortController();
  const job = encodeVerified(tm, INPUT, PAYLOAD, { signal: controller.signal });
  setTimeout(() => controller.abort(), 50);
  await assert.rejects(job, { name: 'AbortError' });
  await assert.rejects(encodeVerified(tm, INPUT, PAYLOAD, { signal: AbortSignal.abort() }), { name: 'AbortError' });
});

const skip = modelSkipReason();
let shared = null;
const instance = () => (shared ??= createTrustmark({ modelDir: MODEL_DIR }));
test.after(() => shared && shared.close());

test('real models: PNG and JPEG q90 outputs verify with the exact payload', { skip, timeout: 300000 }, async () => {
  const input = fs.readFileSync(path.join(IMAGES_DIR, 'ghost.png'));
  const png = await encodeVerified(instance(), input, PAYLOAD);
  assert.equal(png.verification.payload, PAYLOAD);
  const jpeg = await encodeVerified(instance(), fs.readFileSync(path.join(IMAGES_DIR, 'ufo_240.jpg')), PAYLOAD, { format: 'jpeg', quality: 90 });
  assert.equal(jpeg.mimeType, 'image/jpeg');
  assert.equal((await instance().decode(jpeg.bytes)).payload, PAYLOAD, 'the returned bytes carry the payload');
});

test('real models: a JPEG quality that destroys the watermark is not released', { skip, timeout: 300000 }, async () => {
  const input = fs.readFileSync(path.join(IMAGES_DIR, 'ufo_240.jpg'));
  await assert.rejects(encodeVerified(instance(), input, PAYLOAD, { format: 'jpeg', quality: 1, retryStrengths: [1.5] }), (error) => {
    assert.ok(error instanceof WatermarkVerificationError);
    assert.equal(error.attempts.length, 2);
    assert.ok(['not_detected', 'payload_mismatch', 'schema_mismatch'].includes(error.reason), error.reason);
    return true;
  });
});
