// End-to-end tests with the real ONNX models on onnxruntime-node (CPU): embed and read back
// exact payloads on the repository images, every schema and variant, JPEG output, concurrency,
// abort and the integration example.
//
// The models are not downloaded by the tests. Put resizer.onnx, encoder_Q.onnx and
// decoder_Q.onnx, and for the P, B and C tests the encoder_<V>.onnx and decoder_<V>.onnx files
// (from https://cai-watermark.adobe.net/watermarking/trustmark-models/) in a directory and run
// from js/node:
//   TRUSTMARK_MODEL_DIR=/path/to/models npm test
// Tests whose models are missing are reported as skipped.

import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import test from 'node:test';
import { promisify } from 'node:util';

import { IMAGES_DIR, MODEL_DIR, NODE_DIR, modelSkipReason, tempDir } from './helpers.mjs';

const require = createRequire(import.meta.url);
const skip = modelSkipReason();
const { createTrustmark } = require('..');
const { decodeImage } = require('../lib/image.js');

let shared = null;
function instance(t) {
  if (!shared) shared = createTrustmark({ modelDir: MODEL_DIR, maxConcurrency: 2 });
  return shared;
}
test.after(() => shared && shared.close());

function rgba(bytes) {
  return decodeImage(bytes, { maxPixels: 1e8 }).image;
}

function psnr(a, b) {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < a.data.length; i++) {
    if (i % 4 === 3) continue;
    sum += (a.data[i] - b.data[i]) ** 2;
    count++;
  }
  return 10 * Math.log10((255 * 255) / (sum / count));
}

/** A wide (8:3) crop of ripley.jpg, so that only the centre square is watermarked. */
function wideImage() {
  const full = rgba(fs.readFileSync(path.join(IMAGES_DIR, 'ripley.jpg')));
  const width = 2400;
  const height = 900;
  const left = 800;
  const top = 600;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    const start = ((top + y) * full.width + left) * 4;
    data.set(full.data.subarray(start, start + width * 4), y * width * 4);
  }
  return { data, width, height };
}

const CASES = [
  { name: 'ghost.png (RGBA)', input: () => fs.readFileSync(path.join(IMAGES_DIR, 'ghost.png')), payload: '1101010001000100000000111101010000101101000100010011001100111' },
  { name: 'ufo_240.jpg', input: () => fs.readFileSync(path.join(IMAGES_DIR, 'ufo_240.jpg')), payload: '1111000000010000110101111101110101110011100100111110110111000' },
  { name: 'ripley.jpg (4032x2268)', input: () => fs.readFileSync(path.join(IMAGES_DIR, 'ripley.jpg')), payload: '0000111100001111000011110000111100001111000011110000111100001' },
  { name: 'wide 2400x900 raw RGBA (centre square)', input: wideImage, payload: '1' },
];

for (const c of CASES) {
  test(`embeds and reads back the exact payload: ${c.name}`, { skip, timeout: 300000 }, async (t) => {
    const tm = instance(t);
    const input = c.input();
    const cover = Buffer.isBuffer(input) ? rgba(input) : input;
    const result = await tm.encode(input, c.payload);
    assert.equal(result.payload, c.payload.padEnd(61, '0'));
    const stego = rgba(result.bytes);
    assert.deepEqual([stego.width, stego.height], [cover.width, cover.height]);
    for (let i = 3; i < cover.data.length; i += 4) {
      if (stego.data[i] !== cover.data[i]) assert.fail(`alpha changed at pixel ${(i - 3) / 4}`);
    }
    const quality = psnr(cover, stego);
    assert.ok(quality > 30, `PSNR ${quality.toFixed(2)} dB`);

    const decoded = await tm.decode(result.bytes);
    assert.equal(decoded.status, 'detected');
    assert.equal(decoded.payload, result.payload);
    assert.equal(decoded.schema, 'BCH_5');
    t.diagnostic(`PSNR ${quality.toFixed(2)} dB, ${decoded.bitFlips} corrected bits`);

    const original = await tm.decode(input);
    assert.equal(original.status, 'not_detected', 'the original has no watermark');
  });
}

for (const schema of ['BCH_SUPER', 'BCH_4', 'BCH_3']) {
  test(`embeds and reads back every schema: ${schema}`, { skip, timeout: 120000 }, async (t) => {
    const tm = createTrustmark({ modelDir: MODEL_DIR, schema });
    t.after(() => tm.close());
    const payload = '10'.repeat(tm.capacity / 2 | 0).padEnd(tm.capacity, '1');
    const result = await tm.encode(fs.readFileSync(path.join(IMAGES_DIR, 'ufo_240.jpg')), payload);
    const decoded = await tm.decode(result.bytes);
    assert.deepEqual([decoded.status, decoded.schema, decoded.payload], ['detected', schema, payload]);
  });
}

for (const variant of ['P', 'B', 'C']) {
  test(`variant ${variant} embeds and reads back the exact payload`, { skip: modelSkipReason(variant), timeout: 300000 }, async (t) => {
    const tm = createTrustmark({ modelDir: MODEL_DIR, variant });
    t.after(() => tm.close());
    for (const [file, payload] of [['ghost.png', '1100101011110000110010101111000011001010111100001100101011110'], ['ufo_240.jpg', '0110100110010110011010011001011001101001100101100110100110010']]) {
      const input = fs.readFileSync(path.join(IMAGES_DIR, file));
      const result = await tm.encode(input, payload);
      const decoded = await tm.decode(result.bytes);
      assert.deepEqual([decoded.status, decoded.variant, decoded.schema, decoded.payload], ['detected', variant, 'BCH_5', payload], file);
      assert.equal((await tm.decode(input)).status, 'not_detected', `${file} has no watermark`);
      t.diagnostic(`${file}: PSNR ${psnr(rgba(input), rgba(result.bytes)).toFixed(2)} dB, ${decoded.bitFlips} corrected bits`);
    }
  });
}

test('JPEG output at quality 90 still carries the exact payload', { skip, timeout: 120000 }, async (t) => {
  const tm = instance(t);
  const payload = '1010011100001111010101010000111100110011110000111100001111000';
  const result = await tm.encode(fs.readFileSync(path.join(IMAGES_DIR, 'ufo_240.jpg')), payload, { format: 'jpeg', quality: 90 });
  assert.equal(result.mimeType, 'image/jpeg');
  assert.deepEqual(Array.from(result.bytes.subarray(0, 3)), [0xff, 0xd8, 0xff]);
  const decoded = await tm.decode(result.bytes);
  assert.equal(decoded.payload, payload);
});

test('concurrent jobs each get their own payload', { skip, timeout: 300000 }, async (t) => {
  const tm = instance(t);
  const input = fs.readFileSync(path.join(IMAGES_DIR, 'ufo_240.jpg'));
  const payloads = ['1', '01', '001', '0001', '00001', '000001'].map((p) => p.padEnd(61, '1'));
  const encoded = await Promise.all(payloads.map((p) => tm.encode(input, p)));
  const decoded = await Promise.all(encoded.map((r) => tm.decode(r.bytes)));
  assert.deepEqual(decoded.map((d) => d.payload), payloads);
});

test('an aborted job rejects and the instance keeps working', { skip, timeout: 300000 }, async (t) => {
  const tm = instance(t);
  const input = fs.readFileSync(path.join(IMAGES_DIR, 'ripley.jpg'));
  await assert.rejects(tm.encode(input, '1', { signal: AbortSignal.timeout(50) }), { name: 'TimeoutError' });
  const result = await tm.encode(fs.readFileSync(path.join(IMAGES_DIR, 'ufo_240.jpg')), '1');
  assert.equal((await tm.decode(result.bytes)).payload, '1'.padEnd(61, '0'));
});

test('the integration example watermarks a generated image', { skip, timeout: 300000 }, async (t) => {
  const dir = tempDir(t);
  const output = path.join(dir, 'out.png');
  const { stdout } = await promisify(execFile)(process.execPath, [
    path.join(NODE_DIR, 'examples/generate-and-watermark.mjs'), path.join(IMAGES_DIR, 'ufo_240.jpg'), output,
  ], { env: { ...process.env, TRUSTMARK_MODEL_DIR: MODEL_DIR } });
  assert.match(stdout, /watermarked/);
  const tm = instance(t);
  assert.equal((await tm.decode(fs.readFileSync(output))).status, 'detected');
});
