// Tests for the dependency-free core (tm_core.js). No models or network are needed.
// Run from the repository root with: node --test js/tests/*.test.mjs

import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import path from 'node:path';
import test from 'node:test';

const jsDir = path.join(path.dirname(new URL(import.meta.url).pathname), '..');
const require = createRequire(import.meta.url);
// The demo loads tm_core.js as a classic script; in Node it is a CommonJS module
const core = require(path.join(jsDir, 'tm_core.js'));

class Tensor {
  constructor(type, data, dims) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }
}
const ort = { Tensor };

/** RGBA test image whose channel values encode their position. */
function makeImage(width, height) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      data[i] = x % 256;
      data[i + 1] = y % 256;
      data[i + 2] = (x + y) % 256;
      data[i + 3] = 255 - ((x * y) % 256);
    }
  }
  return { data, width, height };
}

test('processes the whole image up to a 2:1 aspect ratio and the centre square beyond', () => {
  assert.deepEqual(core.processingRegion(640, 480), { left: 0, top: 0, width: 640, height: 480 });
  assert.deepEqual(core.processingRegion(400, 200), { left: 0, top: 0, width: 400, height: 200 });
  assert.deepEqual(core.processingRegion(200, 400), { left: 0, top: 0, width: 200, height: 400 });
  assert.deepEqual(core.processingRegion(401, 200), { left: 100, top: 0, width: 200, height: 200 });
  assert.deepEqual(core.processingRegion(200, 401), { left: 0, top: 100, width: 200, height: 200 });
  assert.deepEqual(core.processingRegion(4032, 1800), { left: 1116, top: 0, width: 1800, height: 1800 });
  // Variant P always uses the centre square
  assert.deepEqual(core.processingRegion(640, 480, true), { left: 80, top: 0, width: 480, height: 480 });
  assert.deepEqual(core.processingRegion(480, 640, true), { left: 0, top: 80, width: 480, height: 480 });
});

test('extracts a region as planar RGB in [0, 1] and ignores alpha', () => {
  const image = makeImage(7, 5);
  const region = { left: 2, top: 1, width: 3, height: 2 };
  const planar = core.rgbaToPlanar(image, region);

  assert.equal(planar.length, 3 * 3 * 2);
  for (let y = 0; y < 2; y++) {
    for (let x = 0; x < 3; x++) {
      const k = y * 3 + x;
      const [ix, iy] = [x + 2, y + 1];
      assert.equal(planar[k], Math.fround(ix / 255));
      assert.equal(planar[k + 6], Math.fround(iy / 255));
      assert.equal(planar[k + 12], Math.fround((ix + iy) / 255));
    }
  }
});

test('decodeBits feeds the resized region to the decoder and thresholds its output', async () => {
  const calls = [];
  const resizer = {
    async run(feeds) {
      calls.push({ model: 'resizer', dims: feeds.X.dims, size: Number(feeds.target_size.data[0]), scales: Array.from(feeds.scales.data) });
      const size = Number(feeds.target_size.data[0]);
      return { Y: new Tensor('float32', new Float32Array(3 * size * size).fill(0.5), [1, 3, size, size]) };
    },
  };
  const decoder = {
    async run(feeds) {
      calls.push({ model: 'decoder', dims: feeds.image.dims });
      return { output: new Tensor('float32', Float32Array.from({ length: 100 }, (_, i) => (i % 3 === 0 ? 0.5 : -0.5)), [1, 100]) };
    },
  };
  const queued = [];
  const run = (session, feeds) => { queued.push(session); return session.run(feeds); };

  const bits = await core.decodeBits({ ort, resizer, run }, decoder, makeImage(600, 200), 224, true);

  assert.deepEqual(bits, Array.from({ length: 100 }, (_, i) => i % 3 === 0));
  assert.deepEqual(calls.map((c) => c.model), ['resizer', 'decoder']);
  assert.deepEqual(calls[0].dims, [1, 3, 200, 200]);
  assert.equal(calls[0].size, 224);
  assert.equal(Math.floor(200 * calls[0].scales[2]), 224);
  assert.deepEqual(calls[1].dims, [1, 3, 224, 224]);
  assert.deepEqual(queued, [resizer, decoder], 'every inference goes through the runner');

  await assert.rejects(core.decodeBits({ ort, resizer: null }, decoder, makeImage(4, 4), 256), /resizer model is not loaded/);
});
