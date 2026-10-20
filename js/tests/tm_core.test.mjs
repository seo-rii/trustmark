// Tests for the dependency-free core (tm_core.js). No models or network are needed.
// Run from the repository root with: node --test js/tests/*.test.mjs

import assert from 'node:assert/strict';
import crypto from 'node:crypto';
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

// torch.nn.functional.interpolate(x, size=(outH, outW), mode='bilinear') for
// x[i] = (((i * 37) % 65) - 32) / 64 (values exactly representable in float32)
const BILINEAR_GOLDEN = [
  { channels: 1, inH: 3, inW: 4, outH: 5, outW: 7, expected: [-0.5, -0.2935267, 0.03683038, -0.1406251, -0.3180802, 0.01227688, 0.21875, -0.3875, -0.1810267, 0.1493304, -0.02812504, -0.2055802, 0.1247769, 0.33125, -0.21875, -0.01227675, 0.3180804, 0.140625, -0.03683024, 0.2935269, 0.5, -0.04999996, -0.06116073, -0.07901796, 0.004687483, 0.08839288, 0.07053566, 0.05937489, 0.0625, -0.09375002, -0.34375, -0.08593744, 0.1718749, -0.07812507, -0.234375] },
  { channels: 1, inH: 6, inW: 5, outH: 4, outW: 3, expected: [-0.1770833, -0.1445312, -0.1119791, 0.09635417, 0.3828125, 0.1614583, -0.05338541, 0.1484375, 0.09635407, -0.03385418, -0.0859375, 0.1158855] },
  { channels: 2, inH: 2, inW: 3, outH: 3, outW: 4, expected: [-0.5, -0.1386719, -0.0859375, -0.359375, -0.140625, -0.09667969, -0.04394531, 0, 0.21875, -0.05468749, -0.001953114, 0.359375, -0.078125, 0.2832031, 0.3359375, 0.0625, -0.2265625, 0.1347656, 0.1875, -0.0859375, -0.375, -0.01367188, 0.03906249, -0.234375] },
];

test('bilinear resize matches torch interpolate (align_corners=False)', () => {
  for (const g of BILINEAR_GOLDEN) {
    const src = Float32Array.from({ length: g.channels * g.inH * g.inW }, (_, i) => (((i * 37) % 65) - 32) / 64);
    const out = core.bilinearResize(src, g.channels, g.inH, g.inW, g.outH, g.outW);
    assert.equal(out.length, g.expected.length);
    out.forEach((v, i) => assert.ok(Math.abs(v - g.expected[i]) < 1e-6, `${g.inH}x${g.inW} -> ${g.outH}x${g.outW} [${i}]: ${v} vs ${g.expected[i]}`));
  }
});

// Output of TrustMark.put_the_image_after_processing in the Python implementation for the
// cover and residual below (SHA-256 of the RGB bytes)
const PASTE_GOLDEN = [
  { width: 230, height: 210, strength: 1.0, region: [0, 0, 230, 210], sha256: '54e6555c0dbd368fe368cab304cccb6fc769a432563fb88ab754927a8d1860e8' },
  { width: 520, height: 250, strength: 1.0, region: [135, 0, 250, 250], sha256: '1ad34c63770e51771a21977aad9dc3bc17e6033a48392cba6fdc13465489967d' },
  { width: 40, height: 30, strength: 1.0, region: [0, 0, 40, 30], sha256: '50f7b80ae0f4edf8bc61b7cc4a08e0fbf004e41c3af7376632ab788a6684d65f' },
  { width: 230, height: 210, strength: 0.5, region: [0, 0, 230, 210], sha256: '3116d8c2fa8729c93b4e37a78aa0f45a85145b5af188831229df7e8780b79146' },
];

test('adds the residual and feathers the edges exactly like the Python implementation', () => {
  for (const g of PASTE_GOLDEN) {
    const { width, height } = g;
    const data = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        for (let c = 0; c < 3; c++) data[(y * width + x) * 4 + c] = (x * 7 + y * 13 + c * 50) % 256;
        data[(y * width + x) * 4 + 3] = (x + 3 * y) % 256;
      }
    }
    const image = { data, width, height };
    const region = core.processingRegion(width, height);
    assert.deepEqual([region.left, region.top, region.width, region.height], g.region);

    const plane = region.width * region.height;
    const residual = new Float32Array(3 * plane);
    for (let c = 0; c < 3; c++) {
      for (let ry = 0; ry < region.height; ry++) {
        for (let rx = 0; rx < region.width; rx++) {
          residual[c * plane + ry * region.width + rx] = (((rx * 31 + ry * 17 + c * 11) % 41) - 20) / 256;
        }
      }
    }

    const out = core.applyResidual(image, region, residual, g.strength);
    const rgb = Buffer.alloc(width * height * 3);
    for (let p = 0; p < width * height; p++) {
      for (let c = 0; c < 3; c++) rgb[p * 3 + c] = out[p * 4 + c];
      assert.equal(out[p * 4 + 3], data[p * 4 + 3], 'alpha is preserved');
    }
    assert.equal(crypto.createHash('sha256').update(rgb).digest('hex'), g.sha256, `${width}x${height} strength ${g.strength}`);
    assert.notStrictEqual(out, data, 'the input is not modified in place');
  }
});

test('feather width follows the Python rule', () => {
  assert.equal(core.featherSize(40, 30), 1);
  assert.equal(core.featherSize(230, 210), 2);
  assert.equal(core.featherSize(4032, 2268), 22);
  assert.equal(core.featherSize(9000, 6000), 50);
});

test('encodeWatermark runs the encoder in [-1, 1] and keeps alpha', async () => {
  const feedsSeen = {};
  const resizer = {
    async run(feeds) {
      feedsSeen.resizer = feeds;
      const size = Number(feeds.target_size.data[0]);
      return { Y: new Tensor('float32', new Float32Array(3 * size * size).fill(0.25), [1, 3, size, size]) };
    },
  };
  // An encoder that adds +0.1 to red and -0.1 to blue: after mean removal the residual is 0
  const encoder = {
    async run(feeds) {
      feedsSeen.encoder = feeds;
      const input = feeds['onnx::Concat_0'].data;
      const plane = 256 * 256;
      const out = Float32Array.from(input, (v, i) => v + (i < plane ? 0.1 : i >= 2 * plane ? -0.1 : 0));
      return { image: new Tensor('float32', out, [1, 3, 256, 256]) };
    },
  };
  const image = makeImage(300, 160);
  const packet = Array.from({ length: 100 }, (_, i) => i % 2);

  const result = await core.encodeWatermark({ ort, resizer }, encoder, image, packet);

  assert.deepEqual(feedsSeen.resizer.X.dims, [1, 3, 160, 300]);
  assert.deepEqual(feedsSeen.encoder['onnx::Concat_0'].dims, [1, 3, 256, 256]);
  assert.ok(feedsSeen.encoder['onnx::Concat_0'].data.every((v) => v === -0.5), 'cover is scaled to [-1, 1]');
  assert.deepEqual(feedsSeen.encoder['onnx::Gemm_1'].dims, [1, 100]);
  assert.deepEqual(Array.from(feedsSeen.encoder['onnx::Gemm_1'].data), packet);

  assert.equal(result.width, 300);
  assert.equal(result.height, 160);
  // A zero residual reproduces the cover up to the truncation the Python implementation does
  const zero = core.applyResidual(image, core.processingRegion(300, 160), new Float32Array(3 * 300 * 160), 1.0);
  assert.deepEqual(result.data, zero);
  for (let i = 0; i < image.data.length; i++) {
    if (i % 4 === 3) assert.equal(result.data[i], image.data[i]);
    else assert.ok(image.data[i] - result.data[i] <= 1 && image.data[i] - result.data[i] >= 0);
  }
});

test('encodeWatermark rejects invalid input and missing models', async () => {
  const session = { run: async () => assert.fail('no model should run') };
  const runtime = { ort, resizer: session };
  const image = makeImage(8, 8);
  const packet = new Array(100).fill(0);

  await assert.rejects(core.encodeWatermark(runtime, null, image, packet), /encoder model is not loaded/);
  await assert.rejects(core.encodeWatermark({ ort, resizer: null }, session, image, packet), /resizer model is not loaded/);
  await assert.rejects(core.encodeWatermark(runtime, session, image, packet.slice(1)), /100 bits/);
  await assert.rejects(core.encodeWatermark(runtime, session, image, [...packet.slice(1), 2]), /100 bits/);
  await assert.rejects(core.encodeWatermark(runtime, session, { data: new Uint8Array(10), width: 8, height: 8 }, packet), /RGBA data/);
  await assert.rejects(core.encodeWatermark(runtime, session, image, packet, { strength: -1 }), /strength/);
});
