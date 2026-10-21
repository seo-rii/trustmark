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

test('decodeBits feeds the resized region to the decoder in [-1, 1] and thresholds its output', async () => {
  const calls = [];
  // Resized pixels covering [0, 1]
  const resized = (size) => Float32Array.from({ length: 3 * size * size }, (_, i) => (i % 256) / 255);
  const resizer = {
    async run(feeds) {
      calls.push({ model: 'resizer', dims: feeds.X.dims, size: Number(feeds.target_size.data[0]), scales: Array.from(feeds.scales.data) });
      const size = Number(feeds.target_size.data[0]);
      return { Y: new Tensor('float32', resized(size), [1, 3, size, size]) };
    },
  };
  const decoder = {
    async run(feeds) {
      calls.push({ model: 'decoder', dims: feeds.image.dims, data: feeds.image.data });
      return { output: new Tensor('float32', Float32Array.from({ length: 100 }, (_, i) => (i % 3 === 0 ? 0.5 : -0.5)), [1, 100]) };
    },
  };
  const queued = [];
  const run = (session, feeds) => { queued.push(session); return session.run(feeds); };

  // Variant P: centre square, 224x224
  const bits = await core.decodeBits({ ort, resizer, run }, decoder, makeImage(600, 200), 'P');

  assert.deepEqual(bits, Array.from({ length: 100 }, (_, i) => i % 3 === 0));
  assert.deepEqual(calls.map((c) => c.model), ['resizer', 'decoder']);
  assert.deepEqual(calls[0].dims, [1, 3, 200, 200]);
  assert.equal(calls[0].size, 224);
  assert.equal(Math.floor(200 * calls[0].scales[2]), 224);
  assert.deepEqual(calls[1].dims, [1, 3, 224, 224]);
  // The decoders take [-1, 1] like the Python and Rust decoders (ToTensor() * 2 - 1)
  const expected = resized(224);
  assert.ok(calls[1].data instanceof Float32Array);
  calls[1].data.forEach((v, i) => assert.equal(v, Math.fround(expected[i] * 2 - 1), `decoder input [${i}]`));
  assert.equal(Math.min(...calls[1].data.subarray(0, 256)), -1);
  assert.equal(Math.max(...calls[1].data.subarray(0, 256)), 1);
  assert.deepEqual(queued, [resizer, decoder], 'every inference goes through the runner');

  await assert.rejects(core.decodeBits({ ort, resizer: null }, decoder, makeImage(4, 4), 'Q'), /resizer model is not loaded/);
  await assert.rejects(core.decodeBits({ ort, resizer, run }, decoder, makeImage(4, 4), 'X'), /Unknown TrustMark variant "X"; expected one of Q, P, B, C/);
});

test('decodeBits uses the resolution and crop of each variant', async () => {
  const seen = [];
  const resizer = {
    async run(feeds) {
      const size = Number(feeds.target_size.data[0]);
      seen.push({ dims: feeds.X.dims, size });
      return { Y: new Tensor('float32', new Float32Array(3 * size * size), [1, 3, size, size]) };
    },
  };
  const decoder = { async run() { return { output: new Tensor('float32', new Float32Array(100), [1, 100]) }; } };
  for (const variant of ['Q', 'B', 'C', 'P']) await core.decodeBits({ ort, resizer }, decoder, makeImage(500, 300), variant);
  assert.deepEqual(seen, [
    { dims: [1, 3, 300, 500], size: 256 },
    { dims: [1, 3, 300, 500], size: 256 },
    { dims: [1, 3, 300, 500], size: 256 },
    { dims: [1, 3, 300, 300], size: 224 },
  ]);
});

test('variant settings follow the Python implementation', () => {
  assert.deepEqual(Object.keys(core.VARIANTS), ['Q', 'P', 'B', 'C']);
  for (const [code, v] of Object.entries(core.VARIANTS)) {
    assert.equal(v.code, code);
    assert.equal(v.encoder, `encoder_${code}`);
    assert.equal(v.decoder, `decoder_${code}`);
    assert.equal(v.encodeResolution, 256);
    assert.equal(v.decodeResolution, code === 'P' ? 224 : 256);
    assert.equal(v.forceSquare, code === 'P');
    assert.equal(v.strengthMultiplier, code === 'P' ? 1.25 : 1.0);
    assert.ok(Object.isFrozen(v));
  }
  assert.equal(core.getVariant('C'), core.VARIANTS.C);
  for (const bad of ['q', 'D', '', undefined, null, 0, 'constructor', 'toString']) {
    assert.throws(() => core.getVariant(bad), /Unknown TrustMark variant/);
  }
});

/** RGBA image with some texture: channel c of pixel (x, y) is (7x + 13y + 50c + 5((x * y) % 23)) % 256. */
function makeTexture(width, height) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      for (let c = 0; c < 3; c++) data[i + c] = (x * 7 + y * 13 + c * 50 + ((x * y) % 23) * 5) % 256;
      data[i + 3] = (x + 3 * y) % 256;
    }
  }
  return { data, width, height };
}

// SHA-256 of the planar RGB bytes of Image.resize((w, h), Image.BILINEAR) in Pillow 12.3 for
// crops of makeTexture images converted to RGB
const PILLOW_GOLDEN = [
  { width: 640, height: 480, region: [0, 0, 640, 480], out: [256, 256], sha256: 'f9936364c4b735136c9cc21923d1675c7bc6e35e41ad88b40fb7d37c41128255' },
  { width: 1001, height: 753, region: [0, 0, 1001, 753], out: [224, 224], sha256: 'a2851863c2d8dbca9840826468c8b463c2baf77091da4efcf0e939655d38db9b' },
  { width: 240, height: 240, region: [0, 0, 240, 240], out: [256, 256], sha256: 'fd23731e5f157153f6f53eb5268d1acec8a1894128447d36ca1c3b8a0788a430' },
  { width: 300, height: 256, region: [0, 0, 300, 256], out: [256, 256], sha256: 'b3deeebce7976fac549cc756fb6af92628fd1f1a78b46fa54b392f7d4983fd94' },
  { width: 256, height: 300, region: [0, 0, 256, 300], out: [256, 256], sha256: '6d3e5a75ee0f743301da2e47163b165dfc53dc27eee1e4d159895376ae3452a2' },
  { width: 520, height: 250, region: [135, 0, 250, 250], out: [256, 256], sha256: '00f9e159bc38d5cbcc5f4e46305c6f7a31fbcd9452fcf58c5b6862f27ff3af32' },
  { width: 901, height: 200, region: [350, 0, 200, 200], out: [224, 224], sha256: '78ea3d095aad710fb981ef47cbccf2e9f71804f4d934709464ab984f4677ccef' },
  { width: 37, height: 23, region: [3, 2, 31, 19], out: [61, 17], sha256: 'c47aa27269555e26eac28b00a6e1e918b4d168c4992a7a8fc10834dfd2e3800d' },
];

test('the bilinear resize is bit-exact with Pillow', () => {
  for (const g of PILLOW_GOLDEN) {
    const [left, top, width, height] = g.region;
    const out = core.pillowBilinearResize(makeTexture(g.width, g.height), { left, top, width, height }, g.out[0], g.out[1]);
    assert.equal(out.length, 3 * g.out[0] * g.out[1]);
    assert.equal(crypto.createHash('sha256').update(out).digest('hex'), g.sha256, `${g.width}x${g.height} ${g.region} -> ${g.out}`);
  }
  // A region that keeps its size is copied
  const image = makeTexture(9, 7);
  const copy = core.pillowBilinearResize(image, { left: 2, top: 1, width: 5, height: 4 }, 5, 4);
  assert.deepEqual(Array.from(copy), Array.from(core.rgbaToPlanar(image, { left: 2, top: 1, width: 5, height: 4 }), (v) => Math.round(v * 255)));
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

/** Compares large byte arrays without the slow diff of assert.deepEqual. */
function assertSameBytes(actual, expected, message) {
  assert.equal(actual.length, expected.length, message);
  const index = actual.findIndex((v, i) => v !== expected[i]);
  assert.equal(index, -1, `${message}: first difference at ${index} (${actual[index]} vs ${expected[index]})`);
}

/** Fake encoder session that returns its input plus `delta(channel, index)`. */
function fakeEncoder(delta, feedsSeen) {
  return {
    async run(feeds) {
      feedsSeen.push(feeds);
      const input = feeds['onnx::Concat_0'].data;
      const plane = 256 * 256;
      const out = Float32Array.from(input, (v, i) => v + delta(Math.floor(i / plane), i % plane));
      return { image: new Tensor('float32', out, [1, 3, 256, 256]) };
    },
  };
}

test('encodeWatermark feeds the Pillow-resized region in [-1, 1] and keeps alpha', async () => {
  const feedsSeen = [];
  // An encoder that returns its input: the residual is 0
  const encoder = fakeEncoder(() => 0, feedsSeen);
  const image = makeTexture(300, 160);
  const packet = Array.from({ length: 100 }, (_, i) => i % 2);

  // The encoder does not need the resizer model
  const result = await core.encodeWatermark({ ort }, encoder, image, packet);

  const cover = feedsSeen[0]['onnx::Concat_0'];
  assert.deepEqual(cover.dims, [1, 3, 256, 256]);
  const resized = core.pillowBilinearResize(image, core.processingRegion(300, 160), 256, 256);
  // transforms.ToTensor() then * 2 - 1, in float32
  cover.data.forEach((v, k) => assert.equal(v, Math.fround(Math.fround(resized[k] / 255) * 2 - 1)));
  assert.deepEqual(feedsSeen[0]['onnx::Gemm_1'].dims, [1, 100]);
  assert.deepEqual(Array.from(feedsSeen[0]['onnx::Gemm_1'].data), packet);

  assert.equal(result.width, 300);
  assert.equal(result.height, 160);
  // A zero residual reproduces the cover up to the truncation the Python implementation does
  const zero = core.applyResidual(image, core.processingRegion(300, 160), new Float32Array(3 * 300 * 160), 1.0);
  assertSameBytes(result.data, zero, 'zero residual');
  for (let i = 0; i < image.data.length; i++) {
    if (i % 4 === 3) assert.equal(result.data[i], image.data[i]);
    else assert.ok(image.data[i] - result.data[i] <= 1 && image.data[i] - result.data[i] >= 0);
  }
});

test('encodeWatermark applies the crop and strength multiplier of the variant', async () => {
  // An encoder that adds a pattern with a different mean per channel
  const delta = (c, k) => ((k % 7) - 3) / 64 * (c + 1) + 0.05 * (c - 1);
  const image = makeTexture(500, 300);
  const packet = new Array(100).fill(1);
  // TrustMark.encode: residual = clamp(stego, -1, 1) - cover in float32, minus its per-channel mean
  const expected = (feeds, region, strength) => {
    const cover = feeds['onnx::Concat_0'].data;
    const plane = 256 * 256;
    const residual = new Float32Array(3 * plane);
    for (let c = 0; c < 3; c++) {
      let sum = 0;
      for (let k = 0; k < plane; k++) {
        const stego = Math.fround(cover[c * plane + k] + delta(c, k));
        residual[c * plane + k] = Math.min(1, Math.max(-1, stego)) - cover[c * plane + k];
        sum += residual[c * plane + k];
      }
      for (let k = 0; k < plane; k++) residual[c * plane + k] -= sum / plane;
    }
    const upscaled = core.bilinearResize(residual, 3, 256, 256, region.height, region.width);
    return core.applyResidual(image, region, upscaled, strength);
  };

  for (const [variant, square, multiplier] of [['Q', false, 1.0], ['B', false, 1.0], ['C', false, 1.0], ['P', true, 1.25]]) {
    const feedsSeen = [];
    const result = await core.encodeWatermark({ ort }, fakeEncoder(delta, feedsSeen), image, packet, { variant, strength: 0.8 });
    const region = core.processingRegion(500, 300, square);
    assert.deepEqual(region, square ? { left: 100, top: 0, width: 300, height: 300 } : { left: 0, top: 0, width: 500, height: 300 });
    assertSameBytes(result.data, expected(feedsSeen[0], region, 0.8 * multiplier), variant);
    // The encoder sees the Pillow resize of the region
    const resized = core.pillowBilinearResize(image, region, 256, 256);
    assert.ok(feedsSeen[0]['onnx::Concat_0'].data.every((v, k) => v === Math.fround(Math.fround(resized[k] / 255) * 2 - 1)), variant);
    // P only changes the centre square
    if (square) for (let x = 0; x < 100; x++) assert.equal(result.data[(150 * 500 + x) * 4], image.data[(150 * 500 + x) * 4]);
  }
});

test('encodeWatermark rejects invalid input and missing models', async () => {
  const session = { run: async () => assert.fail('no model should run') };
  const runtime = { ort, resizer: session };
  const image = makeImage(8, 8);
  const packet = new Array(100).fill(0);

  await assert.rejects(core.encodeWatermark(runtime, null, image, packet), /encoder model is not loaded/);
  await assert.rejects(core.encodeWatermark(runtime, session, image, packet.slice(1)), /100 bits/);
  await assert.rejects(core.encodeWatermark(runtime, session, image, [...packet.slice(1), 2]), /100 bits/);
  await assert.rejects(core.encodeWatermark(runtime, session, { data: new Uint8Array(10), width: 8, height: 8 }, packet), /RGBA data/);
  await assert.rejects(core.encodeWatermark(runtime, session, image, packet, { strength: -1 }), /strength must be a non-negative number/);
  await assert.rejects(core.encodeWatermark(runtime, session, image, packet, { strength: '1' }), /strength must be a non-negative number/);
  await assert.rejects(core.encodeWatermark(runtime, session, image, packet, { variant: 'Z' }), /Unknown TrustMark variant "Z"/);
});

test('resolveEncodeOptions applies the defaults', () => {
  assert.deepEqual(core.resolveEncodeOptions(), { variant: core.VARIANTS.Q, strength: 1.0 });
  assert.deepEqual(core.resolveEncodeOptions({ variant: 'P', strength: 0 }), { variant: core.VARIANTS.P, strength: 0 });
  assert.throws(() => core.resolveEncodeOptions({ strength: NaN }), /strength/);
  assert.throws(() => core.resolveEncodeOptions({ strength: Infinity }), /strength/);
});

test('zeroFeeds builds blank inputs for every model', () => {
  const decoder = core.zeroFeeds(ort, 'decoder_P');
  assert.deepEqual(decoder.image.dims, [1, 3, 224, 224]);
  assert.ok(decoder.image.data.every((v) => v === 0));
  const encoder = core.zeroFeeds(ort, 'encoder_C');
  assert.deepEqual(encoder['onnx::Concat_0'].dims, [1, 3, 256, 256]);
  assert.deepEqual(encoder['onnx::Gemm_1'].dims, [1, 100]);
  const resizer = core.zeroFeeds(ort, 'resizer');
  assert.deepEqual(resizer.X.dims, [1, 3, 8, 8]);
  assert.equal(Number(resizer.target_size.data[0]), 4);
  assert.throws(() => core.zeroFeeds(ort, 'decoder_Z'), /Unknown TrustMark variant/);
  assert.throws(() => core.zeroFeeds(ort, 'remover_Q'), /no zero inputs are defined for the remover_Q model/);
});

test('applyResidual upscales a small residual exactly like bilinearResize', () => {
  const residual = Float32Array.from({ length: 3 * 256 * 256 }, (_, i) => (((i * 37) % 65) - 32) / 1024);
  for (const [width, height, square] of [[240, 240, false], [257, 255, false], [1001, 333, false], [61, 37, true]]) {
    const image = makeTexture(width, height);
    const region = core.processingRegion(width, height, square);
    const reference = core.applyResidual(image, region, core.bilinearResize(residual, 3, 256, 256, region.height, region.width), 1.25);
    assertSameBytes(core.applyResidual(image, region, residual, 1.25, 256, 256), reference, `${width}x${height}`);
  }
});
