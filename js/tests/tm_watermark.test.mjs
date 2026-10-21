// Tests for the model lifecycle of tm_watermark.js: readiness, load failures,
// WebGPU -> WASM fallback and the FIFO inference queue. ONNX Runtime, fetch and the
// DOM image loading are replaced by fakes, so neither the models nor a browser are needed.
// Run from the repository root with: node --test js/tests/*.test.mjs

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import vm from 'node:vm';

const jsDir = path.join(path.dirname(new URL(import.meta.url).pathname), '..');

// BCH_5 packet produced by the Python reference implementation (python/trustmark/datalayer.py)
const PAYLOAD = '0100001111001100111110011010001010001001101011001111111100001';
const PACKET = '0100001111001100111110011010001010001001101011001111111100001100001011101010001000011011111011010001';

const BASE_URL = 'https://models.test/';

function deferred() {
  let resolve;
  const promise = new Promise((r) => { resolve = r; });
  return { promise, resolve };
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

// File names of fetched URLs
const fileNames = (urls) => urls.map((url) => url.slice(url.lastIndexOf('/') + 1));

// Values created inside the vm context have that context's prototypes
const plain = (value) => JSON.parse(JSON.stringify(value));

/**
 * Minimal stand-in for onnxruntime-web. Model "files" contain their own name so the fake
 * session knows what it is. Decoders return `decoderOutputs[name]` (100 floats, all -1 by
 * default). `failRun(name, backend)` can make inference fail, `zeroRun(name, backend)` makes it
 * return zeros, and `gpuError(name)` raises a WebGPU error that the error scopes of
 * env.webgpu.device report.
 */
function fakeOrt({ failCreate = () => null, failRun = () => null, zeroRun = () => false, gpuError = () => null, decoderOutputs = {}, runLog = [] } = {}) {
  class Tensor {
    constructor(type, data, dims) {
      this.type = type;
      this.data = data;
      this.dims = dims;
    }
  }
  const created = [];
  const encoderFeeds = [];
  const runs = [];
  const released = [];
  const scopes = [];
  const device = {
    pushErrorScope(filter) { scopes.push({ filter, error: null }); },
    async popErrorScope() { return scopes.pop().error; },
  };
  const InferenceSession = {
    async create(model, options) {
      const name = new TextDecoder().decode(model);
      const backend = options.executionProviders[0];
      created.push({ name, backend });
      const failure = failCreate(name, backend);
      if (failure) throw new Error(failure);
      return {
        async release() { released.push({ name, backend }); },
        async run(feeds) {
          runLog.push(name);
          runs.push({ name, backend, feeds });
          const runFailure = failRun(name, backend);
          if (runFailure) throw new Error(runFailure);
          const error = backend === 'webgpu' ? gpuError(name) : null;
          if (error && scopes.length) scopes[0].error = { message: error };
          if (zeroRun(name, backend)) {
            const output = name.startsWith('encoder_') ? 'image' : 'output';
            return { [output]: new Tensor('float32', new Float32Array(name.startsWith('encoder_') ? 3 * 256 * 256 : 100), []) };
          }
          if (name === 'resizer') {
            const size = Number(feeds.target_size.data[0]);
            return { Y: new Tensor('float32', new Float32Array(3 * size * size), [1, 3, size, size]) };
          }
          if (name.startsWith('encoder_')) {
            encoderFeeds.push({ name, feeds });
            const cover = feeds['onnx::Concat_0'];
            return { image: new Tensor('float32', Float32Array.from(cover.data, (v) => v + 0.5), cover.dims) };
          }
          return { output: new Tensor('float32', Float32Array.from(decoderOutputs[name] || new Array(100).fill(-1)), [1, 100]) };
        },
      };
    },
  };
  return { Tensor, InferenceSession, env: { webgpu: { device } }, created, encoderFeeds, runs, released, scopes };
}

/** Fake fetch serving `${BASE_URL}${name}.onnx`; `gate` can hold responses back. */
function fakeFetch({ missing = [], gate = null, log = [] } = {}) {
  return async (url) => {
    log.push(url);
    if (gate) await gate;
    const name = url.slice(url.lastIndexOf('/') + 1).replace(/\.onnx$/, '');
    if (missing.includes(name)) return { ok: false, status: 404 };
    return { ok: true, status: 200, arrayBuffer: async () => new TextEncoder().encode(name).buffer };
  };
}

// Images "load" asynchronously like in a browser; the src 'broken' fails to load
class FakeImage {
  set src(value) {
    setTimeout(() => (value === 'broken' ? this.onerror() : this.onload()), 0);
  }
  get width() { return 8; }
  get height() { return 8; }
}

const fakeDocument = {
  createElement: () => ({
    getContext: () => ({
      drawImage() {},
      getImageData: (x, y, w, h) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h }),
    }),
  }),
};

const quietConsole = { log() {}, warn() {}, error() {} };

// Load the demo's classic scripts into a fresh context, as the browser would
function loadWatermark({ ort, fetch, autoInit = true }) {
  const context = vm.createContext({
    console: quietConsole, setTimeout, TextDecoder, ort, fetch,
    Image: FakeImage, document: fakeDocument,
  });
  if (!autoInit) context.TRUSTMARK_AUTO_INIT = false;
  for (const file of ['deps/bch_ecc.min.js', 'tm_datalayer.js', 'tm_core.js', 'tm_watermark.js']) {
    vm.runInContext(fs.readFileSync(path.join(jsDir, file), 'utf8'), context, { filename: file });
  }
  // Top-level const/let bindings are not properties of the global object
  const get = (name) => vm.runInContext(name, context);
  return { context, get };
}

const watermarked = Array.from(PACKET, (bit) => (bit === '1' ? 1 : -1));
const noise = Array.from({ length: 100 }, (_, i) => ((i * 7) % 3 === 0 ? 1 : -1));

test('an early runwmark call waits for the models instead of skipping them', async () => {
  const gate = deferred();
  const ort = fakeOrt({ decoderOutputs: { decoder_Q: watermarked, decoder_P: noise } });
  const tm = loadWatermark({ ort, fetch: fakeFetch({ gate: gate.promise }) });

  let settled = false;
  const pending = tm.get('runwmark')('image.png').then((result) => { settled = true; return result; });
  await tick();
  await tick();
  assert.equal(settled, false, 'runwmark must not return before the models have loaded');
  assert.equal(tm.get('trustmarkStatus')().loaded, false);

  gate.resolve();
  const result = await pending;
  assert.equal(result.status, 'detected');
  assert.equal(result.watermark_present, true);
  assert.equal(result.watermark, PAYLOAD);
  assert.equal(result.schema, 'BCH_5');
  assert.equal(result.variant, 'Q');
  assert.equal(result.backend, 'webgpu');
  assert.equal(result.error, null);
  assert.deepEqual(plain(result.decoders.map((d) => d.status)), ['detected', 'not_detected']);

  const ready = await tm.get('trustmarkReady');
  assert.equal(ready.loaded, true);
  assert.deepEqual(plain(Object.keys(ready.models).sort()), ['decoder_P', 'decoder_Q', 'resizer']);
});

test('reports not_detected only when every decoder checked the image', async () => {
  const ort = fakeOrt({ decoderOutputs: { decoder_Q: noise, decoder_P: noise } });
  const tm = loadWatermark({ ort, fetch: fakeFetch() });

  const result = await tm.get('runwmark')('image.png');
  assert.equal(result.status, 'not_detected');
  assert.equal(result.watermark_present, false);
  assert.equal(result.error, null);
  assert.deepEqual(plain(result.decoders.map((d) => d.status)), ['not_detected', 'not_detected']);
});

test('a decoder that failed to load is reported as an error, not as absence', async () => {
  const ort = fakeOrt({
    failCreate: (name) => (name === 'decoder_P' ? 'out of memory' : null),
    decoderOutputs: { decoder_Q: noise },
  });
  const tm = loadWatermark({ ort, fetch: fakeFetch() });

  const status = await tm.get('trustmarkReady');
  assert.equal(status.models.decoder_P.status, 'error');
  assert.match(status.models.decoder_P.error, /webgpu: out of memory; wasm: out of memory/);
  assert.equal(status.models.decoder_Q.status, 'ready');

  const result = await tm.get('runwmark')('image.png');
  assert.equal(result.status, 'error');
  assert.equal(result.watermark_present, false);
  assert.match(result.error, /decoder_P model is not loaded/);
  assert.deepEqual(plain(result.decoders.map((d) => d.status)), ['not_detected', 'error']);
});

test('a watermark found by a loaded decoder is still reported when another failed', async () => {
  const ort = fakeOrt({ decoderOutputs: { decoder_Q: watermarked } });
  const tm = loadWatermark({ ort, fetch: fakeFetch({ missing: ['decoder_P'] }) });

  const result = await tm.get('runwmark')('image.png');
  assert.equal(result.status, 'detected');
  assert.equal(result.watermark, PAYLOAD);
  assert.equal(result.error, null);
  assert.match(result.decoders[1].error, /could not download decoder_P\.onnx: HTTP 404/);
});

test('a missing resizer or an unreadable image is an error', async () => {
  const ort = fakeOrt({ decoderOutputs: { decoder_Q: watermarked, decoder_P: watermarked } });
  const tm = loadWatermark({ ort, fetch: fakeFetch({ missing: ['resizer'] }) });

  const result = await tm.get('runwmark')('image.png');
  assert.equal(result.status, 'error');
  assert.match(result.error, /resizer model is not loaded: could not download resizer\.onnx: HTTP 404/);

  const unreadable = await tm.get('runwmark')('broken');
  assert.equal(unreadable.status, 'error');
  assert.match(unreadable.error, /could not read the image: Failed to load image/);
});

test('falls back to WASM when a WebGPU session cannot be created, and records it', async () => {
  const ort = fakeOrt({
    failCreate: (name, backend) => (backend === 'webgpu' ? 'no available backend found' : null),
    decoderOutputs: { decoder_Q: watermarked, decoder_P: noise },
  });
  const fetchLog = [];
  const tm = loadWatermark({ ort, fetch: fakeFetch({ log: fetchLog }) });

  const { models } = await tm.get('trustmarkReady');
  for (const name of ['decoder_Q', 'decoder_P']) {
    assert.equal(models[name].status, 'ready', name);
    assert.equal(models[name].backend, 'wasm', name);
    assert.equal(models[name].fallbackFrom, 'webgpu', name);
    assert.match(models[name].fallbackReason, /webgpu: no available backend found/, name);
  }
  // The resizer is WASM-only and never tries WebGPU
  assert.equal(models.resizer.backend, 'wasm');
  assert.equal(models.resizer.fallbackFrom, null);
  assert.deepEqual(ort.created.filter((c) => c.name === 'resizer').map((c) => c.backend), ['wasm']);
  // Each model is downloaded once even when a second backend is tried
  assert.equal(fetchLog.length, 3);

  const result = await tm.get('runwmark')('image.png');
  assert.equal(result.status, 'detected');
  assert.equal(result.backend, 'wasm');
});

test('trustmarkInit options apply when auto-initialization is disabled', async () => {
  const ort = fakeOrt({ decoderOutputs: { decoder_Q: noise, decoder_P: noise } });
  const fetchLog = [];
  const tm = loadWatermark({ ort, fetch: fakeFetch({ log: fetchLog }), autoInit: false });

  await tick();
  assert.equal(fetchLog.length, 0, 'nothing loads before trustmarkInit()');

  const status = await tm.get('trustmarkInit')({ modelBaseUrl: 'https://mirror.test/', executionProviders: ['wasm'] });
  assert.equal(status.loaded, true);
  assert.deepEqual(fetchLog, ['https://mirror.test/resizer.onnx', 'https://mirror.test/decoder_Q.onnx', 'https://mirror.test/decoder_P.onnx']);
  assert.ok(ort.created.every((c) => c.backend === 'wasm'));
  // Later calls share the same load
  assert.equal(tm.get('trustmarkInit')(), tm.get('trustmarkInit')());
});

test('inferences run one at a time in FIFO order, and a failure does not stall the queue', async () => {
  const tm = loadWatermark({ ort: fakeOrt(), fetch: fakeFetch() });
  const safeRunInference = tm.get('safeRunInference');

  const events = [];
  let active = 0;
  let maxActive = 0;
  const session = (name, delay, fail = false) => ({
    run: () => new Promise((resolve, reject) => {
      active++;
      maxActive = Math.max(maxActive, active);
      events.push(`start ${name}`);
      setTimeout(() => {
        active--;
        events.push(`end ${name}`);
        if (fail) reject(new Error(`${name} failed`));
        else resolve(name);
      }, delay);
    }),
  });

  const runs = [
    safeRunInference(session('a', 40), {}),
    safeRunInference(session('b', 5, true), {}),
  ];
  // Queued while 'a' is still running; it must not overtake 'b' when 'a' finishes
  await new Promise((resolve) => setTimeout(resolve, 15));
  runs.push(safeRunInference(session('c', 0), {}));
  runs.push(safeRunInference(session('d', 10), {}));
  const outcomes = await Promise.allSettled(runs);

  assert.equal(maxActive, 1);
  assert.deepEqual(events, ['start a', 'end a', 'start b', 'end b', 'start c', 'end c', 'start d', 'end d']);
  assert.deepEqual(outcomes.map((o) => o.status), ['fulfilled', 'rejected', 'fulfilled', 'fulfilled']);
  assert.equal(outcomes[1].reason.message, 'b failed');
});

const rgbaImage = (width, height) => ({ data: new Uint8ClampedArray(width * height * 4).fill(200), width, height });

test('trustmarkEncode embeds the BCH_5 packet and records the encoder backend', async () => {
  const ort = fakeOrt({ failCreate: (name, backend) => (backend === 'webgpu' ? 'no available backend found' : null) });
  const tm = loadWatermark({ ort, fetch: fakeFetch() });

  const result = await tm.get('trustmarkEncode')(rgbaImage(64, 48), PAYLOAD);

  assert.equal(result.packet, PACKET);
  assert.equal(result.payload, PAYLOAD);
  assert.equal(result.schema, 'BCH_5');
  assert.equal(result.variant, 'Q');
  assert.equal(result.backend, 'wasm');
  assert.equal(result.image.width, 64);
  assert.equal(result.image.height, 48);
  assert.equal(ort.encoderFeeds[0].name, 'encoder_Q');
  assert.equal(Array.from(ort.encoderFeeds[0].feeds['onnx::Gemm_1'].data).join(''), PACKET);

  const { models } = tm.get('trustmarkStatus')();
  assert.equal(models.encoder_Q.status, 'ready');
  assert.equal(models.encoder_Q.fallbackFrom, 'webgpu');

  // Short payloads are padded with zeros, like the Python implementation
  const short = await tm.get('trustmarkEncode')(rgbaImage(8, 8), '1');
  assert.equal(short.payload, '1'.padEnd(61, '0'));
});

test('trustmarkEncode fails instead of returning an unwatermarked image', async () => {
  const fetchLog = [];
  const tm = loadWatermark({ ort: fakeOrt(), fetch: fakeFetch({ missing: ['encoder_Q'], log: fetchLog }) });
  const encode = tm.get('trustmarkEncode');

  // Invalid payloads are rejected before the encoder is downloaded
  await assert.rejects(encode(rgbaImage(8, 8), '1'.repeat(62)), /62 bits but BCH_5 allows at most 61 bits/);
  await assert.rejects(encode(rgbaImage(8, 8), 12345), /'0' and '1' characters/);
  assert.ok(!fetchLog.some((url) => url.includes('encoder_Q')));

  await assert.rejects(encode(rgbaImage(8, 8), PAYLOAD), /encoder_Q model is not loaded: could not download encoder_Q\.onnx: HTTP 404/);
  assert.equal(tm.get('trustmarkStatus')().models.encoder_Q.status, 'error');
});

test('trustmarkEncode needs neither the resizer nor a decoder', async () => {
  const fetchLog = [];
  const tm = loadWatermark({ ort: fakeOrt(), fetch: fakeFetch({ missing: ['resizer'], log: fetchLog }), autoInit: false });
  await tm.get('trustmarkInit')({ decoders: [] });

  const result = await tm.get('trustmarkEncode')(rgbaImage(8, 8), PAYLOAD);
  assert.equal(result.packet, PACKET);
  assert.deepEqual(fileNames(fetchLog), ['encoder_Q.onnx'], 'only the encoder is downloaded');
  assert.deepEqual(plain(Object.keys(tm.get('trustmarkStatus')().models)), ['encoder_Q']);

  // Without decoders, runwmark cannot check images
  const decoded = await tm.get('runwmark')('image.png');
  assert.equal(decoded.status, 'error');
  assert.match(decoded.error, /no watermark decoders are loaded/);
});

test('trustmarkInit loads the chosen decoders and runwmark tries them in that order', async () => {
  const fetchLog = [];
  const ort = fakeOrt({ decoderOutputs: { decoder_C: noise, decoder_B: watermarked } });
  const tm = loadWatermark({ ort, fetch: fakeFetch({ log: fetchLog }), autoInit: false });

  const status = await tm.get('trustmarkInit')({ decoders: ['C', 'B', 'C'] });
  assert.deepEqual(fileNames(fetchLog), ['resizer.onnx', 'decoder_C.onnx', 'decoder_B.onnx']);
  assert.deepEqual(plain(Object.keys(status.models)), ['resizer', 'decoder_C', 'decoder_B']);

  const result = await tm.get('runwmark')('image.png');
  assert.equal(result.status, 'detected');
  assert.equal(result.variant, 'B');
  assert.equal(result.watermark, PAYLOAD);
  assert.equal(result.c2padata['c2pa.soft-binding'].alg, 'com.adobe.trustmark.B');
  assert.deepEqual(plain(result.decoders.map((d) => [d.variant, d.status])), [['C', 'not_detected'], ['B', 'detected']]);
});

test('trustmarkInit rejects unknown decoder variants without loading anything', async () => {
  const fetchLog = [];
  const tm = loadWatermark({ ort: fakeOrt(), fetch: fakeFetch({ log: fetchLog }), autoInit: false });
  const init = tm.get('trustmarkInit');

  assert.throws(() => init({ decoders: ['Q', 'X'] }), /Unknown TrustMark variant "X"; expected one of Q, P, B, C/);
  assert.throws(() => init({ decoders: 'Q' }), /decoders must be an array/);
  await tick();
  assert.equal(fetchLog.length, 0);

  // A valid call still works afterwards
  const status = await init({ decoders: ['P'] });
  assert.deepEqual(plain(Object.keys(status.models)), ['resizer', 'decoder_P']);
});

// Packets from the Python data layer (tests/tm_datalayer.test.mjs)
const SCHEMA_VECTORS = [
  { variant: 'P', schema: 'BCH_3', payload: '111010100111011000110001010111110101100111100010001100101000001101001110110', packet: '1110101001110110001100010101111101011001111000100011001010000011010011101100001010001101100110000011' },
  { variant: 'B', schema: 0, payload: '0001000100110011001111111000000010000110', packet: '0001000100110011001111111000000010000110010111010010010001010000110100011100001100011001110111100000' },
  { variant: 'C', schema: 'BCH_4', payload: '11010110101000001010111100111111101111001100000111111001100001111000', packet: '1101011010100000101011110011111110111100110000011111100110000111100000100101100100101101100100010010' },
  { variant: 'Q', schema: 'BCH_SUPER', payload: '1', packet: '1000000000000000000000000000000000000000111101101110010000010111110011001101111101111100000110000000' },
];

test('trustmarkEncode embeds every variant and schema with its own encoder', async () => {
  const fetchLog = [];
  const ort = fakeOrt();
  const tm = loadWatermark({ ort, fetch: fakeFetch({ log: fetchLog }) });
  const encode = tm.get('trustmarkEncode');

  for (const v of SCHEMA_VECTORS) {
    const result = await encode(rgbaImage(40, 30), v.payload, { variant: v.variant, schema: v.schema, strength: 0.5 });
    assert.equal(result.variant, v.variant);
    assert.equal(result.packet, v.packet);
    assert.equal(result.schema, typeof v.schema === 'number' ? 'BCH_SUPER' : v.schema);
    assert.equal(result.payload.length, { BCH_SUPER: 40, BCH_5: 61, BCH_4: 68, BCH_3: 75 }[result.schema]);
    const feeds = ort.encoderFeeds.at(-1);
    assert.equal(feeds.name, `encoder_${v.variant}`);
    assert.equal(Array.from(feeds.feeds['onnx::Gemm_1'].data).join(''), v.packet);
  }
  // Each encoder is downloaded once, on first use
  await encode(rgbaImage(8, 8), '1', { variant: 'P' });
  assert.deepEqual(fileNames(fetchLog).filter((name) => name.startsWith('encoder_')),
    ['encoder_P.onnx', 'encoder_B.onnx', 'encoder_C.onnx', 'encoder_Q.onnx']);
  for (const name of ['encoder_P', 'encoder_B', 'encoder_C', 'encoder_Q']) {
    assert.equal(tm.get('trustmarkStatus')().models[name].status, 'ready', name);
  }
});

test('trustmarkEncode rejects unknown variants, schemas and strengths before downloading', async () => {
  const fetchLog = [];
  const tm = loadWatermark({ ort: fakeOrt(), fetch: fakeFetch({ log: fetchLog }) });
  const encode = tm.get('trustmarkEncode');

  await assert.rejects(encode(rgbaImage(8, 8), '1', { variant: 'Z' }), /Unknown TrustMark variant "Z"; expected one of Q, P, B, C/);
  await assert.rejects(encode(rgbaImage(8, 8), '1', { schema: 'BCH_6' }), /Unknown watermark schema "BCH_6"/);
  await assert.rejects(encode(rgbaImage(8, 8), '1', { schema: 7 }), /Unknown watermark schema 7/);
  await assert.rejects(encode(rgbaImage(8, 8), '1', { strength: -0.5 }), /strength must be a non-negative number/);
  await assert.rejects(encode(rgbaImage(8, 8), '1'.repeat(41), { variant: 'P', schema: 'BCH_SUPER' }), /41 bits but BCH_SUPER allows at most 40 bits/);
  await tm.get('trustmarkReady');
  assert.ok(!fetchLog.some((url) => url.includes('encoder_')), 'no encoder was downloaded');
});

test('a WebGPU session whose first inference fails falls back to the next backend', async () => {
  const ort = fakeOrt({
    failRun: (name, backend) => (backend === 'webgpu' && name === 'decoder_P' ? 'unsupported operator' : null),
    decoderOutputs: { decoder_Q: noise, decoder_P: watermarked },
  });
  const tm = loadWatermark({ ort, fetch: fakeFetch() });

  const { models } = await tm.get('trustmarkReady');
  assert.equal(models.decoder_Q.backend, 'webgpu');
  assert.equal(models.decoder_P.status, 'ready');
  assert.equal(models.decoder_P.backend, 'wasm');
  assert.equal(models.decoder_P.fallbackFrom, 'webgpu');
  assert.match(models.decoder_P.fallbackReason, /webgpu: warm-up inference failed: unsupported operator/);
  assert.deepEqual(plain(ort.released), [{ name: 'decoder_P', backend: 'webgpu' }], 'the failed session is released');

  const result = await tm.get('runwmark')('image.png');
  assert.equal(result.status, 'detected');
  assert.equal(result.variant, 'P');
  assert.equal(result.backend, 'wasm');
});

test('WebGPU sessions that return zeros or raise WebGPU errors fall back to WASM', async () => {
  const ort = fakeOrt({
    zeroRun: (name, backend) => backend === 'webgpu' && (name === 'decoder_Q' || name === 'encoder_Q'),
    gpuError: (name) => (name === 'decoder_P' ? 'Invalid ComputePipeline "Cast"' : null),
    decoderOutputs: { decoder_Q: watermarked, decoder_P: noise },
  });
  const tm = loadWatermark({ ort, fetch: fakeFetch() });

  const { models } = await tm.get('trustmarkReady');
  assert.equal(models.decoder_Q.backend, 'wasm');
  assert.match(models.decoder_Q.fallbackReason, /webgpu: warm-up inference failed: the model returned only zeros/);
  assert.equal(models.decoder_P.backend, 'wasm');
  assert.match(models.decoder_P.fallbackReason, /webgpu: warm-up inference failed: WebGPU error: Invalid ComputePipeline "Cast"/);
  assert.equal(ort.scopes.length, 0, 'every error scope is popped');

  const decoded = await tm.get('runwmark')('image.png');
  assert.equal(decoded.status, 'detected');
  assert.equal(decoded.backend, 'wasm');

  // Encoders are checked on WebGPU before their first use too
  const encoded = await tm.get('trustmarkEncode')(rgbaImage(8, 8), PAYLOAD);
  assert.equal(encoded.backend, 'wasm');
  const { encoder_Q } = tm.get('trustmarkStatus')().models;
  assert.match(encoder_Q.fallbackReason, /webgpu: warm-up inference failed: the model returned only zeros/);
  assert.equal(encoder_Q.warmUpMs, null, 'the WASM encoder is not warmed up');
  assert.deepEqual(plain(ort.runs.filter((r) => r.name === 'encoder_Q').map((r) => r.backend)), ['webgpu', 'wasm']);
});
