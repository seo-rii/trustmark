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

// Values created inside the vm context have that context's prototypes
const plain = (value) => JSON.parse(JSON.stringify(value));

/**
 * Minimal stand-in for onnxruntime-web. Model "files" contain their own name so the fake
 * session knows what it is. Decoders return `decoderOutputs[name]` (100 floats).
 */
function fakeOrt({ failCreate = () => null, decoderOutputs = {}, runLog = [] } = {}) {
  class Tensor {
    constructor(type, data, dims) {
      this.type = type;
      this.data = data;
      this.dims = dims;
    }
  }
  const created = [];
  const InferenceSession = {
    async create(model, options) {
      const name = new TextDecoder().decode(model);
      const backend = options.executionProviders[0];
      created.push({ name, backend });
      const failure = failCreate(name, backend);
      if (failure) throw new Error(failure);
      return {
        async run(feeds) {
          runLog.push(name);
          if (name === 'resizer') {
            const size = Number(feeds.target_size.data[0]);
            return { Y: new Tensor('float32', new Float32Array(3 * size * size), [1, 3, size, size]) };
          }
          return { output: new Tensor('float32', Float32Array.from(decoderOutputs[name]), [1, 100]) };
        },
      };
    },
  };
  return { Tensor, InferenceSession, created };
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
