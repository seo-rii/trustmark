// End-to-end encoder tests with the real ONNX models: embed a Q / BCH_5 watermark with
// tm_core.js and read it back with the Q decoder, on the WASM backend of the bundled
// ONNX Runtime Web (no GPU needed).
//
// The models are not downloaded by the tests. Put encoder_Q.onnx, decoder_Q.onnx and
// resizer.onnx (from https://cai-watermark.adobe.net/watermarking/trustmark-models/)
// in a directory and run from the repository root:
//   TRUSTMARK_MODEL_DIR=/path/to/models node --test js/tests/*.test.mjs
// Without them these tests are reported as skipped.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import test from 'node:test';
import vm from 'node:vm';

const jsDir = path.join(path.dirname(new URL(import.meta.url).pathname), '..');
const require = createRequire(import.meta.url);

const MODEL_FILES = ['encoder_Q.onnx', 'decoder_Q.onnx', 'resizer.onnx'];
const modelDir = process.env.TRUSTMARK_MODEL_DIR;

function skipReason() {
  if (!modelDir) return `set TRUSTMARK_MODEL_DIR to a directory containing ${MODEL_FILES.join(', ')} to run the model tests`;
  const missing = MODEL_FILES.filter((file) => !fs.existsSync(path.join(modelDir, file)));
  return missing.length ? `${missing.join(', ')} not found in TRUSTMARK_MODEL_DIR=${modelDir}` : false;
}
const skip = skipReason();

let loaded = null;
function load() {
  if (!loaded) {
    loaded = (async () => {
      const ort = require(path.join(jsDir, 'deps/ort.webgpu.min.js'));
      const core = require(path.join(jsDir, 'tm_core.js'));
      const session = (file) => ort.InferenceSession.create(fs.readFileSync(path.join(modelDir, file)), { executionProviders: ['wasm'] });
      const [encoder, decoder, resizer] = await Promise.all(MODEL_FILES.map(session));

      const dl = vm.createContext({ console, BigInt, Math });
      for (const file of ['deps/bch_ecc.min.js', 'tm_datalayer.js']) {
        vm.runInContext(fs.readFileSync(path.join(jsDir, file), 'utf8'), dl, { filename: file });
      }
      return { core, runtime: { ort, resizer }, encoder, decoder, dl, eccengine: vm.runInContext('eccengine', dl) };
    })();
  }
  return loaded;
}

/** Deterministic photo-like RGBA image: smooth gradients, texture and some noise. */
function syntheticImage(width, height, seed, withAlpha) {
  const data = new Uint8ClampedArray(width * height * 4);
  let state = seed;
  const noise = () => ((state = (state * 1664525 + 1013904223) >>> 0) / 2 ** 32 - 0.5) * 24;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const u = x / width;
      const v = y / height;
      const i = (y * width + x) * 4;
      data[i] = 40 + 150 * u + 30 * Math.sin(x / 9 + seed) + noise();
      data[i + 1] = 60 + 120 * v + 25 * Math.cos(y / 7 - x / 23) + noise();
      data[i + 2] = 90 + 80 * (1 - u * v) + 20 * Math.sin((x + y) / 13) + noise();
      data[i + 3] = withAlpha ? Math.round(255 * Math.min(1, 0.2 + u + v)) : 255;
    }
  }
  return { data, width, height };
}

function psnr(a, b) {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < a.length; i++) {
    if (i % 4 === 3) continue;
    sum += (a[i] - b[i]) ** 2;
    count++;
  }
  return 10 * Math.log10((255 * 255) / (sum / count));
}

const CASES = [
  { name: 'RGBA 320x240', width: 320, height: 240, seed: 1, alpha: true, payload: '1011001110001111000011111000001111110000001111111000000011111' },
  { name: 'wide 700x300 (centre square)', width: 700, height: 300, seed: 2, alpha: false, payload: '0110100101011010011010010101101001101001010110100110100101011' },
  { name: 'small 180x150 (upscaled)', width: 180, height: 150, seed: 3, alpha: false, payload: '1' },
];

for (const c of CASES) {
  test(`encodes and decodes the exact payload: ${c.name}`, { skip, timeout: 300000 }, async () => {
    const { core, runtime, encoder, decoder, dl, eccengine } = await load();
    const image = syntheticImage(c.width, c.height, c.seed, c.alpha);

    const packet = dl.DataLayer_Encode(c.payload, 1, eccengine);
    const stego = await core.encodeWatermark(runtime, encoder, image, packet);

    assert.equal(stego.width, c.width);
    assert.equal(stego.height, c.height);
    for (let i = 3; i < image.data.length; i += 4) assert.equal(stego.data[i], image.data[i], 'alpha is preserved');
    const quality = psnr(image.data, stego.data);
    assert.ok(quality > 30, `PSNR ${quality.toFixed(2)} dB`);

    const bits = await core.decodeBits(runtime, decoder, stego, 256);
    const decoded = dl.DataLayer_Decode(bits, eccengine, 'Q');
    assert.equal(decoded.valid, true);
    assert.equal(decoded.schema, 'BCH_5');
    assert.equal(decoded.data_binary, c.payload.padEnd(61, '0'));

    // The unwatermarked image does not decode to the payload
    const coverBits = await core.decodeBits(runtime, decoder, image, 256);
    assert.notEqual(dl.DataLayer_Decode(coverBits, eccengine, 'Q').data_binary, c.payload.padEnd(61, '0'));
  });
}
