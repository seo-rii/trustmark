// Shared test helpers: a fake ONNX Runtime, model directories, deferred promises and the
// model-backed test configuration.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

export const NODE_DIR = path.join(path.dirname(new URL(import.meta.url).pathname), '..');
export const REPO_DIR = path.join(NODE_DIR, '..', '..');
export const IMAGES_DIR = path.join(REPO_DIR, 'images');

// Decoder bits that no schema accepts (raw Q decoder output for the unwatermarked images/ufo_240.jpg)
export const NO_WATERMARK_BITS = '0001001010111011100010001101001000100011011001100111101100110110101010110100001001001010100011001011';

// BCH_5 packet produced by the Python reference implementation (python/trustmark/datalayer.py)
export const PAYLOAD = '0100001111001100111110011010001010001001101011001111111100001';
export const PACKET = '0100001111001100111110011010001010001001101011001111111100001100001011101010001000011011111011010001';

export function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

export const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

export function tempDir(t, prefix = 'trustmark-node-') {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}

/** A model directory whose "models" are text files naming their role, for fakeOrt. */
export function fakeModelDir(t, variant = 'Q') {
  const dir = tempDir(t);
  fs.writeFileSync(path.join(dir, 'resizer.onnx'), 'resizer');
  fs.writeFileSync(path.join(dir, `encoder_${variant}.onnx`), 'encoder');
  fs.writeFileSync(path.join(dir, `decoder_${variant}.onnx`), 'decoder');
  return dir;
}

export function bitsToLogits(bits) {
  return Float32Array.from(bits, (b) => (b === '1' || b === 1 || b === true ? 1 : -1));
}

/**
 * Minimal stand-in for onnxruntime-node. Sessions are created from fakeModelDir files. The
 * resizer returns a flat grey image, the encoder returns its input unchanged (so the residual
 * is zero) and records the packet, and the decoder returns `decoderBits()` as logits, by
 * default the last packet the encoder saw (NO_WATERMARK_BITS before any encode).
 * `hook(role, feeds)` runs before every inference and may block or throw.
 */
export function fakeOrt({ failProviders = [], decoderBits = null, hook = null } = {}) {
  const log = { created: [], runs: [], released: [], packets: [], active: 0, maxActive: 0 };
  class Tensor {
    constructor(type, data, dims) {
      this.type = type;
      this.data = data;
      this.dims = dims;
    }
  }
  class Session {
    constructor(role, provider) {
      this.role = role;
      this.provider = provider;
    }

    async run(feeds) {
      log.active++;
      log.maxActive = Math.max(log.maxActive, log.active);
      try {
        log.runs.push(this.role);
        if (hook) await hook(this.role, feeds);
        if (this.role === 'resizer') {
          const size = Number(feeds.target_size.data[0]);
          return { Y: new Tensor('float32', new Float32Array(3 * size * size).fill(0.5), [1, 3, size, size]) };
        }
        if (this.role === 'encoder') {
          const packet = Array.from(feeds['onnx::Gemm_1'].data).join('');
          log.packets.push(packet);
          return { image: new Tensor('float32', Float32Array.from(feeds['onnx::Concat_0'].data), feeds['onnx::Concat_0'].dims) };
        }
        const bits = decoderBits ? decoderBits(feeds) : log.packets[log.packets.length - 1] || NO_WATERMARK_BITS;
        return { output: new Tensor('float32', bitsToLogits(bits), [1, 100]) };
      } finally {
        log.active--;
      }
    }

    async release() {
      log.released.push(this.role);
    }
  }
  const ort = {
    Tensor,
    InferenceSession: {
      async create(bytes, options) {
        const role = Buffer.from(bytes).toString('utf8');
        const provider = options.executionProviders[0];
        const name = typeof provider === 'string' ? provider : provider.name;
        if (failProviders.includes(name)) throw new Error(`${name} execution provider is not available`);
        log.created.push({ role, provider: name });
        return new Session(role, name);
      },
    },
  };
  return { ort, log };
}

/** A small opaque RGBA test image with some texture. */
export function rgbaImage(width, height, { alpha = 255 } = {}) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      data[i] = (x * 7 + y * 3) & 255;
      data[i + 1] = (x * 2 + y * 11) & 255;
      data[i + 2] = (x * 13 + y * 5) & 255;
      data[i + 3] = typeof alpha === 'function' ? alpha(x, y) : alpha;
    }
  }
  return { data, width, height };
}

// Model-backed tests: TRUSTMARK_MODEL_DIR must hold resizer.onnx and the encoder_<V>.onnx and
// decoder_<V>.onnx files of the variants to test (from
// https://cai-watermark.adobe.net/watermarking/trustmark-models/).
export const MODEL_DIR = process.env.TRUSTMARK_MODEL_DIR || '';

/** False when TRUSTMARK_MODEL_DIR has the models of `variant` (and resizer.onnx), else why not. */
export function modelSkipReason(variant = 'Q') {
  const files = ['resizer.onnx', `encoder_${variant}.onnx`, `decoder_${variant}.onnx`];
  if (!MODEL_DIR) return `set TRUSTMARK_MODEL_DIR to a directory containing ${files.join(', ')} to run the model tests`;
  const missing = files.filter((file) => !fs.existsSync(path.join(MODEL_DIR, file)));
  return missing.length ? `${missing.join(', ')} not found in TRUSTMARK_MODEL_DIR=${MODEL_DIR}` : false;
}
