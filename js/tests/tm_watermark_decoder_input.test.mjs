// Tests for the decode path of tm_watermark.js, with fake ONNX Runtime sessions and DOM objects
// (no browser or models needed).
// Run from the repository root with: node --test js/tests/*.test.mjs

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import vm from 'node:vm';

const jsDir = path.join(path.dirname(new URL(import.meta.url).pathname), '..');

class FakeTensor {
  constructor(type, data, dims) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }
}

// A 4x4 image of mid-grey pixels, whatever URL is loaded
class FakeImage {
  set src(_url) {
    this.width = 4;
    this.height = 4;
    setTimeout(() => this.onload(), 0);
  }
}

const fakeDocument = {
  createElement: () => ({
    getContext: () => ({
      drawImage() {},
      getImageData: (x, y, w, h) => ({ data: new Uint8ClampedArray(w * h * 4).fill(128), width: w, height: h }),
    }),
  }),
};

// Loads the demo scripts with sessions that record the decoder inputs. The fake resizer
// returns a tensor of 0.25 everywhere, i.e. pixels in the [0, 1] range.
async function loadWatermark() {
  const decoderInputs = [];
  const fakeSession = (url) => ({
    async run(feeds) {
      if (url.endsWith('resizer.onnx')) {
        const size = Number(feeds.target_size.data[0]);
        return { Y: new FakeTensor('float32', new Float32Array(3 * size * size).fill(0.25), [1, 3, size, size]) };
      }
      decoderInputs.push({ url, image: feeds.image });
      return { output: { cpuData: new Float32Array(100).fill(1) } };
    },
  });
  const ort = {
    Tensor: FakeTensor,
    InferenceSession: { create: async (url) => fakeSession(url) },
  };

  const context = vm.createContext({
    console: { log() {}, warn() {}, error() {} },
    BigInt, BigInt64Array, Math, Image: FakeImage, document: fakeDocument, ort, setTimeout,
  });
  for (const file of ['deps/bch_ecc.min.js', 'tm_datalayer.js', 'tm_watermark.js']) {
    vm.runInContext(fs.readFileSync(path.join(jsDir, file), 'utf8'), context, { filename: file });
  }
  // Let the model loading at the end of tm_watermark.js finish
  await new Promise((resolve) => setTimeout(resolve, 10));
  return { context, decoderInputs };
}

test('decoders get pixels in [-1, 1]', async () => {
  const { context, decoderInputs } = await loadWatermark();
  await context.runwmark('image.png');

  assert.deepEqual(decoderInputs.map((input) => input.url.split('/').pop()), ['decoder_Q.onnx', 'decoder_P.onnx']);
  for (const { url, image } of decoderInputs) {
    assert.ok(image.data.every((v) => v === -0.5), `${url} input is not in [-1, 1]`);
  }
  assert.deepEqual(Array.from(decoderInputs[0].image.dims), [1, 3, 256, 256]);
  assert.deepEqual(Array.from(decoderInputs[1].image.dims), [1, 3, 224, 224]);
});

test('toDecoderInput rescales from [0, 1] to [-1, 1]', async () => {
  const { context } = await loadWatermark();
  const input = new FakeTensor('float32', new Float32Array([0, 0.25, 0.5, 1]), [1, 1, 2, 2]);
  const output = context.toDecoderInput(input);

  assert.deepEqual(Array.from(output.data), [-1, -0.5, 0, 1]);
  assert.deepEqual(output.dims, [1, 1, 2, 2]);
  assert.deepEqual(Array.from(input.data), [0, 0.25, 0.5, 1]);
});
