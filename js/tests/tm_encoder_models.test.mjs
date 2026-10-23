// End-to-end encoder tests with the real ONNX models: embed watermarks of every variant and
// schema with tm_core.js and read them back with the decoder of the same variant, on the WASM
// backend of the bundled ONNX Runtime Web (no GPU needed).
//
// The models are not downloaded by the tests. Put resizer.onnx and the encoder_<V>.onnx and
// decoder_<V>.onnx files of the variants to test (from
// https://cai-watermark.adobe.net/watermarking/trustmark-models/) in a directory and run from
// the repository root:
//   TRUSTMARK_MODEL_DIR=/path/to/models node --test js/tests/*.test.mjs
// Variants whose models are missing are reported as skipped.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import test from 'node:test';
import vm from 'node:vm';
import zlib from 'node:zlib';

const jsDir = path.join(path.dirname(new URL(import.meta.url).pathname), '..');
const require = createRequire(import.meta.url);

const modelDir = process.env.TRUSTMARK_MODEL_DIR;

function skipReason(variant) {
  const files = ['resizer.onnx', `encoder_${variant}.onnx`, `decoder_${variant}.onnx`];
  if (!modelDir) return `set TRUSTMARK_MODEL_DIR to a directory containing ${files.join(', ')} to run the model tests`;
  const missing = files.filter((file) => !fs.existsSync(path.join(modelDir, file)));
  return missing.length ? `${missing.join(', ')} not found in TRUSTMARK_MODEL_DIR=${modelDir}` : false;
}

let loaded = null;
const sessions = {};
function load() {
  if (!loaded) {
    loaded = (async () => {
      const ort = require(path.join(jsDir, 'deps/ort.webgpu.min.js'));
      const core = require(path.join(jsDir, 'tm_core.js'));
      const resizer = await ort.InferenceSession.create(fs.readFileSync(path.join(modelDir, 'resizer.onnx')), { executionProviders: ['wasm'] });

      const dl = vm.createContext({ console, BigInt, Math });
      for (const file of ['deps/bch_ecc.min.js', 'tm_datalayer.js']) {
        vm.runInContext(fs.readFileSync(path.join(jsDir, file), 'utf8'), dl, { filename: file });
      }
      const session = (name) => (sessions[name] ||= ort.InferenceSession.create(fs.readFileSync(path.join(modelDir, `${name}.onnx`)), { executionProviders: ['wasm'] }));
      return { core, runtime: { ort, resizer }, session, dl, eccengine: vm.runInContext('eccengine', dl) };
    })();
  }
  return loaded;
}

/** Decodes an 8-bit, non-interlaced RGB or RGBA PNG into RGBA pixels. */
function readPng(file) {
  const buf = fs.readFileSync(file);
  let width, height, colorType;
  const idat = [];
  for (let pos = 8; pos < buf.length;) {
    const length = buf.readUInt32BE(pos);
    const type = buf.toString('latin1', pos + 4, pos + 8);
    const body = buf.subarray(pos + 8, pos + 8 + length);
    if (type === 'IHDR') {
      width = body.readUInt32BE(0); height = body.readUInt32BE(4); colorType = body[9];
      if (body[8] !== 8 || (colorType !== 2 && colorType !== 6) || body[12] !== 0) throw new Error(`${file}: only 8-bit non-interlaced RGB(A) PNGs are supported`);
    } else if (type === 'IDAT') idat.push(body);
    pos += 12 + length;
  }
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const bpp = colorType === 6 ? 4 : 3;
  const stride = width * bpp;
  const pixels = new Uint8Array(height * stride);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let x = 0; x < stride; x++) {
      const a = x >= bpp ? pixels[y * stride + x - bpp] : 0;
      const b = y > 0 ? pixels[(y - 1) * stride + x] : 0;
      const c = x >= bpp && y > 0 ? pixels[(y - 1) * stride + x - bpp] : 0;
      let predictor = 0;
      if (filter === 1) predictor = a;
      else if (filter === 2) predictor = b;
      else if (filter === 3) predictor = (a + b) >> 1;
      else if (filter === 4) {
        const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        predictor = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      pixels[y * stride + x] = (line[x] + predictor) & 255;
    }
  }
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0, j = 0; i < width * height; i++, j += bpp) {
    data[i * 4] = pixels[j]; data[i * 4 + 1] = pixels[j + 1]; data[i * 4 + 2] = pixels[j + 2];
    data[i * 4 + 3] = bpp === 4 ? pixels[j + 3] : 255;
  }
  return { data, width, height };
}

/** Copies a rectangle of an RGBA image. */
function crop(image, left, top, width, height) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    const start = ((top + y) * image.width + left) * 4;
    data.set(image.data.subarray(start, start + width * 4), y * width * 4);
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

// Every schema on a different crop of images/ghost.png (RGBA): the whole image, wider than 2:1
// (centre square), taller than 3:1 with odd sizes, and smaller than the model resolution (upscaled)
const CASES = [
  { name: 'RGBA 512x512', rect: [0, 0, 512, 512], schema: 'BCH_5', payload: '1011001110001111000011111000001111110000001111111000000011111' },
  { name: 'wide 512x200', rect: [0, 150, 512, 200], schema: 'BCH_SUPER', payload: '0110100101011010011010010101101001101001' },
  { name: 'tall odd 161x501', rect: [170, 5, 161, 501], schema: 'BCH_4', payload: '11010110101000001010111100111111101111001100000111111001100001111000' },
  { name: 'small 180x150', rect: [160, 160, 180, 150], schema: 'BCH_3', payload: '111010100111011000110001010111110101100111100010001100101000001101001110110' },
];

for (const variant of ['Q', 'P', 'B', 'C']) {
  test(`variant ${variant}: every schema decodes to the exact payload`, { skip: skipReason(variant), timeout: 600000 }, async () => {
    const { core, runtime, session, dl, eccengine } = await load();
    const config = core.getVariant(variant);
    const [encoder, decoder] = await Promise.all([session(config.encoder), session(config.decoder)]);

    const ghost = readPng(path.join(jsDir, '../images/ghost.png'));
    for (const c of CASES) {
      const image = crop(ghost, ...c.rect);
      const [, , width, height] = c.rect;
      const version = dl.DataLayer_GetSchemaVersion(c.schema);
      const expected = c.payload.padEnd(dl.DataLayer_GetSchemaDataBits(version), '0');

      const packet = dl.DataLayer_Encode(c.payload, version, eccengine);
      const stego = await core.encodeWatermark(runtime, encoder, image, packet, { variant });

      assert.equal(stego.width, width);
      assert.equal(stego.height, height);
      for (let i = 3; i < image.data.length; i += 4) assert.equal(stego.data[i], image.data[i], 'alpha is preserved');
      const quality = psnr(image.data, stego.data);
      assert.ok(quality > 30, `${c.name}: PSNR ${quality.toFixed(2)} dB`);
      // Pixels outside the processed region (the centre square for P and above 2:1) are unchanged
      const region = core.processingRegion(width, height, config.forceSquare);
      let changedOutside = 0;
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const inside = x >= region.left && x < region.left + region.width && y >= region.top && y < region.top + region.height;
          const i = (y * width + x) * 4;
          if (!inside && (stego.data[i] !== image.data[i] || stego.data[i + 1] !== image.data[i + 1] || stego.data[i + 2] !== image.data[i + 2])) changedOutside++;
        }
      }
      assert.equal(changedOutside, 0, `${c.name}: pixels outside the processed region changed`);

      const bits = await core.decodeBits(runtime, decoder, stego, variant);
      const decoded = dl.DataLayer_Decode(bits, eccengine, variant);
      assert.equal(decoded.valid, true, c.name);
      assert.equal(decoded.schema, c.schema, c.name);
      assert.equal(decoded.data_binary, expected, c.name);

      // The unwatermarked image does not decode to the payload
      const coverBits = await core.decodeBits(runtime, decoder, image, variant);
      assert.notEqual(dl.DataLayer_Decode(coverBits, eccengine, variant).data_binary, expected, c.name);
    }
  });
}
