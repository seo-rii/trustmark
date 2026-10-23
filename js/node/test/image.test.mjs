// Image input/output: PNG and JPEG decoding, EXIF orientation, size limits, alpha handling.
// No models or network needed. Run from js/node with: npm test

import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import test from 'node:test';

import { IMAGES_DIR, NODE_DIR, rgbaImage } from './helpers.mjs';

const require = createRequire(import.meta.url);
const { decodeImage, encodeImage, flatten } = require('../lib/image.js');
const { PNG } = require('pngjs');

const orientation = JSON.parse(fs.readFileSync(path.join(NODE_DIR, 'test/fixtures/orientation.json'), 'utf8'));
const LIMITS = { maxPixels: 25_000_000 };

function rgbOf(image) {
  const out = [];
  for (let i = 0; i < image.data.length; i += 4) out.push(image.data[i], image.data[i + 1], image.data[i + 2]);
  return out;
}

test('PNG output is lossless and keeps the alpha channel', () => {
  const image = rgbaImage(37, 23, { alpha: (x, y) => (x * 9 + y * 5) & 255 });
  const bytes = encodeImage(image, { format: 'png' });
  const png = PNG.sync.read(bytes);
  assert.equal(png.colorType, 6);
  const back = decodeImage(bytes, LIMITS);
  assert.equal(back.format, 'png');
  assert.deepEqual(Array.from(back.image.data), Array.from(image.data));
});

test('PNG output of an opaque image is RGB and still lossless', () => {
  const image = rgbaImage(31, 17);
  const bytes = encodeImage(image, { format: 'png' });
  assert.equal(PNG.sync.read(bytes).colorType, 2);
  assert.deepEqual(Array.from(decodeImage(bytes, LIMITS).image.data), Array.from(image.data));
});

test('JPEG output round-trips approximately and refuses transparency', () => {
  const image = rgbaImage(32, 32);
  const bytes = encodeImage(image, { format: 'jpeg', quality: 95 });
  const back = decodeImage(bytes, LIMITS);
  assert.equal(back.format, 'jpeg');
  assert.equal(back.image.width, 32);
  assert.throws(() => encodeImage(rgbaImage(8, 8, { alpha: 128 }), { format: 'jpeg', quality: 90 }), { code: 'UNSUPPORTED' });
});

test('flatten composites transparent pixels over the background', () => {
  const image = { data: new Uint8ClampedArray([200, 100, 0, 0, 200, 100, 0, 255, 200, 100, 0, 128]), width: 3, height: 1 };
  const flat = flatten(image, [10, 20, 30]);
  assert.deepEqual(Array.from(flat.data), [10, 20, 30, 255, 200, 100, 0, 255, Math.round(200 * 128 / 255 + 10 * 127 / 255), Math.round(100 * 128 / 255 + 20 * 127 / 255), Math.round(30 * 127 / 255), 255]);
});

test('reads grey, grey+alpha and 16-bit PNGs as 8-bit RGBA', () => {
  // Pixels: a 2x1 image in each PNG colour type, built with pngjs
  const rgba = { data: Buffer.from([10, 10, 10, 255, 250, 250, 250, 128]), width: 2, height: 1 };
  const grey = PNG.sync.write({ ...rgba, data: Buffer.from([10, 10, 10, 255, 250, 250, 250, 255]) }, { colorType: 0 });
  const greyAlpha = PNG.sync.write(rgba, { colorType: 4 });
  assert.deepEqual(Array.from(decodeImage(grey, LIMITS).image.data), [10, 10, 10, 255, 250, 250, 250, 255]);
  assert.deepEqual(Array.from(decodeImage(greyAlpha, LIMITS).image.data), [10, 10, 10, 255, 250, 250, 250, 128]);
  const sixteen = PNG.sync.write({ width: 1, height: 1, data: Buffer.from(new Uint16Array([65535, 0, 32896, 65535]).buffer) }, { colorType: 6, bitDepth: 16, inputColorType: 6 });
  assert.deepEqual(Array.from(decodeImage(sixteen, LIMITS).image.data), [255, 0, 128, 255]);
});

test('reads the repository images', () => {
  const ghost = decodeImage(fs.readFileSync(path.join(IMAGES_DIR, 'ghost.png')), LIMITS);
  assert.equal(ghost.image.width, 512);
  assert.ok(Array.from(ghost.image.data).some((v, i) => i % 4 === 3 && v < 255), 'ghost.png has transparency');
  const ufo = decodeImage(fs.readFileSync(path.join(IMAGES_DIR, 'ufo_240.jpg')), LIMITS);
  assert.deepEqual([ufo.format, ufo.image.width, ufo.image.height, ufo.orientation], ['jpeg', 240, 240, 1]);
});

for (let o = 1; o <= 8; o++) {
  test(`EXIF orientation ${o} is applied like Pillow's exif_transpose`, () => {
    const fixture = orientation[o];
    const bytes = Buffer.from(fixture.png, 'base64');
    const shown = decodeImage(bytes, LIMITS);
    assert.equal(shown.orientation, o);
    assert.deepEqual([shown.image.width, shown.image.height], [fixture.width, fixture.height]);
    assert.deepEqual(rgbOf(shown.image), fixture.rgb);

    const stored = decodeImage(bytes, { ...LIMITS, orientation: 'ignore' });
    assert.deepEqual([stored.image.width, stored.image.height], [3, 2]);
    assert.deepEqual(rgbOf(stored.image), orientation[1].rgb);
  });
}

test('EXIF orientation of JPEG files is applied', () => {
  // 16x8, left half red, right half blue, orientation 6 (rotate 90 degrees clockwise)
  const bytes = Buffer.from(orientation.jpeg6, 'base64');
  const { image, orientation: found } = decodeImage(bytes, LIMITS);
  assert.equal(found, 6);
  assert.deepEqual([image.width, image.height], [8, 16]);
  const px = (x, y) => Array.from(image.data.subarray((y * 8 + x) * 4, (y * 8 + x) * 4 + 3));
  assert.ok(px(4, 2)[0] > 200 && px(4, 2)[2] < 60, 'top is red');
  assert.ok(px(4, 13)[2] > 200 && px(4, 13)[0] < 60, 'bottom is blue');
});

test('maxPixels is checked from the header before decoding', () => {
  // A PNG header that claims 100000 x 100000 pixels, with no image data
  const png = encodeImage(rgbaImage(1, 1), { format: 'png' });
  png.writeUInt32BE(100000, 16);
  png.writeUInt32BE(100000, 20);
  assert.throws(() => decodeImage(png, LIMITS), { code: 'IMAGE_TOO_LARGE' });
  assert.throws(() => decodeImage(fs.readFileSync(path.join(IMAGES_DIR, 'ripley.jpg')), { maxPixels: 1_000_000 }), { code: 'IMAGE_TOO_LARGE' });
  assert.throws(() => decodeImage({ data: new Uint8Array(4 * 100), width: 10, height: 10 }, { maxPixels: 99 }), { code: 'IMAGE_TOO_LARGE' });
});

test('rejects unknown, truncated and corrupt image bytes', () => {
  assert.throws(() => decodeImage(Buffer.from('GIF89a'), LIMITS), { code: 'INVALID_IMAGE' });
  const png = encodeImage(rgbaImage(16, 16), { format: 'png' });
  assert.throws(() => decodeImage(png.subarray(0, 40), LIMITS), { code: 'INVALID_IMAGE' });
  const corrupt = Buffer.from(png);
  corrupt[corrupt.length - 20] ^= 0xff;
  assert.throws(() => decodeImage(corrupt, LIMITS), { code: 'INVALID_IMAGE' });
  const jpeg = fs.readFileSync(path.join(IMAGES_DIR, 'ufo_240.jpg'));
  assert.throws(() => decodeImage(jpeg.subarray(0, 200), LIMITS), { code: 'INVALID_IMAGE' });
});

test('accepts raw RGBA and RGB pixels and validates them', () => {
  const rgba = rgbaImage(4, 3);
  assert.equal(decodeImage(rgba, LIMITS).image.data, rgba.data, 'RGBA input is used as is');
  const rgb = { data: new Uint8Array([1, 2, 3, 4, 5, 6]), width: 2, height: 1, channels: 3 };
  assert.deepEqual(Array.from(decodeImage(rgb, LIMITS).image.data), [1, 2, 3, 255, 4, 5, 6, 255]);
  assert.throws(() => decodeImage({ data: new Uint8Array(10), width: 2, height: 2 }, LIMITS), { code: 'INVALID_ARGUMENT' });
  assert.throws(() => decodeImage({ data: new Float32Array(16), width: 2, height: 2 }, LIMITS), { code: 'INVALID_ARGUMENT' });
  assert.throws(() => decodeImage('image.png', LIMITS), { code: 'INVALID_ARGUMENT' });
  assert.equal(decodeImage(new Uint8Array(encodeImage(rgba, { format: 'png' })).buffer, LIMITS).image.width, 4, 'ArrayBuffer input');
});
