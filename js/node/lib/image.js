'use strict';

// Image input and output: PNG and JPEG bytes or raw pixels in, PNG or JPEG bytes out, with
// the pure JavaScript pngjs and jpeg-js codecs. Images are RGBA { data, width, height }.

const { PNG } = require('pngjs');
const jpeg = require('jpeg-js');

const { TrustmarkError } = require('./errors');

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

function invalid(message, cause) {
  return new TrustmarkError('INVALID_IMAGE', message, { cause });
}

function sniffFormat(bytes) {
  if (bytes.length >= 8 && PNG_SIGNATURE.equals(bytes.subarray(0, 8))) return 'png';
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return 'jpeg';
  return null;
}

/** Reads the EXIF orientation (1 to 8) from a TIFF structure, or 1 when absent or invalid. */
function tiffOrientation(tiff) {
  if (tiff.length < 8) return 1;
  const order = tiff.toString('latin1', 0, 2);
  if (order !== 'II' && order !== 'MM') return 1;
  const le = order === 'II';
  const u16 = (o) => (le ? tiff.readUInt16LE(o) : tiff.readUInt16BE(o));
  const u32 = (o) => (le ? tiff.readUInt32LE(o) : tiff.readUInt32BE(o));
  if (u16(2) !== 42) return 1;
  const ifd = u32(4);
  if (ifd + 2 > tiff.length) return 1;
  const entries = u16(ifd);
  for (let i = 0; i < entries; i++) {
    const entry = ifd + 2 + i * 12;
    if (entry + 12 > tiff.length) return 1;
    if (u16(entry) === 0x0112) {
      // SHORT, count 1: the value sits in the first two bytes of the value field
      const value = u16(entry + 2) === 3 && u32(entry + 4) === 1 ? u16(entry + 8) : 1;
      return value >= 1 && value <= 8 ? value : 1;
    }
  }
  return 1;
}

/** Width, height and EXIF orientation from the PNG chunks, without decoding the pixels. */
function pngHeader(bytes) {
  if (bytes.length < 33 || bytes.toString('latin1', 12, 16) !== 'IHDR') throw invalid('Invalid PNG: missing IHDR chunk');
  const width = bytes.readUInt32BE(16);
  const height = bytes.readUInt32BE(20);
  let orientation = 1;
  let offset = 8;
  while (offset + 12 <= bytes.length) {
    const length = bytes.readUInt32BE(offset);
    const type = bytes.toString('latin1', offset + 4, offset + 8);
    const end = offset + 12 + length;
    if (end > bytes.length) break;
    if (type === 'eXIf') {
      orientation = tiffOrientation(bytes.subarray(offset + 8, offset + 8 + length));
      break;
    }
    if (type === 'IDAT' || type === 'IEND') break;  // eXIf must come before IDAT
    offset = end;
  }
  return { width, height, orientation };
}

/** Width, height, components and EXIF orientation from the JPEG markers before the scan. */
function jpegHeader(bytes) {
  let orientation = null;
  let frame = null;
  let offset = 2;
  while (offset + 4 <= bytes.length) {
    if (bytes[offset] !== 0xff) throw invalid('Invalid JPEG: bad marker');
    const marker = bytes[offset + 1];
    if (marker === 0xff) { offset++; continue; }  // fill byte
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) { offset += 2; continue; }
    if (marker === 0xd9 || marker === 0xda) break;  // end of image or start of scan
    const length = bytes.readUInt16BE(offset + 2);
    const start = offset + 4;
    const end = offset + 2 + length;
    if (length < 2 || end > bytes.length) throw invalid('Invalid JPEG: truncated segment');
    if (marker === 0xe1 && orientation === null && length >= 8 && bytes.toString('latin1', start, start + 6) === 'Exif\0\0') {
      orientation = tiffOrientation(bytes.subarray(start + 6, end));
    } else if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc && frame === null) {
      if (length < 8) throw invalid('Invalid JPEG: truncated frame header');
      frame = { height: bytes.readUInt16BE(start + 1), width: bytes.readUInt16BE(start + 3), components: bytes[start + 5] };
    }
    offset = end;
  }
  if (!frame) throw invalid('Invalid JPEG: no frame header before the image data');
  return { ...frame, orientation: orientation || 1 };
}

function checkPixels(width, height, maxPixels) {
  if (!(width > 0 && height > 0)) throw invalid(`Invalid image size ${width}x${height}`);
  if (width * height > maxPixels) {
    throw new TrustmarkError('IMAGE_TOO_LARGE', `The image has ${width}x${height} = ${width * height} pixels, more than maxPixels (${maxPixels})`, { details: { width, height, maxPixels } });
  }
}

/**
 * Applies an EXIF orientation (1 to 8) to RGBA pixels, so that the result is the image as
 * viewers display it.
 */
function applyOrientation(image, orientation) {
  if (orientation === 1) return image;
  const { width: W, height: H, data } = image;
  const swap = orientation >= 5;
  const width = swap ? H : W;
  const height = swap ? W : H;
  const out = new Uint8ClampedArray(width * height * 4);
  // Source pixel (x, y) of each displayed pixel (dx, dy)
  const source = {
    2: (dx, dy) => [W - 1 - dx, dy],
    3: (dx, dy) => [W - 1 - dx, H - 1 - dy],
    4: (dx, dy) => [dx, H - 1 - dy],
    5: (dx, dy) => [dy, dx],
    6: (dx, dy) => [dy, H - 1 - dx],
    7: (dx, dy) => [W - 1 - dy, H - 1 - dx],
    8: (dx, dy) => [W - 1 - dy, dx],
  }[orientation];
  let k = 0;
  for (let dy = 0; dy < height; dy++) {
    for (let dx = 0; dx < width; dx++, k += 4) {
      const [x, y] = source(dx, dy);
      const i = (y * W + x) * 4;
      out[k] = data[i];
      out[k + 1] = data[i + 1];
      out[k + 2] = data[i + 2];
      out[k + 3] = data[i + 3];
    }
  }
  return { data: out, width, height };
}

function isRawImage(input) {
  return input !== null && typeof input === 'object' && !ArrayBuffer.isView(input) && !(input instanceof ArrayBuffer) && 'data' in input;
}

function fromRaw(input, maxPixels) {
  const { data, width, height } = input;
  const channels = input.channels === undefined ? 4 : input.channels;
  if (!Number.isInteger(width) || !Number.isInteger(height)) throw new TrustmarkError('INVALID_ARGUMENT', 'Raw images need integer width and height');
  if (channels !== 3 && channels !== 4) throw new TrustmarkError('INVALID_ARGUMENT', 'Raw images must have 3 (RGB) or 4 (RGBA) channels');
  if (!(data instanceof Uint8Array || data instanceof Uint8ClampedArray)) {
    throw new TrustmarkError('INVALID_ARGUMENT', 'Raw image data must be a Uint8Array, Uint8ClampedArray or Buffer of 8-bit samples');
  }
  checkPixels(width, height, maxPixels);
  if (data.length !== width * height * channels) {
    throw new TrustmarkError('INVALID_ARGUMENT', `Raw image data has ${data.length} bytes, expected ${width}x${height}x${channels} = ${width * height * channels}`);
  }
  if (channels === 4) return { image: { data, width, height }, format: 'raw', orientation: 1 };
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0, j = 0; i < data.length; i += 3, j += 4) {
    rgba[j] = data[i];
    rgba[j + 1] = data[i + 1];
    rgba[j + 2] = data[i + 2];
    rgba[j + 3] = 255;
  }
  return { image: { data: rgba, width, height }, format: 'raw', orientation: 1 };
}

/**
 * Reads an input image.
 *
 * @param {Uint8Array|ArrayBuffer|{data, width, height, channels?}} input - PNG or JPEG bytes,
 *   or raw 8-bit RGBA (or RGB with channels: 3) pixels.
 * @param {{maxPixels: number, orientation: 'apply'|'ignore', tolerant?: boolean}} options
 * @returns {{image: {data, width, height}, format: 'png'|'jpeg'|'raw', orientation: number}}
 *   The RGBA image (oriented for display when orientation is 'apply'), the input format and
 *   the EXIF orientation found in the file.
 */
function decodeImage(input, { maxPixels, orientation: policy = 'apply', tolerant = false }) {
  if (isRawImage(input)) return fromRaw(input, maxPixels);
  let bytes;
  if (input instanceof ArrayBuffer) bytes = Buffer.from(input);
  else if (input instanceof Uint8Array) bytes = Buffer.from(input.buffer, input.byteOffset, input.byteLength);
  else throw new TrustmarkError('INVALID_ARGUMENT', 'The image must be PNG or JPEG bytes (Buffer, Uint8Array or ArrayBuffer) or raw pixels { data, width, height }');

  const format = sniffFormat(bytes);
  if (format === null) throw invalid('Unsupported image format: expected PNG or JPEG bytes');
  const header = format === 'png' ? pngHeader(bytes) : jpegHeader(bytes);
  checkPixels(header.width, header.height, maxPixels);

  let image;
  try {
    if (format === 'png') {
      const png = PNG.sync.read(bytes);  // any colour type and bit depth, converted to 8-bit RGBA
      image = { data: new Uint8ClampedArray(png.data.buffer, png.data.byteOffset, png.data.length), width: png.width, height: png.height };
    } else {
      const decoded = jpeg.decode(bytes, {
        useTArray: true,
        formatAsRGBA: true,
        tolerantDecoding: tolerant,
        maxResolutionInMP: Math.max(1, Math.ceil(maxPixels / 1e6)),
        maxMemoryUsageInMB: Math.max(512, Math.ceil((maxPixels * 16) / 2 ** 20)),
      });
      image = { data: new Uint8ClampedArray(decoded.data.buffer, decoded.data.byteOffset, decoded.data.length), width: decoded.width, height: decoded.height };
    }
  } catch (error) {
    throw invalid(`Could not decode the ${format.toUpperCase()} image: ${error && error.message ? error.message : error}`, error);
  }
  if (image.width !== header.width || image.height !== header.height || image.data.length !== image.width * image.height * 4) {
    throw invalid('The decoded image does not match its header');
  }
  if (policy === 'apply') image = applyOrientation(image, header.orientation);
  return { image, format, orientation: header.orientation };
}

function isOpaque(data) {
  for (let i = 3; i < data.length; i += 4) if (data[i] !== 255) return false;
  return true;
}

/** Composites RGBA pixels over an opaque background colour [r, g, b]. */
function flatten(image, background) {
  const { data } = image;
  const out = new Uint8ClampedArray(data.length);
  for (let i = 0; i < data.length; i += 4) {
    const a = data[i + 3] / 255;
    out[i] = Math.round(data[i] * a + background[0] * (1 - a));
    out[i + 1] = Math.round(data[i + 1] * a + background[1] * (1 - a));
    out[i + 2] = Math.round(data[i + 2] * a + background[2] * (1 - a));
    out[i + 3] = 255;
  }
  return { data: out, width: image.width, height: image.height };
}

/**
 * Encodes RGBA pixels. PNG is lossless and keeps the alpha channel (an RGB PNG is written
 * when every pixel is opaque); JPEG needs opaque pixels. No metadata is written.
 */
function encodeImage(image, { format, quality }) {
  const { data, width, height } = image;
  if (format === 'png') {
    const buffer = Buffer.from(data.buffer, data.byteOffset, data.length);
    const colorType = isOpaque(data) ? 2 : 6;
    // Paeth filtering on every row: about the size of pngjs's adaptive filtering, twice as fast
    return PNG.sync.write({ width, height, data: buffer }, { colorType, inputColorType: 6, inputHasAlpha: true, deflateLevel: 6, filterType: 4 });
  }
  if (format === 'jpeg') {
    if (!isOpaque(data)) throw new TrustmarkError('UNSUPPORTED', 'JPEG cannot store transparency');
    return jpeg.encode({ data, width, height }, quality).data;
  }
  throw new TrustmarkError('UNSUPPORTED', `Unsupported output format ${JSON.stringify(format)}`);
}

module.exports = { decodeImage, encodeImage, flatten, isOpaque };
