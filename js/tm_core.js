/*!
 * TrustMark JS core: image pre-processing and watermark embedding math
 * Copyright 2024 Adobe. All rights reserved.
 * Licensed under the MIT License.
 *
 * NOTICE: Adobe permits you to use, modify, and distribute this file in
 * accordance with the terms of the Adobe license agreement accompanying it.
 */

/*
 * This file has no DOM or global dependencies. ONNX Runtime and the sessions are passed in,
 * so the same code runs in the browser demo and in Node.js. Images are ImageData-like
 * objects: { data: RGBA bytes, width, height }.
 *
 * Most functions take a `runtime` object:
 *   { ort, resizer, run }
 *   - ort: the onnxruntime module (onnxruntime-web or onnxruntime-node), used for ort.Tensor
 *   - resizer: an InferenceSession of resizer.onnx (antialiased downscaler, WASM/CPU only), used
 *     by the decoders; embedding resizes in JavaScript and does not need it
 *   - run: optional (session, feeds) => Promise<outputs>, e.g. a queue that serializes WebGPU
 *     inference; defaults to session.run(feeds)
 *
 * It is a classic script for the no-build demo (it defines the global `TrustmarkCore`) and a
 * CommonJS module in Node.js (`require('./tm_core.js')`).
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  } else {
    root.TrustmarkCore = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {

  // Values from python/trustmark/trustmark.py
  const ENCODER_RESOLUTION = 256;
  const ASPECT_RATIO_LIM = 2.0;      // wider or taller images are processed as a centre square
  const FEATHERING_RESIDUAL = 0.01;  // feather width as a fraction of the processed region
  const MAX_FEATHER = 50;
  const PACKET_BITS = 100;

  /**
   * Settings of each TrustMark variant, shared by embedding and decoding, from TrustMark.__init__
   * and TrustMark.encode in the Python implementation:
   *   - encoder, decoder: ONNX model names (`${name}.onnx`)
   *   - encodeResolution, decodeResolution: model input sizes. The Python decoders of B, C and Q
   *     run at 245x245, but their ONNX exports take 256x256 inputs.
   *   - forceSquare: always process the centred square (P), instead of only above a 2:1 aspect ratio
   *   - strengthMultiplier: factor applied to the requested watermark strength (P: 1.25)
   */
  const VARIANTS = Object.freeze({
    Q: Object.freeze({ code: 'Q', encoder: 'encoder_Q', decoder: 'decoder_Q', encodeResolution: ENCODER_RESOLUTION, decodeResolution: 256, forceSquare: false, strengthMultiplier: 1.0 }),
    P: Object.freeze({ code: 'P', encoder: 'encoder_P', decoder: 'decoder_P', encodeResolution: ENCODER_RESOLUTION, decodeResolution: 224, forceSquare: true, strengthMultiplier: 1.25 }),
    B: Object.freeze({ code: 'B', encoder: 'encoder_B', decoder: 'decoder_B', encodeResolution: ENCODER_RESOLUTION, decodeResolution: 256, forceSquare: false, strengthMultiplier: 1.0 }),
    C: Object.freeze({ code: 'C', encoder: 'encoder_C', decoder: 'decoder_C', encodeResolution: ENCODER_RESOLUTION, decodeResolution: 256, forceSquare: false, strengthMultiplier: 1.0 }),
  });

  /**
   * Returns the settings of a variant.
   * @param {string} code - Variant code: 'Q', 'P', 'B' or 'C'.
   * @returns {Object} The VARIANTS entry.
   * @throws {Error} For an unknown variant.
   */
  function getVariant(code) {
    if (typeof code !== 'string' || !Object.prototype.hasOwnProperty.call(VARIANTS, code)) {
      throw new Error(`Unknown TrustMark variant ${JSON.stringify(code)}; expected one of ${Object.keys(VARIANTS).join(', ')}`);
    }
    return VARIANTS[code];
  }

  /**
   * Validates the options of encodeWatermark and applies their defaults.
   * @param {Object} [options] - { variant = 'Q', strength = 1.0 }.
   * @returns {{variant: Object, strength: number}} The VARIANTS entry and the requested strength
   *   (before the variant's multiplier).
   * @throws {Error} For an unknown variant or a negative or non-numeric strength.
   */
  function resolveEncodeOptions(options = {}) {
    const variant = getVariant(options.variant === undefined ? 'Q' : options.variant);
    const strength = options.strength === undefined ? 1.0 : options.strength;
    if (!(typeof strength === 'number' && strength >= 0 && Number.isFinite(strength))) {
      throw new Error('strength must be a non-negative number');
    }
    return { variant, strength };
  }

  /**
   * Returns all-zero inputs for a model, e.g. for a first inference that absorbs one-time costs
   * (WebAssembly thread start-up, WebGPU shader compilation).
   * @param {Object} ort - The onnxruntime module.
   * @param {string} name - Model name: 'resizer', 'encoder_<variant>' or 'decoder_<variant>'.
   * @returns {Object} The feeds (a small 8x8 image for the resizer).
   * @throws {Error} For another model name.
   */
  function zeroFeeds(ort, name) {
    if (name === 'resizer') {
      const dims = [1, 3, 8, 8];
      return {
        'X': new ort.Tensor('float32', new Float32Array(3 * 8 * 8), dims),
        'scales': new ort.Tensor('float32', computeScalesFixed([4, 4], dims), [4]),
        'target_size': new ort.Tensor('int64', new BigInt64Array([4n]), [1]),
      };
    }
    const match = /^(encoder|decoder)_(.+)$/.exec(name);
    if (!match) throw new Error(`no zero inputs are defined for the ${name} model`);
    const variant = getVariant(match[2]);
    if (match[1] === 'decoder') {
      const size = variant.decodeResolution;
      return { image: new ort.Tensor('float32', new Float32Array(3 * size * size), [1, 3, size, size]) };
    }
    const size = variant.encodeResolution;
    return {
      'onnx::Concat_0': new ort.Tensor('float32', new Float32Array(3 * size * size), [1, 3, size, size]),
      'onnx::Gemm_1': new ort.Tensor('float32', new Float32Array(PACKET_BITS), [1, PACKET_BITS]),
    };
  }

  /**
   * Returns the part of the image the watermark models see: the whole image, or the centred
   * square when the aspect ratio exceeds 2:1 (or when forceSquare is set, as for variant P).
   * Matches TrustMark.get_the_image_for_processing in the Python implementation.
   *
   * @param {number} width - Image width.
   * @param {number} height - Image height.
   * @param {boolean} [forceSquare=false] - Always use the centred square.
   * @returns {{left: number, top: number, width: number, height: number}} The region.
   */
  function processingRegion(width, height, forceSquare = false) {
    const aspectRatio = Math.max(width, height) / Math.min(width, height);
    if (aspectRatio > ASPECT_RATIO_LIM || forceSquare) {
      const size = Math.min(width, height);
      return { left: Math.floor((width - size) / 2), top: Math.floor((height - size) / 2), width: size, height: size };
    }
    return { left: 0, top: 0, width, height };
  }

  // v / 255 in float32 for every byte value v, as transforms.ToTensor() computes it
  const BYTE_TO_UNIT = Float32Array.from({ length: 256 }, (_, v) => v / 255.0);
  // v / 127.5 - 1 in float64 for every byte value v, as NumPy computes the cover in TrustMark.encode
  const COVER_TO_UNIT = Float64Array.from({ length: 256 }, (_, v) => v / 127.5 - 1);

  /**
   * Extracts a region of an RGBA image as planar RGB floats in [0, 1] (CHW layout).
   * The alpha channel is ignored.
   *
   * @param {{data: ArrayLike<number>, width: number}} image - The RGBA image.
   * @param {{left: number, top: number, width: number, height: number}} region - The region.
   * @returns {Float32Array} 3 * region.height * region.width values.
   */
  function rgbaToPlanar(image, region) {
    const { data } = image;
    const plane = region.width * region.height;
    const out = new Float32Array(3 * plane);
    let k = 0;
    for (let y = 0; y < region.height; y++) {
      let i = ((region.top + y) * image.width + region.left) * 4;
      for (let x = 0; x < region.width; x++, i += 4, k++) {
        out[k] = BYTE_TO_UNIT[data[i]];
        out[k + plane] = BYTE_TO_UNIT[data[i + 1]];
        out[k + 2 * plane] = BYTE_TO_UNIT[data[i + 2]];
      }
    }
    return out;
  }

  // Pillow's fixed-point precision for 8-bit images (PRECISION_BITS = 32 - 8 - 2 in Resample.c)
  const PILLOW_PRECISION_BITS = 22;
  const PILLOW_ONE = 1 << PILLOW_PRECISION_BITS;

  /**
   * Resampling bounds and fixed-point weights for one axis of pillowBilinearResize
   * (precompute_coeffs and normalize_coeffs_8bpc in Pillow's Resample.c, bilinear filter).
   * @param {number} inSize - Input size.
   * @param {number} outSize - Output size.
   * @returns {{ksize: number, bounds: Int32Array, weights: Int32Array}} Output index i reads
   *   bounds[2i + 1] inputs from bounds[2i] on, with the weights weights[i * ksize + j].
   */
  function pillowCoefficients(inSize, outSize) {
    const scale = inSize / outSize;
    const filterscale = Math.max(1.0, scale);  // antialiasing: widen the filter when reducing
    const support = 1.0 * filterscale;         // the triangle filter has a support of 1
    const ss = 1.0 / filterscale;
    const ksize = Math.ceil(support) * 2 + 1;
    const bounds = new Int32Array(outSize * 2);
    const weights = new Int32Array(outSize * ksize);
    const kernel = new Float64Array(ksize);
    for (let xx = 0; xx < outSize; xx++) {
      const center = (xx + 0.5) * scale;
      // C truncates towards zero when converting to int
      const xmin = Math.max(0, Math.trunc(center - support + 0.5));
      const count = Math.min(inSize, Math.trunc(center + support + 0.5)) - xmin;
      let total = 0.0;
      for (let x = 0; x < count; x++) {
        const w = Math.max(0.0, 1.0 - Math.abs((x + xmin - center + 0.5) * ss));
        kernel[x] = w;
        total += w;
      }
      for (let x = 0; x < count; x++) {
        const w = total !== 0.0 ? kernel[x] / total : kernel[x];
        weights[xx * ksize + x] = Math.trunc(0.5 + w * PILLOW_ONE);  // the weights are never negative
      }
      bounds[xx * 2] = xmin;
      bounds[xx * 2 + 1] = count;
    }
    return { ksize, bounds, weights };
  }

  /**
   * Rounds and clips a Pillow fixed-point sum to a byte (clip8 in Resample.c).
   * @param {number} sum - The sum, including the rounding offset.
   * @returns {number} 0 to 255.
   */
  function pillowClip8(sum) {
    if (sum >= PILLOW_ONE * 256) return 255;
    if (sum <= 0) return 0;
    return sum >> PILLOW_PRECISION_BITS;
  }

  /**
   * Resizes a region of an RGBA image like Pillow's `Image.resize(size, Image.BILINEAR)` on an
   * RGB image, which the Python implementation uses before its models: an antialiased triangle
   * filter in two passes (horizontal, then vertical) with Pillow's 8-bit fixed-point arithmetic
   * and rounding after each pass, so the result is bit-exact. The alpha channel is ignored.
   * (Pillow 12 resizes images more than 100 times taller than wide vertically first; the regions
   * TrustMark processes are at most 2:1.)
   *
   * @param {{data: ArrayLike<number>, width: number}} image - The RGBA image.
   * @param {{left: number, top: number, width: number, height: number}} region - The region to resize.
   * @param {number} outWidth - Output width.
   * @param {number} outHeight - Output height.
   * @returns {Uint8Array} Planar RGB bytes (CHW layout), 3 * outHeight * outWidth values.
   */
  function pillowBilinearResize(image, region, outWidth, outHeight) {
    const { data } = image;
    const stride = image.width * 4;
    const half = PILLOW_ONE >> 1;
    const vertical = pillowCoefficients(region.height, outHeight);

    // Source of the vertical pass: the horizontally resized rows (RGB) or the image itself (RGBA),
    // addressed as src[origin + row * rowStride + x * step] for region rows and output columns
    let src = data;
    let origin = region.top * stride + region.left * 4;
    let rowStride = stride;
    let step = 4;
    if (outWidth !== region.width) {  // Pillow skips a pass whose size does not change
      // Only the region rows that the vertical pass reads
      const firstRow = vertical.bounds[0];
      const rows = vertical.bounds[2 * outHeight - 2] + vertical.bounds[2 * outHeight - 1] - firstRow;
      const { ksize, bounds, weights } = pillowCoefficients(region.width, outWidth);
      const temp = new Uint8Array(rows * outWidth * 3);
      for (let y = 0, o = 0; y < rows; y++) {
        const rowStart = origin + (firstRow + y) * stride;
        for (let xx = 0; xx < outWidth; xx++, o += 3) {
          const count = bounds[xx * 2 + 1];
          const k = xx * ksize;
          let r = half;
          let g = half;
          let b = half;
          for (let x = 0, i = rowStart + bounds[xx * 2] * 4; x < count; x++, i += 4) {
            const w = weights[k + x];
            r = (r + Math.imul(data[i], w)) | 0;  // int32, like the C code
            g = (g + Math.imul(data[i + 1], w)) | 0;
            b = (b + Math.imul(data[i + 2], w)) | 0;
          }
          temp[o] = pillowClip8(r);
          temp[o + 1] = pillowClip8(g);
          temp[o + 2] = pillowClip8(b);
        }
      }
      src = temp;
      rowStride = outWidth * 3;
      origin = -firstRow * rowStride;
      step = 3;
    }

    const plane = outWidth * outHeight;
    const out = new Uint8Array(3 * plane);
    const { ksize, bounds, weights } = vertical;
    const resize = outHeight !== region.height;
    for (let yy = 0, o = 0; yy < outHeight; yy++) {
      const count = bounds[yy * 2 + 1];
      const k = yy * ksize;
      for (let x = 0; x < outWidth; x++, o++) {
        let i = origin + (resize ? bounds[yy * 2] : yy) * rowStride + x * step;
        if (!resize) {
          out[o] = src[i];
          out[o + plane] = src[i + 1];
          out[o + 2 * plane] = src[i + 2];
          continue;
        }
        let r = half;
        let g = half;
        let b = half;
        for (let y = 0; y < count; y++, i += rowStride) {
          const w = weights[k + y];
          r = (r + Math.imul(src[i], w)) | 0;
          g = (g + Math.imul(src[i + 1], w)) | 0;
          b = (b + Math.imul(src[i + 2], w)) | 0;
        }
        out[o] = pillowClip8(r);
        out[o + plane] = pillowClip8(g);
        out[o + 2 * plane] = pillowClip8(b);
      }
    }
    return out;
  }

  /**
   * Computes scale factors for image resizing with precision.
   * @param {Array<number>} targetDims - Target dimensions for the image.
   * @param {Array<number>} inputDims - Input tensor dimensions.
   * @returns {Float32Array} Scale factors as a tensor.
   */
  function computeScalesFixed(targetDims, inputDims) {
    const [batch, channels, height, width] = inputDims;
    const [targetHeight, targetWidth] = targetDims;

    function computeScale(originalSize, targetSize) {
      let minScale = targetSize / originalSize;
      let maxScale = (targetSize + 1) / originalSize;
      let scale;
      let adjustedSize;

      const tolerance = 1e-12;
      let iterations = 0;
      const maxIterations = 100;

      while (iterations < maxIterations) {
        scale = (minScale + maxScale) / 2;
        adjustedSize = Math.floor(originalSize * scale + tolerance);

        if (adjustedSize < targetSize) {
          minScale = scale;
        } else if (adjustedSize > targetSize) {
          maxScale = scale;
        } else {
          break; // Found the correct scale
        }

        iterations++;
      }

      return scale;
    }

    const scaleH = computeScale(height, targetHeight);
    const scaleW = computeScale(width, targetWidth);

    return new Float32Array([1.0, 1.0, scaleH, scaleW]);
  }

  /**
   * Runs a session through runtime.run, or directly if no runner was given.
   * @param {Object} runtime - { run } (see the top of this file).
   * @param {Object} session - The InferenceSession.
   * @param {Object} feeds - The input feeds.
   * @returns {Promise<Object>} The outputs.
   */
  function runSession(runtime, session, feeds) {
    return runtime.run ? runtime.run(session, feeds) : session.run(feeds);
  }

  /**
   * Crops the processing region and resizes it to size x size with the antialiased
   * resizer model. This is the input of every watermark model.
   *
   * @param {Object} runtime - { ort, resizer, run } (see the top of this file).
   * @param {{data: ArrayLike<number>, width: number, height: number}} image - The RGBA image.
   * @param {number} size - Model resolution.
   * @param {boolean} [forceSquare=false] - Always use the centred square (variant P).
   * @returns {Promise<{region: Object, pixels: Float32Array}>} The region and the resized
   *   planar RGB values in [0, 1] (3 * size * size).
   */
  async function preprocess(runtime, image, size, forceSquare = false) {
    const { ort, resizer } = runtime;
    if (!resizer) throw new Error('resizer model is not loaded');

    const region = processingRegion(image.width, image.height, forceSquare);
    const dims = [1, 3, region.height, region.width];
    const feeds = {
      'X': new ort.Tensor('float32', rgbaToPlanar(image, region), dims),
      'scales': new ort.Tensor('float32', computeScalesFixed([size, size], dims), [4]),
      'target_size': new ort.Tensor('int64', new BigInt64Array([BigInt(size)]), [1]),
    };
    const results = await runSession(runtime, resizer, feeds);
    return { region, pixels: results['Y'].data };
  }

  /**
   * Runs a watermark decoder on an image and returns the raw packet bits.
   * The bits still need error correction, e.g. with DataLayer_Decode.
   * The decoder sees the resized pixels in [-1, 1], the range the decoders were trained on and
   * that the Python (ToTensor() * 2 - 1) and Rust implementations feed them.
   *
   * @param {Object} runtime - { ort, resizer, run } (see the top of this file).
   * @param {Object} decoder - InferenceSession of the decoder model of the variant.
   * @param {{data: ArrayLike<number>, width: number, height: number}} image - The RGBA image.
   * @param {string} [variant='Q'] - Variant code of the decoder; selects its resolution and crop
   *   (see VARIANTS).
   * @returns {Promise<Array<boolean>>} The 100 packet bits.
   */
  async function decodeBits(runtime, decoder, image, variant = 'Q') {
    const { decodeResolution: resolution, forceSquare } = getVariant(variant);
    const { pixels } = await preprocess(runtime, image, resolution, forceSquare);
    const scaled = Float32Array.from(pixels, (v) => v * 2.0 - 1.0);  // [0, 1] -> [-1, 1]
    const input = new runtime.ort.Tensor('float32', scaled, [1, 3, resolution, resolution]);
    const results = await runSession(runtime, decoder, { image: input });
    return Array.from(results['output'].data, (v) => v >= 0);
  }

  /**
   * Source indices and weights of a bilinear resize along one axis, with the half-pixel mapping
   * of torch.nn.functional.interpolate (mode='bilinear', align_corners=False).
   * @param {number} inSize - Input size.
   * @param {number} outSize - Output size.
   * @returns {{i0: Int32Array, i1: Int32Array, w1: Float64Array}} Output index d interpolates
   *   inputs i0[d] and i1[d] with the weights 1 - w1[d] and w1[d].
   */
  function bilinearAxis(inSize, outSize) {
    const scale = inSize / outSize;
    const i0 = new Int32Array(outSize);
    const i1 = new Int32Array(outSize);
    const w1 = new Float64Array(outSize);
    for (let d = 0; d < outSize; d++) {
      const real = Math.max(0, scale * (d + 0.5) - 0.5);
      const index = Math.floor(real);
      i0[d] = index;
      i1[d] = index < inSize - 1 ? index + 1 : index;
      w1[d] = real - index;
    }
    return { i0, i1, w1 };
  }

  /**
   * Bilinear resize of planar float data, matching torch.nn.functional.interpolate with
   * mode='bilinear', align_corners=False and no antialiasing.
   *
   * @param {Float32Array} src - channels * inH * inW values.
   * @param {number} channels - Number of channels.
   * @param {number} inH - Input height.
   * @param {number} inW - Input width.
   * @param {number} outH - Output height.
   * @param {number} outW - Output width.
   * @returns {Float32Array} channels * outH * outW values.
   */
  function bilinearResize(src, channels, inH, inW, outH, outW) {
    const ys = bilinearAxis(inH, outH);
    const xs = bilinearAxis(inW, outW);

    const out = new Float32Array(channels * outH * outW);
    let k = 0;
    for (let c = 0; c < channels; c++) {
      const base = c * inH * inW;
      for (let y = 0; y < outH; y++) {
        const row0 = base + ys.i0[y] * inW;
        const row1 = base + ys.i1[y] * inW;
        const wy1 = ys.w1[y];
        const wy0 = 1 - wy1;
        for (let x = 0; x < outW; x++, k++) {
          const wx1 = xs.w1[x];
          const wx0 = 1 - wx1;
          out[k] = wy0 * (wx0 * src[row0 + xs.i0[x]] + wx1 * src[row0 + xs.i1[x]]) +
                   wy1 * (wx0 * src[row1 + xs.i0[x]] + wx1 * src[row1 + xs.i1[x]]);
        }
      }
    }
    return out;
  }

  /**
   * Returns the feather width used when pasting the watermarked region back,
   * as in TrustMark.put_the_image_after_processing.
   * @param {number} regionWidth - Region width.
   * @param {number} regionHeight - Region height.
   * @returns {number} Feather width in pixels (1 to 50).
   */
  function featherSize(regionWidth, regionHeight) {
    const size = Math.floor(Math.min(regionWidth, regionHeight) * FEATHERING_RESIDUAL);
    return Math.min(Math.max(1, size), MAX_FEATHER);
  }

  /**
   * Adds a residual to the processing region of an image and pastes it back with feathered
   * edges, like the Python encoder (TrustMark.encode followed by put_the_image_after_processing).
   * A residual smaller than the region (the 256x256 encoder output) is upscaled on the fly with
   * exactly the arithmetic of bilinearResize, so no full-resolution copy of it is allocated.
   * Pixels outside the region and the alpha channel are copied unchanged.
   *
   * @param {{data: ArrayLike<number>, width: number, height: number}} image - The RGBA cover image.
   * @param {{left: number, top: number, width: number, height: number}} region - The processed region.
   * @param {Float32Array} residual - Planar RGB residual in [-1, 1] units,
   *   3 * residualHeight * residualWidth values.
   * @param {number} strength - Residual multiplier (WM_STRENGTH).
   * @param {number} [residualWidth=region.width] - Width of the residual.
   * @param {number} [residualHeight=region.height] - Height of the residual.
   * @returns {Uint8ClampedArray} The watermarked RGBA pixels.
   */
  function applyResidual(image, region, residual, strength, residualWidth = region.width, residualHeight = region.height) {
    const out = new Uint8ClampedArray(image.data);
    const { left, top, width: rw, height: rh } = region;
    const plane = residualWidth * residualHeight;
    const feather = featherSize(rw, rh);
    const strength32 = Math.fround(strength);  // NumPy multiplies the float32 residual in float32
    // A residual of the region size maps onto itself with weights 1 and 0
    const ys = bilinearAxis(residualHeight, rh);
    const xs = bilinearAxis(residualWidth, rw);

    // Horizontally interpolated residual rows (3 * rw values), kept in float64 like the
    // intermediate sums of bilinearResize; consecutive region rows mostly share their source rows
    function interpolateRow(row, target) {
      for (let c = 0, p = row * residualWidth, o = 0; c < 3; c++, p += plane) {
        for (let rx = 0; rx < rw; rx++, o++) {
          const wx1 = xs.w1[rx];
          target[o] = (1 - wx1) * residual[p + xs.i0[rx]] + wx1 * residual[p + xs.i1[rx]];
        }
      }
    }
    let rows = [new Float64Array(3 * rw), new Float64Array(3 * rw)];
    let loaded = [-1, -1];

    for (let ry = 0; ry < rh; ry++) {
      const sources = [ys.i0[ry], ys.i1[ry]];
      if (sources[0] === loaded[1] && sources[0] !== loaded[0]) {  // moving down: reuse the lower row
        rows = [rows[1], rows[0]];
        loaded = [loaded[1], loaded[0]];
      }
      for (let j = 0; j < 2; j++) {
        if (loaded[j] !== sources[j]) {
          interpolateRow(sources[j], rows[j]);
          loaded[j] = sources[j];
        }
      }
      const [top0, top1] = rows;
      const wy1 = ys.w1[ry];
      const wy0 = 1 - wy1;
      // Python pastes the region, then blends the top, bottom, left and right bands in that
      // order, each overwriting the previous; the last band covering a pixel decides its weight.
      const rowBand = ry >= rh - feather ? rh - 1 - ry : (ry < feather ? ry : -1);
      for (let rx = 0; rx < rw; rx++) {
        let band = rowBand;
        if (rx >= rw - feather) band = rw - 1 - rx;
        else if (rx < feather) band = rx;
        const alpha = band < 0 ? 1 : (band + 1) / feather;

        const i = ((top + ry) * image.width + left + rx) * 4;
        for (let c = 0, o = rx; c < 3; c++, o += rw) {
          // bilinearResize stores this value in a Float32Array
          const r = Math.fround(wy0 * top0[o] + wy1 * top1[o]);
          const cover = image.data[i + c];
          const stego = Math.min(1, Math.max(-1, Math.fround(r * strength32) + COVER_TO_UNIT[cover])) * 127.5 + 127.5;
          const value = band < 0 ? stego : alpha * stego + (1 - alpha) * cover;
          // NumPy truncates when the Python implementation stores floats into its uint8 image
          out[i + c] = Math.floor(value);
        }
      }
    }
    return out;
  }

  /**
   * Embeds a watermark packet into an image with a TrustMark encoder model, following
   * TrustMark.encode in the Python implementation: the processing region (the centred square for
   * aspect ratios above 2:1, and always for variant P) is downscaled to 256x256 exactly like
   * Pillow's bilinear resize, then come encoder inference, per-channel mean removal of the
   * residual, bilinear upscale of the residual, the variant's strength multiplier and a feathered
   * paste into the original pixels. The alpha channel is preserved.
   *
   * @param {Object} runtime - { ort, run } (see the top of this file); the resizer is not needed.
   * @param {Object} encoder - InferenceSession of the encoder model of the variant (e.g. encoder_Q.onnx).
   * @param {{data: ArrayLike<number>, width: number, height: number}} image - The RGBA cover image.
   * @param {ArrayLike<number>} packet - The 100 packet bits (0/1), e.g. from DataLayer_Encode.
   * @param {Object} [options]
   * @param {string} [options.variant='Q'] - Variant code of the encoder (see VARIANTS).
   * @param {number} [options.strength=1.0] - Watermark strength (WM_STRENGTH in Python); variant P
   *   multiplies it by 1.25 as Python does.
   * @returns {Promise<{data: Uint8ClampedArray, width: number, height: number}>} The watermarked image.
   */
  async function encodeWatermark(runtime, encoder, image, packet, options = {}) {
    const { ort } = runtime;
    const { variant: config, strength } = resolveEncodeOptions(options);
    if (!encoder) throw new Error('encoder model is not loaded');
    if (!packet || packet.length !== PACKET_BITS || !Array.from(packet).every((b) => b === 0 || b === 1)) {
      throw new Error(`the packet must contain ${PACKET_BITS} bits of 0 or 1`);
    }
    if (!image || !(image.width > 0) || !(image.height > 0) || !image.data || image.data.length !== image.width * image.height * 4) {
      throw new Error('the image must be RGBA data with a positive width and height');
    }

    const size = config.encodeResolution;
    const region = processingRegion(image.width, image.height, config.forceSquare);
    // ToTensor() in [0, 1], then scaled to the [-1, 1] range of the encoder, in float32 like PyTorch
    const cover = new Float32Array(3 * size * size);
    pillowBilinearResize(image, region, size, size).forEach((v, k) => { cover[k] = BYTE_TO_UNIT[v] * 2.0 - 1.0; });

    const results = await runSession(runtime, encoder, {
      'onnx::Concat_0': new ort.Tensor('float32', cover, [1, 3, size, size]),
      'onnx::Gemm_1': new ort.Tensor('float32', Float32Array.from(packet), [1, PACKET_BITS]),
    });
    const stego = results['image'].data;

    // residual = clamp(stego) - cover, then remove the per-channel mean to avoid colour shifts
    const plane = size * size;
    const residual = new Float32Array(3 * plane);
    for (let c = 0; c < 3; c++) {
      let sum = 0;
      for (let k = c * plane; k < (c + 1) * plane; k++) {
        residual[k] = Math.min(1, Math.max(-1, stego[k])) - cover[k];
        sum += residual[k];
      }
      const mean = sum / plane;
      for (let k = c * plane; k < (c + 1) * plane; k++) residual[k] -= mean;
    }

    // Upscaled to the region size while it is added, like torch.nn.functional.interpolate
    const data = applyResidual(image, region, residual, strength * config.strengthMultiplier, size, size);
    return { data, width: image.width, height: image.height };
  }

  return {
    ENCODER_RESOLUTION,
    PACKET_BITS,
    VARIANTS,
    getVariant,
    resolveEncodeOptions,
    zeroFeeds,
    processingRegion,
    rgbaToPlanar,
    computeScalesFixed,
    pillowBilinearResize,
    preprocess,
    decodeBits,
    bilinearResize,
    featherSize,
    applyResidual,
    encodeWatermark,
  };
});
