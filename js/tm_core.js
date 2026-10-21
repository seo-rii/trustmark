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
 *   - resizer: an InferenceSession of resizer.onnx (antialiased downscaler, WASM/CPU only)
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
        out[k] = data[i] / 255.0;
        out[k + plane] = data[i + 1] / 255.0;
        out[k + 2 * plane] = data[i + 2] / 255.0;
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
   * @param {Object} decoder - InferenceSession of the decoder model.
   * @param {{data: ArrayLike<number>, width: number, height: number}} image - The RGBA image.
   * @param {number} resolution - Decoder resolution (Q: 256, P: 224).
   * @param {boolean} [forceSquare=false] - Always use the centred square (variant P).
   * @returns {Promise<Array<boolean>>} The 100 packet bits.
   */
  async function decodeBits(runtime, decoder, image, resolution, forceSquare = false) {
    const { pixels } = await preprocess(runtime, image, resolution, forceSquare);
    const scaled = Float32Array.from(pixels, (v) => v * 2.0 - 1.0);  // [0, 1] -> [-1, 1]
    const input = new runtime.ort.Tensor('float32', scaled, [1, 3, resolution, resolution]);
    const results = await runSession(runtime, decoder, { image: input });
    return Array.from(results['output'].data, (v) => v >= 0);
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
    // Source index and weight for each output row/column (PyTorch's half-pixel mapping)
    function axis(inSize, outSize) {
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
    const ys = axis(inH, outH);
    const xs = axis(inW, outW);

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
   * Adds a full-resolution residual to the processing region of an image and pastes it back
   * with feathered edges, like the Python encoder (TrustMark.encode followed by
   * put_the_image_after_processing). Pixels outside the region and the alpha channel are
   * copied unchanged.
   *
   * @param {{data: ArrayLike<number>, width: number, height: number}} image - The RGBA cover image.
   * @param {{left: number, top: number, width: number, height: number}} region - The processed region.
   * @param {Float32Array} residual - Planar RGB residual in [-1, 1] units, 3 * region.height * region.width.
   * @param {number} strength - Residual multiplier (WM_STRENGTH).
   * @returns {Uint8ClampedArray} The watermarked RGBA pixels.
   */
  function applyResidual(image, region, residual, strength) {
    const out = new Uint8ClampedArray(image.data);
    const { left, top, width: rw, height: rh } = region;
    const plane = rw * rh;
    const feather = featherSize(rw, rh);

    for (let ry = 0; ry < rh; ry++) {
      for (let rx = 0; rx < rw; rx++) {
        // Python pastes the region, then blends the top, bottom, left and right bands in that
        // order, each overwriting the previous; the last band covering a pixel decides its weight.
        let band = -1;
        if (rx >= rw - feather) band = rw - 1 - rx;
        else if (rx < feather) band = rx;
        else if (ry >= rh - feather) band = rh - 1 - ry;
        else if (ry < feather) band = ry;
        const alpha = band < 0 ? 1 : (band + 1) / feather;

        const i = ((top + ry) * image.width + left + rx) * 4;
        const k = ry * rw + rx;
        for (let c = 0; c < 3; c++) {
          const cover = image.data[i + c];
          const stego = Math.min(1, Math.max(-1, residual[k + c * plane] * strength + cover / 127.5 - 1)) * 127.5 + 127.5;
          const value = band < 0 ? stego : alpha * stego + (1 - alpha) * cover;
          // NumPy truncates when the Python implementation stores floats into its uint8 image
          out[i + c] = Math.floor(value);
        }
      }
    }
    return out;
  }

  /**
   * Embeds a watermark packet into an image with a TrustMark encoder model, following the
   * Python implementation: centre-square crop for aspect ratios above 2:1, antialiased
   * downscale to 256x256, encoder inference, per-channel mean removal of the residual,
   * bilinear upscale of the residual and feathered paste into the original pixels.
   * The alpha channel is preserved.
   *
   * Only variant Q has been verified; P would need its 1.25 strength multiplier and square crop.
   *
   * @param {Object} runtime - { ort, resizer, run } (see the top of this file).
   * @param {Object} encoder - InferenceSession of the encoder model (encoder_Q.onnx).
   * @param {{data: ArrayLike<number>, width: number, height: number}} image - The RGBA cover image.
   * @param {ArrayLike<number>} packet - The 100 packet bits (0/1), e.g. from DataLayer_Encode.
   * @param {Object} [options]
   * @param {number} [options.strength=1.0] - Watermark strength (WM_STRENGTH in Python).
   * @returns {Promise<{data: Uint8ClampedArray, width: number, height: number}>} The watermarked image.
   */
  async function encodeWatermark(runtime, encoder, image, packet, options = {}) {
    const { ort } = runtime;
    const strength = options.strength === undefined ? 1.0 : options.strength;
    if (!encoder) throw new Error('encoder model is not loaded');
    if (!packet || packet.length !== PACKET_BITS || !Array.from(packet).every((b) => b === 0 || b === 1)) {
      throw new Error(`the packet must contain ${PACKET_BITS} bits of 0 or 1`);
    }
    if (!(typeof strength === 'number' && strength >= 0 && Number.isFinite(strength))) {
      throw new Error('strength must be a non-negative number');
    }
    if (!image || !(image.width > 0) || !(image.height > 0) || !image.data || image.data.length !== image.width * image.height * 4) {
      throw new Error('the image must be RGBA data with a positive width and height');
    }

    const size = ENCODER_RESOLUTION;
    const { region, pixels } = await preprocess(runtime, image, size);
    const cover = Float32Array.from(pixels, (v) => v * 2.0 - 1.0);  // the encoder works in [-1, 1]

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

    const upscaled = bilinearResize(residual, 3, size, size, region.height, region.width);
    return { data: applyResidual(image, region, upscaled, strength), width: image.width, height: image.height };
  }

  return {
    ENCODER_RESOLUTION,
    PACKET_BITS,
    processingRegion,
    rgbaToPlanar,
    computeScalesFixed,
    preprocess,
    decodeBits,
    bilinearResize,
    featherSize,
    applyResidual,
    encodeWatermark,
  };
});
