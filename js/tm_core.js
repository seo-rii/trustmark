/*!
 * TrustMark JS core: image pre-processing shared by the watermark models
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

  // Value from python/trustmark/trustmark.py
  const ASPECT_RATIO_LIM = 2.0;      // wider or taller images are processed as a centre square

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
    const input = new runtime.ort.Tensor('float32', pixels, [1, 3, resolution, resolution]);
    const results = await runSession(runtime, decoder, { image: input });
    return Array.from(results['output'].data, (v) => v >= 0);
  }

  return {
    processingRegion,
    rgbaToPlanar,
    computeScalesFixed,
    preprocess,
    decodeBits,
  };
});
