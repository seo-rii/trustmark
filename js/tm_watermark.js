/*!
 * TrustMark JS Watermarking Decoder Module
 * Copyright 2024 Adobe. All rights reserved.
 * Licensed under the MIT License.
 * 
 * NOTICE: Adobe permits you to use, modify, and distribute this file in
 * accordance with the terms of the Adobe license agreement accompanying it.
 */ 

// Source for ONNX model binaries
const MODEL_BASE_URL = "https://cai-watermark.adobe.net/watermarking/trustmark-models/";

// List all watermark models for decoding
const modelConfigs = [
  { variantcode: 'Q', fname: 'decoder_Q', sessionVar: 'session_wmarkQ', resolution: 256, squarecrop: false }, 
  { variantcode: 'P', fname: 'decoder_P', sessionVar: 'session_wmarkP', resolution: 224, squarecrop: true },
];

// Backends tried, in order, for the decoders. If a WebGPU session cannot be created
// (no adapter, unsupported operator, ...) the decoder falls back to WASM on the CPU.
const DEFAULT_EXECUTION_PROVIDERS = ['webgpu', 'wasm'];

const sessions = {};
let session_resize;

/**
 * Load state of every model, keyed by model name ('resizer', 'decoder_Q', ...).
 * Each entry is { status, backend, fallbackFrom, fallbackReason, error, loadMs } where
 * status is 'pending', 'loading', 'ready' or 'error', backend is the execution provider the
 * session runs on, and fallbackFrom/fallbackReason record a backend that failed before it.
 */
const modelStates = {};

let trustmarkInitPromise = null;
let trustmarkLoaded = false;
let resolveTrustmarkReady;

/**
 * Resolves with trustmarkStatus() once every model has either loaded or failed to load.
 * It never rejects: check the status of each model, or the `status` field returned by runwmark.
 */
const trustmarkReady = new Promise((resolve) => { resolveTrustmarkReady = resolve; });

/**
 * Starts loading the models and resolves once every model has either loaded or failed.
 * This runs automatically when the script loads unless `TRUSTMARK_AUTO_INIT = false` is set
 * before it; set that flag to pass options. Later calls return the same promise.
 *
 * @param {Object} [options]
 * @param {string} [options.modelBaseUrl] - Base URL of the ONNX model files.
 * @param {Array<string>} [options.executionProviders] - Decoder backends to try, in order.
 * @returns {Promise<Object>} Resolves with trustmarkStatus().
 */
function trustmarkInit(options = {}) {
  if (trustmarkInitPromise) {
    if (Object.keys(options).length) {
      console.warn('TrustMark models are already loading, so trustmarkInit() options are ignored. ' +
                   'Set TRUSTMARK_AUTO_INIT = false before loading tm_watermark.js to pass options.');
    }
    return trustmarkInitPromise;
  }

  const baseUrl = options.modelBaseUrl || MODEL_BASE_URL;
  const executionProviders = options.executionProviders || DEFAULT_EXECUTION_PROVIDERS;

  for (const name of ['resizer', ...modelConfigs.map((config) => config.fname)]) {
    modelStates[name] = newModelState('pending');
  }

  trustmarkInitPromise = (async () => {
    // The resizer must run on WASM: WebGPU's Resize does not support antialiasing
    session_resize = await loadModelSession('resizer', baseUrl, ['wasm']);
    for (const config of modelConfigs) {
      const session = await loadModelSession(config.fname, baseUrl, executionProviders);
      if (session) sessions[config.sessionVar] = session;
    }
    trustmarkLoaded = true;
    const status = trustmarkStatus();
    resolveTrustmarkReady(status);
    return status;
  })();
  return trustmarkInitPromise;
}

/**
 * Returns a snapshot of the model load state.
 * @returns {{loaded: boolean, models: Object}} `loaded` is true once trustmarkInit() has
 *   finished (each model is then 'ready' or 'error'); `models` maps names to modelStates entries.
 */
function trustmarkStatus() {
  const models = {};
  for (const [name, state] of Object.entries(modelStates)) {
    models[name] = Object.assign({}, state);
  }
  return { loaded: trustmarkLoaded, models };
}

/**
 * Creates a model state entry.
 * @param {string} status - The initial status.
 * @returns {Object} The state.
 */
function newModelState(status) {
  return { status, backend: null, fallbackFrom: null, fallbackReason: null, error: null, loadMs: null };
}

/**
 * Downloads an ONNX model once and creates a session on the first backend that works.
 * The outcome is recorded in modelStates[name]; failures are recorded rather than thrown.
 *
 * @param {string} name - Model name; the file is `${baseUrl}${name}.onnx`.
 * @param {string} baseUrl - Base URL of the model files.
 * @param {Array<string>} executionProviders - Backends to try, in order.
 * @returns {Promise<ort.InferenceSession|null>} The session, or null if none could be created.
 */
async function loadModelSession(name, baseUrl, executionProviders) {
  const state = newModelState('loading');
  modelStates[name] = state;
  const startTime = Date.now();

  let model;
  try {
    const response = await fetch(`${baseUrl}${name}.onnx`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    model = new Uint8Array(await response.arrayBuffer());
  } catch (error) {
    state.status = 'error';
    state.error = `could not download ${name}.onnx: ${errorMessage(error)}`;
    console.error(`Could not load ${name} model:`, state.error);
    return null;
  }

  const failures = [];
  for (const backend of executionProviders) {
    try {
      const session = await ort.InferenceSession.create(model, { executionProviders: [backend] });
      state.status = 'ready';
      state.backend = backend;
      state.loadMs = Date.now() - startTime;
      if (failures.length) {
        state.fallbackFrom = failures[0].backend;
        state.fallbackReason = failures.map((f) => `${f.backend}: ${f.message}`).join('; ');
        console.warn(`${name} model fell back to ${backend} (${state.fallbackReason})`);
      }
      console.log(`${name} model loaded on ${backend} in ${state.loadMs / 1000} seconds`);
      return session;
    } catch (error) {
      failures.push({ backend, message: errorMessage(error) });
    }
  }

  state.status = 'error';
  state.error = failures.length
    ? `could not create a ${name} session: ` + failures.map((f) => `${f.backend}: ${f.message}`).join('; ')
    : `no execution provider configured for ${name}`;
  console.error(`Could not load ${name} model:`, state.error);
  return null;
}

/**
 * Returns a readable message for any thrown value.
 * @param {*} error - The thrown value.
 * @returns {string} The message.
 */
function errorMessage(error) {
  return (error && error.message) ? error.message : String(error);
}

/* WebGPU will fail silently and intermittently if multiple concurrent inference calls are made.
   Every inference therefore goes through this FIFO promise chain, so only one session.run()
   is in flight at a time and calls run in the order they were made. */
let inferenceQueue = Promise.resolve();
      
/**
 * Runs a session after every previously queued inference has finished.
 * @param {ort.InferenceSession} session - The session to run.
 * @param {Object} feed - The input feeds.
 * @returns {Promise<Object>} The session outputs; rejects if this inference fails.
 */
function safeRunInference(session, feed) {
  const result = inferenceQueue.then(() => session.run(feed));
  // A failed inference must not stall the queue; the caller still receives the rejection
  inferenceQueue = result.then(() => undefined, () => undefined);
  return result;
}
     

/**
 * Converts an image URL to a tensor suitable for processing.
 * @param {string} imageUrl - The URL of the image to load.
 * @returns {Promise<ort.Tensor>} The processed tensor.
 */
async function loadImageAsTensor(imageUrl) {
  const img = new Image();
  img.src = imageUrl;

  return new Promise((resolve, reject) => {
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      const imgData = ctx.getImageData(0, 0, img.width, img.height);

      const { data, width, height } = imgData;
      const totalPixels = width * height;
      const imageTensor = new Float32Array(totalPixels * 3);

      let j = 0;
      const page = width * height;
      const twopage = 2 * page;

      for (let i = 0; i < totalPixels; i++) {
        const index = i * 4;
        imageTensor[j] = data[index] / 255.0; // Red channel
        imageTensor[j + page] = data[index + 1] / 255.0; // Green channel
        imageTensor[j + twopage] = data[index + 2] / 255.0; // Blue channel
        j++;
      }

      resolve(new ort.Tensor('float32', imageTensor, [1, 3, height, width]));
    };

    img.onerror = () => reject(new Error("Failed to load image"));
  });
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
 * Resizes the image tensor to a square size suitable for watermark decoding.
 * @param {ort.Tensor} inputTensor - The input image tensor.
 * @param {number} targetSize - The target size for resizing.
 * @returns {Promise<ort.Tensor>} The resized tensor; rejects if the resizer is unavailable or fails.
 */
async function runResizeModelSquare(inputTensor, targetSize, force_square) {
    try {
        if (!session_resize) {
            const state = modelStates.resizer;
            throw new Error(`resizer model is not loaded${state && state.error ? `: ${state.error}` : ''}`);
        }

        const inputDims = inputTensor.dims;  // Get dimensions of the input tensor
        const [batch, channels, height, width] = inputDims;
        
        // Compute the aspect ratio
        const aspectRatio = width / height;
        const lscape= (aspectRatio>=1.0);
                
        let croppedTensor = inputTensor;
        let cropWidth = width;
        let cropHeight = height;
        
        // If the aspect ratio is greater than 2.0, we need to crop the center square
        if (lscape && (aspectRatio > 2.0 || force_square)) {
            cropWidth = height;  // Take a square from the width
            const offsetX = Math.floor((width - cropWidth) / 2);  // Horizontal center crop
            croppedTensor = await cropTensor(inputTensor, offsetX, 0, cropWidth, height);
        }
        
        if (!lscape && (aspectRatio < 0.5 || force_square)) {
            cropHeight = width;  // Take a square from the height
            const offsetY = Math.floor((height - cropHeight) / 2);  // Vertical center crop
            croppedTensor = await cropTensor(inputTensor, 0, offsetY, width, cropHeight);
        }
    
        // After cropping, resize the tensor to the target size
        const targetDims = [targetSize, targetSize];
        const scales = computeScalesFixed(targetDims, [batch, channels, cropHeight, cropWidth]);
        const scalesTensor = new ort.Tensor('float32', scales, [4]);
                
        // Prepare the target size tensor
        const targetSizeTensor = new ort.Tensor('int64', new BigInt64Array([BigInt(targetSize)]), [1]);
            
        // Set up the feeds for the model
        const feeds = {
            'X': croppedTensor,       // Cropped image tensor
            'scales': scalesTensor,   // Scales tensor
            'target_size': targetSizeTensor  // Dynamic target size tensor
        };

        const results = await safeRunInference(session_resize, feeds);
        return results['Y'];
            
    } catch (error) {
        console.error('Error during resizing:', error);
        throw error;
    }
}

// Helper function to crop the tensor
async function cropTensor(inputTensor, offsetX, offsetY, cropWidth, cropHeight) {
    const [batch, channels, height, width] = inputTensor.dims;  
    const croppedData = new Float32Array(batch * channels * cropWidth * cropHeight);
    const inputData = inputTensor.data;
         
    let k = 0;
    for (let c = 0; c < channels; c++) {
        for (let y = 0; y < cropHeight; y++) {
            for (let x = 0; x < cropWidth; x++) {
                const srcIndex = c * width * height + (y + offsetY) * width + (x + offsetX);
                croppedData[k++] = inputData[srcIndex];
            }
        }
    }
        
    return new ort.Tensor('float32', croppedData, [batch, channels, cropHeight, cropWidth]);
}                  


/**
 * Decodes the watermark from an image. Waits for the models to finish loading first,
 * starting the load if trustmarkInit() has not been called yet.
 *
 * @param {string} base64Image - The image URL, typically a data URL.
 * @returns {Promise<object>} The decoding result:
 *   - status: 'detected', 'not_detected', or 'error'. 'error' means at least one decoder could
 *     not check the image (model not loaded, inference failed, unreadable image) and none found
 *     a watermark, so the absence of a watermark could not be established.
 *   - watermark_present: true only when status is 'detected'.
 *   - watermark, schema, c2padata: payload bits, schema name and C2PA soft binding, or null.
 *   - variant, backend: variant code and execution backend of the detecting decoder, or null.
 *   - error: description of the failures when status is 'error', otherwise null.
 *   - decoders: per decoder { variant, status, backend, error }.
 */
async function runwmark(base64Image) {
  await trustmarkInit();

  const decoders = [];
  let found = null;

  let inputTensor;
  try {
    inputTensor = await loadImageAsTensor(base64Image);
  } catch (error) {
    console.error("Error in watermark decoding:", error);
    return watermarkResult('error', null, decoders, `could not read the image: ${errorMessage(error)}`);
  }

  for (const config of modelConfigs) {
    const outcome = { variant: config.variantcode, status: 'error', backend: null, error: null };
    decoders.push(outcome);

    const session = sessions[config.sessionVar];
    if (!session) {
      const state = modelStates[config.fname];
      outcome.error = `${config.fname} model is not loaded${state && state.error ? `: ${state.error}` : ''}`;
      continue;
    }
    outcome.backend = modelStates[config.fname].backend;

    try {
      const resizedTensorWM = await runResizeModelSquare(inputTensor, config.resolution, config.squarecrop);

      const feeds = { image: resizedTensorWM };

      let startTime = new Date();
      const results = await safeRunInference(session, feeds);

      const watermarkFloat = results['output'].data;
      const watermarkBool = watermarkFloat.map((v) => v >= 0);

      const dataObj = DataLayer_Decode(watermarkBool, eccengine, config.variantcode);
      console.log(`Watermark model inference in ${(new Date() - startTime)} milliseconds`);

      outcome.status = dataObj.valid ? 'detected' : 'not_detected';
      // Report the first detected watermark (if many were)
      if (dataObj.valid && !found) found = { dataObj, outcome };
    } catch (error) {
      console.error(`Error in ${config.fname} watermark decoding:`, error);
      outcome.error = `${config.fname}: ${errorMessage(error)}`;
    }
  }

  if (found) return watermarkResult('detected', found, decoders, null);

  const failed = decoders.filter((d) => d.status === 'error');
  if (failed.length) return watermarkResult('error', null, decoders, failed.map((d) => d.error).join('; '));

  return watermarkResult('not_detected', null, decoders, null);
}

/**
 * Builds the object returned by runwmark.
 * @param {string} status - 'detected', 'not_detected' or 'error'.
 * @param {?Object} found - { dataObj, outcome } of the detecting decoder, or null.
 * @param {Array<Object>} decoders - Per-decoder outcomes.
 * @param {?string} error - Failure description when status is 'error'.
 * @returns {Object} The result.
 */
function watermarkResult(status, found, decoders, error) {
  return {
    status,
    watermark_present: status === 'detected',
    watermark: found ? found.dataObj.data_binary : null,
    schema: found ? found.dataObj.schema : null,
    c2padata: found ? found.dataObj.softBindingInfo : null,
    variant: found ? found.outcome.variant : null,
    backend: found ? found.outcome.backend : null,
    error,
    decoders,
  };
}

// Load the models immediately unless the page opted out to call trustmarkInit(options) itself
if (typeof TRUSTMARK_AUTO_INIT === 'undefined' || TRUSTMARK_AUTO_INIT !== false) {
  trustmarkInit();
}
