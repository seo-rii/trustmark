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
 * Reads an image into RGBA pixels with a canvas.
 * @param {string|Blob|ImageData} source - Image URL (e.g. a data URL), File/Blob, or an
 *   ImageData-like { data, width, height } object, which is returned as is.
 * @returns {Promise<ImageData>} The pixels.
 */
async function loadImageData(source) {
  if (source && source.data && source.width && source.height) return source;

  const url = typeof source === 'string' ? source : URL.createObjectURL(source);
  try {
    const img = await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("Failed to load image"));
      img.src = url;
    });
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    return ctx.getImageData(0, 0, img.width, img.height);
  } finally {
    if (url !== source) URL.revokeObjectURL(url);
  }
}

/**
 * Returns the runtime that TrustmarkCore functions run models with.
 * @returns {Object} { ort, resizer, run }
 */
function coreRuntime() {
  return { ort, resizer: session_resize, run: safeRunInference };
}

/**
 * Returns the error for a model that is not available.
 * @param {string} name - Model name.
 * @returns {Error} The error, including why the model failed to load.
 */
function modelNotLoadedError(name) {
  const state = modelStates[name];
  return new Error(`${name} model is not loaded${state && state.error ? `: ${state.error}` : ''}`);
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

  let image;
  try {
    image = await loadImageData(base64Image);
  } catch (error) {
    console.error("Error in watermark decoding:", error);
    return watermarkResult('error', null, decoders, `could not read the image: ${errorMessage(error)}`);
  }

  for (const config of modelConfigs) {
    const outcome = { variant: config.variantcode, status: 'error', backend: null, error: null };
    decoders.push(outcome);

    const session = sessions[config.sessionVar];
    if (!session) {
      outcome.error = modelNotLoadedError(config.fname).message;
      continue;
    }
    outcome.backend = modelStates[config.fname].backend;

    try {
      if (!session_resize) throw modelNotLoadedError('resizer');

      let startTime = new Date();
      const watermarkBool = await TrustmarkCore.decodeBits(coreRuntime(), session, image, config.resolution, config.squarecrop);

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
