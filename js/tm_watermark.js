/*!
 * TrustMark JS Watermarking Module
 * Copyright 2024 Adobe. All rights reserved.
 * Licensed under the MIT License.
 * 
 * NOTICE: Adobe permits you to use, modify, and distribute this file in
 * accordance with the terms of the Adobe license agreement accompanying it.
 */ 

// Source for ONNX model binaries
const MODEL_BASE_URL = "https://cai-watermark.adobe.net/watermarking/trustmark-models/";

// Variants whose decoders are loaded by default, in the order they are tried. Each decoder
// is a separate download: decoder_Q, decoder_P and decoder_B are 47 MB each, decoder_C 22 MB.
// Choose others with trustmarkInit({ decoders }); the per-variant settings (resolution, crop)
// are TrustmarkCore.VARIANTS.
const DEFAULT_DECODERS = ['Q', 'P'];

// Variants whose decoders were requested from trustmarkInit()
let decoderVariants = [];

// Backends tried, in order, for the decoders. If a WebGPU session cannot be created
// (no adapter, unsupported operator, ...) the decoder falls back to WASM on the CPU.
const DEFAULT_EXECUTION_PROVIDERS = ['webgpu', 'wasm'];

// Sessions keyed by model name ('decoder_Q', 'encoder_P', ...)
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
let trustmarkOptions = null;
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
 * @param {Array<string>} [options.executionProviders] - Decoder and encoder backends to try, in order.
 * @param {Array<string>} [options.decoders=['Q', 'P']] - Variants whose decoders runwmark uses,
 *   in the order they are tried: any of 'Q', 'P', 'B' and 'C'. An empty list loads no decoder,
 *   for pages that only embed watermarks.
 * @returns {Promise<Object>} Resolves with trustmarkStatus().
 * @throws {Error} If `decoders` names an unknown variant (nothing is loaded then).
 */
function trustmarkInit(options = {}) {
  if (trustmarkInitPromise) {
    if (Object.keys(options).length) {
      console.warn('TrustMark models are already loading, so trustmarkInit() options are ignored. ' +
                   'Set TRUSTMARK_AUTO_INIT = false before loading tm_watermark.js to pass options.');
    }
    return trustmarkInitPromise;
  }

  const decoders = options.decoders === undefined ? DEFAULT_DECODERS : options.decoders;
  if (!Array.isArray(decoders)) throw new Error('decoders must be an array of variant codes, e.g. [\'Q\', \'P\']');
  decoders.forEach((code) => TrustmarkCore.getVariant(code));  // throws for an unknown variant
  decoderVariants = [...new Set(decoders)];

  const baseUrl = options.modelBaseUrl || MODEL_BASE_URL;
  const executionProviders = options.executionProviders || DEFAULT_EXECUTION_PROVIDERS;
  trustmarkOptions = { baseUrl, executionProviders };  // also used to load the encoders later

  const decoderNames = decoderVariants.map((code) => TrustmarkCore.getVariant(code).decoder);
  // Only the decoders use the resizer model
  const names = decoderNames.length ? ['resizer', ...decoderNames] : [];
  for (const name of names) modelStates[name] = newModelState('pending');

  trustmarkInitPromise = (async () => {
    // The resizer must run on WASM: WebGPU's Resize does not support antialiasing
    if (decoderNames.length) session_resize = await loadModelSession('resizer', baseUrl, ['wasm']);
    for (const name of decoderNames) {
      // WebGPU sessions can fail silently, so check them first
      const session = await loadModelSession(name, baseUrl, executionProviders, 'webgpu');
      if (session) sessions[name] = session;
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
  return { status, backend: null, fallbackFrom: null, fallbackReason: null, error: null, loadMs: null, warmUpMs: null };
}

/**
 * Downloads an ONNX model once and creates a session on the first backend that works.
 * The outcome is recorded in modelStates[name]; failures are recorded rather than thrown.
 *
 * With warmUp, the session also runs once on all-zero inputs before it is used (warmUpSession):
 * the first inference pays one-time costs (WebAssembly thread start-up, WebGPU shader
 * compilation) that would otherwise delay the first image, and a backend whose inference fails
 * is skipped like one whose session cannot be created.
 *
 * @param {string} name - Model name; the file is `${baseUrl}${name}.onnx`.
 * @param {string} baseUrl - Base URL of the model files.
 * @param {Array<string>} executionProviders - Backends to try, in order.
 * @param {boolean|string} [warmUp=false] - Run a first inference: true on every backend, or
 *   'webgpu' to only check WebGPU sessions.
 * @returns {Promise<ort.InferenceSession|null>} The session, or null if none could be created.
 */
async function loadModelSession(name, baseUrl, executionProviders, warmUp = false) {
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
      if (warmUp === true || (warmUp === 'webgpu' && backend === 'webgpu')) {
        const warmUpStart = Date.now();
        try {
          await warmUpSession(session, name, backend);
        } catch (error) {
          Promise.resolve().then(() => session.release()).catch(() => {});
          throw new Error(`warm-up inference failed: ${errorMessage(error)}`);
        }
        state.warmUpMs = Date.now() - warmUpStart;
      }
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
  return enqueueInference(() => session.run(feed));
}

/**
 * Runs a task that uses a session after every previously queued inference has finished.
 * @param {function(): Promise<*>} task - The task.
 * @returns {Promise<*>} The result of the task; rejects if the task fails.
 */
function enqueueInference(task) {
  const result = inferenceQueue.then(task);
  // A failed inference must not stall the queue; the caller still receives the rejection
  inferenceQueue = result.then(() => undefined, () => undefined);
  return result;
}

/**
 * Runs a new session once on all-zero inputs and checks that it works. ONNX Runtime Web can
 * resolve a WebGPU inference whose shaders failed to compile with outputs that are all zeros,
 * so WebGPU errors raised during the run, and encoder or decoder outputs that are all zeros, are
 * failures too.
 *
 * @param {ort.InferenceSession} session - The session.
 * @param {string} name - Model name.
 * @param {string} backend - Execution provider of the session.
 * @returns {Promise<void>} Rejects if the inference failed.
 */
function warmUpSession(session, name, backend) {
  const feeds = TrustmarkCore.zeroFeeds(ort, name);
  return enqueueInference(async () => {
    const device = backend === 'webgpu' && ort.env && ort.env.webgpu ? ort.env.webgpu.device : null;
    const scopes = device ? ['validation', 'out-of-memory', 'internal'] : [];
    scopes.forEach((filter) => device.pushErrorScope(filter));
    let outputs;
    try {
      outputs = await session.run(feeds);
    } finally {
      const errors = (await Promise.all(scopes.map(() => device.popErrorScope()))).filter(Boolean);
      if (errors.length) throw new Error(`WebGPU error: ${errors[0].message}`);
    }
    if (Object.values(outputs).every((tensor) => tensor.data.every((v) => v === 0))) {
      throw new Error('the model returned only zeros');
    }
  });
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

  if (!decoderVariants.length) {
    return watermarkResult('error', null, decoders, 'no watermark decoders are loaded; pass trustmarkInit({ decoders })');
  }

  let image;
  try {
    image = await loadImageData(base64Image);
  } catch (error) {
    console.error("Error in watermark decoding:", error);
    return watermarkResult('error', null, decoders, `could not read the image: ${errorMessage(error)}`);
  }

  for (const variant of decoderVariants) {
    const name = TrustmarkCore.getVariant(variant).decoder;
    const outcome = { variant, status: 'error', backend: null, error: null };
    decoders.push(outcome);

    const session = sessions[name];
    if (!session) {
      outcome.error = modelNotLoadedError(name).message;
      continue;
    }
    outcome.backend = modelStates[name].backend;

    try {
      if (!session_resize) throw modelNotLoadedError('resizer');

      let startTime = new Date();
      const watermarkBool = await TrustmarkCore.decodeBits(coreRuntime(), session, image, variant);

      const dataObj = DataLayer_Decode(watermarkBool, eccengine, variant);
      console.log(`Watermark model inference in ${(new Date() - startTime)} milliseconds`);

      outcome.status = dataObj.valid ? 'detected' : 'not_detected';
      // Report the first detected watermark (if many were)
      if (dataObj.valid && !found) found = { dataObj, outcome };
    } catch (error) {
      console.error(`Error in ${name} watermark decoding:`, error);
      outcome.error = `${name}: ${errorMessage(error)}`;
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

// Watermark embedding. Each encoder model (17 MB) is only downloaded when the first image is
// watermarked with its variant.
const encoderLoads = {};

/**
 * Loads the encoder model of a variant once, after the models of trustmarkInit().
 * @param {string} variant - Variant code.
 * @returns {Promise<ort.InferenceSession|null>} The session, or null if it failed to load.
 */
function loadEncoder(variant) {
  const name = TrustmarkCore.getVariant(variant).encoder;
  if (!encoderLoads[name]) {
    modelStates[name] = newModelState('pending');
    encoderLoads[name] = trustmarkInit().then(async () => {
      // A WebGPU encoder that silently returns zeros would destroy the image, so check it first
      const session = await loadModelSession(name, trustmarkOptions.baseUrl, trustmarkOptions.executionProviders, 'webgpu');
      if (session) sessions[name] = session;
      return session;
    });
  }
  return encoderLoads[name];
}

/**
 * Embeds a TrustMark watermark into an image, like TrustMark.encode in the Python implementation
 * (MODE='binary'). The alpha channel is kept. Rejects, rather than returning an unwatermarked
 * image, if an option or the payload is invalid, the image cannot be read or the model is
 * unavailable; invalid options and payloads are rejected before anything is downloaded.
 *
 * @param {string|Blob|ImageData} image - Image URL, File/Blob, or RGBA ImageData.
 * @param {string} payload - '0'/'1' characters, at most the capacity of the schema (BCH_SUPER 40,
 *   BCH_5 61, BCH_4 68, BCH_3 75 bits); shorter payloads are padded with zeros.
 * @param {Object} [options]
 * @param {string} [options.variant='Q'] - Model variant: 'Q', 'P', 'B' or 'C'.
 * @param {string|number} [options.schema='BCH_5'] - Error correction schema: 'BCH_SUPER', 'BCH_5',
 *   'BCH_4' or 'BCH_3', or its number (0 to 3, as TrustMark.Encoding in Python).
 * @param {number} [options.strength=1.0] - Watermark strength (variant P multiplies it by 1.25).
 * @returns {Promise<Object>} { image: { data, width, height } (RGBA), payload (the embedded data
 *   bits), packet (the 100 bits fed to the encoder), schema (its name), variant, backend }.
 */
async function trustmarkEncode(image, payload, options = {}) {
  // Validate everything before downloading anything
  const { variant, strength } = TrustmarkCore.resolveEncodeOptions(options);
  const schema = DataLayer_GetSchemaVersion(options.schema === undefined ? 'BCH_5' : options.schema);
  const packet = DataLayer_Encode(payload, schema, eccengine);
  const name = variant.encoder;

  const encoder = await loadEncoder(variant.code);
  if (!encoder) throw modelNotLoadedError(name);

  const pixels = await loadImageData(image);
  const startTime = Date.now();
  const watermarked = await TrustmarkCore.encodeWatermark(coreRuntime(), encoder, pixels, packet, { variant: variant.code, strength });
  console.log(`Watermark embedded in ${Date.now() - startTime} milliseconds`);

  return {
    image: watermarked,
    payload: payload.padEnd(DataLayer_GetSchemaDataBits(schema), '0'),
    packet: packet.join(''),
    schema: DataLayer_GetSchemaName(schema),
    variant: variant.code,
    backend: modelStates[name].backend,
  };
}

/**
 * Encodes RGBA pixels as a PNG file.
 * The canvas stores premultiplied alpha, so the colour of semi-transparent pixels can
 * change slightly, and fully transparent pixels lose their colour.
 *
 * @param {{data: Uint8ClampedArray, width: number, height: number}} image - The pixels.
 * @returns {Promise<Blob>} The PNG.
 */
function imageToPngBlob(image) {
  const canvas = document.createElement('canvas');
  canvas.width = image.width;
  canvas.height = image.height;
  canvas.getContext('2d').putImageData(new ImageData(image.data, image.width, image.height), 0, 0);
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('Could not encode the PNG'))), 'image/png');
  });
}

// Load the models immediately unless the page opted out to call trustmarkInit(options) itself
if (typeof TRUSTMARK_AUTO_INIT === 'undefined' || TRUSTMARK_AUTO_INIT !== false) {
  trustmarkInit();
}
