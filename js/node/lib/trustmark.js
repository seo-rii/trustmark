'use strict';

const path = require('node:path');

const { core, SCHEMAS, variantInfo, schemaInfo, buildPacket, decodePacket } = require('./core');
const { TrustmarkError, throwIfAborted, raceAbort, checkSignal } = require('./errors');
const { decodeImage, encodeImage, flatten, isOpaque } = require('./image');
const { JobQueue } = require('./queue');
const models = require('./models');

const DEFAULT_MAX_PIXELS = 25_000_000;
const DEFAULT_MAX_QUEUE = 100;
const DEFAULT_JPEG_QUALITY = 90;
const MIME_TYPES = { png: 'image/png', jpeg: 'image/jpeg' };

function argumentError(message) {
  return new TrustmarkError('INVALID_ARGUMENT', message);
}

function positiveInteger(value, name, fallback, { allowInfinity = false, min = 1 } = {}) {
  if (value === undefined) return fallback;
  if (allowInfinity && value === Infinity) return value;
  if (!Number.isInteger(value) || value < min) throw argumentError(`${name} must be an integer of at least ${min}`);
  return value;
}

function normalizeProviders(value) {
  if (value === undefined) return ['cpu'];
  if (!Array.isArray(value) || value.length === 0) throw argumentError('executionProviders must be a non-empty array');
  for (const provider of value) {
    const name = models.providerName(provider);
    if (typeof name !== 'string' || name.length === 0) {
      throw argumentError('Each execution provider must be a name such as "cpu" or "cuda", or an object with a name');
    }
  }
  return value.slice();
}

function errorSummary(error) {
  return error ? { code: error.code || null, message: String(error.message || error) } : null;
}

function isAbortOf(error, signal) {
  return !!signal && signal.aborted && error === signal.reason;
}

/**
 * Creates a TrustMark instance for one variant and schema. Models are loaded on first use (or
 * with ready()) and kept until close().
 */
function createTrustmark(options = {}) {
  if (options === null || typeof options !== 'object') throw argumentError('createTrustmark expects an options object');
  const variant = variantInfo(options.variant === undefined ? 'Q' : options.variant);
  const schema = schemaInfo(options.schema === undefined ? 'BCH_5' : options.schema);
  if (typeof options.modelDir !== 'string' || options.modelDir.length === 0) {
    throw argumentError('modelDir is required: the directory holding (or receiving, with allowDownload) the .onnx models');
  }
  if (options.allowDownload !== undefined && typeof options.allowDownload !== 'boolean') throw argumentError('allowDownload must be a boolean');
  if (options.orientation !== undefined && options.orientation !== 'apply' && options.orientation !== 'ignore') {
    throw argumentError("orientation must be 'apply' or 'ignore'");
  }
  if (options.sessionOptions !== undefined && (options.sessionOptions === null || typeof options.sessionOptions !== 'object')) {
    throw argumentError('sessionOptions must be an object');
  }
  const fetchImpl = options.fetch === undefined ? globalThis.fetch : options.fetch;
  if (options.allowDownload && typeof fetchImpl !== 'function') throw argumentError('allowDownload needs a fetch implementation');

  const config = {
    modelDir: path.resolve(options.modelDir),
    allowDownload: options.allowDownload === true,
    baseUrl: models.normalizeBaseUrl(options.modelBaseUrl === undefined ? models.DEFAULT_MODEL_BASE_URL : options.modelBaseUrl),
    digests: models.normalizeDigests(options.expectedDigests),
    requireDigests: options.requireDigests === true,
    maxModelBytes: positiveInteger(options.maxModelBytes, 'maxModelBytes', models.DEFAULT_MAX_MODEL_BYTES),
    fetch: fetchImpl,
  };
  const providers = normalizeProviders(options.executionProviders);
  const sessionOptions = { ...(options.sessionOptions || {}) };
  delete sessionOptions.executionProviders;
  const maxPixels = positiveInteger(options.maxPixels, 'maxPixels', DEFAULT_MAX_PIXELS);
  const orientation = options.orientation || 'apply';
  const queue = new JobQueue(
    positiveInteger(options.maxConcurrency, 'maxConcurrency', 1),
    positiveInteger(options.maxQueue, 'maxQueue', DEFAULT_MAX_QUEUE, { allowInfinity: true, min: 0 }),
  );

  let ort = options.ort || null;
  // Aborted by close(), so that pending downloads stop
  const lifecycle = new AbortController();
  let closing = null;

  // The resizer (antialiased Resize, used before the decoder) always runs on the CPU, as on WASM
  // in the browser demo, so that decoding sees the same pixels whichever provider runs the
  // decoder. Embedding resizes in JavaScript and does not need it.
  const roles = {
    resizer: { file: 'resizer.onnx', providers: ['cpu'] },
    encoder: { file: `${variant.encoder}.onnx`, providers },
    decoder: { file: `${variant.decoder}.onnx`, providers },
  };
  const state = {};
  for (const [role, { file, providers: requested }] of Object.entries(roles)) {
    state[role] = { file, requested, status: 'idle', promise: null, session: null, backend: null, fallbacks: [], info: null, error: null, loadMs: null };
  }

  function ensureOpen() {
    if (closing) throw new TrustmarkError('CLOSED', 'The TrustMark instance is closed');
  }

  function getOrt() {
    if (!ort) {
      try {
        ort = require('onnxruntime-node');
      } catch (error) {
        throw new TrustmarkError('MODEL_LOAD_FAILED', `onnxruntime-node could not be loaded: ${error.message}`, { cause: error });
      }
    }
    return ort;
  }

  /** Loads a model once; a failed load is retried by the next call. */
  function load(role) {
    const s = state[role];
    if (s.promise) return s.promise;
    s.status = 'loading';
    s.error = null;
    const started = Date.now();
    s.promise = (async () => {
      const info = await models.readModel(s.file, config, lifecycle.signal);
      throwIfAborted(lifecycle.signal);
      const created = await models.createSession(getOrt(), info.bytes, s.requested, sessionOptions, s.file);
      if (lifecycle.signal.aborted) {
        await Promise.resolve(created.session.release && created.session.release()).catch(() => {});
        throwIfAborted(lifecycle.signal);
      }
      s.session = created.session;
      s.backend = created.backend;
      s.fallbacks = created.fallbacks;
      s.info = { path: info.path, sha256: info.sha256, size: info.size, downloaded: info.downloaded, verified: info.verified };
      s.loadMs = Date.now() - started;
      s.status = 'ready';
      return s;
    })().catch((error) => {
      s.status = 'error';
      s.error = error;
      s.promise = null;
      throw error;
    });
    return s.promise;
  }

  /** Runtime for tm_core.js that checks the job's signal around every inference. */
  function runtimeFor(resizer, signal) {
    return {
      ort: getOrt(),
      resizer: resizer ? resizer.session : null,
      run: async (session, feeds) => {
        throwIfAborted(signal);
        let outputs;
        try {
          outputs = await session.run(feeds);
        } catch (error) {
          throw new TrustmarkError('INFERENCE_FAILED', `Inference failed: ${error && error.message ? error.message : error}`, { cause: error });
        }
        throwIfAborted(signal);
        return outputs;
      },
    };
  }

  function parseEncodeOptions(opts) {
    if (opts === null || typeof opts !== 'object') throw argumentError('encode options must be an object');
    const format = opts.format === undefined ? 'png' : opts.format;
    if (!Object.prototype.hasOwnProperty.call(MIME_TYPES, format)) throw new TrustmarkError('UNSUPPORTED', `format must be 'png' or 'jpeg', not ${JSON.stringify(format)}`);
    let quality = null;
    if (format === 'jpeg') {
      quality = opts.quality === undefined ? DEFAULT_JPEG_QUALITY : opts.quality;
      if (!Number.isInteger(quality) || quality < 1 || quality > 100) throw argumentError('quality must be an integer from 1 to 100');
    } else if (opts.quality !== undefined) {
      throw argumentError('quality only applies to JPEG output');
    }
    const strength = opts.strength === undefined ? 1.0 : opts.strength;
    if (typeof strength !== 'number' || !Number.isFinite(strength) || strength <= 0) throw argumentError('strength must be a positive number');
    let background = null;
    if (opts.background !== undefined) {
      const b = opts.background;
      if (!Array.isArray(b) || b.length !== 3 || !b.every((v) => Number.isInteger(v) && v >= 0 && v <= 255)) {
        throw argumentError('background must be [r, g, b] with integers from 0 to 255');
      }
      background = b.slice();
    }
    return { format, quality, strength, background, signal: checkSignal(opts.signal) };
  }

  async function encode(input, payloadBits, opts = {}) {
    ensureOpen();
    const { format, quality, strength, background, signal } = parseEncodeOptions(opts);
    const { packet, payload } = buildPacket(payloadBits, schema);

    return queue.run(async () => {
      let { image } = decodeImage(input, { maxPixels, orientation });
      if (format === 'jpeg' && !isOpaque(image.data)) {
        if (!background) {
          throw new TrustmarkError('UNSUPPORTED', 'The image has transparent pixels, which JPEG cannot store; pass background: [r, g, b] to flatten it first, or use PNG');
        }
        image = flatten(image, background);
      }
      const encoder = await raceAbort(load('encoder'), signal);
      throwIfAborted(signal);
      let stego;
      try {
        stego = await core.encodeWatermark(runtimeFor(null, signal), encoder.session, image, packet, { variant: variant.code, strength });
      } catch (error) {
        if (isAbortOf(error, signal) || error instanceof TrustmarkError) throw error;
        throw new TrustmarkError('INFERENCE_FAILED', `Embedding failed: ${error && error.message ? error.message : error}`, { cause: error });
      }
      throwIfAborted(signal);
      const bytes = encodeImage(stego, { format, quality });
      throwIfAborted(signal);
      return {
        bytes,
        format,
        mimeType: MIME_TYPES[format],
        quality,
        payload,
        schema: schema.name,
        variant: variant.code,
        strength,
        backend: encoder.backend,
        width: stego.width,
        height: stego.height,
      };
    }, signal);
  }

  async function decode(input, opts = {}) {
    if (opts === null || typeof opts !== 'object') throw argumentError('decode options must be an object');
    const signal = checkSignal(opts.signal);
    const schemaPolicy = opts.schemaPolicy === undefined ? 'signalled' : opts.schemaPolicy;
    if (schemaPolicy !== 'signalled' && schemaPolicy !== 'any') schemaInfo(schemaPolicy);
    ensureOpen();
    try {
      return await queue.run(async () => {
        const { image } = decodeImage(input, { maxPixels, orientation, tolerant: true });
        const [resizer, decoder] = await raceAbort(Promise.all([load('resizer'), load('decoder')]), signal);
        throwIfAborted(signal);
        let bits;
        try {
          bits = await core.decodeBits(runtimeFor(resizer, signal), decoder.session, image, variant.code);
        } catch (error) {
          if (isAbortOf(error, signal) || error instanceof TrustmarkError) throw error;
          throw new TrustmarkError('INFERENCE_FAILED', `Decoding failed: ${error && error.message ? error.message : error}`, { cause: error });
        }
        const result = decodePacket(bits, variant.code, schemaPolicy);
        return {
          status: result.valid ? 'detected' : 'not_detected',
          payload: result.payload,
          schema: result.schema,
          variant: variant.code,
          backend: decoder.backend,
          bitFlips: result.bitFlips,
          softBinding: result.softBinding,
          error: null,
        };
      }, signal);
    } catch (error) {
      // Aborts, a closed instance, a full queue and invalid arguments are the caller's to
      // handle; anything else means the image could not be checked, which is not "no watermark".
      if (isAbortOf(error, signal)) throw error;
      if (error instanceof TrustmarkError && ['CLOSED', 'QUEUE_FULL', 'INVALID_ARGUMENT'].includes(error.code)) throw error;
      const wrapped = error instanceof TrustmarkError ? error : new TrustmarkError('INFERENCE_FAILED', String(error && error.message ? error.message : error), { cause: error });
      return {
        status: 'error',
        payload: null,
        schema: null,
        variant: variant.code,
        backend: state.decoder.backend,
        bitFlips: null,
        softBinding: null,
        error: wrapped,
      };
    }
  }

  function status() {
    const out = {};
    for (const [role, s] of Object.entries(state)) {
      out[role] = {
        file: s.file,
        status: s.status,
        requestedProviders: s.requested.map(models.providerName),
        backend: s.backend,
        fallbacks: s.fallbacks.map((f) => ({ ...f })),
        path: s.info ? s.info.path : path.join(config.modelDir, s.file),
        sha256: s.info ? s.info.sha256 : null,
        size: s.info ? s.info.size : null,
        verified: s.info ? s.info.verified : false,
        downloaded: s.info ? s.info.downloaded : false,
        loadMs: s.loadMs,
        error: errorSummary(s.error),
      };
    }
    return { variant: variant.code, schema: schema.name, closed: !!closing, models: out };
  }

  async function ready(opts = {}) {
    if (opts === null || typeof opts !== 'object') throw argumentError('ready options must be an object');
    const signal = checkSignal(opts.signal);
    ensureOpen();
    await raceAbort(Promise.all(Object.keys(roles).map(load)), signal);
    return status();
  }

  function close() {
    if (closing) return closing;
    closing = (async () => {
      lifecycle.abort(new TrustmarkError('CLOSED', 'The TrustMark instance was closed while loading models'));
      await queue.close();
      await Promise.all(Object.values(state).map((s) => (s.promise ? s.promise.catch(() => {}) : null)));
      for (const s of Object.values(state)) {
        if (s.session) {
          const session = s.session;
          s.session = null;
          s.status = 'closed';
          if (typeof session.release === 'function') await Promise.resolve(session.release()).catch(() => {});
        }
      }
    })();
    return closing;
  }

  return Object.freeze({
    get variant() { return variant.code; },
    get schema() { return schema.name; },
    get capacity() { return schema.capacity; },
    encode,
    decode,
    ready,
    status,
    close,
  });
}

/** What the JavaScript core supports: variants (encode/decode) and schemas with capacities. */
function capabilities() {
  const variants = {};
  for (const [code, info] of Object.entries(core.VARIANTS)) {
    variants[code] = { encode: typeof info.encoder === 'string', decode: typeof info.decoder === 'string' };
  }
  return { variants, schemas: SCHEMAS.map((s) => ({ name: s.name, capacity: s.capacity })) };
}

module.exports = { createTrustmark, capabilities };
