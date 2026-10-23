'use strict';

// Model files: lookup in modelDir, optional download (temporary file and atomic rename),
// size limit and SHA-256 verification, and session creation on the requested execution
// providers.

const crypto = require('node:crypto');
const fsp = require('node:fs/promises');
const path = require('node:path');

const { TrustmarkError, throwIfAborted } = require('./errors');

const DEFAULT_MODEL_BASE_URL = 'https://cai-watermark.adobe.net/watermarking/trustmark-models/';
const DEFAULT_MAX_MODEL_BYTES = 256 * 1024 * 1024;

function sha256(bytes) {
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

function normalizeDigests(digests) {
  const out = new Map();
  if (digests === undefined || digests === null) return out;
  if (typeof digests !== 'object') throw new TrustmarkError('INVALID_ARGUMENT', 'expectedDigests must map model file names to SHA-256 hex digests');
  for (const [file, value] of Object.entries(digests)) {
    const hex = typeof value === 'string' ? value.trim().toLowerCase().replace(/^sha256[:-]/, '') : '';
    if (!/^[0-9a-f]{64}$/.test(hex)) {
      throw new TrustmarkError('INVALID_ARGUMENT', `expectedDigests[${JSON.stringify(file)}] must be a SHA-256 hex digest`);
    }
    out.set(file, hex);
  }
  return out;
}

/** Validates the base URL: https, or http to a loopback address (for local mirrors and tests). */
function normalizeBaseUrl(value) {
  let url;
  try {
    url = new URL(value);
  } catch (error) {
    throw new TrustmarkError('INVALID_ARGUMENT', `modelBaseUrl is not a valid URL: ${value}`, { cause: error });
  }
  const loopback = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && loopback)) {
    throw new TrustmarkError('INVALID_ARGUMENT', 'modelBaseUrl must use https (plain http is only accepted for localhost)');
  }
  if (!url.pathname.endsWith('/')) url.pathname += '/';
  return url;
}

/**
 * Downloads `url` into `dest` atomically: the body is streamed into a temporary file in the
 * same directory, checked against the size limit and the expected digest, flushed to disk and
 * renamed over `dest`. On any failure the temporary file is removed and `dest` is untouched.
 */
async function download({ url, dest, fetchImpl, maxBytes, digest, signal }) {
  const tmp = path.join(path.dirname(dest), `.${path.basename(dest)}.${process.pid}.${crypto.randomBytes(6).toString('hex')}.part`);
  let handle = null;
  try {
    let response;
    try {
      response = await fetchImpl(url, { signal, redirect: 'follow' });
    } catch (error) {
      throwIfAborted(signal);
      throw new TrustmarkError('MODEL_DOWNLOAD_FAILED', `Could not download ${url}: ${error && error.message ? error.message : error}`, { cause: error });
    }
    if (!response.ok) throw new TrustmarkError('MODEL_DOWNLOAD_FAILED', `Could not download ${url}: HTTP ${response.status}`);
    const declared = Number(response.headers.get('content-length'));
    if (Number.isFinite(declared) && declared > maxBytes) {
      throw new TrustmarkError('MODEL_INTEGRITY', `${url} is ${declared} bytes, more than maxModelBytes (${maxBytes})`);
    }
    if (!response.body) throw new TrustmarkError('MODEL_DOWNLOAD_FAILED', `Could not download ${url}: empty response`);

    handle = await fsp.open(tmp, 'wx', 0o644);
    const hash = crypto.createHash('sha256');
    let size = 0;
    try {
      for await (const chunk of response.body) {
        size += chunk.length;
        if (size > maxBytes) throw new TrustmarkError('MODEL_INTEGRITY', `${url} is larger than maxModelBytes (${maxBytes})`);
        hash.update(chunk);
        await handle.write(chunk);
      }
    } catch (error) {
      throwIfAborted(signal);
      if (error instanceof TrustmarkError) throw error;
      throw new TrustmarkError('MODEL_DOWNLOAD_FAILED', `Could not download ${url}: ${error && error.message ? error.message : error}`, { cause: error });
    }
    const actual = hash.digest('hex');
    if (digest && actual !== digest) {
      throw new TrustmarkError('MODEL_INTEGRITY', `${url} has SHA-256 ${actual}, expected ${digest}`, { details: { expected: digest, actual } });
    }
    await handle.sync();
    await handle.close();
    handle = null;
    throwIfAborted(signal);
    await fsp.rename(tmp, dest);
    return { size, sha256: actual };
  } catch (error) {
    if (handle) await handle.close().catch(() => {});
    await fsp.rm(tmp, { force: true }).catch(() => {});
    throw error;
  }
}

/**
 * Returns the bytes of a model file, downloading it first when it is missing and downloads
 * are allowed. The digest is checked on the bytes that are returned, which are the bytes the
 * session is created from.
 */
async function readModel(file, config, signal) {
  const dest = path.join(config.modelDir, file);
  const digest = config.digests.get(file);
  if (config.requireDigests && !digest) {
    throw new TrustmarkError('MODEL_INTEGRITY', `No expected SHA-256 digest was given for ${file} (requireDigests is set)`);
  }

  let stat = null;
  try {
    stat = await fsp.stat(dest);
  } catch (error) {
    if (error.code !== 'ENOENT') throw new TrustmarkError('MODEL_LOAD_FAILED', `Could not read ${dest}: ${error.message}`, { cause: error });
  }
  let downloaded = false;
  if (!stat) {
    if (!config.allowDownload) {
      throw new TrustmarkError('MODEL_NOT_FOUND', `Model file ${file} was not found in ${config.modelDir} and downloads are disabled (allowDownload: false); get it from ${new URL(file, config.baseUrl)}`, { details: { file, path: dest } });
    }
    await fsp.mkdir(config.modelDir, { recursive: true });
    await download({ url: new URL(file, config.baseUrl).href, dest, fetchImpl: config.fetch, maxBytes: config.maxModelBytes, digest, signal });
    downloaded = true;
    stat = await fsp.stat(dest);
  }
  if (!stat.isFile()) throw new TrustmarkError('MODEL_LOAD_FAILED', `${dest} is not a file`);
  if (stat.size > config.maxModelBytes) {
    throw new TrustmarkError('MODEL_INTEGRITY', `${dest} is ${stat.size} bytes, more than maxModelBytes (${config.maxModelBytes})`);
  }
  const bytes = await fsp.readFile(dest);
  const actual = sha256(bytes);
  if (digest && actual !== digest) {
    throw new TrustmarkError('MODEL_INTEGRITY', `${dest} has SHA-256 ${actual}, expected ${digest}`, { details: { file, path: dest, expected: digest, actual } });
  }
  return { bytes, path: dest, sha256: actual, size: bytes.length, downloaded, verified: !!digest };
}

function providerName(provider) {
  return typeof provider === 'string' ? provider : provider && provider.name;
}

/**
 * Creates an InferenceSession on the first execution provider that works, trying each one on
 * its own so the provider in use is known. ONNX Runtime still runs operators that a provider
 * does not implement on the CPU.
 *
 * @returns {Promise<{session, backend: string, fallbacks: Array<{provider: string, error: string}>}>}
 */
async function createSession(ort, bytes, providers, sessionOptions, label) {
  const fallbacks = [];
  for (const provider of providers) {
    try {
      const session = await ort.InferenceSession.create(bytes, { ...sessionOptions, executionProviders: [provider] });
      return { session, backend: providerName(provider), fallbacks };
    } catch (error) {
      fallbacks.push({ provider: providerName(provider), error: String(error && error.message ? error.message : error) });
    }
  }
  const reasons = fallbacks.map((f) => `${f.provider}: ${f.error}`).join('; ');
  throw new TrustmarkError('MODEL_LOAD_FAILED', `Could not create an ONNX Runtime session for ${label} (${reasons})`, { details: { fallbacks } });
}

module.exports = { DEFAULT_MODEL_BASE_URL, DEFAULT_MAX_MODEL_BYTES, normalizeDigests, normalizeBaseUrl, readModel, createSession, providerName };
