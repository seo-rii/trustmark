// Verified release of generated images: generated -> watermarked -> verified -> released.
//
// A generated image is kept in a private directory. encodeVerified embeds the watermark,
// encodes the final file and reads the watermark back from those exact bytes; only then are the
// bytes staged in the private directory and published into the public directory with an atomic,
// no-clobber hard link. Every state change is recorded in a per-job JSON record, and a job id
// that was already released is not processed again. If anything fails, nothing is written to
// the public directory, and the original never leaves the private one.
//
// What this does NOT guarantee:
// - It is not authentication or proof of origin. Anyone with the public models can read a
//   TrustMark payload, and a watermark can be removed or copied onto another image. Use signed
//   C2PA manifests for provenance and treat a payload as a lookup key.
// - The storage or gateway is the enforcement point. This code only controls what it writes:
//   serve the public directory alone, deny access to the private one, and make sure no other
//   code path writes to the public directory.
// - Job ids and payloads are not secrets: do not use them as access tokens.
// - Verification uses this SDK's decoder on the final bytes. Other decoders (for example
//   libjpeg-based ones) can differ on marginal JPEGs, and later resizing or recompression (CDNs,
//   screenshots, social networks) can remove the watermark.
// - It is not a legal or regulatory compliance guarantee, e.g. for AI-content labelling rules.
// - Idempotency covers one host and filesystem (in-process deduplication plus the no-clobber
//   link). Several hosts need a shared job store and locks. The private and public directories
//   must be on the same filesystem (hard links); otherwise publishing fails closed.

import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';

const require = createRequire(import.meta.url);
const { encodeVerified } = require('../..');  // require('trustmark-node') when installed as a dependency

const JOB_ID = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/;

class ReleaseError extends Error {
  constructor(code, message, options) {
    super(message, options);
    this.name = 'ReleaseError';
    this.code = code;
  }
}

const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');

async function sha256File(file) {
  try {
    return sha256(await fs.readFile(file));
  } catch (error) {
    if (error.code === 'ENOENT') return null;
    throw error;
  }
}

/** Writes a file through a temporary file in the same directory, flushed and renamed. */
async function writeAtomic(file, data, mode) {
  const tmp = path.join(path.dirname(file), `.${path.basename(file)}.${crypto.randomBytes(6).toString('hex')}.tmp`);
  const handle = await fs.open(tmp, 'wx', mode);
  try {
    await handle.writeFile(data);
    await handle.sync();
  } finally {
    await handle.close();
  }
  try {
    await fs.rename(tmp, file);
  } catch (error) {
    await fs.rm(tmp, { force: true });
    throw error;
  }
}

/** A deterministic payload for a job id, so that a retried job embeds the same bits. */
export function payloadForJob(jobId, capacity) {
  const digest = crypto.createHash('sha256').update(`trustmark-job:${jobId}`).digest();
  return Array.from(digest, (byte) => byte.toString(2).padStart(8, '0')).join('').slice(0, capacity);
}

function assertSeparateDirectories(privateDir, publicDir) {
  for (const [parent, child] of [[privateDir, publicDir], [publicDir, privateDir]]) {
    const relative = path.relative(parent, child);
    if (!path.isAbsolute(relative) && relative !== '..' && !relative.startsWith(`..${path.sep}`)) {
      throw new ReleaseError('INVALID_CONFIG', 'The private and public directories must be separate, and neither may contain the other');
    }
  }
}

/**
 * @param {object} config
 * @param {object} config.trustmark - Instance from createTrustmark.
 * @param {string} config.privateDir - Originals, staging files and job records; never served
 *   (created with mode 0700 if missing).
 * @param {string} config.publicDir - Released images only; on the same filesystem.
 * @param {'png'|'jpeg'} [config.format] - Final format (default png).
 * @param {number} [config.quality] - JPEG quality.
 * @param {number[]} [config.retryStrengths] - Passed to encodeVerified.
 */
export function createReleaser({ trustmark, privateDir, publicDir, format = 'png', quality, retryStrengths = [] }) {
  [privateDir, publicDir] = [path.resolve(privateDir), path.resolve(publicDir)];
  assertSeparateDirectories(privateDir, publicDir);
  let jobsDir = path.join(privateDir, 'jobs');
  let preparation = null;
  const extension = format === 'jpeg' ? 'jpg' : 'png';
  const running = new Map();

  function prepareDirectories() {
    if (!preparation) {
      preparation = (async () => {
        await fs.mkdir(privateDir, { recursive: true, mode: 0o700 });
        await fs.mkdir(publicDir, { recursive: true });
        const [privatePath, publicPath] = await Promise.all([fs.realpath(privateDir), fs.realpath(publicDir)]);
        // Resolve aliases before creating any originals, records or staging files, and use
        // the resolved paths for the subsequent writes.
        assertSeparateDirectories(privatePath, publicPath);
        privateDir = privatePath;
        publicDir = publicPath;
        jobsDir = path.join(privateDir, 'jobs');
        await fs.mkdir(jobsDir, { recursive: true, mode: 0o700 });
      })().catch((error) => {
        preparation = null;
        throw error;
      });
    }
    return preparation;
  }

  const recordPath = (jobId) => path.join(jobsDir, `${jobId}.json`);

  async function readRecord(jobId) {
    try {
      return JSON.parse(await fs.readFile(recordPath(jobId), 'utf8'));
    } catch (error) {
      if (error.code === 'ENOENT') return null;
      throw error;
    }
  }

  async function saveRecord(record) {
    record.updatedAt = new Date().toISOString();
    await writeAtomic(recordPath(record.jobId), JSON.stringify(record, null, 2), 0o600);
  }

  async function run(jobId, generated, signal) {
    await prepareDirectories();
    const publicPath = path.join(publicDir, `${jobId}.${extension}`);

    const previous = await readRecord(jobId);
    if (previous && previous.state === 'released') {
      if ((await sha256File(publicPath)) !== previous.sha256) {
        throw new ReleaseError('RELEASE_CONFLICT', `Job ${jobId} was released, but ${publicPath} is missing or was changed`);
      }
      return { jobId, state: 'released', path: publicPath, sha256: previous.sha256, payload: previous.payload, reused: true };
    }

    const payload = payloadForJob(jobId, trustmark.capacity);
    const record = { jobId, state: 'generated', payload, format, sha256: null, verification: null, error: null };
    const originalPath = path.join(privateDir, `${jobId}.original`);
    let staged = null;
    try {
      // generated: the original stays in the private directory
      await writeAtomic(originalPath, generated, 0o600);
      await saveRecord(record);

      // watermarked and verified: encodeVerified only returns bytes that decoded exactly
      const result = await encodeVerified(trustmark, generated, payload, { format, quality, retryStrengths, signal });
      record.state = 'verified';
      record.sha256 = result.sha256;
      record.verification = { schema: result.verification.schema, bitFlips: result.verification.bitFlips, strength: result.strength, attempts: result.verification.attempts.length };
      staged = path.join(privateDir, `.${jobId}.${crypto.randomBytes(6).toString('hex')}.${extension}`);
      await writeAtomic(staged, result.bytes, 0o644);
      await saveRecord(record);
      if (signal) signal.throwIfAborted();

      // released: a hard link publishes the complete file atomically and, unlike rename, fails
      // instead of replacing a file that is already there
      try {
        await fs.link(staged, publicPath);
      } catch (error) {
        if (error.code !== 'EEXIST') throw error;
        if ((await sha256File(publicPath)) !== result.sha256) {
          throw new ReleaseError('RELEASE_CONFLICT', `${publicPath} already exists with different content`, { cause: error });
        }
        // An earlier run of this job published the same bytes but did not record it
      }
      record.state = 'released';
      await saveRecord(record);
      await fs.rm(originalPath, { force: true });
      return { jobId, state: 'released', path: publicPath, sha256: result.sha256, payload: result.payload, reused: false };
    } catch (error) {
      record.state = 'failed';
      record.error = { code: error.code || error.name, reason: error.reason || null, message: error.message };
      await saveRecord(record).catch(() => {});
      throw error;
    } finally {
      if (staged) await fs.rm(staged, { force: true });
    }
  }

  /**
   * Releases a generated image (PNG or JPEG bytes) under `jobId`, at most once. Concurrent
   * calls for the same job share one run; a released job returns the existing file
   * (reused: true).
   *
   * @returns {Promise<{jobId, state: 'released', path, sha256, payload, reused}>}
   */
  function release(jobId, generated, { signal } = {}) {
    if (typeof jobId !== 'string' || !JOB_ID.test(jobId)) {
      return Promise.reject(new ReleaseError('INVALID_JOB_ID', 'Job ids are 1 to 128 letters, digits, "-" or "_", starting with a letter or digit'));
    }
    if (!running.has(jobId)) {
      running.set(jobId, run(jobId, generated, signal).finally(() => running.delete(jobId)));
    }
    return running.get(jobId);
  }

  return { release, record: readRecord };
}
