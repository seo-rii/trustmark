// The verified-release example (examples/verified-release/release.mjs): released files, job
// records, idempotency, and publication failures before and after the public link commits.
// Uses the fake ONNX Runtime, plus the real models when TRUSTMARK_MODEL_DIR is set.
// Run from js/node with: npm test

import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import test from 'node:test';

import { createReleaser, payloadForJob } from '../examples/verified-release/release.mjs';
import { IMAGES_DIR, MODEL_DIR, NO_WATERMARK_BITS, fakeModelDir, fakeOrt, modelSkipReason, rgbaImage, tempDir } from './helpers.mjs';

const require = createRequire(import.meta.url);
const { createTrustmark } = require('..');
const { encodeImage } = require('../lib/image.js');
const { buildPacket, schemaInfo } = require('../lib/core.js');

const GENERATED = encodeImage(rgbaImage(40, 30), { format: 'png' });

function setup(t, fake = {}, options = {}) {
  const { ort, log } = fakeOrt(fake);
  const trustmark = options.trustmark || createTrustmark({ modelDir: fakeModelDir(t), ort });
  t.after(() => trustmark.close());
  const root = tempDir(t);
  const dirs = { privateDir: path.join(root, 'private'), publicDir: path.join(root, 'public') };
  const releaser = createReleaser({ trustmark, ...dirs, ...options });
  return { releaser, log, trustmark, ...dirs };
}

/** Every entry under dir, dot files included. */
function listing(dir) {
  return fs.existsSync(dir) ? fs.readdirSync(dir, { recursive: true }).sort() : [];
}

test('releases the verified image and records each state', async (t) => {
  const { releaser, publicDir, privateDir } = setup(t);
  const released = await releaser.release('job-1', GENERATED);
  assert.equal(released.state, 'released');
  assert.equal(released.reused, false);
  assert.equal(released.path, path.join(publicDir, 'job-1.png'));
  assert.equal(released.payload, payloadForJob('job-1', 61));
  assert.deepEqual(listing(publicDir), ['job-1.png']);
  assert.equal(crypto.createHash('sha256').update(fs.readFileSync(released.path)).digest('hex'), released.sha256);
  assert.deepEqual(listing(privateDir), ['jobs', 'jobs/job-1.json'], 'no original or staging file left behind');
  const record = await releaser.record('job-1');
  assert.deepEqual([record.state, record.sha256, record.verification.schema], ['released', released.sha256, 'BCH_5']);
  assert.equal((fs.statSync(path.join(privateDir, 'jobs')).mode & 0o777), 0o700);
});

test('a released job is not processed again, and concurrent calls share one run', async (t) => {
  const { releaser, log, publicDir } = setup(t);
  const [a, b] = await Promise.all([releaser.release('job-2', GENERATED), releaser.release('job-2', GENERATED)]);
  assert.equal(a, b);
  assert.equal(log.packets.length, 1);
  const again = await releaser.release('job-2', GENERATED);
  assert.equal(again.reused, true);
  assert.equal(again.sha256, a.sha256);
  assert.equal(log.packets.length, 1, 'no second watermarking');
  assert.deepEqual(listing(publicDir), ['job-2.png']);
});

test('a record write failure after publication returns the release and is recovered on retry', async (t) => {
  const { releaser, log, trustmark, privateDir, publicDir } = setup(t);
  const recordPath = path.join(privateDir, 'jobs', 'record-failure.json');
  const rename = fsp.rename;
  t.mock.method(fsp, 'rename', async (from, to) => {
    if (to === recordPath && JSON.parse(await fsp.readFile(from, 'utf8')).state === 'released') {
      throw Object.assign(new Error('cannot save released state'), { code: 'ENOSPC' });
    }
    return rename(from, to);
  });

  const released = await releaser.release('record-failure', GENERATED);
  assert.equal(released.state, 'released');
  assert.equal(released.warnings[0].operation, 'save_record');
  assert.equal(released.warnings[0].code, 'ENOSPC');
  assert.equal(released.warnings[0].path, recordPath);
  assert.equal(crypto.createHash('sha256').update(fs.readFileSync(released.path)).digest('hex'), released.sha256);
  assert.equal((await releaser.record('record-failure')).state, 'verified', 'the durable verification record is retained');

  t.mock.restoreAll();
  const restarted = createReleaser({ trustmark, privateDir, publicDir });
  const recovered = await restarted.release('record-failure', Buffer.from('not a new image'));
  assert.equal(recovered.reused, true);
  assert.equal(recovered.sha256, released.sha256);
  assert.deepEqual(recovered.warnings, []);
  assert.equal((await restarted.record('record-failure')).state, 'released');
  assert.equal(log.packets.length, 1, 'recovery does not encode the job a second time');

  // A verified record only authorizes recovery of exactly the verified bytes.
  const record = await restarted.record('record-failure');
  record.state = 'verified';
  fs.writeFileSync(recordPath, JSON.stringify(record));
  fs.writeFileSync(released.path, 'tampered');
  await assert.rejects(restarted.release('record-failure', GENERATED), { code: 'RELEASE_CONFLICT' });
  assert.equal(log.packets.length, 1);
  assert.equal(fs.readFileSync(released.path, 'utf8'), 'tampered');
});

for (const target of ['original', 'staging']) {
  test(`${target} cleanup failure cannot turn a committed release into a failure`, async (t) => {
    const { releaser, log, privateDir } = setup(t);
    const originalPath = path.join(privateDir, 'cleanup-job.original');
    const rm = fsp.rm;
    t.mock.method(fsp, 'rm', async (file, options) => {
      if (target === 'original' ? file === originalPath
        : path.dirname(file) === privateDir && /^\.cleanup-job\..*\.png$/.test(path.basename(file))) {
        throw Object.assign(new Error('cannot remove private file'), { code: 'EACCES' });
      }
      return rm(file, options);
    });
    const released = await releaser.release('cleanup-job', GENERATED);
    assert.equal(released.state, 'released');
    assert.equal(released.warnings[0].operation, `${target}_cleanup`);
    assert.equal(released.warnings[0].code, 'EACCES');
    assert.ok(fs.existsSync(released.warnings[0].path));
    assert.equal((await releaser.record('cleanup-job')).state, 'released');
    assert.equal(crypto.createHash('sha256').update(fs.readFileSync(released.path)).digest('hex'), released.sha256);
    t.mock.restoreAll();
    const again = await releaser.release('cleanup-job', GENERATED);
    assert.equal(again.reused, true);
    assert.equal(log.packets.length, 1);
    assert.ok(!fs.existsSync(originalPath), 'a retry also completes original cleanup');
  });
}

test('a failed public link still rejects and leaves the image private', async (t) => {
  const { releaser, privateDir, publicDir } = setup(t);
  t.mock.method(fsp, 'link', async () => { throw Object.assign(new Error('link refused'), { code: 'EACCES' }); });
  await assert.rejects(releaser.release('link-failure', GENERATED), { code: 'EACCES' });
  assert.deepEqual(listing(publicDir), []);
  assert.equal((await releaser.record('link-failure')).state, 'failed');
  assert.deepEqual(listing(privateDir), ['jobs', 'jobs/link-failure.json', 'link-failure.original']);
});

const FAILURES = [
  { name: 'verification mismatch', fake: { decoderBits: () => buildPacket('1'.repeat(61), schemaInfo('BCH_5')).packet.join('') }, code: 'VERIFICATION_FAILED', reason: 'payload_mismatch' },
  { name: 'watermark not found', fake: { decoderBits: () => NO_WATERMARK_BITS }, code: 'VERIFICATION_FAILED', reason: 'not_detected' },
  { name: 'decoder error', fake: { hook: (role) => { if (role === 'decoder') throw new Error('decoder crashed'); } }, code: 'VERIFICATION_FAILED', reason: 'decode_error' },
  { name: 'encoder error', fake: { hook: (role) => { if (role === 'encoder') throw new Error('encoder crashed'); } }, code: 'INFERENCE_FAILED' },
];

for (const f of FAILURES) {
  test(`${f.name}: nothing reaches the public directory`, async (t) => {
    const { releaser, publicDir, privateDir } = setup(t, f.fake);
    await assert.rejects(releaser.release('job-3', GENERATED), (error) => {
      assert.equal(error.code, f.code);
      if (f.reason) assert.equal(error.reason, f.reason);
      return true;
    });
    assert.deepEqual(listing(publicDir), []);
    const record = await releaser.record('job-3');
    assert.equal(record.state, 'failed');
    assert.equal(record.error.code, f.code);
    assert.deepEqual(listing(privateDir), ['job-3.original', 'jobs', 'jobs/job-3.json'], 'the original stays private, no staging file is left');
  });
}

test('a timeout or abort before publishing leaves nothing in the public directory', async (t) => {
  const { releaser, publicDir } = setup(t, { hook: (role) => (role === 'decoder' ? new Promise((resolve) => setTimeout(resolve, 300)) : undefined) });
  await assert.rejects(releaser.release('job-4', GENERATED, { signal: AbortSignal.timeout(100) }), { name: 'TimeoutError' });
  await assert.rejects(releaser.release('job-5', GENERATED, { signal: AbortSignal.abort() }), { name: 'AbortError' });
  assert.deepEqual(listing(publicDir), []);
  assert.equal((await releaser.record('job-4')).state, 'failed');
});

test('a failed job can be released by a later call', async (t) => {
  let fail = true;
  const { releaser, publicDir } = setup(t, { hook: (role) => { if (role === 'decoder' && fail) throw new Error('transient'); } });
  await assert.rejects(releaser.release('job-6', GENERATED));
  fail = false;
  assert.equal((await releaser.release('job-6', GENERATED)).state, 'released');
  assert.deepEqual(listing(publicDir), ['job-6.png']);
});

test('never replaces a different file in the public directory', async (t) => {
  const { releaser, publicDir } = setup(t);
  fs.mkdirSync(publicDir, { recursive: true });
  fs.writeFileSync(path.join(publicDir, 'job-7.png'), 'someone else');
  await assert.rejects(releaser.release('job-7', GENERATED), { code: 'RELEASE_CONFLICT' });
  assert.equal(fs.readFileSync(path.join(publicDir, 'job-7.png'), 'utf8'), 'someone else');

  const released = await releaser.release('job-8', GENERATED);
  fs.writeFileSync(released.path, 'tampered');
  await assert.rejects(releaser.release('job-8', GENERATED), { code: 'RELEASE_CONFLICT' });
});

test('rejects unsafe job ids and overlapping directories', async (t) => {
  const { releaser, publicDir, privateDir } = setup(t);
  for (const jobId of ['../escape', '', '.hidden', 'a/b', 'x'.repeat(129), 42]) {
    await assert.rejects(releaser.release(jobId, GENERATED), { code: 'INVALID_JOB_ID' }, String(jobId));
  }
  assert.deepEqual([listing(publicDir), listing(privateDir)], [[], []]);
  const trustmark = createTrustmark({ modelDir: fakeModelDir(t), ort: fakeOrt().ort });
  t.after(() => trustmark.close());
  const root = tempDir(t);
  assert.throws(() => createReleaser({ trustmark, privateDir: path.join(root, 'public', 'private'), publicDir: path.join(root, 'public') }), { code: 'INVALID_CONFIG' });
  assert.throws(() => createReleaser({ trustmark, privateDir: root, publicDir: root }), { code: 'INVALID_CONFIG' });
});

for (const overlap of ['same', 'private-in-public', 'public-in-private']) {
  test(`rejects directory aliases before writing private data (${overlap})`, async (t) => {
    const root = tempDir(t);
    const actual = path.join(root, 'actual');
    const nested = path.join(actual, 'nested');
    const alias = path.join(root, 'alias');
    fs.mkdirSync(nested, { recursive: true });
    fs.symlinkSync(actual, alias, process.platform === 'win32' ? 'junction' : 'dir');
    const dirs = overlap === 'same' ? { privateDir: alias, publicDir: actual }
      : overlap === 'private-in-public' ? { privateDir: path.join(alias, 'nested'), publicDir: actual }
      : { privateDir: actual, publicDir: path.join(alias, 'nested') };
    const { releaser, log } = setup(t, {}, dirs);
    const before = listing(root);
    await assert.rejects(releaser.release('alias-job', GENERATED), { code: 'INVALID_CONFIG' });
    assert.deepEqual(listing(root), before, 'no originals, job records or staging files are written');
    assert.equal(log.runs.length, 0, 'invalid directories are rejected before inference');
  });
}

test('allows symlinks that resolve to separate directories', async (t) => {
  const root = tempDir(t);
  const privateDir = path.join(root, 'private');
  const publicDir = path.join(root, 'public');
  const alias = path.join(root, 'private-alias');
  fs.mkdirSync(privateDir);
  fs.symlinkSync(privateDir, alias, process.platform === 'win32' ? 'junction' : 'dir');
  const { releaser } = setup(t, {}, { privateDir: alias, publicDir });
  const result = await releaser.release('separate-job', GENERATED);
  assert.equal(result.state, 'released');
  assert.deepEqual(listing(publicDir), ['separate-job.png']);
  assert.equal((await releaser.record('separate-job')).state, 'released');
});

const skip = modelSkipReason();

test('real models: releases a verified JPEG and refuses one at quality 1', { skip, timeout: 300000 }, async (t) => {
  const trustmark = createTrustmark({ modelDir: MODEL_DIR });
  const generated = fs.readFileSync(path.join(IMAGES_DIR, 'ufo_240.jpg'));
  const good = setup(t, {}, { trustmark, format: 'jpeg', quality: 90 });
  const released = await good.releaser.release('job-real', generated);
  assert.deepEqual(listing(good.publicDir), ['job-real.jpg']);
  const decoded = await trustmark.decode(fs.readFileSync(released.path));
  assert.equal(decoded.payload, released.payload);

  const bad = setup(t, {}, { trustmark, format: 'jpeg', quality: 1 });
  await assert.rejects(bad.releaser.release('job-real', generated), { code: 'VERIFICATION_FAILED' });
  assert.deepEqual(listing(bad.publicDir), []);
});
