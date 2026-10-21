// Offline mode: with allowDownload false (the default) the SDK never touches the network,
// whether the models are present or missing. A child process runs with a preload that blocks
// and records every socket, DNS lookup and fetch; a control run shows that the guard catches
// network use. Run from js/node with: npm test

import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { promisify } from 'node:util';

import { IMAGES_DIR, MODEL_DIR, NODE_DIR, modelSkipReason, tempDir } from './helpers.mjs';

async function runOffline(t, mode, modelDir) {
  const log = path.join(tempDir(t), 'network.json');
  const { stdout } = await promisify(execFile)(process.execPath, [
    '--require', path.join(NODE_DIR, 'test/fixtures/no-network.cjs'),
    path.join(NODE_DIR, 'test/fixtures/offline-child.mjs'), mode, modelDir, path.join(IMAGES_DIR, 'ufo_240.jpg'),
  ], { env: { ...process.env, NETWORK_LOG: log } });
  return { result: JSON.parse(stdout.trim().split('\n').pop()), attempts: JSON.parse(fs.readFileSync(log, 'utf8')) };
}

test('the network guard records downloads and direct requests (control)', { timeout: 60000 }, async (t) => {
  const { result, attempts } = await runOffline(t, 'control', tempDir(t));
  assert.equal(result.encode, 'MODEL_DOWNLOAD_FAILED');
  assert.equal(result.https, 'ENETBLOCKED');
  assert.ok(attempts.some((a) => a.startsWith('fetch https://cai-watermark.adobe.net/')), attempts.join('\n'));
  assert.ok(attempts.some((a) => a.includes('example.com')), attempts.join('\n'));
});

test('missing models offline: clear errors and no network access', { timeout: 60000 }, async (t) => {
  const { result, attempts } = await runOffline(t, 'offline', tempDir(t));
  assert.deepEqual(result, { encode: 'MODEL_NOT_FOUND', decode: 'MODEL_NOT_FOUND' });
  assert.deepEqual(attempts, []);
});

test('embedding and decoding with local models uses no network', { skip: modelSkipReason(), timeout: 300000 }, async (t) => {
  const { result, attempts } = await runOffline(t, 'offline', MODEL_DIR);
  assert.deepEqual(result, { encode: 'ok', exact: true, decode: 'not_detected' });
  assert.deepEqual(attempts, []);
});
