// Type-checks index.d.ts against test/types/usage.ts with tsc --noEmit (strict).
// Needs the typescript dev dependency (npm install); otherwise reported as skipped.

import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { createRequire } from 'node:module';
import path from 'node:path';
import test from 'node:test';
import { promisify } from 'node:util';

import { NODE_DIR } from './helpers.mjs';

const require = createRequire(import.meta.url);
let tsc = null;
try {
  tsc = require.resolve('typescript/bin/tsc');
} catch {
  // not installed
}

test('index.d.ts type-checks with tsc --noEmit', { skip: tsc ? false : 'typescript is not installed (run npm install in js/node)', timeout: 120000 }, async () => {
  const result = await promisify(execFile)(process.execPath, [tsc, '-p', path.join(NODE_DIR, 'test/types')]).catch((error) => error);
  assert.equal(result.code ?? 0, 0, `${result.stdout || ''}${result.stderr || ''}`);
});
