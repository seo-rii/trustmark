// Checks the data layer (tm_datalayer.js) against the shared packet vectors in tests/vectors.
// Only the decode vectors apply: the JavaScript implementation has no encoder.
// Run from the repository root with: node --test js/tests/*.test.mjs

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const jsDir = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const vectors = JSON.parse(fs.readFileSync(path.join(jsDir, '..', 'tests', 'vectors', 'packets.json'), 'utf8'));

// The demo loads these as classic browser scripts, so evaluate them the same way.
function loadDataLayer() {
  const context = vm.createContext({ console, BigInt, Math });
  for (const file of ['deps/bch_ecc.min.js', 'tm_datalayer.js']) {
    vm.runInContext(fs.readFileSync(path.join(jsDir, file), 'utf8'), context, { filename: file });
  }
  // Top-level `let` bindings are not properties of the global object
  context.eccengine = vm.runInContext('eccengine', context);
  return context;
}

const dataLayer = loadDataLayer();

// Returns { schema, data } for a valid watermark, otherwise null.
function decode(input) {
  const bits = input.split('').map((bit) => bit === '1');
  const result = dataLayer.DataLayer_Decode(bits, dataLayer.eccengine, 'Q');
  return result.valid ? { schema: result.schema, data: result.data_binary } : null;
}

function check(vector, result) {
  const expect = vector.expect.js;
  const original = { schema: vector.schema, data: vector.data };
  if (result !== null) {
    assert.equal(vector.valid_under[result.schema], result.data, 'a valid result must be listed in valid_under');
  }
  switch (expect) {
    case 'valid':
      assert.deepEqual(result, original);
      break;
    case 'invalid':
      assert.equal(result, null);
      break;
    case 'not_original':
      assert.notDeepEqual(result, original);
      break;
    default:
      assert.fail(`unknown expectation ${expect}`);
  }
}

test('vectors are present', () => {
  assert.equal(vectors.format, 'trustmark-packet-vectors');
  assert.ok(vectors.decode.length > 0);
});

for (const vector of vectors.decode) {
  // Known failures still run, but are reported as TODO rather than failing the suite.
  const todo = vector.known_failures?.js;
  test(vector.id, todo ? { todo } : {}, () => check(vector, decode(vector.input)));
}
