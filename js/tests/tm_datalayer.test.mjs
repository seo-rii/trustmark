// Tests for the TrustMark data layer (tm_datalayer.js).
// Run from the repository root with: node --test js/tests/*.test.mjs

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import vm from 'node:vm';

const jsDir = path.join(path.dirname(new URL(import.meta.url).pathname), '..');

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

// 100-bit packets produced by the Python reference implementation
// (python/trustmark/datalayer.py). Each packet decodes under its own schema only,
// so the expected result does not depend on the order schemas are tried in.
const VECTORS = [
  { schema: 0, payload: "0001000100110011001111111000000010000110", packet: "0001000100110011001111111000000010000110010111010010010001010000110100011100001100011001110111100000" },
  { schema: 0, payload: "1011111011101011100111001001111101101110", packet: "1011111011101011100111001001111101101110011010101111010110010000111111100010001011011100010010100000" },
  { schema: 1, payload: "0100001111001100111110011010001010001001101011001111111100001", packet: "0100001111001100111110011010001010001001101011001111111100001100001011101010001000011011111011010001" },
  { schema: 1, payload: "0000000010111011101100110001001100001000000001010101110011001", packet: "0000000010111011101100110001001100001000000001010101110011001101100001110111010011010110110000100001" },
  { schema: 2, payload: "11010110101000001010111100111111101111001100000111111001100001111000", packet: "1101011010100000101011110011111110111100110000011111100110000111100000100101100100101101100100010010" },
  { schema: 2, payload: "01000101101001011101100011000011010000000101010101001011011101001001", packet: "0100010110100101110110001100001101000000010101010100101101110100100110010100010110110101111001010010" },
  { schema: 3, payload: "111010100111011000110001010111110101100111100010001100101000001101001110110", packet: "1110101001110110001100010101111101011001111000100011001010000011010011101100001010001101100110000011" },
  { schema: 3, payload: "110001000001101101100011101011110100110111001100101011000100000110100010101", packet: "1100010000011011011000111010111101001101110011001010110001000001101000101010010000110100110110100011" },
];

const SCHEMA_NAMES = ['BCH_SUPER', 'BCH_5', 'BCH_4', 'BCH_3'];

function toBools(bits) {
  return bits.split('').map((c) => c === '1');
}

// Overwrite the two schema bits at the end of the packet.
function withSchemaBits(bits, version) {
  const out = bits.slice();
  out[98] = Boolean(version & 2);
  out[99] = Boolean(version & 1);
  return out;
}

function softBindingValue(dataobj) {
  return dataobj.softBindingInfo['c2pa.soft-binding'].blocks[0].value;
}

test('recovers every schema whatever the schema bits say', () => {
  const dl = loadDataLayer();
  for (const vector of VECTORS) {
    for (let hinted = 0; hinted < 4; hinted++) {
      const label = `schema ${vector.schema}, schema bits ${hinted}`;
      const dataobj = dl.DataLayer_Decode(withSchemaBits(toBools(vector.packet), hinted), dl.eccengine, 'Q');

      assert.equal(dataobj.valid, true, label);
      assert.equal(dataobj.schema, SCHEMA_NAMES[vector.schema], label);
      assert.equal(dataobj.data_binary, vector.payload, label);
    }
  }
});

test('soft binding uses the recovered schema', () => {
  const dl = loadDataLayer();
  for (const vector of VECTORS) {
    for (let hinted = 0; hinted < 4; hinted++) {
      const label = `schema ${vector.schema}, schema bits ${hinted}`;
      const dataobj = dl.DataLayer_Decode(withSchemaBits(toBools(vector.packet), hinted), dl.eccengine, 'P');

      assert.equal(softBindingValue(dataobj), `${vector.schema}*${vector.payload}`, label);
      assert.equal(dataobj.softBindingInfo['c2pa.soft-binding'].alg, 'com.adobe.trustmark.P', label);
    }
  }
});

test('corrects bit flips within the schema tolerance', () => {
  const dl = loadDataLayer();
  const tolerance = [8, 5, 4, 3];
  for (const vector of VECTORS) {
    const bits = toBools(vector.packet);
    // Spread the flips over the data and ECC bits
    for (let i = 0; i < tolerance[vector.schema]; i++) {
      const pos = (i * 37 + 11) % 96;
      bits[pos] = !bits[pos];
    }
    const dataobj = dl.DataLayer_Decode(bits, dl.eccengine, 'Q');

    assert.equal(dataobj.valid, true, `schema ${vector.schema}`);
    assert.equal(dataobj.data_binary, vector.payload, `schema ${vector.schema}`);
    assert.equal(softBindingValue(dataobj), `${vector.schema}*${vector.payload}`);
  }
});

test('reports no soft binding when no schema recovers the data', () => {
  const dl = loadDataLayer();
  // Too many flips in the data bits for any schema to correct
  const bits = toBools(VECTORS[2].packet);
  for (let pos = 0; pos < 40; pos += 2) bits[pos] = !bits[pos];
  const dataobj = dl.DataLayer_Decode(bits, dl.eccengine, 'Q');

  assert.equal(dataobj.valid, false);
  assert.equal(dataobj.schema, 'BCH_5');
  assert.equal(dataobj.softBindingInfo, null);
});
