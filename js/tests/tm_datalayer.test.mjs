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

// Edge cases from the Python data layer: all-ones payloads, and short payloads that are
// padded with zeros up to the schema capacity
const EDGE_VECTORS = [
  { schema: 0, payload: "1111111111111111111111111111111111111111", packet: "1111111111111111111111111111111111111111110101111101100001100000101111101111000111001101010010010000" },
  { schema: 0, payload: "1", packet: "1000000000000000000000000000000000000000111101101110010000010111110011001101111101111100000110000000" },
  { schema: 0, payload: "10101010101010101010", packet: "1010101010101010101000000000000000000000100001010111011101101010001001101000010001110000101101010000" },
  { schema: 1, payload: "1111111111111111111111111111111111111111111111111111111111111", packet: "1111111111111111111111111111111111111111111111111111111111111111101010101001101001110000100100100001" },
  { schema: 1, payload: "1", packet: "1000000000000000000000000000000000000000000000000000000000000011000010111111011110110110111111100001" },
  { schema: 1, payload: "101010101010101010101010101010", packet: "1010101010101010101010101010100000000000000000000000000000000010110100111110101000010001001111110001" },
  { schema: 2, payload: "11111111111111111111111111111111111111111111111111111111111111111111", packet: "1111111111111111111111111111111111111111111111111111111111111111111111011001100000100011011100000010" },
  { schema: 2, payload: "1", packet: "1000000000000000000000000000000000000000000000000000000000000000000000010101100110111011010111010010" },
  { schema: 3, payload: "111111111111111111111111111111111111111111111111111111111111111111111111111", packet: "1111111111111111111111111111111111111111111111111111111111111111111111111111011011000111110001110011" },
  { schema: 3, payload: "101010101010101010101010101010101010", packet: "1010101010101010101010101010101010100000000000000000000000000000000000000001101100010110110110100011" },
];

const SCHEMA_NAMES = ['BCH_SUPER', 'BCH_5', 'BCH_4', 'BCH_3'];
const SCHEMA_CAPACITY = [40, 61, 68, 75];

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

test('builds the same packets as the Python data layer', () => {
  const dl = loadDataLayer();
  for (const vector of [...VECTORS, ...EDGE_VECTORS]) {
    const packet = dl.DataLayer_Encode(vector.payload, vector.schema, dl.eccengine);
    assert.equal(packet.join(''), vector.packet, `schema ${vector.schema}, payload ${vector.payload}`);
  }
});

test('encoded packets decode to the zero-padded payload', () => {
  const dl = loadDataLayer();
  for (const vector of EDGE_VECTORS) {
    const packet = dl.DataLayer_Encode(vector.payload, vector.schema, dl.eccengine);
    const dataobj = dl.DataLayer_Decode(packet.map((bit) => bit === 1), dl.eccengine, 'Q');

    assert.equal(dataobj.valid, true);
    assert.equal(dataobj.schema, SCHEMA_NAMES[vector.schema]);
    assert.equal(dataobj.data_binary, vector.payload.padEnd(SCHEMA_CAPACITY[vector.schema], '0'));
  }
});

test('rejects payloads that do not fit the schema', () => {
  const dl = loadDataLayer();
  const encode = (payload, version = 1) => () => dl.DataLayer_Encode(payload, version, dl.eccengine);

  assert.throws(encode('1'.repeat(62)), /62 bits but BCH_5 allows at most 61 bits/);
  assert.throws(encode('1'.repeat(41), 0), /41 bits but BCH_SUPER allows at most 40 bits/);
  assert.throws(encode('0102'), /'0' and '1' characters/);
  // A payload given as a number would silently lose bits beyond 2^53
  assert.throws(encode(1234567890123), /'0' and '1' characters/);
  assert.throws(encode('101', 4), /Invalid schema version 4/);
  assert.equal(dl.DataLayer_Encode('1'.repeat(61), 1, dl.eccengine).length, 100);
});

test('maps schema names and numbers to versions like TrustMark.Encoding', () => {
  const dl = loadDataLayer();
  SCHEMA_NAMES.forEach((name, version) => {
    assert.equal(dl.DataLayer_GetSchemaVersion(name), version);
    assert.equal(dl.DataLayer_GetSchemaVersion(version), version);
  });
  for (const bad of ['BCH_6', 'bch_5', '1', 4, -1, 1.5, null, undefined]) {
    assert.throws(() => dl.DataLayer_GetSchemaVersion(bad), /Unknown watermark schema .*; expected BCH_SUPER, BCH_5, BCH_4, BCH_3 or 0 to 3/);
  }
});
