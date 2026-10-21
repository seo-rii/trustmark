// Cross-runtime checks against the Python implementation: images watermarked by this SDK must
// decode to the exact payload in Python, and images watermarked by Python must decode to the
// exact payload here, on the repository images and a wide crop, for every variant.
//
// Needs the models of the four variants (TRUSTMARK_MODEL_DIR, see models.test.mjs) and a Python
// interpreter with the dependencies of python/trustmark (torch, Pillow, numpy, omegaconf, ...)
// and the checkpoints of the four variants:
//   TRUSTMARK_PYTHON=/path/to/venv/bin/python   the interpreter
//   TRUSTMARK_PYTHONPATH=/path/to/python        directory holding the trustmark package
//                                               (default: the python/ directory of this repo)
//   TRUSTMARK_PY_MODEL_DIR=/path                optional model_dir for Python versions with it
// Without them these tests are reported as skipped.

import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import test from 'node:test';
import { promisify } from 'node:util';

import { IMAGES_DIR, MODEL_DIR, NODE_DIR, REPO_DIR, modelSkipReason, tempDir } from './helpers.mjs';

const require = createRequire(import.meta.url);
const { createTrustmark } = require('..');
const { decodeImage, encodeImage } = require('../lib/image.js');

const PYTHON = process.env.TRUSTMARK_PYTHON;
const PYTHONPATH = process.env.TRUSTMARK_PYTHONPATH || path.join(REPO_DIR, 'python');
const skip = ['Q', 'P', 'B', 'C'].map((v) => modelSkipReason(v)).find(Boolean) || (PYTHON ? false : 'set TRUSTMARK_PYTHON (and TRUSTMARK_PYTHONPATH) to run the Python cross-checks');

async function python(job, dir) {
  const jobFile = path.join(dir, `job-${Date.now()}.json`);
  fs.writeFileSync(jobFile, JSON.stringify(job));
  const run = promisify(execFile)(PYTHON, [path.join(NODE_DIR, 'test/fixtures/python_crosscheck.py'), PYTHONPATH, jobFile], { maxBuffer: 1 << 24 });
  run.child.stdin.end();  // Python must not wait for input
  const { stdout } = await run;
  return JSON.parse(stdout.trim().split('\n').pop());
}

/** Writes the inputs: the repository images and a wide (8:3) crop of ripley.jpg as PNG. */
function inputs(dir) {
  const ripley = decodeImage(fs.readFileSync(path.join(IMAGES_DIR, 'ripley.jpg')), { maxPixels: 1e8 }).image;
  const [width, height, left, top] = [2400, 900, 800, 600];
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    const start = ((top + y) * ripley.width + left) * 4;
    data.set(ripley.data.subarray(start, start + width * 4), y * width * 4);
  }
  const wide = path.join(dir, 'wide.png');
  fs.writeFileSync(wide, encodeImage({ data, width, height }, { format: 'png' }));
  return [
    { name: 'ghost.png (RGBA)', file: path.join(IMAGES_DIR, 'ghost.png') },
    { name: 'ufo_240.jpg', file: path.join(IMAGES_DIR, 'ufo_240.jpg') },
    { name: 'ripley.jpg', file: path.join(IMAGES_DIR, 'ripley.jpg') },
    { name: 'wide 2400x900', file: wide },
  ];
}

// Payloads per case: distinct, with every variant and schema
const CASES = [
  { image: 0, variant: 'Q', schema: 'BCH_5', payload: '1101010001000100000000111101010000101101000100010011001100111' },
  { image: 1, variant: 'Q', schema: 'BCH_5', payload: '1111000000010000110101111101110101110011100100111110110111000' },
  { image: 2, variant: 'Q', schema: 'BCH_5', payload: '0000111100001111000011110000111100001111000011110000111100001' },
  { image: 3, variant: 'Q', schema: 'BCH_5', payload: '1010101010101010101010101010101010101010101010101010101010101' },
  { image: 1, variant: 'Q', schema: 'BCH_3', payload: '110010101111000011001010111100001100101011110000110010101111000011001010111' },
  { image: 3, variant: 'Q', schema: 'BCH_SUPER', payload: '0110100110010110011010011001011001101001' },
  { image: 0, variant: 'P', schema: 'BCH_5', payload: '0011010111001010001101011100101000110101110010100011010111001' },
  { image: 3, variant: 'P', schema: 'BCH_SUPER', payload: '1110010011100100111001001110010011100100' },
  { image: 1, variant: 'B', schema: 'BCH_5', payload: '1001011010010110100101101001011010010110100101101001011010010' },
  { image: 2, variant: 'B', schema: 'BCH_4', payload: '01011100010111000101110001011100010111000101110001011100010111000101' },
  { image: 0, variant: 'C', schema: 'BCH_5', payload: '1111100000111110000011111000001111100000111110000011111000001' },
  { image: 3, variant: 'C', schema: 'BCH_3', payload: '101100111000111100001111100000111111000000111111100000001111111100000000111' },
];

test('Node encode -> Python decode and Python encode -> Node decode give the exact payload', { skip, timeout: 900000 }, async (t) => {
  const dir = tempDir(t);
  const images = inputs(dir);
  const instances = {};
  const tm = (variant, schema) => (instances[`${variant}/${schema}`] ??= createTrustmark({ modelDir: MODEL_DIR, variant, schema }));
  t.after(() => Promise.all(Object.values(instances).map((i) => i.close())));

  // Node encodes PNG (and one JPEG at quality 90)
  const nodeOutputs = [];
  for (const [i, c] of CASES.entries()) {
    const result = await tm(c.variant, c.schema).encode(fs.readFileSync(images[c.image].file), c.payload);
    const file = path.join(dir, `node-${i}.png`);
    fs.writeFileSync(file, result.bytes);
    nodeOutputs.push({ ...c, file, format: 'png' });
  }
  const jpeg = await tm('Q', 'BCH_5').encode(fs.readFileSync(images[1].file), CASES[1].payload, { format: 'jpeg', quality: 90 });
  fs.writeFileSync(path.join(dir, 'node-jpeg.jpg'), jpeg.bytes);
  nodeOutputs.push({ ...CASES[1], file: path.join(dir, 'node-jpeg.jpg'), format: 'jpeg q90' });

  const pyOutputs = CASES.map((c, i) => ({ ...c, input: images[c.image].file, output: path.join(dir, `py-${i}.png`) }));
  const py = await python({
    decode: nodeOutputs.map((o) => ({ file: o.file, variant: o.variant })),
    encode: pyOutputs.map((o) => ({ input: o.input, output: o.output, payload: o.payload, variant: o.variant, schema: o.schema })),
  }, dir);

  const rows = [];
  for (const o of nodeOutputs) {
    const r = py.decode[o.file];
    const capacity = tm(o.variant, o.schema).capacity;
    rows.push({ direction: 'node->python', image: images[o.image].name, variant: o.variant, format: o.format, schema: o.schema, exact: r.detected && r.payload === o.payload.padEnd(capacity, '0') && r.schema === o.schema });
  }
  for (const o of pyOutputs) {
    const decoded = await tm(o.variant, 'BCH_5').decode(fs.readFileSync(o.output));
    const capacity = tm(o.variant, o.schema).capacity;
    rows.push({ direction: 'python->node', image: images[o.image].name, variant: o.variant, format: 'png', schema: o.schema, exact: decoded.status === 'detected' && decoded.payload === o.payload.padEnd(capacity, '0') && decoded.schema === o.schema, bitFlips: decoded.bitFlips });
  }
  for (const row of rows) t.diagnostic(JSON.stringify(row));
  assert.deepEqual(rows.filter((r) => !r.exact), []);
});
