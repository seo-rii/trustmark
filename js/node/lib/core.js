'use strict';

// Glue to the shared JavaScript implementation in js/: tm_core.js (pixel processing and model
// calls) and tm_datalayer.js (packet building and BCH error correction). Nothing here
// re-implements them.

const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const { TrustmarkError } = require('./errors');

const JS_DIR = path.join(__dirname, '..', '..');
const core = require(path.join(JS_DIR, 'tm_core.js'));

// tm_datalayer.js and deps/bch_ecc.min.js are classic browser scripts that define globals, so
// they run in their own vm context instead of polluting the global object of the process.
// (vm is used for isolation of names, not as a security boundary: both are files of this
// repository.) The data layer logs with console, which is silenced here.
function loadDataLayer() {
  const silent = { log() {}, warn() {}, error() {}, info() {}, debug() {} };
  const context = vm.createContext({ console: silent });
  for (const file of ['deps/bch_ecc.min.js', 'tm_datalayer.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS_DIR, file), 'utf8'), context, { filename: path.join(JS_DIR, file) });
  }
  // A top-level `let` binding is not a property of the context object
  const eccengine = vm.runInContext('eccengine', context);
  const count = vm.runInContext('DATALAYER_SCHEMA_COUNT', context);
  const schemas = [];
  for (let version = 0; version < count; version++) {
    schemas.push({
      name: context.DataLayer_GetSchemaName(version),
      version,
      capacity: context.DataLayer_GetSchemaDataBits(version),
    });
  }
  return { context, eccengine, schemas };
}

const dataLayer = loadDataLayer();
const SCHEMAS = Object.freeze(dataLayer.schemas.map((s) => Object.freeze(s)));

/**
 * Returns the settings of a variant from TrustmarkCore.VARIANTS ({ code, encoder, decoder,
 * encodeResolution, decodeResolution, forceSquare, strengthMultiplier }), or throws UNSUPPORTED.
 */
function variantInfo(code) {
  try {
    return core.getVariant(code);
  } catch (error) {
    throw new TrustmarkError('UNSUPPORTED', `Variant ${JSON.stringify(code)} is not supported by the JavaScript core; supported variants: ${Object.keys(core.VARIANTS).join(', ')}`, { cause: error });
  }
}

/** Returns { name, version, capacity } or throws UNSUPPORTED. */
function schemaInfo(name) {
  const schema = SCHEMAS.find((s) => s.name === name);
  if (!schema) {
    throw new TrustmarkError('UNSUPPORTED', `Schema ${JSON.stringify(name)} is not supported; supported schemas: ${SCHEMAS.map((s) => s.name).join(', ')}`);
  }
  return schema;
}

/**
 * Builds the 100-bit packet for a payload bit string with DataLayer_Encode.
 * @returns {{ packet: number[], payload: string }} The packet and the payload as embedded
 *   (padded with zeros to the schema capacity).
 */
function buildPacket(payload, schema) {
  if (typeof payload !== 'string') {
    throw new TrustmarkError('INVALID_ARGUMENT', `The payload must be a string of '0' and '1' characters, not ${payload === null ? 'null' : typeof payload}; numbers cannot hold 61-bit payloads exactly`);
  }
  if (payload.length === 0) throw new TrustmarkError('INVALID_ARGUMENT', 'The payload is empty');
  let packet;
  try {
    packet = dataLayer.context.DataLayer_Encode(payload, schema.version, dataLayer.eccengine);
  } catch (error) {
    throw new TrustmarkError('INVALID_ARGUMENT', String(error && error.message ? error.message : error));
  }
  return { packet: Array.from(packet), payload: payload.padEnd(schema.capacity, '0') };
}

/**
 * Error-corrects decoder bits with DataLayer_Decode.
 *
 * DataLayer_Decode tries the schema signalled by the two version bits first and then every
 * other schema, because the version bits are not protected by the ECC. Each extra schema adds
 * false positives on unwatermarked images (BCH_3 accepts about 9% of random packets), so the
 * policy decides which results count:
 *   'signalled'  only the schema in the version bits, like the Python implementation (default)
 *   'any'        whatever schema DataLayer_Decode recovers
 *   a name       only that schema, whatever the version bits say (set to that schema before
 *                decoding, so DataLayer_Decode tries it first)
 *
 * @returns {{ valid: boolean, payload: string|null, schema: string|null, signalledSchema: string,
 *   bitFlips: number|null, softBinding: object|null }}
 */
function decodePacket(bits, variant, policy = 'signalled') {
  let packet = Array.from(bits, Boolean);
  const signalled = dataLayer.context.DataLayer_GetSchemaName(dataLayer.context.DataLayer_GetVersion(packet));
  let accept = null;
  if (policy === 'signalled') {
    accept = signalled;
  } else if (policy !== 'any') {
    const { version } = schemaInfo(policy);
    packet = [...packet.slice(0, -2), !!((version >> 1) & 1), !!(version & 1)];
    accept = policy;
  }
  const result = dataLayer.context.DataLayer_Decode(packet, dataLayer.eccengine, variant);
  if (!result.valid || (accept !== null && result.schema !== accept)) {
    return { valid: false, payload: null, schema: null, signalledSchema: signalled, bitFlips: null, softBinding: null };
  }
  return {
    valid: true,
    payload: String(result.data_binary),
    schema: String(result.schema),
    signalledSchema: signalled,
    bitFlips: typeof result.bitflips === 'number' ? result.bitflips : null,
    // Values created inside the vm context have that context's prototypes
    softBinding: result.softBindingInfo ? JSON.parse(JSON.stringify(result.softBindingInfo)) : null,
  };
}

module.exports = { core, SCHEMAS, variantInfo, schemaInfo, buildPacket, decodePacket };
