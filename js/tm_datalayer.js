/*!
 * TrustMark Data Layer module
 * Copyright 2024 Adobe. All rights reserved.
 * Licensed under the MIT License.
 * 
 * NOTICE: Adobe permits you to use, modify, and distribute this file in
 * accordance with the terms of the Adobe license agreement accompanying it.
 */ 

// Number of supported schema versions (BCH_SUPER, BCH_5, BCH_4, BCH_3)
const DATALAYER_SCHEMA_COUNT = 4;

// Initialize ECC engines for all supported versions
let eccengine = [];
for (let version = 0; version < DATALAYER_SCHEMA_COUNT; version++) {
  eccengine.push(DataLayer_GetECCEngine(version));
}

/**
 * Decodes the watermark data using the given ECC engine and schema.
 * Attempts fallback decoding with alternate schemas if the primary attempt fails.
 *
 * @param {Array<boolean>} watermarkbool - The boolean array representing the watermark.
 * @param {Array<Object>} eccengine - Array of ECC engines for decoding.
 * @returns {Object} Decoded watermark data with schema and soft binding info.
 *   softBindingInfo is null when no schema yields a valid watermark.
 */
function DataLayer_Decode(watermarkbool, eccengine, variant) {
  const hinted = DataLayer_GetVersion(watermarkbool);

  // Try the schema signalled by the version bits first. Those bits are not
  // protected by the ECC, so fall back to every other schema if they are corrupt.
  const candidates = [hinted];
  for (let alt = 0; alt < DATALAYER_SCHEMA_COUNT; alt++) {
    if (alt !== hinted) candidates.push(alt);
  }

  let primary = null;
  for (const version of candidates) {
    const databits = DataLayer_GetSchemaDataBits(version);
    const data = watermarkbool.slice(0, databits);
    const ecc = watermarkbool.slice(databits, 96);

    const dataobj = BCH_Decode(eccengine[version], data, ecc);
    dataobj.schema = DataLayer_GetSchemaName(version);
    if (dataobj.valid) {
      // Use the schema that actually recovered the data, not the hinted one
      dataobj.softBindingInfo = formatSoftBindingData(dataobj.data_binary, version, variant);
      return dataobj;
    }
    if (primary === null) primary = dataobj;
  }

  // No schema recovered the data, so there is no soft binding to report
  primary.softBindingInfo = null;
  return primary;
}

/**
 * Builds the 100-bit watermark packet for a binary payload: the data bits, the BCH ECC bits
 * and 4 version bits ('00' followed by the 2-bit schema). Matches DataLayer.process_encode in
 * the Python implementation bit for bit.
 *
 * @param {string} payload - String of '0' and '1' characters, at most the schema capacity
 *   (BCH_SUPER 40, BCH_5 61, BCH_4 68, BCH_3 75 bits). Shorter payloads are padded with zeros.
 * @param {number} version - The schema version (0 BCH_SUPER, 1 BCH_5, 2 BCH_4, 3 BCH_3).
 * @param {Array<Object>} eccengine - Array of ECC engines indexed by schema version.
 * @returns {Array<number>} The 100 packet bits as 0/1 numbers.
 */
function DataLayer_Encode(payload, version, eccengine) {
  if (!Number.isInteger(version) || version < 0 || version >= DATALAYER_SCHEMA_COUNT) {
    throw new Error(`Invalid schema version ${version}`);
  }
  if (typeof payload !== 'string' || !/^[01]*$/.test(payload)) {
    throw new Error("The payload must be a string of '0' and '1' characters");
  }
  const databits = DataLayer_GetSchemaDataBits(version);
  if (payload.length > databits) {
    throw new Error(`The payload has ${payload.length} bits but ${DataLayer_GetSchemaName(version)} allows at most ${databits} bits`);
  }

  const data = Array.from(payload.padEnd(databits, '0'), (bit) => bit === '1');
  const eccbits = 96 - databits;
  const ecc = [];
  for (const byte of BCH_Encode(eccengine[version], data)) {
    for (let shift = 7; shift >= 0; shift--) ecc.push((byte >> shift) & 1);
  }
  if (ecc.length < eccbits) throw new Error(`The ECC engine returned ${ecc.length} bits, expected ${eccbits}`);

  return [...data.map(Number), ...ecc.slice(0, eccbits), 0, 0, (version >> 1) & 1, version & 1];
}

/**
 * Interprets the watermark data in the context of C2PA.
 *
 * @param {Object} dataobj - The decoded watermark data object.
 * @param {number} version - The version of the schema.
 * @returns {Promise<Object>} Promise resolving with the updated data object.
 */
function interpret_C2PA(dataobj, version) {
  return new Promise((resolve, reject) => {
    if (true) { // Placeholder for schema-specific logic
      fetchSoftBindingInfo(dataobj.data_binary)
        .then(softBindingInfo => {
          if (softBindingInfo) {
            dataobj.softBindingInfo = softBindingInfo;
          } else {
            console.warn("No soft binding info found.");
          }
          resolve(dataobj);
        })
        .catch(error => {
          console.error("Error fetching soft binding info:", error);
          reject(error);
        });
    } else {
      resolve(dataobj);
    }
  });
}

/**
 * Extracts the schema version from the last two bits of the watermark boolean array.
 *
 * @param {Array<boolean>} watermarkbool - The boolean array representing the watermark.
 * @returns {number} The schema version as an integer.
 */
function DataLayer_GetVersion(watermarkbool) {
  watermarkbool = watermarkbool.slice(-2);
  return watermarkbool[0] * 2 + watermarkbool[1];
}

/**
 * Retrieves the ECC engine for the given schema version.
 *
 * @param {number} version - The schema version.
 * @returns {Object} The corresponding ECC engine.
 */
function DataLayer_GetECCEngine(version) {
  switch (version) {
    case 0:
      return BCH(8, 137);
    case 1:
      return BCH(5, 137);
    case 2:
      return BCH(4, 137);
    case 3:
      return BCH(3, 137);
    default:
      return -1;
  }
}

/**
 * Retrieves the number of data bits for the given schema version.
 *
 * @param {number} version - The schema version.
 * @returns {number} The number of data bits.
 */
function DataLayer_GetSchemaDataBits(version) {
  switch (version) {
    case 0:
      return 40;
    case 1:
      return 61;
    case 2:
      return 68;
    case 3:
      return 75;
    default:
      console.error("Invalid schema version");
      return 0;
  }
}

/**
 * Retrieves the name of the schema for the given version.
 *
 * @param {number} version - The schema version.
 * @returns {string} The schema name.
 */
function DataLayer_GetSchemaName(version) {
  switch (version) {
    case 0:
      return "BCH_SUPER";
    case 1:
      return "BCH_5";
    case 2:
      return "BCH_4";
    case 3:
      return "BCH_3";
    default:
      return "Invalid";
  }
}

/**
 * Formats the encoded watermark data into a structured JSON object.
 *
 * @param {Array<boolean>} encodedData - The binary data representing the watermark.
 * @param {number} version - The schema version.
 * @returns {Object|null} Formatted JSON object or null in case of errors.
 */
function formatSoftBindingData(encodedData, version, variant) {
  try {
    const binaryString = Array.isArray(encodedData)
      ? encodedData.join('')
      : String(encodedData);

    return {
      "c2pa.soft-binding": {
        "alg": `com.adobe.trustmark.${variant}`,
        "blocks": [
          {
            "scope": {},
            "value": `${version}*${binaryString}`
          }
        ]
      }
    };
  } catch (error) {
    console.error("Error formatting soft binding data:", error);
    return null;
  }
}

