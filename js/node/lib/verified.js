'use strict';

// encodeVerified: embed, encode the final file, decode those exact bytes again, and return them
// only when the payload and schema match. Policy (who may publish what, where, and when) is
// left to the service that calls it.

const crypto = require('node:crypto');

const { TrustmarkError, checkSignal } = require('./errors');

/**
 * The final bytes did not carry the expected watermark. No bytes are attached: there is
 * nothing in this error that could be published by mistake.
 *
 * reason: 'not_detected' | 'payload_mismatch' | 'schema_mismatch' | 'decode_error' |
 *   'bytes_changed' (of the last attempt; decode_error is not retried)
 * attempts: one { strength, status, payload, schema, bitFlips, error, reason } per attempt
 */
class WatermarkVerificationError extends TrustmarkError {
  constructor(message, { reason, attempts, cause }) {
    super('VERIFICATION_FAILED', message, { cause, details: { reason, attempts } });
    this.name = 'WatermarkVerificationError';
    this.reason = reason;
    this.attempts = attempts;
  }
}

function argumentError(message) {
  return new TrustmarkError('INVALID_ARGUMENT', message);
}

function sha256(bytes) {
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

function strengths(options) {
  const first = options.strength === undefined ? 1.0 : options.strength;
  if (typeof first !== 'number' || !Number.isFinite(first) || first <= 0) throw argumentError('strength must be a positive number');
  const retries = options.retryStrengths === undefined ? [] : options.retryStrengths;
  if (!Array.isArray(retries)) throw argumentError('retryStrengths must be an array of strengths');
  let previous = first;
  for (const s of retries) {
    if (typeof s !== 'number' || !Number.isFinite(s) || s <= previous) {
      throw argumentError('retryStrengths must be finite numbers, each higher than the strength before it');
    }
    previous = s;
  }
  return [first, ...retries];
}

/**
 * Embeds `payload`, encodes the final format and quality, decodes those exact bytes with
 * `options.verifier` (by default the same instance) and returns them only if the decoder
 * recovers exactly the embedded payload and schema. Otherwise it retries with the next of
 * `options.retryStrengths` (none by default) and finally throws WatermarkVerificationError.
 * It never returns the input or bytes that failed verification.
 *
 * The verification decode uses the 'signalled' schema policy, so the version bits must name the
 * right schema, as the Python decoder requires.
 *
 * @param {object} trustmark - An instance from createTrustmark.
 * @param {*} input - As for trustmark.encode.
 * @param {string} payload - Payload bit string.
 * @param {object} [options] - format, quality, background and signal as for encode, plus
 *   strength (first attempt, default 1.0), retryStrengths (higher strengths to try next) and
 *   verifier (an object with decode(), e.g. a second instance on another execution provider).
 * @returns {Promise<object>} The encode result plus sha256 (of bytes) and verification.
 */
async function encodeVerified(trustmark, input, payload, options = {}) {
  if (!trustmark || typeof trustmark.encode !== 'function' || typeof trustmark.decode !== 'function') {
    throw argumentError('encodeVerified needs a TrustMark instance from createTrustmark');
  }
  if (options === null || typeof options !== 'object') throw argumentError('encodeVerified options must be an object');
  const verifier = options.verifier === undefined ? trustmark : options.verifier;
  if (!verifier || typeof verifier.decode !== 'function') throw argumentError('verifier must have a decode() method');
  if (verifier.variant !== undefined && verifier.variant !== trustmark.variant) {
    throw argumentError(`The verifier reads variant ${verifier.variant} but the watermark is variant ${trustmark.variant}`);
  }
  const signal = checkSignal(options.signal);
  const plan = strengths(options);
  const encodeOptions = { format: options.format, quality: options.quality, background: options.background, signal };

  const attempts = [];
  for (const strength of plan) {
    const encoded = await trustmark.encode(input, payload, { ...encodeOptions, strength });
    const digest = sha256(encoded.bytes);
    const decoded = await verifier.decode(encoded.bytes, { signal, schemaPolicy: 'signalled' });
    let reason = null;
    if (decoded.status === 'error') reason = 'decode_error';
    else if (decoded.status !== 'detected') reason = 'not_detected';
    else if (decoded.schema !== encoded.schema) reason = 'schema_mismatch';
    else if (decoded.payload !== encoded.payload) reason = 'payload_mismatch';
    const attempt = {
      strength,
      status: decoded.status,
      payload: decoded.payload,
      schema: decoded.schema,
      bitFlips: decoded.bitFlips === undefined ? null : decoded.bitFlips,
      error: decoded.status === 'error' && decoded.error ? { code: decoded.error.code || null, message: String(decoded.error.message || decoded.error) } : null,
      reason,
    };
    attempts.push(attempt);

    // A decoder that cannot run will not do better at a higher strength
    if (reason === 'decode_error') {
      throw new WatermarkVerificationError(`The watermarked image could not be verified: ${attempt.error ? attempt.error.message : 'decode error'}`, { reason, attempts, cause: decoded.error });
    }
    if (reason === null) {
      // The bytes returned are the bytes that were decoded
      if (sha256(encoded.bytes) !== digest) {
        throw new WatermarkVerificationError('The encoded bytes changed during verification', { reason: 'bytes_changed', attempts });
      }
      return {
        ...encoded,
        sha256: digest,
        verification: { payload: decoded.payload, schema: decoded.schema, bitFlips: attempt.bitFlips, backend: decoded.backend === undefined ? null : decoded.backend, attempts },
      };
    }
  }

  const { reason } = attempts[attempts.length - 1];
  const tried = plan.map((s) => s.toString()).join(', ');
  throw new WatermarkVerificationError(`The watermark could not be verified in the final image (${reason}; strengths tried: ${tried})`, { reason, attempts });
}

module.exports = { encodeVerified, WatermarkVerificationError };
