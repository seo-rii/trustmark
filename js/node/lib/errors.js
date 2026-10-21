'use strict';

/**
 * Error raised by the TrustMark SDK. `code` tells the failures apart, so callers never have to
 * parse messages; in particular a missing or corrupt model is never reported as "no watermark".
 *
 * Codes:
 *   INVALID_ARGUMENT       an option, payload or strength is invalid
 *   UNSUPPORTED            the variant, schema or output format is not supported
 *   INVALID_IMAGE          the input could not be read as an image
 *   IMAGE_TOO_LARGE        the image has more pixels than maxPixels
 *   MODEL_NOT_FOUND        a model file is missing and downloads are disabled
 *   MODEL_DOWNLOAD_FAILED  a model could not be downloaded
 *   MODEL_INTEGRITY        a model file is too large or does not match its expected digest
 *   MODEL_LOAD_FAILED      ONNX Runtime could not create a session on any execution provider
 *   INFERENCE_FAILED       a model failed while running
 *   QUEUE_FULL             too many jobs are waiting (maxQueue)
 *   CLOSED                 the instance was closed
 */
class TrustmarkError extends Error {
  constructor(code, message, options = {}) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    this.name = 'TrustmarkError';
    this.code = code;
    if (options.details !== undefined) this.details = options.details;
  }
}

/** Throws the abort reason of `signal` if it has been aborted (a DOMException by default). */
function throwIfAborted(signal) {
  if (signal && signal.aborted) {
    throw signal.reason !== undefined ? signal.reason : new DOMException('This operation was aborted', 'AbortError');
  }
}

/**
 * Resolves or rejects like `promise`, but rejects with the abort reason as soon as `signal`
 * aborts. The underlying work is not cancelled; its result is ignored.
 */
function raceAbort(promise, signal) {
  if (!signal) return promise;
  throwIfAborted(signal);
  return new Promise((resolve, reject) => {
    const onAbort = () => {
      try { throwIfAborted(signal); } catch (error) { reject(error); }
    };
    signal.addEventListener('abort', onAbort, { once: true });
    promise.then(
      (value) => { signal.removeEventListener('abort', onAbort); resolve(value); },
      (error) => { signal.removeEventListener('abort', onAbort); reject(error); },
    );
  });
}

function checkSignal(signal) {
  if (signal !== undefined && signal !== null && !(typeof signal === 'object' && typeof signal.aborted === 'boolean' && typeof signal.addEventListener === 'function')) {
    throw new TrustmarkError('INVALID_ARGUMENT', 'signal must be an AbortSignal');
  }
  return signal || undefined;
}

module.exports = { TrustmarkError, throwIfAborted, raceAbort, checkSignal };
