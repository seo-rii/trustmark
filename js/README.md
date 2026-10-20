# TrustMark JavaScript example

The [`js`](https://github.com/adobe/trustmark/tree/main/js) directory contains an example JavaScript implementation of decoding TrustMark watermarks embedded in images. It provides a minimal example of a client-side JavaScript application, would could be applied, for example, to a browser extension.

## Overview

The example consists of key modules that handle image preprocessing, watermark detection, and data decoding. It supports the `Q` variant of the TrustMark watermarking schema.

NOTE: The TrustMark JavaScript implementation only decodes watermarked images, in contrast to the Python implementation, which can both encode and decode.

## Components

This example consists of a simple HTML file, `index.html` that loads two JavaScript files:

- [`tm_watermark.js`](https://github.com/adobe/trustmark/blob/main/js/tm_watermark.js), the core module that handles watermark detection and decoding and defines key functions for processing images and extracting watermark data. The `modelConfigs` array specifies parameters such as the TrustMark model variant. As provided, the code checks for both Q and P variants.
- [`tm_datalayer.js`](https://github.com/adobe/trustmark/blob/main/js/tm_datalayer.js) handles data decoding and schema-specific processing.  It also implements error correction and interpretation of binary watermark data.

If GPU compute is available (if you're using Google Chrome, check `chrome://gpu`), then the code will automatically use WebGPU to process the ONNX models.  If you use WebGPU it will only run in a secure context, which means on localhost or an HTTPS link.  You can start a local HTTPS server by running the `server.py` script and a suitable OpenSSL certificate in `server.pem`.  If a WebGPU session cannot be created for a model, that model falls back to WebAssembly (WASM) on the CPU, and the fallback is recorded in its load state.

## Key parameters

The desired TrustMark watermark variants for decoding are listed in the `modelConfigs` array at the top of `tm_watermark.js` for example, B, C, Q and P variants:

```js
const modelConfigs = [
  { variantcode: 'Q', fname: 'decoder_Q', sessionVar: 'session_wmarkQ', resolution: 256, squarecrop: false }, 
  { variantcode: 'P', fname: 'decoder_P', sessionVar: 'session_wmarkP', resolution: 224, squarecrop: true },
];
```

## Model loading and results

`tm_watermark.js` starts downloading the models as soon as it loads. The loading lifecycle is exposed through these globals:

- `trustmarkReady`: a promise that resolves once every model has either loaded or failed to load. It never rejects; it resolves with the same value as `trustmarkStatus()`.
- `trustmarkStatus()`: returns `{ loaded, models }`. `loaded` is `true` once loading has finished. `models` maps each model name (`resizer`, `decoder_Q`, `decoder_P`) to `{ status, backend, fallbackFrom, fallbackReason, error, loadMs }`, where `status` is `pending`, `loading`, `ready` or `error`, and `backend` is the execution provider the session runs on (`webgpu` or `wasm`). When WebGPU fails and the model falls back to WASM, `fallbackFrom` is `webgpu` and `fallbackReason` holds the WebGPU error.
- `trustmarkInit(options)`: starts loading (once) and returns a promise for the status. To pass options, set `TRUSTMARK_AUTO_INIT = false` before loading `tm_watermark.js`, then call it yourself. The options are `modelBaseUrl` (where the `.onnx` files are fetched from) and `executionProviders` (the decoder backends to try in order, by default `['webgpu', 'wasm']`). The resizer always runs on WASM, because WebGPU does not support antialiased resizing.

  ```html
  <script>var TRUSTMARK_AUTO_INIT = false;</script>
  <script src="tm_watermark.js"></script>
  <script>trustmarkInit({ executionProviders: ['wasm'] });</script>
  ```

`runwmark(imageUrl)` waits for the models to load, so it can be called right away. It resolves (it does not throw) with:

| Field | Description |
| --- | --- |
| `status` | `detected`, `not_detected` or `error`. `error` means that at least one decoder could not check the image (for example its model failed to load, inference failed, or the image could not be read) and no decoder found a watermark, so the image may still contain one. |
| `watermark_present` | `true` only when `status` is `detected`. |
| `watermark`, `schema`, `c2padata` | The payload bits, schema name and C2PA soft binding of the detected watermark, otherwise `null`. |
| `variant`, `backend` | The variant code and execution backend of the decoder that found the watermark, otherwise `null`. |
| `error` | A description of the failures when `status` is `error`, otherwise `null`. |
| `decoders` | One `{ variant, status, backend, error }` entry per decoder. |

Inference runs through a FIFO queue so that only one ONNX session runs at a time, because WebGPU fails silently and intermittently when several inferences run concurrently.

## Run the example

To run the example:

1. Start a local web server; for example, using Python: 
    ```
    python -m http.server 8000
    ```
1. Open `index.html` in a browser to run the example.
1. Drag and drop images (for example from the provided [`images`](https://github.com/adobe/trustmark/tree/main/images) directory) onto the indicated area in the web page, which will display information if the image contains a TrustMark watermark (see example below).

To use the code in your own project, simply include the two JavaScript files as usual.

## Run the tests

The tests use the built-in Node.js test runner and need neither a browser nor the models: the data layer tests use packets produced by the Python implementation, and the `tm_watermark.js` tests replace ONNX Runtime, `fetch` and the DOM with fakes to check model loading, backend fallback, the inference queue and the result statuses. From the repository root:

```
node --test js/tests/*.test.mjs
```

## Example output

For an image containing a TrustMark watermark:

```
[4:12:48 PM] Decoding watermark...
[4:12:48 PM] Watermark Found (BCH_5):
1101101111100111100111100101110001111100101100101111010000000
C2PA Assertion:
{
  "c2pa.soft-binding": {
    "alg": "com.adobe.trustmark.Q",
    "blocks": [
      {
        "scope": {},
        "value": "1*1101101111100111100111100101110001111100101100101111010000000"
      }
    ]
  }
}
```

