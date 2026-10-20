# TrustMark JavaScript example

The [`js`](https://github.com/adobe/trustmark/tree/main/js) directory contains an example JavaScript implementation of decoding TrustMark watermarks embedded in images, and of embedding variant Q watermarks. It provides a minimal example of a client-side JavaScript application, would could be applied, for example, to a browser extension.

## Overview

The example consists of key modules that handle image preprocessing, watermark detection, and data decoding. It supports the `Q` variant of the TrustMark watermarking schema.

NOTE: The TrustMark JavaScript implementation decodes Q and P watermarks, but only embeds variant Q watermarks with the `BCH_5` schema (61-bit payloads). The Python implementation supports every variant and schema for both encoding and decoding.

## Components

This example consists of a simple HTML file, `index.html` that loads three JavaScript files:

- [`tm_watermark.js`](https://github.com/adobe/trustmark/blob/main/js/tm_watermark.js), the core module that handles watermark detection and decoding and defines key functions for processing images and extracting watermark data. The `modelConfigs` array specifies parameters such as the TrustMark model variant. As provided, the code checks for both Q and P variants.
- [`tm_datalayer.js`](https://github.com/adobe/trustmark/blob/main/js/tm_datalayer.js) handles data decoding and schema-specific processing.  It also implements error correction and interpretation of binary watermark data.
- [`tm_core.js`](https://github.com/adobe/trustmark/blob/main/js/tm_core.js) holds the image processing shared by the watermark models (centre-square cropping and antialiased resizing) and the watermark embedding math. It has no DOM dependency: ONNX Runtime and the sessions are passed in, and images are `{ data, width, height }` RGBA objects. It defines the global `TrustmarkCore` when loaded as a classic script, and is a CommonJS module in Node.js (`require('./tm_core.js')`).

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

## Embed a watermark

`trustmarkEncode(image, payload, options)` embeds a variant Q watermark with the `BCH_5` schema, following `TrustMark.encode` in the Python implementation:

1. `DataLayer_Encode(payload, 1, eccengine)` builds the 100-bit packet: the payload, padded with zeros to 61 bits, then 35 BCH error correction bits and the 4 version bits `0001`. The payload is a string of `0` and `1` characters, never a number. Payloads that are longer than 61 bits or contain other characters are rejected.
1. Images with an aspect ratio above 2:1 use their centred square; other images are used whole. That region is downscaled to 256x256 with the antialiased `resizer.onnx` model and scaled to [-1, 1].
1. `encoder_Q.onnx` produces the watermarked 256x256 image. The residual (output minus input, with the per-channel mean removed) is upscaled bilinearly to the region size, multiplied by `options.strength` (default 1.0, like Python), added to the original pixels and pasted back with feathered edges. The alpha channel is kept.

```js
const result = await trustmarkEncode(file, '1011001110001111000011111000001111110000001111111000000011111');
// result.image is { data, width, height } (RGBA); result.payload, result.packet, result.schema,
// result.variant and result.backend describe what was embedded and where it ran
const png = await imageToPngBlob(result.image);
```

`image` can be an image URL, a `File`/`Blob` or an `ImageData`. `encoder_Q.onnx` (17 MB) is downloaded on first use, after the models loaded by `trustmarkInit()`, with the same WebGPU to WASM fallback; its load state appears in `trustmarkStatus().models.encoder_Q`. `trustmarkEncode` rejects with an error rather than returning an image without a watermark if a model is unavailable or the input is invalid.

The demo page has an "Embed a watermark" section where you can choose an image and a payload, and download the watermarked PNG.

The pixel processing lives in `tm_core.js` so that it can also run in Node.js, without a DOM:

```js
const ort = require('onnxruntime-node'); // or the bundled deps/ort.webgpu.min.js on WASM
const TrustmarkCore = require('./js/tm_core.js');
const runtime = { ort, resizer: resizerSession };
const watermarked = await TrustmarkCore.encodeWatermark(runtime, encoderSession, { data, width, height }, packet);
const bits = await TrustmarkCore.decodeBits(runtime, decoderSession, watermarked, 256);
```

`tm_datalayer.js` (`DataLayer_Encode`, `DataLayer_Decode`) relies on the globals of `deps/bch_ecc.min.js`, so in Node.js load both into a `vm` context as the tests do.

Limitations:

- Only variant Q with the `BCH_5` schema is supported for embedding. `DataLayer_Encode` matches the Python packets for all four schemas, but variants B, C and P (P needs its own crop and strength) are not implemented yet.
- The 256x256 downscale uses `resizer.onnx` (antialiased bicubic) instead of the antialiased bilinear resize of Pillow that Python uses. Watermarks embedded by either implementation decode in both, but the JavaScript output differs slightly from the Python output and has a PSNR about 0.4 dB lower.
- Pixel values are truncated to integers, as the Python implementation does.
- In the browser, images go through a canvas: colour profiles are applied when the image is read, the canvas stores premultiplied alpha (so fully transparent pixels lose their colour, and semi-transparent pixels can shift slightly), and the downloaded PNG is RGBA without the original metadata. `tm_core.js` itself works on the exact RGBA values it is given.

## Run the example

To run the example:

1. Start a local web server; for example, using Python: 
    ```
    python -m http.server 8000
    ```
1. Open `index.html` in a browser to run the example.
1. Drag and drop images (for example from the provided [`images`](https://github.com/adobe/trustmark/tree/main/images) directory) onto the indicated area in the web page, which will display information if the image contains a TrustMark watermark (see example below).

To use the code in your own project, include the JavaScript files in the same order as `index.html` (`deps/ort.webgpu.min.js`, `deps/bch_ecc.min.js`, `tm_datalayer.js`, `tm_core.js`, `tm_watermark.js`).

## Run the tests

The tests use the built-in Node.js test runner and need neither a browser nor, by default, the models: the data layer tests use packets produced by the Python implementation, the `tm_core.js` tests compare the resizing and residual pasting with PyTorch and Python results, and the `tm_watermark.js` tests replace ONNX Runtime, `fetch` and the DOM with fakes to check model loading, backend fallback, the inference queue, the result statuses and encoder failures. From the repository root:

```
node --test js/tests/*.test.mjs
```

`tests/tm_encoder_models.test.mjs` embeds watermarks with the real models and checks that the Q decoder recovers the exact payload. It runs the bundled ONNX Runtime Web on WASM, so it needs no GPU. The models are not downloaded by the tests: put `encoder_Q.onnx`, `decoder_Q.onnx` and `resizer.onnx` from `https://cai-watermark.adobe.net/watermarking/trustmark-models/` in a directory and pass it in. Without `TRUSTMARK_MODEL_DIR` these tests are reported as skipped.

```
TRUSTMARK_MODEL_DIR=/path/to/models node --test js/tests/*.test.mjs
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

