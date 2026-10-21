# TrustMark JavaScript example

The [`js`](https://github.com/adobe/trustmark/tree/main/js) directory contains an example JavaScript implementation of decoding TrustMark watermarks embedded in images, and of embedding them. It provides a minimal example of a client-side JavaScript application, would could be applied, for example, to a browser extension.

## Overview

The example consists of key modules that handle image preprocessing, watermark detection, and data decoding and encoding. It supports every TrustMark model variant and every error correction schema:

| Variant | Embed | Decode | Encoder model | Decoder model | Model input (encode / decode) | Processed region | Strength |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Q` | yes | yes, loaded by default | `encoder_Q.onnx`, 17 MB | `decoder_Q.onnx`, 47 MB | 256 / 256 | whole image, centre square above 2:1 | x1.0 |
| `P` | yes | yes, loaded by default | `encoder_P.onnx`, 17 MB | `decoder_P.onnx`, 47 MB | 256 / 224 | always the centre square | x1.25 |
| `B` | yes | yes, on request | `encoder_B.onnx`, 17 MB | `decoder_B.onnx`, 47 MB | 256 / 256 | whole image, centre square above 2:1 | x1.0 |
| `C` | yes | yes, on request | `encoder_C.onnx`, 17 MB | `decoder_C.onnx`, 22 MB | 256 / 256 | whole image, centre square above 2:1 | x1.0 |

| Schema | Number | Payload bits |
| --- | --- | --- |
| `BCH_SUPER` | 0 | 40 |
| `BCH_5` (default) | 1 | 61 |
| `BCH_4` | 2 | 68 |
| `BCH_3` | 3 | 75 |

Every variant can embed with every schema. Decoding tries every schema, so the schema does not need to be known. The processed region, the strength multiplier and the model input sizes follow the Python implementation (`TrustMark.__init__` and `TrustMark.encode`), except that the ONNX decoders of B, C and Q take 256x256 inputs where the Python decoders run at 245x245. The settings live in one table, `TrustmarkCore.VARIANTS`, used by both embedding and decoding.

## Components

This example consists of a simple HTML file, `index.html` that loads three JavaScript files:

- [`tm_watermark.js`](https://github.com/adobe/trustmark/blob/main/js/tm_watermark.js), the browser module that loads the models and defines the functions to decode (`runwmark`) and embed (`trustmarkEncode`) watermarks. By default it checks images for Q and P watermarks.
- [`tm_datalayer.js`](https://github.com/adobe/trustmark/blob/main/js/tm_datalayer.js) handles data decoding and schema-specific processing.  It also implements error correction and interpretation of binary watermark data.
- [`tm_core.js`](https://github.com/adobe/trustmark/blob/main/js/tm_core.js) holds the per-variant settings (`VARIANTS`), the image processing shared by the watermark models (centre-square cropping and antialiased resizing) and the watermark embedding math. It has no DOM dependency: ONNX Runtime and the sessions are passed in, and images are `{ data, width, height }` RGBA objects. It defines the global `TrustmarkCore` when loaded as a classic script, and is a CommonJS module in Node.js (`require('./tm_core.js')`).

If GPU compute is available (if you're using Google Chrome, check `chrome://gpu`), then the code will automatically use WebGPU to process the ONNX models.  If you use WebGPU it will only run in a secure context, which means on localhost or an HTTPS link.  You can start a local HTTPS server by running the `server.py` script and a suitable OpenSSL certificate in `server.pem`.  If a WebGPU session cannot be created for a model, that model falls back to WebAssembly (WASM) on the CPU, and the fallback is recorded in its load state.

## Key parameters

The variants whose decoders are loaded, in the order `runwmark` tries them, are set with the `decoders` option of `trustmarkInit` (see below). The default, `['Q', 'P']`, downloads 95 MB of decoders; all four variants (`['Q', 'P', 'B', 'C']`) download 165 MB. An empty list loads no decoder (nor the resizer), for pages that only embed watermarks. Encoders are downloaded separately, on first use.

```html
<script>var TRUSTMARK_AUTO_INIT = false;</script>
<script src="tm_watermark.js"></script>
<script>trustmarkInit({ decoders: ['Q', 'P', 'B', 'C'] });</script>
```

The demo page takes the list from its URL, for example `index.html?decoders=Q,P,B,C`.

## Model loading and results

`tm_watermark.js` starts downloading the models as soon as it loads. The loading lifecycle is exposed through these globals:

- `trustmarkReady`: a promise that resolves once every model has either loaded or failed to load. It never rejects; it resolves with the same value as `trustmarkStatus()`.
- `trustmarkStatus()`: returns `{ loaded, models }`. `loaded` is `true` once loading has finished. `models` maps each model name (`resizer`, `decoder_Q`, `decoder_P`, and `encoder_Q` and so on once an encoder is loaded) to `{ status, backend, fallbackFrom, fallbackReason, error, loadMs }`, where `status` is `pending`, `loading`, `ready` or `error`, and `backend` is the execution provider the session runs on (`webgpu` or `wasm`). When WebGPU fails and the model falls back to WASM, `fallbackFrom` is `webgpu` and `fallbackReason` holds the WebGPU error.
- `trustmarkInit(options)`: starts loading (once) and returns a promise for the status. To pass options, set `TRUSTMARK_AUTO_INIT = false` before loading `tm_watermark.js`, then call it yourself. The options are `modelBaseUrl` (where the `.onnx` files are fetched from), `executionProviders` (the decoder and encoder backends to try in order, by default `['webgpu', 'wasm']`) and `decoders` (the variants to decode, by default `['Q', 'P']`; an unknown variant throws an error and loads nothing). The resizer always runs on WASM, because WebGPU does not support antialiased resizing.

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

`trustmarkEncode(image, payload, options)` embeds a watermark, following `TrustMark.encode` in the Python implementation:

1. `DataLayer_Encode(payload, schema, eccengine)` builds the 100-bit packet: the payload, padded with zeros to the capacity of the schema, then the BCH error correction bits and the 4 version bits (`00` and the 2-bit schema number). The payload is a string of `0` and `1` characters, never a number. Payloads that are longer than the schema allows or contain other characters are rejected.
1. The processed region of the variant (see the table above) is downscaled to 256x256 with the antialiased bilinear filter of Pillow's `Image.resize`, reproduced bit for bit in JavaScript (`TrustmarkCore.pillowBilinearResize`), and scaled to [-1, 1]. Embedding does not use `resizer.onnx`.
1. The encoder of the variant produces the watermarked 256x256 image. The residual (output minus input, with the per-channel mean removed) is upscaled bilinearly to the region size, multiplied by `options.strength` (default 1.0, like Python) times the variant's multiplier (1.25 for P), added to the original pixels and pasted back with feathered edges. The alpha channel is kept.

| Option | Default | Values |
| --- | --- | --- |
| `variant` | `'Q'` | `'Q'`, `'P'`, `'B'` or `'C'` |
| `schema` | `'BCH_5'` | `'BCH_SUPER'`, `'BCH_5'`, `'BCH_4'`, `'BCH_3'`, or their numbers 0 to 3 (`TrustMark.Encoding` in Python) |
| `strength` | `1.0` | a non-negative number (`WM_STRENGTH` in Python) |

```js
const result = await trustmarkEncode(file, '1011001110001111000011111000001111110000001111111000000011111');
const strong = await trustmarkEncode(file, '0110100101011010011010010101101001101001', { variant: 'P', schema: 'BCH_SUPER' });
// result.image is { data, width, height } (RGBA); result.payload, result.packet, result.schema,
// result.variant and result.backend describe what was embedded and where it ran
const png = await imageToPngBlob(result.image);
```

`image` can be an image URL, a `File`/`Blob` or an `ImageData`. The encoder of each variant (17 MB) is downloaded the first time that variant is used, after the models loaded by `trustmarkInit()`, with the same WebGPU to WASM fallback; its load state appears in `trustmarkStatus().models`, for example `encoder_P`. `trustmarkEncode` rejects with an error rather than returning an image without a watermark if a model is unavailable or the input is invalid; an unknown variant or schema, an invalid strength and a payload that does not fit the schema are rejected before anything is downloaded. To read the watermark back with `runwmark`, the decoder of the same variant must be loaded (`trustmarkInit({ decoders })`).

The demo page has an "Embed a watermark" section where you can choose an image, the variant, the schema and the payload, and download the watermarked PNG.

The pixel processing lives in `tm_core.js` so that it can also run in Node.js, without a DOM:

```js
const ort = require('onnxruntime-node'); // or the bundled deps/ort.webgpu.min.js on WASM
const TrustmarkCore = require('./js/tm_core.js');
const runtime = { ort, resizer: resizerSession }; // the resizer is only used for decoding
const watermarked = await TrustmarkCore.encodeWatermark(runtime, encoderPSession, { data, width, height }, packet, { variant: 'P' });
const bits = await TrustmarkCore.decodeBits(runtime, decoderPSession, watermarked, 'P');
```

`tm_datalayer.js` (`DataLayer_Encode`, `DataLayer_Decode`, `DataLayer_GetSchemaVersion`) relies on the globals of `deps/bch_ecc.min.js`, so in Node.js load both into a `vm` context as the tests do.

Agreement with the Python implementation (`TrustMark.encode` with PyTorch on the CPU and Pillow 12.3) for the four variants and four schemas on `images/ghost.png` (RGBA), `ufo_240.jpg`, `ripley.jpg` and 4:1, 1:4 and odd-sized crops of `ripley.jpg` (96 combinations): the JavaScript output differs from the Python output by at most 1 per channel, its PSNR is within 0.06 dB of Python's, the alpha channel is unchanged, and both the Python implementation and the JavaScript decoders recover the exact payload and schema from every JavaScript output; the JavaScript decoders also recover them from every Python output. The remaining differences come from the ONNX encoders and float rounding.

Limitations:

- Pixel values are truncated to integers, as the Python implementation does.
- Like the Python implementation, small regions and payloads that are mostly zeros can be hard to decode, especially with variant P and the `BCH_4` and `BCH_3` schemas, which correct fewer errors.
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

The tests use the built-in Node.js test runner and need neither a browser nor, by default, the models: the data layer tests use packets produced by the Python implementation, the `tm_core.js` tests compare the resizing with Pillow and PyTorch results and the residual pasting with Python results, and the `tm_watermark.js` tests replace ONNX Runtime, `fetch` and the DOM with fakes to check model loading, the decoder selection, backend fallback, the inference queue, the result statuses, and encoder selection and failures. From the repository root:

```
node --test js/tests/*.test.mjs
```

`tests/tm_encoder_models.test.mjs` embeds watermarks of every variant and schema into crops of `images/ghost.png` with the real models and checks that the decoder of the same variant recovers the exact payload. It runs the bundled ONNX Runtime Web on WASM, so it needs no GPU. The models are not downloaded by the tests: put `resizer.onnx` and the `encoder_<V>.onnx` and `decoder_<V>.onnx` files of the variants to test from `https://cai-watermark.adobe.net/watermarking/trustmark-models/` in a directory and pass it in. Variants whose models are missing, and every variant without `TRUSTMARK_MODEL_DIR`, are reported as skipped.

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

