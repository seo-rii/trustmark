# TrustMark — LLM Agent Guide

## Install

The installable Python package lives in this `python/` subdirectory, not the repo root.

```bash
# End-user (preferred)
pip install trustmark

# Development / editable
pip install -e python/   # run from repo root
# or
pip install -e .         # run from inside python/
```

## Minimal working example

```python
from trustmark import TrustMark
from PIL import Image

tm = TrustMark(model_type='Q', loadRemover=False)  # skip remover download for encode/decode only

capacity = tm.schemaCapacity()          # always query — do not hardcode a bit length
secret = ...                            # binary string of exactly `capacity` bits

cover = Image.open('input.jpg').convert('RGB')
watermarked = tm.encode(cover, secret, MODE='binary')
watermarked.save('output.png')

wm_secret, wm_present, wm_schema = tm.decode(watermarked, MODE='binary')
```

**`MODE` must match between `encode` and `decode`.** Mismatching them produces silently
wrong results — no error is raised.

Two modes are available:

- `MODE='binary'` — payload is a bit string (`'0'` and `'1'` characters) of exactly
  `schemaCapacity()` bits.
- `MODE='text'` — payload is a 7-bit ASCII string. Each character occupies 7 bits, so
  the usable capacity is `schemaCapacity() // 7` characters. Non-ASCII (Unicode, bytes)
  is not supported in this mode.

The default is `MODE='text'`. When working with raw bits or arbitrary binary data always
pass `MODE='binary'` explicitly on both calls.

Text mode example:

```python
tm = TrustMark(model_type='Q', loadRemover=False)
max_chars = tm.schemaCapacity() // 7      # e.g. 61 // 7 = 8 characters for BCH_5
secret_text = 'hello'                     # must be 7-bit ASCII, <= max_chars

watermarked = tm.encode(cover, secret_text, MODE='text')
wm_secret, wm_present, wm_schema = tm.decode(watermarked, MODE='text')
```

## Device selection

By default `TrustMark` auto-selects a device: `cuda` if available, else `mps` (Apple
Silicon GPU), else `cpu`. Override explicitly via the `device` constructor argument:

```python
tm = TrustMark(model_type='Q', device='mps')   # or 'cuda', 'cpu'
```

MPS requires `torch>=2.3.1` and `torchvision>=0.18.1` (the package's declared minimum
versions) for full operator coverage — e.g. `torchvision.ops.nms`, used by the bounding
box detector (`loadBBoxDetector=True`), is not implemented for MPS on earlier
torchvision releases and raises `NotImplementedError`.

## Model loading — what gets downloaded and when

On first use, model weights are fetched from Adobe's CDN and cached locally inside the
package directory. Four separate models exist; not all are needed for every use case:

| Model | Constructor arg | Needed for | Approx size |
|---|---|---|---|
| Encoder | `loadEncoder=True` (default) | `encode` | moderate |
| Decoder | `loadDecoder=True` (default) | `decode` (also with `DETECTFIRST=True`) | moderate |
| Remover | `loadRemover=True` (default) | `remove_watermark` | large |
| BBox detector | `loadBBoxDetector=False` (default) | `localize`, `decode(..., DETECTFIRST=True)` | large |

Only the models you construct with are loaded into memory. Calling `encode` on an
instance built with `loadEncoder=False`, or `decode` on one built with `loadDecoder=False`,
raises `ModelNotLoadedError`. The bbox detector only finds watermarked regions; the regions
are still read by the decoder, so `DETECTFIRST=True` needs both the detector and the decoder.

If the encoder or decoder files cannot be fetched, the constructor raises
`ModelNotLoadedError` (`from trustmark import ModelNotLoadedError`, a `RuntimeError`
subclass) naming the model and files, with the download error as its cause. A failed
remover or bbox detector download only prints an error at construction; calling
`remove_watermark` or `localize` afterwards raises `ModelNotLoadedError`.

### Downloading models manually

If automatic download fails (no network access at runtime, corporate proxy, air-gapped
deployment), fetch the files yourself and place them in the package's `models/`
directory — for an editable install from this repo, that's `python/trustmark/models/`.

Each `model_type` (`C`, `Q`, `B`, or `P`) needs a subset of these files, named
`{name}_{TYPE}.{ext}`:

| File pattern | Purpose | Needed for |
|---|---|---|
| `trustmark_{TYPE}.yaml` | encoder/decoder config | `loadEncoder=True` or `loadDecoder=True` (defaults) |
| `decoder_{TYPE}.ckpt` | decoder weights | `loadDecoder=True` (default) |
| `encoder_{TYPE}.ckpt` | encoder weights | `loadEncoder=True` (default) |
| `trustmark_rm_{TYPE}.yaml`, `trustmark_rm_{TYPE}.ckpt` | remover config/weights | `loadRemover=True` (default) |
| `trustmark_bbox_{TYPE}.yaml`, `trustmark_bbox_{TYPE}.ckpt` | bbox detector config/weights | `loadBBoxDetector=True`; **only published for `Q` and `P`** |

Download each file from `https://cai-watermark.adobe.net/watermarking/trustmark-models/<filename>`, e.g.:

```sh
curl -o python/trustmark/models/trustmark_Q.yaml https://cai-watermark.adobe.net/watermarking/trustmark-models/trustmark_Q.yaml
curl -o python/trustmark/models/decoder_Q.ckpt   https://cai-watermark.adobe.net/watermarking/trustmark-models/decoder_Q.ckpt
curl -o python/trustmark/models/encoder_Q.ckpt   https://cai-watermark.adobe.net/watermarking/trustmark-models/encoder_Q.ckpt
```

TrustMark verifies each file's MD5 against a hardcoded checksum (`MODEL_CHECKSUMS` in
`trustmark/trustmark.py`) before using it; a mismatch is treated as a missing file and
triggers an automatic re-download. A manually placed file must match that checksum
exactly — no partial downloads or re-hosted mirrors with different content.

**Skip the remover download** if you only need to embed or verify watermarks:

```python
tm = TrustMark(model_type='Q', loadRemover=False)
```

**Load the bbox detector** only when you need to locate a watermarked region before
decoding (e.g. a cropped or composited image):

```python
tm = TrustMark(model_type='Q', loadRemover=False, loadBBoxDetector=True)
wm_secret, wm_present, wm_schema = tm.decode(img, MODE='binary', DETECTFIRST=True)
```

**Load one network** for a service that only embeds or only verifies watermarks:

```python
embedder = TrustMark(model_type='Q', loadRemover=False, loadDecoder=False)  # encode only
verifier = TrustMark(model_type='Q', loadRemover=False, loadEncoder=False)  # decode only
```

### Long-lived services

Construct `TrustMark` once (e.g. at process start-up) and reuse the instance for every
request. The constructor verifies the model files' MD5 checksums and builds the networks,
which takes seconds and hundreds of MB; `encode` and `decode` do not modify the instance.

```python
from trustmark import TrustMark

VERIFIER = TrustMark(model_type='Q', loadRemover=False, loadEncoder=False)  # once per process

def verify(image):  # called per request
    return VERIFIER.decode(image.convert('RGB'), MODE='binary')
```

Do not create a new `TrustMark` inside the request handler.

## Model variants

| `model_type` | Character | Trade-off |
|---|---|---|
| `'Q'` | Balanced (default) | Good quality and robustness |
| `'P'` | High visual quality | Best PSNR, forces centre-square crop |
| `'B'` | Base | Original paper model |
| `'C'` | Compact decoder | Smaller decoder model |

## ECC encoding types and bit capacity

The `encoding_type` parameter controls error-correction strength and usable payload size.
Pass via `TrustMark.Encoding.*`:

| Encoding | Constant | Payload bits | ECC strength |
|---|---|---|---|
| BCH_5 | `Encoding.BCH_5` | 61 bits | strongest (default) |
| BCH_4 | `Encoding.BCH_4` | 68 bits | moderate |
| BCH_3 | `Encoding.BCH_3` | 75 bits | light |
| BCH_SUPER | `Encoding.BCH_SUPER` | 40 bits | maximum robustness |

Always call `tm.schemaCapacity()` after construction to get the exact capacity for the
chosen encoding — do not hardcode the number.

## decode options

```python
tm.decode(
    img,
    MODE='binary',        # 'binary' (bit string) or 'text' (7-bit ASCII string)
    DETECTFIRST=False,    # True = run bbox detector first (requires loadBBoxDetector=True)
    ROTATION=False,       # True = try 0/90/180/270° rotations
)
# returns: (secret_string, wm_present: bool, wm_schema: int)
# wm_present=False → no watermark detected; secret_string will be ''
```

## encode options

```python
tm.encode(
    cover_image,          # PIL image, any resolution
    string_secret,        # bit string (MODE='binary') or 7-bit ASCII string (MODE='text')
    MODE='binary',
    WM_STRENGTH=1.0,      # increase for stronger watermark, at cost of visual quality
    WM_MERGE='bilinear',  # interpolation mode when upscaling residual back to original res
)
# returns: PIL image (RGB), same resolution as input
```

## Watermark removal

Requires `loadRemover=True` (the default) at construction time. Calling
`remove_watermark` without the remover model loaded raises `ModelNotLoadedError`.

```python
tm = TrustMark(model_type='Q')   # loadRemover=True is the default
recovered = tm.remove_watermark(watermarked_image)
```
