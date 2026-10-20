# TrustMark benchmark (Python)

`benchmark.py` measures watermark quality, robustness and speed of the Python
implementation on a directory of images, reproducibly. For each image it:

1. embeds a random payload derived from `--seed` and the image's relative path
   (`MODE='binary'`, full schema capacity);
2. computes PSNR (and optionally SSIM) between the cover and the watermarked image;
3. applies each transform to the watermarked image and decodes the result;
4. applies each transform to the unwatermarked cover and decodes it too (negatives).

It also times model loading, the first encode/decode call and repeated warm calls.
Only the package's own dependencies (numpy, Pillow, torch) are used.

## Running

From the `python/` directory, so that the repository's `trustmark` package is the one
being measured (the report records the package path that was imported):

```sh
cd python
python -m benchmarks.benchmark --corpus ../images --exclude '*_[PQ].png' \
    --variants Q --schema BCH_5 --seed 1234 \
    --transforms none,png,jpeg:90,jpeg:75,resize:0.5,crop:0.9,rotate:90 \
    --ssim --repeats 5 --out results.json
```

`images/` also holds `*_Q.png` and `*_P.png`, which are already watermarked outputs of
the examples. Exclude them as above: as covers they would be watermarked twice, and as
negatives they would count as false positives.

Model files are loaded like `TrustMark()` does, from the package's `trustmark/models/`
directory (downloaded on first use). The remover is not loaded.

| Option | Default | Meaning |
|---|---|---|
| `--corpus DIR` | required | images under `DIR` (recursive; png, jpg, jpeg, bmp, tif, tiff, webp), sorted by relative path |
| `--exclude GLOB` | none | skip files whose relative path or name matches; repeatable |
| `--max-images N` | all | only the first `N` images |
| `--variants` | `Q` | comma-separated `C`, `Q`, `B`, `P`; one model load per variant |
| `--schema` | `BCH_5` | `BCH_SUPER`, `BCH_5`, `BCH_4` or `BCH_3` |
| `--strength` | `1.0` | `WM_STRENGTH` passed to `encode` |
| `--seed` | `0` | payload seed; also seeds `random`, numpy and torch |
| `--transforms` | see `--help` | comma-separated list, see below |
| `--rotation` | off | decode everything with `ROTATION=True` |
| `--no-negatives` | negatives on | skip decoding the unwatermarked covers |
| `--ssim` | off | also compute SSIM (slow and memory hungry on large images) |
| `--repeats N` | `5` | warm encode/decode repetitions for the timing section |
| `--timing-image PATH` | first corpus image | image used for the timing section |
| `--device` | `cpu` | torch device |
| `--threads N` | torch default | `torch.set_num_threads(N)` |
| `--out PATH` | none | write the full JSON report |

Transforms (applied to RGB images with Pillow):

| Transform | Effect |
|---|---|
| `none` | the image as returned by `encode` |
| `png` | PNG save and reload (lossless; should match `none`) |
| `jpeg:Q` | JPEG save at quality `Q` (Pillow defaults, 4:2:0 chroma subsampling) and reload |
| `resize:S` | bicubic resize by factor `S` (e.g. `0.5` downscale, `2` upscale) |
| `crop:F` | centre crop keeping fraction `F` of each side |
| `rotate:D` | counter-clockwise rotation by `D` degrees; multiples of 90 are exact and decoded with `ROTATION=True`, other angles keep the size and fill the corners with black |

## What is reported

The summary printed at the end is also in the JSON report under `runs[].summary`.

Each decode of a watermarked image has one outcome:

- `exact`: watermark detected with the embedded schema and payload;
- `false_accept`: reported as valid with the embedded schema but a different payload;
- `schema_mismatch`: reported as valid but with another schema (so also a wrong payload);
- `not_detected`: no valid watermark;
- `error`: `decode` raised (the message is recorded).

Negatives are decodes of the unwatermarked covers after the same transforms. Any
detection there is a false positive; `false_positives_by_schema` shows which schema the
decoder claimed. A rotation search tries four orientations, so it gives a negative four
chances to be accepted.

- PSNR is computed in float64 over the RGB channels at the cover's resolution; it is
  reported as `inf` for identical images.
- SSIM is the mean structural similarity over the RGB channels with a 7x7 uniform window
  and sample covariance, as `skimage.metrics.structural_similarity(cover, stego,
  data_range=255, channel_axis=-1)` computes it.
- Timings are wall-clock seconds (`time.perf_counter`): `model_load_s` is the
  `TrustMark(...)` constructor (including the model file checksum check),
  `first_encode_s`/`first_decode_s` are the first calls after loading, and
  `warm_encode`/`warm_decode` give min/mean/p50/p95/max over `--repeats` further calls on
  the timing image (p95 interpolates linearly, so it needs many repeats to mean much).
  `corpus_encode_s`/`corpus_decode_s` cover the corpus loop.

The JSON report also holds the configuration, every per-image and per-transform result
(payloads, decoded schema, bit errors for same-schema decodes, times) and the
environment: Python, torch, torchvision, numpy, Pillow and libjpeg versions, torch thread
counts, CPU count, load average at start and end, the git commit (and whether tracked
files were modified), the imported package path and the MD5 of each model file against
`MODEL_CHECKSUMS`. Infinite values are written as the string `"inf"` so the file is strict
JSON.

## Reproducibility

With the same corpus, options, model files and environment, outcomes, decoded payloads,
PSNR and SSIM are identical between runs. Results can change with the torch version or
thread count (floating point reduction order), and JPEG results with the libjpeg build,
which is why these are recorded. Timings depend on the machine and its load; compare them
only between runs on the same otherwise idle machine.
