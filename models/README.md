# TrustMark model manifest

[`manifest.json`](manifest.json) lists every published TrustMark model artifact,
records which model family and variant each one belongs to, and states which
encoders and decoders interoperate. This directory holds metadata only. The model
files themselves are downloaded from
`https://cai-watermark.adobe.net/watermarking/trustmark-models/<name>` and are not
committed.

The manifest sits at the repository root rather than in one port because the
same files are shared: `python/` loads the PyTorch checkpoints, while `rust/`,
`cpp/`, `js/` and `python-hailo/` load the ONNX files.

> **Digest status:** the sizes, MD5 and SHA-256 digests were computed on
> 2026-10-20 by downloading each file from the CDN (or reading it from the
> repository) and hashing the received bytes. They have not yet been confirmed
> by the maintainers who published the files; see
> [Confirming the digests](#confirming-the-digests).

## Model families

A model family is a lineage of trained weights. Watermarks embedded by an encoder
decode only with a decoder of the **same family and the same variant**.

| Family | Artifacts | Used by |
|---|---|---|
| `trustmark` | PyTorch checkpoints and configs, and their ONNX exports, for variants B, C, P and Q | `python/` (PyTorch); `rust/`, `cpp/`, `js/` and the `'cpu'` backends of `python-hailo/` (ONNX) |
| `trustmark-hailo` | Hailo-8L builds: `*.hef` and the `*.npz` pieces run on the CPU | `python-hailo/` with the `'npu'` backends and its bounding-box detector |

The family ids reuse existing names rather than new ones: `trustmark` is the name of
the project, of the PyPI package built from `python/`, of the `trustmark-models/` CDN
directory and the prefix of the registered C2PA algorithm ids; `trustmark-hailo` is
the PyPI package built from `python-hailo/`. They do not use a year because
variants C and P were published after the 2023 paper.

Within the `trustmark` family the PyTorch checkpoint and the ONNX export of a
variant are interchangeable, given the preprocessing described below. On
2026-10-20 a BCH_5 payload was recovered in every combination of PyTorch or ONNX
encoder and PyTorch or ONNX decoder, for B, C, P and Q, on
`images/ghost.png`, `images/ripley.jpg` and `images/ufo_240.jpg`.

### The Hailo models are a different family

[`python-hailo/CONFIG.md`](../python-hailo/CONFIG.md) states that images encoded
by the PyTorch port cannot be decoded by `python-hailo/`, or the reverse, because
its models are separately trained or quantized. The manifest records that as the
status between `trustmark` and `trustmark-hailo`. It was not tested here: the
`.hef` files need Hailo hardware.

Two observations bear on that statement and should be reviewed by the maintainers:

- The `'cpu'` backends of `python-hailo/` load the very same `encoder_{V}.onnx`
  and `decoder_{V}.onnx` files as `rust/` and `cpp/` (the MD5 values it pins are
  identical), so those backends belong to the `trustmark` family. On 2026-10-20
  `python-hailo/` with `encoder_backend='cpu'` and `decoder_backend='cpu'` and
  `python/` decoded each other's watermarks for B, C, P and Q.
- `secret2image_{B,C,P,Q}.npz` contain exactly the `Gemm` weights of the
  corresponding `trustmark` `encoder_{V}.onnx`, so the NPU encoders were compiled
  from those ONNX files. Quantization changes the remaining weights.

Some Hailo files could not be downloaded on 2026-10-20 because the CDN answered
HTTP 503 every time. Their entries carry the MD5 pinned in
`python-hailo/trustmark_hailo/trustmark_hailo.py`, `null` for `size` and `sha256`,
and a `download_status` note.

## Manifest fields

Top level:

| Field | Meaning |
|---|---|
| `manifest_version` | Version of this file's layout. |
| `base_url` | CDN directory of the published files. |
| `digests` | Date and method of the digest computation, and its confirmation status. |
| `families` | Description of each model family and where its name comes from. |
| `variants` | Description, decoder backbone and typical PSNR of B, C, P and Q (from `python/CONFIG.md`). |
| `wire_format` | The 100-bit packet shared by every family and variant: layout, version bits, BCH polynomial and the four schemas. |
| `c2pa_soft_binding` | The C2PA soft-binding assertion format and the TrustMark algorithm ids found in the C2PA registry. |
| `preprocessing` | Image preprocessing contract shared by the runtimes, per-variant input sizes and per-runtime resampling. |
| `compatibility` | Supported and unsupported encoder/decoder pairs and how they were verified. |
| `artifacts` | One entry per file, keyed by file name. |

Each entry in `artifacts`:

| Field | Meaning |
|---|---|
| `format` | `pytorch-ckpt`, `yaml` (OmegaConf model config), `onnx`, `hailo-hef` or `npz`. |
| `model_family` | `trustmark`, `trustmark-hailo`, or `null` for `resizer.onnx`, which is not a watermark model. |
| `variant` | `B`, `C`, `P` or `Q` (`null` for `resizer.onnx`). |
| `role` | `encoder`, `decoder`, `remover`, `detector` (bounding-box detector), `resizer` or `config`. |
| `config_for` | For `config` entries, the roles the config is used with. |
| `part` | For multi-file Hailo models, which piece of the model the file holds. |
| `url` | Download location. |
| `repo_path` | Location in this repository, for files that also ship here. |
| `size`, `md5`, `sha256` | Size in bytes and digests of the file. |
| `md5_pinned_in` | Source files that pin the same MD5 in a `MODEL_CHECKSUMS` table. |
| `used_by` | Runtimes that load the file: `python`, `rust`, `cpp`, `js`, `python-hailo`. |
| `weights_dtype` | For checkpoints, the stored floating-point type of the weights. |
| `onnx` | For ONNX files: opset, IR version, producer, weight and compute types, and the name, type, shape and meaning of each input and output. |
| `download_status` | Present only when the file could not be downloaded. |

### Preprocessing

All runtimes feed RGB, NCHW, float32 tensors in `[-1, 1]` (`pixel / 127.5 - 1`),
crop the centered square when the long side is more than twice the short side
(variant P always crops it), and resize bilinearly. Encoders take 256x256 images.
The decoder network runs at 224x224:

- `python/` resizes to 245x245 for B, C and Q, and the PyTorch `SecretDecoder`
  then resizes to 224x224 internally.
- The B, C and Q ONNX decoders take 256x256 and contain the same internal resize
  to 224x224.
- Both decoders of variant P take 224x224.

The 245 and 256 input sizes differ intentionally; see `cpp/README.md`. The
`preprocessing.runtimes` section records which resampling each runtime uses,
since their outputs differ slightly.

## Using the manifest

- **Pick a decoder for an encoder:** match `model_family` and `variant`; any
  `format` of that pair works with the matching preprocessing.
- **Verify a download:** compare its SHA-256 (or MD5) with the artifact entry.
- **Write a C2PA soft binding:** use the `alg` listed in `c2pa_soft_binding.registered`
  for the variant. There is no registered id for variant B, and none is assigned to
  `trustmark-hailo`.
- **Python:** `TrustMark.model_info()` returns the model family, the variant and
  the size, MD5 and SHA-256 of each file the instance loaded, and whether each
  file matches its published checksum:

  ```python
  from trustmark import TrustMark
  tm = TrustMark(model_type='Q', loadRemover=False)
  info = tm.model_info()
  # {'model_family': 'trustmark', 'variant': 'Q', 'files': {'decoder_Q.ckpt': {..., 'sha256': ..., 'published': True}, ...}}
  ```

## Confirming the digests

[`check_manifest.py`](check_manifest.py) needs only the Python standard library.

```sh
# Check the manifest's own consistency and that the MODEL_CHECKSUMS tables in
# python/ and python-hailo/ and the .npz files in python-hailo/ agree with it
python3 models/check_manifest.py

# Also check files you have downloaded, for example rust/models after `cargo xtask fetch-models`
python3 models/check_manifest.py --dir rust/models

# Stream every artifact from the CDN and check its size and digests (about 1.3 GB; nothing is saved)
python3 models/check_manifest.py --remote
```

`python/tests/test_model_manifest.py` checks without any model weights that
`MODEL_CHECKSUMS` in `python/trustmark/trustmark.py` agrees with the manifest.

When a published file changes, update its entry (and the matching
`MODEL_CHECKSUMS` table) and the `digests.computed` date.

## Reproducing the ONNX exports

[`python/tools/export_onnx.py`](../python/tools/export_onnx.py) loads the
published PyTorch checkpoints of a variant through the `trustmark` package
(downloading them like `TrustMark()` does), exports its encoder and decoder to
ONNX and verifies the result. It needs `onnx` and `onnxruntime` besides the
package's own dependencies; they are tool requirements, not package
dependencies.

```sh
pip install -e python/ onnx onnxruntime
# The published ONNX files to compare with, e.g. fetched by `cargo xtask fetch-models` in rust/
python python/tools/export_onnx.py --variant Q --variant P --out exported --reference rust/models
```

The published ONNX files store float16 weights and compute in float16 behind
float32 inputs and outputs. The tool reproduces that with the TorchScript-based
exporter at opset 17 and the published input and output names
(`onnx::Concat_0`, `onnx::Gemm_1` and `image` for encoders; `image` and
`output` for decoders). `--precision float32` instead writes full precision
`encoder_<V>_fp32.onnx` and `decoder_<V>_fp32.onnx` and its report to
`export_report_fp32.json`. Existing files are never overwritten.

It then checks, and summarizes in `export_report.json`:

1. the input/output signature, opset and precision against this manifest and,
   with `--reference`, the graph and weights against the published files;
2. the maximum absolute output difference between the PyTorch module, the new
   export and the published file on seeded random inputs, on the images and on
   watermarked images, and for decoders the agreement of the decoded bits;
3. that payloads embedded and read with the new exports, alone or combined with
   the PyTorch modules and the published files, are recovered.

Each export carries `trustmark.model_family`, `trustmark.variant`,
`trustmark.role`, `trustmark.precision`, `trustmark.sources` (name, MD5 and
SHA-256 of the config and checkpoint), `trustmark.preprocessing`,
`trustmark.exporter` and `trustmark.manifest_version` in its ONNX
`metadata_props`, and a `<file>.json` sidecar laid out like an `artifacts`
entry of this manifest.

Results on 2026-10-21 with torch 2.14.1 and onnxruntime 1.31.0, comparing with
the published files on `images/ghost.png`, `images/ripley.jpg` and
`images/ufo_240.jpg`:

| Variant | Encoder vs published | Decoder vs published | Decoded bits on watermarked images, new vs published | Payloads recovered |
|---|---|---|---|---|
| Q | identical graph and weights; identical outputs | identical graph; 0.74% of weight values differ, by at most 0.002 | 300/300 | 15/15 |
| P | identical graph and weights; identical outputs | identical graph; 1.7% of weight values differ, by at most 0.008 | 300/300 | 15/15 |
| B | identical graph; weights differ by at most 0.004 | identical graph; weights differ by at most 0.027 | 300/300 | 15/15 |
| C | identical graph; weights differ by at most 0.004 | identical graph; weights differ by at most 0.025 | 300/300 | 15/15 |

- For Q and P the re-exported encoders differ from the published files only in
  the producer version and the added metadata: with those reset, the files are
  byte-identical. The decoder weight differences sit almost entirely in 1x1
  convolutions, whose batch normalization the exporter folds into the weights,
  and about 98% of them are one or two float16 units in the last place. The
  cause was not determined; the published files were exported with PyTorch
  2.2 (Q) and 2.5 (P), possibly on another device.
- The B and C checkpoints store bfloat16 weights while the published ONNX files
  hold float16 weights. Rounding the published float16 encoder weights to
  bfloat16 gives the checkpoint values for 93.7% of them, the rate expected when
  both are rounded from the same higher precision weights, which are not
  published. A re-export from the checkpoints therefore cannot match the B and C
  files bit for bit, but it agrees with the PyTorch modules more closely than the
  published files do.
- With `--precision float32`, Q's files double in size (35 and 95 MB instead of
  17 and 47 MB) and match the PyTorch modules to within 4e-6 (encoder) and 2e-4
  (decoder logits), against 4e-4 and 0.08 for the float16 files; decoded bits
  and recovered payloads were the same as with float16.
