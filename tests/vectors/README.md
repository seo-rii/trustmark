# Packet conformance vectors

Golden vectors for the 100-bit TrustMark packet: payload bits, BCH error-correction
bits, then four version bits (`00` followed by the 2-bit schema id). The Python, Rust,
C++ and JavaScript data layers are all tested against them.

| Schema | Id | Data bits | ECC bits | Corrects up to |
|---|---|---|---|---|
| `BCH_SUPER` | 0 | 40 | 56 | 8 bit flips |
| `BCH_5` | 1 | 61 | 35 | 5 bit flips |
| `BCH_4` | 2 | 68 | 28 | 4 bit flips |
| `BCH_3` | 3 | 75 | 21 | 3 bit flips |

## Files

- `packets.json`: the vectors, one per line. The Python and JavaScript tests read it.
- `packets.tsv`: the same vectors as tab-separated lines, so the Rust and C++ tests need
  no JSON parser. It leaves out the `flips`, `note` and `known_failures` fields.
- `generate.py`: writes both files. Every packet and every Python expectation comes from
  the Python reference (`python/trustmark/datalayer.py` and `bchecc.py`), loaded without
  the `trustmark` package, so only numpy is needed. The vectors are deterministic (fixed
  seed), and the generator checks every expectation against the reference before writing.

Don't edit the vector files by hand. To regenerate them, or to check that they are up to
date, run this from the repository root:

```sh
python3 tests/vectors/generate.py
python3 tests/vectors/generate.py --check
```

## Tests

None of these need the models. From the repository root:

```sh
python -m pytest python/tests/test_conformance_vectors.py  # needs only numpy and pytest
node --test js/tests/*.test.mjs                            # decode vectors only; JS has no encoder
g++ -std=c++17 -Icpp/src cpp/tests/conformance.cpp cpp/src/bch.cpp -o conformance && ./conformance tests/vectors/packets.tsv
cargo test --manifest-path rust/Cargo.toml --lib conformance_tests
```

The C++ test is also registered with CTest (`trustmark-conformance`). The `Conformance`
workflow runs the Python, JavaScript and C++ tests and checks that the files are up to
date. Rust CI runs the Rust test. The Rust test reads `packets.tsv` with `include_str!`
from outside the crate, so it builds only in a repository checkout. It is compiled only
for tests, so it doesn't affect the published package.

## Format

`encode` vectors have `id`, `case`, `schema` (name), `payload` (the string passed to the
binary-mode encoder), `packet` (what the Python reference produces, or `null` if it
raises), and `expect`. `expect` maps `python`, `rust` and `cpp` to one of these:

- `packet`: the encoder returns `packet`.
- `error:length`: the encoder rejects the payload because it is longer than the schema allows.
- `error:char`: the encoder rejects the payload because it contains a character other than `0` or `1`.

`decode` vectors have `id`, `case`, `input` (100 bits), `schema` and `data` (the result
that `valid` refers to: the original payload, padded with zeros to the schema capacity),
`valid_under`, and `expect`, which maps `python`, `rust`, `cpp` and `js` to one of these:

- `valid`: the decoder reports a valid watermark with exactly `schema` and `data`.
- `invalid`: the decoder reports no valid watermark.
- `not_original`: the decoder must not report `schema` and `data`. It may report no
  watermark, or a valid watermark under another schema.

`valid_under` maps each schema whose BCH decoder accepts the 96 code bits on its own to
the data that decoder returns. Every consumer checks that any valid result it gets is
listed there. A decoder can report a valid watermark only if the BCH code of the reported
schema accepts the bits.

Optional fields: `flips` (positions flipped in the clean packet), `note`, and
`known_failures` (implementation to reason). Known failures still run, but the
JavaScript test reports them as TODO instead of failing.

## Cases

| Case | Vectors | Python | Rust, C++, JS |
|---|---|---|---|
| `random`, `all_zeros`, `all_ones`, `short` (encode) | full-capacity, all-zero, all-one, and 0, 1, 8 and capacity−1 bit payloads (zero padded) | packet | packet |
| `too_long` (encode) | capacity + 1 bits | packet of the first *capacity* bits | `error:length` |
| `invalid_char` (encode, `BCH_5`) | `0102`, `01 1`, `abc`, `1.0`, and Arabic-Indic digits `١٠` | `error:char`, except packet for `١٠` (parsed by `int()`) | `error:char` |
| `clean` | every encoded packet | valid | valid |
| `t_flips` | *t* flips spread over the data and ECC bits | valid | valid |
| `boundary_flips` | flips at the first/last data and ECC bits (up to *t*) | valid | valid |
| `t_plus_1_flips` | *t*+1 flips | invalid | not_original |
| `wrong_schema_bits` | schema bits set to every other schema | invalid | valid |
| `wrong_schema_bits_with_flip` | the same, plus one flip | invalid | valid |
| `reserved_version_bits` | first two version bits `01`, `10` or `11` | valid | valid |
| `no_schema_valid` | all ones, and random words that no schema accepts | invalid | invalid |
| `pad_bit_correction` | a word that decodes only by flipping a padding bit | valid | valid (Rust, JS), invalid (C++) |

The behavior differences are intentional or known:

- **Schema fallback.** Python decodes only the schema named by the schema bits. Rust,
  C++ and JS try the named schema first, then the others. So for packets that only their
  own schema accepts, the fallback decoders recover packets whose schema bits are
  corrupted, and Python does not. The other side of the fallback: roughly one in ten
  random 100-bit strings decodes under some schema in Rust or C++.
- **Why `t_plus_1_flips` is `not_original` and not `invalid`.** With *t*+1 flips, the
  named schema cannot decode the word back to the original packet. A fallback decoder
  then tries the other schemas, and the corrupted word may be within their correction
  radius. The first *t*+1 vector for each schema is chosen so that this happens
  (`valid_under` names the schema). The only safe assertion is that the original payload
  is never reported. The generator only uses words that the named schema rejects, so
  Python reports `invalid`.
- **Reserved version bits.** Python, C++ and JS read only the last two version bits.
  Rust reads all four and treats an unknown version as `BCH_SUPER` before falling back.
  The vectors only use packets where this changes nothing.
- **Payload validation.** The Python reference silently truncates payloads longer than
  the capacity, and accepts any Unicode decimal digit that `int()` parses. Rust and C++
  reject both. When Python starts rejecting them, regenerate the vectors. The `python`
  expectations follow automatically.
- **Padding bits.** Before decoding, the data is padded with zeros to a whole byte. The
  Python, Rust and JS BCH decoders can accept a word by "correcting" one of those padding
  bits. C++ checks the corrected data against its ECC again, and rejects it. Apart from
  `pad_bit_correction`, the vectors avoid words where any decoder could reach such a
  schema.
- **JS fallback order** (`known_failures.js`). `js/tm_datalayer.js` never tries `BCH_3`
  when the schema bits name another schema. When that is fixed, remove
  `JS_NO_BCH_3_FALLBACK` from `generate.py` and regenerate.
