#!/usr/bin/env python3
# Copyright 2026 Adobe
# All Rights Reserved.
#
# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
"""Generate the shared TrustMark 100-bit packet conformance vectors.

Every packet and every Python expectation is computed by the Python reference
data layer (python/trustmark/datalayer.py and bchecc.py). The modules are
loaded directly, without importing the trustmark package, so only numpy is
needed: no torch and no model weights. The other implementations' expectations
follow from the rules documented in tests/vectors/README.md, and the generator
checks them against the reference before writing anything.

Usage, from the repository root:

    python3 tests/vectors/generate.py          # rewrite packets.json and packets.tsv
    python3 tests/vectors/generate.py --check  # exit 1 if the committed files are stale
"""

import argparse
import importlib.util
import json
import pathlib
import random
import sys
import types

VECTOR_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = VECTOR_DIR.parents[1]
JSON_PATH = VECTOR_DIR / 'packets.json'
TSV_PATH = VECTOR_DIR / 'packets.tsv'

FORMAT_VERSION = 1
SEED = 1018

PACKET_BITS = 100
CODE_BITS = 96  # data + ECC; the last 4 bits are '00' + the 2-bit schema id


class Schema:
    def __init__(self, name, schema_id, data_bits, max_errors):
        self.name = name
        self.id = schema_id
        self.data_bits = data_bits
        self.ecc_bits = CODE_BITS - data_bits
        self.max_errors = max_errors

    @property
    def schema_bits(self):
        return format(self.id, '02b')


SCHEMAS = [
    Schema('BCH_SUPER', 0, 40, 8),
    Schema('BCH_5', 1, 61, 5),
    Schema('BCH_4', 2, 68, 4),
    Schema('BCH_3', 3, 75, 3),
]
BY_NAME = {schema.name: schema for schema in SCHEMAS}

IMPLEMENTATIONS = ('python', 'rust', 'cpp', 'js')
ENCODERS = ('python', 'rust', 'cpp')
FALLBACK = ('rust', 'cpp', 'js')

JS_NO_BCH_3_FALLBACK = (
    'js/tm_datalayer.js only tries BCH_SUPER, BCH_5 and BCH_4 when the schema '
    'bits name another schema, so a BCH_3 packet with corrupted schema bits is '
    'reported as invalid')


def load_datalayer():
    """Import python/trustmark/datalayer.py without the package __init__ (and torch)."""
    sys.dont_write_bytecode = True  # keep the source tree free of __pycache__
    source = REPO_ROOT / 'python' / 'trustmark'
    package = types.ModuleType('_trustmark_reference')
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    for name in ('bchecc', 'datalayer'):
        spec = importlib.util.spec_from_file_location(f'{package.__name__}.{name}', source / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules[f'{package.__name__}.datalayer'].DataLayer


class Reference:
    """The Python reference data layer, with string-in/string-out helpers."""

    def __init__(self):
        import numpy as np

        self.np = np
        data_layer = load_datalayer()
        self.layers = [data_layer(PACKET_BITS, verbose=False, encoding_mode=schema.id) for schema in SCHEMAS]

    def encode(self, schema, payload):
        """Return the packet for a binary payload, or None if the encoder raises ValueError."""
        try:
            packet = self.layers[schema.id].encode_binary([payload])[0]
        except ValueError:
            return None
        return ''.join(str(int(bit)) for bit in packet)

    def decode(self, packet):
        """Return (schema name, data) if the reference decoder accepts the packet, else None."""
        bits = self.np.array([[float(bit) for bit in packet]], dtype=self.np.float32)
        data, detected, version = self.layers[0].decode_bitstream(bits, MODE='binary')[0]
        return (SCHEMAS[version].name, data) if detected else None

    def valid_under(self, packet):
        """Decode the code bits under every schema on its own, ignoring the schema bits.

        Returns {schema name: (data, genuine)}. `genuine` is False when the decoder only
        succeeded by "correcting" one of the zero bits that pad the data to a whole
        byte: the corrected word is then not a packet any encoder can produce.
        """
        result = {}
        for schema in SCHEMAS:
            decoded = self.decode(packet[:CODE_BITS] + '00' + schema.schema_bits)
            if decoded is None:
                continue
            data = decoded[1]
            reencoded = self.encode(schema, data)
            distance = sum(a != b for a, b in zip(reencoded[:CODE_BITS], packet[:CODE_BITS]))
            result[schema.name] = (data, distance <= schema.max_errors)
        return result


def named_schema(packet, implementation):
    """The schema a decoder tries first."""
    if implementation == 'rust' and packet[96:98] != '00':
        return 'BCH_SUPER'  # Rust reads all four version bits and defaults to BCH_SUPER
    return SCHEMAS[int(packet[98:], 2)].name


def random_bits(rng, count):
    return ''.join(rng.choice('01') for _ in range(count))


def flip(packet, positions):
    bits = list(packet)
    for position in positions:
        bits[position] = '1' if bits[position] == '0' else '0'
    return ''.join(bits)


def spread_flips(rng, count, schema):
    """Pick `count` distinct code-bit positions that hit both the data and the ECC bits."""
    while True:
        positions = sorted(rng.sample(range(CODE_BITS), count))
        if positions[0] < schema.data_bits <= positions[-1]:
            return positions


class Builder:
    def __init__(self, reference):
        self.ref = reference
        self.rng = random.Random(SEED)
        self.encode = []
        self.decode = []
        self.counters = {}

    def next_id(self, kind, schema, case):
        key = (kind, schema, case)
        index = self.counters.get(key, 0)
        self.counters[key] = index + 1
        return f'{kind}/{schema}/{case}/{index}'

    # Encode vectors -----------------------------------------------------------------

    def add_encode(self, schema, case, payload, error=None, note=None):
        """Add an encode vector; `error` is 'length' or 'char' for payloads Rust and C++ reject.

        `packet` is what the Python reference produces (None if it raises ValueError).
        """
        packet = self.ref.encode(schema, payload)
        python = 'packet' if packet is not None else 'error:' + str(error)
        if error is None:
            assert packet is not None, (schema.name, payload)
            padded = payload + '0' * (schema.data_bits - len(payload))
            assert packet[:schema.data_bits] == padded and packet[-4:] == '00' + schema.schema_bits
        expect = {'python': python, 'rust': python if error is None else 'error:' + error}
        expect['cpp'] = expect['rust']
        vector = {
            'id': self.next_id('encode', schema.name, case),
            'case': case,
            'schema': schema.name,
            'payload': payload,
            'packet': packet,
            'expect': expect,
        }
        if note:
            vector['note'] = note
        self.encode.append(vector)
        return packet

    # Decode vectors -----------------------------------------------------------------

    def add_decode(self, case, packet, original, expect, flips=None, known_failures=None, note=None,
                   padding_errors=False):
        """Add a decode vector after checking its expectations against the reference.

        `original` is the (schema name, data) pair that `valid` refers to, or None.
        `expect` maps every implementation to valid / invalid / not_original.
        The implementations disagree on words that a schema accepts only by "correcting" a
        padding bit, so only pad_bit_correction vectors (`padding_errors`) may let a decoder
        reach such a schema.
        """
        assert len(packet) == PACKET_BITS and set(packet) <= {'0', '1'}
        assert set(expect) == set(IMPLEMENTATIONS)
        valid_under = self.ref.valid_under(packet)
        decodable = {name: data for name, (data, genuine) in valid_under.items() if genuine}
        if len(decodable) < len(valid_under) and not padding_errors:
            # Fine only if every decoder stops at the schema it tries first.
            for implementation in IMPLEMENTATIONS:
                assert named_schema(packet, implementation) in decodable, (case, packet, valid_under)

        python = self.ref.decode(packet)
        if expect['python'] == 'valid':
            assert python == original, (case, packet, python, original)
        else:
            assert expect['python'] == 'invalid' and python is None, (case, packet, python)

        for implementation in FALLBACK:
            outcome = expect[implementation]
            if outcome == 'valid':
                # Fallback decoders return the named schema if it decodes, otherwise any
                # schema that does; the expected result must not depend on the order.
                named = named_schema(packet, implementation)
                if named in valid_under:
                    assert (named, valid_under[named][0]) == original, (case, implementation, packet)
                else:
                    assert list(valid_under) == [original[0]], (case, implementation, packet, valid_under)
                    assert valid_under[original[0]][0] == original[1], (case, implementation, packet)
            elif outcome == 'invalid':
                assert not decodable, (case, implementation, packet, valid_under)
            else:
                assert outcome == 'not_original', outcome
                assert valid_under.get(original[0], (None,))[0] != original[1], (case, packet)
        schema = original[0] if original else None
        vector = {
            'id': self.next_id('decode', schema or 'any', case),
            'case': case,
            'input': packet,
            'schema': schema,
            'data': original[1] if original else None,
            'valid_under': {name: data for name, (data, _) in sorted(valid_under.items(), key=lambda item: BY_NAME[item[0]].id)},
            'expect': expect,
        }
        if flips is not None:
            vector['flips'] = flips
        if known_failures:
            vector['known_failures'] = known_failures
        if note:
            vector['note'] = note
        self.decode.append(vector)

    def no_padding_errors(self, word):
        return all(genuine for _, genuine in self.ref.valid_under(word).values())

    def unique_packet(self, schema):
        """A random full-capacity payload whose packet decodes under its own schema only."""
        while True:
            payload = random_bits(self.rng, schema.data_bits)
            packet = self.ref.encode(schema, payload)
            if self.ref.valid_under(packet) == {schema.name: (payload, True)}:
                return payload, packet

    def build_schema(self, schema):
        t = schema.max_errors
        same = dict.fromkeys(IMPLEMENTATIONS, 'valid')

        # Packets ----------------------------------------------------------------------
        randoms = []
        for _ in range(3):
            payload, packet = self.unique_packet(schema)
            self.add_encode(schema, 'random', payload)
            randoms.append((payload, packet))
        specials = []
        for case, payload in (('all_zeros', '0' * schema.data_bits), ('all_ones', '1' * schema.data_bits)):
            specials.append((payload, self.add_encode(schema, case, payload)))
        shorts = []
        for length in (0, 1, 8, schema.data_bits - 1):
            payload = '1' if length == 1 else random_bits(self.rng, length)
            padded = payload + '0' * (schema.data_bits - length)
            shorts.append((padded, self.add_encode(schema, 'short', payload)))
        self.add_encode(
            schema, 'too_long', random_bits(self.rng, schema.data_bits + 1), error='length',
            note='The Python reference silently drops the bits past the capacity; Rust and C++ reject the payload.')

        # Clean packets ----------------------------------------------------------------
        for payload, packet in randoms + specials + shorts:
            self.add_decode('clean', packet, (schema.name, payload), same)

        # Up to t bit flips: always recovered under the schema the schema bits name.
        for payload, packet in randoms + specials:
            while True:
                flips = spread_flips(self.rng, t, schema)
                if self.no_padding_errors(flip(packet, flips)):
                    break
            self.add_decode('t_flips', flip(packet, flips), (schema.name, payload), same, flips=flips)
        # The first and last data and ECC bits (as many as the schema corrects).
        flips = sorted({0, schema.data_bits - 1, schema.data_bits, CODE_BITS - 1})[-t:]
        payload, packet = next((payload, packet) for payload, packet in randoms
                               if self.no_padding_errors(flip(packet, flips)))
        self.add_decode('boundary_flips', flip(packet, flips), (schema.name, payload), same, flips=flips)

        # t+1 bit flips: the named schema cannot recover the packet. The first vector is
        # chosen so another schema does accept the corrupted word.
        not_original = {'python': 'invalid', 'rust': 'not_original', 'cpp': 'not_original', 'js': 'not_original'}
        for index, (payload, packet) in enumerate(randoms):
            while True:
                flips = spread_flips(self.rng, t + 1, schema)
                corrupted = flip(packet, flips)
                valid_under = self.ref.valid_under(corrupted)
                if schema.name in valid_under or not self.no_padding_errors(corrupted):
                    continue
                if bool(valid_under) == (index == 0):
                    break
            self.add_decode('t_plus_1_flips', corrupted, (schema.name, payload), not_original, flips=flips)

        # Corrupted schema bits: Python decodes only the named schema; the other decoders
        # fall back to the remaining schemas.
        fallback = {'python': 'invalid', 'rust': 'valid', 'cpp': 'valid', 'js': 'valid'}
        known = {'js': JS_NO_BCH_3_FALLBACK} if schema.name == 'BCH_3' else None
        for payload, packet in randoms[:2]:
            for other in SCHEMAS:
                if other is not schema:
                    self.add_decode('wrong_schema_bits', packet[:98] + other.schema_bits,
                                    (schema.name, payload), fallback, known_failures=known)
        payload, packet = randoms[2]
        other = SCHEMAS[(schema.id + 1) % len(SCHEMAS)]
        while True:
            flips = [self.rng.randrange(schema.data_bits)]
            corrupted = flip(packet, flips)[:98] + other.schema_bits
            if self.ref.valid_under(corrupted) == {schema.name: (payload, True)}:
                break
        self.add_decode('wrong_schema_bits_with_flip', corrupted, (schema.name, payload), fallback,
                        flips=flips, known_failures=known)

        # The first two of the four version bits are reserved and always '00'. Python, C++
        # and JS read only the last two; Rust treats the packet as BCH_SUPER and falls back.
        payload, packet = randoms[0]
        for reserved in ('01', '10', '11'):
            self.add_decode('reserved_version_bits', packet[:96] + reserved + packet[98:],
                            (schema.name, payload), same)

    def build_pad_bit_correction(self, schema):
        """A word that decodes only by flipping one of the zero bits padding the data to a byte."""
        layer = self.ref.layers[schema.id]
        padding = -schema.data_bits % 8
        while True:
            data = random_bits(self.rng, schema.data_bits)
            padded = data + '0' * (padding - 1) + '1'
            ecc = layer.bch_encoder.encode(bytearray(int(padded[i:i + 8], 2) for i in range(0, len(padded), 8)))
            ecc_bits = ''.join(format(byte, '08b') for byte in ecc)[:schema.ecc_bits]
            word = data + ecc_bits + '00' + schema.schema_bits
            if self.ref.valid_under(word) == {schema.name: (data, False)}:
                break
        self.add_decode(
            'pad_bit_correction', word, (schema.name, data),
            {'python': 'valid', 'rust': 'valid', 'cpp': 'invalid', 'js': 'valid'}, padding_errors=True,
            note='Python, Rust and JS accept a word whose only error is in the zero bits that pad the '
                 'data to a whole byte before decoding; C++ checks the corrected data against its ECC '
                 'again and rejects it. No encoder can produce this word.')

    def build(self):
        for schema in SCHEMAS:
            self.build_schema(schema)

        bch_5 = BY_NAME['BCH_5']
        for payload in ('0102', '01 1', 'abc', '1.0'):
            self.add_encode(bch_5, 'invalid_char', payload, error='char')
        self.add_encode(
            bch_5, 'invalid_char', '\u0661\u0660', error='char',
            note="ARABIC-INDIC DIGIT ONE and ZERO: the Python reference parses them with int() "
                 "and encodes the payload '10'.")

        reject = dict.fromkeys(IMPLEMENTATIONS, 'invalid')
        self.add_decode('no_schema_valid', '1' * PACKET_BITS, None, reject)
        for schema in SCHEMAS:
            while True:
                word = random_bits(self.rng, CODE_BITS) + '00' + schema.schema_bits
                if not self.ref.valid_under(word):
                    break
            self.add_decode('no_schema_valid', word, None, reject)

        for schema in SCHEMAS:
            if schema.data_bits % 8:
                self.build_pad_bit_correction(schema)
        return self.encode, self.decode


def render_json(encode, decode):
    header = {
        'format': 'trustmark-packet-vectors',
        'version': FORMAT_VERSION,
        'generator': 'tests/vectors/generate.py',
        'seed': SEED,
        'schemas': [
            {'name': s.name, 'id': s.id, 'data_bits': s.data_bits, 'ecc_bits': s.ecc_bits, 'max_errors': s.max_errors}
            for s in SCHEMAS
        ],
    }
    lines = ['{']
    for key, value in header.items():
        lines.append(f'  {json.dumps(key)}: {json.dumps(value)},')
    for key, vectors in (('encode', encode), ('decode', decode)):
        lines.append(f'  {json.dumps(key)}: [')
        lines.extend(f'    {json.dumps(v)}{"," if i < len(vectors) - 1 else ""}' for i, v in enumerate(vectors))
        lines.append('  ],' if key == 'encode' else '  ]')
    lines.append('}')
    return '\n'.join(lines) + '\n'


def render_tsv(encode, decode):
    lines = [
        '# TrustMark packet conformance vectors, generated by tests/vectors/generate.py from packets.json.',
        '# Do not edit; see tests/vectors/README.md. Fields are separated by one tab; empty fields are allowed.',
        '# encode<TAB>id<TAB>schema<TAB>payload<TAB>packet<TAB>python<TAB>rust<TAB>cpp',
        '# decode<TAB>id<TAB>input<TAB>schema<TAB>data<TAB>valid_under<TAB>python<TAB>rust<TAB>cpp<TAB>js',
    ]
    for v in encode:
        assert '\t' not in v['payload'] and '\n' not in v['payload']
        lines.append('\t'.join(['encode', v['id'], v['schema'], v['payload'], v['packet'] or '']
                               + [v['expect'][i] for i in ENCODERS]))
    for v in decode:
        valid_under = ';'.join(f'{name}={data}' for name, data in v['valid_under'].items())
        lines.append('\t'.join(['decode', v['id'], v['input'], v['schema'] or '', v['data'] or '', valid_under]
                               + [v['expect'][i] for i in IMPLEMENTATIONS]))
    return '\n'.join(lines) + '\n'


def generate():
    encode, decode = Builder(Reference()).build()
    return {JSON_PATH: render_json(encode, decode), TSV_PATH: render_tsv(encode, decode)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true', help='fail if the committed vector files are stale')
    args = parser.parse_args()
    outputs = generate()
    stale = [path for path, text in outputs.items() if not path.exists() or path.read_text(encoding='utf-8') != text]
    if args.check:
        for path in stale:
            print(f'{path.relative_to(REPO_ROOT)} is stale; run python3 tests/vectors/generate.py', file=sys.stderr)
        return 1 if stale else 0
    for path in stale:
        with open(path, 'w', encoding='utf-8', newline='\n') as file:
            file.write(outputs[path])
        print(f'wrote {path.relative_to(REPO_ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
