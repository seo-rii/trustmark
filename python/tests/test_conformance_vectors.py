"""Check the Python reference data layer against the shared packet vectors.

The vectors live in tests/vectors at the repository root; see the README there.
Only numpy and pytest are needed: the data layer is loaded without importing the
trustmark package, so neither torch nor the model weights are required.

Run from the repository root with:

    python -m pytest python/tests/test_conformance_vectors.py
"""

import importlib.util
import json
import pathlib

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location('trustmark_vector_generator', REPO_ROOT / 'tests' / 'vectors' / 'generate.py')
generator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generator)

VECTORS = json.loads(generator.JSON_PATH.read_text(encoding='utf-8'))
LAYERS = generator.Reference().layers
SCHEMA_NAMES = [schema.name for schema in generator.SCHEMAS]


def test_vector_files_are_up_to_date():
    for path, text in generator.generate().items():
        assert path.read_text(encoding='utf-8') == text, f'{path.name} is stale; run python3 tests/vectors/generate.py'


def test_vectors_cover_every_schema():
    assert {v['schema'] for v in VECTORS['encode']} == set(SCHEMA_NAMES)
    assert {v['schema'] for v in VECTORS['decode']} == set(SCHEMA_NAMES) | {None}


@pytest.mark.parametrize('vector', VECTORS['encode'], ids=lambda v: v['id'])
def test_encode(vector):
    layer = LAYERS[SCHEMA_NAMES.index(vector['schema'])]
    expect = vector['expect']['python']
    if expect == 'packet':
        packet = layer.encode_binary([vector['payload']])
        assert packet.shape == (1, 100)
        assert ''.join(str(int(bit)) for bit in packet[0]) == vector['packet']
    else:
        assert expect.startswith('error:')
        with pytest.raises(ValueError):
            layer.encode_binary([vector['payload']])


@pytest.mark.parametrize('vector', VECTORS['decode'], ids=lambda v: v['id'])
def test_decode(vector):
    bits = np.array([[float(bit) for bit in vector['input']]], dtype=np.float32)
    # The decoders do not depend on the layer's encoding mode
    for layer in LAYERS:
        data, detected, version = layer.decode_bitstream(bits, MODE='binary')[0]
        if vector['expect']['python'] == 'valid':
            assert (detected, SCHEMA_NAMES[version], data) == (True, vector['schema'], vector['data'])
        else:
            assert vector['expect']['python'] == 'invalid'
            assert not detected
