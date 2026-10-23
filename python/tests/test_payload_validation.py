"""Check that encode rejects payloads that do not fit, without model weights."""

import numpy as np
import pytest
from PIL import Image

from trustmark import TrustMark
from trustmark.datalayer import DataLayer

# (encoding mode, payload capacity in bits)
SCHEMAS = [(0, 40), (1, 61), (2, 68), (3, 75)]


def roundtrip(layer, payload, mode):
    if mode == 'binary':
        packet = layer.encode_binary([payload])
    else:
        packet = layer.encode_text([payload])
    return layer.decode_bitstream(packet, MODE=mode)[0]


@pytest.mark.parametrize('mode,capacity', SCHEMAS)
def test_binary_payload_at_capacity_round_trips(mode, capacity):
    layer = DataLayer(100, encoding_mode=mode)
    payload = ''.join('0' if i % 3 else '1' for i in range(capacity))
    assert layer.schemaCapacity(mode) == capacity
    assert roundtrip(layer, payload, 'binary') == (payload, True, mode)


@pytest.mark.parametrize('mode,capacity', SCHEMAS)
def test_binary_payload_over_capacity_is_rejected(mode, capacity):
    layer = DataLayer(100, encoding_mode=mode)
    with pytest.raises(ValueError, match=r'at most %d bits' % capacity):
        layer.encode_binary(['1' * (capacity + 1)])


@pytest.mark.parametrize('payload', ['0102', '01 1', 'abc', '1.0'])
def test_binary_payload_with_invalid_characters_is_rejected(payload):
    layer = DataLayer(100, encoding_mode=1)
    with pytest.raises(ValueError, match="'0' and '1'"):
        layer.encode_binary([payload])


def test_short_binary_payload_is_zero_padded():
    layer = DataLayer(100, encoding_mode=1)
    packet = layer.encode_binary(['101'])
    assert packet.shape == (1, 100)
    assert np.array_equal(packet, layer.encode_binary(['101' + '0' * 58]))
    assert roundtrip(layer, '101', 'binary') == ('101' + '0' * 58, True, 1)


@pytest.mark.parametrize('mode,capacity', SCHEMAS)
def test_text_payload_limit(mode, capacity):
    layer = DataLayer(100, encoding_mode=mode)
    max_chars = capacity // 7
    text = 'TrustMark!'[:max_chars]
    assert len(text) == max_chars
    assert roundtrip(layer, text, 'text') == (text, True, mode)
    with pytest.raises(ValueError, match=r'at most %d characters' % max_chars):
        layer.encode_text([text + 'x'])


def test_short_text_payload_round_trips():
    layer = DataLayer(100, encoding_mode=1)
    assert roundtrip(layer, 'hi', 'text') == ('hi', True, 1)


@pytest.mark.parametrize('text', ['caf\u00e9', '\u2603', 'a\x80'])
def test_text_payload_with_non_ascii_is_rejected(text):
    layer = DataLayer(100, encoding_mode=1)
    with pytest.raises(ValueError, match='7-bit ASCII'):
        layer.encode_text([text])


def trustmark_without_models(use_ECC):
    # just enough state for encode() to reach payload validation
    tm = TrustMark.__new__(TrustMark)
    tm.use_ECC = use_ECC
    tm.secret_len = 100
    tm.model_type = 'Q'
    tm.device = 'cpu'
    tm.enctyp = TrustMark.Encoding.BCH_5
    tm.ecc = DataLayer(100, encoding_mode=tm.enctyp)
    return tm


@pytest.mark.parametrize('payload,mode', [('1' * 62, 'binary'), ('012', 'binary'), ('mysecret!', 'text'), ('caf\u00e9', 'text')],
                         ids=['too-many-bits', 'non-binary', 'too-many-chars', 'non-ascii'])
def test_encode_rejects_invalid_payload_before_model_work(payload, mode):
    tm = trustmark_without_models(use_ECC=True)
    with pytest.raises(ValueError):
        tm.encode(Image.new('RGB', (64, 64)), payload, MODE=mode)


@pytest.mark.parametrize('payload,mode', [('1' * 99, 'binary'), ('1' * 101, 'binary'), ('mysecret', 'text')],
                         ids=['99-bits', '101-bits', 'text'])
def test_encode_without_ecc_requires_secret_len_bits(payload, mode):
    tm = trustmark_without_models(use_ECC=False)
    with pytest.raises(ValueError, match='exactly 100 bits'):
        tm.encode(Image.new('RGB', (64, 64)), payload, MODE=mode)
