"""Check the inputs encode, decode and remove_watermark accept.

The networks are replaced by stubs, so no model weights are needed, except for the
round trips at the end, which only run when the Q model files are already present.
"""

import io
import math
import pathlib

import numpy as np
import pytest
import torch
from PIL import Image

import trustmark
from trustmark import TrustMark, load_image
from trustmark.datalayer import DataLayer

PAYLOAD = ''.join('1' if i % 3 == 0 else '0' for i in range(61))


class StubEncoder:
    # returns the cover unchanged, so encode() adds a zero residual
    device = 'cpu'

    def __call__(self, cover, secret):
        assert cover.shape[1] == 3
        return cover, None


class StubRemover:
    device = 'cpu'

    def __call__(self, image):
        assert image.shape[1] == 3
        return image


class StubDecoder:
    # reports the packet of PAYLOAD for any 3-channel input
    device = 'cpu'

    def __init__(self, packet):
        self.inputs = []
        logits = torch.from_numpy(np.asarray(packet, dtype=np.float32)) * 2 - 1

        def decoder(image):
            assert image.shape[1] == 3
            self.inputs.append(image)
            return logits.reshape(1, -1)
        self.decoder = decoder


def stub_trustmark(concentrate_wm_region=1.0):
    # just enough state for encode, decode and remove_watermark, with stub networks
    tm = TrustMark.__new__(TrustMark)
    tm.use_ECC = True
    tm.secret_len = 100
    tm.model_type = 'Q'
    tm.device = 'cpu'
    tm.enctyp = TrustMark.Encoding.BCH_5
    tm.ecc = DataLayer(100, verbose=False, encoding_mode=tm.enctyp)
    tm.aspect_ratio_lim = 2.0
    tm.concentrate_wm_region = concentrate_wm_region
    tm.model_resolution_enc = 256
    tm.model_resolution_dec = 245
    tm.model_resolution_remove = 256
    tm.encoder = StubEncoder()
    tm.decoder = StubDecoder(tm.ecc.encode_binary([PAYLOAD])[0])
    tm.removal = StubRemover()
    tm.detector = None
    return tm


def call(tm, method, image, **kwargs):
    # encode/decode in binary mode, remove_watermark as is
    if method == 'encode':
        return tm.encode(image, PAYLOAD, MODE='binary', **kwargs)
    if method == 'decode':
        return tm.decode(image, MODE='binary', **kwargs)
    return tm.remove_watermark(image, **kwargs)


def rgba_noise(size=(48, 36), seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 4), dtype=np.uint8), 'RGBA')


def palette_image(transparency=None):
    image = rgba_noise().convert('RGB').convert('P', palette=Image.Palette.ADAPTIVE, colors=8)
    if transparency is not None:
        image.info['transparency'] = transparency
    return image


# name -> (input image, expected output mode, expected alpha band or None)
def cases():
    rgba = rgba_noise()
    la = rgba.convert('LA')
    p_index = palette_image(transparency=3)
    p_alpha = palette_image(transparency=bytes([0, 64, 128, 255, 255, 10, 200, 255]))
    return {
        'RGB': (rgba.convert('RGB'), 'RGB', None),
        'RGBA': (rgba, 'RGBA', rgba.getchannel('A')),
        'LA': (la, 'RGBA', la.getchannel('A')),
        'L': (rgba.convert('L'), 'RGB', None),
        'P': (palette_image(), 'RGB', None),
        'P-transparent-index': (p_index, 'RGBA', p_index.convert('RGBA').getchannel('A')),
        'P-alpha-palette': (p_alpha, 'RGBA', p_alpha.convert('RGBA').getchannel('A')),
        'CMYK': (rgba.convert('CMYK'), 'RGB', None),
    }


CASES = list(cases())


def rgb_content(image):
    # the colour content encode is expected to watermark
    if image.mode == 'P' and 'transparency' in image.info:
        image = image.convert('RGBA')
    return image.convert('RGB')


@pytest.mark.parametrize('name', CASES)
@pytest.mark.parametrize('method', ['encode', 'remove_watermark'])
def test_output_mode_and_alpha(name, method):
    image, out_mode, alpha = cases()[name]
    tm = stub_trustmark()
    run = lambda im: call(tm, method, im)
    out = run(image)
    assert out.mode == out_mode
    assert out.size == image.size
    # the colour channels are what the RGB path produces for the converted image
    assert out.convert('RGB').tobytes() == run(rgb_content(image)).tobytes()
    if alpha is None:
        assert 'A' not in out.getbands()
    else:
        assert out.getchannel('A').tobytes() == alpha.tobytes()


@pytest.mark.parametrize('method', ['encode', 'remove_watermark'])
def test_rgba_palette_alpha_is_preserved(method):
    rgba = Image.new('RGBA', (48, 36), (255, 0, 0, 0))
    rgba.paste((0, 255, 0, 128), (24, 0, 48, 36))
    image = rgba.convert('P')
    # Pillow stores this transparency in the palette, without an info entry.
    assert image.palette.mode == 'RGBA'
    assert 'transparency' not in image.info
    expected = image.convert('RGBA')

    out = call(stub_trustmark(), method, image)

    assert out.mode == 'RGBA'
    assert out.size == image.size
    assert out.getchannel('A').tobytes() == expected.getchannel('A').tobytes()
    rgb_out = call(stub_trustmark(), method, expected.convert('RGB'))
    assert out.convert('RGB').tobytes() == rgb_out.tobytes()


def test_rgb_input_is_used_without_conversion(monkeypatch):
    image = rgba_noise().convert('RGB')

    def no_convert(self, *args, **kwargs):
        raise AssertionError('RGB input must not be converted')
    monkeypatch.setattr(Image.Image, 'convert', no_convert)
    tm = stub_trustmark()
    assert tm.encode(image, PAYLOAD, MODE='binary').mode == 'RGB'
    assert tm.remove_watermark(image).mode == 'RGB'
    assert tm.decode(image, MODE='binary') == (PAYLOAD, True, TrustMark.Encoding.BCH_5)


@pytest.mark.parametrize('name', CASES)
def test_decode_reads_rgb_content(name):
    image = cases()[name][0]
    tm = stub_trustmark()
    assert tm.decode(image, MODE='binary') == (PAYLOAD, True, TrustMark.Encoding.BCH_5)
    expected = stub_trustmark()
    expected.decode(rgb_content(image), MODE='binary')
    assert torch.equal(tm.decoder.inputs[0], expected.decoder.inputs[0])


def unsupported_image(mode):
    if mode == 'I;16':
        return Image.fromarray(np.full((8, 8), 40000, dtype=np.uint16))
    return Image.new(mode, (8, 8))


@pytest.mark.parametrize('mode', ['I;16', 'I;16B', 'I', 'F'])
@pytest.mark.parametrize('method', ['encode', 'decode', 'remove_watermark'])
def test_high_bit_depth_modes_are_rejected(mode, method):
    tm = stub_trustmark()
    with pytest.raises(ValueError, match=r"'%s' has more than 8 bits.*adobe/trustmark#36" % mode):
        call(tm, method, unsupported_image(mode))


@pytest.mark.parametrize('mode', ['1', 'YCbCr', 'PA', 'RGBX', 'HSV'])
@pytest.mark.parametrize('method', ['encode', 'decode', 'remove_watermark'])
def test_other_modes_are_rejected(mode, method):
    tm = stub_trustmark()
    with pytest.raises(ValueError, match="unsupported image mode '%s'" % mode):
        call(tm, method, Image.new(mode, (8, 8)))


@pytest.mark.parametrize('method', ['encode', 'decode', 'remove_watermark'])
def test_non_pil_input_is_rejected(method):
    tm = stub_trustmark()
    with pytest.raises(TypeError, match='expected a PIL.Image.Image, got ndarray'):
        call(tm, method, np.zeros((8, 8, 3), dtype=np.uint8))


@pytest.mark.parametrize('size', [(0, 0), (0, 5), (5, 0)])
@pytest.mark.parametrize('method', ['encode', 'decode', 'remove_watermark'])
def test_empty_image_is_rejected(size, method):
    tm = stub_trustmark()
    with pytest.raises(ValueError, match='image has no pixels'):
        call(tm, method, Image.new('RGB', size))


@pytest.mark.parametrize('size', [(1, 1), (1, 3), (3, 1), (2, 2)])
def test_tiny_images_are_processed(size):
    tm = stub_trustmark()
    assert tm.encode(Image.new('RGB', size), PAYLOAD, MODE='binary').size == size
    assert tm.decode(Image.new('RGB', size), MODE='binary')[1]


@pytest.mark.parametrize('method', ['encode', 'decode', 'remove_watermark'])
def test_region_emptied_by_concentration_is_rejected(method):
    tm = stub_trustmark(concentrate_wm_region=0.1)
    with pytest.raises(ValueError, match='too small for concentrate_wm_region=0.1'):
        call(tm, method, Image.new('RGB', (9, 9)))
    call(tm, method, Image.new('RGB', (10, 10)))


@pytest.mark.parametrize('strength', [math.nan, math.inf, -math.inf, -0.5, '1.0', None, 1j])
@pytest.mark.parametrize('method', ['encode', 'remove_watermark'])
def test_invalid_strength_is_rejected(strength, method):
    tm = stub_trustmark()
    with pytest.raises(ValueError, match='WM_STRENGTH must be a finite number >= 0'):
        call(tm, method, Image.new('RGB', (8, 8)), WM_STRENGTH=strength)


@pytest.mark.parametrize('strength', [0, 0.0, 1, 1.5, np.float32(0.8), np.float64(2.0)])
@pytest.mark.parametrize('method', ['encode', 'remove_watermark'])
def test_valid_strength_is_accepted(strength, method):
    tm = stub_trustmark()
    assert call(tm, method, Image.new('RGB', (8, 8)), WM_STRENGTH=strength).size == (8, 8)


def construct(monkeypatch, **kwargs):
    def load_model(self, *args, **kw):
        return object()
    monkeypatch.setattr(TrustMark, 'load_model', load_model)
    return TrustMark(verbose=False, device='cpu', model_type='Q', loadRemover=False, **kwargs)


@pytest.mark.parametrize('value', [0, 0.0, -0.2, 1.0001, 2, math.nan, math.inf, '0.5', None])
def test_invalid_concentrate_wm_region_is_rejected(monkeypatch, value):
    with pytest.raises(ValueError, match=r'concentrate_wm_region must be a finite number in \(0, 1\]'):
        construct(monkeypatch, concentrate_wm_region=value)


@pytest.mark.parametrize('value', [1, 1.0, 0.5, 0.01, np.float32(0.8)])
def test_valid_concentrate_wm_region_is_accepted(monkeypatch, value):
    assert construct(monkeypatch, concentrate_wm_region=value).concentrate_wm_region == value


def save_with_orientation(path, image, orientation, **kwargs):
    exif = Image.Exif()
    exif[0x0112] = orientation  # Orientation
    exif[0x010F] = 'TrustMark test'  # Make
    image.save(path, exif=exif.tobytes(), **kwargs)


def fake_icc_profile(colour_space):
    # only the header fields load_image looks at; 128-byte ICC header
    header = bytearray(128)
    header[16:20] = colour_space
    return bytes(header)


@pytest.mark.parametrize('orientation,transpose', [
    (1, None),
    (3, Image.Transpose.ROTATE_180),
    (6, Image.Transpose.ROTATE_270),
    (8, Image.Transpose.ROTATE_90),
])
def test_load_image_applies_exif_orientation(tmp_path, orientation, transpose):
    stored = rgba_noise(size=(6, 4))
    path = tmp_path / 'oriented.png'
    icc = fake_icc_profile(b'RGB ')
    save_with_orientation(path, stored, orientation, icc_profile=icc, dpi=(300, 300))

    image, metadata = load_image(path)

    upright = stored if transpose is None else stored.transpose(transpose)
    assert image.mode == 'RGBA'
    assert image.tobytes() == upright.tobytes()
    assert sorted(metadata) == ['dpi', 'exif', 'icc_profile']
    exif = Image.Exif()
    exif.load(metadata['exif'])
    assert 0x0112 not in exif
    assert exif[0x010F] == 'TrustMark test'
    assert metadata['icc_profile'] == icc
    assert metadata['dpi'] == pytest.approx((300, 300), abs=0.01)


def test_load_image_metadata_round_trips_through_save(tmp_path):
    path = tmp_path / 'in.jpg'
    save_with_orientation(path, rgba_noise(size=(6, 4)).convert('RGB'), 6, dpi=(72, 72))
    image, metadata = load_image(path)
    out = tmp_path / 'out.jpg'
    image.save(out, **metadata)
    with Image.open(out) as reloaded:
        assert reloaded.size == (4, 6)
        assert 0x0112 not in reloaded.getexif()
        assert reloaded.getexif()[0x010F] == 'TrustMark test'


@pytest.mark.parametrize('mode,colour_space', [('L', b'GRAY'), ('CMYK', b'CMYK')])
def test_load_image_drops_non_rgb_icc_profile(tmp_path, mode, colour_space):
    path = tmp_path / 'image.jpg'
    Image.new(mode, (4, 4)).save(path, icc_profile=fake_icc_profile(colour_space))
    image, metadata = load_image(path)
    assert image.mode == mode
    assert 'icc_profile' not in metadata


def test_load_image_without_metadata(tmp_path):
    path = tmp_path / 'plain.png'
    Image.new('RGB', (4, 4)).save(path)
    image, metadata = load_image(path)
    assert metadata == {}
    assert image.size == (4, 4)


def test_load_image_is_exported():
    assert trustmark.load_image is load_image


# round trips with the real Q weights, only when they are already present (no download)

IMAGES = pathlib.Path(__file__).resolve().parents[2] / 'images'
MODELS = pathlib.Path(trustmark.__file__).resolve().parent / 'models'
Q_FILES = ['trustmark_Q.yaml', 'encoder_Q.ckpt', 'decoder_Q.ckpt']
needs_q_weights = pytest.mark.skipif(not all((MODELS / f).is_file() for f in Q_FILES) or not IMAGES.is_dir(),
                                     reason='Q model files or repository images not present')


def ghost_with_palette_transparency():
    # 255 opaque colours plus a fully transparent index where the ghost is mostly transparent
    ghost = Image.open(IMAGES / 'ghost.png')
    opaque = ghost.convert('RGB').quantize(colors=255)
    indices = np.asarray(opaque).copy()
    indices[np.asarray(ghost.getchannel('A')) < 128] = 255
    image = Image.frombytes('P', ghost.size, indices.tobytes())
    image.putpalette(opaque.getpalette()[:255 * 3] + [0, 0, 0])
    image.info['transparency'] = 255
    return image


@pytest.fixture(scope='module')
def q_model():
    return TrustMark(verbose=False, device='cpu', model_type='Q', loadRemover=False)


@needs_q_weights
@pytest.mark.parametrize('name,make', [
    ('ghost RGBA', lambda: Image.open(IMAGES / 'ghost.png')),
    ('ufo L', lambda: Image.open(IMAGES / 'ufo_240.jpg').convert('L')),
    ('ufo P', lambda: Image.open(IMAGES / 'ufo_240.jpg').convert('P', palette=Image.Palette.ADAPTIVE)),
    ('ghost P with transparency', ghost_with_palette_transparency),
])
def test_real_round_trip(q_model, name, make):
    image = make()
    image.load()
    watermarked = q_model.encode(image, PAYLOAD, MODE='binary')
    if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
        assert 'with transparency' not in name or 0 in watermarked.getchannel('A').getextrema()
        assert watermarked.mode == 'RGBA'
        expected_alpha = image.getchannel('A') if image.mode == 'RGBA' else image.convert('RGBA').getchannel('A')
        assert watermarked.getchannel('A').tobytes() == expected_alpha.tobytes()
    else:
        assert watermarked.mode == 'RGB'
    buffer = io.BytesIO()
    watermarked.save(buffer, 'PNG')
    buffer.seek(0)
    assert q_model.decode(Image.open(buffer), MODE='binary') == (PAYLOAD, True, TrustMark.Encoding.BCH_5)
