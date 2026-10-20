"""Check loadEncoder/loadDecoder without model weights (load_model is stubbed)."""

import numpy as np
import pytest
import torch
from PIL import Image

from trustmark import TrustMark, ModelNotLoadedError


class StubEncoder:
    # returns the cover unchanged, so encode() adds a zero residual
    device = 'cpu'

    def __call__(self, cover, secret):
        return cover, None


class StubDecoder:
    # always reports the packet it was given, as decoder logits (+1 for 1, -1 for 0)
    device = 'cpu'

    def __init__(self, packet):
        logits = torch.from_numpy(np.asarray(packet, dtype=np.float32)) * 2 - 1
        self.decoder = lambda image: logits.reshape(1, -1)


class FailingDetector:
    def yolo(self, images):
        raise AssertionError('the detector must not run when the decoder is missing')


def construct(monkeypatch, **kwargs):
    # returns the instance and the model parts the constructor asked load_model for
    parts = []

    def load_model(self, config_path, weight_path, device, secret_len, part='all'):
        parts.append(part)
        return object()

    monkeypatch.setattr(TrustMark, 'load_model', load_model)
    tm = TrustMark(verbose=False, device='cpu', model_type='Q', **kwargs)
    return tm, parts


def test_default_loads_encoder_decoder_and_remover(monkeypatch):
    tm, parts = construct(monkeypatch)
    assert sorted(parts) == ['decoder', 'encoder', 'remover']
    assert tm.encoder is not None and tm.decoder is not None


@pytest.mark.parametrize('kwargs,expected', [
    (dict(loadEncoder=False), ['decoder']),
    (dict(loadDecoder=False), ['encoder']),
    (dict(loadEncoder=False, loadDecoder=False), []),
    (dict(loadEncoder=False, loadBBoxDetector=True), ['decoder', 'detector']),
], ids=['decoder-only', 'encoder-only', 'none', 'decoder-and-detector'])
def test_only_requested_models_are_loaded(monkeypatch, kwargs, expected):
    tm, parts = construct(monkeypatch, loadRemover=False, **kwargs)
    assert sorted(parts) == expected
    assert (tm.encoder is None) == ('encoder' not in expected)
    assert (tm.decoder is None) == ('decoder' not in expected)


def test_encode_without_encoder_raises_model_not_loaded(monkeypatch):
    tm, _ = construct(monkeypatch, loadRemover=False, loadEncoder=False)
    with pytest.raises(ModelNotLoadedError, match='loadEncoder=True'):
        tm.encode(Image.new('RGB', (64, 64)), '1' * 61, MODE='binary')


@pytest.mark.parametrize('kwargs', [dict(), dict(DETECTFIRST=True), dict(ROTATION=True)],
                         ids=['plain', 'detectfirst', 'rotation'])
def test_decode_without_decoder_raises_model_not_loaded(monkeypatch, kwargs):
    tm, _ = construct(monkeypatch, loadRemover=False, loadDecoder=False)
    tm.detector = FailingDetector()
    with pytest.raises(ModelNotLoadedError, match='loadDecoder=True'):
        tm.decode(Image.new('RGB', (64, 64)), MODE='binary', **kwargs)


def test_encoder_only_instance_encodes(monkeypatch):
    tm, _ = construct(monkeypatch, loadRemover=False, loadDecoder=False)
    tm.encoder = StubEncoder()
    cover = Image.new('RGB', (80, 60), (120, 30, 200))
    stego = tm.encode(cover, '1' * 61, MODE='binary')
    assert stego.size == cover.size
    # the stub encoder leaves the cover as is
    assert np.array_equal(np.asarray(stego), np.asarray(cover))


def test_decoder_only_instance_decodes(monkeypatch):
    tm, _ = construct(monkeypatch, loadRemover=False, loadEncoder=False)
    payload = '10' * 30 + '1'
    tm.decoder = StubDecoder(tm.ecc.encode_binary([payload])[0])
    assert tm.decode(Image.new('RGB', (64, 64)), MODE='binary') == (payload, True, TrustMark.Encoding.BCH_5)


DETECTOR_CONFIG = """
model:
  target: trustmark.model.TrustMark_Arch
  params:
    cover_key: image
    secret_key: secret
    secret_len: 100
    resolution: 256
    secret_encoder_config:
      target: trustmark.unet.Unet1
      params: {width: 32, ndown: 4, nmiddle: 1, activ: silu}
    secret_decoder_config:
      target: trustmark.unet.SecretDecoder
      params: {arch: fasterRCNN2}
    yolo_config:
      target: trustmark.model.Identity
    discriminator_config: {target: trustmark.munit.MsDCDisGP}
    loss_config: {target: trustmark.loss.ImageSecretLoss}
    noise_config: {target: trustmark.utils.transformations2.TransformNet}
"""


def test_detector_skips_unused_secret_decoder(monkeypatch, tmp_path):
    # same structure as trustmark_bbox_{Q,P}.yaml, with a stand-in for the box predictor;
    # the bbox checkpoints only hold 'yolo.*' weights
    from trustmark.model import Identity
    config_path = tmp_path / 'trustmark_bbox_Q.yaml'
    config_path.write_text(DETECTOR_CONFIG)
    monkeypatch.setattr(TrustMark, 'check_and_download', lambda self, filename: None)
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: {})
    tm = TrustMark.__new__(TrustMark)
    detector = tm.load_model(str(config_path), str(tmp_path / 'trustmark_bbox_Q.ckpt'), 'cpu', 100, part='detector')
    assert isinstance(detector.decoder, Identity)
    assert isinstance(detector.encoder, Identity)
    assert isinstance(detector.yolo, Identity)
