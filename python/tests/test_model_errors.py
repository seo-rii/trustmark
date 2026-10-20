import pytest
from PIL import Image

import trustmark
from trustmark import TrustMark
from trustmark.datalayer import DataLayer


def _failing_download(self, filename):
    raise OSError('simulated download failure for ' + filename)


def _load_with_failing_parts(monkeypatch, failing_parts):
    # parts in failing_parts go through the real load_model with downloads failing,
    # the others return a stand-in so no model files are needed
    real_load_model = TrustMark.load_model

    def load_model(self, config_path, weight_path, device, secret_len, part='all'):
        if part in failing_parts:
            return real_load_model(self, config_path, weight_path, device, secret_len, part=part)
        return object()

    monkeypatch.setattr(TrustMark, 'check_and_download', _failing_download)
    monkeypatch.setattr(TrustMark, 'load_model', load_model)


def test_model_not_loaded_error_is_public_runtime_error():
    assert issubclass(trustmark.ModelNotLoadedError, RuntimeError)


def test_remove_watermark_without_remover_raises_model_not_loaded():
    tm = object.__new__(TrustMark)
    tm.removal = None
    with pytest.raises(trustmark.ModelNotLoadedError, match='loadRemover=True'):
        tm.remove_watermark(Image.new('RGB', (64, 64)))


def test_localize_without_detector_raises_model_not_loaded():
    tm = object.__new__(TrustMark)
    tm.detector = None
    with pytest.raises(trustmark.ModelNotLoadedError, match='BBox detector model is not loaded'):
        tm.localize(Image.new('RGB', (64, 64)))


@pytest.mark.parametrize('part', ['decoder', 'encoder'])
def test_constructor_raises_when_required_model_cannot_be_loaded(monkeypatch, part):
    _load_with_failing_parts(monkeypatch, {part})
    with pytest.raises(trustmark.ModelNotLoadedError, match=part + ' model') as excinfo:
        TrustMark(verbose=False, device='cpu', model_type='Q', loadRemover=False)
    assert part + '_Q.ckpt' in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, OSError)


def test_optional_models_fail_softly_then_raise_when_used(monkeypatch):
    _load_with_failing_parts(monkeypatch, {'remover', 'detector'})
    tm = TrustMark(verbose=False, device='cpu', model_type='Q', loadRemover=True, loadBBoxDetector=True)
    assert tm.removal is None
    assert tm.detector is None

    image = Image.new('RGB', (64, 64))
    with pytest.raises(trustmark.ModelNotLoadedError):
        tm.remove_watermark(image)
    with pytest.raises(trustmark.ModelNotLoadedError):
        tm.localize(image)


@pytest.mark.parametrize('encoding', [0, 1, 2, 3])
def test_datalayer_decode_text(encoding):
    dl = DataLayer(100, verbose=False, encoding_mode=encoding)
    packet = dl.encode_text(['hello'])
    assert dl.decode_text(packet) == [('hello', True, encoding)]


@pytest.mark.parametrize('encoding', [0, 1, 2, 3])
def test_datalayer_decode_binary(encoding):
    dl = DataLayer(100, verbose=False, encoding_mode=encoding)
    bits = ('1101' * 20)[:dl.schemaCapacity(encoding)]
    packet = dl.encode_binary([bits])
    assert dl.decode_binary(packet) == [(bits, True, encoding)]
