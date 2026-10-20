# Copyright 2023 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

# Regression tests for the model_dir and allow_download constructor options.
# load_model is replaced so no weights are needed, and urllib.request.urlopen is
# stubbed so the network is never used.

import io
import os
import urllib.request
from hashlib import md5

import pytest

import trustmark.trustmark as tm_module
from trustmark.trustmark import TrustMark

PACKAGE_MODELS = os.path.join(os.path.dirname(os.path.realpath(tm_module.__file__)), 'models')

COMPONENT_FILES = {
    'decoder': ('trustmark_{}.yaml', 'decoder_{}.ckpt'),
    'encoder': ('trustmark_{}.yaml', 'encoder_{}.ckpt'),
    'remover': ('trustmark_rm_{}.yaml', 'trustmark_rm_{}.ckpt'),
    'detector': ('trustmark_bbox_{}.yaml', 'trustmark_bbox_{}.ckpt'),
}


@pytest.fixture
def network(monkeypatch):
    # Records every download attempt; serves `state['serve']` bytes if set,
    # otherwise fails the test instead of reaching the CDN.
    calls = []
    state = {'serve': None}

    def fake_urlopen(url, data=None, timeout=None):
        calls.append(url)
        if state['serve'] is None:
            raise AssertionError('unexpected network access: ' + url)
        return io.BytesIO(state['serve'])

    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
    return calls, state


@pytest.fixture
def loads(monkeypatch):
    # Stands in for load_model: records the paths for each component and runs only
    # the model file acquisition step, so construction needs no weights.
    loaded = []

    def fake_load_model(self, config_path, weight_path, device, secret_len, part='all'):
        loaded.append((part, config_path, weight_path))
        self.check_and_download(config_path)
        self.check_and_download(weight_path)
        return None

    monkeypatch.setattr(TrustMark, 'load_model', fake_load_model)
    return loaded


def fake_model_files(monkeypatch, model_dir, model_type, parts):
    # Writes small stand-in files for the given components and pins their checksums
    model_dir.mkdir(parents=True, exist_ok=True)
    for part in parts:
        for pattern in COMPONENT_FILES[part]:
            name = pattern.format(model_type)
            content = ('stand-in for ' + name).encode()
            (model_dir / name).write_bytes(content)
            monkeypatch.setitem(tm_module.MODEL_CHECKSUMS, name, md5(content).hexdigest())


def test_model_dir_is_used_for_every_component(tmp_path, monkeypatch, network, loads):
    calls, _ = network
    fake_model_files(monkeypatch, tmp_path, 'Q', COMPONENT_FILES)

    tm = TrustMark(verbose=False, device='cpu', model_type='Q', loadRemover=True, loadBBoxDetector=True,
                   model_dir=tmp_path, allow_download=False)

    assert tm.model_dir == str(tmp_path)
    assert sorted(part for part, _, _ in loads) == sorted(COMPONENT_FILES)
    for part, config_path, weight_path in loads:
        config_name, weight_name = COMPONENT_FILES[part]
        assert config_path == str(tmp_path / config_name.format('Q'))
        assert weight_path == str(tmp_path / weight_name.format('Q'))
    assert calls == []


def test_default_model_dir_is_package_directory(loads, monkeypatch):
    monkeypatch.setattr(TrustMark, 'check_and_download', lambda self, filename: None)

    tm = TrustMark(verbose=False, device='cpu', model_type='P', loadRemover=True, loadBBoxDetector=True)

    assert os.path.realpath(tm.model_dir) == PACKAGE_MODELS
    assert tm.allow_download is True
    for _, config_path, weight_path in loads:
        assert os.path.dirname(os.path.realpath(config_path)) == PACKAGE_MODELS
        assert os.path.dirname(os.path.realpath(weight_path)) == PACKAGE_MODELS


def test_existing_positional_arguments_are_unchanged(loads, monkeypatch):
    monkeypatch.setattr(TrustMark, 'check_and_download', lambda self, filename: None)

    tm = TrustMark(True, False, 100, 'cpu', 'Q', TrustMark.Encoding.BCH_5, 1.0, False, False)

    assert sorted(part for part, _, _ in loads) == ['decoder', 'encoder']
    assert os.path.realpath(tm.model_dir) == PACKAGE_MODELS


def test_offline_missing_file_raises_without_network(tmp_path, network, loads):
    calls, _ = network
    model_dir = tmp_path / 'not-created'

    with pytest.raises(FileNotFoundError) as excinfo:
        TrustMark(verbose=False, device='cpu', model_type='Q', loadRemover=False,
                  model_dir=model_dir, allow_download=False)

    message = str(excinfo.value)
    assert 'trustmark_Q.yaml' in message
    assert str(model_dir) in message
    assert calls == []
    assert not model_dir.exists()


def test_offline_checksum_mismatch_raises_without_network(tmp_path, monkeypatch, network, loads):
    calls, _ = network
    fake_model_files(monkeypatch, tmp_path, 'Q', ['decoder', 'encoder'])
    (tmp_path / 'decoder_Q.ckpt').write_bytes(b'truncated')

    with pytest.raises(OSError) as excinfo:
        TrustMark(verbose=False, device='cpu', model_type='Q', loadRemover=False,
                  model_dir=tmp_path, allow_download=False)

    message = str(excinfo.value)
    assert 'decoder_Q.ckpt' in message
    assert str(tmp_path) in message
    assert tm_module.MODEL_CHECKSUMS['decoder_Q.ckpt'] in message
    assert calls == []
    assert (tmp_path / 'decoder_Q.ckpt').read_bytes() == b'truncated'


def test_offline_with_valid_files_loads_without_network(tmp_path, monkeypatch, network, loads):
    calls, _ = network
    fake_model_files(monkeypatch, tmp_path, 'Q', ['decoder', 'encoder'])

    TrustMark(verbose=False, device='cpu', model_type='Q', loadRemover=False,
              model_dir=str(tmp_path), allow_download=False)

    assert sorted(part for part, _, _ in loads) == ['decoder', 'encoder']
    assert calls == []


def test_download_creates_model_dir(tmp_path, monkeypatch, network, loads):
    calls, state = network
    content = b'downloaded stand-in'
    state['serve'] = content
    for name in ('trustmark_Q.yaml', 'decoder_Q.ckpt', 'encoder_Q.ckpt'):
        monkeypatch.setitem(tm_module.MODEL_CHECKSUMS, name, md5(content).hexdigest())
    model_dir = tmp_path / 'cache' / 'trustmark'

    TrustMark(verbose=False, device='cpu', model_type='Q', loadRemover=False, model_dir=model_dir)

    assert sorted(os.listdir(model_dir)) == ['decoder_Q.ckpt', 'encoder_Q.ckpt', 'trustmark_Q.yaml']
    assert sorted(set(calls)) == sorted(tm_module.MODEL_REMOTE_HOST + name for name in os.listdir(model_dir))
