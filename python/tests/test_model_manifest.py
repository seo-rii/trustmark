# Copyright 2026 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

# Checks that the model checksums pinned by the package agree with models/manifest.json at the
# root of the repository, and tests TrustMark.model_info(). No model weights are needed.

import hashlib
import json
import os

import pytest
import torch

import trustmark.trustmark as tm_module
from trustmark.trustmark import MODEL_CHECKSUMS, MODEL_FAMILY, TrustMark

MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models', 'manifest.json')


@pytest.fixture(scope='module')
def artifacts():
    if not os.path.isfile(MANIFEST):
        pytest.skip('models/manifest.json is only available in a repository checkout')
    with open(MANIFEST, encoding='utf-8') as f:
        return json.load(f)['artifacts']


def test_model_checksums_match_manifest(artifacts):
    for name, md5 in MODEL_CHECKSUMS.items():
        assert name in artifacts, f'{name} is pinned in MODEL_CHECKSUMS but missing from the manifest'
        entry = artifacts[name]
        assert entry['md5'] == md5, f'{name}: MODEL_CHECKSUMS has {md5}, manifest has {entry["md5"]}'
        assert entry['model_family'] == MODEL_FAMILY
        assert entry['format'] in ('pytorch-ckpt', 'yaml')
        assert 'python/trustmark/trustmark.py' in entry['md5_pinned_in']


def test_manifest_python_artifacts_are_pinned(artifacts):
    used = {name for name, entry in artifacts.items() if 'python' in entry['used_by']}
    assert used == set(MODEL_CHECKSUMS)


def write_fake_models(directory):
    # A config that load_model can instantiate as torch.nn.Identity, and empty state dicts
    config = directory / 'trustmark_Q.yaml'
    config.write_text('model:\n  target: torch.nn.Identity\n  params:\n'
                      + ''.join(f'    {key}:\n      target: unused\n' for key in
                                ('secret_encoder_config', 'secret_decoder_config', 'discriminator_config',
                                 'loss_config', 'noise_config')))
    weights = []
    for name in ('decoder_Q.ckpt', 'encoder_Q.ckpt'):
        torch.save({}, directory / name)
        weights.append(directory / name)
    return config, weights


def test_model_info_reports_loaded_files(tmp_path, monkeypatch):
    config, (decoder, encoder) = write_fake_models(tmp_path)
    monkeypatch.setattr(TrustMark, 'check_and_download', lambda self, filename: None)
    config_md5 = hashlib.md5(config.read_bytes()).hexdigest()
    monkeypatch.setitem(tm_module.MODEL_CHECKSUMS, 'trustmark_Q.yaml', config_md5)

    tm = TrustMark.__new__(TrustMark)
    tm.model_type = 'Q'
    tm.model_files = dict()
    tm.load_model(str(config), str(decoder), 'cpu', 100, part='decoder')
    tm.load_model(str(config), str(encoder), 'cpu', 100, part='encoder')

    info = tm.model_info()
    assert info['model_family'] == 'trustmark'
    assert info['variant'] == 'Q'
    assert set(info['files']) == {'trustmark_Q.yaml', 'decoder_Q.ckpt', 'encoder_Q.ckpt'}

    yaml_info = info['files']['trustmark_Q.yaml']
    assert yaml_info['parts'] == ['decoder', 'encoder']
    assert yaml_info['md5'] == config_md5
    assert yaml_info['published'] is True

    data = decoder.read_bytes()
    decoder_info = info['files']['decoder_Q.ckpt']
    assert decoder_info['path'] == str(decoder)
    assert decoder_info['parts'] == ['decoder']
    assert decoder_info['size'] == len(data)
    assert decoder_info['md5'] == hashlib.md5(data).hexdigest()
    assert decoder_info['sha256'] == hashlib.sha256(data).hexdigest()
    assert decoder_info['published'] is False


def test_model_info_without_loaded_models():
    tm = TrustMark.__new__(TrustMark)
    tm.model_type = 'P'
    tm.model_files = dict()
    assert tm.model_info() == {'model_family': 'trustmark', 'variant': 'P', 'files': {}}
