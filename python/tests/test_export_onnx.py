# Copyright 2026 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

# Tests python/tools/export_onnx.py on randomly initialised networks of the published
# architectures, so no model weights are needed: the exports must have the input/output
# signature, opset and precision that models/manifest.json records for the published files.

import importlib.util
import json
import os

import numpy as np
import pytest
import torch

onnx = pytest.importorskip('onnx')
ort = pytest.importorskip('onnxruntime')

from trustmark.unet import SecretDecoder, Unet1

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'export_onnx.py')


@pytest.fixture(scope='module')
def tool():
    spec = importlib.util.spec_from_file_location('export_onnx', TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not os.path.isfile(module.MANIFEST):
        pytest.skip('models/manifest.json is only available in a repository checkout')
    return module


@pytest.fixture(scope='module')
def manifest(tool):
    with open(tool.MANIFEST, encoding='utf-8') as f:
        return json.load(f)


def networks():
    # The architectures of variant C (trustmark_C.yaml): a Unet1 encoder, narrowed from width 32
    # to 4 because float16 convolutions are slow on CPU, and a ResNet-18 decoder
    torch.manual_seed(0)
    encoder = Unet1(resolution=256, secret_len=100, width=4, ndown=4, nmiddle=1, activ='silu').eval()
    decoder = SecretDecoder(arch='resnet18', resolution=224, secret_len=100).eval()
    return {'encoder': encoder, 'decoder': decoder}


def example_inputs(role):
    generator = torch.Generator().manual_seed(1)
    if role == 'encoder':
        return (torch.rand(1, 3, 256, 256, generator=generator) * 2 - 1,
                torch.randint(0, 2, (1, 100), generator=generator).float())
    return (torch.rand(1, 3, 256, 256, generator=generator) * 2 - 1,)


@pytest.mark.parametrize('precision', ['float16', 'float32'])
def test_export_matches_published_signature(tool, manifest, tmp_path, precision):
    for role, module in networks().items():
        path = str(tmp_path / f'{role}_C.onnx')
        inputs = example_inputs(role)
        tool.export(tool.prepare(module, precision), inputs, path, role)

        model = onnx.load(path)
        described = tool.describe_onnx(model)
        published = manifest['artifacts'][f'{role}_C.onnx']['onnx']
        assert described['inputs'] == tool.strip_semantics(published['inputs'])
        assert described['outputs'] == tool.strip_semantics(published['outputs'])
        assert described['opset'] == published['opset']
        assert described['weights_dtype'] == precision
        assert described['compute_dtype'] == precision
        if precision == 'float16':
            assert described['ir_version'] == published['ir_version']

        with torch.no_grad():
            expected = module(*inputs).numpy()
        session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        actual = session.run(None, dict(zip(tool.INPUT_NAMES[role], [x.numpy() for x in inputs])))[0]
        assert actual.shape == expected.shape
        assert np.abs(actual - expected).max() < (5e-2 if precision == 'float16' else 1e-4)


def test_prepare_leaves_the_source_module_unchanged(tool):
    module = networks()['decoder']
    prepared = tool.prepare(module, 'float16')
    assert next(prepared.parameters()).dtype == torch.float16
    assert next(module.parameters()).dtype == torch.float32
    assert 'forward' not in vars(module)


def test_add_metadata(tool, tmp_path):
    path = str(tmp_path / 'decoder_C.onnx')
    tool.export(tool.prepare(networks()['decoder'], 'float32'), example_inputs('decoder'), path, 'decoder')
    tool.add_metadata(path, {'variant': 'C', 'sources': [{'name': 'decoder_C.ckpt'}]})
    properties = {p.key: p.value for p in onnx.load(path).metadata_props}
    assert properties == {'trustmark.variant': 'C', 'trustmark.sources': '[{"name": "decoder_C.ckpt"}]'}
