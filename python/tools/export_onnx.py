# Copyright 2026 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

"""Export TrustMark encoders and decoders to ONNX and verify the exports.

Loads the published PyTorch checkpoints of each variant through the trustmark package,
exports the encoder and decoder with the same input/output names, shapes, types and opset
as the published ONNX files (see models/manifest.json), embeds metadata, and verifies:

  (a) the input/output signature and opset against models/manifest.json and, with
      --reference, the graph and weights against the published ONNX files;
  (b) numerical agreement between the PyTorch modules, the new exports and, with
      --reference, the published files on seeded random inputs, real images and
      watermarked images (max abs difference, and decoder bit agreement);
  (c) end-to-end payload recovery when the new exports encode and decode.

The published files store float16 weights and compute in float16 behind float32 inputs and
outputs. The default --precision float16 reproduces that by running the module in half
precision and casting its inputs and output in its forward pass, which gives the same graph,
node names included. --precision float32 writes a full precision export as <name>_fp32.onnx.

Each export gets trustmark.* entries in its ONNX metadata_props and a <file>.json sidecar laid
out like an artifact entry of models/manifest.json. A summary of every check goes to
export_report.json in the output directory. Existing files are never overwritten.

Requirements beyond the trustmark package: onnx and onnxruntime. Run from a repository
checkout, since models/manifest.json is read:

    pip install -e python/ onnx onnxruntime
    python python/tools/export_onnx.py --variant Q --out exported --reference rust/models

Exits with status 1 if a signature differs or a payload is not recovered.
"""

import argparse
import copy
import hashlib
import inspect
import json
import os
import sys
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, numpy_helper
from PIL import Image

from trustmark import TrustMark

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(REPO, 'models', 'manifest.json')
DEFAULT_IMAGES = [os.path.join(REPO, 'images', name) for name in ('ghost.png', 'ripley.jpg', 'ufo_240.jpg')]
OPSET = 17
INPUT_NAMES = {'encoder': ['onnx::Concat_0', 'onnx::Gemm_1'], 'decoder': ['image']}
OUTPUT_NAMES = {'encoder': ['image'], 'decoder': ['output']}
DTYPES = {TensorProto.FLOAT: 'float32', TensorProto.FLOAT16: 'float16', TensorProto.INT64: 'int64'}


def file_digests(path):
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            md5.update(chunk)
            sha256.update(chunk)
    return {'size': os.path.getsize(path), 'md5': md5.hexdigest(), 'sha256': sha256.hexdigest()}


def describe_onnx(model):
    # Same fields as the 'onnx' object of a manifest artifact entry
    graph = model.graph
    initializers = {i.name for i in graph.initializer}

    def signature(values):
        return [{'name': v.name, 'dtype': DTYPES[v.type.tensor_type.elem_type],
                 'shape': [d.dim_value if d.HasField('dim_value') else d.dim_param for d in v.type.tensor_type.shape.dim]}
                for v in values if v.name not in initializers]

    inputs = {v.name for v in graph.input} - initializers
    input_casts = [a.i for n in graph.node if n.op_type == 'Cast' and n.input[0] in inputs for a in n.attribute if a.name == 'to']
    weight_types = sorted({DTYPES[i.data_type] for i in graph.initializer})
    return {
        'opset': [o.version for o in model.opset_import if o.domain in ('', 'ai.onnx')][0],
        'ir_version': model.ir_version,
        'producer': ('%s %s' % (model.producer_name, model.producer_version)).strip(),
        'weights_dtype': '/'.join(weight_types) if weight_types else None,
        'compute_dtype': 'float16' if input_casts and all(t == TensorProto.FLOAT16 for t in input_casts) else 'float32',
        'inputs': signature(graph.input),
        'outputs': signature(graph.output),
    }


def strip_semantics(values):
    return [{k: v[k] for k in ('name', 'dtype', 'shape')} for v in values]


def prepare(module, precision):
    module = copy.deepcopy(module).eval()
    if precision == 'float16':
        module = module.half()
        forward = module.forward
        # Casting inside the root module's forward yields the published graph's /Cast nodes
        module.forward = lambda *inputs: forward(*(x.half() for x in inputs)).float()
    return module


def export(module, example_inputs, path, role):
    options = dict(opset_version=OPSET, input_names=INPUT_NAMES[role], output_names=OUTPUT_NAMES[role], do_constant_folding=True)
    if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
        options['dynamo'] = False  # the published files come from the TorchScript-based exporter
    with warnings.catch_warnings(), torch.no_grad():
        # The decoder's resize to 224 depends on the input size, which is fixed at export time as intended
        warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
        torch.onnx.export(module, example_inputs, path, **options)


def add_metadata(path, properties):
    model = onnx.load(path)
    for key, value in properties.items():
        entry = model.metadata_props.add()
        entry.key = 'trustmark.' + key
        entry.value = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    onnx.save(model, path)


def compare_graphs(new, reference):
    initializers = [{i.name: numpy_helper.to_array(i) for i in m.graph.initializer} for m in (new, reference)]
    same_names = set(initializers[0]) == set(initializers[1])
    result = {
        'graph_bytes_identical': new.graph.SerializeToString() == reference.graph.SerializeToString(),
        'nodes_identical': [(n.op_type, n.name, list(n.input), list(n.output)) for n in new.graph.node]
                           == [(n.op_type, n.name, list(n.input), list(n.output)) for n in reference.graph.node],
        'initializer_names_identical': same_names,
    }
    if same_names:
        a, b = initializers
        comparable = [k for k in a if a[k].shape == b[k].shape and a[k].dtype == b[k].dtype]
        result.update({
            'initializers_identical': '%d/%d' % (sum(np.array_equal(a[k], b[k]) for k in comparable), len(a)),
            'weight_elements_differing': '%d/%d' % (sum(int((a[k] != b[k]).sum()) for k in comparable), sum(a[k].size for k in a)),
            'weight_max_abs_diff': max([float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max()) for k in comparable if a[k].size] or [0.0]),
        })
    return result


def image_tensor(tm, image, size):
    # python/ preprocessing: crop rule, PIL bilinear resize, [-1, 1]
    image = tm.get_the_image_for_processing(image.convert('RGB')).resize((size, size), Image.BILINEAR)
    return (np.asarray(image, dtype=np.float32) / 255.0 * 2.0 - 1.0).transpose(2, 0, 1)[None].copy()


class OnnxEncoder:
    # Stands in for the PyTorch encoder inside TrustMark.encode()
    device = 'cpu'

    def __init__(self, session):
        self.session = session

    def __call__(self, cover, secret):
        stego = self.session.run(['image'], {'onnx::Concat_0': cover.numpy().astype(np.float32),
                                             'onnx::Gemm_1': secret.numpy().astype(np.float32)})[0]
        stego = torch.from_numpy(stego)
        return stego, stego - cover


def encode(tm, encoder, image, payload):
    saved = tm.encoder
    if encoder is not None:
        tm.encoder = encoder
    try:
        return tm.encode(image, payload, MODE='binary')
    finally:
        tm.encoder = saved


def decode(tm, session, image):
    if session is None:
        return tm.decode(image, MODE='binary')
    x = image_tensor(tm, image, session.get_inputs()[0].shape[2])
    return tm.ecc.decode_bitstream(session.run(['output'], {'image': x})[0] > 0, 'binary')[0]


def psnr(a, b):
    mse = np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2)
    return float(10 * np.log10(255.0 ** 2 / mse)) if mse else float('inf')


def max_abs(a, b):
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


def export_variant(variant, args, manifest):
    report = {'variant': variant, 'precision': args.precision, 'files': {}}
    failures = []
    artifacts = manifest['artifacts']
    profile = dict(manifest['preprocessing']['common'], **manifest['preprocessing']['variants'][variant])
    sizes = {'encoder': profile['encoder_input_size'], 'decoder': profile['decoder_input_size']['onnx']}
    suffix = '' if args.precision == 'float16' else '_fp32'

    tm = TrustMark(model_type=variant, loadRemover=False, device='cpu', verbose=False)
    info = tm.model_info()
    modules = {'encoder': tm.encoder.encoder, 'decoder': tm.decoder.decoder}
    examples = {'encoder': (torch.zeros(1, 3, sizes['encoder'], sizes['encoder']), torch.zeros(1, 100)),
                'decoder': (torch.zeros(1, 3, sizes['decoder'], sizes['decoder']),)}
    sessions = {}
    for role in ('encoder', 'decoder'):
        published = '%s_%s.onnx' % (role, variant)
        path = os.path.join(args.out, '%s_%s%s.onnx' % (role, variant, suffix))
        sources = [dict({'name': name}, **{k: info['files'][name][k] for k in ('md5', 'sha256')})
                   for name in ('trustmark_%s.yaml' % variant, '%s_%s.ckpt' % (role, variant))]
        for source in sources:
            if source['sha256'] != artifacts[source['name']]['sha256']:
                failures.append('%s: source %s does not match models/manifest.json' % (published, source['name']))
        export(prepare(modules[role], args.precision), examples[role], path, role)
        metadata = {
            'model_family': info['model_family'],
            'variant': variant,
            'role': role,
            'precision': args.precision,
            'sources': sources,
            'preprocessing': dict(profile, input_size=sizes[role]),
            'exporter': {'tool': 'python/tools/export_onnx.py', 'torch': torch.__version__, 'opset': OPSET},
            'manifest_version': str(manifest['manifest_version']),
        }
        add_metadata(path, metadata)
        model = onnx.load(path)
        entry = dict(format='onnx', model_family=metadata['model_family'], variant=variant, role=role,
                     **file_digests(path), onnx=describe_onnx(model), precision=args.precision,
                     sources=sources, exporter=metadata['exporter'])
        with open(path + '.json', 'w', encoding='utf-8') as f:
            json.dump(entry, f, indent=2)
            f.write('\n')

        # (a) signature
        expected = artifacts[published]['onnx']
        checks = {
            'inputs_match_manifest': entry['onnx']['inputs'] == strip_semantics(expected['inputs']),
            'outputs_match_manifest': entry['onnx']['outputs'] == strip_semantics(expected['outputs']),
            'opset_matches_manifest': entry['onnx']['opset'] == expected['opset'],
        }
        if args.precision == 'float16':
            checks['dtypes_match_manifest'] = all(entry['onnx'][k] == expected[k] for k in ('weights_dtype', 'compute_dtype'))
        failures += ['%s: %s is false' % (os.path.basename(path), k) for k, ok in checks.items() if not ok]
        file_report = {'path': path, 'size': entry['size'], 'sha256': entry['sha256'], 'signature': checks}
        sessions[role] = {'new': ort.InferenceSession(path, providers=['CPUExecutionProvider'])}
        if args.reference:
            reference = os.path.join(args.reference, published)
            reference_digests = file_digests(reference)
            file_report['reference'] = {'path': reference, 'size': reference_digests['size'],
                                        'is_published_file': reference_digests['sha256'] == artifacts[published]['sha256']}
            if args.precision == 'float16':
                file_report['reference'].update(compare_graphs(model, onnx.load(reference)))
            sessions[role]['published'] = ort.InferenceSession(reference, providers=['CPUExecutionProvider'])
        report['files'][published] = file_report
        print('%s: exported %s' % (published, path), flush=True)

    report['numerics'] = compare_outputs(tm, modules, sessions, sizes, args)
    report['end_to_end'] = end_to_end(tm, sessions, args)
    failures += ['%s: payload not recovered with %s encoder and %s decoder' % (case[4], case[0], case[1])
                 for case in report['end_to_end'] if not case[2]]
    report['failures'] = failures
    return report


def compare_outputs(tm, modules, sessions, sizes, args):
    # Max abs differences between the PyTorch module ('torch', float32 as used by python/),
    # the new export ('new') and the published file ('published'), plus decoder bit agreement
    rng = np.random.default_rng(args.seed)
    images = [Image.open(path).convert('RGB') for path in args.images]
    results = {}

    def run_all(role, feeds):
        outputs = {}
        with torch.no_grad():
            outputs['torch'] = modules[role](*[torch.from_numpy(x) for x in feeds]).numpy()
        names = INPUT_NAMES[role]
        for label, session in sessions[role].items():
            outputs[label] = session.run(None, dict(zip(names, feeds)))[0]
        return outputs

    def summarize(role, cases):
        summary = {}
        for a, b in (('torch', 'new'), ('torch', 'published'), ('new', 'published')):
            if b not in cases[0]:
                continue
            item = {'max_abs_diff': max(max_abs(c[a], c[b]) for c in cases)}
            if role == 'decoder':
                agree = sum(int(((c[a] > 0) == (c[b] > 0)).sum()) for c in cases)
                item['bit_agreement'] = '%d/%d' % (agree, sum(c[a].size for c in cases))
            summary['%s_vs_%s' % (a, b)] = item
        return summary

    def random_image(size):
        return rng.uniform(-1, 1, (1, 3, size, size)).astype(np.float32)

    def random_bits():
        return rng.integers(0, 2, (1, 100)).astype(np.float32)

    results['encoder'] = {
        'random_inputs': summarize('encoder', [run_all('encoder', [random_image(sizes['encoder']), random_bits()]) for _ in range(args.samples)]),
        'images': summarize('encoder', [run_all('encoder', [image_tensor(tm, im, sizes['encoder']), random_bits()]) for im in images]),
    }
    payloads = [''.join(rng.choice(['0', '1'], tm.schemaCapacity())) for _ in images]
    watermarked = [encode(tm, None, im, p) for im, p in zip(images, payloads)]
    results['decoder'] = {
        'random_inputs': summarize('decoder', [run_all('decoder', [random_image(sizes['decoder'])]) for _ in range(args.samples)]),
        'images': summarize('decoder', [run_all('decoder', [image_tensor(tm, im, sizes['decoder'])]) for im in images]),
        'watermarked_images': summarize('decoder', [run_all('decoder', [image_tensor(tm, im, sizes['decoder'])]) for im in watermarked]),
    }
    return results


def end_to_end(tm, sessions, args):
    # [encoder, decoder, payload recovered, PSNR of the watermarked image] for each image
    rng = np.random.default_rng(args.seed + 1)
    encoders = {'torch': None}
    decoders = {'torch': None}
    for label in sessions['encoder']:
        encoders[label] = OnnxEncoder(sessions['encoder'][label])
        decoders[label] = sessions['decoder'][label]
    cases = []
    for path in args.images:
        cover = Image.open(path).convert('RGB')
        payload = ''.join(rng.choice(['0', '1'], tm.schemaCapacity()))
        for encoder_label, encoder in encoders.items():
            stego = encode(tm, encoder, cover, payload)
            for decoder_label, decoder in decoders.items():
                if 'new' not in (encoder_label, decoder_label):
                    continue
                secret, present, _ = decode(tm, decoder, stego)
                cases.append([encoder_label, decoder_label, bool(present and secret == payload),
                              round(psnr(stego, cover), 2), os.path.basename(path)])
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', action='append', choices=['B', 'C', 'P', 'Q'], required=True,
                        help='variant to export; repeat for several')
    parser.add_argument('--out', required=True, help='output directory (existing exports are not overwritten)')
    parser.add_argument('--reference', help='directory holding the published encoder_<V>.onnx and decoder_<V>.onnx to compare with')
    parser.add_argument('--precision', choices=['float16', 'float32'], default='float16',
                        help='float16 reproduces the published files (default); float32 writes <name>_fp32.onnx')
    parser.add_argument('--images', nargs='+', default=DEFAULT_IMAGES, help='images used for the checks')
    parser.add_argument('--samples', type=int, default=4, help='number of random inputs per model')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    with open(MANIFEST, encoding='utf-8') as f:
        manifest = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    if args.reference and os.path.samefile(args.out, args.reference):
        parser.error('--out must differ from --reference')
    suffix = '' if args.precision == 'float16' else '_fp32'
    existing = [name for v in args.variant for name in ('encoder_%s%s.onnx' % (v, suffix), 'decoder_%s%s.onnx' % (v, suffix))
                if os.path.exists(os.path.join(args.out, name))]
    if existing:
        parser.error('refusing to overwrite %s in %s' % (', '.join(existing), args.out))

    reports = [export_variant(v, args, manifest) for v in args.variant]
    report_path = os.path.join(args.out, 'export_report%s.json' % suffix)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(reports, f, indent=2)
        f.write('\n')
    print(json.dumps(reports, indent=2))
    failures = [failure for report in reports for failure in report['failures']]
    for failure in failures:
        print('error: ' + failure)
    print('report written to %s; %d errors' % (report_path, len(failures)))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
