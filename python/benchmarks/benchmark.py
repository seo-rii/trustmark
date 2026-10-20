"""Reproducible quality, robustness and performance benchmark for TrustMark (Python).

For every image in a corpus this embeds a seeded random payload, applies a list of
transforms to the watermarked image, decodes each result and reports exact payload
recovery, false accepts, schema mismatches and misses, together with PSNR (and
optionally SSIM), false positives on the unwatermarked originals and timings.

Run from the python/ directory so the repository's trustmark package is used:

    python -m benchmarks.benchmark --corpus ../images --exclude '*_[PQ].png' --out results.json

See benchmarks/README.md for the options and the meaning of each number.
"""

import argparse
import fnmatch
import hashlib
import io
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
from PIL import Image, features

if __package__ in (None, ''):
    # run as a script (python benchmarks/benchmark.py): make the sibling packages importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import metrics  # noqa: E402

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
SCHEMAS = {'BCH_SUPER': 0, 'BCH_5': 1, 'BCH_4': 2, 'BCH_3': 3}
SCHEMA_NAMES = {v: k for k, v in SCHEMAS.items()}
DEFAULT_TRANSFORMS = 'none,png,jpeg:90,jpeg:75,jpeg:50,resize:0.5,resize:2.0,crop:0.8,rotate:90'

Transform = namedtuple('Transform', 'label kind param rotation_search')


def parse_transform(spec):
    """Parse 'none', 'png', 'jpeg:Q', 'resize:S', 'crop:F' or 'rotate:D' into a Transform."""
    kind, _, arg = spec.strip().partition(':')
    kind = kind.lower()
    try:
        if kind in ('none', 'png'):
            if arg:
                raise ValueError
            return Transform(kind, kind, None, False)
        if kind == 'jpeg':
            quality = int(arg)
            if not 1 <= quality <= 100:
                raise ValueError
            return Transform('jpeg:%d' % quality, kind, quality, False)
        if kind in ('resize', 'crop', 'rotate'):
            value = float(arg)
            if not math.isfinite(value) or (kind == 'resize' and value <= 0) or (kind == 'crop' and not 0 < value <= 1):
                raise ValueError
            label = '%s:%g' % (kind, value)
            # decode(ROTATION=True) only searches 90 degree steps
            rotation_search = kind == 'rotate' and value % 360 in (90, 180, 270)
            return Transform(label, kind, value, rotation_search)
    except ValueError:
        pass
    raise ValueError('invalid transform %r; expected none, png, jpeg:<1-100>, resize:<factor > 0>, '
                     'crop:<fraction in (0,1]> or rotate:<degrees>' % spec)


def apply_transform(image, transform):
    """Apply a Transform to an RGB PIL image and return a new RGB PIL image."""
    kind, param = transform.kind, transform.param
    if kind == 'none':
        return image
    if kind in ('png', 'jpeg'):
        buf = io.BytesIO()
        if kind == 'png':
            image.save(buf, format='PNG')
        else:
            image.save(buf, format='JPEG', quality=param)  # Pillow defaults, 4:2:0 subsampling
        buf.seek(0)
        out = Image.open(buf)
        out.load()
        return out.convert('RGB')
    w, h = image.size
    if kind == 'resize':
        size = (max(1, int(round(w * param))), max(1, int(round(h * param))))
        return image.resize(size, Image.Resampling.BICUBIC)
    if kind == 'crop':
        cw, ch = max(1, int(round(w * param))), max(1, int(round(h * param)))
        left, top = (w - cw) // 2, (h - ch) // 2
        return image.crop((left, top, left + cw, top + ch))
    if kind == 'rotate':
        angle = param % 360
        steps = {0: None, 90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180,
                 270: Image.Transpose.ROTATE_270}
        if angle in steps:
            return image if steps[angle] is None else image.transpose(steps[angle])
        # counter-clockwise, same size, uncovered corners black
        return image.rotate(angle, resample=Image.Resampling.BICUBIC)
    raise ValueError('unknown transform kind %r' % kind)


def find_images(corpus, exclude=(), max_images=None):
    """Image files under corpus (recursive), sorted by relative path, minus exclude globs."""
    corpus = Path(corpus)
    found = []
    for path in sorted(corpus.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(corpus).as_posix()
        if any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(path.name, g) for g in exclude):
            continue
        found.append((rel, path))
    return found[:max_images] if max_images else found


def make_payload(seed, key, nbits):
    """Seeded random bit string; the same seed and key always give the same payload."""
    rng = random.Random('%s/%s' % (seed, key))  # str seeds are hashed with SHA-512, stable across runs
    return ''.join(rng.choice('01') for _ in range(nbits))


def load_rgb(path):
    with Image.open(path) as im:
        mode = im.mode
        im.load()
        return im.convert('RGB'), mode


def file_md5(path):
    digest = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_info(directory):
    def git(*cmd):
        try:
            return subprocess.run(['git', '-C', str(directory)] + list(cmd), capture_output=True,
                                  text=True, check=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    commit = git('rev-parse', 'HEAD')
    status = git('status', '--porcelain', '--untracked-files=no')
    return {'commit': commit, 'dirty': (bool(status) if commit and status is not None else None)}


def loadavg():
    try:
        return list(os.getloadavg())
    except (AttributeError, OSError):
        return None


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def model_files(trustmark_module, variant):
    """md5 of the encoder/decoder files the given variant loads, checked against MODEL_CHECKSUMS."""
    models_dir = Path(trustmark_module.__file__).resolve().parent / 'models'
    info = {}
    for name in ('trustmark_%s.yaml' % variant, 'encoder_%s.ckpt' % variant, 'decoder_%s.ckpt' % variant):
        path = models_dir / name
        expected = trustmark_module.MODEL_CHECKSUMS.get(name)
        if path.is_file():
            md5 = file_md5(path)
            info[name] = {'md5': md5, 'bytes': path.stat().st_size, 'expected_md5': expected,
                          'matches_expected': md5 == expected}
        else:
            info[name] = {'md5': None, 'bytes': None, 'expected_md5': expected, 'matches_expected': False}
    return info


def decode_record(tm, image, transform, expected_payload, expected_schema, force_rotation):
    rotation = transform.rotation_search or force_rotation
    record = {'transform': transform.label, 'rotation_search': rotation, 'size': list(image.size)}
    try:
        (payload, detected, schema), record['decode_s'] = timed(tm.decode, image, MODE='binary', ROTATION=rotation)
    except Exception as e:  # keep benchmarking the rest of the corpus
        record.update(outcome='error', error='%s: %s' % (type(e).__name__, e))
        return record
    detected, schema = bool(detected), int(schema)
    record.update(detected=detected, schema=schema, schema_name=SCHEMA_NAMES.get(schema), payload=payload)
    if expected_payload is not None:
        record['outcome'] = metrics.classify(expected_payload, expected_schema, detected, payload, schema)
        record['bit_errors'] = metrics.bit_errors(expected_payload, payload) if schema == expected_schema else None
    return record


def run_variant(args, variant, schema, transforms, images):
    import trustmark.trustmark as tm_module
    from trustmark import TrustMark

    print('[%s] loading model' % variant, file=sys.stderr)
    tm, load_s = timed(TrustMark, verbose=False, model_type=variant, encoding_type=schema,
                       device=args.device, loadRemover=False)
    capacity = tm.schemaCapacity()

    # timing: first (cold) calls, then warm repeats on one image
    timing_rel, timing_path = (Path(args.timing_image).name, Path(args.timing_image)) if args.timing_image else images[0]
    timing_image, _ = load_rgb(timing_path)
    timing_payload = make_payload(args.seed, 'timing', capacity)
    stego, first_encode_s = timed(tm.encode, timing_image, timing_payload, MODE='binary', WM_STRENGTH=args.strength)
    _, first_decode_s = timed(tm.decode, stego, MODE='binary')
    warm_encode, warm_decode = [], []
    for _ in range(args.repeats):
        stego, t = timed(tm.encode, timing_image, timing_payload, MODE='binary', WM_STRENGTH=args.strength)
        warm_encode.append(t)
        _, t = timed(tm.decode, stego, MODE='binary')
        warm_decode.append(t)
    timings = {'model_load_s': load_s, 'timing_image': timing_rel, 'timing_image_size': list(timing_image.size),
               'first_encode_s': first_encode_s, 'first_decode_s': first_decode_s,
               'warm_encode': metrics.timing_stats(warm_encode), 'warm_decode': metrics.timing_stats(warm_decode)}

    per_image = []
    for rel, path in images:
        print('[%s] %s' % (variant, rel), file=sys.stderr)
        image, mode = load_rgb(path)
        payload = make_payload(args.seed, rel, capacity)
        entry = {'image': rel, 'mode': mode, 'size': list(image.size), 'payload': payload}
        try:
            stego, entry['encode_s'] = timed(tm.encode, image, payload, MODE='binary', WM_STRENGTH=args.strength)
        except Exception as e:
            entry['encode_error'] = '%s: %s' % (type(e).__name__, e)
            per_image.append(entry)
            continue
        entry['psnr'] = metrics.psnr(image, stego)
        if args.ssim:
            entry['ssim'] = metrics.ssim(image, stego)
        entry['decodes'] = [decode_record(tm, apply_transform(stego, t), t, payload, schema, args.rotation)
                            for t in transforms]
        if args.negatives:
            entry['negatives'] = [decode_record(tm, apply_transform(image, t), t, None, None, args.rotation)
                                  for t in transforms]
        per_image.append(entry)

    encoded = [e for e in per_image if 'encode_error' not in e]
    per_transform = {}
    for i, t in enumerate(transforms):
        summary = metrics.summarize_outcomes([e['decodes'][i]['outcome'] for e in encoded])
        if args.negatives:
            summary['negatives'] = metrics.summarize_negatives(e['negatives'][i] for e in encoded)
        per_transform[t.label] = summary
    summary = {
        'images': len(per_image),
        'encode_errors': len(per_image) - len(encoded),
        'psnr': metrics.value_stats(e['psnr'] for e in encoded),
        'ssim': metrics.value_stats(e['ssim'] for e in encoded) if args.ssim else None,
        'per_transform': per_transform,
        'negatives': (metrics.summarize_negatives(n for e in encoded for n in e['negatives'])
                      if args.negatives else None),
        'corpus_encode_s': metrics.timing_stats([e['encode_s'] for e in encoded]),
        'corpus_decode_s': metrics.timing_stats([d['decode_s'] for e in encoded for d in e['decodes'] if 'decode_s' in d]),
    }
    return {'variant': variant, 'schema': SCHEMA_NAMES[schema], 'schema_id': schema, 'capacity_bits': capacity,
            'device': str(tm.device), 'model_files': model_files(tm_module, variant),
            'timings': timings, 'summary': summary, 'images': per_image}


def environment():
    import torch
    import trustmark
    env = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'cpu_count': os.cpu_count(),
        'torch': torch.__version__,
        'torch_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(),
        'numpy': np.__version__,
        'pillow': Image.__version__,
        'libjpeg': features.version('jpg'),
        'trustmark_path': str(Path(trustmark.__file__).resolve().parent),
        'git': git_info(Path(__file__).resolve().parent),
    }
    try:
        import torchvision
        env['torchvision'] = torchvision.__version__
    except ImportError:
        pass
    return env


def format_summary(result):
    lines = []
    cfg = result['config']
    env = result['environment']
    lines.append('seed %s, strength %s, %d image(s), torch %s, %s thread(s), commit %s%s' % (
        cfg['seed'], cfg['strength'], len(result['runs'][0]['images']) if result['runs'] else 0, env['torch'],
        env['torch_threads'], (env['git']['commit'] or 'unknown')[:12], ' (dirty)' if env['git']['dirty'] else ''))
    for run in result['runs']:
        s, t = run['summary'], run['timings']
        lines.append('')
        lines.append('variant %s, schema %s (%d bits), device %s' % (run['variant'], run['schema'], run['capacity_bits'], run['device']))
        lines.append('  load %.2fs, first encode %.3fs, first decode %.3fs (%s %dx%d)' % (
            t['model_load_s'], t['first_encode_s'], t['first_decode_s'], t['timing_image'], *t['timing_image_size']))
        for name in ('warm_encode', 'warm_decode'):
            if t[name]:
                lines.append('  %s p50 %.3fs, p95 %.3fs (n=%d)' % (name.replace('_', ' '), t[name]['p50'], t[name]['p95'], t[name]['n']))
        if s['psnr']:
            lines.append('  PSNR mean %.2f dB, min %.2f dB' % (s['psnr']['mean'], s['psnr']['min']))
        if s['ssim']:
            lines.append('  SSIM mean %.4f, min %.4f' % (s['ssim']['mean'], s['ssim']['min']))
        if s['encode_errors']:
            lines.append('  encode errors: %d' % s['encode_errors'])
        lines.append('  %-12s %3s %6s %13s %16s %13s %6s %13s' % (
            'transform', 'n', 'exact', 'false_accept', 'schema_mismatch', 'not_detected', 'error', 'negative_fp'))
        for label, ts in s['per_transform'].items():
            c = ts['counts']
            neg = ts.get('negatives')
            lines.append('  %-12s %3d %6d %13d %16d %13d %6d %13s' % (
                label, ts['n'], c['exact'], c['false_accept'], c['schema_mismatch'], c['not_detected'], c['error'],
                '%d/%d' % (neg['false_positives'], neg['n']) if neg else '-'))
        neg = s['negatives']
        if neg:
            by_schema = ', '.join('%s: %d' % kv for kv in sorted(neg['false_positives_by_schema'].items()))
            lines.append('  negatives: %d/%d false positives%s' % (
                neg['false_positives'], neg['n'], ' (%s)' % by_schema if by_schema else ''))
    return '\n'.join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--corpus', required=True, help='directory of cover images (searched recursively)')
    p.add_argument('--exclude', action='append', default=[], metavar='GLOB',
                   help='skip images whose relative path or file name matches GLOB (repeatable)')
    p.add_argument('--max-images', type=int, default=None, help='use only the first N images (sorted by path)')
    p.add_argument('--variants', default='Q', help='comma-separated model variants (C, Q, B, P); default Q')
    p.add_argument('--schema', default='BCH_5', choices=sorted(SCHEMAS), help='ECC schema; default BCH_5')
    p.add_argument('--strength', type=float, default=1.0, help='WM_STRENGTH passed to encode; default 1.0')
    p.add_argument('--seed', type=int, default=0, help='seed for payloads and RNGs; default 0')
    p.add_argument('--transforms', default=DEFAULT_TRANSFORMS,
                   help='comma-separated transforms applied before decoding; default %(default)s')
    p.add_argument('--rotation', action='store_true',
                   help='decode every image with ROTATION=True (by default only rotate:90/180/270 do)')
    p.add_argument('--no-negatives', dest='negatives', action='store_false',
                   help='skip decoding the unwatermarked originals')
    p.add_argument('--ssim', action='store_true', help='also compute SSIM (slower on large images)')
    p.add_argument('--repeats', type=int, default=5, help='warm encode/decode repeats for timing; default 5')
    p.add_argument('--timing-image', default=None, help='image used for timing; default the first corpus image')
    p.add_argument('--device', default='cpu', help="torch device ('cpu', 'cuda', 'mps'); default cpu")
    p.add_argument('--threads', type=int, default=None, help='torch.set_num_threads value; default torch default')
    p.add_argument('--out', default=None, help='write the full JSON report to this path')
    args = p.parse_args(argv)
    args.variants = [v.strip().upper() for v in args.variants.split(',') if v.strip()]
    for v in args.variants:
        if v not in ('C', 'Q', 'B', 'P'):
            p.error('unknown variant %r' % v)
    if not math.isfinite(args.strength) or args.strength < 0:
        p.error('--strength must be a finite number >= 0')
    if args.repeats < 0:
        p.error('--repeats must be >= 0')
    try:
        args.transform_list = [parse_transform(s) for s in args.transforms.split(',') if s.strip()]
    except ValueError as e:
        p.error(str(e))
    if not args.transform_list:
        p.error('--transforms must list at least one transform')
    return args


def main(argv=None):
    args = parse_args(argv)
    images = find_images(args.corpus, args.exclude, args.max_images)
    if not images:
        sys.exit('no images found in %s' % args.corpus)

    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = environment()
    env['loadavg_start'] = loadavg()
    config = {k: v for k, v in vars(args).items() if k != 'transform_list'}
    config['transforms'] = [t.label for t in args.transform_list]
    config['images'] = [rel for rel, _ in images]
    runs = [run_variant(args, v, SCHEMAS[args.schema], args.transform_list, images) for v in args.variants]
    env['loadavg_end'] = loadavg()

    result = {'benchmark': 'trustmark-python', 'format_version': 1, 'config': config,
              'environment': env, 'runs': runs}
    print(format_summary(result))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(metrics.json_safe(result), f, indent=1)
            f.write('\n')
    return result


if __name__ == '__main__':
    main()
