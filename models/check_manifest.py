# Copyright 2026 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

"""Check models/manifest.json against the repository and, optionally, against model files.

Without options, checks the manifest's own consistency, that the MODEL_CHECKSUMS tables of
python/ and python-hailo/ agree with it, and the digests of artifacts stored in the repository.

  --dir DIR   also check the size and digests of every artifact present in DIR
  --remote    also download every artifact (streamed, nothing is saved) and check it

Uses only the Python standard library. Exits with status 1 if any check fails.
"""

import argparse
import ast
import fnmatch
import hashlib
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, 'models', 'manifest.json')
PIN_FILES = ['python/trustmark/trustmark.py', 'python-hailo/trustmark_hailo/trustmark_hailo.py']
FORMATS = {'pytorch-ckpt', 'yaml', 'onnx', 'hailo-hef', 'npz'}
ROLES = {'encoder', 'decoder', 'remover', 'detector', 'resizer', 'config'}


def pinned_checksums(path):
    # MODEL_CHECKSUMS entries of a source file, read without importing it
    pins = {}
    for node in ast.walk(ast.parse(open(path, encoding='utf-8').read())):
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        if isinstance(target, ast.Subscript) and getattr(target.value, 'id', None) == 'MODEL_CHECKSUMS':
            pins[ast.literal_eval(target.slice)] = ast.literal_eval(node.value)
        elif isinstance(target, ast.Name) and target.id == 'MODEL_CHECKSUMS' and isinstance(node.value, ast.Dict):
            pins.update(ast.literal_eval(node.value))
    return pins


def digest_stream(stream):
    size, md5, sha256 = 0, hashlib.md5(), hashlib.sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b''):
        size += len(chunk)
        md5.update(chunk)
        sha256.update(chunk)
    return {'size': size, 'md5': md5.hexdigest(), 'sha256': sha256.hexdigest()}


def compare(name, entry, actual):
    return ['%s: %s is %s, manifest says %s' % (name, key, actual[key], entry[key])
            for key in ('size', 'md5', 'sha256') if entry.get(key) is not None and entry[key] != actual[key]]


def check_consistency(manifest):
    errors = []
    families, variants = manifest['families'], manifest['variants']
    for name, entry in manifest['artifacts'].items():
        if entry['format'] not in FORMATS:
            errors.append('%s: unknown format %r' % (name, entry['format']))
        if entry['role'] not in ROLES:
            errors.append('%s: unknown role %r' % (name, entry['role']))
        if entry['model_family'] is not None and entry['model_family'] not in families:
            errors.append('%s: unknown model_family %r' % (name, entry['model_family']))
        if entry['variant'] is not None and entry['variant'] not in variants:
            errors.append('%s: unknown variant %r' % (name, entry['variant']))
        if 'url' in entry and entry['url'] != manifest['base_url'] + name:
            errors.append('%s: url %s is not under base_url' % (name, entry['url']))
        if 'url' not in entry and 'repo_path' not in entry:
            errors.append('%s: neither url nor repo_path' % name)
        if len(entry['md5']) != 32 or (entry['sha256'] is not None and len(entry['sha256']) != 64):
            errors.append('%s: malformed digest' % name)
        if entry['sha256'] is None and 'download_status' not in entry:
            errors.append('%s: sha256 missing without a download_status' % name)
    for pair in manifest['compatibility']['pairs']:
        for side in ('encoder', 'decoder'):
            if pair[side]['model_family'] not in families or pair[side]['variant'] not in variants:
                errors.append('compatibility pair %r refers to an unknown family or variant' % pair)
    return errors


def check_repository(manifest):
    errors = []
    artifacts = manifest['artifacts']
    pinned_in = {}
    for pin_file in PIN_FILES:
        path = os.path.join(ROOT, pin_file)
        if not os.path.isfile(path):
            continue
        for name, md5 in pinned_checksums(path).items():
            pinned_in.setdefault(name, []).append(pin_file)
            if name not in artifacts:
                errors.append('%s pins %s, which is not in the manifest' % (pin_file, name))
            elif artifacts[name]['md5'] != md5:
                errors.append('%s pins md5 %s for %s, manifest says %s' % (pin_file, md5, name, artifacts[name]['md5']))
    for name, entry in artifacts.items():
        if sorted(entry.get('md5_pinned_in', [])) != sorted(pinned_in.get(name, [])):
            errors.append('%s: md5_pinned_in is %s, but it is pinned in %s' % (name, entry.get('md5_pinned_in', []), pinned_in.get(name, [])))
        if 'repo_path' in entry:
            path = os.path.join(ROOT, entry['repo_path'])
            if not os.path.isfile(path):
                errors.append('%s: %s does not exist' % (name, entry['repo_path']))
                continue
            with open(path, 'rb') as f:
                errors += compare(name, entry, digest_stream(f))
    return errors


def check_files(manifest, directory, patterns):
    errors, checked = [], 0
    for name, entry in manifest['artifacts'].items():
        path = os.path.join(directory, name)
        if not os.path.isfile(path) or not any(fnmatch.fnmatch(name, p) for p in patterns):
            continue
        with open(path, 'rb') as f:
            errors += compare(name, entry, digest_stream(f))
        checked += 1
    print('checked %d files in %s' % (checked, directory))
    return errors


def check_remote(manifest, patterns):
    errors = []
    for name, entry in manifest['artifacts'].items():
        if 'url' not in entry or not any(fnmatch.fnmatch(name, p) for p in patterns):
            continue
        try:
            with urllib.request.urlopen(entry['url'], timeout=60) as response:
                problems = compare(name, entry, digest_stream(response))
        except OSError as e:
            problems = ['%s: could not download %s: %s' % (name, entry['url'], e)]
        print('%s %s' % ('FAIL' if problems else 'ok  ', name), flush=True)
        errors += problems
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dir', action='append', default=[], help='directory holding downloaded model files')
    parser.add_argument('--remote', action='store_true', help='download every artifact from the CDN and check it')
    parser.add_argument('--only', action='append', default=[], help='with --dir/--remote, check only names matching this glob')
    args = parser.parse_args()

    with open(MANIFEST, encoding='utf-8') as f:
        manifest = json.load(f)
    patterns = args.only or ['*']
    errors = check_consistency(manifest) + check_repository(manifest)
    for directory in args.dir:
        errors += check_files(manifest, directory, patterns)
    if args.remote:
        errors += check_remote(manifest, patterns)

    for error in errors:
        print('error: ' + error)
    print('%d artifacts, %d errors' % (len(manifest['artifacts']), len(errors)))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
