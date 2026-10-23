# Python side of crosscheck.test.mjs: decodes images and embeds payloads with the Python
# TrustMark implementation.
#
# Usage: python python_crosscheck.py <dir containing the trustmark package> <job.json>
# job.json: {"decode": [{"file": "f.png", "variant": "Q"}, ...],
#            "encode": [{"input": "f", "output": "f.png", "payload": "0101", "variant": "Q", "schema": "BCH_5"}, ...]}
# Prints {"decode": {file: {"detected", "payload", "schema"}}, "encode": [output, ...]} as JSON.
# TRUSTMARK_PY_MODEL_DIR, when set, is passed as model_dir (with downloads disabled) to Python
# versions that support it.
import inspect
import json
import os
import sys

sys.path.insert(0, sys.argv[1])
from PIL import Image  # noqa: E402
from trustmark import TrustMark  # noqa: E402

SCHEMAS = {'BCH_SUPER': 0, 'BCH_5': 1, 'BCH_4': 2, 'BCH_3': 3}
NAMES = {v: k for k, v in SCHEMAS.items()}
job = json.load(open(sys.argv[2]))
instances = {}


def trustmark(variant, schema):
    if (variant, schema) not in instances:
        kwargs = dict(verbose=False, model_type=variant, encoding_type=SCHEMAS[schema], loadRemover=False, device='cpu')
        model_dir = os.environ.get('TRUSTMARK_PY_MODEL_DIR')
        if model_dir and 'model_dir' in inspect.signature(TrustMark.__init__).parameters:
            kwargs.update(model_dir=model_dir, allow_download=False)
        instances[(variant, schema)] = TrustMark(**kwargs)
    return instances[(variant, schema)]


out = {'decode': {}, 'encode': []}
for item in job.get('decode', []):
    path = item['file']
    secret, detected, version = trustmark(item['variant'], 'BCH_5').decode(Image.open(path).convert('RGB'), MODE='binary')
    out['decode'][path] = {'detected': bool(detected), 'payload': secret if detected else None, 'schema': NAMES.get(version) if detected else None}
for item in job.get('encode', []):
    cover = Image.open(item['input'])
    alpha = cover.getchannel('A') if cover.mode == 'RGBA' else None
    stego = trustmark(item['variant'], item['schema']).encode(cover.convert('RGB'), item['payload'], MODE='binary')
    if alpha is not None:
        stego.putalpha(alpha)
    stego.save(item['output'])
    out['encode'].append(item['output'])
print(json.dumps(out))
