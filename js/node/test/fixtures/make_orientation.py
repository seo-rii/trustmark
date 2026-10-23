# Regenerates orientation.json: tiny PNGs carrying each EXIF orientation (1 to 8) in an eXIf
# chunk, the pixels Pillow's ImageOps.exif_transpose shows for them, and a 16x8 JPEG (left half
# red, right half blue) with orientation 6.
#   python make_orientation.py > orientation.json
import base64, io, json, sys

import numpy as np
from PIL import Image, ImageOps

W, H = 3, 2
a = np.zeros((H, W, 3), np.uint8)
for y in range(H):
    for x in range(W):
        a[y, x] = [x * 40 + 10, y * 100 + 20, (y * W + x) * 30 + 5]
src = Image.fromarray(a, 'RGB')
out = {}
for o in range(1, 9):
    exif = Image.Exif()
    exif[0x0112] = o
    buf = io.BytesIO()
    src.save(buf, 'PNG', exif=exif.tobytes())
    shown = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(buf.getvalue()))).convert('RGB'))
    out[o] = {'png': base64.b64encode(buf.getvalue()).decode(), 'width': shown.shape[1], 'height': shown.shape[0], 'rgb': shown.reshape(-1).tolist()}
j = np.zeros((8, 16, 3), np.uint8)
j[:, :8] = [250, 10, 10]
j[:, 8:] = [10, 10, 250]
exif = Image.Exif()
exif[0x0112] = 6
buf = io.BytesIO()
Image.fromarray(j).save(buf, 'JPEG', quality=95, exif=exif.tobytes())
out['jpeg6'] = base64.b64encode(buf.getvalue()).decode()
json.dump(out, sys.stdout)
