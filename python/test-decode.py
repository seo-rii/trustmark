# Copyright 2025 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


from trustmark import TrustMark, load_image
from PIL import Image
from pathlib import Path
import math,random
import numpy as np


#EXAMPLE_FILE = '../images/ufo_240.jpg'     # JPEG example
#EXAMPLE_FILE = '../images/ghost.png'        # PNG RGBA example
EXAMPLE_FILE = '../images/ripley.jpg'     # JPEG example

# Available modes: Q=balance, P=high visual quality, C=compact decoder, B=base from paper
MODE='P'
DETECTFIRST=False  # False = decode full image watermark, True = detect watermarked region before decoding
tm=TrustMark(verbose=True, model_type=MODE, encoding_type=TrustMark.Encoding.BCH_5,  loadBBoxDetector=DETECTFIRST)

# encoding example
cover, metadata = load_image(EXAMPLE_FILE)  # upright per EXIF orientation, plus the metadata to keep
rgb=cover.convert('RGB')

random.seed(1234)
capacity=tm.schemaCapacity()
bitstring=''.join([random.choice(['0', '1']) for _ in range(capacity)])
encoded=tm.encode(cover, bitstring, MODE='binary')  # RGBA covers keep their alpha channel

outfile=Path(EXAMPLE_FILE).stem+'_'+MODE+'.png'
encoded.save(outfile, **metadata)


# decoding example
stego = Image.open(outfile).convert('RGB')
wm_secret, wm_present, wm_schema = tm.decode(stego, MODE='binary', DETECTFIRST=DETECTFIRST, ROTATION=False)
if wm_present:
  print(f'Extracted secret: {wm_secret} (schema {wm_schema})')
else:
  print('No valid watermark decoded')

# psnr (quality, higher is better)
mse = np.mean(np.square(np.subtract(np.asarray(stego).astype(np.int16), np.asarray(rgb).astype(np.int16))))
if mse > 0:
  PIXEL_MAX = 255.0
  psnr= 20 * math.log10(PIXEL_MAX) - 10 * math.log10(mse)
  print('PSNR = %f' % psnr)

# removal
stego, metadata = load_image(outfile)
im_recover = tm.remove_watermark(stego)
rm_wm_secret, rm_wm_present, rm_wm_schema = tm.decode(im_recover, MODE='binary', DETECTFIRST=DETECTFIRST)
if rm_wm_present and rm_wm_schema==wm_schema:
  print(f'Extracted secret: {rm_wm_secret} (schema {rm_wm_schema})')
else:
   print('No valid secret decoded after removal')
im_recover.save('recovered.png', **metadata)

