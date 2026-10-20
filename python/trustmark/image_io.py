"""Image input/output helpers shared by the TrustMark methods."""

from PIL import Image, ImageOps

# 8-bit modes accepted by TrustMark.encode, decode and remove_watermark
SUPPORTED_MODES = ('RGB', 'RGBA', 'LA', 'L', 'P', 'CMYK')

EXIF_ORIENTATION = 0x0112


def split_alpha(image):
    """Split a supported PIL image into its RGB content and its alpha band.

    Returns (rgb, alpha): rgb is an 'RGB' image (the input itself when it is already RGB);
    alpha is the 'L' alpha band of RGBA, LA and palette images with transparency, else None.
    Raises TypeError for non-PIL input and ValueError for unsupported modes and empty images.
    """
    if not isinstance(image, Image.Image):
        raise TypeError('expected a PIL.Image.Image, got %s' % type(image).__name__)
    mode = image.mode
    if mode not in SUPPORTED_MODES:
        if mode in ('I', 'F') or mode.startswith('I;16'):
            raise ValueError("image mode '%s' has more than 8 bits per channel, which is not supported; "
                             "reduce the image to 8 bits per channel first (16-bit PNG support is proposed in "
                             "adobe/trustmark#36)" % mode)
        raise ValueError("unsupported image mode '%s'; use one of %s, e.g. image.convert('RGB')"
                         % (mode, ', '.join(SUPPORTED_MODES)))
    if image.width < 1 or image.height < 1:
        raise ValueError('image has no pixels (size %dx%d)' % image.size)
    if mode == 'RGB':
        return image, None
    if mode == 'P' and 'transparency' in image.info:
        image = image.convert('RGBA')  # palette transparency becomes an alpha band
        mode = 'RGBA'
    alpha = image.getchannel('A') if mode in ('RGBA', 'LA') else None
    return image.convert('RGB'), alpha


def load_image(path):
    """Open an image file for TrustMark with its EXIF orientation applied.

    Returns (image, metadata). image is the first frame, turned upright with
    ImageOps.exif_transpose and kept in the file's mode (TrustMark accepts the modes in
    SUPPORTED_MODES). metadata holds Image.save keyword arguments that carry the file's
    metadata over to the watermarked output, e.g. ``watermarked.save(out, **metadata)``:

    - 'exif': the EXIF data without the orientation tag, since the pixels are now upright
    - 'icc_profile': the ICC profile, only when it describes an RGB colour space, because
      encode returns RGB pixels (grayscale and CMYK profiles are dropped)
    - 'dpi': the resolution

    Keys are omitted when the file has no such metadata.
    """
    with Image.open(path) as im:
        image = ImageOps.exif_transpose(im)
        if image is im:  # older Pillow returns the input unchanged; detach it from the file
            image = im.copy()
    metadata = {}
    exif = image.getexif()
    exif.pop(EXIF_ORIENTATION, None)
    if len(exif):
        metadata['exif'] = exif.tobytes()
    icc_profile = image.info.get('icc_profile')
    if icc_profile and icc_profile[16:20] == b'RGB ':  # colour space field of the ICC header
        metadata['icc_profile'] = icc_profile
    if 'dpi' in image.info:
        metadata['dpi'] = image.info['dpi']
    return image, metadata
