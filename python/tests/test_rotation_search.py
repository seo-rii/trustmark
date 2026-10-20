import pytest
from PIL import Image

from trustmark import TrustMark

FULL_BOX = [0.0, 0.0, 1.0, 1.0]
NOT_DETECTED = ('', False, -1)


def _make_image():
    # non-square with a marker pixel so every orientation is distinguishable
    img = Image.new('RGB', (40, 20))
    img.putpixel((0, 0), (255, 0, 0))
    return img


def _rotations(img):
    return {0: img,
            90: img.transpose(Image.ROTATE_90),
            180: img.transpose(Image.ROTATE_180),
            270: img.transpose(Image.ROTATE_270)}


def _stub_trustmark(image, boxes_by_angle=None, detected_angles=(), detector=True):
    # TrustMark without weights; localize/subimage_decode are stubbed and report
    # which orientation of `image` they were called with
    rotations = _rotations(image)

    def angle_of(img):
        for angle, rotated in rotations.items():
            if img.size == rotated.size and img.tobytes() == rotated.tobytes():
                return angle
        raise AssertionError('unexpected image passed to stub')

    calls = {'localize': [], 'decode': []}

    def localize(img, return_all=False):
        assert return_all
        angle = angle_of(img)
        calls['localize'].append(angle)
        return (boxes_by_angle or {}).get(angle)

    def subimage_decode(img, MODE):
        angle = angle_of(img)
        calls['decode'].append(angle)
        if angle in detected_angles:
            return 'secret-%d' % angle, True, 1
        return NOT_DETECTED

    tm = object.__new__(TrustMark)
    tm.detector = object() if detector else None
    tm.concentrate_wm_region = 1.0
    tm.aspect_ratio_lim = 2.0
    if detector:
        tm.localize = localize
    tm.subimage_decode = subimage_decode
    return tm, calls


def test_detectfirst_rotation_tries_next_orientation_when_no_boxes():
    image = _make_image()
    tm, calls = _stub_trustmark(image, boxes_by_angle={0: None, 90: [FULL_BOX]}, detected_angles={90})

    assert tm.decode(image, MODE='binary', DETECTFIRST=True, ROTATION=True) == ('secret-90', True, 1)
    assert calls['localize'] == [0, 90]
    assert calls['decode'] == [90]


@pytest.mark.parametrize('target', [180, 270])
def test_detectfirst_rotation_finds_late_orientation(target):
    image = _make_image()
    boxes = {0: None, 90: [FULL_BOX], 180: None if target == 270 else [FULL_BOX], 270: [FULL_BOX]}
    tm, calls = _stub_trustmark(image, boxes_by_angle=boxes, detected_angles={target})

    assert tm.decode(image, MODE='binary', DETECTFIRST=True, ROTATION=True) == ('secret-%d' % target, True, 1)
    assert calls['localize'] == [a for a in (0, 90, 180, 270) if a <= target]
    assert calls['decode'] == [a for a in (90, 180, 270) if a <= target and boxes[a]]


def test_detectfirst_rotation_exhausts_all_orientations_and_boxes():
    image = _make_image()
    boxes = {0: None, 90: [FULL_BOX, FULL_BOX], 180: None, 270: [FULL_BOX]}
    tm, calls = _stub_trustmark(image, boxes_by_angle=boxes)

    assert tm.decode(image, MODE='binary', DETECTFIRST=True, ROTATION=True) == NOT_DETECTED
    assert calls['localize'] == [0, 90, 180, 270]
    assert calls['decode'] == [90, 90, 270]


@pytest.mark.parametrize('rotation', [False, True])
def test_detectfirst_without_detector_raises(rotation):
    image = _make_image()
    tm, calls = _stub_trustmark(image, detected_angles={0}, detector=False)

    with pytest.raises(RuntimeError, match='BBox detector model is not loaded'):
        tm.decode(image, MODE='binary', DETECTFIRST=True, ROTATION=rotation)
    assert calls['decode'] == []


@pytest.mark.parametrize('model_type', ['B', 'C'])
def test_bbox_detector_unavailable_for_model_type_raises_before_loading(monkeypatch, model_type):
    loaded = []
    monkeypatch.setattr(TrustMark, 'load_model', lambda self, *args, **kwargs: loaded.append(kwargs.get('part')))

    with pytest.raises(ValueError, match=r'\b%s\b' % model_type):
        TrustMark(verbose=False, device='cpu', model_type=model_type, loadRemover=False, loadBBoxDetector=True)
    assert loaded == []


@pytest.mark.parametrize('model_type', ['Q', 'P'])
def test_bbox_detector_loads_for_q_and_p(monkeypatch, model_type):
    loaded = []
    monkeypatch.setattr(TrustMark, 'load_model', lambda self, *args, **kwargs: loaded.append(kwargs.get('part')) or object())

    tm = TrustMark(verbose=False, device='cpu', model_type=model_type, loadRemover=False, loadBBoxDetector=True)
    assert loaded == ['decoder', 'encoder', 'detector']
    assert tm.detector is not None


@pytest.mark.parametrize('model_type', ['B', 'C'])
def test_b_and_c_without_bbox_detector_still_construct(monkeypatch, model_type):
    monkeypatch.setattr(TrustMark, 'load_model', lambda self, *args, **kwargs: object())
    tm = TrustMark(verbose=False, device='cpu', model_type=model_type, loadRemover=False)
    assert tm.detector is None


# unchanged paths

def test_detectfirst_without_rotation_no_boxes_returns_not_detected():
    image = _make_image()
    tm, calls = _stub_trustmark(image, boxes_by_angle={0: None, 90: [FULL_BOX]}, detected_angles={90})

    assert tm.decode(image, MODE='binary', DETECTFIRST=True, ROTATION=False) == NOT_DETECTED
    assert calls['localize'] == [0]
    assert calls['decode'] == []


def test_detectfirst_without_rotation_decodes_box():
    image = _make_image()
    tm, calls = _stub_trustmark(image, boxes_by_angle={0: [FULL_BOX]}, detected_angles={0})

    assert tm.decode(image, MODE='binary', DETECTFIRST=True, ROTATION=False) == ('secret-0', True, 1)
    assert calls['localize'] == [0]
    assert calls['decode'] == [0]


@pytest.mark.parametrize('detector', [False, True])
def test_whole_image_rotation_search(detector):
    image = _make_image()
    tm, calls = _stub_trustmark(image, detected_angles={180}, detector=detector)

    assert tm.decode(image, MODE='binary', DETECTFIRST=False, ROTATION=True) == ('secret-180', True, 1)
    assert calls['localize'] == []
    assert calls['decode'] == [0, 90, 180]


@pytest.mark.parametrize('detector', [False, True])
def test_whole_image_without_rotation(detector):
    image = _make_image()
    tm, calls = _stub_trustmark(image, detected_angles={90}, detector=detector)

    assert tm.decode(image, MODE='binary', DETECTFIRST=False, ROTATION=False) == NOT_DETECTED
    assert calls['localize'] == []
    assert calls['decode'] == [0]
