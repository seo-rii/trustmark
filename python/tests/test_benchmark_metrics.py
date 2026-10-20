"""Weight-free checks of the benchmark metrics, transforms and corpus helpers."""

import json
import math

import numpy as np
import pytest
from PIL import Image

from benchmarks import benchmark, metrics


def uniform_pair(diff, shape=(8, 8, 3)):
    return np.zeros(shape, dtype=np.uint8), np.full(shape, diff, dtype=np.uint8)


def test_psnr_identical_images_is_inf():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, size=(16, 12, 3), dtype=np.uint8)
    assert metrics.psnr(a, a.copy()) == math.inf
    assert metrics.psnr(*uniform_pair(0)) == math.inf


@pytest.mark.parametrize('diff', [1, 181, 182, 255])
def test_psnr_uniform_difference_matches_closed_form(diff):
    # 182 and above overflow when squared in int16; the metric must not
    expected = 20 * math.log10(255.0) - 10 * math.log10(diff * diff)
    a, b = uniform_pair(diff)
    assert metrics.psnr(a, b) == pytest.approx(expected, abs=1e-12)
    assert metrics.psnr(b, a) == pytest.approx(expected, abs=1e-12)


def test_psnr_full_range_difference_is_zero_db():
    a, b = uniform_pair(255)
    assert metrics.psnr(a, b) == pytest.approx(0.0, abs=1e-12)


def test_psnr_accepts_pil_images():
    a = Image.new('RGB', (10, 6), (10, 20, 30))
    b = Image.new('RGB', (10, 6), (11, 20, 30))
    # one channel in three differs by 1 -> mse 1/3
    assert metrics.psnr(a, b) == pytest.approx(20 * math.log10(255.0) - 10 * math.log10(1 / 3))


@pytest.mark.parametrize('a,b', [
    (np.zeros((4, 4, 3)), np.zeros((4, 5, 3))),
    (np.zeros((4, 4, 3)), np.zeros((4, 4))),
    (np.zeros((0, 4, 3)), np.zeros((0, 4, 3))),
])
def test_psnr_rejects_mismatched_or_empty_images(a, b):
    with pytest.raises(ValueError):
        metrics.psnr(a, b)


def naive_ssim_channel(x, y, win=7, data_range=255.0):
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    scores = []
    for i in range(x.shape[0] - win + 1):
        for j in range(x.shape[1] - win + 1):
            px = x[i:i + win, j:j + win].ravel()
            py = y[i:i + win, j:j + win].ravel()
            mx, my = px.mean(), py.mean()
            vx, vy = px.var(ddof=1), py.var(ddof=1)
            cxy = np.cov(px, py, ddof=1)[0, 1]
            scores.append(((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2)))
    return np.mean(scores)


def test_ssim_matches_windowed_definition():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, size=(13, 11, 3)).astype(np.uint8)
    b = np.clip(a.astype(np.int16) + rng.integers(-40, 41, size=a.shape), 0, 255).astype(np.uint8)
    expected = np.mean([naive_ssim_channel(a[..., c].astype(np.float64), b[..., c].astype(np.float64)) for c in range(3)])
    assert metrics.ssim(a, b) == pytest.approx(expected, rel=1e-9)
    assert metrics.ssim(a[..., 0], b[..., 0]) == pytest.approx(
        naive_ssim_channel(a[..., 0].astype(np.float64), b[..., 0].astype(np.float64)), rel=1e-9)


def test_ssim_identical_is_one_and_symmetric():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
    b = rng.integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
    assert metrics.ssim(a, a) == pytest.approx(1.0)
    assert metrics.ssim(a, b) == pytest.approx(metrics.ssim(b, a))
    assert metrics.ssim(a, b) < 0.5


def test_ssim_rejects_images_smaller_than_window():
    with pytest.raises(ValueError, match='at least 7'):
        metrics.ssim(np.zeros((6, 20, 3)), np.zeros((6, 20, 3)))


@pytest.mark.parametrize('detected,payload,schema,outcome', [
    (True, '1010', 1, 'exact'),
    (True, '1011', 1, 'false_accept'),
    (True, '1010', 2, 'schema_mismatch'),
    (True, '10101', 3, 'schema_mismatch'),
    (False, '', -1, 'not_detected'),
    (False, '1010', 1, 'not_detected'),
])
def test_classify(detected, payload, schema, outcome):
    assert metrics.classify('1010', 1, detected, payload, schema) == outcome


def test_bit_errors():
    assert metrics.bit_errors('1010', '1010') == 0
    assert metrics.bit_errors('1010', '0101') == 4
    assert metrics.bit_errors('1010', '1011') == 1
    assert metrics.bit_errors('1010', '101') is None
    assert metrics.bit_errors('1010', None) is None


def test_summarize_outcomes():
    s = metrics.summarize_outcomes(['exact', 'exact', 'false_accept', 'not_detected'])
    assert s['n'] == 4
    assert s['counts'] == {'exact': 2, 'false_accept': 1, 'schema_mismatch': 0, 'not_detected': 1, 'error': 0}
    assert s['rates']['exact'] == 0.5
    assert s['rates']['schema_mismatch'] == 0.0
    assert metrics.summarize_outcomes([])['rates']['exact'] is None
    with pytest.raises(ValueError):
        metrics.summarize_outcomes(['exact', 'maybe'])


def test_summarize_negatives():
    records = [
        {'detected': False, 'schema_name': None},
        {'detected': True, 'schema_name': 'BCH_3'},
        {'detected': True, 'schema_name': 'BCH_3'},
        {'detected': True, 'schema_name': 'BCH_5'},
        {'error': 'RuntimeError: boom'},
    ]
    assert metrics.summarize_negatives(records) == {
        'n': 5, 'false_positives': 3, 'false_positive_rate': 0.6,
        'false_positives_by_schema': {'BCH_3': 2, 'BCH_5': 1}, 'errors': 1}
    assert metrics.summarize_negatives([]) == {
        'n': 0, 'false_positives': 0, 'false_positive_rate': None, 'false_positives_by_schema': {}, 'errors': 0}


def test_timing_and_value_stats():
    t = metrics.timing_stats([0.5, 0.1, 0.2, 0.3, 0.4])
    assert t['n'] == 5
    assert t['min'] == 0.1 and t['max'] == 0.5
    assert t['p50'] == pytest.approx(0.3)
    assert t['p95'] == pytest.approx(0.48)  # linear interpolation between 0.4 and 0.5
    assert metrics.timing_stats([]) is None

    v = metrics.value_stats([40.0, math.inf, 50.0, None])
    assert v['n'] == 3 and v['min'] == 40.0 and v['mean'] == math.inf and v['median'] == 50.0
    assert metrics.value_stats([]) is None


def test_json_safe_produces_strict_json():
    data = {'psnr': math.inf, 'neg': -math.inf, 'nan': math.nan, 'x': [np.float64(1.5), (2, math.inf)], 's': 'ok'}
    safe = metrics.json_safe(data)
    assert safe == {'psnr': 'inf', 'neg': '-inf', 'nan': 'nan', 'x': [1.5, [2, 'inf']], 's': 'ok'}
    json.dumps(safe, allow_nan=False)


@pytest.mark.parametrize('spec,label,rotation_search', [
    ('none', 'none', False),
    ('PNG', 'png', False),
    ('jpeg:75', 'jpeg:75', False),
    ('resize:0.5', 'resize:0.5', False),
    ('crop:1', 'crop:1', False),
    ('rotate:90', 'rotate:90', True),
    ('rotate:-90', 'rotate:-90', True),
    ('rotate:5', 'rotate:5', False),
    ('rotate:360', 'rotate:360', False),
])
def test_parse_transform(spec, label, rotation_search):
    t = benchmark.parse_transform(spec)
    assert t.label == label
    assert t.rotation_search == rotation_search


@pytest.mark.parametrize('spec', ['', 'none:1', 'jpeg', 'jpeg:0', 'jpeg:101', 'jpeg:x', 'resize:0', 'resize:-2',
                                  'crop:0', 'crop:1.5', 'rotate:nan', 'blur:2'])
def test_parse_transform_rejects_invalid_specs(spec):
    with pytest.raises(ValueError, match='invalid transform'):
        benchmark.parse_transform(spec)


def rgb_image(w, h, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8))


def test_apply_transform_geometry_and_codecs():
    im = rgb_image(40, 30)
    apply = lambda spec: benchmark.apply_transform(im, benchmark.parse_transform(spec))
    assert apply('none') is im
    png = apply('png')
    assert png.mode == 'RGB' and np.array_equal(np.asarray(png), np.asarray(im))
    jpeg = apply('jpeg:50')
    assert jpeg.mode == 'RGB' and jpeg.size == im.size and not np.array_equal(np.asarray(jpeg), np.asarray(im))
    assert apply('resize:0.5').size == (20, 15)
    assert apply('resize:2').size == (80, 60)
    crop = apply('crop:0.5')
    assert crop.size == (20, 15)
    assert np.array_equal(np.asarray(crop), np.asarray(im)[7:22, 10:30])
    assert np.array_equal(np.asarray(apply('rotate:90')), np.rot90(np.asarray(im)))
    assert np.array_equal(np.asarray(apply('rotate:360')), np.asarray(im))
    assert apply('rotate:5').size == im.size


def test_make_payload_is_seeded():
    a = benchmark.make_payload(7, 'img.png', 61)
    assert len(a) == 61 and set(a) <= {'0', '1'}
    assert a == benchmark.make_payload(7, 'img.png', 61)
    assert a != benchmark.make_payload(8, 'img.png', 61)
    assert a != benchmark.make_payload(7, 'other.png', 61)


def test_find_images_sorts_filters_and_excludes(tmp_path):
    (tmp_path / 'sub').mkdir()
    for name in ['b.png', 'a.JPG', 'sub/c.webp', 'a_Q.png', 'notes.txt']:
        (tmp_path / name).write_bytes(b'')
    found = [rel for rel, _ in benchmark.find_images(tmp_path, exclude=['*_Q.png'])]
    assert found == ['a.JPG', 'b.png', 'sub/c.webp']
    assert [rel for rel, _ in benchmark.find_images(tmp_path, max_images=2)] == ['a.JPG', 'a_Q.png']


@pytest.mark.parametrize('argv', [
    ['--strength', 'nan'], ['--strength', '-1'], ['--variants', 'X'], ['--transforms', 'jpeg:0'], ['--repeats', '-1'],
])
def test_parse_args_rejects_invalid_options(argv):
    with pytest.raises(SystemExit):
        benchmark.parse_args(['--corpus', '.'] + argv)
