"""Metrics and summaries for the TrustMark benchmark.

Only numpy is needed here, so these functions can be tested without model weights.
"""

import math

import numpy as np

# Possible results of decoding a watermarked image, as recorded per transform.
#   exact            watermark found, schema and payload as embedded
#   false_accept     watermark reported as valid with the expected schema but a different payload
#   schema_mismatch  watermark reported as valid but with another schema (a wrong payload too)
#   not_detected     no valid watermark found
#   error            decode raised an exception
OUTCOMES = ('exact', 'false_accept', 'schema_mismatch', 'not_detected', 'error')


def _pair_as_float64(reference, test):
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(test, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError('images must have the same shape, got %s and %s' % (a.shape, b.shape))
    if a.size == 0:
        raise ValueError('images must not be empty')
    return a, b


def psnr(reference, test, data_range=255.0):
    """Peak signal-to-noise ratio in dB, computed in float64.

    Accepts PIL images or arrays of the same shape. Returns math.inf for identical images.
    """
    a, b = _pair_as_float64(reference, test)
    mse = np.mean(np.square(a - b))
    if mse == 0:
        return math.inf
    return 20 * math.log10(data_range) - 10 * math.log10(mse)


def _box_mean(x, size):
    # mean over every size x size window that lies fully inside x (a 'valid' box filter)
    c = np.cumsum(np.cumsum(x, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    s = c[size:, size:] - c[:-size, size:] - c[size:, :-size] + c[:-size, :-size]
    return s / (size * size)


def ssim(reference, test, data_range=255.0, win_size=7):
    """Mean structural similarity (Wang et al. 2004) over all channels, computed in float64.

    Uses a win_size x win_size uniform window, sample (N-1) covariance, K1=0.01, K2=0.03 and
    only windows that lie fully inside the image, averaged per channel then across channels,
    which corresponds to the defaults of skimage.metrics.structural_similarity with
    channel_axis=-1 and the given data_range.
    """
    a, b = _pair_as_float64(reference, test)
    if a.ndim == 2:
        a, b = a[..., None], b[..., None]
    if a.ndim != 3:
        raise ValueError('expected an HxW or HxWxC image, got shape %s' % (a.shape,))
    if min(a.shape[:2]) < win_size:
        raise ValueError('images must be at least %d pixels in each dimension for SSIM' % win_size)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    n = win_size * win_size
    cov_norm = n / (n - 1.0)
    scores = []
    for ch in range(a.shape[2]):
        x, y = a[..., ch], b[..., ch]
        mu_x, mu_y = _box_mean(x, win_size), _box_mean(y, win_size)
        var_x = cov_norm * (_box_mean(x * x, win_size) - mu_x * mu_x)
        var_y = cov_norm * (_box_mean(y * y, win_size) - mu_y * mu_y)
        cov_xy = cov_norm * (_box_mean(x * y, win_size) - mu_x * mu_y)
        s = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
        scores.append(s.mean())
    return float(np.mean(scores))


def classify(expected_payload, expected_schema, detected, payload, schema):
    """Classify one decode of a watermarked image into an OUTCOMES entry (except 'error')."""
    if not detected:
        return 'not_detected'
    if schema != expected_schema:
        return 'schema_mismatch'
    if payload != expected_payload:
        return 'false_accept'
    return 'exact'


def bit_errors(expected_payload, payload):
    """Number of differing bits, or None when the decoded payload has another length."""
    if payload is None or len(payload) != len(expected_payload):
        return None
    return sum(a != b for a, b in zip(expected_payload, payload))


def summarize_outcomes(outcomes):
    """Counts and rates of each outcome in OUTCOMES."""
    counts = dict.fromkeys(OUTCOMES, 0)
    for outcome in outcomes:
        if outcome not in counts:
            raise ValueError('unknown outcome %r' % (outcome,))
        counts[outcome] += 1
    n = sum(counts.values())
    rates = {k: (v / n if n else None) for k, v in counts.items()}
    return {'n': n, 'counts': counts, 'rates': rates}


def summarize_negatives(records):
    """False positives among decodes of images that carry no watermark.

    Each record is a dict with 'detected' and 'schema_name', or with 'error' if decode failed.
    """
    records = list(records)
    by_schema = {}
    for r in records:
        if r.get('detected'):
            by_schema[r['schema_name']] = by_schema.get(r['schema_name'], 0) + 1
    false_positives = sum(by_schema.values())
    return {'n': len(records), 'false_positives': false_positives,
            'false_positive_rate': (false_positives / len(records) if records else None),
            'false_positives_by_schema': by_schema, 'errors': sum(1 for r in records if 'error' in r)}


def value_stats(values):
    """n/mean/min/median/max of a list of numbers (inf propagates), or None when empty."""
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return None
    return {'n': int(arr.size), 'mean': float(np.mean(arr)), 'min': float(np.min(arr)),
            'median': float(np.median(arr)), 'max': float(np.max(arr))}


def timing_stats(samples):
    """n/min/mean/p50/p95/max of timing samples in seconds (linear percentile interpolation)."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return None
    return {'n': int(arr.size), 'min': float(np.min(arr)), 'mean': float(np.mean(arr)),
            'p50': float(np.percentile(arr, 50)), 'p95': float(np.percentile(arr, 95)),
            'max': float(np.max(arr))}


def json_safe(obj):
    """Copy of obj with non-finite floats replaced by 'inf', '-inf' or 'nan' (strict JSON)."""
    if isinstance(obj, float):
        if math.isnan(obj):
            return 'nan'
        if math.isinf(obj):
            return 'inf' if obj > 0 else '-inf'
        return obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj
