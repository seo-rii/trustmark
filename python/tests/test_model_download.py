# Copyright 2023 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

# Regression tests for TrustMark.check_and_download. They use a fake model file
# name with a patched checksum and a stubbed urllib.request.urlopen, so they need
# neither the network nor the real model weights.

import io
import os
import urllib.request
from hashlib import md5

import pytest

import trustmark.trustmark as tm_module
from trustmark.trustmark import TrustMark

NAME = 'decoder_TEST.ckpt'
GOOD = b'verified model bytes' * 1000
URL = tm_module.MODEL_REMOTE_HOST + NAME


class FakeResponse(io.BytesIO):
    def info(self):
        return {}


class InterruptedResponse(FakeResponse):
    def read(self, *args):
        if self.tell() > 0:
            raise ConnectionResetError('connection reset by peer')
        return super().read(len(GOOD) // 2)


@pytest.fixture
def network(monkeypatch):
    # Replaces urllib.request.urlopen; tests choose what it serves. Any
    # unexpected network use fails the test instead of reaching the CDN.
    calls = []
    state = {'serve': None}

    def fake_urlopen(url, data=None, timeout=None):
        calls.append((url, timeout))
        if state['serve'] is None:
            raise AssertionError('unexpected network access: ' + url)
        return state['serve']()

    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
    monkeypatch.setitem(tm_module.MODEL_CHECKSUMS, NAME, md5(GOOD).hexdigest())
    return calls, state


def tm():
    # check_and_download does not depend on loaded models; skip __init__
    return TrustMark.__new__(TrustMark)


def test_valid_existing_file_is_not_downloaded(tmp_path, network):
    calls, _ = network
    target = tmp_path / NAME
    target.write_bytes(GOOD)

    tm().check_and_download(str(target))

    assert calls == []
    assert target.read_bytes() == GOOD


def test_corrupt_existing_file_is_redownloaded_and_replaced(tmp_path, network):
    calls, state = network
    state['serve'] = lambda: FakeResponse(GOOD)
    target = tmp_path / NAME
    target.write_bytes(b'truncated')

    tm().check_and_download(str(target))

    assert target.read_bytes() == GOOD
    assert os.listdir(tmp_path) == [NAME]
    assert len(calls) == 1
    url, timeout = calls[0]
    assert url == URL
    assert timeout is not None and timeout > 0


def test_missing_file_is_downloaded(tmp_path, network):
    calls, state = network
    state['serve'] = lambda: FakeResponse(GOOD)
    target = tmp_path / NAME

    tm().check_and_download(str(target))

    assert target.read_bytes() == GOOD
    assert os.listdir(tmp_path) == [NAME]
    assert [url for url, _ in calls] == [URL]


def test_download_with_wrong_checksum_raises_and_leaves_no_file(tmp_path, network):
    _, state = network
    state['serve'] = lambda: FakeResponse(b'<html>not a model</html>')
    target = tmp_path / NAME

    with pytest.raises(OSError) as excinfo:
        tm().check_and_download(str(target))

    message = str(excinfo.value)
    assert URL in message
    assert md5(GOOD).hexdigest() in message
    assert md5(b'<html>not a model</html>').hexdigest() in message
    assert os.listdir(tmp_path) == []


def test_wrong_checksum_does_not_clobber_existing_file(tmp_path, network):
    _, state = network
    state['serve'] = lambda: FakeResponse(b'still wrong')
    target = tmp_path / NAME
    target.write_bytes(b'old corrupt copy')

    with pytest.raises(OSError):
        tm().check_and_download(str(target))

    # the unverified download never replaces what was there
    assert target.read_bytes() == b'old corrupt copy'
    assert os.listdir(tmp_path) == [NAME]


def test_interrupted_download_raises_and_leaves_no_file(tmp_path, network):
    _, state = network
    state['serve'] = lambda: InterruptedResponse(GOOD)
    target = tmp_path / NAME

    with pytest.raises(OSError) as excinfo:
        tm().check_and_download(str(target))

    # no truncated file under the final name and no temporary file left behind
    assert os.listdir(tmp_path) == []
    assert URL in str(excinfo.value)


def test_unknown_filename_raises_without_download(tmp_path, network):
    calls, _ = network
    target = tmp_path / 'decoder_UNKNOWN.ckpt'

    with pytest.raises(ValueError) as excinfo:
        tm().check_and_download(str(target))

    assert 'decoder_UNKNOWN.ckpt' in str(excinfo.value)
    assert calls == []
    assert os.listdir(tmp_path) == []
