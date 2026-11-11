"""Features must not change when the signer moves or stands closer — that is the whole point
of normalize.py. These tests prove it and check the fixed (64, 239) contract."""

import numpy as np

from sanket.core.normalize import FEATURES, SEQ_LEN, model_input, normalize_sequence, resample
from tests.synthetic import clip


def test_shape_and_no_nan():
    x = model_input(clip(0, np.random.default_rng(1)))
    assert x.shape == (SEQ_LEN, FEATURES)
    assert np.isfinite(x).all()


def test_shift_invariant():
    f = clip(1, np.random.default_rng(2))
    g = f.copy()
    g[..., 0] += 0.2  # signer stands further right
    g[..., 1] -= 0.1
    assert np.allclose(model_input(f), model_input(g), atol=1e-5)


def test_scale_invariant():
    f = clip(2, np.random.default_rng(3))
    c = np.array([0.5, 0.45, 0.0], np.float32)
    g = (f - c) * 1.7 + c  # signer stands closer to the camera
    assert np.allclose(model_input(f), model_input(g), atol=1e-4)


def test_hand_flags():
    f = clip(0, np.random.default_rng(4))  # kind 0 = right hand only
    x = normalize_sequence(f)
    assert x[:, -2].max() == 0.0  # left hand never seen
    assert x[:, -1].min() == 1.0  # right hand always seen


def test_resample_edges():
    assert resample(np.zeros((0, FEATURES))).shape == (SEQ_LEN, FEATURES)
    assert resample(np.ones((1, FEATURES))).shape == (SEQ_LEN, FEATURES)
    seq = np.arange(10, dtype=np.float32)[:, None].repeat(FEATURES, 1)
    out = resample(seq)
    assert out[0, 0] == 0 and out[-1, 0] == 9


def test_missing_shoulders_does_not_crash():
    f = clip(0, np.random.default_rng(5))
    f[:, 42:59] = np.nan
    assert np.isfinite(model_input(f)).all()
