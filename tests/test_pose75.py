"""The shipped ISL model only works if the app builds its input exactly like training did.
These tests prove sanket/core/pose75.py is identical to the experiment's verified sanket_core.py
(which was checked bit-for-bit against the Colab training code), and that IslRecognizer loads the
model, maps its 262 classes to lexicon glosses, and respects the confidence threshold."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from sanket.core import pose75
from sanket.core.lexicon import default_lexicon
from sanket.core.recognizer import IslRecognizer, _gloss_key

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments" / "include_pose_baseline" / "sanket_core.py"
ISL = REPO / "models" / "isl"


def load_experiment():
    spec = importlib.util.spec_from_file_location("exp_core", EXP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def person(rng, t=40, hand=True):
    """Random but realistic 75-point clip in pixel units."""
    f = np.zeros((t, 75, 3), np.float32)
    f[:, :33, :2] = rng.uniform(200, 700, (t, 33, 2))
    f[:, 11, :2], f[:, 12, :2] = (700, 300), (500, 300)
    if hand:
        f[:, 54:75, :2] = rng.uniform(420, 560, (t, 21, 2))
        f[:, 54:75, 1] -= 150
    return f


def test_identical_to_verified_experiment_code():
    exp = load_experiment()
    rng = np.random.default_rng(0)
    for _ in range(20):
        clip = person(rng, t=int(rng.integers(20, 120)), hand=bool(rng.integers(0, 2)))
        clip[rng.random(len(clip)) < 0.2, 33:54] = 0  # hand A sometimes missing
        assert np.array_equal(pose75.build_input(list(clip)), exp.build_input(list(clip), 64))
    assert pose75.KEEP == exp.KEEP and pose75.D_IN == exp.D_IN == 136


def test_input_shape():
    x = pose75.build_input(list(person(np.random.default_rng(1))))
    assert x.shape == (1, 64, 136) and x.dtype == np.float32 and np.isfinite(x).all()


def test_segmenter_one_sign_and_rest():
    seg, out = pose75.Segmenter75(), []
    rest = np.zeros((75, 3), np.float32)
    rest[11, :2], rest[12, :2] = (380, 300), (260, 300)
    up = rest.copy()
    up[54:75, :2] = (250, 330)  # wrist above the raise line
    for f in [rest] * 10 + [up] * 30 + [rest] * 15 + [rest] * 50:
        r = seg.push(f)
        if r is not None:
            out.append(r)
    assert len(out) == 1 and 16 <= len(out[0]) <= 60


def test_overlay_points():
    f = np.zeros((75, 3), np.float32)
    f[54:75, :2] = (320, 240)
    o = pose75.overlay_points(f, 640, 480)
    assert o.shape == (42, 2) and np.isnan(o[:21]).all() and np.allclose(o[21:], 0.5)


def test_class_names_map_to_lexicon():
    lex = default_lexicon()
    assert _gloss_key("big large") == "biglarge" and _gloss_key("ex. monsoon") == "exmonsoon"
    if (ISL / "config.json").exists():
        classes = json.loads((ISL / "config.json").read_text())["classes"]
        assert all(lex.get(_gloss_key(c)) for c in classes)


def test_demo_mode_never_confident(tmp_path):
    rec = IslRecognizer(tmp_path)
    assert rec.demo_mode and not rec.predict(list(person(np.random.default_rng(2)))).accepted


@pytest.mark.skipif(not (ISL / "sanket_include.onnx").exists(), reason="ISL model not downloaded")
def test_real_model_runs_and_thresholds():
    rec = IslRecognizer(ISL)
    assert not rec.demo_mode and len(rec.classes) == 262
    p = rec.predict(list(person(np.random.default_rng(3))))
    assert len(p.top) == 3 and abs(sum(1 for _ in p.top) - 3) == 0
    assert p.accepted == (p.prob >= rec.threshold)
    assert IslRecognizer(ISL, threshold=1.01).predict(list(person(np.random.default_rng(4)))).accepted is False
