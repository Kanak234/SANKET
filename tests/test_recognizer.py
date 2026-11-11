"""Recognizer: loads an ONNX model, returns top-3, honours the confidence threshold, and falls
back to DEMO mode (never confident) when the model file is missing."""

import numpy as np
import pytest

from sanket.core.recognizer import Recognizer, softmax
from tests.synthetic import clip


@pytest.fixture(scope="module")
def tiny_onnx(tmp_path_factory):
    torch = pytest.importorskip("torch")
    from training.model import SignTransformer

    path = tmp_path_factory.mktemp("m") / "r.onnx"
    model = SignTransformer(n_classes=263, d_model=32, layers=1, heads=2, ff=64).eval()
    torch.onnx.export(model, (torch.zeros(1, 64, 239),), str(path), input_names=["landmarks"],
                      output_names=["logits"], dynamic_axes={"landmarks": {0: "b"}, "logits": {0: "b"}},
                      opset_version=17, dynamo=False)
    return path


def test_predict_shapes(tiny_onnx):
    rec = Recognizer(tiny_onnx, threshold=0.0)
    p = rec.predict(clip(0, np.random.default_rng(0)))
    assert not rec.demo_mode
    assert len(p.top) == 3 and p.top[0][1] >= p.top[1][1] >= p.top[2][1]
    assert p.accepted  # threshold 0 accepts everything


def test_threshold_rejects(tiny_onnx):
    rec = Recognizer(tiny_onnx, threshold=0.99)
    assert not rec.predict(clip(1, np.random.default_rng(1))).accepted


def test_demo_mode(tmp_path):
    rec = Recognizer(tmp_path / "missing.onnx")
    assert rec.demo_mode
    assert not rec.predict(clip(0, np.random.default_rng(0))).accepted  # never fakes a result


def test_softmax_stable():
    p = softmax(np.array([[1000.0, 1000.0, -1000.0]]))
    assert np.isclose(p.sum(), 1) and np.isfinite(p).all()
