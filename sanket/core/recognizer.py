"""recognizer.py — runs SANKET's own sign-recognition model on one sign segment.

WHY: this is the "brain" of direction A. The model (recognizer.onnx) was trained FROM SCRATCH
     by training/train.py on INCLUDE landmarks; here we only run it. ONNX Runtime is used so the
     same file runs on Windows now and on Android later, and the app never needs PyTorch.
WHAT: Recognizer.predict(segment) → Prediction(gloss, prob, top3, accepted)
      `accepted` is False when the model is not confident enough (PRD FR-A5): the UI then says
      "Samajh nahi aaya — dobara karein" instead of guessing.
NEXT: accepted glosses are collected into a sentence → grammar.glosses_to_sentence().
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sanket.core.lexicon import Lexicon, default_lexicon
from sanket.core.normalize import FEATURES, SEQ_LEN, model_input
from sanket.paths import resource

log = logging.getLogger(__name__)

MODEL_PATH = resource("models", "recognizer.onnx")
META_PATH = resource("models", "recognizer.json")  # written by training/evaluate.py
DEFAULT_THRESHOLD = 0.60


@dataclass
class Prediction:
    """What the UI needs to show one recognised sign as a chip."""

    gloss: str  # best guess, e.g. "hello"
    prob: float  # its probability 0..1
    top: list[tuple[str, float]]  # top-3 alternatives (user can tap to correct)
    accepted: bool  # prob >= threshold


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax (subtract max before exp so large logits don't overflow)."""
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class Recognizer:
    """Loads recognizer.onnx once; predict() is called once per segmented sign (~10 ms on CPU)."""

    def __init__(
        self,
        model_path: Path | str = MODEL_PATH,
        lexicon: Lexicon | None = None,
        threshold: float | None = None,
    ) -> None:
        self.lexicon = lexicon or default_lexicon()
        self.model_path = Path(model_path)
        self.demo_mode = not self.model_path.exists()
        self.threshold = threshold if threshold is not None else self._load_threshold()
        if self.demo_mode:
            # WHY: lets the UI, tests and self-test run before the model is trained. In demo
            # mode every prediction is below threshold, so SANKET never shows a fake result.
            log.warning("recognizer.onnx not found at %s — running in DEMO mode (no recognition)", model_path)
            self._session = None
            return
        import onnxruntime as ort  # imported lazily: tests of other modules stay fast

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2  # leave CPU for MediaPipe and the UI
        self._session = ort.InferenceSession(str(self.model_path), opts, providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name
        n_out = self._session.get_outputs()[0].shape[-1]
        if isinstance(n_out, int) and n_out != len(self.lexicon.labels):
            raise ValueError(f"model has {n_out} classes but lexicon has {len(self.lexicon.labels)}")

    def _load_threshold(self) -> float:
        """Use the threshold calibrated on the validation set, if training produced one."""
        try:
            return float(json.loads(Path(META_PATH).read_text())["threshold"])
        except (OSError, KeyError, ValueError):
            return DEFAULT_THRESHOLD

    def predict_features(self, x: np.ndarray) -> np.ndarray:
        """(B, 64, 239) features → (B, n_classes) probabilities."""
        x = np.asarray(x, dtype=np.float32)
        assert x.shape[1:] == (SEQ_LEN, FEATURES), x.shape
        if self._session is None:  # demo mode → uniform = never confident
            return np.full((len(x), len(self.lexicon.labels)), 1.0 / len(self.lexicon.labels), np.float32)
        logits = self._session.run(None, {self._input: x})[0]
        return softmax(logits)

    def predict(self, segment: np.ndarray, k: int = 3) -> Prediction:
        """One raw segment (T, 79, 3) from the segmenter → Prediction."""
        probs = self.predict_features(model_input(segment)[None])[0]
        order = np.argsort(-probs)[:k]
        top = [(self.lexicon.labels[i], float(probs[i])) for i in order]
        best, p = top[0]
        return Prediction(gloss=best, prob=p, top=top, accepted=p >= self.threshold)


# ---------------------------------------------------------------------------------------------
# The SHIPPED recognizer: own BiGRU model (2.14 M params, random init, trained from scratch on
# INCLUDE pose keypoints, test top-1 93.8 %). Files in models/isl/: sanket_include.onnx + config.json.
# ---------------------------------------------------------------------------------------------
ISL_DIR = resource("models", "isl")
ISL_THRESHOLD = 0.45  # same default the experiment's live demo was tuned with


def _gloss_key(name: str) -> str:
    """'big large' → 'biglarge', 'ex. monsoon' → 'exmonsoon' — the lexicon's gloss keys."""
    return "".join(ch for ch in name.lower() if "a" <= ch <= "z")


class IslRecognizer:
    """Same interface as Recognizer (predict(segment) → Prediction, demo_mode, threshold),
    but takes a list of (75, 3) pose frames from pose75.Segmenter75.

    WHY a separate class: different input layout (64, 136) and its own class list (262 words —
    INCLUDE's 'second (number)' is not in this model). config.json is checked against the code's
    layout so a mismatched model file fails loudly instead of answering wrongly.
    """

    def __init__(self, model_dir: Path | str = ISL_DIR, lexicon: Lexicon | None = None,
                 threshold: float | None = None) -> None:
        from sanket.core.pose75 import D_IN, KEEP

        self.lexicon = lexicon or default_lexicon()
        self.threshold = ISL_THRESHOLD if threshold is None else threshold
        model_dir = Path(model_dir)
        onnx = sorted(model_dir.glob("sanket_*.onnx")) if model_dir.exists() else []
        cfg_path = model_dir / "config.json"
        self.demo_mode = not (onnx and cfg_path.exists())
        if self.demo_mode:
            log.warning("ISL model not found in %s — DEMO mode (no recognition)", model_dir)
            self._session, self.classes, self.T = None, list(self.lexicon.labels), 64
            return
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg.get("d_in", D_IN) != D_IN or list(cfg.get("keep", KEEP)) != KEEP:
            raise ValueError("models/isl/config.json does not match the code's feature layout")
        self.T = int(cfg["T"])
        self.test_top1 = cfg.get("test_top1")
        # Map model class names to lexicon glosses (needed for Hindi/English text and the avatar).
        self.classes = [_gloss_key(c) for c in cfg["classes"]]
        unknown = [c for c in self.classes if self.lexicon.get(c) is None]
        if unknown:
            raise ValueError(f"model classes missing from lexicon.json: {unknown[:5]}")
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2  # leave CPU for MediaPipe and the UI
        self._session = ort.InferenceSession(str(onnx[0]), opts, providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    def predict(self, frames75, k: int = 3) -> Prediction:
        """Segment (list of (75, 3) frames) → Prediction. Demo mode is never confident."""
        from sanket.core.pose75 import build_input

        if self._session is None:
            g = self.classes[0]
            return Prediction(gloss=g, prob=0.0, top=[(g, 0.0)], accepted=False)
        logits = self._session.run(None, {self._input: build_input(frames75, self.T)})[0]
        probs = softmax(logits)[0]
        order = np.argsort(-probs)[:k]
        top = [(self.classes[i], float(probs[i])) for i in order]
        best, p = top[0]
        return Prediction(gloss=best, prob=p, top=top, accepted=p >= self.threshold)


def load_recognizer(threshold: float | None = None):
    """The app's recognizer: the shipped ISL pose model if present, else the Holistic one (or demo)."""
    if (ISL_DIR / "config.json").exists():
        return IslRecognizer(threshold=threshold)
    return Recognizer(threshold=threshold)
