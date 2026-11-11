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
