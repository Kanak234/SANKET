"""workers.py — background threads so the window never freezes.

WHY: MediaPipe (~25 ms/frame), the recognizer and Vosk speech recognition are slow compared with
     a 60 Hz screen. Running them on the UI thread would make buttons lag and video stutter.
WHAT:
  • CameraWorker (QThread): webcam → MediaPipe points → Segmenter → Recognizer. Emits the
    mirrored preview image, the live points (for the overlay), "signing now" state, each
    Prediction, per-stage timings (F12 debug overlay) and a low-light hint.
  • SpeechJob (QRunnable): after push-to-talk is released, turns audio into text and glosses
    off the UI thread, then emits the result.
  • Raw camera frames and audio never leave memory (PRD FR-C2).
NEXT: ConversationView and ContributeView connect to these signals.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThread, Signal
from PySide6.QtGui import QImage

from sanket.core.grammar import text_to_glosses
from sanket.core.recognizer import Recognizer
from sanket.core.segmenter import Segmenter
from sanket.core.stt import SpeechToText

log = logging.getLogger(__name__)


class CameraWorker(QThread):
    frameReady = Signal(QImage, object)  # mirrored preview image, (79,3) points of that frame
    signingChanged = Signal(bool)  # True while a sign is being collected
    predicted = Signal(object)  # Prediction
    segmentReady = Signal(object)  # raw (T,79,3) segment (ContributeView saves these)
    timings = Signal(dict)  # {"mediapipe_ms": .., "recognizer_ms": .., "fps": ..}
    hint = Signal(str)  # "low_light" etc.
    failed = Signal(str)  # camera could not be opened

    def __init__(self, recognizer: Recognizer, camera_index: int = 0, recognize: bool = True) -> None:
        super().__init__()
        self.recognizer = recognizer
        self.camera_index = camera_index
        self.recognize = recognize  # False in Contribute mode: only collect segments
        self.segmenter = Segmenter()
        self._running = False
        self._flush = False

    def stop(self) -> None:
        self._running = False
        self.wait(3000)

    def flush_sign(self) -> None:
        """Enter key: close the current sign now (handled on the worker thread)."""
        self._flush = True

    def run(self) -> None:
        import cv2

        from sanket.core.landmarks import HolisticExtractor

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
        if not cap.isOpened():
            self.failed.emit("no_camera")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        try:
            extractor = HolisticExtractor()
        except FileNotFoundError as exc:
            self.failed.emit(str(exc))
            cap.release()
            return
        self._running = True
        t0 = time.monotonic()
        last_fps_t, frames_since = t0, 0
        no_hand_since = None
        was_signing = False
        while self._running:
            ok, bgr = cap.read()
            if not ok:
                self.failed.emit("no_camera")
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ts = int((time.monotonic() - t0) * 1000)
            a = time.perf_counter()
            pts = extractor.process(rgb, ts)  # UN-mirrored frame → correct left/right hands
            mp_ms = (time.perf_counter() - a) * 1000

            # Preview: mirrored like a selfie camera, so the signer sees a natural reflection.
            view = np.ascontiguousarray(rgb[:, ::-1])
            h, w = view.shape[:2]
            img = QImage(view.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.frameReady.emit(img, pts)

            # Low-light hint: person visible but no hand found for 3 s.
            person = np.isfinite(pts[42:59, 0]).any()
            hands = np.isfinite(pts[:42, 0]).any()
            if person and not hands and float(rgb.mean()) < 60:
                no_hand_since = no_hand_since or time.monotonic()
                if time.monotonic() - no_hand_since > 3:
                    self.hint.emit("low_light")
                    no_hand_since = None
            else:
                no_hand_since = None

            seg = self.segmenter.push(pts)
            if self._flush:
                self._flush = False
                seg = seg if seg is not None else self.segmenter.flush()
            if self.segmenter.active != was_signing:
                was_signing = self.segmenter.active
                self.signingChanged.emit(was_signing)
            rec_ms = 0.0
            if seg is not None:
                self.segmentReady.emit(seg)
                if self.recognize:
                    b = time.perf_counter()
                    self.predicted.emit(self.recognizer.predict(seg))
                    rec_ms = (time.perf_counter() - b) * 1000
            frames_since += 1
            now = time.monotonic()
            if now - last_fps_t >= 1.0:
                self.timings.emit({"mediapipe_ms": round(mp_ms, 1), "recognizer_ms": round(rec_ms, 1),
                                   "fps": round(frames_since / (now - last_fps_t), 1)})
                last_fps_t, frames_since = now, 0
        extractor.close()
        cap.release()


class SpeechSignals(QObject):
    done = Signal(str, list)  # text, glosses
    failed = Signal(str)


class SpeechJob(QRunnable):
    """Runs SpeechToText.stop() (which does the recognition) on a thread-pool thread."""

    def __init__(self, stt: SpeechToText) -> None:
        super().__init__()
        self.stt = stt
        self.signals = SpeechSignals()

    def run(self) -> None:
        try:
            text = self.stt.stop()
            self.signals.done.emit(text, text_to_glosses(text) if text else [])
        except Exception as exc:  # microphone unplugged mid-recording, etc.
            log.exception("speech recognition failed")
            self.signals.failed.emit(str(exc))
