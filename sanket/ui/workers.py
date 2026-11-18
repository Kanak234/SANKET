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
from sanket.core.recognizer import IslRecognizer, Recognizer
from sanket.core.segmenter import Segmenter
from sanket.core.stt import SpeechToText

log = logging.getLogger(__name__)


class CameraWorker(QThread):
    frameReady = Signal(QImage, object)  # mirrored preview image, (42,2) hand overlay points (0-1, NaN=missing)
    signingChanged = Signal(bool)  # True while a sign is being collected
    predicted = Signal(object)  # Prediction
    segmentReady = Signal(object)  # raw (T,79,3) segment (ContributeView saves these)
    timings = Signal(dict)  # {"mediapipe_ms": .., "recognizer_ms": .., "fps": ..}
    hint = Signal(str)  # "low_light" etc.
    failed = Signal(str)  # camera could not be opened

    def __init__(self, recognizer: Recognizer | IslRecognizer, camera_index: int = 0, recognize: bool = True) -> None:
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

    # ------------------------------------------------------------------ camera opening
    # 18 Nov FIX (friend ke laptop par "camera kaam nahi kar raha"):
    #   WHY: pehle sirf ek try tha — cv2.VideoCapture(index, CAP_DSHOW). Kai laptops
    #        (Intel/IR webcams, Windows Hello cameras) DirectShow par khulte hi nahi ya
    #        pehle 10-20 frame khaali dete hain. Ek fail = turant "no_camera" = app bekaar.
    #   WHAT: backends ki list (DSHOW -> MSMF -> ANY) x camera index (settings wala pehle,
    #        phir 0..3) — jo pehla asli frame de, wahi use hota hai.
    #   NEXT: run() isi ka result le kar MediaPipe loop chalata hai.
    def _open_camera(self):
        import cv2

        backends = []
        for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY"):  # Windows par DSHOW sabse tez, MSMF naye laptops ke liye
            if hasattr(cv2, name) and getattr(cv2, name) not in backends:
                backends.append(getattr(cv2, name))
        indices = [self.camera_index] + [i for i in range(4) if i != self.camera_index]
        for idx in indices:
            for be in backends:
                cap = cv2.VideoCapture(idx, be)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                # Warm-up: kuch cameras pehle kuch frame None dete hain — 40 tries (~2 s) tak ruko
                for _ in range(40):
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.size > 0:
                        log.info("camera %s opened with backend %s", idx, be)
                        return cap
                    time.sleep(0.05)
                cap.release()
        return None

    def run(self) -> None:
        # WHY try/except poore run par: QThread ke andar koi bhi exception chupchaap thread
        #     maar deta tha — user ko sirf kaala camera box dikhta tha, koi message nahi.
        #     Ab har error "failed" signal se UI banner tak pahunchta hai.
        try:
            self._run()
        except Exception as exc:  # MediaPipe/ONNX/OpenCV ka koi bhi native error
            log.exception("camera worker crashed")
            self.failed.emit(f"Camera/model error: {exc}")

    def _run(self) -> None:
        import cv2

        # Pick the landmark pipeline that matches the loaded model:
        #   IslRecognizer (shipped 93.8% model) → MediaPipe Pose + Hand, 75 points, Segmenter75
        #   Recognizer (Holistic training pipeline) → MediaPipe Holistic, 79 points, Segmenter
        from sanket.core.recognizer import IslRecognizer

        pose_mode = isinstance(self.recognizer, IslRecognizer)
        cap = self._open_camera()
        if cap is None:
            self.failed.emit("no_camera")
            return
        try:
            if pose_mode:
                from sanket.core.landmarks import PoseHandExtractor
                from sanket.core.pose75 import Segmenter75

                extractor = PoseHandExtractor()
                self.segmenter = Segmenter75()
            else:
                from sanket.core.landmarks import HolisticExtractor

                extractor = HolisticExtractor()
        except Exception as exc:  # pehle sirf FileNotFoundError pakadte the — baaki errors thread maar dete the
            log.exception("MediaPipe could not start")
            self.failed.emit(f"MediaPipe start nahi hua: {exc}")
            cap.release()
            return
        self._running = True
        t0 = time.monotonic()
        last_fps_t, frames_since = t0, 0
        no_hand_since = None
        was_signing = False
        bad_reads = 0  # WHY: USB/laptop camera kabhi-kabhi 1-2 frame gira deta hai — ek miss par band mat karo
        while self._running:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                bad_reads += 1
                if bad_reads > 30:  # ~1.5 s tak lagataar koi frame nahi = camera sach mein gaya
                    self.failed.emit("no_camera")
                    break
                time.sleep(0.05)
                continue
            bad_reads = 0
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            ts = int((time.monotonic() - t0) * 1000)
            a = time.perf_counter()
            pts = extractor.process(rgb, ts)  # UN-mirrored frame → correct left/right hands
            mp_ms = (time.perf_counter() - a) * 1000

            # Overlay points (42 hand points, 0-1 image units, NaN = missing) — same format for both
            # pipelines, so CameraView does not need to know which model is running.
            if pose_mode:
                from sanket.core.pose75 import overlay_points

                overlay = overlay_points(pts, w, h)
                person = bool(np.abs(pts[:33, :2]).sum() > 0)
                hands = bool(np.isfinite(overlay[:, 0]).any())
            else:
                overlay = pts[:42, :2].copy()
                overlay[:, 0] /= w / float(h)  # undo landmarks.py's W/H scaling
                person = bool(np.isfinite(pts[42:59, 0]).any())
                hands = bool(np.isfinite(pts[:42, 0]).any())

            # Preview: mirrored like a selfie camera, so the signer sees a natural reflection.
            view = np.ascontiguousarray(rgb[:, ::-1])
            img = QImage(view.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.frameReady.emit(img, overlay)

            # Low-light hint: person visible but no hand found for 3 s in a dark frame.
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
                self.segmentReady.emit(np.stack(seg) if isinstance(seg, list) else seg)
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
