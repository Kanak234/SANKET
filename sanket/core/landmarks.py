"""landmarks.py — turns a camera/video frame into SANKET's 79 body points.

WHY: the recognizer never looks at pixels. It looks at where the hands, arms and a few face
     points are. Google's MediaPipe Holistic model finds those points; this file wraps it so the
     rest of SANKET only deals with a simple (79, 3) NumPy array.
WHAT: `HolisticExtractor.process(frame_rgb, timestamp_ms)` → (79, 3) float32, NaN where missing.
      MediaPipe is used FROZEN, exactly as Google ships it — nothing is trained or fine-tuned.
NEXT: the live app passes each frame's points to segmenter.py; training/extract_landmarks.py
      runs the same extractor over every INCLUDE video so training sees identical numbers.

IMPORTANT: always pass the UN-mirrored camera frame. INCLUDE videos are not mirrored, and ISL
           meaning depends on which hand does what. The UI mirrors only the picture it displays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sanket.core.normalize import FACE_IDX, N_HAND, N_POINTS, POSE_IDX, empty_frame
from sanket.paths import resource

# Official Google download location of the Holistic model (fetched once at build time by
# scripts/fetch_models.py — the running app never goes online).
HOLISTIC_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)
HOLISTIC_PATH = resource("models", "mediapipe", "holistic_landmarker.task")


def _fill(dst: np.ndarray, offset: int, landmarks, indices: list[int] | None = None) -> None:
    """Copy MediaPipe landmark objects into rows of `dst` starting at `offset`.

    WHY a helper: hands, pose and face all need the same copy loop; `indices` picks only the
    pose/face points SANKET uses (17 of 33 pose points, 20 of 478 face points).
    """
    if not landmarks:  # that body part was not detected in this frame → rows stay NaN
        return
    picks = indices if indices is not None else range(len(landmarks))
    for row, idx in enumerate(picks):
        if idx < len(landmarks):
            lm = landmarks[idx]
            dst[offset + row] = (lm.x, lm.y, lm.z)


def result_to_points(result) -> np.ndarray:
    """MediaPipe HolisticLandmarkerResult → (79, 3) array in SANKET's fixed point order.

    ORDER (must match normalize.py): left hand 21, right hand 21, pose 17, face 20.
    """
    pts = empty_frame()
    if result is None:
        return pts
    _fill(pts, 0, result.left_hand_landmarks)
    _fill(pts, N_HAND, result.right_hand_landmarks)
    _fill(pts, 2 * N_HAND, result.pose_landmarks, POSE_IDX)
    _fill(pts, 2 * N_HAND + len(POSE_IDX), result.face_landmarks, FACE_IDX)
    assert pts.shape == (N_POINTS, 3)
    return pts


class HolisticExtractor:
    """Owns one MediaPipe Holistic model in VIDEO mode (it tracks between frames → smoother).

    USE: create once, call process() for every frame with an increasing timestamp, close() at end.
    NEXT: returned points go to Segmenter.push() in the app or to an .npz file in training.
    """

    def __init__(self, model_path: Path | str = HOLISTIC_PATH) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model missing: {model_path}. Run `python scripts/fetch_models.py` once."
            )
        # Imported here (not at top) so tests and tools that never touch the camera stay fast.
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,  # VIDEO = uses previous frame to track faster
            min_pose_detection_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
        )
        self._landmarker = vision.HolisticLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_rgb: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """One RGB uint8 frame (H, W, 3) → (79, 3) points. Timestamps must strictly increase."""
        import mediapipe as mp

        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)  # MediaPipe rejects repeats
        self._last_ts = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        pts = result_to_points(result)
        # MediaPipe gives x as a fraction of WIDTH and y as a fraction of HEIGHT. A 16:9 INCLUDE
        # video and a 4:3 webcam would then stretch the same sign differently. Multiplying x (and
        # z, which MediaPipe scales like x) by W/H puts both axes in the same units, so the
        # model sees identical shapes from any camera.
        h, w = frame_rgb.shape[:2]
        pts[:, [0, 2]] *= w / float(h)
        return pts

    def close(self) -> None:
        """Free the native MediaPipe graph (important in the multi-process extractor)."""
        self._landmarker.close()

    def __enter__(self) -> HolisticExtractor:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
