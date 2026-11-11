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


# ---------------------------------------------------------------------------------------------
# Pose + Hand extractor — the input format of the SHIPPED ISL model (models/isl/).
# WHY separate from Holistic: the 93.8% model was trained on AI4Bharat's INCLUDE pose keypoints,
#      made with MediaPipe Pose (33 points) + Hand (21 x 2). Holistic gives slightly different
#      points, so the shipped model must be fed by these two models instead.
# ---------------------------------------------------------------------------------------------
ISL_DIR = resource("models", "isl")
POSE_TASK = ISL_DIR / "pose_landmarker_full.task"
HAND_TASK = ISL_DIR / "hand_landmarker.task"
POSE_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task")


class PoseHandExtractor:
    """One webcam frame → (75, 3) points in PIXEL units (training layout, see core/pose75.py).

    USE: create once, process() every frame with increasing timestamps, close() at the end.
    NEXT: the frame goes to pose75.Segmenter75.push(); finished segments go to IslRecognizer.
    The .task files are bundled at build time (scripts/fetch_models.py) — never downloaded at runtime.
    """

    def __init__(self, pose_task: Path | str = POSE_TASK, hand_task: Path | str = HAND_TASK) -> None:
        for p in (Path(pose_task), Path(hand_task)):
            if not p.exists():
                raise FileNotFoundError(f"MediaPipe model missing: {p}. Run `python scripts/fetch_models.py` once.")
        from mediapipe.tasks.python import BaseOptions, vision

        # VIDEO mode = tracks between frames instead of detecting from scratch every frame → faster
        self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_task)),
            running_mode=vision.RunningMode.VIDEO, num_poses=1))
        self._hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_task)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.5, min_tracking_confidence=0.5))
        self._last_ts = -1

    def process(self, frame_rgb: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """RGB uint8 frame (H, W, 3), UN-mirrored → (75, 3) pixel-unit points (0 = missing)."""
        import mediapipe as mp

        from sanket.core.pose75 import frame_to_75

        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)  # MediaPipe rejects repeated timestamps
        self._last_ts = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        h, w = frame_rgb.shape[:2]
        return frame_to_75(self._pose.detect_for_video(image, timestamp_ms),
                           self._hands.detect_for_video(image, timestamp_ms), w, h)

    def close(self) -> None:
        self._pose.close()
        self._hands.close()
