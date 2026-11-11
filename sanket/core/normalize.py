"""normalize.py — THE single definition of SANKET's landmark features.

WHY: the recognizer only works if training (on INCLUDE videos) and the live app (webcam) turn
     landmarks into numbers in exactly the same way. Two copies of this logic would drift apart,
     so both `training/` and `sanket/` import this one file.
WHAT: converts raw MediaPipe landmarks of one sign, shape (T, 79, 3) with NaN for missing points,
      into a fixed-size model input of shape (64, 239):
        79 points x (x, y, z) = 237 numbers  + 2 hand-present flags = 239 per frame,
        resampled to 64 frames.
NEXT: recognizer.py feeds the result to recognizer.onnx; training/dataset.py feeds it to the
      PyTorch model; signs.py uses the normalised points to draw the avatar.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------------------------
# Point layout (fixed contract — changing it means retraining the model).
# WHY these points: ISL meaning lives mostly in the hands, then arm/shoulder position, then a
# few facial cues (eyebrows for questions, lips for mouthing). 79 points keep the model small.
# ---------------------------------------------------------------------------------------------
N_HAND = 21  # MediaPipe hand model gives 21 points per hand
POSE_IDX = list(range(17))  # pose 0-16 = face outline + shoulders, elbows, wrists, hand tips (upper body only)
FACE_IDX = [
    70, 63, 105, 66, 107,  # left eyebrow  (raised eyebrows mark yes/no questions in ISL)
    336, 296, 334, 293, 300,  # right eyebrow
    61, 291, 0, 17, 37, 267, 84, 314,  # outer lips (mouthing of words)
    1,  # nose tip
    152,  # chin
]
N_POSE = len(POSE_IDX)  # 17
N_FACE = len(FACE_IDX)  # 20
N_POINTS = 2 * N_HAND + N_POSE + N_FACE  # 79

# Where each block starts inside the 79-point array (used by segmenter and avatar too).
LEFT_HAND = slice(0, 21)
RIGHT_HAND = slice(21, 42)
POSE = slice(42, 59)
FACE = slice(59, 79)
L_SHOULDER = 42 + 11  # pose landmark 11 = left shoulder
R_SHOULDER = 42 + 12  # pose landmark 12 = right shoulder
L_WRIST = 42 + 15
R_WRIST = 42 + 16

N_FLAGS = 2  # left-hand-present, right-hand-present
FEATURES = N_POINTS * 3 + N_FLAGS  # 239 numbers per frame
SEQ_LEN = 64  # every sign is stretched/squeezed to 64 frames before the model sees it


def empty_frame() -> np.ndarray:
    """A frame with every point missing (NaN). USE: when MediaPipe finds no person at all."""
    return np.full((N_POINTS, 3), np.nan, dtype=np.float32)


def _body_frame(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame origin (shoulder midpoint) and scale (shoulder width).

    WHY: people stand at different distances and positions in front of the camera. Measuring
         every point relative to the shoulders makes the same sign give the same numbers.
    HOW: frames where shoulders are missing borrow the median of frames where they were seen;
         if they were never seen, fall back to the image centre and a typical width.
    NEXT: normalize_sequence() subtracts origin and divides by scale.
    """
    ls, rs = frames[:, L_SHOULDER, :2], frames[:, R_SHOULDER, :2]
    origin = (ls + rs) / 2.0  # (T, 2), NaN where a shoulder is missing
    scale = np.linalg.norm(ls - rs, axis=1)  # (T,)
    ok = np.isfinite(origin).all(axis=1) & np.isfinite(scale) & (scale > 1e-4)
    if ok.any():
        fallback_origin = np.median(origin[ok], axis=0)
        fallback_scale = float(np.median(scale[ok]))
    else:  # no shoulders anywhere in the clip (very rare: person cut off at the neck)
        fallback_origin = np.array([0.5, 0.5], dtype=np.float32)
        fallback_scale = 0.3
    origin = np.where(ok[:, None], origin, fallback_origin)
    scale = np.where(ok, scale, fallback_scale)
    return origin.astype(np.float32), scale.astype(np.float32)


def normalize_sequence(frames: np.ndarray) -> np.ndarray:
    """Raw landmarks (T, 79, 3) with NaN → features (T, 239) with no NaN.

    STEPS:
      1. shift so the shoulder midpoint is (0, 0) and divide x, y, z by shoulder width;
      2. record whether each hand was seen (the model must know "no hand" from "hand at 0,0");
      3. replace every remaining NaN with 0.
    NEXT: resample() makes the length exactly SEQ_LEN.
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1:] != (N_POINTS, 3):
        raise ValueError(f"expected (T, {N_POINTS}, 3), got {frames.shape}")
    origin, scale = _body_frame(frames)
    pts = frames.copy()
    pts[:, :, :2] -= origin[:, None, :]  # 1a. centre on shoulders
    pts /= scale[:, None, None]  # 1b. size-independent (z uses the same scale)
    left_seen = np.isfinite(frames[:, LEFT_HAND, 0]).any(axis=1).astype(np.float32)  # 2.
    right_seen = np.isfinite(frames[:, RIGHT_HAND, 0]).any(axis=1).astype(np.float32)
    pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)  # 3.
    pts = np.clip(pts, -10.0, 10.0)  # guard against a wild detection blowing up the model
    flat = pts.reshape(len(pts), N_POINTS * 3)
    return np.concatenate([flat, left_seen[:, None], right_seen[:, None]], axis=1).astype(np.float32)


def resample(seq: np.ndarray, length: int = SEQ_LEN) -> np.ndarray:
    """Stretch or squeeze a (T, F) sequence to (length, F) by linear interpolation in time.

    WHY: signers sign at different speeds and clips have different lengths, but the
         transformer needs a fixed length. Interpolation keeps the motion's shape.
    """
    seq = np.asarray(seq, dtype=np.float32)
    t = len(seq)
    if t == 0:
        return np.zeros((length, seq.shape[1] if seq.ndim == 2 else FEATURES), np.float32)
    if t == 1:
        return np.repeat(seq, length, axis=0)
    src = np.linspace(0.0, t - 1, num=length)  # where each output frame samples the input
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (src - lo)[:, None].astype(np.float32)
    return (seq[lo] * (1 - w) + seq[hi] * w).astype(np.float32)


def model_input(frames: np.ndarray) -> np.ndarray:
    """Raw landmarks of one sign → the exact (64, 239) float32 array the recognizer expects.

    USE: the only function the recognizer and the training dataset need to call.
    """
    return resample(normalize_sequence(frames), SEQ_LEN)


def normalized_points(frames: np.ndarray) -> np.ndarray:
    """Like normalize_sequence() but keeps the (T, 79, 3) shape and leaves NaN for missing points.

    USE: avatar drawing — it must know which points are missing so it doesn't draw a hand at 0,0.
    """
    frames = np.asarray(frames, dtype=np.float32)
    origin, scale = _body_frame(frames)
    pts = frames.copy()
    pts[:, :, :2] -= origin[:, None, :]
    pts /= scale[:, None, None]
    return pts
