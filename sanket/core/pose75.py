"""pose75.py — the feature layout of SANKET's shipped ISL model (the 93.8% INCLUDE pose model).

WHY this file exists: the recognizer that ships in the app was trained on AI4Bharat's pre-extracted
     INCLUDE pose keypoints (Zenodo 6674324), whose layout is MediaPipe Pose (33 body points) +
     two hands (21 + 21) = 75 points in PIXEL units. The model only works if the live app builds
     its input exactly the way training did. So everything here is a line-for-line port of
     experiments/include_pose_baseline/sanket_core.py, which was verified bit-for-bit against the
     Colab training code (tests/test_pose75.py re-checks that against the experiment file).
WHAT:
  • assign_hands() / frame_to_75(): MediaPipe Pose + Hand results → one (75, 3) frame
  • normalize() / build_input(): list of frames → model input (1, 64, 136)
  • hand_active() / Segmenter75: cut the live stream into single signs
NEXT: landmarks.PoseHandExtractor produces the frames, recognizer.IslRecognizer consumes the
      segments, ui/workers.CameraWorker connects the two.

IMPORTANT: do not "improve" normalize()/build_input() — any change, however small, makes the live
           numbers differ from training and the model will answer wrongly. Retrain if you change it.
"""

from __future__ import annotations

from collections import deque

import numpy as np

# ---------------------------------------------------------------------------------------------
# Layout constants — must match training exactly.
# 75 points = 0-32 body (MediaPipe Pose) | 33-53 hand A | 54-74 hand B
# Hand A = the hand nearest body wrist 15, hand B = nearest body wrist 16 (INCLUDE layout).
# ---------------------------------------------------------------------------------------------
POSE_N, HAND_N = 33, 21
HAND_A0, HAND_B0 = 33, 54  # where each hand starts in the 75-point array
WRIST_A, WRIST_B = 15, 16  # body landmarks 15 = left wrist, 16 = right wrist
L_SHOULDER, R_SHOULDER = 11, 12
KEEP = list(range(25)) + list(range(33, 75))  # training dropped legs (25-32) → 67 points
HA, HB = slice(25, 46), slice(46, 67)  # position of the hands inside KEEP
D_IN = 67 * 2 + 2  # 134 (x, y) + 2 hand-present flags = 136 numbers per frame


# ---------------------------------------------------------------------------------------------
# STEP 1: MediaPipe output → one (75, 3) array in the training layout
# ---------------------------------------------------------------------------------------------
def _to_px(lms, w: int, h: int) -> np.ndarray:
    """Normalised (0-1) landmarks → pixels.

    WHY: the training data was in pixel units. x must be multiplied by width and y by height,
         otherwise a 16:9 frame squashes the shape and the shoulder width (the scale) is wrong.
    """
    return np.array([[p.x * w, p.y * h, p.z] for p in lms], np.float32)


def assign_hands(pose_px, hands_px, handed_labels):
    """Put the detected hands into slot A / B.

    WHY: MediaPipe's 'Left/Right' label is confusing with mirroring. Training used the rule
         'the hand nearest body wrist 15 is A'. The same rule is applied here (by distance),
         so live and training mean the same thing.
    RETURNS: (hand_a, hand_b) — each a (21, 3) array or None.
    """
    if not hands_px:
        return None, None
    if pose_px is not None:
        wa, wb = pose_px[WRIST_A, :2], pose_px[WRIST_B, :2]

        def d(hnd, w):  # distance between the hand's wrist (hand point 0) and a body wrist
            return float(np.linalg.norm(hnd[0, :2] - w))

        if len(hands_px) == 1:
            h0 = hands_px[0]
            return (h0, None) if d(h0, wa) <= d(h0, wb) else (None, h0)
        h0, h1 = hands_px[0], hands_px[1]
        # Two hands: of the two possible pairings, the one with the smaller total distance is right.
        if d(h0, wa) + d(h1, wb) <= d(h0, wb) + d(h1, wa):
            return h0, h1
        return h1, h0
    # No body found (rare): fall back to MediaPipe's label. Hand Landmarker assumes a mirrored
    # (selfie) image and we pass the raw frame, so the label reads inverted: 'Right' = slot A.
    a = b = None
    for hnd, lab in zip(hands_px, handed_labels, strict=True):
        if lab == "Right" and a is None:
            a = hnd
        elif b is None:
            b = hnd
        elif a is None:
            a = hnd
    return a, b


def frame_to_75(pose_result, hand_result, w: int, h: int) -> np.ndarray:
    """One frame's MediaPipe Pose + Hand results → (75, 3) float32. Missing points are 0 (as in training).

    NEXT: the array goes to Segmenter75.push().
    """
    out = np.zeros((75, 3), np.float32)
    pose_px = None
    if pose_result is not None and pose_result.pose_landmarks:
        pose_px = _to_px(pose_result.pose_landmarks[0], w, h)
        out[:POSE_N] = pose_px
    hands_px, labels = [], []
    if hand_result is not None and hand_result.hand_landmarks:
        for i, lms in enumerate(hand_result.hand_landmarks[:2]):
            hands_px.append(_to_px(lms, w, h))
            cat = hand_result.handedness[i][0] if hand_result.handedness and hand_result.handedness[i] else None
            labels.append(cat.category_name if cat else "")
    ha, hb = assign_hands(pose_px, hands_px, labels)
    if ha is not None:
        out[HAND_A0 : HAND_A0 + HAND_N] = ha
    if hb is not None:
        out[HAND_B0 : HAND_B0 + HAND_N] = hb
    return out


# ---------------------------------------------------------------------------------------------
# STEP 2: normalisation + model input — exact copy of the Colab training code
# ---------------------------------------------------------------------------------------------
def normalize(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre on the shoulder midpoint (11, 12) and scale by shoulder width.

    WHY: removes camera distance / resolution differences, so a laptop webcam and the INCLUDE
         camera end up on the same scale. Median per segment = one or two bad frames don't spoil
         the whole sign. Clip ±6 keeps outliers in check.
    """
    k = k[:, KEEP, :2].astype(np.float32)
    pres = np.abs(k).sum(-1) > 0
    sh_c = (k[:, 11] + k[:, 12]) / 2
    sh_w = np.linalg.norm(k[:, 11] - k[:, 12], axis=-1)
    ok = sh_w > 1e-6
    c = np.median(sh_c[ok], 0) if ok.any() else np.zeros(2, np.float32)
    s = np.median(sh_w[ok]) if ok.any() else 1.0
    k = np.clip((k - c) / s, -6, 6)
    k[~pres] = 0
    return k, pres


def build_input(frames75, T: int = 64) -> np.ndarray:
    """List of (75, 3) frames → model input (1, T, 136).

    Like training's eval mode: pick T evenly spaced frames from the whole segment (no random crop).
    NEXT: IslRecognizer feeds this to the ONNX model.
    """
    k, pres = normalize(np.stack(frames75))
    idx = np.linspace(0, len(k) - 1, T).round().astype(int)
    k, pres = k[idx], pres[idx]
    flags = np.stack([pres[:, HA].any(1), pres[:, HB].any(1)], 1).astype(np.float32)
    x = np.concatenate([k.reshape(T, -1), flags], 1).astype(np.float32)
    return x[None]


# ---------------------------------------------------------------------------------------------
# STEP 3: Segmenter — cut single signs out of the continuous webcam stream
# ---------------------------------------------------------------------------------------------
def hand_active(f75: np.ndarray, raise_k: float) -> bool:
    """Is a hand in 'signing position'?

    RULE: a hand is detected AND its wrist is above shoulder_y + raise_k * shoulder width.
    WHY: MediaPipe also finds hands hanging at rest; starting a sign on 'detected' alone would
         produce predictions without any signing.
    """
    hands = [f75[HAND_A0 : HAND_A0 + HAND_N], f75[HAND_B0 : HAND_B0 + HAND_N]]
    present = [np.abs(hd[:, :2]).sum() > 0 for hd in hands]
    if not any(present):
        return False
    ls, rs = f75[L_SHOULDER, :2], f75[R_SHOULDER, :2]
    sw = float(np.linalg.norm(ls - rs))
    if sw < 1e-6:  # no body found → a visible hand is enough
        return True
    limit_y = (ls[1] + rs[1]) / 2 + raise_k * sw  # image y grows downward
    return any(p and hd[0, 1] < limit_y for hd, p in zip(hands, present, strict=True))


class Segmenter75:
    """Small state machine:  IDLE --(hand raised for start_frames)--> RECORDING
                             RECORDING --(hand down for end_frames / max_len)--> segment ready

    pre_roll: a few frames from before the start are included, because INCLUDE videos contain a
    little rest position before the sign, so the model expects that shape too.
    """

    def __init__(self, raise_k=1.2, start_frames=3, end_frames=10, pre_roll=8, min_len=16, max_len=160):
        self.raise_k, self.start_frames, self.end_frames = raise_k, start_frames, end_frames
        self.min_len, self.max_len = min_len, max_len
        self.history: deque = deque(maxlen=pre_roll + start_frames)  # IDLE: recent frames (pre-roll)
        self.buf: list = []
        self.recording, self.on, self.off = False, 0, 0

    @property
    def active(self) -> bool:
        """True while a sign is being recorded (UI shows the red dot)."""
        return self.recording

    def reset(self) -> None:
        self.history.clear()
        self.buf, self.recording, self.on, self.off = [], False, 0, 0

    def push(self, f75: np.ndarray):
        """Add one frame. Returns the list of frames as soon as a sign is complete, else None.

        NEXT: the returned list goes to IslRecognizer.predict().
        """
        active = hand_active(f75, self.raise_k)
        if not self.recording:
            self.history.append(f75)
            self.on = self.on + 1 if active else 0
            if self.on >= self.start_frames:  # a real start (not a 1-2 frame twitch)
                self.recording, self.buf, self.off = True, list(self.history), 0
            return None
        self.buf.append(f75)
        self.off = 0 if active else self.off + 1
        if self.off >= self.end_frames or len(self.buf) >= self.max_len:
            seg = self.buf[: len(self.buf) - max(0, self.off - 4)]  # keep only 4 of the trailing empty frames
            self.reset()
            return seg if len(seg) >= self.min_len else None  # too short = accidental hand movement
        return None

    def flush(self):
        """Enter key: close the current sign now (returns it if long enough)."""
        if not self.recording:
            return None
        seg = self.buf
        self.reset()
        return seg if len(seg) >= self.min_len else None


def overlay_points(f75: np.ndarray, w: int, h: int) -> np.ndarray:
    """Hand points for the camera overlay: (42, 2) in 0-1 image units, NaN where missing.

    USE: ui CameraView draws these on the mirrored preview (it flips x itself).
    """
    hands = np.concatenate([f75[HAND_A0 : HAND_A0 + HAND_N, :2], f75[HAND_B0 : HAND_B0 + HAND_N, :2]])
    out = hands / np.array([w, h], np.float32)
    out[np.abs(hands).sum(1) == 0] = np.nan
    return out.astype(np.float32)
