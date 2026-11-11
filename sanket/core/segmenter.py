"""segmenter.py — decides where one sign starts and ends in the live camera stream.

WHY: the recognizer classifies ONE isolated sign at a time (that is how INCLUDE is recorded).
     A live signer produces a continuous stream, so something must cut it into single signs.
WHAT: a small state machine fed one frame of points at a time:
        IDLE   → hands down / not moving
        ACTIVE → a hand is raised into the signing space or clearly moving
      When ACTIVE ends (hands drop, or the hands stay still for `still_frames`), the collected
      frames are returned as one sign segment.
NEXT: the app sends each returned segment to Recognizer.predict().

Edge cases handled: one hand out of frame (still a sign), no hands for the whole segment
(discarded), signer never pausing (forced cut at `max_frames`), tiny twitches (min length).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from sanket.core.normalize import L_SHOULDER, L_WRIST, LEFT_HAND, R_SHOULDER, R_WRIST, RIGHT_HAND


@dataclass
class SegmenterConfig:
    """Tunable thresholds (Settings screen can change sensitivity later)."""

    raise_ratio: float = 1.4  # wrist counts as "raised" if it is above shoulder_y + ratio*shoulder_width
    motion_on: float = 0.035  # motion (in shoulder-widths per frame) that starts a sign
    motion_still: float = 0.012  # below this, the hands count as still
    still_frames: int = 10  # this many still/lowered frames in a row ends a sign (~0.33 s at 30 fps)
    min_frames: int = 8  # shorter segments are twitches, not signs
    max_frames: int = 120  # 4 s cap so a non-stop signer still gets results
    pre_roll: int = 4  # frames kept from just before the start (the sign's beginning)


@dataclass
class Segmenter:
    """Feed frames with push(); it returns a finished segment or None."""

    config: SegmenterConfig = field(default_factory=SegmenterConfig)

    def __post_init__(self) -> None:
        self._history: deque[np.ndarray] = deque(maxlen=self.config.pre_roll)  # pre-roll buffer
        self._active: list[np.ndarray] = []  # frames of the sign in progress
        self._quiet = 0  # consecutive still/lowered frames while ACTIVE
        self._prev: np.ndarray | None = None  # previous frame, for motion
        self._raised_run = 0  # consecutive frames with a hand in signing space while IDLE
        self._armed = True  # False right after a sign ends until hands drop or move again,
        #                     so a hand still held up after a sign does not re-trigger forever

    @property
    def active(self) -> bool:
        """True while a sign is being collected (UI shows a red "recording" dot)."""
        return bool(self._active)

    @staticmethod
    def _hands_raised(pts: np.ndarray, ratio: float) -> bool:
        """Is either visible wrist inside the signing space (roughly chest height or above)?

        Image y grows downward, so "above" means a smaller y than shoulder_y + ratio*width.
        """
        ls, rs = pts[L_SHOULDER, :2], pts[R_SHOULDER, :2]
        if not (np.isfinite(ls).all() and np.isfinite(rs).all()):
            # No shoulders visible: fall back to "a hand is visible at all".
            return bool(np.isfinite(pts[LEFT_HAND, 0]).any() or np.isfinite(pts[RIGHT_HAND, 0]).any())
        width = float(np.linalg.norm(ls - rs)) or 1e-3
        limit = (ls[1] + rs[1]) / 2 + ratio * width
        for wrist, hand in ((L_WRIST, LEFT_HAND), (R_WRIST, RIGHT_HAND)):
            hand_seen = np.isfinite(pts[hand, 0]).any()
            y = pts[hand, 1][0] if hand_seen else pts[wrist, 1]  # hand point 0 = wrist
            if hand_seen and np.isfinite(y) and y < limit:
                return True
        return False

    @staticmethod
    def _motion(prev: np.ndarray | None, cur: np.ndarray) -> float:
        """Average hand movement between two frames, in shoulder-widths (scale-independent)."""
        if prev is None:
            return 0.0
        ls, rs = cur[L_SHOULDER, :2], cur[R_SHOULDER, :2]
        width = float(np.linalg.norm(ls - rs)) if np.isfinite(ls).all() and np.isfinite(rs).all() else 0.25
        width = max(width, 1e-3)
        moves = []
        for hand in (LEFT_HAND, RIGHT_HAND):
            a, b = prev[hand, :2], cur[hand, :2]
            ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
            if ok.any():
                moves.append(float(np.linalg.norm(a[ok] - b[ok], axis=1).mean()))
        return (max(moves) / width) if moves else 0.0

    def push(self, pts: np.ndarray) -> np.ndarray | None:
        """Add one frame of (79, 3) points. Returns a finished (T, 79, 3) segment or None."""
        cfg = self.config
        raised = self._hands_raised(pts, cfg.raise_ratio)
        motion = self._motion(self._prev, pts)
        self._prev = pts

        if not self._active:  # ---------------- IDLE
            # Start when a hand is in signing space AND either moving clearly or held there for
            # 3 frames (static handshapes like colours barely move once they are in place).
            if not self._armed and (not raised or motion >= cfg.motion_on):
                self._armed = True  # signer lowered hands or started a new movement
            self._raised_run = self._raised_run + 1 if raised else 0
            if self._armed and raised and (motion >= cfg.motion_on or self._raised_run >= 3):
                self._active = list(self._history) + [pts]  # start: include pre-roll frames
                self._quiet = 0
            else:
                self._history.append(pts)
            return None

        # ---------------- ACTIVE
        self._active.append(pts)
        self._quiet = self._quiet + 1 if (not raised or motion < cfg.motion_still) else 0
        if self._quiet >= cfg.still_frames or len(self._active) >= cfg.max_frames:
            return self._finish()
        return None

    def flush(self) -> np.ndarray | None:
        """Force-close the current sign (user pressed Enter). Returns it if long enough."""
        return self._finish() if self._active else None

    def _finish(self) -> np.ndarray | None:
        """Close the segment, drop trailing still frames, reject twitches and hand-less segments."""
        frames = self._active
        if self._quiet and len(frames) > self._quiet:
            frames = frames[: len(frames) - self._quiet + 2]  # keep 2 frames of the hold
        self._active, self._quiet = [], 0
        self._history.clear()
        self._armed, self._raised_run = False, 0  # wait for a fresh movement before the next sign
        if len(frames) < self.config.min_frames:
            return None
        seg = np.stack(frames).astype(np.float32)
        hands_ever = np.isfinite(seg[:, LEFT_HAND, 0]).any() or np.isfinite(seg[:, RIGHT_HAND, 0]).any()
        return seg if hands_ever else None
