"""signs.py — the avatar's sign dictionary and the player that strings signs together.

WHY: direction B must SHOW a sign for each word. We do not invent motions: for every one of the
     263 words, training/build_sign_dictionary.py picked the most typical real INCLUDE recording
     (signed by Deaf students) and stored its landmark motion in models/signs.npz.
WHAT: SignDictionary loads those motions. SignPlayer.timeline(tokens) turns a gloss list into
      one smooth list of avatar frames:
        • each gloss → its stored motion (optionally slowed to 0.5x);
        • 6 blended frames between signs so the avatar does not "jump";
        • "TEXT:" tokens (no sign yet) → a short hold of the rest pose while the caption shows.
NEXT: ui/avatar_widget.py draws one frame every 1/30 s and highlights the current caption word.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sanket.core.grammar import is_text_token
from sanket.core.normalize import N_POINTS
from sanket.paths import resource

log = logging.getLogger(__name__)

SIGNS_PATH = resource("models", "signs.npz")
BLEND_FRAMES = 6  # PRD FR-B4
TEXT_HOLD_FRAMES = 24  # ~0.8 s pause for a word without a sign


@dataclass
class Frame:
    """One avatar frame and which token (word) it belongs to (for caption highlighting)."""

    points: np.ndarray  # (79, 3) normalised, NaN = not drawn
    token_index: int


class SignDictionary:
    """gloss → (T, 79, 3) float32 motion, plus the rest pose shown between sentences."""

    def __init__(self, path: Path | str = SIGNS_PATH) -> None:
        self.path = Path(path)
        self.motions: dict[str, np.ndarray] = {}
        self.fps = 30.0
        if not self.path.exists():
            log.warning("signs.npz not found at %s — avatar will only show captions", self.path)
            self.rest = default_rest_pose()
            return
        with np.load(self.path, allow_pickle=False) as data:
            for key in data.files:
                if key == "__fps__":
                    self.fps = float(data[key])
                elif key == "__rest__":
                    continue
                else:
                    self.motions[key] = data[key].astype(np.float32)
            self.rest = data["__rest__"].astype(np.float32) if "__rest__" in data.files else default_rest_pose()

    def has(self, gloss: str) -> bool:
        return gloss in self.motions


def default_rest_pose() -> np.ndarray:
    """A neutral standing pose in normalised units (shoulders at ±0.5, hands down, face above).

    USE: shown before the first sign and for words without a sign, so the avatar never vanishes.
    """
    pts = np.full((N_POINTS, 3), np.nan, dtype=np.float32)
    pose = {0: (0.0, -1.1), 2: (-0.12, -1.18), 5: (0.12, -1.18), 7: (-0.3, -1.1), 8: (0.3, -1.1),
            9: (-0.06, -0.95), 10: (0.06, -0.95), 11: (0.5, 0.0), 12: (-0.5, 0.0),
            13: (0.62, 0.75), 14: (-0.62, 0.75), 15: (0.6, 1.45), 16: (-0.6, 1.45)}
    for idx, (x, y) in pose.items():
        pts[42 + idx] = (x, y, 0.0)
    # Face points (same order as normalize.FACE_IDX) so the head shows brows + lips, not a blank oval.
    face = [(-0.22, -1.2), (-0.18, -1.23), (-0.14, -1.24), (-0.10, -1.23), (-0.06, -1.21),
            (0.06, -1.21), (0.10, -1.23), (0.14, -1.24), (0.18, -1.23), (0.22, -1.2),
            (-0.11, -0.84), (0.11, -0.84), (0.0, -0.87), (0.0, -0.79), (-0.05, -0.86), (0.05, -0.86),
            (-0.05, -0.80), (0.05, -0.80), (0.0, -0.98), (0.0, -0.66)]
    for i, (x, y) in enumerate(face):
        pts[59 + i] = (x, y, 0.0)
    return pts


def _blend(a: np.ndarray, b: np.ndarray, n: int) -> list[np.ndarray]:
    """n frames moving linearly from pose a to pose b. Missing points simply take b's value."""
    out = []
    for i in range(1, n + 1):
        w = i / (n + 1)
        mix = a * (1 - w) + b * w
        mix = np.where(np.isfinite(a) & np.isfinite(b), mix, b)
        out.append(mix.astype(np.float32))
    return out


def _slow(motion: np.ndarray, speed: float) -> np.ndarray:
    """Resample a motion in time: speed 0.5 → twice as many frames (learners can follow)."""
    if speed >= 0.999 or len(motion) < 2:
        return motion
    n = max(2, int(round(len(motion) / speed)))
    src = np.linspace(0, len(motion) - 1, n)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, len(motion) - 1)
    w = (src - lo)[:, None, None]
    mix = motion[lo] * (1 - w) + motion[hi] * w
    return np.where(np.isfinite(motion[lo]) & np.isfinite(motion[hi]), mix, motion[lo]).astype(np.float32)


class SignPlayer:
    """Builds the full frame list for a sentence; the widget just plays it back."""

    def __init__(self, dictionary: SignDictionary) -> None:
        self.dict = dictionary

    def timeline(self, tokens: list[str], speed: float = 1.0) -> list[Frame]:
        frames: list[Frame] = []
        current = self.dict.rest
        for ti, tok in enumerate(tokens):
            if is_text_token(tok) or not self.dict.has(tok):
                target = self.dict.rest  # no sign → hold rest pose, caption carries the word
                frames += [Frame(p, ti) for p in _blend(current, target, BLEND_FRAMES)]
                frames += [Frame(target, ti) for _ in range(int(TEXT_HOLD_FRAMES / max(speed, 0.25)))]
                current = target
                continue
            motion = _slow(self.dict.motions[tok], speed)
            frames += [Frame(p, ti) for p in _blend(current, motion[0], BLEND_FRAMES)]
            frames += [Frame(p, ti) for p in motion]
            current = motion[-1]
        frames += [Frame(p, len(tokens) - 1) for p in _blend(current, self.dict.rest, BLEND_FRAMES)]
        return frames
