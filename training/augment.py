"""augment.py — makes ~3,100 training videos behave like many more.

WHY: INCLUDE has only ~12 clips per word, from a few signers. Without variety the model would
     memorise those exact people and fail on Kanak's webcam. Random, realistic changes teach it
     "the same sign, slightly different person / camera / speed".
WHAT: every function takes raw landmarks (T, 79, 3) with NaN and returns a modified copy.
      All changes are ones a real camera or signer could produce:
        rotate ±15° (camera tilt) · stretch x/y separately (body shape, lens) · time-warp
        0.8–1.2x (signing speed) · drop 10% of frames (webcam lag) · tiny jitter (detector
        noise) · lose a hand now and then (hand leaves the frame).
      NO left/right mirroring: in ISL the dominant hand matters, mirroring would teach errors.
NEXT: dataset.py applies `augment()` to training clips only (never to val/test).
"""

from __future__ import annotations

import numpy as np

from sanket.core.normalize import L_SHOULDER, LEFT_HAND, R_SHOULDER, RIGHT_HAND


def _centre(f: np.ndarray) -> np.ndarray:
    mid = np.nanmean((f[:, L_SHOULDER, :2] + f[:, R_SHOULDER, :2]) / 2, axis=0)
    return np.where(np.isfinite(mid), mid, 0.5)


def rotate_scale(f: np.ndarray, rng: np.random.Generator, max_deg=15.0, scale=(0.85, 1.15)) -> np.ndarray:
    """Rotate and stretch around the shoulder midpoint (camera tilt / different body shapes)."""
    c = _centre(f)
    a = np.deg2rad(rng.uniform(-max_deg, max_deg))
    sx, sy = rng.uniform(*scale), rng.uniform(*scale)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    out = f.copy()
    xy = (out[..., :2] - c) @ rot.T
    out[..., 0] = xy[..., 0] * sx + c[0]
    out[..., 1] = xy[..., 1] * sy + c[1]
    return out


def time_warp(f: np.ndarray, rng: np.random.Generator, lo=0.8, hi=1.2) -> np.ndarray:
    """Play the sign faster or slower by resampling time (nearest frame keeps NaN pattern)."""
    t = len(f)
    n = max(4, int(round(t * rng.uniform(lo, hi))))
    idx = np.clip(np.round(np.linspace(0, t - 1, n)).astype(int), 0, t - 1)
    return f[idx]


def drop_frames(f: np.ndarray, rng: np.random.Generator, p=0.1) -> np.ndarray:
    """Remove random frames (like a laggy webcam)."""
    if len(f) < 10:
        return f
    keep = rng.random(len(f)) >= p
    keep[0] = keep[-1] = True
    return f[keep]


def jitter(f: np.ndarray, rng: np.random.Generator, sigma=0.004) -> np.ndarray:
    """Small random noise on every point (MediaPipe's own frame-to-frame wobble)."""
    return f + rng.normal(0, sigma, f.shape).astype(np.float32)


def drop_hand(f: np.ndarray, rng: np.random.Generator, p=0.1) -> np.ndarray:
    """Occasionally hide one hand for a stretch of frames (hand leaves the camera view)."""
    if rng.random() >= p or len(f) < 6:
        return f
    out = f.copy()
    hand = LEFT_HAND if rng.random() < 0.5 else RIGHT_HAND
    a = rng.integers(0, len(f) // 2)
    b = rng.integers(a + 1, len(f))
    out[a:b, hand] = np.nan
    return out


def augment(frames: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply the full random chain to one training clip."""
    f = rotate_scale(frames, rng)
    f = time_warp(f, rng)
    f = drop_frames(f, rng)
    f = jitter(f, rng)
    return drop_hand(f, rng)
