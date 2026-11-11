"""build_sign_dictionary.py — chooses one real recording per word for the avatar to perform.

WHY: the avatar must sign CORRECT ISL, and SANKET has no Deaf reviewer yet. So we do not
     generate motions: for each of the 263 words we pick the most typical of its real INCLUDE
     recordings (made by Deaf students) — the "medoid", i.e. the clip most similar on average to
     all other clips of that word. That avoids picking an odd or badly-detected take.
WHAT: for each word: load its training clips → normalise → distance between clips with
      Dynamic Time Warping (handles different speeds) → pick the medoid → fill tiny detection
      gaps → smooth jitter (One-Euro filter) → resample to 30 fps → store as float16.
      Output: models/signs.npz  {gloss: (T, 79, 3), "__fps__": 30, "__rest__": (79, 3)}
      Credit for these motions is shown in the app (About + Dictionary), as CC BY 4.0 requires.
NEXT: sanket/core/signs.py loads the file; ui/avatar_widget.py draws it.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from sanket.core.normalize import FACE, N_POINTS, normalized_points
from training.common import MODELS, cache_path, label_from_path, split_paths

OUT_FPS = 30.0
DTW_LEN = 32  # clips are shortened to 32 frames just for comparison (DTW cost grows with length²)


def fill_gaps(pts: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """Linearly fill NaN runs of up to `max_gap` frames per coordinate (detector blinks).

    Longer gaps stay NaN on purpose: the hand really was out of view, so the avatar hides it.
    """
    out = pts.copy()
    t = len(out)
    flat = out.reshape(t, -1)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        bad = ~np.isfinite(col)
        if not bad.any() or bad.all():
            continue
        good = np.flatnonzero(~bad)
        i = 0
        while i < t:
            if bad[i]:
                j = i
                while j < t and bad[j]:
                    j += 1
                if i > 0 and j < t and (j - i) <= max_gap:
                    col[i:j] = np.interp(np.arange(i, j), good, col[good])
                i = j
            else:
                i += 1
    return flat.reshape(out.shape)


def one_euro(pts: np.ndarray, fps: float, min_cutoff: float = 1.5, beta: float = 0.3) -> np.ndarray:
    """One-Euro filter: smooths small jitter but keeps fast real movements sharp."""
    out = pts.copy()
    prev = None
    dprev = np.zeros_like(pts[0])

    def alpha(cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * fps)

    for i in range(len(pts)):
        x = pts[i]
        if prev is None:
            prev = x.copy()
            continue
        ok = np.isfinite(x) & np.isfinite(prev)
        dx = np.where(ok, (x - prev) * fps, 0.0)
        dprev = dprev + alpha(1.0) * (dx - dprev)
        cutoff = min_cutoff + beta * np.abs(dprev)
        a = alpha(cutoff)
        smooth = np.where(ok, prev + a * (x - prev), x)
        out[i] = smooth
        prev = np.where(np.isfinite(smooth), smooth, prev)
    return out


def resample_time(pts: np.ndarray, src_fps: float, dst_fps: float = OUT_FPS) -> np.ndarray:
    """Change frame rate keeping real duration (INCLUDE is ~25 fps; the avatar plays at 30)."""
    n = max(2, int(round(len(pts) * dst_fps / src_fps)))
    idx = np.linspace(0, len(pts) - 1, n)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(pts) - 1)
    w = (idx - lo)[:, None, None]
    mix = pts[lo] * (1 - w) + pts[hi] * w
    return np.where(np.isfinite(pts[lo]) & np.isfinite(pts[hi]), mix, pts[lo]).astype(np.float32)


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Dynamic Time Warping distance between two (T, F) sequences."""
    n, m = len(a), len(b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, m] / (n + m))


def compare_vector(pts: np.ndarray) -> np.ndarray:
    """Hands + arms only, NaN→0, 32 frames: what matters for "is this the same sign"."""
    sel = pts[:, :59, :2]  # both hands + upper body (face excluded)
    idx = np.linspace(0, len(sel) - 1, DTW_LEN).round().astype(int)
    return np.nan_to_num(sel[idx]).reshape(DTW_LEN, -1)


def medoid(clips: list[np.ndarray]) -> int:
    """Index of the clip with the smallest mean DTW distance to the others."""
    if len(clips) <= 2:
        return 0
    vecs = [compare_vector(c) for c in clips]
    d = np.zeros((len(vecs), len(vecs)))
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            d[i, j] = d[j, i] = dtw(vecs[i], vecs[j])
    return int(d.mean(1).argmin())


def trim_idle(pts: np.ndarray) -> np.ndarray:
    """Drop leading/trailing frames where no hand is visible (signer waiting to start)."""
    seen = np.isfinite(pts[:, :42, 0]).any(axis=1)
    if not seen.any():
        return pts
    a, b = np.flatnonzero(seen)[[0, -1]]
    return pts[max(0, a - 3) : b + 4]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-clips", type=int, default=12, help="clips compared per word (speed)")
    ap.add_argument("--out", default=str(MODELS / "signs.npz"))
    args = ap.parse_args(argv)

    by_word: dict[str, list] = defaultdict(list)
    for split in ("train", "val"):
        for rel in split_paths(split):
            p = cache_path(rel)
            if p.exists():
                by_word[label_from_path(rel)].append(p)
    if not by_word:
        raise SystemExit("No cached landmarks found. Run training.extract_landmarks first.")

    out: dict[str, np.ndarray] = {}
    firsts = []
    for word, paths in sorted(by_word.items()):
        clips, fpss = [], []
        for p in paths[: args.max_clips]:
            with np.load(p) as d:
                pts = normalized_points(d["frames"])
                clips.append(trim_idle(pts))
                fpss.append(float(d["fps"]))
        k = medoid(clips)
        motion = one_euro(fill_gaps(clips[k]), fpss[k])
        motion = resample_time(motion, fpss[k])
        motion[:, FACE, 2] = 0.0  # face depth is noisy and unused by the 2D avatar
        out[word] = motion.astype(np.float16)
        firsts.append(motion[0])
        print(f"{word:16s} {len(clips):2d} clips → medoid #{k}, {len(motion)} frames")

    rest = np.nanmedian(np.stack(firsts), axis=0).astype(np.float32)  # typical starting pose
    rest[:42] = np.nan  # rest pose: hands hidden (down), only body + face drawn
    assert rest.shape == (N_POINTS, 3)
    np.savez_compressed(args.out, __fps__=np.float32(OUT_FPS), __rest__=rest, **out)
    print(f"saved {len(out)} signs → {args.out}")


if __name__ == "__main__":
    main()
