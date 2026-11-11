"""synthetic.py — fake but realistic-shaped landmark clips for tests.

WHY: the real INCLUDE dataset is 56.8 GB and cannot live in CI. These generators produce
     (T, 79, 3) clips with the same layout (shoulders, two hands, face) and a different motion per
     class, so the whole training → evaluation → ONNX → avatar chain can be proven end-to-end.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from sanket.core.normalize import FACE, LEFT_HAND, N_POINTS, POSE, RIGHT_HAND

REPO = Path(__file__).resolve().parent.parent
CLASSES = {  # rel-folder → gloss (gloss must exist in the real label map)
    "Greetings/1. hello": "hello",
    "Greetings/9. thank you": "thankyou",
    "Adjectives/2. good": "good",
}


def clip(kind: int, rng: np.random.Generator, t: int | None = None) -> np.ndarray:
    """One synthetic sign. kind 0 = right hand circles, 1 = both hands up-down, 2 = left hand sideways."""
    t = t or int(rng.integers(30, 55))
    f = np.full((t, N_POINTS, 3), np.nan, np.float32)
    cx, cy = 0.5 + rng.normal(0, 0.03), 0.45 + rng.normal(0, 0.03)
    w = 0.22 * rng.uniform(0.85, 1.15)
    pose = f[:, POSE]
    pose[:, 11] = (cx + w / 2, cy, 0)
    pose[:, 12] = (cx - w / 2, cy, 0)
    pose[:, 0] = (cx, cy - 0.25 * w * 4, 0)
    f[:, POSE] = pose
    f[:, FACE] = np.stack([np.full(20, cx), np.full(20, cy - w), np.zeros(20)], axis=1)
    ph = np.linspace(0, 2 * np.pi, t)
    hand_shape = rng.normal(0, 0.01, (21, 3)).astype(np.float32)
    if kind == 0:
        x, y = cx - 0.3 * w + 0.4 * w * np.cos(ph), cy + 0.5 * w + 0.4 * w * np.sin(ph)
        f[:, RIGHT_HAND] = np.stack([x, y, np.zeros(t)], 1)[:, None, :] + hand_shape
    elif kind == 1:
        y = cy + w * (0.2 + 0.6 * np.abs(np.sin(ph)))
        f[:, RIGHT_HAND] = np.stack([np.full(t, cx - 0.5 * w), y, np.zeros(t)], 1)[:, None, :] + hand_shape
        f[:, LEFT_HAND] = np.stack([np.full(t, cx + 0.5 * w), y, np.zeros(t)], 1)[:, None, :] + hand_shape
    else:
        x = cx + w * (0.2 + 0.8 * np.sin(ph))
        f[:, LEFT_HAND] = np.stack([x, np.full(t, cy + 0.6 * w), np.zeros(t)], 1)[:, None, :] + hand_shape
    return f + rng.normal(0, 0.002, f.shape).astype(np.float32)


def make_dataset(root: Path, per_class: dict[str, int] | None = None, seed: int = 0) -> dict[str, str]:
    """Write fake splits + cache under `root`. Returns env vars that point training/ at it."""
    per_class = per_class or {"train": 14, "val": 4, "test": 5}
    rng = np.random.default_rng(seed)
    splits, cache = root / "splits", root / "cache"
    splits.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "training" / "splits" / "label_map_include.json", splits / "label_map_include.json")
    for split, n in per_class.items():
        lines = []
        for k, folder in enumerate(CLASSES):
            for i in range(n):
                rel = f"{folder}/{split}_{i}.MOV"
                lines.append(rel)
                dest = (cache / rel).with_suffix(".npz")
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, frames=clip(k, rng), fps=np.float32(25))
        (splits / f"include_{split}.txt").write_text("\n".join(lines))
    return {
        "SANKET_SPLITS": str(splits),
        "SANKET_CACHE": str(cache),
        "SANKET_CHECKPOINTS": str(root / "ckpt"),
        "SANKET_REPORTS": str(root / "reports"),
        "SANKET_MODELS_OUT": str(root / "models"),
    }
