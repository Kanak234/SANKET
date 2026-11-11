"""common.py — small helpers shared by every training script.

WHY: download, extraction, training, evaluation and the sign dictionary all need to agree on
     (a) where data lives and (b) how a video path becomes a class label. Keeping that here means
     one fix fixes all scripts.
NEXT: imported by every other file in training/.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPLITS = Path(os.environ.get("SANKET_SPLITS", REPO / "training" / "splits"))  # official AI4Bharat lists (MIT)

# Data folders. Defaults match Kanak's i9 desktop (D: drive); override with environment variables.
DATA_ROOT = Path(os.environ.get("SANKET_DATA", r"D:\datasets\INCLUDE"))
ZIPS = DATA_ROOT / "zips"
VIDEOS = DATA_ROOT / "videos"
CACHE = Path(os.environ.get("SANKET_CACHE", REPO / "data" / "cache"))  # landmark .npz files
CHECKPOINTS = Path(os.environ.get("SANKET_CHECKPOINTS", REPO / "checkpoints"))
REPORTS = Path(os.environ.get("SANKET_REPORTS", REPO / "reports"))
LEXICON = REPO / "models" / "lexicon.json"  # read-only input (the shipped word table)
# Where trained artefacts go (recognizer.onnx / .json, signs.npz). Tests point this at a temp dir
# so they can never overwrite the real models.
MODELS = Path(os.environ.get("SANKET_MODELS_OUT", REPO / "models"))


def label_from_path(rel_path: str) -> str:
    """'Seasons/63. Winter/MVI_4997.MOV' → 'winter';  'Pronouns/9. you (plural)/x.MOV' → 'youplural'.

    RULE (checked against all 263 labels): take the word folder, drop the leading "NN. ",
    lowercase, keep only letters. This reproduces AI4Bharat's label_map_include.json exactly.
    """
    folder = Path(rel_path.replace("\\", "/")).parts[1]
    return re.sub(r"[^a-z]", "", re.sub(r"^\d+\.\s*", "", folder).lower())


def label_map() -> dict[str, int]:
    """gloss → class index, the official order (must equal models/lexicon.json indices)."""
    return json.loads((SPLITS / "label_map_include.json").read_text())


def split_paths(split: str, subset: str = "include") -> list[str]:
    """Relative video paths for 'train' / 'val' / 'test' from the official split files."""
    name = f"{subset}_{split}.txt"
    lines = (SPLITS / name).read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def cache_path(rel_path: str) -> Path:
    """Where the landmarks of one video are stored: data/cache/<Category>/<NN. Word>/<file>.npz."""
    return CACHE / Path(rel_path.replace("\\", "/")).with_suffix(".npz")
