"""dataset.py — feeds cached INCLUDE landmarks to PyTorch.

WHY: PyTorch's training loop wants (features, label) pairs in batches. This file reads the .npz
     cache made by extract_landmarks.py, applies augmentation (train split only), and turns each
     clip into the exact (64, 239) input the app will later produce — via sanket.core.normalize,
     the same code the live app runs.
NEXT: train.py and evaluate.py wrap SignDataset in a DataLoader.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from sanket.core.normalize import model_input
from training.augment import augment
from training.common import cache_path, label_from_path, label_map, split_paths


class SignDataset(Dataset):
    """One item = one INCLUDE video → (tensor (64, 239), class index)."""

    def __init__(self, split: str, subset: str = "include", train: bool = False, seed: int = 0) -> None:
        self.labels = label_map()
        rels = split_paths(split, subset)
        # Keep only videos whose landmarks exist (failed/missing videos are listed in reports/).
        self.items = [(cache_path(r), self.labels[label_from_path(r)]) for r in rels if cache_path(r).exists()]
        self.missing = len(rels) - len(self.items)
        self.train = train
        self.rng = np.random.default_rng(seed)
        self._cache: dict[int, np.ndarray] = {}  # all clips fit in RAM (~400 MB) → no disk reads per epoch

    def __len__(self) -> int:
        return len(self.items)

    def frames(self, i: int) -> np.ndarray:
        if i not in self._cache:
            with np.load(self.items[i][0]) as d:
                self._cache[i] = d["frames"].astype(np.float32)
        return self._cache[i]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        f = self.frames(i)
        if self.train:
            # A per-worker RNG would repeat augmentations across workers; mixing in torch's
            # seed (unique per worker and epoch) keeps them different.
            rng = np.random.default_rng(torch.randint(0, 2**31 - 1, (1,)).item())
            f = augment(f, rng)
        return torch.from_numpy(model_input(f)), self.items[i][1]
