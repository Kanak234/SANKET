# Open questions

These are the things the docs don't decide yet. Until Kanak decides each one, the most conservative option stays in place.

## 1. KP watermark file missing (AGENTS.md rule 8)
- `assets/logo-mark.svg` is not in the repo.
- `experiments/include_pose_baseline/sanket_live.py` shows a text placeholder `[KP logo]` for now. It does not draw its own logo.
- **TODO:** add `assets/logo-mark.svg`, plus a transparent PNG export `assets/logo-mark.png` (OpenCV can't read SVG).

## 2. Should the INCLUDE pose baseline move into the main pipeline? (2026-11-11)
- The experiment in `experiments/include_pose_baseline/` reached **93.8% top-1** on INCLUDE (262 words). It was trained on Colab T4 in ~5 min from pre-extracted pose keypoints (632 MB), compared with the 56.8 GB video download for the main pipeline.
- It uses its own 75-point layout and normalization, which differ from `sanket/core/normalize.py` (Holistic, 79 points). For that reason it is **not merged** (rule 3).
- **Decide:** (a) keep both, (b) port the experiment's 75-point features into `sanket/core/normalize.py` and train the main model on Zenodo pose data, or (c) drop the experiment once the main model is trained.
- Before deciding, measure accuracy on a real webcam. That has not been tested yet.

## 3. Licence direction
- `LICENSE` currently says "all rights reserved". Kanak has said he wants SANKET to be open source, but no licence has been chosen yet (and he has said not Apache 2.0).
- Until he picks one, the existing proprietary LICENSE stays unchanged.

## 4. Model hosting
- The experiment model goes on Hugging Face (`KANAKq/sanket-isl-include`, created **private** by default) instead of Git LFS. Model files are gitignored.
