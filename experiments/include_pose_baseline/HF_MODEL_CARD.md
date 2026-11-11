---
license: other
license_name: proprietary-kanak-prabhakar
language:
  - en
  - hi
tags:
  - sign-language
  - indian-sign-language
  - isl
  - onnx
  - pose
  - gru
datasets:
  - INCLUDE
pipeline_tag: video-classification
---

# SANKET — ISL word recognizer (INCLUDE pose baseline)

This model recognizes isolated Indian Sign Language (ISL) words from body and hand keypoint sequences. It is part of **[SANKET](https://github.com/Kanak234/SANKET)**, an offline ISL ⇄ speech translator.

| | INCLUDE (262 words) |
|---|---|
| Test top-1 | **93.8%** |
| Test top-5 | **99.1%** |
| Test videos | 816 |

## Files
- `sanket_include.onnx`: the model. Input `seq` of shape **(1, 64, 136)**, output `logits` of shape (1, 262). Runs on CPU with ONNX Runtime.
- `config.json`: class names, `T`, keypoint indices, and normalization notes.
- `best.pt`: PyTorch state dict.

## Model
- BiGRU (2 layers, hidden 256, bidirectional) with attention pooling, 2.14M params.
- **Trained from scratch** (random init), with no pretrained weights.
- Input per frame: 67 keypoints (25 upper-body pose points + 21 + 21 hand points), giving (x, y) = 134 values, plus 2 hand-present flags.
- Normalization: each sequence is centered on the shoulder midpoint and scaled by shoulder width, using the median per sequence, then clipped to ±6. Missing points are set to 0.
- Each sequence is resampled to 64 frames.

## Use
See the live webcam demo (MediaPipe → this model → speech):
https://github.com/Kanak234/SANKET/tree/main/experiments/include_pose_baseline

## Limits
- Covers only isolated words (262), not continuous signing or fingerspelling.
- The training data comes from one recording setup. Accuracy on other webcams and other signers has not been measured yet.
- This is a research preview and is not a replacement for an interpreter.

## Data & credit
Trained on the **INCLUDE** dataset (Sridhar, Ganesan, Kumar, Khapra, ACM Multimedia 2020; AI4Bharat / IIT Madras), CC BY 4.0. Pose keypoints come from Zenodo record 6674324. The signs were recorded by Deaf students of St. Louis School for the Deaf, Adyar, Chennai.

## Licence
Copyright © 2026 Kanak Prabhakar. All rights reserved. See the SANKET repository LICENSE.
