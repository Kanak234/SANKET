# Third-party notices

SANKET itself is proprietary — Copyright © 2026 Kanak Prabhakar. All rights reserved.
It includes or uses the third-party components below. Each stays under its own licence; those
licences apply only to the component named, never to SANKET's own code, model or data files.

## Data

**INCLUDE — Indian Lexicon Sign Language Dataset** — CC BY 4.0
Sridhar, Ganesan, Kumar, Khapra. *INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition*, ACM Multimedia 2020. AI4Bharat / IIT Madras. https://zenodo.org/records/4010759
Signs recorded by Deaf students of St. Louis School for the Deaf, Adyar, Chennai.
SANKET trains its own model on landmarks extracted from these videos and its avatar replays landmark motions derived from them (changes: landmarks extracted, normalised, smoothed and resampled). The videos themselves are not distributed. Licence: https://creativecommons.org/licenses/by/4.0/

**INCLUDE train/val/test split lists and label map** — MIT License, Copyright (c) 2021 AI4Bharat (see `training/splits/LICENSE-AI4Bharat-INCLUDE-MIT.txt`).

## Models (used as shipped, not trained or fine-tuned by SANKET)

**MediaPipe Holistic Landmarker** (Google) — Apache License 2.0. https://github.com/google-ai-edge/mediapipe
**Vosk speech recognition toolkit and small Hindi / Indian-English models** (Alpha Cephei) — toolkit Apache License 2.0; each model's own licence file is reproduced in `third_party/` of the build. https://alphacephei.com/vosk/

## Libraries

| Component | Licence |
|---|---|
| Qt 6 / PySide6 (dynamically linked, unmodified) | LGPL-3.0 — source: https://code.qt.io ; you may replace the Qt libraries in the installed folder |
| ONNX Runtime | MIT |
| NumPy | BSD-3-Clause |
| OpenCV (opencv-python-headless) | Apache License 2.0 |
| MediaPipe Python package | Apache License 2.0 |
| vosk (Python) | Apache License 2.0 |
| sounddevice / PortAudio | MIT |
| pyttsx3 | MPL-2.0 (unmodified) |
| comtypes / pywin32 (via pyttsx3 on Windows) | MIT / PSF |

The full text of the Apache License 2.0 is available at https://www.apache.org/licenses/LICENSE-2.0, the LGPL-3.0 at https://www.gnu.org/licenses/lgpl-3.0.html, the MPL-2.0 at https://www.mozilla.org/MPL/2.0/ and the MIT License at https://opensource.org/license/mit.
