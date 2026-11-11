# SANKET — Data & Licences

| Component | Source | Licence | How used | Status |
|---|---|---|---|---|
| INCLUDE ISL dataset (4,292 videos, 263 words, 15 categories, 56.8 GB) | AI4Bharat / IIT Madras — Zenodo record 4010759 | **CC BY 4.0** (verified on Zenodo 2026-11-11) | Train recognizer; medoid clips → avatar sign dictionary. Videos are never shipped — only derived landmarks | ✅ Allowed with credit |
| Credit line (mandatory) | — | — | "Signs recorded by Deaf students of St. Louis School for the Deaf, Adyar, Chennai. INCLUDE dataset — Sridhar, Ganesan, Kumar, Khapra (ACM MM 2020), AI4Bharat/IIT Madras, CC BY 4.0." | Show in About, Dictionary, README |
| MediaPipe Holistic Landmarker model | Google | Apache 2.0 | Frozen landmark extraction | ✅ |
| INCLUDE split lists + label map | AI4Bharat (GitHub) | MIT | Official train/val/test split | ✅ — `training/splits/LICENSE-AI4Bharat-INCLUDE-MIT.txt` |
| Vosk + small Hindi / English models | Alpha Cephei | Toolkit Apache 2.0; each model's README/licence is copied to `third_party/` by `scripts/fetch_models.py` and must be checked before the first Store submission | Frozen offline STT | ⚠️ check once after first fetch |
| onnxruntime | Microsoft | MIT | Inference | ✅ |
| PySide6 / Qt 6 | Qt | LGPLv3 (dynamic linking via PyInstaller onedir is fine) | UI | ✅ — ship LGPL notice + Qt licence text |
| pyttsx3 / Windows SAPI voices | — | MPL-2.0 / OS component | TTS | ✅ |
| Noto Sans, Noto Sans Devanagari | Google | SIL OFL 1.1 | Bundled fonts | ✅ |
| PyTorch | Meta | BSD-3 | Training only (not shipped) | ✅ |
| KP orbit logo | Kanak Prabhakar | Own (all rights reserved) | Watermark | ✅ |
| **SANKET code, recognizer model, lexicon.json, signs.npz, UI, docs** | Kanak Prabhakar | **MIT** (open source) | The product | ✅ |

## Download (Windows, no bash)
`training/download_include.py` reads the file list + md5 from the Zenodo record, downloads each zip to `D:\datasets\INCLUDE\zips` with HTTP Range resume, verifies md5, extracts to `D:\datasets\INCLUDE\videos`, and skips anything already verified. Train/val/test split lists and the label map come from AI4Bharat's INCLUDE GitHub repo (MIT) and are already in `training/splits/` — no manual download.

## User data
- No account, no analytics, no network.
- Camera/mic data processed in memory only.
- Contribute mode: landmarks `.npz` + consent record saved locally in `%LOCALAPPDATA%\SANKET\contrib\`; the user shares it manually.
- Privacy policy (required by Microsoft Store) = `docs/PRIVACY.md`, published on kanakprabhakar.online/sanket/privacy.

## SANKET's own licence (MIT)

Since 2026-11-12 SANKET's code, model, lexicon and docs are released under the MIT licence (see `LICENSE`). Kanak chose open source so researchers, deaf organisations and students can use and improve it. Exception: the KP orbit logo stays all rights reserved. Third-party components keep their own licences.

## Why third-party licences stay as they are
MediaPipe and Vosk are Apache 2.0, Qt is LGPLv3, INCLUDE is CC BY 4.0. SANKET cannot relicense them; it only uses them and must ship their notices. Their licences give nobody any right over SANKET's own code or model. Replacing them would mean building an own hand/pose landmark model and an own speech recogniser — possible later, not in v1.

## Contributor agreement (for friends / community contributors)
```
By submitting code, data (including sign recordings or landmark files) or any other material to
SANKET, you confirm that it is your own work or you have the right to submit it, and you assign
all copyright and related rights in that contribution to Kanak Prabhakar, who may use, modify and
license it without restriction. Sign-data contributors also confirm they consented to being
recorded and may ask for their data to be removed from future versions.
```
Every contributor must agree in writing (GitHub PR checkbox or signed form) before their work is merged.
