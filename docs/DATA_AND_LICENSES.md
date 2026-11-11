# SANKET — Data & Licences

| Component | Source | Licence | How used | Status |
|---|---|---|---|---|
| INCLUDE ISL dataset (4,292 videos, 263 words, 15 categories, 56.8 GB) | AI4Bharat / IIT Madras — Zenodo record 4010759 | **CC BY 4.0** (verified on Zenodo 2026-11-11) | Train recognizer; medoid clips → avatar sign dictionary. Videos are never shipped — only derived landmarks | ✅ Allowed with credit |
| Credit line (mandatory) | — | — | "Signs recorded by Deaf students of St. Louis School for the Deaf, Adyar, Chennai. INCLUDE dataset — Sridhar, Ganesan, Kumar, Khapra (ACM MM 2020), AI4Bharat/IIT Madras, CC BY 4.0." | Show in About, Dictionary, README |
| MediaPipe Tasks models (hand, pose lite, face) | Google | Apache 2.0 | Frozen landmark extraction | ✅ |
| Vosk + small Hindi / English models | Alpha Cephei | Toolkit Apache 2.0; **model licence must be read from each model's README before bundling** | Frozen offline STT | ⚠️ Agent must verify and record here |
| onnxruntime | Microsoft | MIT | Inference | ✅ |
| PySide6 / Qt 6 | Qt | LGPLv3 (dynamic linking via PyInstaller onedir is fine) | UI | ✅ — ship LGPL notice + Qt licence text |
| pyttsx3 / Windows SAPI voices | — | MPL-2.0 / OS component | TTS | ✅ |
| Noto Sans, Noto Sans Devanagari | Google | SIL OFL 1.1 | Bundled fonts | ✅ |
| PyTorch | Meta | BSD-3 | Training only (not shipped) | ✅ |
| KP orbit logo | Kanak Prabhakar | Own (all rights reserved) | Watermark | ✅ |
| **SANKET code, recognizer model, lexicon.json, signs.npz, UI, docs** | Kanak Prabhakar | **Proprietary — all rights reserved** | The product | ✅ |

## Download (Windows, no bash)
`training/download_include.py` reads the file list + md5 from the Zenodo record, downloads each zip to `D:\datasets\INCLUDE\zips` with HTTP Range resume, verifies md5, extracts to `D:\datasets\INCLUDE\videos`, and skips anything already verified. Train/test split CSVs: from the Google Drive link on the Zenodo page (Kanak downloads once manually and puts them in `D:\datasets\INCLUDE\splits\`).

## User data
- No account, no analytics, no network.
- Camera/mic data processed in memory only.
- Contribute mode: landmarks `.npz` + consent record saved locally in `%LOCALAPPDATA%\SANKET\contrib\`; the user shares it manually.
- Privacy policy (required by Microsoft Store) = `docs/PRIVACY.md`, published on kanakprabhakar.online/sanket/privacy.

## SANKET's own licence (proprietary)
```
Copyright (c) 2026 Kanak Prabhakar. All rights reserved.

SANKET (source code, trained models, lexicon, sign dictionary data files, user interface,
artwork and documentation) is proprietary. No licence is granted to copy, modify, distribute,
sublicense, train on, or create derivative works from any part of it without prior written
permission from Kanak Prabhakar. Viewing the public repository does not grant any rights.

Third-party components bundled with SANKET remain under their own licences, listed in
THIRD_PARTY_NOTICES.md. Those licences apply only to those components, not to SANKET's own work.
```

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
