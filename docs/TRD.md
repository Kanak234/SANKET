# SANKET v1 — Technical Requirements Document (TRD)

> Pairs with `PRD.md`. Every decision here is final for v1 unless changed in this file (add an ADR line in §12).

---

## 1. Architecture

```
                   ┌──────────────────────── SANKET (Windows, PySide6, offline) ────────────────────────┐
 Webcam ─► capture ─► landmarks (MediaPipe Tasks) ─► normalise ─► segmenter ─► recognizer.onnx ─► gloss[]
                                                                                                  │
                                                                     gloss→text (rules) ◄─────────┘
                                                                          │
                                                          captions (hi/en) + Windows TTS ─► 🔊

 Mic ─► push-to-talk ─► Vosk STT (offline) ─► text→gloss (rules) ─► sign_dictionary ─► avatar renderer ─► 🧍
                                                                     (landmark clips)     (QPainter 2D)
 └────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 Offline training side (Kanak's i9 + RTX 5060 Ti):
 INCLUDE videos ─► extract_landmarks.py ─► .npz cache ─► train.py (PyTorch, from scratch) ─► export_onnx.py
                                                     └─► build_sign_dictionary.py ─► signs.npz
```

The **runtime never imports PyTorch** — only `onnxruntime`, so the same `recognizer.onnx` + `signs.npz` + `lexicon.json` move unchanged to Android in v2.

## 2. Stack

| Layer | Choice | Why |
|---|---|---|
| Language | **Python 3.12** (uv-managed venv) | PyTorch wheels exist for 3.12, not 3.14 |
| UI | **PySide6** (Qt 6) | Kanak already ships PySide6 apps (MEDHĀ); native, fast QPainter for avatar |
| Landmarks | **MediaPipe Tasks — HolisticLandmarker** (one model: pose + face + both hands), used frozen | Same model exists on Android; one tracker keeps hands, body and face in sync |
| Training | **PyTorch 2.x + CUDA** on RTX 5060 Ti 16 GB | From-scratch model, no pretrained weights |
| Inference | **onnxruntime** (CPU EP; DirectML EP optional) | Portable to Android (onnxruntime-mobile) |
| STT | **Vosk** + `vosk-model-small-hi` (+ small English model) — used **as-is, no training** | Offline, ~50 MB, has Android SDK |
| TTS | Windows SAPI via `pyttsx3` (Hindi voice "Kalpana/Hemant" if installed) | Offline, zero size |
| Packaging | PyInstaller (onedir) → **MSIX** (makeappx + AppxManifest) → Microsoft Store | Store signs MSIX; same path used for Likhit/MEDHĀ |
| Tests / lint | pytest, ruff | — |

**Model rules check:** the recognizer is SANKET's own model, trained from random init (no fine-tuning). It is a separate product model, outside the 3 research models / TARKA. MediaPipe and Vosk are used frozen as off-the-shelf components; nothing is fine-tuned.

## 3. Repository layout

```
sanket/
├─ AGENTS.md                  # rules for any coding agent (read first)
├─ README.md                  # star-gate README (demo GIF, features, credits, limits)
├─ pyproject.toml             # deps + ruff + pytest config
├─ assets/  logo-mark.svg, app icons, fonts
├─ data/                      # NOT in git (gitignored): raw INCLUDE, landmark cache
├─ models/                    # shipped artefacts: recognizer.onnx, signs.npz, lexicon.json, labels.json
├─ docs/                      # PRD, TRD, UI_UX_BRIEF, ROADMAP_AND_GATE21, DATA_AND_LICENSES, HOW_IT_WORKS
├─ sanket/
│  ├─ app.py                  # entry point
│  ├─ core/
│  │  ├─ capture.py           # webcam thread (QThread), frame queue
│  │  ├─ landmarks.py         # MediaPipe wrapper → (79,3) array per frame
│  │  ├─ normalize.py         # shoulder-centred, shoulder-width-scaled features (shared by train + runtime)
│  │  ├─ segmenter.py         # sign start/end detection
│  │  ├─ recognizer.py        # onnxruntime session, resample to 64 frames, top-k
│  │  ├─ grammar.py           # gloss→hi/en sentence, hi/en text→gloss
│  │  ├─ stt.py               # Vosk push-to-talk
│  │  ├─ tts.py               # pyttsx3 wrapper, voice selection
│  │  └─ signs.py             # sign dictionary loader + clip blending
│  ├─ ui/
│  │  ├─ main_window.py, conversation_view.py, avatar_widget.py,
│  │  ├─ dictionary_view.py, settings_view.py, contribute_view.py, theme.py
│  └─ i18n/ hinglish.json, english.json
├─ training/
│  ├─ download_include.py     # resumable, md5-checked downloader (Windows-friendly, no bash)
│  ├─ extract_landmarks.py    # videos → data/cache/*.npz (multiprocess, resumable)
│  ├─ dataset.py, augment.py, model.py, train.py (checkpoint every epoch, --resume)
│  ├─ evaluate.py             # writes reports/metrics.json + confusion matrix PNG
│  ├─ export_onnx.py          # + parity check
│  └─ build_sign_dictionary.py
├─ packaging/ sanket.spec, AppxManifest.xml (template), build_msix.ps1
├─ scripts/ run_full_pipeline.ps1   # one command: download→extract→train→eval→export→dictionary
└─ tests/
```

## 4. Landmark features (shared contract)

Per frame → array `(79, 3)` float32, missing points = 0 and a mask bit:
- Hands: left 21 + right 21 = 42 (MediaPipe handedness, **no mirroring**)
- Pose upper body: indices 0–16 = 17
- Face: 20 points — eyebrows (10), outer lips (8), nose tip, chin
- Aspect fix (in `landmarks.py`): x and z are multiplied by image W/H so 16:9 INCLUDE videos and 4:3 webcams give identical shapes.
- Normalisation: origin = midpoint of shoulders (pose 11, 12); scale = shoulder distance; z scaled the same.
- Extra per-frame features: hand-present flags (2). Final per-frame vector = 79×3 + 2 = **239**.
- Sequence: resample (linear interpolation) to **T = 64** frames.

`normalize.py` is imported by both `training/` and `sanket/` — one implementation only.

## 5. Recognizer model (own, from scratch)

- Input `(B, 64, 239)` → Linear 239→192 → + learned positional embedding → [CLS] token
- 4× Transformer encoder layers (d=192, 6 heads, FF 384, dropout 0.2, pre-norm)
- CLS → LayerNorm → Linear 192→263
- ~2 M parameters, ONNX ≈ 8 MB.
- Training: AdamW lr 3e-4, wd 0.05, cosine schedule, warmup 5 epochs, 200 epochs, batch 64, label smoothing 0.1, mixed precision.
- Augmentation: rotation ±15°, scale 0.85–1.15, shift ±0.1, time-warp 0.8–1.2, random frame drop 10%, Gaussian jitter σ=0.01, random hand drop (p=0.1).
- Split: INCLUDE official `train.csv` / `test.csv`; 10% of train held out for validation (by signer if signer IDs available, else random).
- **Power-cut safety:** checkpoint (`model`, `optimizer`, `scheduler`, `epoch`, RNG states) every epoch to `checkpoints/last.pt`; `--resume` continues exactly.
- Reject: softmax top-1 < threshold → "unknown". Threshold calibrated on validation so that ≤ 5% of correct predictions are rejected.

## 6. Grammar rules (v1, rule-based, in `grammar.py`)

**Gloss → sentence:** ISL and Hindi are both SOV, so the gloss order is kept. `lexicon.json` holds, per gloss: `hi`, `en`, `pos`, `synonyms_hi[]`, `synonyms_en[]`. Rules: pronoun handling (I→मैं, you→आप), add "है/हैं" for adjective/noun predicates, question gloss at end → "?" Example: `YOU NAME WHAT` → "आपका नाम क्या है?" / "What is your name?"

**Text → gloss:** tokenise → lower/normalise → drop stopwords (है, हैं, था, का, की, के, को, ने, से, में, पर, तो, भी, a, an, the, is, am, are, to, of) → map each token through synonym index → move question words (क्या, कहाँ, कब, कौन, क्यों, कैसे / what, where, when, who, why, how) to the end → unknown tokens become `TEXT:<word>` chips.

Both directions are unit-tested with ≥ 30 sentence pairs in `tests/test_grammar.py`.

## 7. Sign dictionary (avatar data)

- For each of the 263 glosses: take all INCLUDE training clips → normalise → resample to each clip's natural length (keep timing) → choose the **medoid** clip (min mean DTW distance to others) → smooth with a One-Euro filter → store.
- `signs.npz`: `{gloss: float16 array (T,79,3)}` + `fps`. Target size < 15 MB.
- Credit: "Signs recorded by Deaf students of St. Louis School for the Deaf, Adyar, Chennai — INCLUDE dataset, AI4Bharat/IIT Madras, CC BY 4.0" shown in About + Dictionary.

## 8. Avatar renderer

`avatar_widget.py`, QPainter at 30 fps: torso + head silhouette from pose points, rounded-capsule arms, detailed hands (21 points, finger bones as tapered lines, palm polygon filled), eyebrows + mouth lines from face points. Two colour themes (light/dark, high contrast). Between clips: 6-frame linear blend. Speed 0.5× / 1×. No external 3D engine in v1.

## 9. Security & privacy

- No sockets: `tests/test_offline.py` monkeypatches `socket.socket` to raise and runs a full pipeline smoke test.
- No frames/audio to disk; contribute mode stores landmarks `.npz` only after explicit consent (checkbox + date).
- No secrets anywhere (there are no keys in this project).
- MSIX capabilities: `webcam`, `microphone`, `runFullTrust` only.

## 10. Performance budget (CPU only, Ryzen 5 5600H)

| Stage | Budget |
|---|---|
| MediaPipe (hands + pose lite + face) | ≤ 25 ms/frame (run face every 2nd frame if needed) |
| Recognizer ONNX (64×239) | ≤ 15 ms per segment |
| Vosk small hi, 5 s audio | ≤ 1 s |
| Avatar paint | ≤ 8 ms/frame |

## 11. Verification

`pytest` (unit + offline + ONNX parity + grammar), `ruff check .`, `python -m sanket --selftest` (loads every model, runs one synthetic sign and one synthetic sentence, exits 0), `evaluate.py` metrics vs PRD G2, Windows App Cert Kit on the MSIX.

## 12. Decision log (ADR-lite)

- 2026-11-11 — Own model, from scratch; no Sarvam/cloud API; fully offline. (Kanak)
- 2026-11-11 — v1 vocabulary = INCLUDE 263 words; 2D avatar; Windows first, Android v2. (defaults taken by Claude; Kanak can change)
- 2026-11-11 — Face lock & voice unlock dropped.
- 2026-11-11 — Vosk + Windows TTS used as-is (no training) for speech parts.
- 2026-11-11 — Built by Claude directly (Kanak's decision), not by an external agent. MediaPipe Holistic chosen over three separate landmarkers.
- 2026-11-11 — Official AI4Bharat INCLUDE split lists + label map (MIT) vendored in `training/splits/`, so no manual Google-Drive download is needed.
- 2026-11-11 — Partner Center name reserved: SANKET, Store ID 9NH6TTQJQTGJ, identity KanakPrabhakar.SANKET.
