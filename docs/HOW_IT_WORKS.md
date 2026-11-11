# How SANKET works — explained file by file (understanding gate)

Read this once top to bottom and you can explain every part of SANKET yourself.

## The big picture
```
SIGN → VOICE   camera → landmarks.py → segmenter.py → recognizer.py → grammar.py (A) → tts.py
VOICE → SIGN   mic → stt.py → grammar.py (B) → signs.py → ui/avatar_widget.py
```
Nothing goes to the internet. The only model we TRAIN is the recognizer. MediaPipe (body points) and Vosk (speech-to-text) are used exactly as shipped.

## Direction A — sign to voice

| Step | File | What happens | Why this way |
|---|---|---|---|
| 1 | `core/landmarks.py` | MediaPipe Holistic finds 79 points per frame: 21 per hand, 17 upper-body, 20 face (brows, lips). x and z are multiplied by width/height. | The model works on points, not pixels → small, fast, privacy-friendly. The width/height fix makes a 16:9 video and a 4:3 webcam give the same shape. |
| 2 | `core/segmenter.py` | Watches the points. A sign starts when a hand is raised to chest height and moves; it ends after ~0.3 s of stillness or when hands drop. | INCLUDE has one sign per video, so the model expects one sign at a time. |
| 3 | `core/normalize.py` | Centres every point on the shoulder midpoint, divides by shoulder width, adds 2 "hand seen" flags, stretches the sign to 64 frames → (64, 239). | Same numbers no matter where the person stands or how fast they sign. Training uses this SAME file, so numbers always match. |
| 4 | `core/recognizer.py` | Runs `models/recognizer.onnx` → probabilities for 263 words → top-3. Below the confidence threshold → "Samajh nahi aaya". | Never guesses silently. Threshold is calibrated on validation data so only ~5% of right answers get rejected. |
| 5 | `core/grammar.py` (A) | Joins words in ISL order (which is also Hindi order) and adds "hoon/hai/hain" + "am/is/are" when the sentence ends in an adjective. | ISL has no "is/the"; a few rules make normal Hindi/English. |
| 6 | `core/tts.py` | Speaks with the Windows Hindi voice (falls back to English and tells the user how to add Hindi). | Offline, 0 MB extra. |

## Direction B — voice to sign

| Step | File | What happens | Why |
|---|---|---|---|
| 1 | `core/stt.py` | Push-to-talk (Space). Vosk small Hindi model turns audio into text. Max 15 s, audio stays in memory. | Offline, ~40 MB, also exists for Android. |
| 2 | `core/grammar.py` (B) | Matches longest phrases first ("aap kaise hain" → HOWAREYOU), drops "hai/ka/ko…", turns "kal" into YESTERDAY if the sentence is in past tense, moves question words to the end, unknown words become caption chips. | ISL grammar, and nothing is ever silently dropped. |
| 3 | `core/signs.py` | Looks up each word's motion in `models/signs.npz`, blends 6 frames between signs, holds a rest pose for words without a sign. | Motions are REAL recordings by Deaf signers (INCLUDE), not invented. |
| 4 | `ui/avatar_widget.py` | Draws torso, arms, detailed hands, brows and lips at 30 fps and underlines the caption word being signed. | 2D is fast and readable; hands are drawn biggest because they carry the meaning. |

## Training (only on your GPU desktop)

| File | Job |
|---|---|
| `training/download_include.py` | Downloads 56.8 GB with resume + MD5 check (power-cut safe). |
| `training/extract_landmarks.py` | Runs the same `HolisticExtractor` over every video once → small `.npz` files. Parallel, resumable, failures logged. |
| `training/augment.py` | Random rotation, stretch, speed change, frame drops, jitter, hand loss. No mirroring (left/right matters in ISL). |
| `training/model.py` | Own transformer, ~1.9 M parameters, random init (no fine-tuning). |
| `training/train.py` | AdamW + warm-up + cosine, label smoothing, mixed precision, checkpoint every epoch, auto-resume. |
| `training/evaluate.py` | Test accuracy vs the 80% goal, per-category chart, confusion matrix, threshold calibration. |
| `training/export_onnx.py` | Saves ONNX and proves PyTorch and ONNX give the same answer (diff < 1e-4). |
| `training/build_sign_dictionary.py` | For each word picks the most typical real clip (medoid by DTW), fills tiny gaps, smooths, 30 fps. |

## Edge cases already handled
Camera missing/busy (banner + retry, voice side still works) · no mic or no Hindi model (Speak disabled with reason) · one hand out of frame · nobody in frame · low light hint · signer never pauses (4 s cap) · hand left raised after a sign (no re-trigger) · low confidence (ask again, tap to pick from top-3) · Hindi voice missing (English + notice) · names/numbers in speech (caption chips) · power cut during download/extraction/training (resume) · model files missing (DEMO mode, never fake results).

## Known limits (be honest in demos)
263 isolated words only (no continuous sentences, no fingerspelling yet) · INCLUDE signers are from one school in Chennai, Jharkhand signers may differ → contribute mode + community data · avatar signs not yet checked by a Deaf reviewer.
