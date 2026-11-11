# SANKET v1 — Product Requirements Document (PRD)

> **SANKET (संकेत)** = "sign / signal". Offline, two-way Indian Sign Language (ISL) ⇄ speech translator.
> Owner: Kanak Prabhakar · Status: v1 spec (2026-11-11) · Source of truth for v1 — update this file when a decision changes.

---

## 1. Overview

SANKET lets a **Deaf or mute person** and a **hearing person** (shopkeeper, customer, clerk, doctor) hold a **two-way conversation** on one device, fully **offline**:

- **Direction A — Sign → Voice:** the Deaf/mute person signs ISL in front of the camera → SANKET recognises the signs → shows Hindi + English text → speaks it aloud.
- **Direction B — Voice → Sign:** the hearing person speaks Hindi (or English) → SANKET transcribes it → converts it to ISL word order → a **2D signing avatar** signs it back, with large captions underneath.

v1 ships on **Windows (Microsoft Store)**. Android (Play Store) is v2 and reuses the same ONNX models and data files.

## 2. Problem

- India has roughly 300 certified ISL interpreters (ISLRTC figure, cited in the iSign benchmark paper) for millions of Deaf people.
- In day-to-day life (shop, bank counter, hospital desk, office) there is usually **no interpreter**, so communication falls back to writing on paper or guessing.
- Existing ISL tools are mostly hackathon/academic demos: small vocabulary, browser/Streamlit, one direction only, often need internet.
- The 2026 India AI Impact Summit work (live ISL avatar for the PM's speech, MIB sign avatars) covers **speech → sign for broadcasts**. The **sign → speech** direction for everyday face-to-face use remains open.

## 3. Goals (v1)

| # | Goal | Measured by |
|---|---|---|
| G1 | Two-way conversation on one Windows PC, 100% offline after install | Works with network adapter disabled |
| G2 | Recognise the 263 everyday ISL word signs of the INCLUDE dataset | Top-1 accuracy on INCLUDE test split ≥ **80%** (INCLUDE-50 ≥ 90%) |
| G3 | Real-time on an ordinary laptop CPU | Sign → text latency ≤ **1.5 s** after the sign ends; UI ≥ 24 fps on Ryzen 5 5600H CPU-only |
| G4 | Voice → avatar for Hindi speech | Every in-vocabulary word in a spoken sentence is signed; out-of-vocabulary words are shown as captions, never silently dropped |
| G5 | Honest output | Low-confidence signs show "Samajh nahi aaya — dobara karein" instead of a guess |
| G6 | Publishable | Passes Microsoft Store certification; MSIX with webcam + microphone capabilities |

## 4. Non-goals (v1)

- Continuous sentence-level ISL recognition (v1 recognises **isolated signs**, one word at a time, and builds the sentence from them).
- Fingerspelling recognition and fingerspelled avatar output (v2 — needs alphabet data).
- 3D photoreal avatar (v1 avatar is stylised 2D).
- Regional languages other than Hindi and English.
- Any cloud API, account, login, analytics or telemetry.
- Face lock / voice unlock (dropped — voice unlock cannot serve mute users).

## 5. Users

| User | Context | Needs |
|---|---|---|
| **Deaf / mute signer** (primary) | Knows ISL, may read basic Hindi/English | Be understood quickly; see clearly what the other person said |
| **Hearing person** (secondary) | Shopkeeper, customer, clerk; knows no ISL | Hear/read what was signed; speak normally and be understood |
| **Contributor** (later) | Deaf community member, ISL teacher, friends | Record/verify new signs with consent (data-collection mode) |

## 6. Core workflows

### W1 — Conversation (main screen)
1. App opens directly on **Conversation** screen. Camera + mic permission requested once.
2. Signer signs → live landmark overlay → recognised word appears as a **chip** with confidence colour.
3. Signer pauses ~1 s (hands down or still) → chips form a sentence → Hindi + English shown in big text → spoken aloud.
4. Hearing person presses and holds **Bolo / Speak** (or Space bar) → speaks → releases.
5. Transcript appears → ISL gloss shown → avatar signs each word in ISL order; captions highlight the current word.
6. Both turns are appended to the on-screen transcript (session only — cleared on exit).

### W2 — Learn / Dictionary
Browse all 263 signs by category, search Hindi/English, watch the avatar perform any sign at 0.5× / 1× speed.

### W3 — Settings
Camera choice, mic choice, voice (Hindi/English), avatar speed, confidence threshold, text size, high-contrast theme, language of UI (Hinglish/English).

### W4 — Contribute (hidden behind Settings → "Help improve SANKET", off by default)
Consent screen → record a chosen word 5× → saves **landmarks only (no video)** to a local folder the user can zip and share manually. Nothing is uploaded.

## 7. Functional requirements

**FR-A Recognition (Direction A)**
- FR-A1 Capture webcam at 640×480, 30 fps target.
- FR-A2 Extract per-frame landmarks: 2 hands × 21, 17 upper-body pose points, 20 selected face points (eyebrows, lips, nose).
- FR-A3 Segment signs automatically: a sign starts when either wrist rises above the hip line or motion energy exceeds threshold; it ends after 10 still frames or hands drop.
- FR-A4 Classify each segment with SANKET's own recognizer (ONNX); return top-3 with probabilities.
- FR-A5 If top-1 probability < threshold (default 0.60) → show "not understood", do not speak.
- FR-A6 Sentence closes after 1.2 s with no new sign, or on Enter key.
- FR-A7 Gloss sequence → Hindi + English sentence (rule-based, see TRD §6).
- FR-A8 Speak Hindi via an installed Windows Hindi voice; fall back to English voice with a one-time notice.

**FR-B Voice → Sign (Direction B)**
- FR-B1 Push-to-talk recording (button or Space), max 15 s.
- FR-B2 Offline speech-to-text (Vosk small Hindi model; English model selectable).
- FR-B3 Hindi text → ISL gloss: drop function words, keep SOV order, move question words to the end, map synonyms to dictionary glosses.
- FR-B4 Each in-vocabulary gloss → play its avatar clip; blend 6 frames between clips.
- FR-B5 Out-of-vocabulary words → shown in an orange caption chip "(no sign yet)" in sentence order.

**FR-C General**
- FR-C1 No network calls anywhere in the app (enforced by a test that blocks sockets).
- FR-C2 No camera frames or audio written to disk (except W4 landmarks, opt-in).
- FR-C3 KP orbit logo watermark (`assets/logo-mark.svg`) in the window corner and About screen.
- FR-C4 Full keyboard operation; every control has an accessible name.
- FR-C5 Visible "Preview — accuracy limited to 263 words" label in v1.

## 8. Edge cases

| Case | Expected behaviour |
|---|---|
| No camera / camera busy | Banner: camera not available + button to retry; Direction B still works |
| No microphone | Speak button disabled with tooltip; Direction A still works |
| Two people in frame | Use the person whose shoulders are largest/closest; ignore others |
| One hand out of frame | Still classify; if both hands missing for whole segment → discard segment |
| Low light / backlight | Show "Roshni kam hai" hint when hand-detection confidence stays low for 3 s |
| Signer signs too fast (no pause) | Segmenter also splits on a sharp drop of motion; worst case: user presses Enter |
| Left-handed signer | Supported (no mirroring in training; both hands are features) |
| Hindi voice not installed | English TTS + notice with steps to add the Windows Hindi speech pack |
| Speech contains numbers/names | Shown as caption chips (no sign) |
| Very long speech | Stop at 15 s, process what was recorded |
| Power cut / crash mid-session | Nothing to recover — session transcript is in memory only by design |

## 9. Acceptance criteria (v1 "done")

1. All FRs implemented; `pytest` passes; `ruff check` clean.
2. Recognizer: INCLUDE test top-1 ≥ 80%, INCLUDE-50 ≥ 90%, reported in `reports/metrics.json`.
3. ONNX vs PyTorch output max abs diff < 1e-4 on 50 test samples.
4. End-to-end run on the HP Victus (CPU only): sign → speech ≤ 1.5 s after sign end.
5. App runs with Wi-Fi off from a clean Windows user account.
6. MSIX installs, launches, asks for camera/mic, passes Windows App Cert Kit.
7. Kanak can explain every module (understanding gate): `docs/HOW_IT_WORKS.md` written in simple language.

## 10. Risks & honest limits (shown in README and Store listing)

- 263 words only; it is a **preview**, not a replacement for an interpreter.
- INCLUDE signers are from one school in Chennai; regional ISL variation will reduce accuracy in Jharkhand → fixed by community data in v1.x.
- Avatar signs are copied from INCLUDE recordings, not yet verified by a Deaf reviewer → Store listing must say "community verification in progress".
