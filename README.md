# SANKET (संकेत)

**Offline, two-way Indian Sign Language ⇄ speech translator.**
A Deaf or mute person signs in front of the camera and SANKET speaks it aloud in Hindi or English. The hearing person replies by voice, and an on-screen avatar signs the reply back in ISL. Everything runs on the device — no internet, no account.

> **Status:** v1 in development (specification stage). Windows first, Android next.

## How it works
- **Sign → Voice:** camera → hand/pose/face landmarks → SANKET's own sign recognizer (trained from scratch) → Hindi/English text → offline speech.
- **Voice → Sign:** microphone → offline speech-to-text → ISL word order → 2D signing avatar with captions.

See [`docs/PRD.md`](docs/PRD.md), [`docs/TRD.md`](docs/TRD.md) and [`docs/UI_UX_BRIEF.md`](docs/UI_UX_BRIEF.md) for the full specification.

## v1 scope (honest limits)
- 263 everyday ISL word signs (INCLUDE dataset vocabulary) — a **preview**, not a replacement for an interpreter.
- Avatar signs are pending verification by Deaf community reviewers.

## Credits
Signs recorded by Deaf students of St. Louis School for the Deaf, Adyar, Chennai — INCLUDE dataset (Sridhar, Ganesan, Kumar, Khapra, ACM MM 2020), AI4Bharat / IIT Madras, CC BY 4.0.

## Want to help?
Deaf signers, ISL teachers and developers are welcome. Contributions are accepted only under the [contributor agreement](docs/DATA_AND_LICENSES.md#contributor-agreement-for-friends--community-contributors).

## Licence
Copyright (c) 2026 Kanak Prabhakar. All rights reserved. See [`LICENSE`](LICENSE).
