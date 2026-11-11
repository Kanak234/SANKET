# SANKET — Roadmap + Gate #21 (Visualize · Worst case · Victory)

## Roadmap

| Phase | What | Who | Done when |
|---|---|---|---|
| **P0** Setup | Repo `Kanak234/sanket`, venv, deps, docs in place | Nemotron (OpenCode) | `pytest` runs (empty), `ruff` clean |
| **P1** Code | All of `sanket/`, `training/`, `packaging/`, tests with synthetic fixtures | Nemotron | Tests green with fake data; `--selftest` passes with a dummy ONNX |
| **P2** Data | Download INCLUDE (56.8 GB → D:), extract landmarks | Kanak runs `scripts/run_full_pipeline.ps1` | `data/cache/` has ~4,292 `.npz` files |
| **P3** Train | Train recognizer from scratch, evaluate, export ONNX, build sign dictionary | same script (resumable) | metrics.json meets PRD G2 |
| **P4** Verify | Claude reviews code + metrics + runs checklist; Kanak tests on laptop with webcam | Claude + Kanak | PRD §9 all ticked |
| **P5** Ship | MSIX → Microsoft Store (Preview), GitHub release, 30–45 s reel | Kanak (Partner Center), Claude writes listing + reel script | Store live |
| **P6** Community | Social post asking Deaf signers / ISL teachers to help; school friends onboarded; contribute mode data | Kanak + team | First external signer verifies avatar signs |
| v2 | Android app (Kotlin + onnxruntime-mobile + MediaPipe Android), fingerspelling, more words | team | Play Store |

## 1. Visualize

- `training/evaluate.py` saves `reports/confusion_matrix.png`, per-category accuracy bar chart, and the 20 most-confused sign pairs → shows exactly which signs need more data.
- App has a **debug overlay** (F12): live landmarks, segment boundaries, top-3 probabilities, latency per stage in ms.
- README demo GIF: split screen sign → speech and speech → avatar.

## 2. Prepare for the worst

| Worst case | Prepared response |
|---|---|
| Power cut during 56.8 GB download | Downloader resumes per file, md5-verified |
| Power cut during training | Checkpoint every epoch, `--resume` restores RNG + optimizer |
| Accuracy stuck < 80% | Fallback: ship INCLUDE-50 vocabulary first (paper reports 94.5% possible on it); add more augmentation; check landmark extraction failures per class |
| Real users in Jharkhand sign differently | "Preview" label; contribute mode; per-signer calibration in v1.x |
| Microsoft Store rejects | Common causes pre-checked: privacy policy URL, webcam/mic capability justification, age rating, App Cert Kit run locally first |
| Wrong avatar sign offends/misleads | Report button in Dictionary ("This sign looks wrong") writes a local note; listing says "community verification in progress" |
| Laptop too slow | Face landmarks every 2nd frame; pose "lite"; reduce camera to 480p; DirectML EP |
| Nobody responds to the social-media call | Contact ISLRTC (islrtc.nic.in) workshops, Deaf schools in Ranchi/Hazaribagh, NGOs — list kept in `docs/OUTREACH.md` (v1.x) |
| Copyright / licence complaint | Only CC BY 4.0 data + OFL fonts + permissive libs; all credited in About and `DATA_AND_LICENSES.md` |

## 3. Prepare for victory

- **Scale-ready data format:** `signs.npz`, `lexicon.json`, `labels.json` are versioned; adding words = adding entries + retraining, no code change.
- **Android path ready:** runtime uses only ONNX + MediaPipe + Vosk, all available on Android.
- **Team-ready repo:** `CONTRIBUTING.md`, issue templates ("new sign request", "wrong sign"), good-first-issues list for school friends.
- **If it goes viral:** pinned GitHub Discussion for signers, a simple form (offline-collected `.npz` zipped and emailed) for contributions, roadmap issue for v2.
- **Funding/credibility:** metrics + Deaf-community endorsement make it eligible for accessibility grants/hackathons later.
