# SANKET — Start here

## 1. Train the model + build everything (i9 desktop, RTX 5060 Ti) — one command

```powershell
git clone https://github.com/Kanak234/SANKET D:\Projects\sanket; Set-Location D:\Projects\sanket; .\scripts\run_full_pipeline.ps1
```

What it does (every step resumes after a power cut — just run the same command again):
install deps → fetch MediaPipe + Vosk → download INCLUDE (56.8 GB) → extract landmarks → train
from scratch → evaluate (PASS/FAIL vs 80% goal) → export ONNX (parity check) → build avatar
signs → self-test.

Then review `reports\metrics.json`, `reports\confusion_matrix.png` and `reports\per_category.png`.

## 2. Try the app

```powershell
Set-Location D:\Projects\sanket; uv run python -m sanket
```
Space (hold) = speak · Enter = finish sentence · Esc = clear · Ctrl+F = swap sides · F12 = debug timings.

## 3. Build the Store package

```powershell
Set-Location D:\Projects\sanket; .\packaging\build_msix.ps1
```
`packaging/store_identity.json` already holds the Partner Center identity (Store ID 9NH6TTQJQTGJ).
Upload `dist\SANKET_1.0.0.0_x64.msix` in Partner Center → SANKET → Packages.

## Where things are
| Folder | What |
|---|---|
| `sanket/core` | all logic: landmarks, normalisation, segmenter, recognizer, grammar, speech, avatar data |
| `sanket/ui` | PySide6 screens |
| `training` | download, extraction, model, training, evaluation, export, sign dictionary |
| `tests` | 99 tests incl. a full synthetic train → export → app chain |
| `docs` | PRD, TRD, UI/UX brief, roadmap + Gate 21, licences, how it works, Store listing, privacy |
