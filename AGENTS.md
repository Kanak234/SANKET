# AGENTS.md — rules for any coding agent working on SANKET

Read this file and every file in `docs/` before writing code. `docs/` is the source of truth. If something you need is not in the docs, write it in `docs/OPEN_QUESTIONS.md` and pick the most conservative option — do not silently invent requirements.

## Environment
- Windows 11, PowerShell. Project root: `D:\Projects\sanket`. Datasets: `D:\datasets\INCLUDE`.
- Python **3.12** in a `uv` venv at `.venv` (never system Python, never 3.14).
- GPU for training: NVIDIA RTX 5060 Ti 16 GB (CUDA). Runtime must work **CPU-only**.

## Code rules
1. **Heavy inline comments**: after each logical block write a comment explaining (a) why it exists, (b) what it does, (c) what it is used for, (d) where the data flows next. The owner must be able to explain every file from the comments alone.
2. Type hints everywhere; small functions; no global mutable state except Qt app singletons.
3. `sanket/core/normalize.py` is the ONLY feature implementation — training and runtime both import it.
4. Runtime (`sanket/`) must never import `torch`. Only `onnxruntime`, `numpy`, `mediapipe`, `vosk`, `pyttsx3`, `PySide6`.
5. No network access in runtime code. No telemetry. No secrets. No writing camera frames or audio to disk.
6. All long jobs (download, extraction, training) must be **resumable** (power cuts are frequent).
7. UI strings live in `sanket/i18n/*.json`, Hinglish in Roman script (not Devanagari) for UI labels; Hindi output sentences in Devanagari.
8. Every window shows the KP watermark from `assets/logo-mark.svg`. If the file is missing, use a placeholder and add a TODO in `docs/OPEN_QUESTIONS.md` — never draw your own version of the logo.
9. No pretrained weights for the recognizer. Random init only. No fine-tuning of any model.

## Git rules (strict)
- Author and committer: `Kanak Prabhakar <kanakprabhakar72@gmail.com>`.
- Never add "Co-Authored-By", "Generated with …", agent names, or session IDs to commits, PRs, README or code.
- Small commits with clear messages (`feat(core): add segmenter`, `test(grammar): …`).
- `data/`, `checkpoints/`, `.venv/`, `dist/`, `build/` are gitignored. `models/*.onnx` and `models/signs.npz` are committed via Git LFS (or GitHub Release assets if LFS is not set up — note it in OPEN_QUESTIONS).

## Definition of done for any task
- `uv run ruff check .` clean
- `uv run pytest -q` green
- `uv run python -m sanket --selftest` exits 0
- Report: files changed, what each does, how it was verified, anything NOT verified (say it plainly).
