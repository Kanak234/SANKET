# SANKET v1 — Start here

Offline two-way ISL ⇄ speech translator. This kit holds the full spec + the build prompt for Nemotron 3 Ultra (OpenCode).

## What's inside
| File | Purpose |
|---|---|
| `docs/PRD.md` | What v1 must do, users, workflows, edge cases, acceptance criteria |
| `docs/TRD.md` | Architecture, stack, model design, grammar rules, avatar, privacy, performance |
| `docs/UI_UX_BRIEF.md` | Screens, layout, states, accessibility |
| `docs/ROADMAP_AND_GATE21.md` | Phases P0–P6 + Visualize / Worst case / Victory |
| `docs/DATA_AND_LICENSES.md` | INCLUDE (CC BY 4.0) + every library licence + credit line |
| `AGENTS.md` | Coding rules (heavy comments, no torch at runtime, resumable jobs, git author = Kanak) |
| `OPENCODE_PROMPT.md` | The full build instruction Nemotron executes |

## Before running
Put your `logo-mark.svg` into `assets\` after cloning (optional — a placeholder is used if missing).

## One command (PowerShell, on the i9 desktop) — clone the repo + start the build
```powershell
git clone https://github.com/Kanak234/Sankat D:\Projects\sanket; Set-Location D:\Projects\sanket; git config user.name "Kanak Prabhakar"; git config user.email "kanakprabhakar72@gmail.com"; opencode run -m nvidia/nvidia/nemotron-3-ultra-550b-a55b "Read OPENCODE_PROMPT.md in this folder and execute every step in it fully, following AGENTS.md and docs/."
```

## After Nemotron finishes
1. Send Claude `docs/BUILD_REPORT.md` (or connect the `D:\Projects\sanket` folder) → Claude verifies the code.
2. Download the INCLUDE train/test split CSVs (Google Drive link on the Zenodo page) into `D:\datasets\INCLUDE\splits\`.
3. Run the data + training pipeline (resumable after power cuts):
```powershell
Set-Location D:\Projects\sanket; .\scripts\run_full_pipeline.ps1
```
4. Send Claude `reports\metrics.json` + `reports\confusion_matrix.png` → Claude checks against PRD G2.
5. Store: reserve the name in Partner Center → fill `packaging\store_identity.json` → `.\packaging\build_msix.ps1` → submit.
