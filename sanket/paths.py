"""paths.py — one place that knows where SANKET's files live.

WHY: the app runs in three situations — from source (`python -m sanket`), as a PyInstaller
     build, and inside an MSIX package. Each puts files in a different folder. If every module
     guessed paths on its own, one of these three would break.
WHAT: exposes `resource()` for read-only shipped files (models, assets, i18n) and `user_dir()`
      for the few things SANKET may write (settings, opt-in contribution data).
NEXT: imported by recognizer.py, signs.py, lexicon.py, stt.py and the UI to open their files.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def app_root() -> Path:
    """Folder that contains `models/`, `assets/` and the `sanket/` package.

    WHY: PyInstaller unpacks bundled data next to the exe and sets `sys._MEIPASS`;
         from source, the repo root is two levels above this file.
    USE: base for every read-only resource path.
    """
    frozen_root = getattr(sys, "_MEIPASS", None)  # set only inside a PyInstaller build
    if frozen_root:
        return Path(frozen_root)
    return Path(__file__).resolve().parent.parent  # repo root when running from source


def resource(*parts: str) -> Path:
    """Path to a shipped, read-only file, e.g. resource("models", "lexicon.json").

    NEXT: callers open the returned path; nothing here checks existence so the caller can
          decide whether a missing file is fatal (recognizer) or just a warning (avatar).
    """
    return app_root().joinpath(*parts)


def user_dir() -> Path:
    """Writable per-user folder (%LOCALAPPDATA%\\SANKET on Windows, ~/.sanket elsewhere).

    WHY: MSIX installs are read-only, so settings and opt-in contribution files must go
         to the user's own app-data folder. Camera frames and audio are NEVER written here.
    """
    base = os.environ.get("LOCALAPPDATA")
    folder = Path(base) / "SANKET" if base else Path.home() / ".sanket"
    folder.mkdir(parents=True, exist_ok=True)  # create on first use so callers can just write
    return folder
