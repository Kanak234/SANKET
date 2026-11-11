"""Windows PowerShell 5.1 reads .ps1 files without a BOM as ANSI, so any non-ASCII character
(e.g. an em dash) breaks string parsing — this happened on Kanak's PC with build_msix.ps1.
Keep every PowerShell and batch script pure ASCII."""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_windows_scripts_are_ascii():
    files = subprocess.check_output(["git", "ls-files", "*.ps1", "*.bat"], cwd=REPO, text=True).split()
    assert files
    bad = {f: sorted({c for c in (REPO / f).read_text(encoding="utf-8") if ord(c) > 127}) for f in files}
    assert not {f: c for f, c in bad.items() if c}, bad
