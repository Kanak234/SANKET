"""fetch_models.py — downloads the frozen third-party models SANKET ships, ONCE, at build time.

WHY: the app must work fully offline, so these files are bundled inside the installer. They are
     downloaded here (by the developer / CI), never by the running app.
WHAT: models/mediapipe/holistic_landmarker.task  (Google MediaPipe, Apache 2.0)
      models/vosk/hi/  vosk-model-small-hi-0.22   (Alpha Cephei, offline Hindi speech-to-text)
      models/vosk/en/  vosk-model-small-en-in-0.4 (Alpha Cephei, offline Indian-English STT)
      Each model's own README/LICENSE is copied to third_party/<name>/ so THIRD_PARTY_NOTICES can
      quote it exactly. Re-running skips anything already present.
NEXT: packaging/sanket.spec bundles the models/ folder.
"""

from __future__ import annotations

import io
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"
THIRD = REPO / "third_party"
sys.path.insert(0, str(REPO))

from sanket.core.landmarks import HOLISTIC_URL  # noqa: E402  (after sys.path tweak)

VOSK = {
    "hi": "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip",
    "en": "https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip",
}


def get(url: str) -> bytes:
    import requests

    print(f"downloading {url}")
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    return r.content


def fetch_holistic() -> None:
    dest = MODELS / "mediapipe" / "holistic_landmarker.task"
    if dest.exists():
        print(f"ok   {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(get(HOLISTIC_URL))
    print(f"saved {dest}")


def fetch_vosk(lang: str, url: str) -> None:
    dest = MODELS / "vosk" / lang
    if (dest / "am").exists() or (dest / "conf").exists():
        print(f"ok   {dest}")
        return
    data = get(url)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root = z.namelist()[0].split("/")[0]  # zip contains one top folder, e.g. vosk-model-small-hi-0.22
        tmp = MODELS / "vosk" / f"_{lang}"
        if tmp.exists():
            shutil.rmtree(tmp)
        z.extractall(tmp)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(tmp / root), dest)
    shutil.rmtree(tmp, ignore_errors=True)
    notice_dir = THIRD / root
    notice_dir.mkdir(parents=True, exist_ok=True)
    for f in dest.iterdir():
        if f.is_file() and f.name.lower().startswith(("readme", "license", "copying", "notice")):
            shutil.copy(f, notice_dir / f.name)
    print(f"saved {dest} (licence files → {notice_dir})")


def main() -> None:
    fetch_holistic()
    for lang, url in VOSK.items():
        fetch_vosk(lang, url)
    print("all models present")


if __name__ == "__main__":
    main()
