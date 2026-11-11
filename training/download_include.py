"""download_include.py — downloads the 56.8 GB INCLUDE dataset safely on a home connection.

WHY: Hazaribagh has frequent power cuts and the download takes hours. A normal browser download
     restarts from zero after a cut. This script resumes each file from the exact byte where it
     stopped (HTTP Range), checks every file's MD5, and skips files already verified.
WHAT: reads the file list + checksums from Zenodo's public API for record 4010759 (CC BY 4.0),
      downloads each zip into DATA_ROOT/zips, verifies it, then extracts it into DATA_ROOT/videos.
NEXT: extract_landmarks.py reads the extracted videos.

Run:  uv run python -m training.download_include            (all 15 categories)
      uv run python -m training.download_include --only Greetings Colours   (quick test)
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

from training.common import VIDEOS, ZIPS

RECORD_API = "https://zenodo.org/api/records/4010759"
CHUNK = 1 << 20  # 1 MB per write — small enough to lose little on a power cut


def md5sum(path: Path) -> str:
    """MD5 of a file (Zenodo publishes these, so we can prove the zip is complete and unchanged)."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def file_list(session) -> list[dict]:
    """[{name, url, size, md5}] for every zip in the record."""
    rec = session.get(RECORD_API, timeout=60)
    rec.raise_for_status()
    out = []
    for f in rec.json()["files"]:
        if not f["key"].endswith(".zip"):
            continue
        out.append({
            "name": f["key"],
            "url": f["links"]["self"],
            "size": int(f["size"]),
            "md5": f["checksum"].split(":", 1)[-1],
        })
    return sorted(out, key=lambda x: x["name"])


def download(session, item: dict) -> Path:
    """Download one zip with resume. Returns the verified local path."""
    from tqdm import tqdm

    dest = ZIPS / item["name"]
    done_marker = dest.with_suffix(".zip.ok")  # written only after the MD5 matched
    if done_marker.exists() and dest.exists():
        return dest
    have = dest.stat().st_size if dest.exists() else 0
    if have > item["size"]:  # corrupted/oversized leftover → start again
        dest.unlink()
        have = 0
    if have < item["size"]:
        headers = {"Range": f"bytes={have}-"} if have else {}
        with session.get(item["url"], headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            if have and r.status_code != 206:  # server ignored Range → restart cleanly
                have = 0
            mode = "ab" if have else "wb"
            with dest.open(mode) as f, tqdm(total=item["size"], initial=have, unit="B", unit_scale=True,
                                            desc=item["name"]) as bar:
                for block in r.iter_content(CHUNK):
                    f.write(block)
                    bar.update(len(block))
    if md5sum(dest) != item["md5"]:
        dest.unlink()  # bad file: delete so the next run downloads it fresh
        raise RuntimeError(f"MD5 mismatch for {item['name']} — deleted, please re-run")
    done_marker.write_text(item["md5"])
    return dest


def extract(zip_path: Path) -> None:
    """Unzip into VIDEOS (skipped if already extracted — marker file)."""
    marker = VIDEOS / f".{zip_path.stem}.extracted"
    if marker.exists():
        return
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(VIDEOS)
    marker.write_text("ok")


def main() -> None:
    import requests

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="category prefixes to download, e.g. Greetings Colours")
    args = ap.parse_args()
    ZIPS.mkdir(parents=True, exist_ok=True)
    VIDEOS.mkdir(parents=True, exist_ok=True)
    with requests.Session() as s:
        items = file_list(s)
        if args.only:
            items = [i for i in items if any(i["name"].startswith(p) for p in args.only)]
        total = sum(i["size"] for i in items) / 1e9
        print(f"{len(items)} files, {total:.1f} GB → {ZIPS}")
        for item in items:
            extract(download(s, item))
    print(f"Done. Videos in {VIDEOS}")


if __name__ == "__main__":
    main()
