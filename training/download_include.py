"""download_include.py — downloads the 56.8 GB INCLUDE dataset safely on a home connection.

WHY: Hazaribagh has frequent power cuts and the download takes hours. A normal browser download
     restarts from zero after a cut. This script resumes each file from the exact byte where it
     stopped (HTTP Range), checks every file's MD5, and skips files already verified.
WHAT: reads the file list + checksums from Zenodo's public API for record 4010759 (CC BY 4.0),
      downloads each zip into DATA_ROOT/zips, verifies it, then extracts it into DATA_ROOT/videos.
NEXT: extract_landmarks.py reads the extracted videos.

Run:  uv run python -m training.download_include            (all 15 categories, 6 files at a time)
      uv run python -m training.download_include --only Greetings Colours   (quick test)
"""

from __future__ import annotations

import argparse
import hashlib
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from training.common import VIDEOS, ZIPS

RECORD_API = "https://zenodo.org/api/records/4010759"
CHUNK = 1 << 20  # 1 MB blocks for MD5 hashing
READ_CHUNK = 64 * 1024  # network read size: a dropped connection loses at most 64 KB of progress


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


def _fetch_once(session, item: dict, dest: Path, bar) -> None:
    """One attempt: continue the file from the byte already on disk until it is complete.

    Raises on any network problem; the caller decides whether to retry.
    """
    have = dest.stat().st_size if dest.exists() else 0
    if have >= item["size"]:
        return
    headers = {"Range": f"bytes={have}-"} if have else {}
    # timeout=(connect, read): a stalled connection fails after 60 s instead of hanging forever
    with session.get(item["url"], headers=headers, stream=True, timeout=(30, 60)) as r:
        r.raise_for_status()
        if have and r.status_code != 206:  # server ignored Range → start this file again
            have = 0
            bar.reset(total=item["size"])
        with dest.open("ab" if have else "wb") as f:
            for block in r.iter_content(READ_CHUNK):
                f.write(block)
                bar.update(len(block))


def download(item: dict, retries: int = 200, position: int = 0) -> Path:
    """Download one zip, retrying broken connections automatically, then verify its MD5.

    WHY: Zenodo connections from home internet often drop after a few MB (seen on Kanak's PC:
         'IncompleteRead' after 7 MB). Each retry continues from the last byte written
         (HTTP Range), waits a little longer each time (2 s → 60 s), and the retry counter
         resets whenever progress was made — so only a truly dead link gives up.
    """
    import requests
    from tqdm import tqdm

    dest = ZIPS / item["name"]
    done_marker = dest.with_suffix(".zip.ok")  # written only after the MD5 matched
    if done_marker.exists() and dest.exists():
        return dest
    if dest.exists() and dest.stat().st_size > item["size"]:  # corrupted/oversized leftover
        dest.unlink()
    have = dest.stat().st_size if dest.exists() else 0
    failures, wait = 0, 2.0
    with requests.Session() as session, tqdm(total=item["size"], initial=have, unit="B", unit_scale=True,
                                             desc=item["name"][:28].ljust(28), position=position,
                                             leave=True) as bar:
        while True:
            before = dest.stat().st_size if dest.exists() else 0
            try:
                _fetch_once(session, item, dest, bar)
                break
            except (requests.RequestException, OSError) as exc:
                after = dest.stat().st_size if dest.exists() else 0
                if after > before:  # progress was made → this is a flaky line, not a dead link
                    failures, wait = 0, 2.0
                failures += 1
                if failures > retries:
                    raise RuntimeError(f"{item['name']}: giving up after {retries} retries ({exc})") from exc
                bar.set_postfix_str(f"retry {failures} in {wait:.0f}s")
                time.sleep(wait)
                wait = min(wait * 2, 60.0)
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
    ap.add_argument("--parallel", type=int, default=6,
                    help="files downloaded at the same time (a slow server is faster with several connections)")
    ap.add_argument("--retries", type=int, default=200, help="retries per file without any progress")
    args = ap.parse_args()
    ZIPS.mkdir(parents=True, exist_ok=True)
    VIDEOS.mkdir(parents=True, exist_ok=True)
    with requests.Session() as s:
        items = file_list(s)
    if args.only:
        items = [i for i in items if any(i["name"].startswith(p) for p in args.only)]
    total = sum(i["size"] for i in items) / 1e9
    done = sum(1 for i in items if (ZIPS / i["name"]).with_suffix(".zip.ok").exists())
    print(f"{len(items)} files, {total:.1f} GB → {ZIPS}  ({done} already complete, {args.parallel} at a time)")
    errors = []
    # Downloads run in parallel threads; each finished zip is extracted right away on this thread,
    # so disk work overlaps with the remaining downloads.
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(download, item, args.retries, n % args.parallel): item
                   for n, item in enumerate(items)}
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                extract(fut.result())
            except Exception as exc:  # one bad file must not stop the others
                errors.append(f"{item['name']}: {exc}")
    if errors:
        print("\nSome files failed (re-run the same command to continue them):")
        for e in errors:
            print("  " + e)
        raise SystemExit(1)
    print(f"Done. Videos in {VIDEOS}")


if __name__ == "__main__":
    main()
