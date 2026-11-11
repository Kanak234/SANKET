"""colab_stream.py — downloads INCLUDE and extracts landmarks ONE ZIP AT A TIME (for Google Colab).

WHY: on Kanak's home internet Zenodo gives ~67 kB/s → ~10 days for 56.8 GB. A Colab VM sits in a
     Google data centre and downloads far faster. But the full dataset (zips + extracted videos
     ≈ 114 GB) may not fit on a Colab disk, and a Colab session can disconnect. So this script:
       1. downloads one zip (with the same auto-retry/resume downloader as at home);
       2. unzips it into a temporary folder;
       3. runs MediaPipe Holistic on every video of that zip, in parallel on all CPU cores;
       4. writes each video's landmarks (.npz, ~100 KB) straight into the cache on Google Drive;
       5. deletes the zip and the videos, and writes a "done" marker on Drive;
     while the NEXT zip is already downloading in the background. Disk use stays ~3 GB.
     If the session dies, run it again: finished zips are skipped via their Drive markers.
WHAT IT PRODUCES: <SANKET_CACHE>/<Category>/<NN. Word>/<video>.npz — exactly the cache layout that
     extract_landmarks.py makes at home, so training/train.py uses it unchanged.
NEXT: the notebook packs the cache into one SANKET_cache.zip on Drive; at home it is unpacked into
      data/cache and `run_full_pipeline.ps1 -FromCache` trains on the RTX 5060 Ti.

Run (inside the notebook):  python -m training.colab_stream --workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import time
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from training.common import CACHE, cache_path, split_paths

VIDEO_EXT = (".mov", ".mp4", ".avi")


def split_lookup() -> dict[tuple[str, str], str]:
    """(word folder, file name) → official split path, for every train/val/test video.

    WHY: zips may extract with slightly different top folders; matching on the last two path
         parts (e.g. '63. winter', 'mvi_4997.mov') finds each video's official name reliably.
    """
    table = {}
    for split in ("train", "val", "test"):
        for rel in split_paths(split):
            parts = Path(rel.replace("\\", "/")).parts
            table[(parts[-2].lower(), parts[-1].lower())] = rel
    return table


def jobs_in(folder: Path, lookup: dict[tuple[str, str], str]) -> list[tuple[str, str]]:
    """(official rel path, local file) for every video in `folder` that is in a split and not cached yet."""
    jobs = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if not f.lower().endswith(VIDEO_EXT):
                continue
            p = Path(root) / f
            rel = lookup.get((p.parent.name.lower(), f.lower()))
            if rel and not cache_path(rel).exists():
                jobs.append((rel, str(p)))
    return sorted(jobs)


def run_parallel(jobs: list[tuple[str, str]], workers: int, worker_fn=None) -> tuple[int, list[str]]:
    """Extract landmarks for `jobs` using `workers` processes. Returns (ok count, failure lines).

    `worker_fn` defaults to extract_landmarks._worker (one MediaPipe per process); tests pass a stub.
    """
    if not jobs:
        return 0, []
    if worker_fn is None:
        from training.extract_landmarks import _worker as worker_fn
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    chunks = [jobs[i::workers] for i in range(workers)]
    procs = [ctx.Process(target=worker_fn, args=(c, q)) for c in chunks if c]
    for p in procs:
        p.start()
    ok, fails = 0, []
    for _ in range(len(jobs)):
        status, rel, err = q.get()
        if status == "ok":
            ok += 1
        else:
            fails.append(f"{rel},{err}")
    for p in procs:
        p.join()
    return ok, fails


def process_zip(zip_path: Path, work: Path, lookup, workers: int, worker_fn=None) -> tuple[int, list[str]]:
    """Unzip → extract landmarks → delete the videos. The zip itself is deleted by the caller."""
    tmp = work / zip_path.stem
    if tmp.exists():
        shutil.rmtree(tmp)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    try:
        return run_parallel(jobs_in(tmp, lookup), workers, worker_fn)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # free disk before the next zip


def main(argv: list[str] | None = None) -> None:
    import requests

    import training.download_include as dl

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2, help="parallel MediaPipe processes")
    ap.add_argument("--work", type=Path, default=Path("/content/sanket_work"), help="local temp folder")
    ap.add_argument("--only", nargs="*", help="zip name prefixes, e.g. Greetings (quick test)")
    args = ap.parse_args(argv)

    args.work.mkdir(parents=True, exist_ok=True)
    dl.ZIPS = args.work / "zips"  # zips go to the fast local disk, never to Drive
    dl.ZIPS.mkdir(parents=True, exist_ok=True)
    markers = CACHE / "_zips_done"  # on Drive → survives a disconnected session
    markers.mkdir(parents=True, exist_ok=True)
    fail_log = CACHE / "_extract_failures.csv"

    with requests.Session() as s:
        items = dl.file_list(s)
    if args.only:
        items = [i for i in items if any(i["name"].startswith(p) for p in args.only)]
    todo = [i for i in items if not (markers / f"{i['name']}.done").exists()]
    print(f"{len(items)} zips total, {len(items) - len(todo)} already done, {len(todo)} to go → cache {CACHE}")
    lookup = split_lookup()

    t0 = time.time()
    total_ok = 0
    # One background thread downloads the NEXT zip while this thread runs MediaPipe on the current one.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending: Future | None = pool.submit(dl.download, todo[0]) if todo else None
        for n, item in enumerate(todo):
            zip_path = pending.result()  # wait for this zip (already downloading in the background)
            pending = pool.submit(dl.download, todo[n + 1]) if n + 1 < len(todo) else None
            ok, fails = process_zip(zip_path, args.work, lookup, args.workers)
            total_ok += ok
            if fails:
                with fail_log.open("a", encoding="utf-8") as fh:
                    fh.write("\n".join(fails) + "\n")
            zip_path.unlink(missing_ok=True)
            zip_path.with_suffix(".zip.ok").unlink(missing_ok=True)
            (markers / f"{item['name']}.done").write_text(f"{ok} ok, {len(fails)} failed")
            mins = (time.time() - t0) / 60
            print(f"[{n + 1}/{len(todo)}] {item['name']}: {ok} videos ok, {len(fails)} failed · {mins:.0f} min so far",
                  flush=True)
    cached = sum(1 for _ in CACHE.rglob("*.npz"))
    print(f"\nDONE. {total_ok} new landmark files this run, {cached} in the cache in total.")


if __name__ == "__main__":
    main()
