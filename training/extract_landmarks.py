"""extract_landmarks.py — runs MediaPipe over every INCLUDE video once and caches the points.

WHY: decoding 4,292 HD videos and running MediaPipe on them takes hours; training runs for many
     epochs. Doing it once and saving small .npz files (~100 KB each) makes training fast and lets
     us re-train any number of times. It uses the SAME HolisticExtractor as the live app, so the
     model learns from exactly the numbers it will see from a webcam.
WHAT: for each video in the official train/val/test lists → data/cache/<...>.npz containing
        frames: (T, 79, 3) float32 (NaN = point not found), fps: float
      Resumable (skips finished files), parallel (one MediaPipe per worker process), and every
      failure is logged to reports/extract_failures.csv instead of stopping the run.
NEXT: training/dataset.py loads these files.

Run:  uv run python -m training.extract_landmarks --workers 6
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from training.common import REPORTS, VIDEOS, cache_path, split_paths

MAX_SIDE = 960  # downscale HD frames before MediaPipe: same landmarks, ~3x faster


def find_video(rel: str, index: dict[tuple[str, str], Path]) -> Path | None:
    """Locate a video even if the zip extracted with a slightly different top folder."""
    direct = VIDEOS / rel
    if direct.exists():
        return direct
    parts = Path(rel.replace("\\", "/")).parts
    return index.get((parts[-2].lower(), parts[-1].lower()))


def build_index() -> dict[tuple[str, str], Path]:
    """(word folder, file name) → path, for every video under VIDEOS (fallback lookup)."""
    idx = {}
    for root, _dirs, files in os.walk(VIDEOS):
        for f in files:
            if f.lower().endswith((".mov", ".mp4", ".avi")):
                p = Path(root) / f
                idx[(p.parent.name.lower(), f.lower())] = p
    return idx


def video_to_points(path: Path, extractor) -> tuple[np.ndarray, float]:
    """Decode one video and return (T, 79, 3) landmarks + its frame rate."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = bgr.shape[:2]
        if max(h, w) > MAX_SIDE:  # keeps aspect ratio, so landmarks.py's W/H correction stays valid
            s = MAX_SIDE / max(h, w)
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(extractor.process(rgb, int(i * 1000 / fps)))
        i += 1
    cap.release()
    if not frames:
        raise RuntimeError("no frames decoded")
    return np.stack(frames).astype(np.float32), float(fps)


def _worker(jobs: list[tuple[str, str]], out_q) -> None:
    """One process = one MediaPipe instance; results/errors go back through the queue."""
    from sanket.core.landmarks import HolisticExtractor

    for rel, path in jobs:
        try:
            with HolisticExtractor() as ex:  # fresh tracker per video (VIDEO mode keeps state)
                pts, fps = video_to_points(Path(path), ex)
            dest = cache_path(rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, frames=pts, fps=np.float32(fps))
            os.replace(tmp, dest)  # atomic: a power cut never leaves a half-written .npz
            out_q.put(("ok", rel, ""))
        except Exception as exc:  # keep going; record what failed
            out_q.put(("fail", rel, repr(exc)))


def main() -> None:
    from tqdm import tqdm

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--subset", default="include", choices=["include", "include50"])
    args = ap.parse_args()

    rels = sorted({p for s in ("train", "val", "test") for p in split_paths(s, args.subset)})
    todo = [r for r in rels if not cache_path(r).exists()]
    print(f"{len(rels)} videos in splits, {len(rels) - len(todo)} already cached, {len(todo)} to do")
    if not todo:
        return
    index = build_index()
    jobs, missing = [], []
    for r in todo:
        p = find_video(r, index)
        (jobs.append((r, str(p))) if p else missing.append(r))
    REPORTS.mkdir(parents=True, exist_ok=True)
    fail_log = REPORTS / "extract_failures.csv"
    with fail_log.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for r in missing:
            w.writerow([r, "video file not found — is the download complete?"])
        chunks = [jobs[i :: args.workers] for i in range(args.workers)]
        q = mp.get_context("spawn").Queue()
        procs = [mp.get_context("spawn").Process(target=_worker, args=(c, q)) for c in chunks if c]
        for p in procs:
            p.start()
        fails = 0
        for _ in tqdm(range(len(jobs)), desc="landmarks"):
            status, rel, err = q.get()
            if status == "fail":
                fails += 1
                w.writerow([rel, err])
                fh.flush()
        for p in procs:
            p.join()
    print(f"done: {len(jobs) - fails} ok, {fails} failed, {len(missing)} missing (see {fail_log})")


if __name__ == "__main__":
    main()
