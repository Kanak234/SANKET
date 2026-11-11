"""Zip-by-zip Colab extraction: videos are matched to the official split names, already-cached
videos are skipped, parallel workers report back, and the temporary videos are deleted."""

import os
import zipfile
from pathlib import Path

import numpy as np

import training.colab_stream as cs
from training.common import split_paths


def stub_worker(jobs, out_q):
    """Stands in for MediaPipe: writes a tiny .npz per video (must be top-level for 'spawn')."""
    out = Path(os.environ["STUB_OUT"])
    for rel, path in jobs:
        if "bad" in Path(path).name.lower():
            out_q.put(("fail", rel, "decode error"))
            continue
        dest = (out / rel).with_suffix(".npz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dest, frames=np.zeros((3, 79, 3), np.float32), fps=np.float32(25))
        out_q.put(("ok", rel, ""))


def make_zip(tmp: Path, rels: list[str]) -> Path:
    z = tmp / "Greetings_1of2.zip"
    with zipfile.ZipFile(z, "w") as f:
        for rel in rels:
            f.writestr(rel, b"fake video")
        f.writestr("Greetings/README.txt", b"not a video")
    return z


def test_process_zip(tmp_path, monkeypatch):
    out = tmp_path / "cache"
    monkeypatch.setenv("STUB_OUT", str(out))
    monkeypatch.setattr(cs, "cache_path", lambda rel: (out / rel).with_suffix(".npz"))
    rels = [r for r in split_paths("train") if r.startswith("Greetings/")][:4]
    assert len(rels) == 4
    # one video already cached → must be skipped
    done = (out / rels[0]).with_suffix(".npz")
    done.parent.mkdir(parents=True, exist_ok=True)
    np.savez(done, frames=np.zeros((1, 79, 3)))
    z = make_zip(tmp_path, rels)

    ok, fails = cs.process_zip(z, tmp_path / "work", cs.split_lookup(), workers=2, worker_fn=stub_worker)
    assert ok == 3 and fails == []
    assert all((out / r).with_suffix(".npz").exists() for r in rels)
    assert not (tmp_path / "work" / z.stem).exists()  # videos deleted after extraction


def test_lookup_covers_all_splits():
    lookup = cs.split_lookup()
    assert len(lookup) == sum(len(split_paths(s)) for s in ("train", "val", "test"))


def test_main_loop_prefetch_and_resume(tmp_path, monkeypatch):
    """main(): every zip processed once, markers written to the (Drive) cache, re-run skips all."""
    import training.download_include as dl

    items = [{"name": f"Cat_{i}of3.zip"} for i in range(1, 4)]
    downloaded, processed = [], []

    def fake_download(item, *a, **k):
        downloaded.append(item["name"])
        p = dl.ZIPS / item["name"]
        p.write_bytes(b"zip")
        return p

    monkeypatch.setattr(cs, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(dl, "file_list", lambda s: items)
    monkeypatch.setattr(dl, "download", fake_download)
    monkeypatch.setattr(cs, "process_zip", lambda z, *a, **k: (processed.append(z.name) or (5, [])))
    cs.main(["--work", str(tmp_path / "work"), "--workers", "1"])
    assert downloaded == processed == [i["name"] for i in items]
    assert len(list((tmp_path / "cache" / "_zips_done").glob("*.done"))) == 3
    assert not list((tmp_path / "work" / "zips").glob("*.zip"))  # zips deleted
    downloaded.clear()
    cs.main(["--work", str(tmp_path / "work"), "--workers", "1"])  # session restarted
    assert downloaded == []
