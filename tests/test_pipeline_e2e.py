"""End-to-end proof of the training chain on a synthetic 3-class dataset:
train (from scratch) → resume → evaluate → export ONNX (parity) → sign dictionary → app recognizer.
Runs each step as a real subprocess with env vars pointing at temp folders, exactly like the
PowerShell pipeline does on Kanak's desktop — and can never overwrite the real models/."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.synthetic import CLASSES, clip, make_dataset

REPO = Path(__file__).resolve().parent.parent
pytest.importorskip("torch")


def run(mod, env, *args):
    r = subprocess.run([sys.executable, "-m", mod, *args], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0, f"{mod} failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}"
    return r.stdout


def test_full_chain(tmp_path):
    env = {**os.environ, **make_dataset(tmp_path), "PYTHONPATH": str(REPO)}
    out = run("training.train", env, "--epochs", "6", "--batch", "16", "--workers", "0", "--warmup", "1")
    assert "best validation accuracy" in out
    # resume: a second run with more epochs must continue, not restart
    (tmp_path / "ckpt" / "include" / "done.txt").unlink()
    out = run("training.train", env, "--epochs", "30", "--batch", "16", "--workers", "0", "--warmup", "1")
    assert "resumed from epoch 5" in out
    metrics = json.loads(run("training.evaluate", env, "--no-plots").split("\n", 1)[0] and
                         (tmp_path / "reports" / "metrics.json").read_text())
    assert metrics["test_clips"] == 3 * 5
    assert metrics["test_top1"] >= 0.9, metrics  # 3 very different motions → must be learnable
    out = run("training.export_onnx", env)
    assert "PASS" in out
    run("training.build_sign_dictionary", env)
    signs = np.load(tmp_path / "models" / "signs.npz")
    assert set(CLASSES.values()) <= set(signs.files)

    # The exported model, used through the APP's recognizer, must recognise new synthetic clips.
    from sanket.core.recognizer import Recognizer

    rec = Recognizer(tmp_path / "models" / "recognizer.onnx", threshold=0.0)
    rng = np.random.default_rng(123)
    hits = sum(rec.predict(clip(k, rng)).gloss == g for k, g in enumerate(CLASSES.values()) for _ in range(4))
    assert hits >= 10, hits
