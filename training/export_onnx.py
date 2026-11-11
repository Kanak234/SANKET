"""export_onnx.py — saves the trained PyTorch model as models/recognizer.onnx for the app.

WHY: the shipped app must not carry PyTorch (~2 GB). ONNX is a portable model file that
     ONNX Runtime runs on Windows now and on Android later — same file, no retraining.
WHAT: exports checkpoints/<subset>/best.pt with a dynamic batch size, then PROVES the export is
      faithful: runs 50 test clips through both PyTorch and ONNX Runtime and requires the
      largest difference in output to be < 1e-4 (PRD acceptance criterion 3).
NEXT: sanket/core/recognizer.py loads models/recognizer.onnx.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sanket.core.normalize import FEATURES, SEQ_LEN
from training.common import CHECKPOINTS, MODELS, REPORTS
from training.dataset import SignDataset
from training.model import SignTransformer


def export(ckpt_path: Path, out_path: Path) -> SignTransformer:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SignTransformer(n_classes=ckpt["n_classes"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    dummy = torch.zeros(1, SEQ_LEN, FEATURES)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (dummy,), str(out_path),
        input_names=["landmarks"], output_names=["logits"],
        dynamic_axes={"landmarks": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    return model


def parity(model: SignTransformer, onnx_path: Path, samples: np.ndarray) -> float:
    """Max absolute difference between PyTorch and ONNX Runtime logits on the same inputs."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(torch.from_numpy(samples)).numpy()
    got = sess.run(None, {"landmarks": samples})[0]
    return float(np.abs(ref - got).max())


def main(argv: list[str] | None = None) -> float:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="include", choices=["include", "include50"])
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=MODELS / "recognizer.onnx")
    args = ap.parse_args(argv)
    model = export(args.ckpt or CHECKPOINTS / args.subset / "best.pt", args.out)

    test = SignDataset("test", args.subset)
    if len(test):
        samples = np.stack([test[i][0].numpy() for i in range(min(50, len(test)))])
    else:  # no data available (e.g. CI) → random inputs still prove the export math
        samples = np.random.default_rng(0).normal(size=(50, SEQ_LEN, FEATURES)).astype(np.float32)
    diff = parity(model, args.out, samples)
    ok = diff < 1e-4
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "onnx_parity.json").write_text(json.dumps({"max_abs_diff": diff, "pass": ok}, indent=2))
    print(f"exported {args.out} ({args.out.stat().st_size / 1e6:.1f} MB); parity max diff {diff:.2e} → "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("ONNX parity check failed")
    return diff


if __name__ == "__main__":
    main()
