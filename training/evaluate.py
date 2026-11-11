"""evaluate.py — honest scorecard of the trained model + the confidence threshold the app uses.

WHY: PRD goal G2 says the model must reach ≥ 80% top-1 on the INCLUDE TEST split (videos it
     never saw during training). This script measures that, shows WHERE it fails (Gate #21:
     Visualize), and calibrates the "not sure → ask again" threshold on the VALIDATION split.
WHAT: writes
      reports/metrics.json          top-1/top-5 test accuracy, per-category accuracy, PASS/FAIL
      reports/confusion_matrix.png  which signs get mixed up
      reports/per_category.png      accuracy bar per INCLUDE category
      models/recognizer.json        {threshold, ...} read by sanket/core/recognizer.py
NEXT: export_onnx.py; Kanak sends metrics.json + the PNGs to Claude for review.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.common import CHECKPOINTS, LEXICON, MODELS, REPORTS
from training.dataset import SignDataset
from training.model import SignTransformer

GOAL = {"include": 0.80, "include50": 0.90}  # PRD G2


@torch.no_grad()
def predict(model, ds: SignDataset, device: str) -> tuple[np.ndarray, np.ndarray]:
    """All softmax probabilities (N, C) and true labels (N,) for a dataset."""
    probs, ys = [], []
    for x, y in DataLoader(ds, batch_size=128, shuffle=False):
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def calibrate_threshold(probs: np.ndarray, ys: np.ndarray, reject_correct: float = 0.05) -> float:
    """Threshold such that only ~5% of CORRECT validation predictions would be rejected.

    WHY: too high → the app keeps saying "dobara karein" for right answers; too low → it speaks
         wrong words. Rejecting at most 5% of right answers is the PRD's chosen balance.
    """
    conf = probs.max(1)
    right = probs.argmax(1) == ys
    if right.sum() < 5:
        return 0.6
    return float(np.clip(np.quantile(conf[right], reject_correct), 0.3, 0.9))


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="include", choices=["include", "include50"])
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt or CHECKPOINTS / args.subset / "best.pt", map_location=device, weights_only=False)
    model = SignTransformer(n_classes=ckpt["n_classes"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_ds, test_ds = SignDataset("val", args.subset), SignDataset("test", args.subset)
    vp, vy = predict(model, val_ds, device)
    tp, ty = predict(model, test_ds, device)
    threshold = calibrate_threshold(vp, vy)

    top1 = float((tp.argmax(1) == ty).mean()) if len(ty) else 0.0
    top5 = float(np.mean([y in np.argsort(-p)[:5] for p, y in zip(tp, ty, strict=True)])) if len(ty) else 0.0
    accepted = tp.max(1) >= threshold
    acc_when_accepted = float((tp.argmax(1) == ty)[accepted].mean()) if accepted.any() else 0.0

    idx2gloss = {v: k for k, v in val_ds.labels.items()}
    lex = json.loads(LEXICON.read_text(encoding="utf-8"))["entries"]
    per_cat: dict[str, list[bool]] = defaultdict(list)
    confusions: dict[tuple[str, str], int] = defaultdict(int)
    for p, y in zip(tp, ty, strict=True):
        g, pred = idx2gloss[int(y)], idx2gloss[int(p.argmax())]
        per_cat[lex[g]["category"]].append(pred == g)
        if pred != g:
            confusions[(g, pred)] += 1
    per_cat_acc = {c: float(np.mean(v)) for c, v in sorted(per_cat.items())}
    worst = sorted(confusions.items(), key=lambda kv: -kv[1])[:20]

    goal = GOAL[args.subset]
    metrics = {
        "subset": args.subset,
        "test_clips": int(len(ty)),
        "test_top1": round(top1, 4),
        "test_top5": round(top5, 4),
        "threshold": round(threshold, 4),
        "test_accept_rate": round(float(accepted.mean()) if len(ty) else 0.0, 4),
        "test_top1_when_accepted": round(acc_when_accepted, 4),
        "goal_top1": goal,
        "result": "PASS" if top1 >= goal else "FAIL",
        "per_category_top1": per_cat_acc,
        "most_confused": [{"true": a, "predicted": b, "count": n} for (a, b), n in worst],
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODELS / "recognizer.json").write_text(json.dumps(
        {"threshold": metrics["threshold"], "test_top1": metrics["test_top1"], "subset": args.subset}, indent=2))
    if not args.no_plots:
        _plots(tp, ty, per_cat_acc, len(val_ds.labels))
    print(json.dumps({k: metrics[k] for k in ("test_top1", "test_top5", "threshold", "result")}, indent=2))
    return metrics


def _plots(tp, ty, per_cat_acc, n_classes) -> None:
    """Confusion matrix + per-category bars (Gate #21: Visualize)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.zeros((n_classes, n_classes), np.float32)
    for p, y in zip(tp.argmax(1), ty, strict=True):
        cm[y, p] += 1
    cm /= np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cm, cmap="viridis")
    ax.set_title("SANKET recognizer — confusion matrix (rows = true sign)")
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(list(per_cat_acc), list(per_cat_acc.values()), color="#0F6E6E")
    ax.axhline(0.8, color="#B45309", linestyle="--", label="goal 80%")
    ax.set_ylim(0, 1)
    ax.set_ylabel("test top-1 accuracy")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "per_category.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
