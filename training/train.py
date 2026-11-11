"""train.py — trains SignTransformer from scratch on INCLUDE landmarks (resumable).

WHY resumable: power cuts at home are frequent and a full run takes ~30–60 min on the
     RTX 5060 Ti. A checkpoint (model + optimizer + scheduler + epoch + random-generator states)
     is written atomically after EVERY epoch, and `--resume` (default when a checkpoint exists)
     continues exactly where it stopped — not from zero.
WHAT: AdamW + 5-epoch warm-up + cosine decay, label smoothing 0.1, mixed precision on GPU,
      best model chosen by validation accuracy, early stop after 40 epochs without improvement.
      Writes checkpoints/<subset>/last.pt, best.pt and reports/train_log_<subset>.csv.
NEXT: evaluate.py measures best.pt on the TEST split; export_onnx.py turns it into
      models/recognizer.onnx.

Run:  uv run python -m training.train                 (full 263 words)
      uv run python -m training.train --subset include50 --epochs 120   (quick check)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from training.common import CHECKPOINTS, REPORTS
from training.dataset import SignDataset
from training.model import SignTransformer, count_params


def rng_state() -> dict:
    """Every random generator's state, so a resumed run continues the same random sequence."""
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_atomic(obj: dict, path: Path) -> None:
    """Write to a temp file then rename — a power cut mid-write can never corrupt last.pt."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def lr_lambda(epoch: int, warmup: int, total: int) -> float:
    """Learning-rate multiplier: linear warm-up, then cosine down to 2% of the base rate."""
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    """(accuracy, mean loss) over a loader — used for validation each epoch."""
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += ce(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return correct / max(1, total), loss_sum / max(1, total)


def main(argv: list[str] | None = None) -> float:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="include", choices=["include", "include50"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint and start over")
    ap.add_argument("--ckpt-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = args.ckpt_dir or CHECKPOINTS / args.subset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    last, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    done = ckpt_dir / "done.txt"  # written when training finished, so a re-run of the pipeline skips it
    if done.exists() and not args.fresh:
        print(f"training already finished ({done.read_text().strip()}); use --fresh to retrain")
        return float(done.read_text().split()[-1])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_ds = SignDataset("train", args.subset, train=True, seed=args.seed)
    val_ds = SignDataset("val", args.subset, train=False)
    if len(train_ds) == 0:
        raise SystemExit("No cached landmarks found. Run training.extract_landmarks first.")
    print(f"train {len(train_ds)} clips (missing {train_ds.missing}), val {len(val_ds)} clips, device {device}")
    pin = device == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                          pin_memory=pin, drop_last=len(train_ds) > args.batch, persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    model = SignTransformer(n_classes=len(train_ds.labels)).to(device)
    print(f"model parameters: {count_params(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: lr_lambda(e, args.warmup, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_epoch, best_acc, stale = 0, -1.0, 0
    if last.exists() and not args.fresh:  # ---- resume after a power cut
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        set_rng_state(ck["rng"])
        start_epoch, best_acc, stale = ck["epoch"] + 1, ck["best_acc"], ck["stale"]
        print(f"resumed from epoch {ck['epoch']} (best val acc {best_acc:.3f})")

    log_path = REPORTS / f"train_log_{args.subset}.csv"
    new_log = not log_path.exists() or start_epoch == 0
    with log_path.open("w" if new_log else "a", newline="") as fh:
        w = csv.writer(fh)
        if new_log:
            w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"])
        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            model.train()
            loss_sum = correct = seen = 0
            for x, y in train_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                    logits = model(x)
                    loss = ce(logits, y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stops rare exploding steps
                scaler.step(opt)
                scaler.update()
                loss_sum += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                seen += len(y)
            sched.step()
            val_acc, val_loss = evaluate_loader(model, val_dl, device)
            if val_acc > best_acc:
                best_acc, stale = val_acc, 0
                save_atomic({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc,
                             "n_classes": len(train_ds.labels)}, best_path)
            else:
                stale += 1
            save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                         "scaler": scaler.state_dict(), "rng": rng_state(), "epoch": epoch,
                         "best_acc": best_acc, "stale": stale}, last)
            lr = opt.param_groups[0]["lr"]
            w.writerow([epoch, f"{loss_sum / seen:.4f}", f"{correct / seen:.4f}", f"{val_loss:.4f}",
                        f"{val_acc:.4f}", f"{lr:.2e}", f"{time.time() - t0:.1f}"])
            fh.flush()
            print(f"epoch {epoch:3d}  train {correct / seen:.3f}  val {val_acc:.3f}  best {best_acc:.3f}  lr {lr:.1e}")
            if stale >= args.patience:
                print(f"early stop: no improvement for {args.patience} epochs")
                break
    print(f"best validation accuracy {best_acc:.3f} → {best_path}")
    done.write_text(f"best_val_acc {best_acc:.4f}")
    return best_acc


if __name__ == "__main__":
    main()
