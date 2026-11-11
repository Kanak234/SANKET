# ruff: noqa: E402
"""
SANKET — INCLUDE pose baseline: setup + train + export (Google Colab, T4 GPU, ek cell).

Result (2026-11-11, T4, ~5 min): INCLUDE 262 words -> TEST top-1 0.938 | top-5 0.991
                                 INCLUDE-50          -> TEST top-1 0.979 | top-5 1.000

Chalana: Colab -> Runtime -> T4 GPU -> naya cell me ye poori file paste karke run.
SUBSET = 'include50' (50 words, ~2 min) ya 'include' (262 words, ~5 min).

Flow: packages -> data (Zenodo 6674324, INCLUDE ke pose .pkl) -> CSV<->pkl matching -> normalize
      -> BiGRU train -> test -> ONNX (batch=1) export + 20 asli samples pe PyTorch se verify -> zip download

Data ke 3 sabak (is script me pehle se handle):
  1. Zenodo zip ke andar 2 aur zip hain (Pose_Signs.zip, Train_Test_Split.zip).
  2. CSV me .MOV path hai aur naam alag likhe hain ("Means of Transportation" vs
     "Means_of_Transportation", "GoodMorning" vs "Good Morning") -> nk() dono ko ek jaisa banata hai.
  3. Keypoints pixel units me, outliers ke saath -> kandhe se center/scale + clip ±6.
Normalization yahan aur sanket_core.py me HUBAHU same hai (tests/test_core.py isse check karta hai).
"""

import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "-q", "install", "onnx", "onnxscript", "onnxruntime"], check=True
)

import collections
import glob
import json
import os
import pickle
import random
import re
import shutil
import time
import zipfile

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn

# ---------------- CONFIG ----------------
SUBSET = "include"  # 'include50' = 50 words (fast) | 'include' = 262 words
T = 64  # har video 64 frames pe (median 61, 95th pct 95)
EPOCHS, LR = 120, 1e-3
BATCH, HID = (64, 256) if SUBSET == "include" else (32, 192)  # zyada classes -> bada model
ROOT = "/content/include_pose"
BASE = f"{ROOT}/INCLUDE"
POSE_ROOT = f"{BASE}/Pose_Signs/Pose_Signs"
SPLIT = f"{BASE}/Train_Test_Split/Train_Test_Split"
OUT = f"/content/sanket_{SUBSET}"
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu":
    print("⚠️ GPU nahi mila — Runtime > Change runtime type > T4 GPU")
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# ---------------- STEP 0: data (runtime reset hua ho tabhi download) ----------------
# --timeout/--tries: Zenodo connection kabhi latak jaata hai; ye khud retry karta hai.
if not os.path.isdir(POSE_ROOT):
    print("data download ho raha hai (632 MB, ~4-5 min)...")
    subprocess.run(
        "wget -q -c --timeout=30 --tries=5 -O /content/INCLUDE_pose.zip "
        '"https://zenodo.org/record/6674324/files/INCLUDE.zip?download=1"',
        shell=True,
        check=True,
    )
    zipfile.ZipFile("/content/INCLUDE_pose.zip").extractall(ROOT)
    os.remove("/content/INCLUDE_pose.zip")
    for z in glob.glob(f"{BASE}/*.zip"):  # zip ke andar zip
        zipfile.ZipFile(z).extractall(z[:-4])
        os.remove(z)
print("data ready:", len(glob.glob(f"{POSE_ROOT}/**/*.pkl", recursive=True)), "pkl files")

# Keypoint layout (verified 2026-11-11): 0-32 body | 33-53 hand A (wrist 15 ke paas) | 54-74 hand B (wrist 16)
KEEP = list(range(25)) + list(range(33, 75))  # pair (25-32) hataye -> 67 points
HA, HB = slice(25, 46), slice(46, 67)


# ---------------- STEP A: CSV rows -> .pkl ----------------
def clean(w):
    return re.sub(r"^\s*\d+\.\s*", "", str(w)).strip().lower()  # "51. Good Morning" -> "good morning"


def nk(s):
    return re.sub(r"[^a-z0-9]", "", clean(s))  # -> "goodmorning" (space/_/punctuation ka farak khatam)


pkl_by_key = {}
for p in glob.glob(f"{POSE_ROOT}/**/*.pkl", recursive=True):
    cat, word = os.path.relpath(p, POSE_ROOT).split(os.sep)[:2]
    pkl_by_key[(nk(cat), nk(word), os.path.splitext(os.path.basename(p))[0])] = (p, clean(word))


def load_split(name):
    # Key me word bhi hai, kyunki ek hi video-naam (MVI_0048) alag words me bhi milta hai.
    rows, miss = [], collections.Counter()
    for r in pd.read_csv(f"{SPLIT}/{name}.csv").itertuples():
        hit = pkl_by_key.get((nk(r.Category), nk(r.Word), os.path.splitext(os.path.basename(r.Video))[0]))
        if hit:
            rows.append(hit)
        else:
            miss["Extra/ (pose nahi bani)" if "/" in r.Video else "unmatched"] += 1
    print(f"{name}: {len(rows)} matched | missing: {dict(miss)}")
    return rows


train_rows, test_rows = load_split(f"train_{SUBSET}"), load_split(f"test_{SUBSET}")
classes = sorted({w for _, w in train_rows})
cls2id = {c: i for i, c in enumerate(classes)}
test_rows = [r for r in test_rows if r[1] in cls2id]
print("classes:", len(classes))


# ---------------- STEP B: normalization + augmentation + Dataset ----------------
def normalize(k):
    # sanket_core.normalize() ki hubahu copy — badlo to dono jagah badlo (test isse pakdega).
    k = k[:, KEEP, :2].astype(np.float32)
    pres = np.abs(k).sum(-1) > 0
    sh_c, sh_w = (k[:, 11] + k[:, 12]) / 2, np.linalg.norm(k[:, 11] - k[:, 12], axis=-1)
    ok = sh_w > 1e-6
    c = np.median(sh_c[ok], 0) if ok.any() else np.zeros(2, np.float32)
    s = np.median(sh_w[ok]) if ok.any() else 1.0
    k = np.clip((k - c) / s, -6, 6)
    k[~pres] = 0
    return k, pres


def frame_idx(n, train):
    # Train me random 80-100% crop = alag speed ke signs ki nakal. Eval me poora video.
    L, s = n, 0
    if train and n > 8:
        L = int(n * random.uniform(0.8, 1.0))
        s = random.randint(0, n - L)
    return np.linspace(s, s + L - 1, T).round().astype(int)


class SignDS(torch.utils.data.Dataset):
    def __init__(self, rows, train):
        self.train = train
        self.items = [(*normalize(pickle.load(open(p, "rb"))["keypoints"]), cls2id[w]) for p, w in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        k, pres, y = self.items[i]
        idx = frame_idx(len(k), self.train)
        k, pres = k[idx].copy(), pres[idx]
        if self.train:  # rotate/scale/shift/noise -> alag camera angle/distance; overfit rokta hai
            a = np.deg2rad(random.uniform(-15, 15))
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
            k = (k @ R.T) * random.uniform(0.9, 1.1) + np.random.uniform(-0.1, 0.1, 2).astype(np.float32)
            k += np.random.normal(0, 0.01, k.shape).astype(np.float32)
            k[~pres] = 0
        flags = np.stack([pres[:, HA].any(1), pres[:, HB].any(1)], 1).astype(np.float32)
        return torch.from_numpy(np.concatenate([k.reshape(T, -1), flags], 1)), y  # (64, 136)


random.shuffle(train_rows)
nv = max(1, len(train_rows) // 10)  # 10% val -> best checkpoint; test sirf aakhri report ke liye


def dl(rows, tr):
    return torch.utils.data.DataLoader(SignDS(rows, tr), BATCH, shuffle=tr, num_workers=2)


tr_dl, va_dl, te_dl = dl(train_rows[nv:], True), dl(train_rows[:nv], False), dl(test_rows, False)
D_IN = 67 * 2 + 2


# ---------------- STEP C: model (BiGRU + attention pooling, random init) ----------------
# BiGRU: chhote data (~15 videos/word) pe kam overfit, ONNX export aasaan, CPU pe real-time.
class SignGRU(nn.Module):
    def __init__(self, d_in, n_cls, h=HID):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.2))
        self.gru = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.att = nn.Linear(2 * h, 1)  # har frame ki importance
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * h, n_cls))

    def forward(self, x):
        z, _ = self.gru(self.inp(x))
        w = torch.softmax(self.att(z), 1)
        return self.head((w * z).sum(1))


model = SignGRU(D_IN, len(classes)).to(DEV)
print("params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * len(tr_dl))
crit = nn.CrossEntropyLoss(label_smoothing=0.1)


@torch.no_grad()
def evaluate(loader):
    model.eval()
    c1 = c5 = n = 0
    for x, y in loader:
        top = model(x.to(DEV)).topk(5, 1).indices.cpu()
        c1 += (top[:, 0] == y).sum().item()
        c5 += (top == y[:, None]).any(1).sum().item()
        n += len(y)
    return c1 / n, c5 / n


# ---------------- STEP D: training ----------------
best, t0 = -1, time.time()
for ep in range(1, EPOCHS + 1):
    model.train()
    tot = 0
    for x, y in tr_dl:
        loss = crit(model(x.to(DEV)), y.to(DEV))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        tot += loss.item()
    v1, v5 = evaluate(va_dl)
    if v1 >= best:
        best = v1
        torch.save(model.state_dict(), f"{OUT}/best.pt")
    if ep % 10 == 0 or ep == 1:
        print(
            f"ep {ep:3d} | loss {tot / len(tr_dl):.3f} | val top1 {v1:.3f} top5 {v5:.3f} | {time.time() - t0:.0f}s"
        )

# ---------------- STEP E: test ----------------
model.load_state_dict(torch.load(f"{OUT}/best.pt"))
t1, t5 = evaluate(te_dl)
print(f"\nTEST ({SUBSET}, {len(classes)} classes) -> top1 {t1:.3f} | top5 {t5:.3f}  (best val {best:.3f})")

# ---------------- STEP F: ONNX export (batch=1) + verify ----------------
# batch=1: GRU ka ONNX batch-size se bandh jaata hai; app ek waqt me ek sign chalata hai.
model.eval().cpu()
onnx_path = f"{OUT}/sanket_{SUBSET}.onnx"
torch.onnx.export(
    model,
    torch.zeros(1, T, D_IN),
    onnx_path,
    opset_version=17,
    dynamo=False,
    input_names=["seq"],
    output_names=["logits"],
)
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
same, diff = 0, 0.0
for i in random.sample(range(len(te_dl.dataset)), 20):  # 20 asli test videos
    x, _ = te_dl.dataset[i]
    x = x[None]
    with torch.no_grad():
        pt = model(x).numpy()
    ox = sess.run(None, {"seq": x.numpy()})[0]
    same += int(pt.argmax() == ox.argmax())
    diff = max(diff, float(np.abs(pt - ox).max()))
print(f"ONNX check: same prediction {same}/20 | max diff {diff:.2e}")  # 20/20 aur ~1e-6 hona chahiye

json.dump(
    {
        "classes": classes,
        "T": T,
        "keep": KEEP,
        "d_in": D_IN,
        "hidden": HID,
        "test_top1": t1,
        "test_top5": t5,
        "norm": "center=shoulder mid (11,12), scale=shoulder width, median per video, clip ±6",
    },
    open(f"{OUT}/config.json", "w"),
    indent=1,
)
shutil.make_archive(f"/content/sanket_{SUBSET}_model", "zip", OUT)
from google.colab import files  # Colab ke bahar ye line hata do

files.download(f"/content/sanket_{SUBSET}_model.zip")
