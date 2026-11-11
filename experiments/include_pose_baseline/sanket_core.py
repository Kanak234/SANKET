"""
SANKET core — webcam ke bina test hone wala saara logic yahan hai.

Flow (har frame):
  MediaPipe landmarks  ->  frame_to_75()  (75 points, pixel coords, training jaisa layout)
                       ->  Segmenter.push()  (haath utha = sign shuru, haath gira = sign khatam)
  Sign khatam hone par ->  Predictor.predict(frames)
                       ->  build_input()  (normalize + 64 frames, BILKUL training jaisa)
                       ->  ONNX model  ->  top-k words + probability

Is file ko alag isliye rakha: sanket_live.py sirf camera/UI sambhale, aur yahan ka har function
tests/test_core.py se bina webcam ke verify ho sake.
"""

import json
import os
import urllib.request
from collections import deque

import numpy as np

# ------------------------------------------------------------------------------------------------
# Layout constants — training (Colab) ke saath EXACT match hone chahiye, warna model galat padhega.
# 75 points = 0-32 body (MediaPipe Pose) | 33-53 hand A | 54-74 hand B
# Hand A = jo haath body wrist 15 ke paas hai, Hand B = jo wrist 16 ke paas (Colab layout check se verified).
# ------------------------------------------------------------------------------------------------
POSE_N, HAND_N = 33, 21
HAND_A0, HAND_B0 = 33, 54  # 75-array me dono haathon ka starting index
WRIST_A, WRIST_B = 15, 16  # body landmarks: 15 = left wrist, 16 = right wrist
L_SHOULDER, R_SHOULDER = 11, 12
KEEP = list(range(25)) + list(range(33, 75))  # training me pair (25-32) hataye the -> 67 points
HA, HB = slice(25, 46), slice(46, 67)  # KEEP ke andar haathon ki position
D_IN = 67 * 2 + 2  # 134 (x,y) + 2 hand-present flags = 136

# MediaPipe .task models — pehli baar internet se download, uske baad hamesha offline
TASK_URLS = {
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task",
}


def ensure_task_models(model_dir):
    """MediaPipe ke 2 .task files model/ me na hon to ek baar download karo.
    Kyun: app offline hai, par landmark models pip package me nahi aate. Ek baar download ke baad
    internet ki zaroorat nahi. Return: {naam: path}, jo sanket_live.py landmarkers banane me use karta hai."""
    paths = {}
    for name, url in TASK_URLS.items():
        path = os.path.join(model_dir, name)
        if not os.path.exists(path):
            print(f"[setup] {name} download ho raha hai (sirf pehli baar)...")
            tmp = path + ".part"  # adhoora download asli naam se save na ho
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
        paths[name] = path
    return paths


# ------------------------------------------------------------------------------------------------
# STEP 1: MediaPipe output -> ek (75, 3) array, training data ke format me
# ------------------------------------------------------------------------------------------------
def _to_px(lms, w, h):
    """Normalized (0-1) landmarks ko pixel me badlo.
    Kyun: training data pixel units me tha. x ko width se aur y ko height se guna karna zaroori hai,
    warna 16:9 frame me shape chapti ho jaati hai aur kandhe ki chaudai (scale) galat aati hai."""
    return np.array([[p.x * w, p.y * h, p.z] for p in lms], np.float32)


def assign_hands(pose_px, hands_px, handed_labels):
    """Detected haathon ko A/B slot me daalo.
    Kyun: MediaPipe ka 'Left/Right' label mirror ke hisaab se confuse karta hai. Training data me
    rule tha: 'jo haath wrist 15 ke paas = A'. Wahi rule yahan lagate hain (distance se),
    taaki live aur training ka matlab ek hi rahe.
    Return: (hand_a, hand_b) — har ek (21,3) array ya None."""
    if not hands_px:
        return None, None
    if pose_px is not None:
        wa, wb = pose_px[WRIST_A, :2], pose_px[WRIST_B, :2]

        def d(hnd, w):  # haath ki wrist (hand point 0) se body wrist ki doori
            return float(np.linalg.norm(hnd[0, :2] - w))

        if len(hands_px) == 1:
            h0 = hands_px[0]
            return (h0, None) if d(h0, wa) <= d(h0, wb) else (None, h0)
        h0, h1 = hands_px[0], hands_px[1]
        # Do haath: dono possible jodiyon me se jiski total doori kam ho, wahi sahi assignment
        if d(h0, wa) + d(h1, wb) <= d(h0, wb) + d(h1, wa):
            return h0, h1
        return h1, h0
    # Body na mile (rare): MediaPipe ka label use karo. Hand Landmarker mirrored (selfie) image maanta
    # hai, aur hum raw (non-mirrored) frame dete hain, isliye label ulta padhna padta hai:
    # 'Right' = insaan ka left = slot A.  (Inference — body milne pe ye rasta use hi nahi hota.)
    a = b = None
    for hnd, lab in zip(hands_px, handed_labels, strict=True):
        if lab == "Right" and a is None:
            a = hnd
        elif b is None:
            b = hnd
        elif a is None:
            a = hnd
    return a, b


def frame_to_75(pose_result, hand_result, w, h):
    """Ek frame ke MediaPipe results -> (75,3) float32 array. Jo nahi mila wo 0 (training jaisa).
    Aage: ye array Segmenter.push() me jaata hai."""
    out = np.zeros((75, 3), np.float32)
    pose_px = None
    if pose_result is not None and pose_result.pose_landmarks:
        pose_px = _to_px(pose_result.pose_landmarks[0], w, h)
        out[:POSE_N] = pose_px
    hands_px, labels = [], []
    if hand_result is not None and hand_result.hand_landmarks:
        for i, lms in enumerate(hand_result.hand_landmarks[:2]):
            hands_px.append(_to_px(lms, w, h))
            cat = (
                hand_result.handedness[i][0] if hand_result.handedness and hand_result.handedness[i] else None
            )
            labels.append(cat.category_name if cat else "")
    ha, hb = assign_hands(pose_px, hands_px, labels)
    if ha is not None:
        out[HAND_A0 : HAND_A0 + HAND_N] = ha
    if hb is not None:
        out[HAND_B0 : HAND_B0 + HAND_N] = hb
    return out


# ------------------------------------------------------------------------------------------------
# STEP 2: normalization + model input — Colab training code ki HUBAHU copy (ek bhi farak = galat result)
# ------------------------------------------------------------------------------------------------
def normalize(k):
    """Kandhon (11,12) ke beech center + kandhe ki chaudai se scale.
    Kyun: camera distance/resolution ka farak khatam -> laptop webcam aur INCLUDE ka camera same scale pe.
    Median per segment = ek-do kharab frames poore sign ko nahi bigaadte. Clip ±6 = outliers kaabu me."""
    k = k[:, KEEP, :2].astype(np.float32)
    pres = np.abs(k).sum(-1) > 0
    sh_c = (k[:, 11] + k[:, 12]) / 2
    sh_w = np.linalg.norm(k[:, 11] - k[:, 12], axis=-1)
    ok = sh_w > 1e-6
    c = np.median(sh_c[ok], 0) if ok.any() else np.zeros(2, np.float32)
    s = np.median(sh_w[ok]) if ok.any() else 1.0
    k = np.clip((k - c) / s, -6, 6)
    k[~pres] = 0
    return k, pres


def build_input(frames75, T):
    """List of (75,3) frames -> model input (1, T, 136).
    Training ke eval mode jaisa: poore segment se T frames barabar doori pe chuno (koi random crop nahi).
    Aage: Predictor isse ONNX ko deta hai."""
    k, pres = normalize(np.stack(frames75))
    idx = np.linspace(0, len(k) - 1, T).round().astype(int)
    k, pres = k[idx], pres[idx]
    flags = np.stack([pres[:, HA].any(1), pres[:, HB].any(1)], 1).astype(np.float32)
    x = np.concatenate([k.reshape(T, -1), flags], 1).astype(np.float32)
    return x[None]


# ------------------------------------------------------------------------------------------------
# STEP 3: Segmenter — continuous webcam stream me se ek-ek sign ka hissa kaatna
# ------------------------------------------------------------------------------------------------
def hand_active(f75, raise_k):
    """Kya koi haath 'sign karne ki position' me hai?
    Rule: haath detect hua AUR uski wrist kandhe se raise_k * (kandhe ki chaudai) se upar hai.
    Kyun: haath neeche latke hon (rest position) to bhi MediaPipe unhe pakad leta hai. Sirf 'detect hua'
    pe sign shuru karte to bina sign ke bhi predictions aate rehte."""
    hands = [f75[HAND_A0 : HAND_A0 + HAND_N], f75[HAND_B0 : HAND_B0 + HAND_N]]
    present = [np.abs(hd[:, :2]).sum() > 0 for hd in hands]
    if not any(present):
        return False
    ls, rs = f75[L_SHOULDER, :2], f75[R_SHOULDER, :2]
    sw = float(np.linalg.norm(ls - rs))
    if sw < 1e-6:  # body nahi mili -> sirf haath dikhna hi kaafi maan lo
        return True
    limit_y = (ls[1] + rs[1]) / 2 + raise_k * sw  # image me y neeche badhta hai
    return any(p and hd[0, 1] < limit_y for hd, p in zip(hands, present, strict=True))


class Segmenter:
    """Chhoti state machine:  IDLE --(haath utha, start_frames tak)--> RECORDING
                              RECORDING --(haath gira, end_frames tak / max_len)--> segment ready
    pre_roll: shuru hone se pehle ke kuch frames bhi jodte hain. Kyun: INCLUDE ke videos me sign se
    pehle ka thoda rest-position wala hissa bhi hota hai, isliye model ko wahi shape milni chahiye."""

    def __init__(self, raise_k=1.2, start_frames=3, end_frames=10, pre_roll=8, min_len=16, max_len=160):
        self.raise_k, self.start_frames, self.end_frames = raise_k, start_frames, end_frames
        self.min_len, self.max_len = min_len, max_len
        self.history = deque(maxlen=pre_roll + start_frames)  # IDLE me pichle frames (pre-roll)
        self.buf, self.recording, self.on, self.off = [], False, 0, 0

    def reset(self):
        self.history.clear()
        self.buf, self.recording, self.on, self.off = [], False, 0, 0

    def push(self, f75):
        """Ek frame do. Sign poora hote hi frames ki list lautata hai, warna None.
        Aage: lautaayi list Predictor.predict() me jaati hai."""
        active = hand_active(f75, self.raise_k)
        if not self.recording:
            self.history.append(f75)
            self.on = self.on + 1 if active else 0
            if self.on >= self.start_frames:  # pakka sign shuru (1-2 frame ka jhatka nahi)
                self.recording, self.buf, self.off = True, list(self.history), 0
            return None
        self.buf.append(f75)
        self.off = 0 if active else self.off + 1
        if self.off >= self.end_frames or len(self.buf) >= self.max_len:
            seg = self.buf[: len(self.buf) - max(0, self.off - 4)]  # aakhri khaali frames me se 4 hi rakho
            self.reset()
            return seg if len(seg) >= self.min_len else None  # bahut chhota = galti se haath hila
        return None


# ------------------------------------------------------------------------------------------------
# STEP 4: Predictor — ONNX model + config (Colab se aayi zip)
# ------------------------------------------------------------------------------------------------
class Predictor:
    """config.json + .onnx load karta hai. predict(frames) -> [(word, prob), ...] top-k.
    Config me T aur d_in check karte hain, taaki galat model file pe chupchaap galat jawab na aaye."""

    def __init__(self, model_dir):
        import onnxruntime as ort  # yahan import: tests bina onnxruntime ke bhi chal sakein

        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
        onnx = [f for f in os.listdir(model_dir) if f.startswith("sanket_") and f.endswith(".onnx")]
        if not onnx:
            raise FileNotFoundError(
                f"{model_dir} me sanket_*.onnx nahi mila — Colab wali zip yahan unzip karo"
            )
        if cfg.get("d_in", D_IN) != D_IN or list(cfg.get("keep", KEEP)) != KEEP:
            raise ValueError(
                "config.json ka layout is code se match nahi karta — model aur code alag version ke hain"
            )
        self.classes, self.T = cfg["classes"], int(cfg["T"])
        self.sess = ort.InferenceSession(
            os.path.join(model_dir, sorted(onnx)[0]), providers=["CPUExecutionProvider"]
        )
        self.inp = self.sess.get_inputs()[0].name
        self.model_name = sorted(onnx)[0]

    def predict(self, frames75, k=3):
        logits = self.sess.run(None, {self.inp: build_input(frames75, self.T)})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()  # softmax -> probability
        top = np.argsort(-p)[:k]
        return [(self.classes[i], float(p[i])) for i in top]
