"""
SANKET core tests — webcam ke bina. Chalao:  python tests/test_core.py
Kya pakka karte hain:
  1. normalize/build_input Colab training code se BIT-for-BIT same hai (sabse zaroori — farak = galat words)
  2. haath sahi slot (A/B) me jaate hain
  3. Segmenter asli sign pakadta hai, aur jhatke / neeche latke haath ko sign nahi maanta
  4. Predictor ONNX + config sahi padhta hai, aur galat config pe saaf error deta hai
"""

import json
import os
import sys
import tempfile
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sanket_core as core  # noqa: E402

rng = np.random.default_rng(0)


# ---------- Colab training code ki exact copy (reference) ----------
KEEP_TR = list(range(25)) + list(range(33, 75))
HA_TR, HB_TR = slice(25, 46), slice(46, 67)


def normalize_training(k):
    k = k[:, KEEP_TR, :2].astype(np.float32)
    pres = np.abs(k).sum(-1) > 0
    sh_c, sh_w = (k[:, 11] + k[:, 12]) / 2, np.linalg.norm(k[:, 11] - k[:, 12], axis=-1)
    ok = sh_w > 1e-6
    c = np.median(sh_c[ok], 0) if ok.any() else np.zeros(2, np.float32)
    s = np.median(sh_w[ok]) if ok.any() else 1.0
    k = np.clip((k - c) / s, -6, 6)
    k[~pres] = 0
    return k, pres


def training_eval_input(k75, T=64):
    k, pres = normalize_training(k75)
    idx = np.linspace(0, len(k) - 1, T).round().astype(int)  # frame_idx(n, train=False)
    k, pres = k[idx].copy(), pres[idx]
    flags = np.stack([pres[:, HA_TR].any(1), pres[:, HB_TR].any(1)], 1).astype(np.float32)
    return np.concatenate([k.reshape(T, -1), flags], 1)


# ---------- helpers: nakli insaan ----------
def person_frame(hand_y_a=None, hand_y_b=None, missing_a=False, missing_b=False):
    """(75,3) frame: kandhe y=300 pe (chaudai 200px), wrists body pe; haath ki y di ho to wahan."""
    f = np.zeros((75, 3), np.float32)
    f[:33, 0] = rng.uniform(500, 700, 33)
    f[:33, 1] = rng.uniform(250, 700, 33)
    f[11, :2] = [700, 300]
    f[12, :2] = [500, 300]  # 11 = left shoulder (image me right side)
    ya = 650 if hand_y_a is None else hand_y_a
    yb = 650 if hand_y_b is None else hand_y_b
    f[15, :2] = [720, ya]
    f[16, :2] = [480, yb]
    if not missing_a:
        f[33:54, :2] = np.array([720, ya]) + rng.normal(0, 15, (21, 2))
        f[33, :2] = [720, ya]
    if not missing_b:
        f[54:75, :2] = np.array([480, yb]) + rng.normal(0, 15, (21, 2))
        f[54, :2] = [480, yb]
    return f


def test_normalize_matches_training():
    for trial in range(20):
        n = int(rng.integers(20, 140))
        k = rng.uniform(-80, 3100, (n, 75, 3)).astype(np.float64)  # INCLUDE jaisa range, float64 jaisa pkl
        k[rng.random((n, 75)) < 0.2] = 0  # kuch points missing
        k[:, 11, :2] = rng.uniform(600, 800, (n, 2))
        k[:, 12, :2] = rng.uniform(300, 500, (n, 2))
        if trial == 0:
            k[:, 33:75] = 0  # dono haath kabhi nahi dikhe
        ours = core.build_input(list(k), 64)[0]
        ref = training_eval_input(k, 64)
        assert ours.shape == (64, core.D_IN) and ours.dtype == np.float32
        assert np.array_equal(ours, ref.astype(np.float32)), f"trial {trial}: training se mismatch!"
    print("ok  normalize + build_input == Colab training code (20 random videos, exact)")


def test_assign_hands():
    pose = np.zeros((33, 3), np.float32)
    pose[15, :2] = [700, 400]
    pose[16, :2] = [300, 400]
    near15 = np.zeros((21, 3), np.float32)
    near15[:, :2] = [705, 410]
    near16 = np.zeros((21, 3), np.float32)
    near16[:, :2] = [295, 395]
    a, b = core.assign_hands(pose, [near16, near15], ["Left", "Right"])  # ulte order me diye
    assert a is near15 and b is near16
    a, b = core.assign_hands(pose, [near16], ["Left"])
    assert a is None and b is near16
    a, b = core.assign_hands(None, [near16], ["Right"])  # body nahi -> label se
    assert a is near16 and b is None
    print("ok  assign_hands: wrist-15 wala haath slot A, order/label ka farak nahi padta")


def test_frame_to_75_pixels():
    def lm(x, y):
        return NS(x=x, y=y, z=0.0)

    pose = NS(pose_landmarks=[[lm(0.5, 0.5)] * 33])
    pose.pose_landmarks[0] = list(pose.pose_landmarks[0])
    pose.pose_landmarks[0][15] = lm(0.6, 0.6)
    pose.pose_landmarks[0][16] = lm(0.4, 0.6)
    hands = NS(hand_landmarks=[[lm(0.4, 0.6)] * 21], handedness=[[NS(category_name="Left")]])
    f = core.frame_to_75(pose, hands, 1280, 720)
    assert f.shape == (75, 3)
    assert np.allclose(f[0, :2], [640, 360])  # x*W, y*H (aspect ratio sahi)
    assert np.allclose(f[54, :2], [512, 432])  # wrist 16 ke paas -> slot B
    assert np.abs(f[33:54]).sum() == 0  # slot A khaali
    empty = core.frame_to_75(NS(pose_landmarks=[]), NS(hand_landmarks=[], handedness=[]), 1280, 720)
    assert np.abs(empty).sum() == 0
    print("ok  frame_to_75: pixel conversion + slots + khaali frame")


def run(seg, frames):
    out = []
    for f in frames:
        s = seg.push(f)
        if s is not None:
            out.append(s)
    return out


def test_segmenter():
    idle = [person_frame() for _ in range(30)]  # haath neeche (y=650 > 300+1.2*200)
    sign = [person_frame(hand_y_a=330) for _ in range(45)]  # ek haath upar
    segs = run(core.Segmenter(), idle + sign + idle)
    assert len(segs) == 1, segs
    assert 45 <= len(segs[0]) <= 45 + 8 + 3 + 4, len(segs[0])  # sign + pre-roll + aakhri 4
    segs = run(core.Segmenter(), idle + sign[:2] + idle)  # 2 frame ka jhatka
    assert segs == []
    segs = run(core.Segmenter(), idle + [person_frame(missing_a=True, missing_b=True) for _ in range(40)])
    assert segs == []  # haath dikhe hi nahi
    segs = run(core.Segmenter(), idle + sign + idle + sign + idle)
    assert len(segs) == 2  # do sign lagataar
    segs = run(core.Segmenter(max_len=60), idle + [person_frame(hand_y_a=330)] * 200)
    assert len(segs) >= 2 and all(len(s) <= 60 for s in segs)  # haath upar hi rakha -> max_len pe toota
    print("ok  segmenter: 1 sign, jhatka ignore, rest-haath ignore, 2 signs, max_len")


def make_dummy_model(d, n_cls=5, T=64):
    """Asli model jaisa I/O (seq (1,64,136) -> logits (1,N)) wala chhota ONNX, Predictor test ke liye."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    W = numpy_helper.from_array(rng.normal(0, 1, (core.D_IN, n_cls)).astype(np.float32), "W")
    axes = numpy_helper.from_array(np.array([1], np.int64), "axes")
    g = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["seq", "axes"], ["m"], keepdims=0),
            helper.make_node("MatMul", ["m", "W"], ["logits"]),
        ],
        "dummy",
        [helper.make_tensor_value_info("seq", TensorProto.FLOAT, [1, T, core.D_IN])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, n_cls])],
        [W, axes],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
    m.ir_version = (
        9  # naye onnx ka default IR 14 hai jo onnxruntime nahi padhta; torch export bhi ~8-9 deta hai
    )
    onnx.save(m, os.path.join(d, "sanket_include.onnx"))
    cls = ["hello", "thank you", "good morning", "cat", "india"][:n_cls]
    json.dump(
        {"classes": cls, "T": T, "keep": core.KEEP, "d_in": core.D_IN},
        open(os.path.join(d, "config.json"), "w"),
    )
    return cls


def test_predictor():
    with tempfile.TemporaryDirectory() as d:
        cls = make_dummy_model(d)
        p = core.Predictor(d)
        top = p.predict([person_frame(hand_y_a=330) for _ in range(50)], k=3)
        assert len(top) == 3 and all(w in cls for w, _ in top)
        assert abs(sum(q for _, q in p.predict([person_frame()] * 20, k=5)) - 1) < 1e-5  # softmax
        assert top[0][1] >= top[1][1] >= top[2][1]
        json.dump(
            {"classes": cls, "T": 64, "keep": list(range(67)), "d_in": 136},
            open(os.path.join(d, "config.json"), "w"),
        )
        try:
            core.Predictor(d)
            raise AssertionError("galat keep pe error aana chahiye tha")
        except ValueError:
            pass
    print("ok  predictor: ONNX run, softmax, sorted top-k, galat config pe saaf error")


if __name__ == "__main__":
    for t in [
        test_normalize_matches_training,
        test_assign_hands,
        test_frame_to_75_pixels,
        test_segmenter,
        test_predictor,
    ]:
        t()
    print("\nALL TESTS PASSED")
