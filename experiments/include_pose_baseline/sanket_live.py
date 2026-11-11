"""
SANKET Live — webcam se Indian Sign Language (ISL) -> text + aawaz. Poora offline.

Chalane ka tareeka:  python sanket_live.py            (auto mode: haath uthao, sign karo, haath girao)
                     python sanket_live.py --manual   (SPACE dabake record shuru/band)
Keys:  SPACE = manual record | M = auto/manual badlo | S = aawaz on/off | C = sentence saaf | Q/ESC = band

Flow:  camera frame -> MediaPipe (pose + 2 haath) -> sanket_core.frame_to_75 -> Segmenter -> Predictor (ONNX)
       -> screen pe top-3 words + sentence + TTS aawaz
"""

import argparse
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np
import sanket_core as core

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------------------------------------
# Aawaz (offline TTS) — alag thread me, taaki bolte waqt camera atke nahi
# ------------------------------------------------------------------------------------------------
class Speaker:
    """Queue me word daalo, background thread bolta rahega.
    Kyun naya engine har baar: Windows pe pyttsx3 ka runAndWait() thread ke andar doosri baar
    kabhi-kabhi atak jaata hai. Har word pe naya engine thoda dheema hai par bharosemand.
    pyttsx3 na mile ya fail ho -> app chalti rahegi, sirf aawaz band (text phir bhi dikhega)."""

    def __init__(self, enabled=True):
        self.enabled, self.q = enabled, queue.Queue()
        try:
            import pyttsx3  # noqa: F401

            self.ok = True
        except Exception:
            self.ok = False
            print("[tts] pyttsx3 nahi mila — aawaz band, text chalega")
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, text):
        if self.enabled and self.ok:
            self.q.put(text)

    def _run(self):
        while True:
            text = self.q.get()
            try:
                import pyttsx3

                eng = pyttsx3.init()
                eng.setProperty("rate", 160)
                eng.say(text)
                eng.runAndWait()
                eng.stop()
            except Exception as e:  # aawaz fail = crash nahi, sirf log
                print("[tts] error:", e)


# ------------------------------------------------------------------------------------------------
# MediaPipe landmarkers (VIDEO mode = frames ke beech tracking, har frame naya detection nahi -> fast)
# ------------------------------------------------------------------------------------------------
def make_landmarkers(model_dir):
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    paths = core.ensure_task_models(model_dir)
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=paths["pose_landmarker_full.task"]),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
    )
    hands = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=paths["hand_landmarker.task"]),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    return mp, pose, hands


# ------------------------------------------------------------------------------------------------
# Drawing helpers — landmarks raw frame pe, text mirror ke BAAD (warna text ulta dikhega)
# ------------------------------------------------------------------------------------------------
UPPER_BODY = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]
COL_A, COL_B, COL_BODY = (255, 170, 60), (60, 200, 255), (200, 200, 200)  # BGR


def draw_skeleton(img, f75):
    def lines(base, edges, col, n):
        pts = f75[base : base + n, :2]
        if np.abs(pts).sum() == 0:
            return
        for a, b in edges:
            pa, pb = pts[a], pts[b]
            if np.abs(pa).sum() and np.abs(pb).sum():
                cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), col, 2, cv2.LINE_AA)

    lines(0, UPPER_BODY, COL_BODY, core.POSE_N)
    lines(core.HAND_A0, HAND_EDGES, COL_A, core.HAND_N)
    lines(core.HAND_B0, HAND_EDGES, COL_B, core.HAND_N)


def load_logo():
    """KP watermark (AGENTS.md rule 8). Repo ke assets/logo-mark.svg ka PNG roop (assets/logo-mark.png,
    transparent) ho to wahi. Na ho to sirf text placeholder — apna banaya logo kabhi nahi (rule 8)."""
    p = os.path.join(HERE, "..", "..", "assets", "logo-mark.png")
    if os.path.exists(p):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[2] == 4:
            return cv2.resize(img, (56, int(56 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
    return None


def draw_watermark(img, logo):
    h, w = img.shape[:2]
    if logo is not None:  # alpha blend PNG, top-right
        lh, lw = logo.shape[:2]
        x0, y0 = w - lw - 14, 12
        roi = img[y0 : y0 + lh, x0 : x0 + lw].astype(np.float32)
        a = logo[:, :, 3:4].astype(np.float32) / 255 * 0.85
        img[y0 : y0 + lh, x0 : x0 + lw] = (roi * (1 - a) + logo[:, :, :3] * a).astype(np.uint8)
        return
    # Placeholder (TODO docs/OPEN_QUESTIONS.md): asli logo-mark.png aane tak sirf saada text
    cv2.putText(
        img, "[KP logo]", (w - 120, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA
    )


def panel(img, x, y, w, h, alpha=0.55):
    """Text ke peeche halka kaala box — kisi bhi background pe text padha ja sake (colour contrast)."""
    sub = img[y : y + h, x : x + w]
    img[y : y + h, x : x + w] = (sub * (1 - alpha)).astype(np.uint8)


def draw_ui(img, st):
    h, w = img.shape[:2]
    # --- upar: status + mode + fps ---
    panel(img, 0, 0, w, 44)
    rec = st["recording"]
    cv2.circle(img, (22, 22), 8, (0, 0, 255) if rec else (90, 90, 90), -1, cv2.LINE_AA)
    status = (
        "RECORDING"
        if rec
        else ("Haath uthao aur sign karo" if st["mode"] == "auto" else "SPACE dabao record ke liye")
    )
    cv2.putText(
        img,
        f"SANKET  |  {st['mode'].upper()}  |  {status}",
        (40, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"{st['fps']:.0f} fps",
        (w - 160, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    # --- right: top-3 prediction bars ---
    if st["top"]:
        panel(img, w - 300, 60, 290, 30 + 34 * len(st["top"]))
        cv2.putText(
            img, "Prediction", (w - 288, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )
        for i, (word, p) in enumerate(st["top"]):
            yy = 96 + 34 * i
            col = (
                (60, 140, 40) if (i == 0 and p >= st["thr"]) else (110, 110, 110)
            )  # gehra rang = white text saaf padhe
            cv2.rectangle(img, (w - 288, yy), (w - 288 + int(260 * p), yy + 22), col, -1)
            cv2.putText(
                img,
                f"{word}  {p * 100:.0f}%",
                (w - 282, yy + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    # --- neeche: sentence + keys ---
    panel(img, 0, h - 78, w, 78)
    sent = " ".join(st["sentence"][-8:]) or "..."
    cv2.putText(img, sent, (16, h - 42), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        img,
        f"SPACE record  M mode  S voice:{'on' if st['tts'] else 'off'}  C clear  Q quit",
        (16, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )


# ------------------------------------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="SANKET Live — ISL to text + speech (offline)")
    ap.add_argument("--camera", type=int, default=0, help="webcam number (0 = default)")
    ap.add_argument("--model-dir", default=os.path.join(HERE, "model"))
    ap.add_argument("--manual", action="store_true", help="SPACE se record (auto segmenting band)")
    ap.add_argument(
        "--threshold", type=float, default=0.45, help="isse kam confidence pe word bola/joda nahi jayega"
    )
    ap.add_argument("--raise-k", type=float, default=1.2, help='haath kitna upar ho to "sign shuru" maane')
    ap.add_argument("--no-tts", action="store_true")
    a = ap.parse_args()

    # 1) model + landmarkers (yahin fail ho to saaf message, aadha khula window nahi)
    try:
        pred = core.Predictor(a.model_dir)
    except Exception as e:
        sys.exit(f'[model] {e}\n-> Colab wali sanket_include_model.zip ko "{a.model_dir}" me unzip karo.')
    try:
        mp, pose_lm, hand_lm = make_landmarkers(a.model_dir)
    except Exception as e:
        sys.exit(
            f"[mediapipe] {e}\n-> Pehli baar internet chahiye (2 chhoti .task files download hoti hain)."
        )
    print(f"[ok] model {pred.model_name} | {len(pred.classes)} words | T={pred.T}")

    # 2) camera — Windows pe DirectShow jaldi khulta hai
    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        sys.exit(f"[camera] webcam {a.camera} nahi khula — dusra number try karo: --camera 1")

    seg = core.Segmenter(raise_k=a.raise_k)
    spk = Speaker(enabled=not a.no_tts)
    logo = load_logo()
    st = {
        "mode": "manual" if a.manual else "auto",
        "recording": False,
        "top": [],
        "sentence": [],
        "fps": 0.0,
        "thr": a.threshold,
        "tts": not a.no_tts,
    }
    manual_buf, last_ts, t_prev = [], -1, time.time()
    win = "SANKET - ISL to Speech"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def finish(frames):
        """Ek poora sign mila -> predict -> screen + sentence + aawaz."""
        if not frames:
            return
        st["top"] = pred.predict(frames, k=3)
        word, p = st["top"][0]
        print(f"[sign] {len(frames):3d} frames -> " + ", ".join(f"{w} {q:.2f}" for w, q in st["top"]))
        if p >= a.threshold:  # kam confidence = galat bolne se behtar chup
            st["sentence"].append(word)
            spk.say(word)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[camera] frame nahi mila — band kar rahe hain")
            break
        h, w = frame.shape[:2]

        # 3) MediaPipe: RGB image + har baar badhta timestamp (VIDEO mode ki shart)
        ts = max(int(time.monotonic() * 1000), last_ts + 1)
        last_ts = ts
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        f75 = core.frame_to_75(pose_lm.detect_for_video(img, ts), hand_lm.detect_for_video(img, ts), w, h)

        # 4) segmenting: auto = Segmenter khud; manual = SPACE ke beech ke frames
        if st["mode"] == "auto":
            done = seg.push(f75)
            st["recording"] = seg.recording
            if done:
                finish(done)
        elif st["recording"]:
            manual_buf.append(f75)
            if len(manual_buf) >= seg.max_len:  # bhool gaye to bhi limit pe khud ruk jaaye
                finish(manual_buf)
                manual_buf, st["recording"] = [], False

        # 5) display: skeleton raw frame pe -> mirror (selfie jaisa) -> text upar se
        draw_skeleton(frame, f75)
        frame = cv2.flip(frame, 1)
        now = time.time()
        st["fps"] = 0.9 * st["fps"] + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        draw_ui(frame, st)
        draw_watermark(frame, logo)
        cv2.imshow(win, frame)

        # 6) keyboard
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            st["sentence"], st["top"] = [], []
        if key == ord("s"):
            spk.enabled = st["tts"] = not st["tts"]
        if key == ord("m"):
            st["mode"] = "manual" if st["mode"] == "auto" else "auto"
            seg.reset()
            manual_buf, st["recording"] = [], False
        if key == ord(" ") and st["mode"] == "manual":
            if st["recording"]:
                finish(manual_buf if len(manual_buf) >= seg.min_len else [])
                manual_buf = []
            st["recording"] = not st["recording"]
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:  # window ka X dabaya
            break

    cap.release()
    cv2.destroyAllWindows()
    pose_lm.close()
    hand_lm.close()


if __name__ == "__main__":
    main()
