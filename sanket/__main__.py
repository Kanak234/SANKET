"""sanket/__main__.py — entry point:  `python -m sanket`  (or the SANKET.exe in the installer).

WHY two modes:
  • normal run → opens the PySide6 window (ui/main_window.py);
  • `--selftest` → checks every model file and both translation directions WITHOUT a camera,
    microphone or screen, then exits 0 (all good) or 1 (something broken). Used by the training
    pipeline's last step, by CI before building the installer, and by Kanak after installing.
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys

import numpy as np


def selftest() -> int:
    """Run SANKET's whole logic chain once on synthetic data. Prints a checklist."""
    from sanket.core.grammar import glosses_to_sentence, text_to_glosses
    from sanket.core.lexicon import default_lexicon
    from sanket.core.normalize import FEATURES, N_POINTS, SEQ_LEN, model_input
    from sanket.core.recognizer import IslRecognizer, load_recognizer
    from sanket.core.segmenter import Segmenter
    from sanket.core.signs import SignDictionary, SignPlayer
    from sanket.core.stt import SpeechToText
    from sanket.paths import resource

    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, note: str = "") -> None:
        results.append((name, bool(ok), note))

    # Block the network for the whole self-test: proves the logic needs no internet (PRD FR-C1).
    real_socket = socket.socket

    def no_network(*_a, **_k):
        raise OSError("network disabled during SANKET self-test")

    socket.socket = no_network  # type: ignore[assignment]
    try:
        lex = default_lexicon()
        check("lexicon: 263 signs", len(lex) == 263, f"{len(lex)} entries")

        rec = load_recognizer()
        is_isl = isinstance(rec, IslRecognizer)
        check("recognizer model", not rec.demo_mode, "DEMO mode — no model in models/isl or models/"
              if rec.demo_mode else f"{'ISL pose model' if is_isl else 'Holistic model'}, threshold {rec.threshold:.2f}")
        if is_isl and not rec.demo_mode:
            from sanket.core.pose75 import Segmenter75

            # Synthetic 75-point signer (pixel units): shoulders at y=300, right hand raised and moving.
            seg75, found = Segmenter75(), []
            for t in range(70):
                f = np.zeros((75, 3), np.float32)
                f[11, :2], f[12, :2] = (380, 300), (260, 300)
                f[15, :2], f[16, :2] = (390, 520), (250, 520)
                if 5 <= t < 45:
                    f[16, :2] = (250 + 20 * np.sin(t / 3), 330)
                    f[54:75, :2] = f[16, :2] + np.random.default_rng(t).normal(0, 3, (21, 2))
                out = seg75.push(f)
                if out is not None:
                    found.append(out)
            check("pose segmenter finds the sign", len(found) == 1, f"{len(found)} segment(s)")
            if found:
                p75 = rec.predict(found[0])
                check("ISL model runs", 0.0 <= p75.prob <= 1.0, f"top: {p75.gloss} {p75.prob:.2f}")

        # Synthetic sign: a hand rising, waving, then dropping — must produce one segment.
        seg = Segmenter()
        segments = []
        for t in range(60):
            pts = np.full((N_POINTS, 3), np.nan, np.float32)
            pts[42 + 11] = (0.6, 0.5, 0)  # shoulders
            pts[42 + 12] = (0.4, 0.5, 0)
            if 5 <= t < 40:
                pts[21:42, 0] = 0.45 + 0.05 * np.sin(t / 2)  # right hand moving at chest height
                pts[21:42, 1] = 0.55
                pts[21:42, 2] = 0
            out = seg.push(pts)
            if out is not None:
                segments.append(out)
        check("segmenter finds the sign", len(segments) == 1, f"{len(segments)} segment(s)")
        if segments:
            x = model_input(segments[0])
            check("features shape", x.shape == (SEQ_LEN, FEATURES) and np.isfinite(x).all(), str(x.shape))
            if not is_isl:
                pred = rec.predict(segments[0])
                check("recognizer runs", 0.0 <= pred.prob <= 1.0, f"top: {pred.gloss} {pred.prob:.2f}")

        s = glosses_to_sentence(["i", "happy"])
        check("signs → sentence", s.hi == "मैं ख़ुश हूँ" and s.en == "I am happy.", f"{s.hi} / {s.en}")
        g = text_to_glosses("मैं कल बाज़ार गया था")
        check("speech text → signs", g[:3] == ["i", "yesterday", "market"], " ".join(g))

        signs = SignDictionary()  # optional in v1: without it the avatar shows captions only
        print(f"  [info] avatar sign dictionary: {len(signs.motions)} signs"
              + ("" if signs.motions else " (models/signs.npz missing - captions only)"))
        frames = SignPlayer(signs).timeline(["hello", "TEXT:राम"])
        check("avatar timeline", len(frames) > 0, f"{len(frames)} frames")

        stt = SpeechToText("hi")
        check("Hindi speech model", stt.available, "" if stt.available else "models/vosk/hi missing")
        if stt.available:
            check("speech-to-text runs offline", stt.transcribe_pcm(b"\x00\x00" * 16000) == "", "1 s silence")

        if is_isl:
            for task in ("pose_landmarker_full.task", "hand_landmarker.task"):
                check(f"MediaPipe {task}", resource("models", "isl", task).exists())
        else:
            check("MediaPipe model", resource("models", "mediapipe", "holistic_landmarker.task").exists())
    finally:
        socket.socket = real_socket  # type: ignore[assignment]

    width = max(len(n) for n, _, _ in results)
    for name, ok, note in results:
        print(f"  [{'OK ' if ok else 'XX '}] {name.ljust(width)}  {note}")
    failed = [n for n, ok, _ in results if not ok]
    print(f"\nSANKET self-test: {len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sanket", description="SANKET — offline ISL ⇄ speech translator")
    ap.add_argument("--selftest", action="store_true", help="check models and logic, then exit")
    ap.add_argument("--debug", action="store_true", help="verbose logging")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.selftest:
        return selftest()
    from sanket.ui.main_window import run

    return run()


if __name__ == "__main__":
    sys.exit(main())
