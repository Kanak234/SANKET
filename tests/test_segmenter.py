"""The segmenter must cut a live stream into single signs: one sign in, one segment out;
noise and hand-less movement must not create segments."""

import numpy as np

from sanket.core.normalize import N_POINTS
from sanket.core.segmenter import Segmenter


def frame(hand_y=None, hand_x=0.45):
    p = np.full((N_POINTS, 3), np.nan, np.float32)
    p[42 + 11] = (0.6, 0.5, 0)
    p[42 + 12] = (0.4, 0.5, 0)
    if hand_y is not None:
        p[21:42] = (hand_x, hand_y, 0)
    return p


def run(frames):
    s, out = Segmenter(), []
    for f in frames:
        r = s.push(f)
        if r is not None:
            out.append(r)
    return out, s


def test_one_sign():
    stream = [frame()] * 10 + [frame(0.55, 0.45 + 0.03 * np.sin(i)) for i in range(30)] + [frame()] * 15
    segs, _ = run(stream)
    assert len(segs) == 1 and 20 <= len(segs[0]) <= 40


def test_two_signs_with_pause():
    sign = [frame(0.55, 0.45 + 0.03 * np.sin(i)) for i in range(25)]
    segs, _ = run([frame()] * 5 + sign + [frame()] * 15 + sign + [frame()] * 15)
    assert len(segs) == 2


def test_idle_stream_gives_nothing():
    segs, _ = run([frame()] * 100)
    assert segs == []


def test_hands_down_not_a_sign():
    segs, _ = run([frame(0.95, 0.45 + 0.03 * np.sin(i)) for i in range(60)])  # hand below chest
    assert segs == []


def test_max_length_cut():
    segs, _ = run([frame(0.55, 0.45 + 0.03 * np.sin(i)) for i in range(300)])
    assert len(segs) >= 2 and all(len(s) <= 120 for s in segs)


def test_flush():
    s = Segmenter()
    for i in range(20):
        s.push(frame(0.55, 0.45 + 0.03 * np.sin(i)))
    assert s.active
    seg = s.flush()
    assert seg is not None and not s.active


def test_held_hand_does_not_retrigger():
    sign = [frame(0.55, 0.45 + 0.03 * np.sin(i)) for i in range(25)]
    held = [frame(0.55, 0.45)] * 80  # hand stays up, motionless, after the sign
    segs, _ = run(sign + held)
    assert len(segs) == 1
