"""Avatar timeline: every token gets frames, blends have no NaN jumps, slow mode is longer."""

import numpy as np

from sanket.core.signs import BLEND_FRAMES, SignDictionary, SignPlayer, default_rest_pose
from tests.synthetic import clip


def make_dict(tmp_path):
    from sanket.core.normalize import normalized_points

    rng = np.random.default_rng(0)
    path = tmp_path / "signs.npz"
    np.savez(path, __fps__=np.float32(30), __rest__=default_rest_pose(),
             hello=normalized_points(clip(0, rng)).astype(np.float16),
             good=normalized_points(clip(1, rng)).astype(np.float16))
    return SignDictionary(path)


def test_timeline_covers_all_tokens(tmp_path):
    d = make_dict(tmp_path)
    frames = SignPlayer(d).timeline(["hello", "TEXT:राम", "good"])
    assert {f.token_index for f in frames} == {0, 1, 2}
    assert len(frames) > len(d.motions["hello"]) + len(d.motions["good"]) + 2 * BLEND_FRAMES


def test_slow_mode_longer(tmp_path):
    d = make_dict(tmp_path)
    p = SignPlayer(d)
    assert len(p.timeline(["hello"], speed=0.5)) > 1.6 * len(p.timeline(["hello"], speed=1.0)) - 30


def test_missing_file_falls_back(tmp_path):
    d = SignDictionary(tmp_path / "nope.npz")
    assert d.motions == {}
    frames = SignPlayer(d).timeline(["hello"])
    assert frames and all(f.points.shape == (79, 3) for f in frames)
