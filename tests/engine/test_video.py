"""Video datasets: which lengths a VAE takes, which frames each clip takes, and reading them."""

import math
import types

import pytest
import torch

from zentrainer.harness import data, video
from zentrainer.harness.video import Probe

# the temporal formulas of two real video VAEs (comfy/sd.py)
H3 = types.SimpleNamespace(downscale_ratio=(lambda a: max(1, (a - 5) // 17 * 5 + 2) if a > 1 else 1, 16, 16),
                           upscale_ratio=(lambda a: max(1, (a - 2) // 5 * 17 + 5), 16, 16))
WAN = types.SimpleNamespace(downscale_ratio=(lambda a: max(0, math.floor((a + 3) / 4)), 8, 8),
                            upscale_ratio=(lambda a: max(0, a * 4 - 3), 8, 8))


def test_valid_lengths_come_from_the_vae():
    assert video.valid_lengths(H3, 60) == [1, 5, 22, 39, 56]
    assert video.valid_lengths(WAN, 21) == [1, 5, 9, 13, 17, 21]
    assert video.valid_lengths(types.SimpleNamespace(downscale_ratio=8, upscale_ratio=8), 49) == [1]  # an image VAE


def test_plan_head_uniform_chunks():
    p = Probe(640, 360, 24.0, 240)  # 10 s
    lengths = video.valid_lengths(H3, 200)
    head = video.plan(p, 0, 39, "head", 3, lengths)
    assert len(head) == 1 and head[0] == list(range(39))
    uni = video.plan(p, 0, 39, "uniform", 3, lengths)
    assert [w[0] for w in uni] == [0, 100, 201] and all(len(w) == 39 for w in uni)
    assert uni[-1][-1] == 239  # the last window ends on the last frame
    chunks = video.plan(p, 0, 39, "chunks", 99, lengths)
    assert len(chunks) == 6 and chunks[1][0] == 39  # back to back, no overlap


def test_plan_resamples_and_fits_short_videos():
    p = Probe(640, 360, 30.0, 45)  # 1.5 s at 30 fps
    lengths = video.valid_lengths(WAN, 200)
    w = video.plan(p, 15, 49, "head", 1, lengths)
    assert len(w[0]) == 21  # 22 frames at 15 fps; the longest valid length that fits is 21
    assert w[0][:3] == [0, 2, 4]  # every other source frame
    assert video.plan(Probe(64, 64, 24.0, 3), 0, 49, "head", 1, video.valid_lengths(H3, 49)) == [[0]]  # 3 frames: a still


def test_repeats_follow_the_item_a_clip_came_from(tmp_path):
    items = [data.Item(tmp_path / "a.mp4", "", repeats=2), data.Item(tmp_path / "b.png", "", repeats=1)]
    lat = torch.zeros(4, 3, 8, 8)
    ds = data.Dataset([data.Sample(lat, 0), data.Sample(lat, 0), data.Sample(torch.zeros(4, 1, 8, 8), 0)],
                      [], [""], 0, owners=[0, 0, 1])
    data._expand_repeats(ds, items)
    assert sorted(ds.buckets[(4, 3, 8, 8)]) == [0, 0, 1, 1]  # both clips of the video, twice each
    assert ds.buckets[(4, 1, 8, 8)] == [2]


def test_read_and_probe_a_real_clip(tmp_path):
    av = pytest.importorskip("av")
    import numpy as np

    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=24)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        for i in range(30):
            frame = np.full((48, 64, 3), i * 8, dtype=np.uint8)
            for pkt in s.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    p = video.probe(path)
    assert (p.width, p.height, p.frames) == (64, 48, 30) and p.fps == pytest.approx(24)
    frames = video.read(path, [0, 10, 29], (32, 32))
    assert frames.shape == (3, 32, 32, 3)
    assert frames[0].mean() < frames[1].mean() < frames[2].mean()  # brightness rises with the index
    assert data.scan(tmp_path, warn=False)[0].path == path


def test_write_and_read_back_sound(tmp_path):
    pytest.importorskip("av")
    t = torch.linspace(0, 1, 32000)
    wave = torch.stack([torch.sin(2 * math.pi * 440 * t), torch.sin(2 * math.pi * 220 * t)]) * 0.5
    frames = torch.rand(24, 32, 48, 3)
    path = tmp_path / "s.mp4"
    video.write(path, frames, 24, (wave, 32000))
    p = video.probe(path)
    assert (p.width, p.height) == (48, 32) and p.frames == 24
    got = video.read_audio(path, 0.25, 0.75)
    assert got is not None
    a, sr = got
    assert sr == 32000 and a.shape == (2, 16000)
    assert a.abs().max() > 0.2  # the tone is there, not silence
    assert video.read_audio(tmp_path / "s.mp4", 0, 0.1)[0].shape[1] == 3200


def test_cuts_found_and_clips_stay_inside_shots(tmp_path):
    av = pytest.importorskip("av")
    import numpy as np

    path = tmp_path / "cut.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=24)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        for i in range(60):  # a slow fade, then a hard cut at 30 to a different picture
            v = (i % 30) * 2 + (0 if i < 30 else 160)
            frame = np.full((48, 64, 3), v, dtype=np.uint8)
            for pkt in s.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    found = video.cuts(path)
    assert found == [30]
    p = video.probe(path)
    spans = video.shots(p, found)
    assert spans == [(0, 30), (30, 60)]
    wins = video.plan_shots(p, 0, 22, "uniform", 3, video.valid_lengths(H3, 60), spans)
    assert wins and all(len(w) == 22 for w in wins)
    assert all((w[0] < 30) == (w[-1] < 30) for w in wins)  # none crosses the cut
