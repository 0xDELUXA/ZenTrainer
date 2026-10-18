"""Video in datasets: probing clips, choosing which frames train, and decoding them.

A video trains as one or more clips. Each clip is a window of frames resampled to the run's
frame rate and cut to a length the model's VAE can encode exactly: video VAEs compress time in
fixed steps (Wan 4k+1 frames, MiniMax H3 17k+5), so a clip of 50 frames becomes 49 or 39. The
valid lengths come from the VAE itself (its temporal down/up-scale formulas), so a new video
model needs nothing here. Which windows train is the clip mode:

- head: the first window (good when the opening matters, e.g. image-to-video).
- uniform: `clips` windows spread evenly over the video (the whole video, a few samples).
- chunks: consecutive windows from the start, up to `clips` of them (dense coverage).

Every window caches as its own latent, so a long video gives several training samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
MODES = ("uniform", "head", "chunks")


@dataclass
class Probe:
    width: int
    height: int
    fps: float
    frames: int

    @property
    def duration(self) -> float:
        return self.frames / self.fps if self.fps else 0.0


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def probe(path: Path) -> Probe | None:
    import av

    try:
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            fps = float(s.average_rate or s.guessed_rate or 0) or 24.0
            frames = s.frames or 0
            if not frames and s.duration and s.time_base:
                frames = int(round(float(s.duration * s.time_base) * fps))
            if not frames and c.duration:
                frames = int(round(c.duration / 1_000_000 * fps))
            return Probe(s.codec_context.width, s.codec_context.height, fps, frames)
    except Exception:
        return None


def valid_lengths(vae, longest: int) -> list[int]:
    """Frame counts up to `longest` this VAE encodes and decodes back to the same count."""
    down = getattr(vae, "downscale_ratio", None)
    up = getattr(vae, "upscale_ratio", None)
    if not (isinstance(down, tuple) and callable(down[0]) and isinstance(up, tuple) and callable(up[0])):
        return [1]  # an image VAE: one frame
    return [f for f in range(1, longest + 1) if up[0](down[0](f)) == f]


def plan(p: Probe, fps: float, frames: int, mode: str, clips: int, lengths: list[int]) -> list[list[int]]:
    """Which source frames each clip takes: a list of windows, each a list of source frame
    indices (resampled to `fps`; 0 = the video's own). Clip length is the longest valid length
    up to `frames` that the video can fill."""
    rate = fps or p.fps
    step = p.fps / rate  # source frames per output frame
    available = int(p.frames / step) if step > 0 else p.frames
    fit = [n for n in lengths if n <= min(frames, available)]
    if not fit:
        return []
    n = max(fit)
    if mode == "head":
        starts = [0]
    elif mode == "chunks":
        starts = list(range(0, available - n + 1, n))[: max(1, clips)]
    else:  # uniform
        k = max(1, clips)
        span = available - n
        starts = [0] if k == 1 or span <= 0 else sorted({round(span * i / (k - 1)) for i in range(k)})
    return [[min(int(round((s + i) * step)), p.frames - 1) for i in range(n)] for s in starts]


def read(path: Path, indices: list[int], size: tuple[int, int]) -> torch.Tensor:
    """These source frames, resized to cover `size` (w, h) and center-cropped: (T, H, W, 3) in [0, 1]."""
    import av
    import numpy as np
    from PIL import Image

    want = sorted(set(indices))
    got: dict[int, torch.Tensor] = {}
    w, h = size
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(c.decode(stream)):
            if i > want[-1]:
                break
            if i in want:
                im = frame.to_image()
                scale = max(w / im.width, h / im.height)
                im = im.resize((max(w, round(im.width * scale)), max(h, round(im.height * scale))), Image.Resampling.LANCZOS)
                left, top = (im.width - w) // 2, (im.height - h) // 2
                im = im.crop((left, top, left + w, top + h)).convert("RGB")
                got[i] = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
    last = got[max(got)] if got else torch.zeros((h, w, 3))
    return torch.stack([got.get(i, last) for i in indices])


def poster(path: Path, at: float = 0.0):
    """A PIL frame of the video (`at` 0..1 through it), for thumbnails and captioning."""
    import av

    p = probe(path)
    target = int((p.frames - 1) * at) if p and p.frames else 0
    with av.open(str(path)) as c:
        for i, frame in enumerate(c.decode(c.streams.video[0])):
            if i >= target:
                return frame.to_image()
    return None
