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


def cuts(path: Path, threshold: float = 4.0) -> list[int]:
    """Frames where a new shot starts (hard cuts), from how much each frame differs from the
    last on a tiny greyscale copy: a cut is a jump well above the video's usual motion."""
    import av
    import numpy as np

    diffs, prev = [], None
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in c.decode(stream):
            g = frame.reformat(width=64, height=36, format="gray").to_ndarray().astype(np.float32)
            diffs.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
            prev = g
    if len(diffs) < 3:
        return []
    d = np.array(diffs)
    floor = max(12.0, float(np.median(d[1:])) * threshold)
    return [i for i in range(1, len(d)) if d[i] > floor and d[i] >= d[max(0, i - 1)] and d[i] >= d[min(len(d) - 1, i + 1)]]


def shots(p: Probe, cut_frames: list[int]) -> list[tuple[int, int]]:
    """[start, end) source-frame ranges between cuts."""
    bounds = [0, *[c for c in cut_frames if 0 < c < p.frames], p.frames]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


def plan_shots(p: Probe, fps: float, frames: int, mode: str, clips: int, lengths: list[int],
               spans: list[tuple[int, int]]) -> list[list[int]]:
    """As plan(), but no clip crosses a cut: windows only inside one shot. The clip length is
    the longest valid one the longest shot can fill."""
    rate = fps or p.fps
    step = p.fps / rate
    out_spans = [(int(a / step), int(b / step)) for a, b in spans]  # in output frames
    longest = max((b - a for a, b in out_spans), default=0)
    fit = [n for n in lengths if n <= min(frames, longest)]
    if not fit:
        return []
    n = max(fit)
    cand = [s for a, b in out_spans for s in range(a, b - n + 1)]  # every start that fits in a shot
    if not cand:
        return []
    if mode == "head":
        starts = [cand[0]]
    elif mode == "chunks":
        starts = [s for a, b in out_spans for s in range(a, b - n + 1, n)][: max(1, clips)]
    else:  # uniform over the positions that fit
        k = max(1, min(clips, len(cand)))
        starts = sorted({cand[round((len(cand) - 1) * i / (k - 1))] for i in range(k)}) if k > 1 else [cand[len(cand) // 2]]
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


def split(src: Path, out_dir: Path, seconds: float, count: int, fps: float = 24.0, short_side: int = 768,
          stem: str | None = None) -> list[Path]:
    """Cut a long video into `count` clips of `seconds` each, spread evenly over it (or every
    whole clip when `count` is 0), re-encoded small: `fps`, the short side at most
    `short_side`, the soundtrack kept as AAC (joint audio-video models can train on it). Clips land in `out_dir` as <stem>_<n>.mp4; existing ones are kept.
    Uses ffmpeg (seeks, so a clip costs a few seconds whatever the source length)."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("splitting videos needs ffmpeg on the PATH")
    p = probe(src)
    if p is None or not p.duration:
        raise ValueError(f"can't read {src.name}")
    whole = int(p.duration // seconds)
    if whole < 1:
        return []
    n = whole if count <= 0 else min(count, whole)
    span = p.duration - seconds
    starts = [span * i / max(n - 1, 1) for i in range(n)] if n > 1 else [0.0]
    stem = stem or src.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = f"scale='if(lt(iw,ih),min({short_side},iw),-2)':'if(lt(iw,ih),-2,min({short_side},ih))'"
    made = []
    for i, t in enumerate(starts):
        dst = out_dir / f"{stem}_{i:03d}.mp4"
        if not dst.exists():
            tmp = dst.with_suffix(".part.mp4")
            subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(src), "-t", f"{seconds:.3f}",
                            "-vf", f"{scale},fps={fps}", "-c:a", "aac", "-b:a", "192k", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                            "-pix_fmt", "yuv420p", str(tmp)], check=True)
            tmp.replace(dst)
        made.append(dst)
    return made


def write(path: Path, frames: torch.Tensor, fps: float, sound: tuple[torch.Tensor, int] | None = None) -> None:
    """(T, H, W, 3) in [0, 1] to an H.264 .mp4 (sides rounded down to even, as yuv420 needs),
    with `sound` ((channels, samples) in [-1, 1], rate) as AAC when given."""
    import av
    import numpy as np

    arr = (frames.clamp(0, 1).cpu().float().numpy() * 255).astype(np.uint8)
    h, w = arr.shape[1] // 2 * 2, arr.shape[2] // 2 * 2
    tmp = path.with_suffix(".part.mp4")
    with av.open(str(tmp), "w") as c:
        s = c.add_stream("libx264", rate=round(fps))
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.options = {"crf": "18", "preset": "veryfast"}
        a = None
        if sound is not None:
            wave, sr = sound
            wave = wave.clamp(-1, 1).numpy().astype(np.float32)
            a = c.add_stream("aac", rate=sr)
            a.layout = "stereo" if wave.shape[0] >= 2 else "mono"
        for f in arr:
            for pkt in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f[:h, :w]), format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
        if a is not None:
            wave = wave[:2]
            for i in range(0, wave.shape[1], 1024):
                frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(wave[:, i:i + 1024]), format="fltp", layout=a.layout.name)
                frame.sample_rate = sr
                for pkt in a.encode(frame):
                    c.mux(pkt)
            for pkt in a.encode():
                c.mux(pkt)
    tmp.replace(path)


def read_audio(path: Path, t0: float, t1: float) -> tuple[torch.Tensor, int] | None:
    """The audio from t0 to t1 s as ((channels, samples) float32 in [-1, 1], sample rate); None
    when the video has no audio track."""
    import av
    import numpy as np

    with av.open(str(path)) as c:
        if not c.streams.audio:
            return None
        s = c.streams.audio[0]
        sr = s.codec_context.sample_rate
        layout = "stereo" if (s.codec_context.channels or 1) >= 2 else "mono"
        rs = av.AudioResampler(format="fltp", layout=layout, rate=sr)
        c.seek(max(0, int((t0 - 0.5) / s.time_base)), stream=s, any_frame=False, backward=True)
        chunks, start = [], None
        for frame in c.decode(s):
            if frame.pts is not None and frame.pts * s.time_base > t1:
                break
            for f in rs.resample(frame):
                if start is None:
                    start = float(f.pts * f.time_base) if f.pts is not None else t0
                chunks.append(f.to_ndarray())
        if not chunks:
            return None
    wave = np.concatenate(chunks, axis=1)
    a = max(0, int(round((t0 - start) * sr)))
    n = int(round((t1 - t0) * sr))
    wave = wave[:, a:a + n]
    if wave.shape[1] < n:
        wave = np.pad(wave, ((0, 0), (0, n - wave.shape[1])))
    return torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32)), sr
