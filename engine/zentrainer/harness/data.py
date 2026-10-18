"""Dataset: scan → bucket → encode once → cache → batches.

Latents (VAE) and conditioning (text encoder) are computed once and cached on disk,
keyed by the files, their mtimes and every setting that changes the encoding, so a
re-run on the same data skips straight to training.

Masks (optional, white = learn, black = ignore) are found per image as
`_masks/<stem>.<ext>`, `<stem>-masklabel.<ext>` (OneTrainer) or `<stem>.mask.<ext>`, get
the image's exact resize/crop, and are stored at latent resolution for the loss.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..spec import DatasetSpec
from . import video
from .video import VIDEO_EXTS, is_video

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS


MASK_SUFFIXES = ("-masklabel", ".mask")
# edit pairs: <stem>_control.<ext> is the input picture for <stem> (the target), the caption is
# the instruction; _control2, _control3... add more references
CONTROL = re.compile(r"_control\d*$")
SET_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}$")


@dataclass
class Item:
    path: Path
    caption: str  # the "default" set (<stem>.txt)
    repeats: int = 1
    mask: Path | None = None
    captions: dict[str, str] = field(default_factory=dict)  # set name -> text, "default" included
    controls: list[Path] = field(default_factory=list)  # edit pairs: the input picture(s)


@dataclass
class Sample:
    latent: torch.Tensor  # (C, [T,] h, w) raw VAE latent (before process_latent_in); T for video VAEs
    caption: int  # index into Dataset.conditionings
    mask: torch.Tensor | None = None  # (h, w) in [0, 1] at latent resolution
    variants: list[int] = field(default_factory=list)  # caption choices when mixing sets
    audio: torch.Tensor | None = None  # (C, 2, Ta): a clip's soundtrack latent (joint audio-video models)
    refs: list[torch.Tensor] | None = None  # edit pairs: the control pictures' raw latents, as reference_latents
    # grounded edits: this image's own conditioning (the encoder saw its inputs), and the empty
    # caption with them for caption dropout
    cond: list | None = None
    cond_empty: list | None = None


@dataclass
class Dataset:
    samples: list[Sample]
    conditionings: list  # one CLIP conditioning per unique caption
    captions: list[str]
    empty_caption: int  # index of "" (for caption dropout)
    buckets: dict[tuple, list[int]] = field(default_factory=dict)  # latent shape -> sample idx
    owners: list[int] = field(default_factory=list)  # sample -> the scanned item it came from (a video gives several)


def caption_sets(img: Path, caption_ext: str = ".txt") -> dict[str, str]:
    """All caption sets for an image: <stem>.txt is "default", <stem>.<set>.txt the rest."""
    out = {}
    default = img.with_suffix(caption_ext)
    if default.is_file():
        out["default"] = default.read_text(encoding="utf-8").strip()
    for p in img.parent.glob(f"{glob_escape(img.stem)}.*{caption_ext}"):
        name = p.name[len(img.stem) + 1 : -len(caption_ext)]
        if SET_NAME.match(name):
            out[name] = p.read_text(encoding="utf-8").strip()
    return out


def caption_path(img: Path, set_name: str, caption_ext: str = ".txt") -> Path:
    return img.with_suffix(caption_ext) if set_name == "default" else img.with_name(f"{img.stem}.{set_name}{caption_ext}")


def glob_escape(s: str) -> str:
    return re.sub(r"([*?\[\]])", r"[\1]", s)


def is_mask_file(p: Path) -> bool:
    return p.stem.endswith(MASK_SUFFIXES)


def is_control_file(p: Path) -> bool:
    return bool(CONTROL.search(p.stem))


CONTROL_EXTS = MEDIA_EXTS | {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"}


def find_controls(img: Path) -> list[Path]:
    """An edit pair's input(s): <stem>_control, then _control2, _control3... Pictures for every
    model; videos and audio too for MiniMax H3's references."""
    out = []
    for n in ["", *map(str, range(2, 10))]:
        hit = next((c for e in sorted(CONTROL_EXTS) if (c := img.with_name(f"{img.stem}_control{n}{e}")).is_file()), None)
        if hit is None:
            break
        out.append(hit)
    return out


def find_mask(img: Path) -> Path | None:
    if is_video(img):
        return None  # masks are per image
    for cand in [img.parent / "_masks" / (img.stem + e) for e in sorted(IMAGE_EXTS)] + [
        img.with_name(img.stem + suffix + e) for suffix in MASK_SUFFIXES for e in sorted(IMAGE_EXTS)
    ]:
        if cand.is_file():
            return cand
    return None


def scan(root: Path, caption_ext: str = ".txt", warn: bool = True) -> list[Item]:
    """Images and videos under root with same-stem captions (and masks, see module doc). "N_name"
    subfolders repeat N times; folders starting with "." or "_" are ignored."""
    items = []
    for dirpath, dirs, files in os.walk(root):
        # skip hidden and tool folders: other trainers leave .thumbs/, _latent_cache/, ...
        dirs[:] = sorted(x for x in dirs if not x.startswith((".", "_")))
        d = Path(dirpath)
        repeats = 1
        head = d.name.split("_", 1)[0] if d != root else ""
        if head.isdigit() and "_" in d.name:
            repeats = max(1, int(head))
        for f in sorted(files):
            p = d / f
            if p.suffix.lower() not in MEDIA_EXTS or is_mask_file(p) or is_control_file(p):
                continue
            caps = caption_sets(p, caption_ext)
            if "default" not in caps and warn:
                logging.warning("[ZenTrainer] no caption for %s; training it with an empty caption", p.name)
            items.append(Item(p, caps.get("default", ""), repeats, find_mask(p), caps, find_controls(p)))
    return items


def bucket_sizes(resolution: int, step: int, max_ratio: float = 4.0) -> list[tuple[int, int]]:
    """(w, h) pairs, multiples of `step`, with w*h <= resolution², aspect within max_ratio."""
    area = resolution * resolution
    out = set()
    w = step
    while w <= resolution * max_ratio:
        h = (area // w) // step * step
        if h >= step and max(w / h, h / w) <= max_ratio:
            out.add((w, h))
        w += step
    return sorted(out)


def pick_bucket(w: int, h: int, sizes: list[tuple[int, int]]) -> tuple[int, int]:
    ar = math.log(w / h)
    return min(sizes, key=lambda s: (abs(math.log(s[0] / s[1]) - ar), -s[0] * s[1]))


def load_image(path: Path, size: tuple[int, int], mode: str = "RGB") -> torch.Tensor:
    """Resize to cover `size`, center-crop. Returns (H, W, 3) float in [0, 1], or (H, W)
    for mode "L" (masks get exactly the same geometry as their image)."""
    from PIL import Image, ImageOps

    import numpy as np

    img = ImageOps.exif_transpose(Image.open(path)).convert(mode)
    tw, th = size
    scale = max(tw / img.width, th / img.height)
    rw, rh = max(tw, round(img.width * scale)), max(th, round(img.height * scale))
    img = img.resize((rw, rh), Image.Resampling.LANCZOS)
    left, top = (rw - tw) // 2, (rh - th) // 2
    img = img.crop((left, top, left + tw, top + th))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)


def apply_trigger(text: str, trigger: str, prepend: bool) -> str:
    """ai-toolkit semantics: "[trigger]" -> the word; if the caption still doesn't
    contain it, prepend "trigger, "."""
    if not trigger:
        return text.replace("[trigger]", "").strip(" ,")
    text = text.replace("[trigger]", trigger)
    if prepend and trigger.lower() not in text.lower():
        text = f"{trigger}, {text}" if text else trigger
    return text


def cache_key(items: list[Item], spec: DatasetSpec, encoder_id: str) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([spec.resolution, spec.buckets, spec.bucket_step, encoder_id]).encode())
    if any(is_video(it.path) for it in items):  # only then: image datasets keep their caches
        h.update(json.dumps([spec.video_frames, spec.video_fps, spec.video_mode, spec.video_clips,
                             spec.video_resolution]).encode())
        if spec.video_audio:
            h.update(b"audio")
        if spec.video_avoid_cuts:
            h.update(b"avoid_cuts")
        if spec.clip_stills:
            h.update(b"clip_stills")
    for it in items:
        st = it.path.stat()
        h.update(f"{it.path}|{st.st_mtime_ns}|{st.st_size}|{it.caption}".encode())
        if it.mask:
            ms = it.mask.stat()
            h.update(f"|mask:{it.mask}|{ms.st_mtime_ns}|{ms.st_size}".encode())
        if spec.keyframes:
            h.update(f"|keyframes:{spec.keyframes}".encode())
        if spec.edit:
            if spec.edit_grounded:
                h.update(b"grounded")
            for c in it.controls:
                cs = c.stat()
                h.update(f"|control:{c}|{cs.st_mtime_ns}|{cs.st_size}".encode())
    return h.hexdigest()[:20]


def build(spec: DatasetSpec, clip, vae, root: Path, cache_dir: Path, encoder_id: str, progress=None,
          audio_vae=None, h3: bool = False) -> Dataset:
    # The harness runs outside ComfyUI's inference_mode, where the frozen VAE/text
    # encoder parameters still require grad: encode under no_grad, or every cached
    # latent drags an autograd graph (and a VAE backward) into training.
    with torch.no_grad():
        return _build(spec, clip, vae, root, cache_dir, encoder_id, progress, audio_vae, h3)


MUTES_FILE = ".zentrainer.json"  # in the dataset folder: {"muted": {set: [relative paths]}}; travels with it


def muted(root: Path) -> dict[str, set[str]]:
    """The pictures muted in each caption set (kept, captions and all, but not trained on in that set)."""
    try:
        data = json.loads((root / MUTES_FILE).read_text())
    except (OSError, ValueError):
        return {}
    return {s: set(v) for s, v in (data.get("muted") or {}).items()}


def _build(spec, clip, vae, root, cache_dir, encoder_id, progress, audio_vae=None, h3: bool = False) -> Dataset:
    items = scan(root, spec.caption_ext)
    if not items:
        raise FileNotFoundError(f"no images found in {root}")
    mutes = muted(root)
    if mutes:  # a picture trains from the sets it isn't muted in; muted in all of them, it sits out
        used = spec.caption_mix or [spec.caption_set]
        before = len(items)
        items = [it for it in items if any(str(it.path.relative_to(root)) not in mutes.get(s, ()) for s in used)]
        if len(items) < before:
            logging.info("[ZenTrainer] %d muted picture(s) sit out (caption set %s)", before - len(items), ", ".join(used))
        if not items:
            raise FileNotFoundError(f"every picture in {root} is muted in {', '.join(used)}")
    variants: list[list[str]] = []
    for it in items:  # pick each image's caption(s), trigger applied, before the cache key
        rel = str(it.path.relative_to(root))
        if spec.caption_mix:
            texts = [it.captions[s] for s in spec.caption_mix if it.captions.get(s) and rel not in mutes.get(s, ())]
        else:
            texts = [it.captions.get(spec.caption_set, "")] if it.captions.get(spec.caption_set) else []
        texts = texts or [it.captions.get("default", "")]
        texts = [apply_trigger(t, spec.trigger, spec.trigger_prepend) for t in texts]
        if spec.edit and spec.edit_template:
            from .edit import instruction

            texts = [instruction(t, spec.edit_template) for t in texts]
        variants.append(texts)
        it.caption = " ||| ".join(texts)  # only feeds the cache key

    # H3 builds references/keyframes its own way; other runs keep their caches
    key = cache_key(items, spec, encoder_id + ("|h3" if h3 and (spec.edit or spec.keyframes) else ""))
    path = cache_dir / key
    if (path / "dataset.pt").is_file():
        logging.info("[ZenTrainer] dataset cache hit %s (%d images)", key, len(items))
        return _expand_repeats(_from_plain(torch.load(path / "dataset.pt", weights_only=False)), items)

    sizes = bucket_sizes(spec.resolution, spec.bucket_step) if spec.buckets else [(spec.resolution, spec.resolution)]
    captions = sorted({t for v in variants for t in v} | {""})
    cap_index = {c: i for i, c in enumerate(captions)}

    from PIL import Image

    video_sizes = (bucket_sizes(spec.video_resolution, spec.bucket_step) if spec.buckets
                   else [(spec.video_resolution, spec.video_resolution)])
    lengths = None  # the clip lengths this VAE encodes exactly, worked out on the first video

    samples: list[Sample] = []
    owners: list[int] = []
    grounded: list[tuple[int, list[Path], str]] = []  # (sample, inputs, caption): encoded with the captions
    # MiniMax H3: (sample, inputs, first frame, last frame, caption, (w, h), frames), each sample's
    # conditioning built by ComfyUI's own H3 nodes after the pictures
    h3_jobs: list[tuple] = []
    sounds: list[tuple[int, Path, float, float]] = []  # (sample, video, t0, t1): encoded after the pictures
    for n, it in enumerate(items):
        idx = [cap_index[t] for t in variants[n]]
        if is_video(it.path):
            p = video.probe(it.path)
            if p is None or not p.frames:
                logging.warning("[ZenTrainer] can't read %s; skipping it", it.path.name)
                continue
            if lengths is None:
                lengths = video.valid_lengths(vae, max(spec.video_frames, 1))
            if spec.video_avoid_cuts:
                spans = video.shots(p, video.cuts(it.path))
                windows = video.plan_shots(p, spec.video_fps, spec.video_frames, spec.video_mode, spec.video_clips, lengths, spans)
            else:
                windows = video.plan(p, spec.video_fps, spec.video_frames, spec.video_mode, spec.video_clips, lengths)
            if not windows:
                logging.warning("[ZenTrainer] %s is too short for a clip; skipping it", it.path.name)
            size = pick_bucket(p.width, p.height, video_sizes)
            for k, frames in enumerate(windows):
                px = video.read(it.path, frames, size)
                latent = _encode(vae, px, it.path)
                if h3 and ((spec.edit and it.controls) or spec.keyframes):
                    first = px[:1].clone() if spec.keyframes else None
                    last = px[-1:].clone() if spec.keyframes == "first_last" else None
                    h3_jobs.append((len(samples), it.controls if spec.edit else [], first, last, variants[n][0],
                                    size, len(frames)))
                if spec.video_audio and audio_vae is not None:
                    sounds.append((len(samples), it.path, frames[0] / p.fps, (frames[-1] + 1) / p.fps))
                samples.append(Sample(latent, idx[0], None, idx))
                owners.append(n)
                if progress:
                    progress(n + 1, len(items) + len(captions), f"encoding {it.path.name} ({k + 1}/{len(windows)})")
            if spec.clip_stills and windows:  # its sharpest frame, trained as a photo with the clip's caption
                f = video.sharpest_frame(it.path, p)
                still = video.read(it.path, [f], pick_bucket(p.width, p.height, sizes))
                samples.append(Sample(_encode(vae, still, it.path), idx[0], None, idx))
                owners.append(n)
            continue
        with Image.open(it.path) as im:
            w, h = im.size
        size = pick_bucket(w, h, sizes)
        latent = _encode(vae, load_image(it.path, size).unsqueeze(0), it.path)
        mask = None
        if it.mask:
            m = load_image(it.mask, size, mode="L")[None, None]  # (1, 1, H, W)
            mask = torch.nn.functional.interpolate(m, size=tuple(latent.shape[-2:]), mode="area")[0, 0]
        refs = None
        if h3 and spec.edit and it.controls:  # H3 takes them in its own conditioning, built below
            h3_jobs.append((len(samples), it.controls, None, None, variants[n][0], size, 1))
        elif spec.edit and it.controls:  # the input picture(s) at the target's size, as the model's references
            for c in it.controls:
                with Image.open(c) as cim:
                    cw, ch = cim.size
                if abs((cw / ch) / (size[0] / size[1]) - 1) > 0.08:
                    logging.warning("[ZenTrainer] %s is a different shape from %s: it's cropped to match, which "
                                    "cuts part of it off", c.name, it.path.name)
            refs = [_encode(vae, load_image(c, size).unsqueeze(0), c) for c in it.controls]
            if spec.edit_grounded:
                grounded.append((len(samples), it.controls, variants[n][0]))
        samples.append(Sample(latent, idx[0], mask, idx, refs=refs))
        owners.append(n)
        if progress:
            progress(n + 1, len(items) + len(captions), f"encoding {it.path.name}")
    if not samples:
        raise FileNotFoundError(f"nothing in {root} could be encoded")
    # soundtracks in one pass after every picture: the video and audio VAEs each load once,
    # instead of swapping in and out per clip
    for k, (i, path, t0, t1) in enumerate(sounds):
        samples[i].audio = _encode_audio(audio_vae, path, t0, t1)
        if progress and k % 10 == 0:
            progress(k + 1, len(sounds), f"encoding soundtracks ({path.name})")

    conditionings = []
    for n, c in enumerate(captions):
        tokens = clip.tokenize(c)
        conditionings.append(_to_cpu(clip.encode_from_tokens_scheduled(tokens)))
        if progress:
            progress(len(items) + n + 1, len(items) + len(captions), "encoding captions")

    if grounded:  # each with the encoder looking at its own inputs; the empty one only if dropout uses it
        from .edit import encode_grounded, grounding_pixels

        for k, (i, inputs, text) in enumerate(grounded):
            pics = [grounding_pixels(c) for c in inputs]
            samples[i].cond = _to_cpu(encode_grounded(clip, text, pics))
            if spec.caption_dropout:
                samples[i].cond_empty = _to_cpu(encode_grounded(clip, "", pics))
            if progress:
                progress(k + 1, len(grounded), "encoding instructions with their pictures")

    if h3_jobs:  # H3: references or keyframes, and the text encoder seeing them
        from .edit import h3_keyframe_cond, h3_reference_cond

        def h3_cond(text, inputs, first, last, size, nframes):
            if inputs:
                return h3_reference_cond(clip, vae, audio_vae, text, inputs, size[0], size[1], nframes)
            return h3_keyframe_cond(clip, vae, text, first, last, size[0], size[1], nframes)

        for k, (i, inputs, first, last, text, size, nframes) in enumerate(h3_jobs):
            samples[i].cond = _to_cpu(h3_cond(text, inputs, first, last, size, nframes))
            if spec.caption_dropout:
                samples[i].cond_empty = _to_cpu(h3_cond("", inputs, first, last, size, nframes))
            if progress:
                progress(k + 1, len(h3_jobs), "encoding captions with their references")

    ds = Dataset(samples, conditionings, captions, cap_index[""], owners=owners)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(_to_plain(ds), path / "dataset.pt")
    return _expand_repeats(ds, items)


def _encode(vae, pixels: torch.Tensor, path: Path) -> torch.Tensor:
    """pixels (T, H, W, 3) in [0, 1]: one image, or a clip's frames -> the latent, batch dim off."""
    latent = vae.encode(pixels)[0].detach().cpu()
    if not torch.isfinite(latent).all():
        # never cache this: every later run with the same key would train on it
        raise ValueError(f"the VAE returned NaN/inf for {path.name}. The VAE is likely broken or missing "
                         "weights (check ComfyUI's log for 'Missing VAE keys'); try a separate VAE file")
    return latent


# The cache holds only tensors, lists, dicts and strings. Pickling our own classes
# would tie the file to a module path, and ComfyUI imports custom nodes by path.
def _encode_audio(audio_vae, path: Path, t0: float, t1: float) -> torch.Tensor | None:
    """The soundtrack from t0 to t1 s as the audio VAE's latent (C, 2, Ta); None for a silent video."""
    import comfy.audio

    got = video.read_audio(path, t0, t1)
    if got is None:
        return None
    wave, sr = got  # (channels, samples)
    if wave.shape[0] == 1:
        wave = wave.repeat(2, 1)
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    wave = wave[None, :2]
    if sr != vae_sr:
        wave = comfy.audio.resample(wave, sr, vae_sr)
    return audio_vae.encode(wave.movedim(1, -1))[0].float().cpu()  # (C, 2, Ta)


def _to_plain(ds: Dataset) -> dict:
    return {
        "version": 3,
        "latents": [s.latent for s in ds.samples],
        "variants": [s.variants for s in ds.samples],
        "masks": [s.mask for s in ds.samples],
        "caption_idx": [s.caption for s in ds.samples],
        "conditionings": ds.conditionings,
        "captions": ds.captions,
        "empty_caption": ds.empty_caption,
        "owners": ds.owners,
        "audio": [s.audio for s in ds.samples],
        "refs": [s.refs for s in ds.samples],
        "conds": [s.cond for s in ds.samples],
        "conds_empty": [s.cond_empty for s in ds.samples],
    }


def _from_plain(d: dict) -> Dataset:
    masks = d.get("masks") or [None] * len(d["latents"])
    variants = d.get("variants") or [[c] for c in d["caption_idx"]]
    audio = d.get("audio") or [None] * len(d["latents"])
    n = len(d["latents"])
    refs, conds, empties = (d.get(k) or [None] * n for k in ("refs", "conds", "conds_empty"))
    samples = [Sample(lat, c, m, v, a, r, g, e)
               for lat, c, m, v, a, r, g, e in zip(d["latents"], d["caption_idx"], masks, variants, audio, refs, conds, empties)]
    owners = d.get("owners") or list(range(len(samples)))  # before videos: one sample per item
    return Dataset(samples, d["conditionings"], d["captions"], d["empty_caption"], owners=owners)


def _expand_repeats(ds: Dataset, items: list[Item]) -> Dataset:
    """Repeats are an index trick, not copies; buckets are recomputed per run."""
    order: list[int] = []
    owners = ds.owners or list(range(len(ds.samples)))
    for i, owner in enumerate(owners):
        order += [i] * items[owner].repeats
    buckets: dict[tuple, list[int]] = {}
    for i in order:
        buckets.setdefault(tuple(ds.samples[i].latent.shape), []).append(i)
    ds.buckets = buckets
    return ds


def _to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_cpu(v) for v in x)
    return x


class Batches:
    """Endless iterator of same-shape index batches, reshuffled each epoch."""

    def __init__(self, ds: Dataset, batch_size: int, seed: int):
        self.ds = ds
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.epoch = 0
        self._queue: list[list[int]] = []

    def _refill(self) -> None:
        batches = []
        for idx in self.ds.buckets.values():
            idx = idx[:]
            self.rng.shuffle(idx)
            batches += [idx[i : i + self.batch_size] for i in range(0, len(idx), self.batch_size)]
        self.rng.shuffle(batches)
        self._queue = batches
        self.epoch += 1

    def __next__(self) -> list[int]:
        if not self._queue:
            self._refill()
        return self._queue.pop()

    def __iter__(self):
        return self
