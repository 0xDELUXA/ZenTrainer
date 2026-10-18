"""Auto-captioning with ComfyUI's native text generation (the same path as its
TextGenerate node): a vision-language text encoder (Qwen3-VL, Qwen2.5-VL, Gemma, ...)
sees each image and writes its caption into a caption set.

A job is queued as a one-node ComfyUI prompt (ZenCaptionSession) so it takes its turn
on the GPU like a training run. Jobs are kept on disk (user/zentrainer/caption_jobs.json):
what they wrote, and what any caption they replaced said before, so a job can be reviewed
and undone after a restart. A preview job writes nothing; it only returns its captions.
"""

from __future__ import annotations

import json
import re
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from . import events

STYLES = {
    "nl": ("Natural language",
           "Describe this image for training an image model: the subject, their appearance, clothing and pose, "
           "the setting, lighting and photographic or artistic style. One detailed paragraph. "
           "Start directly with the description; no preamble."),
    "short": ("Short caption", "Write a single-sentence caption for this image. Start directly; no preamble."),
    "tags": ("Tags",
             "List comma-separated tags for this image: subject, features, clothing, pose, setting, lighting, style, "
             "most important first. Output only the tags."),
}

CAPTION = "zentrainer.caption"  # {job_id, dataset, set, status, done, total, file, text, message, ...}
MODES = ("missing", "all", "selected")  # which images: no caption in the set yet / every one / `paths`
ACTIVE = ("queued", "running")
KEEP = 60  # jobs remembered


@dataclass
class CaptionJob:
    job_id: str
    dataset: str
    set_name: str
    clip_name: str
    clip_type: str
    prompt: str
    mode: str = "missing"
    paths: list[str] = field(default_factory=list)  # dataset-relative; mode "selected"
    subject: str = ""  # what to call the main subject (a trigger word, a name)
    preview: int = 0  # > 0: caption this many images, write nothing
    max_length: int = 300
    status: str = "queued"  # queued | running | done | error | stopped | undone
    done: int = 0
    total: int = 0
    file: str = ""
    text: str = ""
    message: str = ""
    prompt_id: str = ""
    written: dict[str, str] = field(default_factory=dict)  # path -> the caption this job wrote
    previous: dict[str, str] = field(default_factory=dict)  # path -> the caption it replaced
    results: list[dict] = field(default_factory=list)  # preview: [{path, text}]
    created: float = field(default_factory=time.time)
    finished: float | None = None

    def public(self) -> dict:
        d = asdict(self)
        d.pop("previous")
        d["written"] = list(self.written)
        d["replaced"] = len(self.previous)
        return d


_jobs: dict[str, CaptionJob] = {}
_lock = threading.Lock()
_loaded = False


def _file() -> Path:
    import folder_paths

    return Path(folder_paths.get_user_directory()) / "zentrainer" / "caption_jobs.json"


def _load() -> None:
    """Jobs from disk, once. One that was queued or running when ComfyUI went down didn't
    finish: what it wrote stays (and can be undone)."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        rows = json.loads(_file().read_text())
    except (OSError, ValueError):
        return
    names = {f.name for f in fields(CaptionJob)}
    for r in rows:
        job = CaptionJob(**{k: v for k, v in r.items() if k in names})
        if job.status in ACTIVE:
            job.status, job.message = "stopped", "ComfyUI restarted before it finished"
        _jobs[job.job_id] = job


def _save() -> None:
    keep = sorted(_jobs.values(), key=lambda j: j.created, reverse=True)[:KEEP]
    for j in list(_jobs.values()):
        if j not in keep:
            del _jobs[j.job_id]
    path = _file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([asdict(j) for j in keep]))
    tmp.replace(path)


def create(**kw) -> CaptionJob:
    with _lock:
        _load()
        job = CaptionJob(job_id="cap-" + secrets.token_hex(4), **kw)
        _jobs[job.job_id] = job
        if not job.preview:
            _save()
    _emit(job)
    return job


def get(job_id: str) -> CaptionJob | None:
    with _lock:
        _load()
        return _jobs.get(job_id)


def update(job: CaptionJob, **kw) -> None:
    with _lock:
        for k, v in kw.items():
            setattr(job, k, v)
        if not job.preview:
            _save()
    _emit(job)


def models() -> list[str]:
    """Text encoders that actually resolve to a file (dangling symlinks are listed by
    ComfyUI but can't load), vision-language ones first."""
    import os

    import folder_paths

    names = [n for n in folder_paths.get_filename_list("text_encoders")
             if (p := folder_paths.get_full_path("text_encoders", n)) and os.path.exists(p)]
    vl = re.compile(r"vl|gemma", re.I)
    return [n for n in names if vl.search(n)] + [n for n in names if not vl.search(n)]


def recent(dataset: str | None = None) -> list[dict]:
    with _lock:
        _load()
        jobs = [j for j in _jobs.values() if not j.preview and (dataset is None or j.dataset == dataset)]
    return [j.public() for j in sorted(jobs, key=lambda j: j.created, reverse=True)[:20]]


def _emit(job: CaptionJob) -> None:
    d = job.public()
    for k in ("prompt", "written", "paths"):
        d.pop(k)
    events.emit(CAPTION, d)


def clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^\s*(caption|description|tags)\s*:\s*", "", text.strip(), flags=re.I)
    return text.strip().strip('"').strip()


def instruction(job: CaptionJob) -> str:
    if not job.subject:
        return job.prompt
    return (f"{job.prompt}\nRefer to the main subject as \"{job.subject}\" (exactly that, not a description "
            f"of who they are).")


def todo(job: CaptionJob, items: list) -> list:
    """The images this job captions, from the dataset's scan."""
    from .harness.data import caption_path

    root = None
    if job.mode == "selected":
        want = set(job.paths)
        out = []
        for it in items:
            root = root or _dataset_root(job)
            if str(it.path.relative_to(root)) in want:
                out.append(it)
    elif job.mode == "all":
        out = list(items)
    else:
        out = [it for it in items if not caption_path(it.path, job.set_name).is_file()]
    return out[: job.preview] if job.preview else out


def _dataset_root(job: CaptionJob) -> Path:
    from . import datasets

    return datasets.folder(job.dataset)


def run(job: CaptionJob) -> None:
    """Caption the job's images. Runs inside a ComfyUI prompt."""
    import comfy.model_management as mm
    import comfy.utils
    import nodes
    import torch

    from .harness.data import caption_path, scan
    from .harness.trainer import _comfy_compiler

    if job.status != "queued":  # cancelled while it waited
        return
    update(job, status="running")
    # ComfyUI's model compiler (CUDA graphs) breaks token-by-token generation: a device-side
    # assert on the second token that takes the whole server down. Off while captioning, as
    # it is while training.
    _comfy_compiler(False)
    try:
        with torch.inference_mode():
            root = _dataset_root(job)
            items = todo(job, scan(root, warn=False))
            update(job, total=len(items))
            if not items:
                update(job, status="done", message="nothing to caption: every image already has one in this set")
                return
            clip = nodes.CLIPLoader().load_clip(job.clip_name, job.clip_type)[0]
            if not hasattr(clip, "generate"):
                raise RuntimeError(f"{job.clip_name} can't generate text; pick a vision-language model (Qwen-VL, Gemma)")
            prompt = instruction(job)
            pbar = comfy.utils.ProgressBar(len(items))
            for it in items:
                if job.status != "running":  # cancelled
                    break
                image = _picture(it.path)  # (1, H, W, 3) in [0, 1]
                tokens = clip.tokenize(prompt, image=image, min_length=1)
                ids = clip.generate(tokens, do_sample=False, max_length=job.max_length)
                text = clean(clip.decode(ids))
                rel = str(it.path.relative_to(root))
                if job.preview:
                    job.results.append({"path": rel, "text": text})
                else:
                    target = caption_path(it.path, job.set_name)
                    if target.is_file() and rel not in job.previous:
                        job.previous[rel] = target.read_text(encoding="utf-8")
                    target.write_text(text + "\n", encoding="utf-8")
                    job.written[rel] = text
                update(job, done=job.done + 1, file=it.path.name, text=text)
                pbar.update_absolute(job.done)  # also raises on ComfyUI interrupt
                mm.throw_exception_if_processing_interrupted()
        if job.status == "running":
            update(job, status="done", finished=time.time())
    except BaseException as e:
        stopped = type(e).__name__ == "InterruptProcessingException"
        update(job, status="stopped" if stopped else "error", message=str(e) or type(e).__name__, finished=time.time())
        if not stopped:
            raise
    finally:
        _comfy_compiler(True)


def _picture(path: Path):
    """What the model looks at: the image, or a video's middle frame; at most 1024 px a side,
    in multiples of 28 (the vision encoder's patch grid)."""
    import numpy as np
    import torch
    from PIL import Image

    from .harness import video

    im = video.poster(path, 0.5) if video.is_video(path) else Image.open(path)
    with im:
        w, h = im.size
        scale = min(1.0, 1024 / max(w, h))
        size = (max(28, round(w * scale / 28) * 28), max(28, round(h * scale / 28) * 28))
        rgb = im.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0)[None]


def cancel(job: CaptionJob) -> None:
    """Stop at the next image (what's written stays, and can be undone)."""
    if job.status in ACTIVE:
        update(job, status="stopped", message="cancelled", finished=time.time())


def undo(job: CaptionJob) -> int:
    """Put every caption the job wrote back as it was: replaced ones get their old text,
    new ones are removed. Only while nothing else has changed them since."""
    from .harness.data import caption_path

    if job.status in ACTIVE:
        raise ValueError("stop the job before undoing it")
    if job.status == "undone":
        return 0
    root = _dataset_root(job)
    restored = kept = 0
    for rel, wrote in job.written.items():
        target = caption_path(root / rel, job.set_name)
        current = target.read_text(encoding="utf-8").strip() if target.is_file() else None
        if current != wrote.strip():  # edited since: that edit wins
            kept += 1
            continue
        if rel in job.previous:
            target.write_text(job.previous[rel], encoding="utf-8")
        else:
            target.unlink(missing_ok=True)
        restored += 1
    msg = f"undid {restored} caption{'s' if restored != 1 else ''}"
    if kept:
        msg += f"; kept {kept} you edited since"
    update(job, status="undone", message=msg)
    return restored
