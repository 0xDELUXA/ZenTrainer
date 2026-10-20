"""Masks from SAM3 for the dataset mask editor: a text prompt ("person", "face:1") and/or clicked
points (keep / exclude) on one image, run through ComfyUI's queue like captioning so it waits
for the GPU instead of fighting a training run for it. The result is a greyscale PNG the editor
refines (feather, grow, brush) before saving it as the image's mask.

SAM3 is ComfyUI's own (comfy_extras/nodes_sam3.py), loaded as a checkpoint
(sam3.1_multiplex_fp16.safetensors) and kept loaded between requests so clicks stay quick.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ACTIVE = ("queued", "running")
_jobs: dict[str, "SegmentJob"] = {}
_lock = threading.Lock()
_model: dict[str, tuple] = {}  # checkpoint name -> (MODEL, CLIP)


@dataclass
class SegmentJob:
    job_id: str
    image: str  # absolute path
    text: str = ""
    positive: list[dict] = field(default_factory=list)  # [{x, y}] in image pixels
    negative: list[dict] = field(default_factory=list)
    threshold: float = 0.5
    refine: int = 2
    status: str = "queued"  # queued | running | done | error
    message: str = ""
    created: float = field(default_factory=time.time)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("image")
        return d


def models() -> list[str]:
    """SAM3 checkpoints ComfyUI can load."""
    import folder_paths

    return [n for n in folder_paths.get_filename_list("checkpoints") if "sam3" in n.lower()]


def create(image: Path, **kw) -> SegmentJob:
    job = SegmentJob("seg-" + secrets.token_hex(4), str(image), **kw)
    with _lock:
        _jobs[job.job_id] = job
        for old in sorted(_jobs.values(), key=lambda j: j.created)[:-50]:  # keep the last 50
            _jobs.pop(old.job_id, None)
            _out(old.job_id).unlink(missing_ok=True)
    return job


def get(job_id: str) -> SegmentJob | None:
    return _jobs.get(job_id)


def _out(job_id: str) -> Path:
    import folder_paths

    d = Path(folder_paths.get_user_directory()) / "zentrainer" / "segment"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{job_id}.png"


def result(job_id: str) -> Path | None:
    p = _out(job_id)
    return p if p.is_file() else None


def run(job: SegmentJob) -> None:
    """Runs inside a ComfyUI prompt (ZenSegmentSession)."""
    import numpy as np
    import nodes
    import torch
    from comfy_extras.nodes_sam3 import SAM3_Detect
    from PIL import Image

    job.status = "running"
    try:
        names = models()
        if not names:
            raise RuntimeError("no SAM3 model: put sam3.1_multiplex_fp16.safetensors in models/checkpoints (Models page)")
        name = names[0]
        if name not in _model:
            _model.clear()
            _model[name] = nodes.CheckpointLoaderSimple().load_checkpoint(name)[:2]
        model, clip = _model[name]
        with Image.open(job.image) as im:
            rgb = im.convert("RGB")
        image = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0)[None]
        cond = None
        if job.text.strip():
            cond = nodes.CLIPTextEncode().encode(clip, job.text.strip())[0]
        pos = json.dumps([{"x": int(p["x"]), "y": int(p["y"])} for p in job.positive]) if job.positive else None
        neg = json.dumps([{"x": int(p["x"]), "y": int(p["y"])} for p in job.negative]) if job.negative else None
        if cond is None and not pos:
            raise ValueError("give it something to find: a word, or click on the subject")
        out = SAM3_Detect.execute(model, image, conditioning=cond, positive_coords=pos, negative_coords=neg,
                                  threshold=job.threshold, refine_iterations=job.refine, individual_masks=False)
        mask = out.result[0] if hasattr(out, "result") else out[0]
        m = mask.reshape(-1, *mask.shape[-2:])[0].float().clamp(0, 1).cpu().numpy()
        Image.fromarray((m * 255).astype(np.uint8), mode="L").resize(rgb.size, Image.Resampling.BILINEAR).save(_out(job.job_id))
        job.status = "done"
    except Exception as e:
        job.status, job.message = "error", str(e) or type(e).__name__
