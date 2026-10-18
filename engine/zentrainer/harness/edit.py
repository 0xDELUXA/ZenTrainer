"""Edit training: the model sees an input picture (the pair's _control image) as reference
latents beside the noisy target, and learns to produce the target from it and the caption (the
instruction). The same path ComfyUI's ReferenceLatent node takes, so any family whose model reads
`reference_latents` (Flux 3 Klein, Flux 2, Kontext, Qwen Image Edit, ...) trains and samples
this way unchanged."""

from __future__ import annotations

import inspect

import torch


def takes_references(model) -> bool:
    """Whether this ComfyUI model reads reference_latents from its conditioning."""
    for cls in type(model).__mro__:
        fn = cls.__dict__.get("extra_conds")
        if fn is None:
            continue
        try:
            if "reference_latents" in inspect.getsource(fn):
                return True
        except (OSError, TypeError):
            pass
    return False


def method(model, chosen: str = "") -> str | None:
    """How the references join the sequence ("index", "offset", "index_timestep_zero"...): the
    spec's choice, else the model's own default; a model without one (Krea 2) would ignore them,
    so it gets "index" (each reference tagged on the position axis the image doesn't use, the
    NK2E / Kontext-style scheme)."""
    if chosen:
        return chosen
    dm = getattr(model, "diffusion_model", None)
    own = getattr(dm, "default_ref_method", None) or getattr(getattr(dm, "params", None), "default_ref_method", None)
    return None if own else "index"


def with_refs(conditioning: list, refs: list[torch.Tensor], ref_method: str | None = None) -> list:
    """A copy of a CLIP conditioning ([[cond, {...}]]) carrying these raw latents, as the
    ReferenceLatent node adds them (batch dim on, appended to any already there)."""
    out = []
    for cond, opts in conditioning:
        opts = dict(opts)
        opts["reference_latents"] = list(opts.get("reference_latents") or []) + [r if r.ndim == 4 else r[None] for r in refs]
        if ref_method:
            opts["reference_latents_method"] = ref_method
        out.append([cond, opts])
    return out


VISION = "<|vision_start|><|image_pad|><|vision_end|>"
GROUNDING_PX = 768  # longest side of the picture the text encoder sees (krea2edit's grounding_px)


def instruction(text: str, template: str) -> str:
    return template.replace("{}", text) if template else text


def grounded_template(clip, n: int) -> str | None:
    """The text encoder's own chat template with n pictures before the instruction, or None
    when it isn't a vision-language chat encoder. Its own template matters: Krea 2's carries a
    system prompt that qwen3vl's default picture template doesn't."""
    tok = getattr(clip, "tokenizer", None)
    base = getattr(tok, "llama_template", None)
    if not isinstance(base, str) or "user\n" not in base:
        return None
    return base.replace("user\n", "user\n" + VISION * n, 1)


def grounding_pixels(path) -> torch.Tensor:
    """An input picture as the text encoder sees it: whole (never cropped to the target),
    longest side at most GROUNDING_PX. (1, H, W, 3) in [0, 1]."""
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > GROUNDING_PX:
            s = GROUNDING_PX / max(im.size)
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.Resampling.LANCZOS)
        return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None]


def encode_grounded(clip, text: str, pictures: list[torch.Tensor]):
    """A caption's conditioning with the encoder looking at these pictures."""
    template = grounded_template(clip, len(pictures))
    if template is None:
        raise ValueError("grounded edits need a vision-language text encoder (Qwen3-VL: Krea 2, Qwen Image)")
    tokens = clip.tokenize(text, images=pictures, llama_template=template)
    return clip.encode_from_tokens_scheduled(tokens)
