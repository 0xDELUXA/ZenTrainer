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


def with_refs(conditioning: list, refs: list[torch.Tensor]) -> list:
    """A copy of a CLIP conditioning ([[cond, {...}]]) carrying these raw latents, as the
    ReferenceLatent node adds them (batch dim on, appended to any already there)."""
    out = []
    for cond, opts in conditioning:
        opts = dict(opts)
        opts["reference_latents"] = list(opts.get("reference_latents") or []) + [r if r.ndim == 4 else r[None] for r in refs]
        out.append([cond, opts])
    return out
