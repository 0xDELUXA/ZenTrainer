"""torch.compile on the model's repeated transformer block (memory.compile_blocks): the block is
compiled once and reused for every layer, so the elementwise work around the big matmuls (norms,
modulation, the SwiGLU, residual gates, the LoRA add) fuses into a few kernels instead of dozens.
OneTrainer's second-biggest speed-up. Compiles each block's own forward (dynamic shapes, so
TREAD's changing token counts and new buckets don't recompile every step); graph breaks at
ComfyUI's weight streaming just fall back to eager for that part. Undone when the run ends: the
model is cached between runs.
"""

from __future__ import annotations

import contextlib
import logging

import torch

from .memory import block_lists


@contextlib.contextmanager
def compiled_blocks(diffusion_model, enabled: bool):
    if not enabled:
        yield 0
        return
    import torch._dynamo

    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
    blocks = block_lists(diffusion_model, 1)
    done = []
    for m in blocks:
        if "forward" in m.__dict__:  # already wrapped by someone else: leave it
            continue
        m.forward = torch.compile(m.forward, dynamic=True)
        done.append(m)
    logging.info("[ZenTrainer] torch.compile on %d blocks (the first steps compile)", len(done))
    try:
        yield len(done)
    finally:
        for m in done:
            m.__dict__.pop("forward", None)
