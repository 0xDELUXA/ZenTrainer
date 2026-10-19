"""Gradient checkpointing and offload-aware recompute.

UPSTREAM SEAM: `patch`/`unpatch` (and the OffloadCheckpointFunction they use) are
private functions in comfy_extras/nodes_train.py. This is
the only module that reaches into them. The proposal is to move them to a public
`comfy/train/` module that the native node and ZenTrainer both import.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager


def block_lists(model, depth: int = 1) -> list:
    """The repeated blocks to checkpoint: children of the outermost nn.ModuleLists
    (blocks, double_blocks, single_blocks, input_blocks, ...). depth=2 goes one list
    deeper (e.g. the transformer_blocks inside SD's input_blocks) for finer segments."""
    import torch.nn as nn

    found: list = []

    def walk(m, level: int) -> None:
        for child in m.children():
            if isinstance(child, nn.ModuleList) and len(child) > 1:
                if level == depth:
                    found.extend(child)
                else:
                    for block in child:
                        walk(block, level + 1)
            elif not isinstance(child, nn.ModuleList):
                walk(child, level)

    walk(model, 1)
    return found


@contextmanager
def gradient_checkpointing(diffusion_model, enabled: bool, depth: int, offloading: bool, skip: int = 0) -> Iterator[int]:
    """Recompute each block's activations in backward instead of keeping them. Yields
    how many blocks are checkpointed.

    ComfyUI's native helper (find_modules_at_depth) counts the model itself as depth 1,
    so "depth 1" there wraps the whole network as one segment: backward then rebuilds
    every activation at once and saves nothing. Checkpointing per block is what bounds
    memory to roughly one block's activations. `skip` leaves the last blocks as they are
    (keeping their activations, no recompute)."""
    import types

    import torch.nn as nn

    if not enabled:
        yield types.SimpleNamespace(blocks=0, input_grad=False)
        return
    from comfy_extras.nodes_train import patch, unpatch

    blocks = block_lists(diffusion_model, max(1, depth))
    if not blocks:  # no repeated blocks (unusual): fall back to the top-level children
        blocks = [c for c in diffusion_model.children()]
    if skip > 0:
        blocks = blocks[: max(0, len(blocks) - skip)]
    for m in blocks:
        patch(m, offloading=offloading)
    logging.info("[ZenTrainer] gradient checkpointing on %d blocks (depth %d%s)", len(blocks), depth,
                 f", last {skip} kept" if skip > 0 else "")
    if offloading:
        # An offloaded layer's weight lives in a streaming buffer that the next layers reuse, so
        # autograd can't keep it for backward ("modified by an inplace operation" on step one).
        # Layers outside the checkpointed blocks (Krea 2's txtmlp, the final layer, the kept
        # last blocks) recompute too, then: torch's checkpoint, which unlike the offload one
        # also works when the input needs no grad. Any leaf with a weight: ComfyUI's quantized
        # Linear (MixedPrecisionOps) is a plain nn.Module, not an nn.Linear.
        import torch

        inside = {id(sub) for b in blocks for sub in b.modules()}
        loose = [m for m in diffusion_model.modules()
                 if id(m) not in inside and next(m.children(), None) is None
                 and isinstance(getattr(m, "weight", None), torch.Tensor)]
        for m in loose:
            patch(m, offloading=False)
        if loose:
            logging.info("[ZenTrainer] %d layers outside the checkpointed blocks recompute too (offloaded weights)", len(loose))
    # ComfyUI's per-layer offload checkpoint (a custom autograd Function, used when a patched
    # module is itself a Linear/Conv) only records when its input requires grad; block-level
    # checkpointing is torch's non-reentrant one, which doesn't. Without that need, the input
    # stays grad-free and the backward stops at the first trained layer.
    input_grad = offloading and any(isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)) for m in blocks)
    try:
        yield types.SimpleNamespace(blocks=len(blocks), input_grad=input_grad)
    finally:
        for m in diffusion_model.modules():
            unpatch(m)


def attention_override(backend: str):
    """transformer_options entry routing every attention call to `backend`.

    Flash (a custom op with no registered autograd) and sage are forward-only, so a
    server launched with --use-sage-attention would fail on the first backward.
    """
    if not backend or backend == "default":
        return {}
    from comfy.ldm.modules.attention import get_attention_function

    fn = get_attention_function(backend)

    def override(_func, *args, **kwargs):
        return fn(*args, **kwargs)

    return {"optimized_attention_override": override}
