"""Overlapped weight streaming while an offloaded model trains.

ComfyUI casts an offloaded layer's weights on one of two offload streams, and picking a stream
(comfy.model_management.get_offload_stream) makes it wait for everything already queued on the
compute stream. So layer N+1's copy can't start until layer N has finished: every copy sits
between two layers instead of under one. Profiled on Krea 2 int8, 16 GB: the step's extra time
over a fully loaded one was almost exactly its host-to-device copies (410 ms).

That wait guards the stream's cast buffer, and uncast_bias_weight already guards it precisely:
after a layer's last use of its weights the stream waits for that layer. With two streams in
turn, layer N+1's copy waits only for layer N-1 (its buffer's last user) and runs while layer N
computes. Every training-time caller releases its stream that way (ComfyUI's layers through
CastBiasWeightContext, as does int8_backward). The dynamic-VRAM path (ops.cast_modules_with_vbar)
serves sampling models, so this is installed for training steps only and taken off for samples.
Checked: identical first-step loss with and without, steps 1.74 -> 1.57 s.
"""

from __future__ import annotations

import logging

_original = None


def _stream(device):
    import torch

    import comfy.model_management as mm

    if mm.NUM_STREAMS == 0 or torch.compiler.is_compiling() or device not in mm.STREAMS:
        return _original(device)  # its own setup (first use) and its compile-time None
    ss = mm.STREAMS[device]
    c = (mm.stream_counters.get(device, 0) + 1) % len(ss)
    mm.stream_counters[device] = c
    return ss[c]


def overlap(on: bool) -> None:
    """Weight copies overlap the previous layer's compute (on), or ComfyUI's own (off)."""
    global _original
    import comfy.model_management as mm

    if on and _original is None:
        if mm.NUM_STREAMS == 0:
            return
        _original, mm.get_offload_stream = mm.get_offload_stream, _stream
        logging.info("[ZenTrainer] offloaded weights stream under the previous layer's compute")
    elif not on and _original is not None:
        mm.get_offload_stream, _original = _original, None
