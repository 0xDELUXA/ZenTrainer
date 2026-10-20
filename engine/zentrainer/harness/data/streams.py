"""Models whose latent is several streams at once: MiniMax H3 denoises video and stereo audio
together. ComfyUI's sampler hands such a model one flat latent with every stream packed end to
end plus their shapes (`latent_shapes`); the model unpacks it, and its sampling (noise, sigma,
x0) runs on the pack as if it were one tensor. Training does the same, so the objective, the
noise and the loss stay shape-agnostic.

The dataset caches the picture's (video) latent and, when it trains on sound, the clip's audio
latent. Streams it has nothing for are the model's own empty ones (its latent format's
`fix_empty_latent`: silence for H3) and the loss leaves them out (zero weight), so a LoRA
learns the picture without unlearning the sound; a real soundtrack takes the audio stream's
place and counts toward the loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class Packed:
    x0: torch.Tensor  # (B, 1, N): every stream, flattened end to end
    shapes: list  # each stream's shape, for the model's conds
    weight: torch.Tensor  # (B, 1, N): 1 over the trained stream (scaled by any mask), 0 elsewhere


class Streams:
    def __init__(self, model):
        self.model = model  # comfy BaseModel
        fmt = model.latent_format
        probe = torch.zeros((1, fmt.latent_channels) + (1,) * (fmt.latent_dimensions + 1))
        self.multi = bool(getattr(fmt.fix_empty_latent(probe), "is_nested", False))

    def pack(self, latent: torch.Tensor, mask_weight: torch.Tensor | None = None,
             audio: list[torch.Tensor | None] | None = None) -> Packed | None:
        """`latent` (B, C, [T,] h, w), the dataset's own; None when the model takes it as is.
        `mask_weight` (B, h, w) weights the picture's positions (masked training). `audio`, one
        per sample: its soundtrack latent (C, 2, Ta), fitted to the audio stream's length and
        trained on; None keeps the silence, left out of the loss."""
        if not self.multi:
            return None
        import comfy.utils

        streams = list(self.model.latent_format.fix_empty_latent(latent).unbind())
        heard = torch.zeros(latent.shape[0], dtype=torch.bool)
        if audio is not None and len(streams) > 1 and any(a is not None for a in audio):
            sound = streams[1].clone()
            for b, a in enumerate(audio):
                if a is not None:
                    a = a.to(sound)
                    ta = min(a.shape[-1], sound.shape[-1])
                    sound[b] = 0
                    sound[b, ..., :ta] = a[..., :ta]
                    heard[b] = True
            streams[1] = sound
        x0, shapes = comfy.utils.pack_latents(streams)
        first = streams[0]
        w = torch.ones_like(first) if mask_weight is None else _spread(mask_weight.to(first), first)
        n = math.prod(first.shape[1:])
        weight = torch.zeros_like(x0)
        weight[..., :n] = w.reshape(w.shape[0], 1, -1)
        if heard.any():
            m = math.prod(streams[1].shape[1:])
            weight[heard, :, n:n + m] = 1.0
        return Packed(x0, shapes, weight)

    def packed_shape(self, latent_shape: tuple) -> tuple:
        """The packed shape of one dataset latent (no batch dim given), with batch 1."""
        if not self.multi:
            return (1, *latent_shape)
        return tuple(self.pack(torch.zeros((1, *latent_shape))).x0.shape)


def _spread(w: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """(B, h, w) -> like's shape (B, C, [T,] h, w)."""
    w = w.reshape(w.shape[0], *([1] * (like.ndim - 3)), *w.shape[1:])
    return w.expand_as(like)


def first_stream(latent):
    """The picture out of a sampler's result (a NestedTensor for multi-stream models)."""
    return latent.unbind()[0] if getattr(latent, "is_nested", False) else latent
