"""Slider LoRAs (spec.slider): a LoRA whose strength moves an image along one attribute, +1 towards
`positive`, -1 towards `negative`, 0 the base model.

Text (concept) sliders, as AI Toolkit's ConceptSliderTrainer: each step the base model (the LoRA at
strength 0, no grad) predicts the same noisy latents for the positive, neutral and negative prompts
(and any anchors); the LoRA then trains on the neutral prompt at +1 towards

    neutral + g * (positive - negative)

and at -1 towards neutral - g * (positive - negative), each renormalised to the neutral
prediction's mean and std (what keeps g = 3 from blowing the range up), with anchors' predictions
under the LoRA pulled to their base-model ones. The targets are made in the objective's own
training space (the model's output space for a "native" target, as AI Toolkit does it), then
handed to Objective.loss as x0 targets, so the loss is in that space too, for any model.
"""

from __future__ import annotations

import torch

from .objective import _bcast


def to_space(target: str, x_t: torch.Tensor, x0: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """An x0 prediction in the objective's training space: native = (x_t - x0) / sigma, else x0."""
    if target != "native":
        return x0
    return (x_t.float() - x0.float()) / _bcast(sigma, x0).float()


def from_space(target: str, x_t: torch.Tensor, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    if target != "native":
        return y
    return x_t.float() - _bcast(sigma, y).float() * y


def norm_like(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """t rescaled to ref's mean and std, per sample."""
    dims = tuple(range(1, t.ndim))
    tm, ts = t.mean(dims, keepdim=True), t.std(dims, keepdim=True).clamp_min(1e-8)
    return (t - tm) / ts * ref.std(dims, keepdim=True) + ref.mean(dims, keepdim=True)


def text_targets(target: str, x_t: torch.Tensor, sigma: torch.Tensor, positive: torch.Tensor, neutral: torch.Tensor,
                 negative: torch.Tensor, guidance: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The +1 and -1 targets, as x0, from the base model's x0 predictions for the three prompts."""
    pos, neu, neg = (to_space(target, x_t, p, sigma) for p in (positive, neutral, negative))
    delta = guidance * (pos - neg)
    plus, minus = norm_like(neu + delta, neu), norm_like(neu - delta, neu)
    return from_space(target, x_t, plus, sigma), from_space(target, x_t, minus, sigma)
