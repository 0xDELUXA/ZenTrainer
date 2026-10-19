"""What the model is trained to predict, at which noise levels, under which loss.

Family-agnostic by construction: everything goes through ComfyUI's `model_sampling`,
which already knows each model's noise schedule (eps / v / flow, shift, sigma range).

ComfyUI models always return a denoised estimate x0. For a forward-noised sample

    x_t = noise_scaling(sigma, noise, x0)        # x0 + σ·ε  or  (1-σ)·x0 + σ·ε

the residual r = (x_t - x0) / σ is ε for SD-style models and the velocity (ε - x0) for
flow models — each family's own training target. "native" compares residuals,
"x0" compares denoised images (ComfyUI's native trainer). They differ by a 1/σ²
per-sample weight, which matters most at low noise.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from ..spec import ObjectiveSpec

LOSSES: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "mse": lambda a, b: F.mse_loss(a, b, reduction="none"),
    "l1": lambda a, b: F.l1_loss(a, b, reduction="none"),
    "huber": lambda a, b: F.huber_loss(a, b, reduction="none"),
    "smooth_l1": lambda a, b: F.smooth_l1_loss(a, b, reduction="none"),
}


def sample_percents(spec: ObjectiveSpec, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Noise levels as "percent of the schedule" in (0, 1): 0 = full noise, 1 = clean."""
    if spec.timesteps == "logit_normal":
        z = torch.randn(n, generator=generator) * spec.logit_std + spec.logit_mean
        t = torch.sigmoid(z)  # t = noise fraction
        if spec.uniform_mix > 0:
            pick = torch.rand(n, generator=generator) < spec.uniform_mix
            t = torch.where(pick, torch.rand(n, generator=generator), t)
    else:
        t = torch.rand(n, generator=generator)
    # keep away from exact endpoints: σ=0 has no residual, σ=max can be inf for eps models
    return (1.0 - t).clamp(1e-3, 1.0 - 1e-3)


def sigmas_for(model_sampling, percents: torch.Tensor, shift: float = 0.0) -> torch.Tensor:
    """σ for each percent: the model's own schedule, or for flow models with `shift` set, the
    flow schedule re-skewed by that shift instead of the sampling one."""
    if shift > 0:
        t = (1.0 - percents).clamp(0.0, 1.0)  # noise fraction
        return shift * t / (1 + (shift - 1) * t)  # comfy.model_sampling.time_snr_shift
    return torch.tensor([float(model_sampling.percent_to_sigma(float(p))) for p in percents])


def is_flow(model_sampling) -> bool:
    """σ in [0, 1] with x_t = (1-σ)x0 + σ·noise: where a training shift means something."""
    import comfy.model_sampling as cms

    # ModelSamplingFlux (Flux, Krea 2, Chroma) is flow matching too, just not a ModelSamplingDiscreteFlow
    flow = (cms.ModelSamplingDiscreteFlow, cms.ModelSamplingFlux)
    return isinstance(model_sampling, flow) and isinstance(model_sampling, cms.CONST)


class Objective:
    def __init__(self, spec: ObjectiveSpec):
        self.spec = spec
        self.loss_fn = LOSSES[spec.loss]

    def noisy(self, model_sampling, sigma: torch.Tensor, noise: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        # noise_scaling reshapes a (B,) sigma itself, as in ComfyUI's samplers
        return model_sampling.noise_scaling(sigma.to(x0.device), noise, x0, False)

    def loss(self, x_t: torch.Tensor, x0: torch.Tensor, x0_pred: torch.Tensor, sigma: torch.Tensor,
             weight: torch.Tensor | None = None) -> torch.Tensor:
        """Per-sample loss, shape (B,). `weight` (B, h, w) weights latent positions (masked
        training); each sample's loss is normalised by its total weight so masked and
        unmasked samples stay on the same scale."""
        x0, x0_pred = x0.float(), x0_pred.float()
        if self.spec.target == "native":
            s = _bcast(sigma, x0).float()
            pred, target = (x_t.float() - x0_pred) / s, (x_t.float() - x0) / s
        else:
            pred, target = x0_pred, x0
        per_elem = self.loss_fn(pred, target)
        if weight is None:
            return per_elem.reshape(per_elem.shape[0], -1).mean(dim=1)
        w = weight.to(per_elem)
        w = w.reshape(w.shape[0], *([1] * (per_elem.ndim - w.ndim)), *w.shape[1:])  # (B,1,[1,]h,w)
        w = w.expand_as(per_elem)
        b = per_elem.shape[0]
        return (per_elem * w).reshape(b, -1).sum(1) / w.reshape(b, -1).sum(1).clamp_min(1e-8)


def _bcast(sigma: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return sigma.to(like.device).reshape(sigma.shape[0], *([1] * (like.ndim - 1)))
