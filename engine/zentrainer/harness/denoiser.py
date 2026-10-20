"""One differentiable forward pass of a ComfyUI diffusion model.

ComfyUI's samplers do: conditioning → `process_conds` (the model's `extra_conds`
turn a CLIP output into what this family needs: SDXL pooled + size embeddings, Flux
guidance, ...) → `get_area_and_mult` per chunk → `cond_cat` → `apply_model`. We do
the same, minus CFG and areas, so every family ComfyUI supports works unchanged.
"""

from __future__ import annotations

from typing import Any

import torch


class Denoiser:
    def __init__(self, patcher, transformer_options: dict[str, Any] | None = None):
        self.patcher = patcher
        self.model = patcher.model
        self.transformer_options = transformer_options or {}

    @property
    def model_sampling(self):
        return self.model.model_sampling

    def prepare(self, conditioning: list, latent: torch.Tensor, device: torch.device,
                latent_shapes: list | None = None) -> dict:
        """Turn one sample's CLIP conditioning ([[cond, {pooled_output, ...}]]) into the
        processed cond dict for a latent of this shape (a packed one with its stream shapes,
        see streams.py). Cache the result per sample."""
        import comfy.sampler_helpers
        import comfy.samplers

        conds = {"positive": comfy.sampler_helpers.convert_cond(conditioning)}
        noise = torch.zeros_like(latent, device=device)
        conds = comfy.samplers.process_conds(self.model, noise, conds, device, latent_image=noise, seed=0,
                                             latent_shapes=latent_shapes)
        return conds["positive"][0]

    def __call__(self, x_t: torch.Tensor, sigma: torch.Tensor, conds: list[dict], extra: dict | None = None) -> torch.Tensor:
        """x_t (B, C, ...), sigma (B,), one prepared cond per sample. Returns x0 predictions."""
        import comfy.patcher_extension
        import comfy.samplers

        parts = []
        for i, c in enumerate(conds):
            res = comfy.samplers.get_area_and_mult(c, x_t[i : i + 1], sigma[i : i + 1])
            parts.append(res.conditioning)

        to = self.patcher.apply_hooks(hooks=None)
        to = comfy.patcher_extension.merge_nested_dicts(to, self.patcher.model_options.get("transformer_options", {}),
                                                        copy_dict1=False)
        to = comfy.patcher_extension.merge_nested_dicts(to, self.transformer_options, copy_dict1=False)
        if extra:  # this call only (e.g. TREAD's block replacements for one training step)
            to = comfy.patcher_extension.merge_nested_dicts(to, extra, copy_dict1=True)

        groups = batches(parts, comfy.samplers.cond_equal_size)
        if len(groups) == 1:
            return self._apply(x_t, sigma, parts, [c["uuid"] for c in conds], to)
        # conds that can't share a batch (MOSS's text runs at its true length, so two captions of
        # different lengths don't stack) run as separate calls, as ComfyUI's sampler does
        out = [None] * len(conds)
        for g in groups:
            pred = self._apply(x_t[g], sigma[g], [parts[i] for i in g], [conds[i]["uuid"] for i in g], to)
            for j, i in enumerate(g):
                out[i] = pred[j : j + 1]
        return torch.cat(out)

    def _apply(self, x_t, sigma, parts, uuids, to):
        import comfy.samplers

        kwargs = comfy.samplers.cond_cat(parts)
        to = dict(to)
        to["cond_or_uncond"] = [0] * len(parts)
        to["uuids"] = uuids
        to["sigmas"] = sigma
        kwargs["transformer_options"] = to
        return self.model.apply_model(x_t, sigma, **kwargs)


def batches(parts: list, fits) -> list[list[int]]:
    """Sample indices grouped so each group's conds stack into one call (fits(a, b): they do)."""
    groups: list[list[int]] = []
    for i, p in enumerate(parts):
        g = next((g for g in groups if fits(parts[g[0]], p)), None)
        if g is None:
            groups.append([i])
        else:
            g.append(i)
    return groups
