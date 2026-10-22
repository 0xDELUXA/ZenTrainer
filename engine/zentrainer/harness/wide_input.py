"""Channel inputs ("concat" edits): the input pictures' latents ride beside the noisy latent as
extra channels, and the model's input layer is widened to read them. The layout of ComfyUI's
MOSS-Image control adapters (Comfy-Org/ComfyUI e3bee7f, darask0/MOSS-Image-color-depth-control):
MOSS's `video_embedder`, a Linear over the latent's 128 channels, becomes [W | C] over
128 × (1 + inputs), the noisy latent first and each input after it in order; a zero input latent
means no input.

Trained as ComfyUI weight wrappers rather than a new module, so offloading, int8 and weight
streaming see the same layer: the wrapper returns the wider weight at each forward. The new
columns start at zero (the run starts as the base model) and train at CONTROL_LR_SCALE times the
LoRA's rate, as darask0's adapters did ("at 1x they barely move"); the original columns and the
bias train as small deltas at the LoRA's rate. Saved as ComfyUI's "set" patches
(`<layer>.set_weight`, `<layer>.bias.set_weight`), which its LoRA loader applies, widening the
layer, from that commit on.
"""

from __future__ import annotations

import torch

CONTROL_LR_SCALE = 10.0


def input_layer(model) -> tuple[str, torch.nn.Module] | None:
    """(name, module) of the layer a model's latent channels go into, for models whose input
    layer is a Linear over the latent's channels (MOSS Image); None for the rest."""
    dm = getattr(model, "diffusion_model", None)
    layer = getattr(dm, "video_embedder", None)
    channels = getattr(dm, "latent_channels", None)
    if layer is None or channels is None or getattr(layer, "weight", None) is None or layer.weight.ndim != 2:
        return None
    if layer.weight.shape[1] != channels:
        return None
    return "diffusion_model.video_embedder", layer


class WideInput(torch.nn.Module):
    """The trained part of a widened input layer: the new columns, and deltas on the original
    weight and bias."""

    def __init__(self, layer: torch.nn.Module, inputs: int, dtype: torch.dtype, device):
        super().__init__()
        out_f, in_f = layer.weight.shape
        self.in_features = in_f
        self.inputs = inputs
        self.delta = torch.nn.Parameter(torch.zeros(out_f, in_f, dtype=dtype, device=device))
        self.control = torch.nn.Parameter(torch.zeros(out_f, in_f * inputs, dtype=dtype, device=device))
        has_bias = getattr(layer, "bias", None) is not None
        self.bias_delta = torch.nn.Parameter(torch.zeros(out_f, dtype=dtype, device=device)) if has_bias else None

    def weight(self, w: torch.Tensor) -> torch.Tensor:
        return torch.cat([w + self.delta.to(w.dtype), self.control.to(w.dtype)], dim=1)

    def bias(self, b: torch.Tensor) -> torch.Tensor:
        return b + self.bias_delta.to(b.dtype)

    def load(self, weight: torch.Tensor, bias: torch.Tensor | None, base_weight: torch.Tensor,
             base_bias: torch.Tensor | None) -> None:
        """Start from a saved widened layer (continuing a run): its columns minus the base's."""
        want = self.in_features * (1 + self.inputs)
        if weight.shape[1] != want:
            raise ValueError(f"the LoRA's input layer takes {weight.shape[1] // self.in_features - 1} input(s), "
                             f"this run {self.inputs}: pick the same number of input sets")
        with torch.no_grad():
            self.delta.copy_(weight[:, : self.in_features].float() - base_weight.float())
            self.control.copy_(weight[:, self.in_features :])
            if self.bias_delta is not None and bias is not None and base_bias is not None:
                self.bias_delta.copy_(bias.float() - base_bias.float())

    def full(self, base_weight: torch.Tensor, base_bias: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The whole widened weight and bias, as a LoRA file stores them."""
        dev = base_weight.device  # the base comes from the CPU; the trained part may be on the GPU
        w = torch.cat([base_weight.float() + self.delta.float().to(dev), self.control.float().to(dev)], dim=1)
        b = (base_bias.float() + self.bias_delta.float().to(base_bias.device)
             if base_bias is not None and self.bias_delta is not None else None)
        return w, b


def zero_inputs(latent: torch.Tensor, inputs: int) -> torch.Tensor:
    """The "no input" channels for a latent: zeros, inputs × its channels."""
    return latent.new_zeros((latent.shape[0], latent.shape[1] * inputs) + tuple(latent.shape[2:]))
