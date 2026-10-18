"""A faster backward for ComfyUI's int8 ConvRot layers while training (memory.fused_int8_backward).

ComfyUI's QuantLinearFunc keeps the int8 forward fast but, in backward, dequantizes the whole
weight to bf16 for every frozen layer on every step just to get the input gradient. For a ConvRot
weight, W = (q * s) @ blockdiag(H^T) (int8 q, a scale per output row s, a normalised Hadamard H
over groups of input columns), so

    grad_in = grad_out @ W = ((grad_out * s) @ q) @ blockdiag(H^T)

The first product is the fused Triton kernel (kernels/w8a16_backward.py: never builds the bf16
weight), the rotation a small per-group matmul. Training only; anything else (other layouts, an
fp8 backward, a weight that trains, no Triton) and any error falls back to ComfyUI's own path.
"""

from __future__ import annotations

import logging

import torch

_enabled = False
_patched = False
_warned = False


def enable(on: bool) -> None:
    """Turn the fused path on or off (patching ComfyUI's QuantLinearFunc once)."""
    global _enabled
    _enabled = on
    if on:
        _patch()


def _hadamard(size: int, device, dtype) -> torch.Tensor:
    from comfy_kitchen.tensor.int8_utils import _build_hadamard

    return _build_hadamard(size, device=device, dtype=dtype)


def input_grad(grad_2d: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, group: int) -> torch.Tensor:
    """grad_out @ dequantized ConvRot weight, without dequantizing it. grad_2d (M, out) bf16."""
    from .kernels.w8a16_backward import fused_w8a16_input_grad

    # fp32 out of the kernel and through the rotation, rounded once at the end like the eager path
    g = fused_w8a16_input_grad(grad_2d.to(torch.bfloat16), q, scale, out_dtype=torch.float32)  # (M, in), rotated
    if group:
        m, k = g.shape
        h = _hadamard(group, g.device, torch.float32)
        g = torch.matmul(g.view(m, k // group, group), h.T).view(m, k)
    return g.to(torch.bfloat16)


def _patch() -> None:
    global _patched
    if _patched:
        return
    try:
        import comfy.model_management as mm
        import comfy.ops as ops
    except Exception:
        return
    func = getattr(ops, "QuantLinearFunc", None)
    if func is None:
        return
    original = func.backward

    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        global _warned
        if _enabled and mm.in_training and not ctx.fp8_bwd and not ctx.weight_requires_grad:
            try:
                _input, weight = ctx.saved_tensors
                params = getattr(weight, "_params", None)
                if params is not None and getattr(params, "convrot", False) and getattr(weight, "_qdata", None) is not None \
                        and weight._qdata.dtype == torch.int8 and grad_output.is_cuda:
                    grad_2d = grad_output.flatten(0, -2)
                    gi = input_grad(grad_2d, weight._qdata, params.scale, int(params.convrot_groupsize))
                    gi = gi.to(ctx.compute_dtype)
                    if len(ctx.input_shape) > 2:
                        gi = gi.unflatten(0, ctx.input_shape[:-1])
                    gb = grad_2d.to(ctx.compute_dtype).sum(dim=0) if ctx.has_bias else None
                    return gi, None, gb, None, None, None
            except Exception:
                if not _warned:
                    _warned = True
                    logging.warning("[ZenTrainer] fused int8 backward failed; using ComfyUI's", exc_info=True)
        return original(ctx, grad_output)

    func.backward = staticmethod(backward)
    _patched = True
    logging.info("[ZenTrainer] fused int8 ConvRot backward available")
