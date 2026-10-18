"""A faster backward for ComfyUI's int8 ConvRot layers while training (memory.fused_int8_backward).

ComfyUI's QuantLinearFunc keeps the int8 forward fast but, in backward, dequantizes the whole
weight to bf16 for every frozen layer on every step just to get the input gradient. For a ConvRot
weight, W = (q * s) @ blockdiag(H^T) (int8 q, a scale per output row s, a normalised Hadamard H
over groups of input columns), so

    grad_in = grad_out @ W = ((grad_out * s) @ q) @ blockdiag(H^T)

Two ways to take the first product:

- "fused" (memory.fused_int8_backward): a Triton kernel (kernels/w8a16_backward.py) that never
  builds the bf16 weight. Still 16-bit math: saves memory, not time.
- "w8a8" (memory.w8a8_backward): the gradient quantized to int8 per row too, and the product on
  int8 tensor cores (torch._int_mm), as OneTrainer does. Measured on H3's shapes at clip size
  (RTX 5090): 6.0 ms for the four GEMMs against 16.8 in bf16. cuBLAS wants the weight
  column-major for that, so each call takes a transposed copy (int8: a quarter of a bf16 one).
  Costs some gradient precision (an int8 step of each row's largest value), rounded
  stochastically so it stays unbiased.

The rotation after either is a small per-group matmul. Training only; anything else (other
layouts, an fp8 backward, a weight that trains, no Triton) and any error falls back to
ComfyUI's own path.
"""

from __future__ import annotations

import logging

import torch

_mode = ""  # "", "fused" or "w8a8"
_patched = False
_warned = False


def enable(mode: str | bool) -> None:
    """Pick the backward ("" = ComfyUI's), patching ComfyUI's QuantLinearFunc once."""
    global _mode
    _mode = "fused" if mode is True else (mode or "")
    if _mode:
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


def w8a8_input_grad(grad_2d: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, group: int) -> torch.Tensor | None:
    """grad_out @ dequantized ConvRot weight on int8 tensor cores: (grad_out * s) quantized per
    row, times q, rescaled, then rotated. None when the shapes don't suit _int_mm."""
    m, n = grad_2d.shape
    k = q.shape[1]
    if m <= 16 or n % 8 or k % 8:
        return None
    gs = grad_2d.float() * scale.reshape(1, -1).float()
    step = gs.abs().amax(dim=1, keepdim=True).clamp_min(1e-30) / 127.0
    # stochastic rounding: each value lands on a neighbouring int8 step with the odds that keep
    # its expectation exact, so the rounding error averages out over steps instead of biasing
    gq = gs.div_(step).add_(torch.rand_like(gs)).floor_().clamp_(-127, 127).to(torch.int8)
    del gs
    acc = torch._int_mm(gq, q.t().contiguous().t())  # (m, k) int32; the weight column-major
    g = acc.float().mul_(step)
    del acc
    if group:
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
        if _mode and mm.in_training and not ctx.fp8_bwd and not ctx.weight_requires_grad:
            try:
                _input, weight = ctx.saved_tensors
                params = getattr(weight, "_params", None)
                if params is not None and getattr(params, "convrot", False) and getattr(weight, "_qdata", None) is not None \
                        and weight._qdata.dtype == torch.int8 and grad_output.is_cuda:
                    grad_2d = grad_output.flatten(0, -2)
                    group = int(params.convrot_groupsize)
                    gi = w8a8_input_grad(grad_2d, weight._qdata, params.scale, group) if _mode == "w8a8" else None
                    if gi is None:
                        gi = input_grad(grad_2d, weight._qdata, params.scale, group)
                    gi = gi.to(ctx.compute_dtype)
                    if len(ctx.input_shape) > 2:
                        gi = gi.unflatten(0, ctx.input_shape[:-1])
                    gb = grad_2d.to(ctx.compute_dtype).sum(dim=0) if ctx.has_bias else None
                    return gi, None, gb, None, None, None
            except Exception:
                if not _warned:
                    _warned = True
                    logging.warning("[ZenTrainer] %s int8 backward failed; using ComfyUI's", _mode, exc_info=True)
        return original(ctx, grad_output)

    func.backward = staticmethod(backward)
    _patched = True
    logging.info("[ZenTrainer] int8 ConvRot backward: %s", _mode)
