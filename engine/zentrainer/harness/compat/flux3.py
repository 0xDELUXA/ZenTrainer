"""FLUX.3 (comfy/ldm/flux3/model.py):
- The joint sequence is a preallocated buffer filled by `torch.cat(..., out=x)`, which autograd
  refuses ("functions with out=... arguments don't support automatic differentiation") once the
  image tokens carry grad: every step. In training the cat is copied into the buffer instead, which
  autograd records.
- Each block adds its gated output into the residual stream in place, but the block's norm saved
  that stream for its backward ("modified by an inplace operation"), and a checkpointed block
  would recompute from its changed input. Out of place while gradients are on, as for H3's
  `_mod_gate`. Without them it's ComfyUI's own, which `preprocess_conds` needs: it ignores the
  block's return value and relies on the update in place. So that one runs without gradients in
  training, as it does from extra_conds (the text and references are prepared once and cached).
"""

from __future__ import annotations

import contextlib
import logging

import torch

from .. import compat


def apply() -> None:
    if "flux3" in compat._applied:
        return
    try:
        import comfy.ldm.flux3.model as flux3
    except Exception:  # an older ComfyUI without FLUX.3
        return
    import types

    import comfy.model_management as mm

    block = getattr(flux3, "Block", None)
    if block is None:
        return
    sampling_forward = block.forward

    def forward(self, x, spans, pe, transformer_options={}):
        if not (mm.in_training and torch.is_grad_enabled()):
            return sampling_forward(self, x, spans, pe, transformer_options)
        B, N, _ = x.shape
        mod = spans[0][2]
        x_mod = flux3.modulated_norm(x, self.pre_norm, mod.scale, mod.shift)
        for start, end, mod in spans[1:]:
            x_mod[:, start:end] = flux3.modulated_norm(x[:, start:end], self.pre_norm, mod.scale, mod.shift)
        if self.fused_qkv:
            q, k, v = self.qkv(x_mod).chunk(3, dim=-1)
        else:
            q, k, v = self.q_proj(x_mod), self.k_proj(x_mod), self.v_proj(x_mod)
        mlp = self.mlp_in(x_mod)
        q, k = self.norm(q.view(B, N, self.num_heads, -1), k.view(B, N, self.num_heads, -1), pe)
        q, k, v = (flux3.AttentionTensorContainer(q.flatten(2)), flux3.AttentionTensorContainer(k.flatten(2)),
                   flux3.AttentionTensorContainer(v))
        out = self.attn_out(flux3.optimized_attention(q, k, v, self.num_heads, transformer_options=transformer_options,
                                                      preferred_attention=self.comfy_attention))
        out = flux3.comfy.ops.linear_input_act(self.mlp_out, mlp, "swiglu", residual=out,
                                               residual_scale=out.new_ones(out.shape[-1]))
        parts, end = [], 0  # the spans are consecutive row ranges
        for a, b, mod in spans:
            if a > end:
                parts.append(x[:, end:a])
            parts.append(torch.addcmul(x[:, a:b], out[:, a:b], mod.gate))
            end = b
        if end < N:
            parts.append(x[:, end:])
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    preprocess = flux3.Flux3.preprocess_conds

    def preprocess_conds(self, *args, **kwargs):
        with torch.no_grad() if mm.in_training else contextlib.nullcontext():
            return preprocess(self, *args, **kwargs)

    def cat(tensors, dim=0, *, out=None):
        if out is None or not mm.in_training:
            return torch.cat(tensors, dim=dim, out=out)
        return out.copy_(torch.cat(tensors, dim=dim))

    # the model module's `torch`, with that one cat: the rest is torch's own
    class Torch(types.ModuleType):
        def __getattr__(self, name):
            return getattr(torch, name)

    shim = Torch("torch")
    shim.cat = cat
    flux3.torch = shim
    block.forward = forward
    flux3.Flux3.preprocess_conds = preprocess_conds
    compat._applied.add("flux3")
    logging.info("[ZenTrainer] FLUX.3: trainable joint sequence and residual patched in")
