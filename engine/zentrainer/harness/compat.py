"""Fixes for ComfyUI model code that runs fine for sampling but can't train, applied at runtime
from here so the pinned ComfyUI checkout stays untouched. Each patch only changes what runs
while `comfy.model_management.in_training` is set: sampling keeps ComfyUI's fused kernels.
Each is the change worth sending upstream; drop it here once ComfyUI has it.

MiniMax H3 (comfy/ldm/minimax/model.py):
- Attention: the training branch calls comfy_kitchen's fused RMSNorm + rope, which has no
  backward. Plain torch here: the q/k RMSNorm modules, then the partial split-half rope (the
  first `rot` of each head's dims rotate, the rest pass through). Matches the kitchen op to ~1e-7.
- `_mod_gate` adds the gated branch into the residual stream in place, but norm1/norm2 saved
  that stream for their backward ("modified by an inplace operation"). Out of place in training.
"""

from __future__ import annotations

import logging

import torch

_applied: set[str] = set()


def apply() -> None:
    _minimax_h3()


def _minimax_h3() -> None:
    if "minimax_h3" in _applied:
        return
    try:
        import comfy.ldm.minimax.model as h3
    except Exception:  # an older ComfyUI without H3
        return
    import comfy.model_management as mm

    attention = getattr(h3, "Attention", None)
    gate = getattr(h3, "_mod_gate", None)
    if attention is None or gate is None:
        return
    sampling_forward = attention.forward

    def forward(self, x, rope_freqs=None, transformer_options={}):
        if not mm.in_training or rope_freqs is None:
            return sampling_forward(self, x, rope_freqs, transformer_options)
        s = x.shape[0]
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        v = v.view(s, self.heads, self.head_dim)
        rot = rope_freqs.shape[-3] * 2
        q = _rope_split_half(self.q_norm(q.view(1, s, self.heads, self.head_dim)), rope_freqs, rot)[0]
        k = _rope_split_half(self.k_norm(k.view(1, s, self.heads, self.head_dim)), rope_freqs, rot)[0]
        q = h3.AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
        k = h3.AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
        v = h3.AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
        out = h3.optimized_attention(q, k, v, self.heads, preferred_attention=self.comfy_attention, mask=None,
                                     skip_reshape=True, transformer_options=transformer_options)
        return self.out_proj(out.squeeze(0))

    def mod_gate(x, g, other, segments):
        if not mm.in_training:
            return gate(x, g, other, segments)
        parts, end = [], 0
        for a, b, row in segments:
            if a > end:
                parts.append(x[end:a])
            parts.append(torch.addcmul(x[a:b], other[a:b], h3._mod_row(g, row, x.dtype)))
            end = b
        if end < x.shape[0]:
            parts.append(x[end:])
        return torch.cat(parts) if len(parts) > 1 else parts[0]

    attention.forward = forward
    h3._mod_gate = mod_gate
    _applied.add("minimax_h3")
    logging.info("[ZenTrainer] MiniMax H3: trainable attention and residual gate patched in")


def prune_unused(patcher) -> int:
    """Drop weights the training forward never touches, so they don't take VRAM a run could
    keep the model in. MiniMax H3 int8 builds carry the sparse-attention (VSA) gate in every
    block, ~1.9 GB the dense forward never calls; None is ComfyUI's own "no gate" state.
    Returns the bytes freed."""
    import comfy.model_management as mm

    freed = 0
    for m in patcher.model.modules():
        gate = getattr(m, "to_gate_compress", None)
        if isinstance(gate, torch.nn.Module):
            freed += mm.module_size(gate)
            m.to_gate_compress = None
    if freed:
        patcher.size = 0  # recount
        logging.info("[ZenTrainer] dropped %.1f GB of weights training never uses (VSA gates)", freed / 1024 ** 3)
    return freed


def _rope_split_half(x: torch.Tensor, pe: torch.Tensor, rot: int) -> torch.Tensor:
    """Rotate the first `rot` dims of each head (split-half pairs); pe is (1, S, 1, rot/2, 2, 2)."""
    xr, xp = x[..., :rot], x[..., rot:]
    x_ = xr.to(pe.dtype).unflatten(-1, (2, -1)).movedim(-2, -1).unsqueeze(-2)
    out = (pe[..., 0] * x_[..., 0] + pe[..., 1] * x_[..., 1]).movedim(-1, -2).flatten(-2).type_as(x)
    return torch.cat([out, xp], -1)
