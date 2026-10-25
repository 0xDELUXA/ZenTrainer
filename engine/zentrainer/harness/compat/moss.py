"""MOSS-Image (comfy/ldm/moss_image/model.py), with memory.fused_qk_rope:
- Attention: the q/k RMSNorms and the rope run as one op that trains (kernels/qk_norm_rope.py)
  instead of a norm, a cast and a rope each. Falls back to ComfyUI's for attention patches, a
  norm weight that trains (or is patched), or no rope. Its sampling already fuses them.
"""

from __future__ import annotations

import torch

from .. import compat


def apply() -> None:
    if "moss" in compat._applied:
        return
    try:
        import comfy.ldm.moss_image.model as moss
    except Exception:  # an older ComfyUI without MOSS-Image
        return
    import comfy.model_management as mm

    sampling_attention = getattr(moss, "attention", None)
    if sampling_attention is None:
        return

    def plain(norm) -> bool:
        """A norm whose weight is a frozen tensor with nothing patched onto it."""
        w = getattr(norm, "weight", None)
        return isinstance(w, torch.Tensor) and not w.requires_grad and not getattr(norm, "weight_function", None)

    # Newer ComfyUI passes more (ref_kv, kv_out, mask: MOSS-Image edit, #17296 #17320), positionally
    # or by name; they go through untouched, and the fused path takes only calls that use none.
    def attention(q, k, v, q_norm, k_norm, pe, heads, comfy_attention, transformer_options, *more, **named):
        if not (compat.fused_qk_rope and mm.in_training) or not (plain(q_norm) and plain(k_norm)) \
                or any(x is not None for x in (*more, *named.values())):
            return sampling_attention(q, k, v, q_norm, k_norm, pe, heads, comfy_attention, transformer_options,
                                      *more, **named)
        from ..kernels.qk_norm_rope import qk_norm_rope

        B, N, _ = q.shape
        wq = mm.cast_to(q_norm.weight, dtype=torch.float32, device=q.device)
        wk = mm.cast_to(k_norm.weight, dtype=torch.float32, device=k.device)
        q, k = qk_norm_rope(q.view(B, N, heads, -1), k.view(B, N, heads, -1), wq, wk, pe, q_norm.eps,
                            split_half=True, seq_first=True)
        q, k, v = (moss.AttentionTensorContainer(q.flatten(2)), moss.AttentionTensorContainer(k.flatten(2)),
                   moss.AttentionTensorContainer(v))
        return moss.optimized_attention(q, k, v, heads, transformer_options=transformer_options,
                                        preferred_attention=comfy_attention)

    moss.attention = attention
    compat._applied.add("moss")
