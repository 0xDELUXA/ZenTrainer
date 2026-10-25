"""Krea 2 (comfy/ldm/krea2/model.py), with memory.fused_qk_rope:
- Attention: the q/k RMSNorms and the rope run as one op that trains (kernels/qk_norm_rope.py)
  instead of a norm, a cast and a rope each. Falls back to ComfyUI's for attention patches, a
  norm weight that trains (or is patched), or no rope. Its sampling already fuses them.
"""

from __future__ import annotations

import torch

from .. import compat


def apply() -> None:
    if "krea2" in compat._applied:
        return
    try:
        import comfy.ldm.krea2.model as k2
    except Exception:  # an older ComfyUI without Krea 2
        return
    import comfy.model_management as mm

    attention = getattr(k2, "Attention", None)
    if attention is None:
        return
    sampling_forward = attention.forward

    def forward(self, x, freqs=None, mask=None, transformer_options={}):
        norms = self.qknorm
        if not (compat.fused_qk_rope and mm.in_training and freqs is not None) \
                or ("block_index" in transformer_options and "attn1_patch" in transformer_options.get("patches", {})) \
                or norms.qnorm.scale.requires_grad or norms.knorm.scale.requires_grad:
            return sampling_forward(self, x, freqs, mask, transformer_options)
        from ..kernels.qk_norm_rope import qk_norm_rope

        q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
        q = k2.rearrange(q, "B L (H D) -> B H L D", H=self.heads)
        k = k2.rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
        v = k2.rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)
        # the norms' (1 + scale) weights in fp32, as ComfyUI's RMSNorm builds them
        wq = mm.cast_to(norms.qnorm.scale, dtype=torch.float32, device=x.device) + 1.0
        wk = mm.cast_to(norms.knorm.scale, dtype=torch.float32, device=x.device) + 1.0
        q, k = qk_norm_rope(q, k, wq, wk, freqs, norms.qnorm.eps)
        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        q, k, v = k2.AttentionTensorContainer(q), k2.AttentionTensorContainer(k), k2.AttentionTensorContainer(v)
        out = k2.optimized_attention_masked(q, k, v, self.heads, mask=mask, skip_reshape=True,
                                            preferred_attention=self.comfy_attention, transformer_options=transformer_options)
        patches = transformer_options.get("patches", {})
        if "block_index" in transformer_options and "attn1_output_patch" in patches:
            extra_options = transformer_options.copy()
            for p in patches["attn1_output_patch"]:
                out = p(out, extra_options)
        return self.wo(out * k2.F.sigmoid(gate))

    attention.forward = forward
    compat._applied.add("krea2")
