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

FLUX.3 (comfy/ldm/flux3/model.py):
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

Krea 2 (comfy/ldm/krea2/model.py) and MOSS-Image (comfy/ldm/moss_image/model.py), with
memory.fused_qk_rope:
- Attention: the q/k RMSNorms and the rope run as one op that trains (kernels/qk_norm_rope.py)
  instead of a norm, a cast and a rope each. Falls back to ComfyUI's for attention patches, a
  norm weight that trains (or is patched), or no rope. Their sampling already fuses them.
"""

from __future__ import annotations

import contextlib
import logging

import torch

_applied: set[str] = set()
fused_qk_rope = False  # memory.fused_qk_rope, set by the trainer before each run


def apply(qk_rope: bool = False) -> None:
    global fused_qk_rope
    fused_qk_rope = qk_rope
    _minimax_h3()
    _flux3()
    _krea2()
    _moss()
    if qk_rope:
        logging.info("[ZenTrainer] fused q/k norm + rope while training, for Krea 2 and MOSS (memory.fused_qk_rope)")


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


def _flux3() -> None:
    if "flux3" in _applied:
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
    _applied.add("flux3")
    logging.info("[ZenTrainer] FLUX.3: trainable joint sequence and residual patched in")


def _krea2() -> None:
    if "krea2" in _applied:
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
        if not (fused_qk_rope and mm.in_training and freqs is not None) \
                or ("block_index" in transformer_options and "attn1_patch" in transformer_options.get("patches", {})) \
                or norms.qnorm.scale.requires_grad or norms.knorm.scale.requires_grad:
            return sampling_forward(self, x, freqs, mask, transformer_options)
        from .kernels.qk_norm_rope import qk_norm_rope

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
    _applied.add("krea2")


def _moss() -> None:
    if "moss" in _applied:
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
        if not (fused_qk_rope and mm.in_training) or not (plain(q_norm) and plain(k_norm)) \
                or any(x is not None for x in (*more, *named.values())):
            return sampling_attention(q, k, v, q_norm, k_norm, pe, heads, comfy_attention, transformer_options,
                                      *more, **named)
        from .kernels.qk_norm_rope import qk_norm_rope

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
    _applied.add("moss")


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
