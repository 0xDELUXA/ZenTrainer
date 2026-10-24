"""Per-head QK RMSNorm and RoPE as one op that trains (memory.fused_qk_rope): Krea 2's and MOSS's
attention run their q/k norms and then the rope as separate ops while training (comfy_kitchen's
fused rms_rope has no backward), each a pass over q and k plus casts to fp32 and back.

    forward:   y = rope(n * w),  n = x * r,  r = rsqrt(mean(x²) + eps)
               comfy_kitchen's rms_rope / rms_rope_split_half: fp32 inside, rounded once
    backward:  g_z = Mᵀ g_y      each pair rotated back (M: the rope's 2x2 per pair, from freqs)
               g_n = g_z * w
               g_x = r * (g_n - n * mean(g_n * n))
               one Triton kernel, a program per row; plain torch without Triton or off CUDA

Tensors come in the model's own layout: (B, H, L, D) with freqs (B or 1, 1, L, D/2, 2, 2) (Krea 2),
or seq_first (B, L, H, D) with freqs (B or 1, L, 1, D/2, 2, 2) (MOSS); any strides. A pair is
elements 2i and 2i+1 (interleaved, Krea 2) or i and i + D/2 (split_half, MOSS). w is the norm's
weight (D,) in fp32 (Krea 2's 1 + scale: in bf16 it would round to steps of 1/128). The weights and
freqs get no gradient: callers use the separate ops when a norm's weight trains. Registered as
custom ops, so torch.compile keeps the fused kernels whole instead of tracing into them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRITON_AVAILABLE = False


def _pairs(t: torch.Tensor, split_half: bool) -> tuple[torch.Tensor, torch.Tensor]:
    d = t.shape[-1] // 2
    return (t[..., :d], t[..., d:]) if split_half else (t[..., 0::2], t[..., 1::2])


def _unpairs(a: torch.Tensor, b: torch.Tensor, split_half: bool) -> torch.Tensor:
    return torch.cat((a, b), -1) if split_half else torch.stack((a, b), -1).flatten(-2)


def _freqs(freqs: torch.Tensor, x: torch.Tensor, seq_first: bool, dt: torch.dtype) -> torch.Tensor:
    """freqs cut to x's length (as ComfyUI's rope does), in dt."""
    seq = 1 if seq_first else 2
    if freqs.shape[seq] != x.shape[seq]:
        freqs = freqs.narrow(seq, 0, x.shape[seq])
    return freqs.to(dt)


def reference(x: torch.Tensor, w: torch.Tensor, freqs: torch.Tensor, eps: float, split_half: bool = False,
              seq_first: bool = False) -> torch.Tensor:
    """The maths in plain torch at x's precision (at least fp32): what the op computes."""
    dt = torch.promote_types(x.dtype, torch.float32)
    n0, n1 = _pairs(F.rms_norm(x.to(dt), (x.shape[-1],), weight=w.to(dt), eps=eps), split_half)
    f = _freqs(freqs, x, seq_first, dt)  # f[..., i, j] = M[i][j]
    return _unpairs(f[..., 0, 0] * n0 + f[..., 0, 1] * n1, f[..., 1, 0] * n0 + f[..., 1, 1] * n1, split_half).to(x.dtype)


def _backward_torch(g, x, w, freqs, eps, split_half, seq_first):
    dt = torch.promote_types(x.dtype, torch.float32)
    f = _freqs(freqs, x, seq_first, dt)
    g0, g1 = _pairs(g.to(dt), split_half)
    gz = _unpairs(f[..., 0, 0] * g0 + f[..., 1, 0] * g1, f[..., 0, 1] * g0 + f[..., 1, 1] * g1, split_half)
    gn = gz * w.to(dt)
    xf = x.to(dt)
    r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    n = xf * r
    return (r * (gn - n * (gn * n).mean(-1, keepdim=True))).to(x.dtype)


if TRITON_AVAILABLE:

    @triton.jit
    def _backward_kernel(G, X, W, FR, OUT, H, L, sgb, sgh, sgl, sxb, sxh, sxl, sob, soh, sol, sfb, eps,
                         D: tl.constexpr, SPLIT: tl.constexpr):
        row = tl.program_id(0)
        b = (row // (H * L)).to(tl.int64)
        h = ((row // L) % H).to(tl.int64)
        t = (row % L).to(tl.int64)
        k = tl.arange(0, D // 2)
        if SPLIT:  # pair k: elements k and k + D/2
            e, o = k, k + D // 2
        else:  # pair k: elements 2k and 2k + 1
            e, o = 2 * k, 2 * k + 1
        g = G + b * sgb + h * sgh + t * sgl
        x = X + b * sxb + h * sxh + t * sxl
        out = OUT + b * sob + h * soh + t * sol
        f = FR + b * sfb + t * (D * 2) + k * 4
        xe = tl.load(x + e).to(tl.float32)
        xo = tl.load(x + o).to(tl.float32)
        ge = tl.load(g + e).to(tl.float32)
        go = tl.load(g + o).to(tl.float32)
        we = tl.load(W + e)
        wo = tl.load(W + o)
        f00 = tl.load(f)
        f01 = tl.load(f + 1)
        f10 = tl.load(f + 2)
        f11 = tl.load(f + 3)
        r = tl.rsqrt((tl.sum(xe * xe, 0) + tl.sum(xo * xo, 0)) / D + eps)
        ne = xe * r
        no = xo * r
        gne = (f00 * ge + f10 * go) * we
        gno = (f01 * ge + f11 * go) * wo
        dot = (tl.sum(gne * ne, 0) + tl.sum(gno * no, 0)) / D
        tl.store(out + e, (r * (gne - ne * dot)).to(OUT.dtype.element_ty))
        tl.store(out + o, (r * (gno - no * dot)).to(OUT.dtype.element_ty))


def _backward(g, x, w, freqs, eps, split_half, seq_first):
    D = x.shape[-1]
    if not (TRITON_AVAILABLE and x.is_cuda and D in (64, 128, 256) and x.dtype in (torch.bfloat16, torch.float16, torch.float32)):
        return _backward_torch(g, x, w, freqs, eps, split_half, seq_first)
    hd, sd = (2, 1) if seq_first else (1, 2)  # which dims hold the heads and the sequence
    B, H, L = x.shape[0], x.shape[hd], x.shape[sd]
    f = _freqs(freqs, x, seq_first, torch.float32)
    f = (f[:, :, 0] if seq_first else f[:, 0]).contiguous()  # (B or 1, L, D/2, 2, 2)
    g, x = (t if t.stride(-1) == 1 else t.contiguous() for t in (g, x))
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _backward_kernel[(B * H * L,)](g, x, w.float().contiguous(), f, out, H, L,
                                   g.stride(0), g.stride(hd), g.stride(sd), x.stride(0), x.stride(hd), x.stride(sd),
                                   out.stride(0), out.stride(hd), out.stride(sd), f.stride(0) if f.shape[0] > 1 else 0,
                                   eps, D=D, SPLIT=split_half, num_warps=1 if D <= 64 else 2)
    return out


@torch.library.custom_op("zentrainer::qk_norm_rope", mutates_args=())
def qk_norm_rope(q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, freqs: torch.Tensor,
                 eps: float, split_half: bool = False, seq_first: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    if q.is_cuda and q.dtype != torch.float64:
        try:
            import comfy.quant_ops

            fused = comfy.quant_ops.ck.rms_rope_split_half if split_half else comfy.quant_ops.ck.rms_rope
            oq, ok = fused(q, k, freqs, wq.float(), wk.float(), eps)
            return oq.contiguous(), ok.contiguous()
        except (ImportError, AttributeError):
            pass
    return (reference(q, wq, freqs, eps, split_half, seq_first).contiguous(),
            reference(k, wk, freqs, eps, split_half, seq_first).contiguous())


@qk_norm_rope.register_fake
def _(q, k, wq, wk, freqs, eps, split_half=False, seq_first=False):
    return q.new_empty(q.shape), k.new_empty(k.shape)


@torch.library.custom_op("zentrainer::qk_norm_rope_backward", mutates_args=())
def qk_norm_rope_backward(gq: torch.Tensor, gk: torch.Tensor, q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor,
                          wk: torch.Tensor, freqs: torch.Tensor, eps: float, split_half: bool,
                          seq_first: bool) -> tuple[torch.Tensor, torch.Tensor]:
    return (_backward(gq, q, wq, freqs, eps, split_half, seq_first),
            _backward(gk, k, wk, freqs, eps, split_half, seq_first))


@qk_norm_rope_backward.register_fake
def _(gq, gk, q, k, wq, wk, freqs, eps, split_half, seq_first):
    return q.new_empty(q.shape), k.new_empty(k.shape)


def _setup(ctx, inputs, output):
    q, k, wq, wk, freqs, eps, split_half, seq_first = inputs
    ctx.save_for_backward(q, k, wq, wk, freqs)
    ctx.eps, ctx.split_half, ctx.seq_first = eps, split_half, seq_first


def _grad(ctx, gq, gk):
    q, k, wq, wk, freqs = ctx.saved_tensors
    dq, dk = qk_norm_rope_backward(gq, gk, q, k, wq, wk, freqs, ctx.eps, ctx.split_half, ctx.seq_first)
    return dq, dk, None, None, None, None, None, None


qk_norm_rope.register_autograd(_grad, setup_context=_setup)
