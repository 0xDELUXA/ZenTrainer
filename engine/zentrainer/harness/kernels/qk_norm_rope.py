"""Per-head QK RMSNorm and RoPE as one op that trains (memory.fused_qk_rope): Krea 2's attention
runs its q/k norms and then the rope as separate ops while training (comfy_kitchen's fused
rms_rope has no backward), each a pass over q and k plus casts to fp32 and back.

    forward:   y = rope(n * w),  n = x * r,  r = rsqrt(mean(x²) + eps)
               comfy_kitchen's rms_rope: fp32 inside, rounded once
    backward:  g_z = Mᵀ g_y      each pair rotated back (M: the rope's 2x2 per pair, from freqs)
               g_n = g_z * w
               g_x = r * (g_n - n * mean(g_n * n))
               one Triton kernel, a program per row; plain torch without Triton or off CUDA

x is (B, H, L, D), any strides; w the norm's weight (D,) in fp32 (Krea 2's 1 + scale: in bf16 it
would round to steps of 1/128); freqs ComfyUI's (B or 1, 1, L, D/2, 2, 2). The weights and freqs
get no gradient: callers use the separate ops when a norm's weight trains. Registered as custom
ops, so torch.compile keeps the fused kernels whole instead of tracing into them.
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


def reference(x: torch.Tensor, w: torch.Tensor, freqs: torch.Tensor, eps: float) -> torch.Tensor:
    """The maths in plain torch at x's precision (at least fp32): what the op computes."""
    dt = torch.promote_types(x.dtype, torch.float32)
    n = F.rms_norm(x.to(dt), (x.shape[-1],), weight=w.to(dt), eps=eps)
    p = n.reshape(*n.shape[:-1], -1, 1, 2)
    f = freqs.to(dt)
    return (f[..., 0] * p[..., 0] + f[..., 1] * p[..., 1]).reshape(x.shape).to(x.dtype)


def _backward_torch(g: torch.Tensor, x: torch.Tensor, w: torch.Tensor, freqs: torch.Tensor, eps: float) -> torch.Tensor:
    dt = torch.promote_types(x.dtype, torch.float32)
    f = freqs.to(dt)  # (..., D/2, 2, 2): f[..., i, j] = M[i][j]
    g0, g1 = g.to(dt)[..., 0::2], g.to(dt)[..., 1::2]
    gz = torch.stack((f[..., 0, 0] * g0 + f[..., 1, 0] * g1, f[..., 0, 1] * g0 + f[..., 1, 1] * g1), -1).flatten(-2)
    gn = gz * w.to(dt)
    xf = x.to(dt)
    r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    n = xf * r
    return (r * (gn - n * (gn * n).mean(-1, keepdim=True))).to(x.dtype)


if TRITON_AVAILABLE:

    @triton.jit
    def _backward_kernel(G, X, W, FR, OUT, H, L, sgb, sgh, sgl, sxb, sxh, sxl, sfb, eps,
                         D: tl.constexpr):
        row = tl.program_id(0)
        b = row // (H * L)
        h = (row // L) % H
        t = row % L
        k = tl.arange(0, D // 2)
        g = G + b.to(tl.int64) * sgb + h.to(tl.int64) * sgh + t.to(tl.int64) * sgl
        x = X + b.to(tl.int64) * sxb + h.to(tl.int64) * sxh + t.to(tl.int64) * sxl
        f = FR + b.to(tl.int64) * sfb + t.to(tl.int64) * (D * 2) + k * 4
        xe = tl.load(x + 2 * k).to(tl.float32)
        xo = tl.load(x + 2 * k + 1).to(tl.float32)
        ge = tl.load(g + 2 * k).to(tl.float32)
        go = tl.load(g + 2 * k + 1).to(tl.float32)
        we = tl.load(W + 2 * k)
        wo = tl.load(W + 2 * k + 1)
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
        out = OUT + row.to(tl.int64) * D
        tl.store(out + 2 * k, (r * (gne - ne * dot)).to(OUT.dtype.element_ty))
        tl.store(out + 2 * k + 1, (r * (gno - no * dot)).to(OUT.dtype.element_ty))


def _freqs_for(freqs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """freqs as (B or 1, L, D/2, 2, 2), fp32, contiguous, cut to x's length (as ComfyUI's rope does)."""
    f = freqs[:, 0] if freqs.dim() == 6 else freqs
    if f.shape[1] != x.shape[2]:
        f = f[:, :x.shape[2]]
    return f.float().contiguous()


def _backward(g: torch.Tensor, x: torch.Tensor, w: torch.Tensor, freqs: torch.Tensor, eps: float) -> torch.Tensor:
    B, H, L, D = x.shape
    if not (TRITON_AVAILABLE and x.is_cuda and D in (64, 128, 256) and x.dtype in (torch.bfloat16, torch.float16, torch.float32)):
        return _backward_torch(g, x, w, freqs, eps)
    f = _freqs_for(freqs, x)
    g, x = (t if t.stride(-1) == 1 else t.contiguous() for t in (g, x))
    out = torch.empty((B, H, L, D), device=x.device, dtype=x.dtype)
    _backward_kernel[(B * H * L,)](g, x, w.float().contiguous(), f, out, H, L, g.stride(0), g.stride(1), g.stride(2),
                                   x.stride(0), x.stride(1), x.stride(2), f.stride(0) if f.shape[0] > 1 else 0, eps,
                                   D=D, num_warps=1 if D <= 64 else 2)
    return out


def _forward(x: torch.Tensor, w: torch.Tensor, freqs: torch.Tensor, eps: float) -> torch.Tensor:
    return reference(x, w, freqs, eps)


@torch.library.custom_op("zentrainer::qk_norm_rope", mutates_args=())
def qk_norm_rope(q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, freqs: torch.Tensor,
                 eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    if q.is_cuda and q.dtype != torch.float64:
        try:
            import comfy.quant_ops

            oq, ok = comfy.quant_ops.ck.rms_rope(q, k, freqs, wq.float(), wk.float(), eps)
            return oq.contiguous(), ok.contiguous()
        except (ImportError, AttributeError):
            pass
    return _forward(q, wq, freqs, eps).contiguous(), _forward(k, wk, freqs, eps).contiguous()


@qk_norm_rope.register_fake
def _(q, k, wq, wk, freqs, eps):
    return q.new_empty(q.shape), k.new_empty(k.shape)


@torch.library.custom_op("zentrainer::qk_norm_rope_backward", mutates_args=())
def qk_norm_rope_backward(gq: torch.Tensor, gk: torch.Tensor, q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor,
                          wk: torch.Tensor, freqs: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    return _backward(gq, q, wq, freqs, eps), _backward(gk, k, wk, freqs, eps)


@qk_norm_rope_backward.register_fake
def _(gq, gk, q, k, wq, wk, freqs, eps):
    return q.new_empty(q.shape), k.new_empty(k.shape)


def _setup(ctx, inputs, output):
    q, k, wq, wk, freqs, eps = inputs
    ctx.save_for_backward(q, k, wq, wk, freqs)
    ctx.eps = eps


def _grad(ctx, gq, gk):
    q, k, wq, wk, freqs = ctx.saved_tensors
    dq, dk = qk_norm_rope_backward(gq, gk, q, k, wq, wk, freqs, ctx.eps)
    return dq, dk, None, None, None, None


qk_norm_rope.register_autograd(_grad, setup_context=_setup)
