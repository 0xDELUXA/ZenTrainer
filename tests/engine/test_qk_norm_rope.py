"""The fused q/k RMSNorm + rope (kernels/qk_norm_rope.py) against the separate ops it replaces:
the same forward, and a backward that matches autograd and finite differences (fp64, CPU)."""

import torch
import torch.nn.functional as F

from zentrainer.harness.kernels.qk_norm_rope import qk_norm_rope

B, H, KV, L, D = 2, 4, 2, 9, 16


def _freqs():
    """ComfyUI's rope layout (EmbedND): (1, 1, L, D/2, 2, 2), each pair's [[cos, -sin], [sin, cos]]."""
    a = torch.arange(L, dtype=torch.float64)[:, None] / (10000 ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    return torch.stack([a.cos(), -a.sin(), a.sin(), a.cos()], -1).reshape(1, 1, L, D // 2, 2, 2)


def _separate(x, w, f):
    """Krea 2's norm then ComfyUI's _apply_rope1."""
    n = F.rms_norm(x, (D,), weight=w, eps=1e-5)
    p = n.reshape(*n.shape[:-1], -1, 1, 2)
    return (f[..., 0] * p[..., 0] + f[..., 1] * p[..., 1]).reshape(x.shape)


def test_the_fused_op_trains_like_the_separate_ones():
    torch.manual_seed(0)
    f = _freqs()
    wq, wk = 1 + torch.randn(D, dtype=torch.float64) * 0.3, 1 + torch.randn(D, dtype=torch.float64) * 0.3
    # laid out as Krea 2's rearrange leaves them (not contiguous), k with fewer heads (GQA)
    q0 = torch.randn(B, L, H * D, dtype=torch.float64).view(B, L, H, D).transpose(1, 2)
    k0 = torch.randn(B, L, KV * D, dtype=torch.float64).view(B, L, KV, D).transpose(1, 2)
    assert torch.autograd.gradcheck(lambda q, k: qk_norm_rope(q, k, wq, wk, f, 1e-5),
                                    (q0.clone().requires_grad_(), k0.clone().requires_grad_()), eps=1e-6, atol=1e-7)
    q, k = q0.clone().requires_grad_(), k0.clone().requires_grad_()
    qs, ks = q0.clone().requires_grad_(), k0.clone().requires_grad_()
    gq, gk = torch.randn_like(q0), torch.randn_like(k0)
    a, b = qk_norm_rope(q, k, wq, wk, f, 1e-5)
    torch.autograd.backward((a, b), (gq, gk))
    torch.autograd.backward((_separate(qs, wq, f), _separate(ks, wk, f)), (gq, gk))
    assert torch.allclose(a, _separate(q0, wq, f)) and torch.allclose(b, _separate(k0, wk, f))
    assert torch.allclose(q.grad, qs.grad, atol=1e-12) and torch.allclose(k.grad, ks.grad, atol=1e-12)
