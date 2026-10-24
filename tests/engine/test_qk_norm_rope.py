"""The fused q/k RMSNorm + rope (kernels/qk_norm_rope.py) against the separate ops it replaces, in
Krea 2's form and MOSS's: the same forward, and a backward that matches autograd and finite
differences (fp64, CPU)."""

import pytest
import torch
import torch.nn.functional as F

from zentrainer.harness.kernels.qk_norm_rope import qk_norm_rope

B, H, KV, L, D = 2, 4, 2, 9, 16


def _freqs():
    """ComfyUI's rope layout (EmbedND): (1, 1, L, D/2, 2, 2), each pair's [[cos, -sin], [sin, cos]]."""
    a = torch.arange(L, dtype=torch.float64)[:, None] / (10000 ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    return torch.stack([a.cos(), -a.sin(), a.sin(), a.cos()], -1).reshape(1, 1, L, D // 2, 2, 2)


def _krea2(x, w, f):
    """Krea 2's norm then ComfyUI's _apply_rope1 (pairs 2i, 2i+1)."""
    n = F.rms_norm(x, (D,), weight=w, eps=1e-5)
    p = n.reshape(*n.shape[:-1], -1, 1, 2)
    return (f[..., 0] * p[..., 0] + f[..., 1] * p[..., 1]).reshape(x.shape)


def _moss(x, w, f):
    """MOSS's norm then its rope_split_half (pairs i, i + D/2)."""
    n = F.rms_norm(x, (D,), weight=w, eps=1e-5)
    p = n.unflatten(-1, (2, -1)).movedim(-2, -1).unsqueeze(-2)
    return (f[..., 0] * p[..., 0] + f[..., 1] * p[..., 1]).movedim(-1, -2).flatten(-2)


@pytest.mark.parametrize("model", ["krea2", "moss"])
def test_the_fused_op_trains_like_the_separate_ones(model):
    torch.manual_seed(0)
    moss = model == "moss"
    f = _freqs().transpose(1, 2) if moss else _freqs()  # MOSS: (1, L, 1, D/2, 2, 2)
    separate = _moss if moss else _krea2
    wq, wk = 1 + torch.randn(D, dtype=torch.float64) * 0.3, 1 + torch.randn(D, dtype=torch.float64) * 0.3
    q0 = torch.randn(B, L, H * D, dtype=torch.float64).view(B, L, H, D)
    k0 = torch.randn(B, L, KV * D, dtype=torch.float64).view(B, L, KV, D)
    if not moss:  # as Krea 2's rearrange leaves them: (B, H, L, D), not contiguous
        q0, k0 = q0.transpose(1, 2), k0.transpose(1, 2)
    op = lambda q, k: qk_norm_rope(q, k, wq, wk, f, 1e-5, split_half=moss, seq_first=moss)  # noqa: E731
    assert torch.autograd.gradcheck(op, (q0.clone().requires_grad_(), k0.clone().requires_grad_()), eps=1e-6, atol=1e-7)
    q, k = q0.clone().requires_grad_(), k0.clone().requires_grad_()
    qs, ks = q0.clone().requires_grad_(), k0.clone().requires_grad_()
    gq, gk = torch.randn_like(q0), torch.randn_like(k0)
    a, b = op(q, k)
    torch.autograd.backward((a, b), (gq, gk))
    torch.autograd.backward((separate(qs, wq, f), separate(ks, wk, f)), (gq, gk))
    assert torch.allclose(a, separate(q0, wq, f)) and torch.allclose(b, separate(k0, wk, f))
    assert torch.allclose(q.grad, qs.grad, atol=1e-12) and torch.allclose(k.grad, ks.grad, atol=1e-12)
