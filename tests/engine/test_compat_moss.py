"""The MOSS attention patch (harness/compat.py _moss) on a ComfyUI whose attention takes more."""

import sys
import types

import torch

from zentrainer.harness import compat


def _install(monkeypatch, in_training):
    calls = []

    def attention(q, k, v, q_norm, k_norm, pe, heads, comfy_attention, transformer_options,
                  ref_kv=None, kv_out=None, mask=None):  # as ComfyUI after #17296 / #17320
        calls.append((ref_kv, kv_out, mask))
        return "comfy"

    moss = types.ModuleType("comfy.ldm.moss_image.model")
    moss.attention = attention
    mm = types.ModuleType("comfy.model_management")
    mm.in_training = in_training
    for name, mod in (("comfy", types.ModuleType("comfy")), ("comfy.ldm", types.ModuleType("comfy.ldm")),
                      ("comfy.ldm.moss_image", types.ModuleType("comfy.ldm.moss_image")),
                      ("comfy.ldm.moss_image.model", moss), ("comfy.model_management", mm)):
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules["comfy"].model_management = mm
    monkeypatch.setattr(compat, "_applied", set())
    monkeypatch.setattr(compat, "fused_qk_rope", True)
    compat._moss()
    return moss, calls


def test_new_arguments_reach_comfyui(monkeypatch):
    moss, calls = _install(monkeypatch, in_training=True)
    norm = torch.nn.RMSNorm(4).requires_grad_(False)
    args = (torch.zeros(1, 2, 8),) * 3 + (norm, norm, None, 2, None, {})
    mask = torch.zeros(1, 2)
    assert moss.attention(*args, mask=mask) == "comfy"  # by name (TextBlock)
    assert moss.attention(*args, "ref", None, mask) == "comfy"  # positionally (the image blocks)
    assert calls[0][2] is mask and calls[1] == ("ref", None, mask)


def test_sampling_goes_to_comfyui(monkeypatch):
    moss, calls = _install(monkeypatch, in_training=False)
    norm = torch.nn.RMSNorm(4).requires_grad_(False)
    assert moss.attention(*(torch.zeros(1, 2, 8),) * 3, norm, norm, None, 2, None, {}) == "comfy"
    assert calls == [(None, None, None)]
