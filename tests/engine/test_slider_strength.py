"""A scalable network's strength (sliders): 1 is the trained adapter as it is, 0 the base model
exactly, -1 the same change the other way; in both attachment paths, and for trained norm/bias
deltas too."""

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("comfy_aimdo")  # ComfyUI's own environment (its .venv): else the path would leak into other tests
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "comfyui"))
lora = pytest.importorskip("comfy.weight_adapter.lora")

from zentrainer.harness.network import _Assisted, _BiasDiff, _Switch  # noqa: E402


def _trained_lora(out=6, inp=5):
    torch.manual_seed(0)
    w = torch.randn(out, inp)
    adapter = lora.LoRAAdapter.create_train(w, rank=2, alpha=2.0)
    with torch.no_grad():
        for p in adapter.parameters():
            p.add_(torch.randn_like(p) * 0.3)  # a LoRA that has learned something (up starts at 0)
    return w, adapter


def test_strength_scales_the_weight_path():
    w, adapter = _trained_lora()
    switch = _Switch()
    layer = _Assisted(adapter, None, switch)
    delta = adapter(w) - w
    assert delta.abs().max() > 0
    for s in (1.0, 0.0, -1.0, 0.5):
        switch.strength = s
        assert torch.allclose(layer(w), w + s * delta, atol=1e-6), s
    switch.strength = 0.0
    assert torch.equal(layer(w), w)  # exactly the base model


def test_strength_scales_the_bypass_path():
    w, adapter = _trained_lora()
    switch = _Switch()
    layer = _Assisted(adapter, None, switch)
    layer.multiplier, layer.is_conv, layer.kw_dict = 1.0, False, {}
    x = torch.randn(3, w.shape[1])
    base_out = x @ w.T
    switch.strength = 1.0
    one = layer.h(x, base_out)
    for s in (0.0, -1.0, 0.5):
        switch.strength = s
        got = layer.h(x, base_out)
        assert torch.allclose(torch.as_tensor(got, dtype=one.dtype).expand_as(one), s * one, atol=1e-6), s


def test_strength_scales_trained_norms_and_biases():
    switch = _Switch()
    d = _BiasDiff((4,), torch.float32, switch)
    with torch.no_grad():
        d.diff.copy_(torch.tensor([1.0, -2.0, 0.5, 3.0]))
    b = torch.ones(4)
    for s in (1.0, 0.0, -1.0):
        switch.strength = s
        assert torch.allclose(d(b), b + s * d.diff), s
    assert torch.allclose(_BiasDiff((4,), torch.float32)(b), b)  # no switch: as before
