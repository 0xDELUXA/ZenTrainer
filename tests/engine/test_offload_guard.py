"""The backward guard on streamed weights (memory/offload.py guard_backward)."""

import sys
import types

import pytest
import torch

from zentrainer.harness.memory import offload


@pytest.fixture
def comfy_ops(monkeypatch):
    ops = types.ModuleType("comfy.ops")
    ops.result = None
    ops.cast_bias_weight = lambda *a, **k: ops.result
    pkg = types.ModuleType("comfy")
    pkg.ops = ops
    monkeypatch.setitem(sys.modules, "comfy", pkg)
    monkeypatch.setitem(sys.modules, "comfy.ops", ops)
    yield ops
    offload.guard_backward(False)


def test_streamed_weights_are_copied_while_training(comfy_ops):
    original = comfy_ops.cast_bias_weight
    offload.guard_backward(True)
    w, b = torch.ones(2, 2), torch.ones(2)
    comfy_ops.result = (w, b, ("offload-stream", w, b))
    weight, bias, state = comfy_ops.cast_bias_weight("layer", offloadable=True)
    assert weight is not w and bias is not b and torch.equal(weight, w) and torch.equal(bias, b)
    assert state == ("offload-stream", w, b)  # ComfyUI still releases its own


def test_left_alone_when_nothing_to_guard(comfy_ops):
    offload.guard_backward(True)
    w, b = torch.ones(2, 2), torch.ones(2)
    converted = w.to(torch.float64)  # already the compute stream's own (a dtype or LoRA change)
    for result in ((w, b), (w, b, None), (w, b, (None, w, b)), (converted, None, ("s", w, None))):
        comfy_ops.result = result
        assert all(x is y for x, y in zip(comfy_ops.cast_bias_weight("layer"), result))
    comfy_ops.result = (w, b, ("s", w, b))
    with torch.no_grad():  # samples, the no-grad teacher passes: nothing keeps it
        assert comfy_ops.cast_bias_weight("layer")[0] is w


def test_guard_comes_off(comfy_ops):
    original = comfy_ops.cast_bias_weight
    offload.guard_backward(True)
    offload.guard_backward(True)  # twice is once
    offload.guard_backward(False)
    assert comfy_ops.cast_bias_weight is original


def test_quantized_weights_copy_their_storage():
    kitchen = pytest.importorskip("comfy_kitchen.tensor")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    q = kitchen.QuantizedTensor.from_float(torch.randn(64, 32, device="cuda", dtype=torch.bfloat16), "TensorWiseINT8Layout")
    c = offload._own(q)
    assert type(c) is type(q) and c._qdata.data_ptr() != q._qdata.data_ptr()
    assert torch.equal(c.dequantize(), q.dequantize())
