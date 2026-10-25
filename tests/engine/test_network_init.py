"""Continuing from a LoRA file trains it as ComfyUI's LoRA loader applies it: with the file's own
alpha, not this run's (they differ whenever the file's alpha isn't the spec's rank)."""

import sys
import types
from pathlib import Path

import pytest
import torch

pytest.importorskip("comfy_aimdo")  # ComfyUI's own environment (its .venv): else the path would leak into other tests
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "comfyui"))
pytest.importorskip("comfy.weight_adapter")

from comfy.weight_adapter import adapters  # noqa: E402
from zentrainer.harness.network import Network  # noqa: E402
from zentrainer.spec import NetworkSpec  # noqa: E402


def _patcher(w: torch.Tensor):
    model = torch.nn.Module()
    model.attn = torch.nn.Module()
    model.attn.proj = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
    model.attn.proj.weight.data.copy_(w)
    model.attn.proj.weight_function = []
    wrappers = {}
    return types.SimpleNamespace(model=model, add_weight_wrapper=lambda k, f: wrappers.__setitem__(k, f)), wrappers


def _stock(sd: dict, w: torch.Tensor) -> torch.Tensor:
    """The weight ComfyUI's loader makes of the file (comfy.lora reads alpha from <name>.alpha)."""
    for cls in adapters:
        a = cls.load("attn.proj", sd, float(sd["attn.proj.alpha"]), None)
        if a is not None:
            return a.calculate_weight(w.clone(), "attn.proj.weight", 1.0, 1.0, None, lambda x: x, intermediate_dtype=torch.float32)
    raise AssertionError("ComfyUI loads no adapter from it")


@pytest.mark.parametrize("kind", ["lokr_factored_w2", "lora"])
def test_continuing_keeps_the_files_alpha(kind):
    torch.manual_seed(0)
    w = torch.randn(8, 8)
    if kind == "lora":  # rank 2, alpha 8: scale 4
        sd = {"attn.proj.lora_down.weight": torch.randn(2, 8), "attn.proj.lora_up.weight": torch.randn(8, 2),
              "attn.proj.alpha": torch.tensor(8.0)}
        spec = NetworkSpec(algorithm="LoRA", rank=4, targets="all")
    else:  # w2 factored at rank 2, alpha 8
        sd = {"attn.proj.lokr_w1": torch.randn(2, 2), "attn.proj.lokr_w2_a": torch.randn(4, 2),
              "attn.proj.lokr_w2_b": torch.randn(2, 4), "attn.proj.alpha": torch.tensor(8.0)}
        spec = NetworkSpec(algorithm="LoKr", rank=4, targets="all")  # this run's alpha would be 4
    patcher, wrappers = _patcher(w)
    net = Network(patcher, spec, torch.device("cpu"), init_weights=sd)
    assert float(net.state_dict()["attn.proj.alpha"]) == 8.0
    with torch.no_grad():
        trained = wrappers["attn.proj.weight"](w.clone())
    assert torch.allclose(trained, _stock(sd, w), atol=1e-5)
