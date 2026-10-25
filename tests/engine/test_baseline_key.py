"""A run's step-0 samples are reused from an earlier run only when they'd look the same: a base LoRA
or a turbo LoRA (both on for samples) makes a baseline of its own."""

import types

from zentrainer.harness.callbacks import Samples
from zentrainer.spec import TrainSpec


def _key(**over) -> str:
    d = {"name": "x", "model": {"ckpt_name": "m.safetensors"}, "dataset": {"folder": "d"},
         "samples": {"prompts": ["a portrait"]}}
    for section, values in over.items():
        d.setdefault(section, {}).update(values)
    spec = TrainSpec.from_dict(d)
    s = object.__new__(Samples)
    s.items = [types.SimpleNamespace(width=1024, height=1024, frames=1, seed=7, control=None)]
    s.prompts = ["a portrait"]
    return s._baseline_key(types.SimpleNamespace(spec=spec), 0)


def test_the_same_settings_share_a_baseline():
    assert _key() == _key()


def test_a_base_or_turbo_lora_makes_its_own_baseline():
    plain = _key()
    assert _key(network={"base_lora": "base.safetensors"}) != plain
    assert _key(network={"base_lora": "base.safetensors", "base_strength": 0.5}) != _key(network={"base_lora": "base.safetensors"})
    assert _key(samples={"prompts": ["a portrait"], "turbo_lora": "turbo.safetensors"}) != plain
