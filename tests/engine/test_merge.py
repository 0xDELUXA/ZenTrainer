"""LoRA merge is exact; resize keeps what matters."""

import torch

from zentrainer.loras import merge


def _lora(out_dim, in_dim, rank, alpha, seed):
    g = torch.Generator().manual_seed(seed)
    return {"m.lora_up.weight": torch.randn(out_dim, rank, generator=g),
            "m.lora_down.weight": torch.randn(rank, in_dim, generator=g),
            "m.alpha": torch.tensor(float(alpha))}


def _delta(sd, strength=1.0):
    r = sd["m.lora_down.weight"].shape[0]
    return sd["m.lora_up.weight"] @ sd["m.lora_down.weight"] * (float(sd["m.alpha"]) / r) * strength


def test_merge_equals_stacking():
    a, b = _lora(12, 10, 4, 4, 0), _lora(12, 10, 2, 1, 1)
    m = merge.merge([(a, 0.7), (b, 0.5)])
    assert m["m.lora_down.weight"].shape == (6, 10)
    assert torch.allclose(_delta(m), _delta(a, 0.7) + _delta(b, 0.5), atol=1e-5)


def test_resize_keeps_a_low_rank_change_exactly_and_caps_rank():
    a = _lora(32, 24, 3, 3, 2)
    big = merge.merge([(a, 1.0), (a, 1.0)])  # rank 6 but the change is rank 3
    small, stats = merge.resize(big, max_rank=8, device="cpu")
    assert small["m.lora_down.weight"].shape[0] == 3
    assert torch.allclose(_delta(small), _delta(a, 2.0), atol=1e-4)
    capped, _ = merge.resize(merge.merge([(_lora(32, 24, 8, 8, 3), 1.0)]), max_rank=2, energy=1.0, device="cpu")
    assert capped["m.lora_down.weight"].shape[0] == 2
