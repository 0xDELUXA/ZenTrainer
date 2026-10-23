"""Compact keeps the file's strongest directions across all its layers, writes LoRA and LoKr
files that load as ComfyUI computes them, and can weigh layers by the base model's weights."""

import torch
from safetensors.torch import load_file, save_file

from zentrainer.loras import compact


def _lokr_delta(sd, name):
    """ComfyUI's LoKr: kron(w1, w2) * alpha / rank of the decomposed factor (1 if neither is)."""
    p = {k[len(name) + 1:]: v.float() for k, v in sd.items() if k.startswith(name + ".")}
    w1 = p["lokr_w1"] if "lokr_w1" in p else p["lokr_w1_a"] @ p["lokr_w1_b"]
    w2 = p["lokr_w2"] if "lokr_w2" in p else p["lokr_w2_a"] @ p["lokr_w2_b"]
    dim = p["lokr_w2_b"].shape[0] if "lokr_w2" not in p else p["lokr_w1_b"].shape[0] if "lokr_w1" not in p else None
    scale = float(p["alpha"]) / dim if dim else 1.0
    return torch.kron(w1, w2) * scale


def _lora_delta(sd, name):
    up, down = sd[f"{name}.lora_up.weight"].float(), sd[f"{name}.lora_down.weight"].float()
    return up @ down * (float(sd[f"{name}.alpha"]) / down.shape[0])


def _save(tmp_path, sd, file="in.safetensors"):
    path = tmp_path / file
    save_file(sd, str(path))
    return path


def test_a_lora_budget_goes_to_the_layer_that_changes_most(tmp_path):
    g = torch.Generator().manual_seed(0)
    sd = {"big.lora_up.weight": torch.randn(32, 8, generator=g), "big.lora_down.weight": torch.randn(8, 24, generator=g),
          "big.alpha": torch.tensor(8.0),
          "quiet.lora_up.weight": torch.randn(32, 8, generator=g) * 1e-3, "quiet.lora_down.weight": torch.randn(8, 24, generator=g),
          "quiet.alpha": torch.tensor(8.0), "norm.weight": torch.ones(4)}
    path = _save(tmp_path, sd)
    an = compact.analyze(path)
    ks = compact.ranks(an, keep=0.99)
    by = dict(zip([layer.name for layer in an.layers], ks))
    assert by["quiet"] == 0 and 0 < by["big"] <= 8
    full = compact.ranks(an, keep=1.0)
    out = tmp_path / "out.safetensors"
    compact.build(path, out, an, full, dtype="fp32")
    o = load_file(str(out))
    for name in ("big", "quiet"):
        assert torch.allclose(_lora_delta(o, name), _lora_delta(sd, name), atol=1e-4)
    assert torch.equal(o["norm.weight"], sd["norm.weight"])
    small = tmp_path / "small.safetensors"
    st = compact.build(path, small, an, ks, dtype="fp32")
    assert "quiet.lora_up.weight" not in load_file(str(small)) and st["layers_kept"] == 1 and st["kept"] >= 0.99


def test_a_parameter_budget_is_never_overshot(tmp_path):
    g = torch.Generator().manual_seed(1)
    sd = {}
    for n in ("a", "b", "c"):
        sd.update({f"{n}.lora_up.weight": torch.randn(16, 6, generator=g), f"{n}.lora_down.weight": torch.randn(6, 16, generator=g),
                   f"{n}.alpha": torch.tensor(6.0)})
    an = compact.analyze(_save(tmp_path, sd))
    for budget in (40, 100, 300):
        assert compact.summary(an, compact.ranks(an, params=budget))["params"] <= budget
    shares = [p[1] for p in compact.curve(an)]
    assert shares == sorted(shares) and shares[-1] > 0.999


def _lokr(g, out1=4, in1=4, out2=12, in2=10):
    return {"m.lokr_w1": torch.randn(out1, in1, generator=g), "m.lokr_w2": torch.randn(out2, in2, generator=g),
            "m.alpha": torch.tensor(4.0)}


def test_a_lokr_kept_as_one_is_its_best_cut_of_w2(tmp_path):
    g = torch.Generator().manual_seed(2)
    sd = _lokr(g)
    path = _save(tmp_path, sd)
    an = compact.analyze(path)
    assert an.layers[0].kind == "lokr" and an.layers[0].s.numel() == 10
    full, out = tmp_path / "full.safetensors", tmp_path / "k3.safetensors"
    compact.build(path, full, an, [10], dtype="fp32")
    assert torch.allclose(_lokr_delta(load_file(str(full)), "m"), _lokr_delta(sd, "m"), atol=1e-4)
    compact.build(path, out, an, [3], dtype="fp32")
    o = load_file(str(out))
    assert o["m.lokr_w2_a"].shape == (12, 3) and o["m.lokr_w2_b"].shape == (3, 10)
    u, s, v = torch.linalg.svd(sd["m.lokr_w2"])
    best = torch.kron(sd["m.lokr_w1"], u[:, :3] * s[:3] @ v[:3])
    assert torch.allclose(_lokr_delta(o, "m"), best, atol=1e-4)


def test_a_decomposed_lokr_keeps_comfyuis_scale(tmp_path):
    g = torch.Generator().manual_seed(3)
    sd = {"m.lokr_w1": torch.randn(2, 3, generator=g), "m.lokr_w2_a": torch.randn(8, 4, generator=g),
          "m.lokr_w2_b": torch.randn(4, 6, generator=g), "m.alpha": torch.tensor(2.0)}
    path = _save(tmp_path, sd)
    an = compact.analyze(path)
    out = tmp_path / "o.safetensors"
    compact.build(path, out, an, compact.ranks(an, keep=1.0), dtype="fp32")
    assert torch.allclose(_lokr_delta(load_file(str(out)), "m"), _lokr_delta(sd, "m"), atol=1e-4)


def test_a_lokr_made_a_lora_is_exact_at_full_rank_and_best_below(tmp_path):
    g = torch.Generator().manual_seed(4)
    sd = _lokr(g, 3, 2, 5, 4)
    path = _save(tmp_path, sd)
    an = compact.analyze(path, fmt="lora")
    assert an.layers[0].kind == "lokr_lora" and an.layers[0].s.numel() == 2 * 4
    full = tmp_path / "full.safetensors"
    compact.build(path, full, an, [8], dtype="fp32")
    want = _lokr_delta(sd, "m")
    assert torch.allclose(_lora_delta(load_file(str(full)), "m"), want, atol=1e-4)
    k3 = tmp_path / "k3.safetensors"
    compact.build(path, k3, an, [3], dtype="fp32")
    u, s, v = torch.linalg.svd(want)
    assert torch.allclose(_lora_delta(load_file(str(k3)), "m"), u[:, :3] * s[:3] @ v[:3], atol=1e-4)


def test_the_base_model_lowers_a_layer_beside_a_large_weight(tmp_path):
    g = torch.Generator().manual_seed(5)
    sd = {}
    for n in ("a", "b"):
        sd.update({f"diffusion_model.{n}.lora_up.weight": torch.randn(16, 4, generator=g),
                   f"diffusion_model.{n}.lora_down.weight": torch.randn(4, 16, generator=g),
                   f"diffusion_model.{n}.alpha": torch.tensor(4.0)})
    path = _save(tmp_path, sd)
    base = tmp_path / "base.safetensors"
    q = torch.randint(-127, 127, (16, 16), dtype=torch.int8)
    save_file({"a.weight": torch.randn(16, 16) * 0.01,  # small weight: a's change matters a lot
               "b.weight": q, "b.weight_scale": torch.full((16, 1), 10.0)}, str(base))  # large, int8 rows
    an = compact.analyze(path, base=base)
    assert an.base_matched == 2
    ks = dict(zip([layer.name for layer in an.layers], compact.ranks(an, keep=0.999)))
    assert ks["diffusion_model.a"] == 4 and ks["diffusion_model.b"] == 0
