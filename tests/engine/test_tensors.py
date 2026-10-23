"""The safetensors inspector reads a file from its header alone, and a copy streams it: recast,
pruned, with new metadata, and still a file safetensors loads."""

import pytest
import torch
from safetensors.torch import load_file, save_file

from zentrainer.models import tensors


def _lora(path):
    sd = {f"diffusion_model.layers.{i}.attn.qkv.{part}": torch.randn(*shape)
          for i in (2, 10, 22) for part, shape in (("lora_down.weight", (4, 16)), ("lora_up.weight", (24, 4)))}
    sd.update({f"diffusion_model.layers.{i}.attn.qkv.alpha": torch.tensor(4.0) for i in (2, 10, 22)})
    save_file(sd, str(path), metadata={"ss_network_dim": "4", "note": "hi"})
    return sd


def test_inspect_reads_the_header(tmp_path):
    path = tmp_path / "a.safetensors"
    _lora(path)
    info = tensors.inspect(path)
    assert info["kind"] == "lora" and info["algorithm"] == "LoRA" and info["ranks"] == [4] and info["layers"] == 3
    assert info["metadata"] == {"ss_network_dim": "4", "note": "hi"}
    assert info["params"] == 3 * (4 * 16 + 24 * 4 + 1)
    names = [t["name"] for t in info["tensors"]]
    assert names.index("diffusion_model.layers.2.attn.qkv.alpha") < names.index("diffusion_model.layers.10.attn.qkv.alpha")
    assert info["by_dtype"]["F32"]["tensors"] == 9


def test_a_copy_recasts_prunes_and_takes_new_metadata(tmp_path):
    src, dst = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    sd = _lora(src)
    want = tensors.plan(src, "bf16", ["diffusion_model.layers.22"])
    stats = tensors.save_copy(src, dst, {"note": "edited"}, "bf16", ["diffusion_model.layers.22"])
    assert stats["tensors"] == want["tensors"] == 6 and stats["dropped"] == 3 and stats["converted"] == 6
    out = load_file(str(dst))
    assert not any(".22." in k for k in out)
    for k, v in out.items():
        assert v.dtype == torch.bfloat16 and torch.equal(v, sd[k].to(torch.bfloat16))
    assert tensors.read_header(dst)[0] == {"note": "edited"}
    assert dst.stat().st_size - tensors.read_header(dst)[2] == want["bytes"]


def test_a_plain_copy_is_byte_for_byte_and_keeps_metadata(tmp_path):
    src, dst = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    sd = _lora(src)
    tensors.save_copy(src, dst)
    out = load_file(str(dst))
    assert out.keys() == sd.keys() and all(torch.equal(out[k], sd[k]) for k in sd)
    assert tensors.read_header(dst)[0] == {"ss_network_dim": "4", "note": "hi"}
    with pytest.raises(FileExistsError):
        tensors.save_copy(src, dst)
    with pytest.raises(ValueError):
        tensors.save_copy(src, src, overwrite=True)


def test_quantized_weights_and_their_scales_are_left_as_they_are(tmp_path):
    src, dst = tmp_path / "q.safetensors", tmp_path / "q2.safetensors"
    sd = {"blocks.0.weight": torch.randint(-127, 127, (8, 8), dtype=torch.int8),
          "blocks.0.weight_scale": torch.rand(8, dtype=torch.float32),
          "blocks.0.bias": torch.randn(8, dtype=torch.float32)}
    save_file(sd, str(src))
    assert tensors.inspect(src)["quantized"]
    stats = tensors.save_copy(src, dst, dtype="bf16")
    out = load_file(str(dst))
    assert stats["converted"] == 1 and out["blocks.0.bias"].dtype == torch.bfloat16
    assert torch.equal(out["blocks.0.weight"], sd["blocks.0.weight"])
    assert out["blocks.0.weight_scale"].dtype == torch.float32
