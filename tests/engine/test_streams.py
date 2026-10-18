"""Multi-stream latents (MiniMax H3: video + audio) packed the way ComfyUI's sampler packs them."""

import math
import sys
import types

import torch

from zentrainer.harness.streams import Streams, first_stream


class _Nested:
    is_nested = True

    def __init__(self, ts):
        self.ts = ts

    def unbind(self):
        return self.ts


class _AV:  # a two-stream latent format: the picture plus an empty (zero) second stream
    latent_channels = 4
    latent_dimensions = 3

    def fix_empty_latent(self, latent):
        return _Nested((latent, latent.new_zeros((latent.shape[0], 3, 2, 5))))


class _Plain:
    latent_channels = 4
    latent_dimensions = 2

    def fix_empty_latent(self, latent):
        return latent


def _comfy_utils(monkeypatch):
    def pack_latents(latents):
        shapes = [t.shape for t in latents]
        return torch.cat([t.reshape(t.shape[0], 1, -1) for t in latents], dim=-1), shapes

    mod = types.ModuleType("comfy.utils")
    mod.pack_latents = pack_latents
    pkg = types.ModuleType("comfy")
    pkg.utils = mod
    monkeypatch.setitem(sys.modules, "comfy", pkg)
    monkeypatch.setitem(sys.modules, "comfy.utils", mod)


def test_single_stream_models_are_left_alone():
    s = Streams(types.SimpleNamespace(latent_format=_Plain()))
    assert not s.multi
    assert s.pack(torch.zeros(1, 4, 8, 8)) is None
    assert s.packed_shape((4, 8, 8)) == (1, 4, 8, 8)


def test_pack_weights_only_the_picture(monkeypatch):
    _comfy_utils(monkeypatch)
    s = Streams(types.SimpleNamespace(latent_format=_AV()))
    assert s.multi
    latent = torch.randn(1, 4, 1, 6, 8)
    p = s.pack(latent)
    n = math.prod(latent.shape[1:])
    assert p.x0.shape == (1, 1, n + 3 * 2 * 5)
    assert torch.equal(p.x0[0, 0, :n], latent.flatten())
    assert p.weight[..., :n].eq(1).all() and p.weight[..., n:].eq(0).all()
    assert s.packed_shape((4, 1, 6, 8)) == tuple(p.x0.shape)


def test_pack_spreads_a_mask_over_the_picture(monkeypatch):
    _comfy_utils(monkeypatch)
    s = Streams(types.SimpleNamespace(latent_format=_AV()))
    mask = torch.zeros(1, 6, 8)
    mask[:, :, :4] = 1  # left half
    p = s.pack(torch.randn(1, 4, 1, 6, 8), mask)
    w = p.weight[0, 0, : 4 * 6 * 8].reshape(4, 1, 6, 8)
    assert w[..., :4].eq(1).all() and w[..., 4:].eq(0).all()


def test_first_stream():
    t = torch.zeros(2)
    assert first_stream(t) is t
    assert first_stream(_Nested((t, torch.ones(1)))) is t


def test_h3_targets():
    from zentrainer.harness.network import matches

    for name in ("blocks.3.attn.qkv_proj", "blocks.3.attn.out_proj", "blocks.3.mlp.fc1", "blocks.49.mlp.fc2"):
        assert matches(name, "attn_mlp"), name
    for name in ("token_refiner.blocks.0.attn.qkv_proj", "condition_proj", "blocks.3.adaln_proj.linear",
                 "blocks.3.attn.to_gate_compress", "video_patch_proj"):
        assert not matches(name, "attn_mlp"), name
    assert matches("blocks.3.attn.qkv_proj", "attn") and not matches("blocks.3.mlp.fc1", "attn")


def test_rope_split_half_rotates_only_the_first_dims():
    from zentrainer.harness.compat import _rope_split_half

    x = torch.randn(1, 5, 2, 8)
    pe = torch.zeros(1, 5, 1, 2, 2, 2)
    pe[..., 0, 0] = 1  # identity rotation: cos 1, sin 0
    pe[..., 1, 1] = 1
    out = _rope_split_half(x, pe, 4)
    assert torch.allclose(out, x)


class _Frozen(torch.nn.Module):  # a stand-in adapter: adds a constant to the weight / the output
    def __init__(self, k):
        super().__init__()
        self.k = k

    def __call__(self, w):
        return w + self.k

    def h(self, x, base_out):
        return torch.full_like(base_out, self.k)

    def g(self, y):
        return y


def test_training_adapter_is_switchable_and_not_saved():
    from zentrainer.harness.network import _Assisted, _Switch

    sw = _Switch()
    a = _Assisted(_Frozen(1.0), _Frozen(10.0), sw)
    w = torch.zeros(2)
    assert a(w).tolist() == [11.0, 11.0]  # both while training
    sw.on = False
    assert a(w).tolist() == [1.0, 1.0]  # samples: the trained adapter alone
    sw.on = True
    a.multiplier = 0.5  # bypass sets layer info on what it holds; the children need it too
    assert a.h(w, w).tolist() == [11.0, 11.0] and a.trained.multiplier == 0.5
    only = _Assisted(None, _Frozen(10.0), sw)  # a layer only the training adapter touches
    sw.on = False
    assert only(w).tolist() == [0.0, 0.0] and only.h(w, w) == 0


def test_pack_trains_a_real_soundtrack(monkeypatch):
    _comfy_utils(monkeypatch)
    s = Streams(types.SimpleNamespace(latent_format=_AV()))
    lat = torch.ones(2, 4, 2, 3, 3)
    sound = torch.full((3, 2, 7), 0.5)  # longer than the stream (5): cropped
    p = s.pack(lat, None, [sound, None])
    n, m = 4 * 2 * 3 * 3, 3 * 2 * 5
    assert p.x0.shape == (2, 1, n + m)
    assert torch.all(p.x0[0, 0, n:] == 0.5) and torch.all(p.x0[1, 0, n:] == 0)
    assert torch.all(p.weight[0, 0, n:] == 1) and torch.all(p.weight[1, 0, n:] == 0)  # only heard audio counts
    assert torch.all(p.weight[:, 0, :n] == 1)


def test_sample_prompts_fill_from_the_run():
    from zentrainer.spec import SampleSpec, sample_prompts

    s = SampleSpec(prompts=["a", {"prompt": "b", "width": 480, "height": 832, "frames": 73}], width=1024, height=1024, seed=7)
    a, b = sample_prompts(s)
    assert (a.prompt, a.width, a.height, a.frames, a.seed) == ("a", 1024, 1024, 1, 7)
    assert (b.prompt, b.width, b.height, b.frames, b.seed) == ("b", 480, 832, 73, 8)


def test_lora_plus_groups_keep_their_ratio_through_set_lr():
    from zentrainer.harness import optim
    from zentrainer.spec import OptimSpec

    down, up = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
    spec = OptimSpec(learning_rate=1e-4, lora_plus=4)
    opt, sched = optim.build([{"params": [down], "lr": 1e-4}, {"params": [up], "lr": 4e-4, "lr_mult": 4.0}], spec)
    assert [g["lr"] for g in opt.param_groups] == [1e-4, 4e-4]
    optim.set_lr(opt, sched, 2e-4)
    assert [round(g["lr"], 12) for g in opt.param_groups] == [2e-4, 8e-4]


def test_network_disabled_zeroes_then_restores():
    import types

    from zentrainer.harness.network import Network

    p = torch.nn.Parameter(torch.ones(3))
    net = types.SimpleNamespace(params={"a.lora_up.weight": p})
    with Network.disabled(net):
        assert torch.all(p == 0)
    assert torch.all(p == 1)
