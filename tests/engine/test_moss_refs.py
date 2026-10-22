"""MOSS Image references (harness/moss_refs.py) on a tiny random MOSS model from the pinned ComfyUI:
without references it's MOSS's own forward, and references reach the picture."""

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("comfy_aimdo")  # ComfyUI's own environment (its .venv): else the path would leak into other tests
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "comfyui"))
moss = pytest.importorskip("comfy.ldm.moss_image.model")

import comfy.model_management  # noqa: E402
import comfy.ops  # noqa: E402

from zentrainer.harness import moss_refs  # noqa: E402


@pytest.fixture
def dm(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "in_training", True)  # the plain torch attention, no kernels
    torch.manual_seed(0)
    m = moss.MOSSImage(dim=64, ffn_dim=128, num_heads=2, num_layers=2, text_dim=32, latent_channels=8,
                       operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.2)
    return m.eval()


def _inputs():
    torch.manual_seed(1)
    return torch.randn(1, 8, 4, 6), torch.tensor([0.5]), torch.randn(1, 5, 32)


def test_rope_without_references_is_moss_own(dm):
    for opts in ({}, {"rope_options": {"shift_y": 1.0, "scale_x": 0.0, "shift_t": 2.0}}):
        assert torch.equal(moss_refs.rope(dm, 4, 6, 5, [], "cpu", opts), dm.rope(4, 6, 5, "cpu", opts))


def test_references_go_on_their_own_rope_axis_value(dm):
    pe = moss_refs.rope(dm, 4, 6, 5, [(4, 6), (2, 2)], "cpu")
    assert pe.shape[1] == 24 + 5 + 24 + 4
    own = dm.rope(4, 6, 5, "cpu")
    assert torch.equal(pe[:, :29], own)
    assert not torch.equal(pe[:, 29:53], own[:, :24])  # r = 1: same rows and columns, told apart


def test_forward_without_references_matches_and_references_matter(dm):
    x, t, ctx = _inputs()
    with torch.no_grad():
        own = dm._forward(x, t, ctx)
        assert torch.allclose(moss_refs.forward(dm, x, t, ctx, []), own, atol=1e-5)
        a = moss_refs.forward(dm, x, t, ctx, [torch.randn(1, 8, 4, 6)])
        b = moss_refs.forward(dm, x, t, ctx, [torch.randn(1, 8, 3, 5)])  # any size
    assert a.shape == own.shape == b.shape
    assert not torch.allclose(a, own) and not torch.allclose(a, b)


def test_wrapper_passes_through_without_references(dm):
    x, t, ctx = _inputs()

    class Executor:
        class_obj = dm

        def __call__(self, *a, **kw):
            return ("own", kw)

    assert moss_refs._wrapper(Executor(), x, t, ctx, {}, control=None)[0] == "own"
    with torch.no_grad():
        out = moss_refs._wrapper(Executor(), x, t, ctx, {}, ref_latents=[torch.randn(1, 8, 4, 6)])
    assert out.shape == x.shape
    with pytest.raises(ValueError, match="index_timestep_zero"):
        moss_refs._wrapper(Executor(), x, t, ctx, {}, ref_latents=[x], ref_latents_method="offset")


def test_clean_references_have_their_own_pass_and_train(dm):
    x, t, ctx = _inputs()
    ref = torch.randn(1, 8, 4, 6)
    with torch.no_grad():
        clean = moss_refs.forward_t0(dm, x, t, ctx, [ref])
        assert not torch.allclose(clean, moss_refs.forward(dm, x, t, ctx, [ref]))  # another layout
        assert not torch.allclose(clean, dm._forward(x, t, ctx))  # the references reach the picture
        assert not torch.allclose(clean, moss_refs.forward_t0(dm, x, t, ctx, [torch.randn(1, 8, 4, 6)]))
    # trains: the picture's pass carries the gradient, the references' pass none
    w = dm.layers[1].proj["q"].weight
    w.requires_grad_(True)
    moss_refs.forward_t0(dm, x, t, ctx, [ref]).square().mean().backward()
    assert w.grad is not None and w.grad.abs().sum() > 0


def test_the_hook_keeps_the_trainers_attention_override(dm):
    x, t, ctx = _inputs()
    seen = []

    def backend(func, *args, **kwargs):
        seen.append(args[1].shape[1])  # keys
        return func(*args, **kwargs)

    with torch.no_grad():
        moss_refs.forward_t0(dm, x, t, ctx, [torch.randn(1, 8, 4, 6)], {"optimized_attention_override": backend})
    # the references' own pass (24 keys), the text embedder's blocks (5), then the picture and text
    # with the references beside (24 + 5 + 24)
    assert seen == [24] * 2 + [5] * 2 + [53] * 2


def test_skip_layer_guidance_replaces_only_the_pictures_blocks(dm):
    x, t, ctx = _inputs()
    seen = []

    def skip(args, extra):  # as ComfyUI's SkipLayerGuidanceDiT: the block, for the picture's batch
        seen.append(args["img"].shape[1])
        return extra["original_block"](args)

    to = {"patches_replace": {"dit": {("single_block", 1): skip}}}
    with torch.no_grad():
        moss_refs.forward_t0(dm, x, t, ctx, [torch.randn(1, 8, 4, 6)], to)
    assert seen == [24 + 5]  # once, in the picture's pass; the references' own pass runs the block itself
