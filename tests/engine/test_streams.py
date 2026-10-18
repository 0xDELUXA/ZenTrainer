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


def test_released_text_encoder_loads_again():
    from zentrainer.harness.models import Models

    loads = []
    m = Models(patcher=None, _clip=None, vae=None, _clip_loader=lambda: loads.append(1) or "clip")
    assert m.clip == "clip"
    m.release_clip()
    assert m._clip is None
    assert m.clip == "clip" and len(loads) == 2  # a later prompt can still be encoded


def test_manual_tries_batch_into_one_session_and_pick_up_late_ones():
    from zentrainer.harness.control import Control
    from zentrainer.harness.trainer import Trainer

    calls = []

    class Samples:
        def render(self, t):
            calls.append("run samples")

        def render_manual(self, t, tries, more):
            batch = list(tries)
            t.control.send("sample", prompt="late")  # queued while the first batch renders
            nxt = more()
            calls.append(("session", [a["prompt"] for a in batch], [a["prompt"] for a in nxt]))

    t = object.__new__(Trainer)
    t.control = Control()
    t._samples = Samples()
    t.control.send("sample", prompt="a")
    t.control.send("sample", prompt="b")
    t._apply(t.control.drain())
    assert calls == [("session", ["a", "b"], ["late"])]


def test_block_range():
    from zentrainer.harness.network import in_blocks

    assert in_blocks("diffusion_model.blocks.25.attn.qkv", "20-49")
    assert not in_blocks("diffusion_model.blocks.3.attn.qkv", "20-49")
    assert in_blocks("diffusion_model.double_blocks.7.img_mlp.0", "5-")
    assert in_blocks("diffusion_model.layers.2.mlp", "-2") and not in_blocks("diffusion_model.layers.3.mlp", "-2")
    assert not in_blocks("diffusion_model.final_layer.linear", "20-49")
    assert in_blocks("diffusion_model.final_layer.linear", "")


def test_late_blocks_train_without_input_grad_and_early_blocks_skip_backward():
    import torch.utils.checkpoint as cp

    early, late = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    early.requires_grad_(False)
    late.requires_grad_(False)
    lora = torch.nn.Parameter(torch.zeros(4))  # stands in for an adapter on the late block
    hits = []
    early.register_full_backward_hook(lambda *a: hits.append("early"))
    x = torch.randn(2, 4)  # no requires_grad: the input carries none
    h = cp.checkpoint(early, x, use_reentrant=False)
    y = cp.checkpoint(lambda t: late(t) + lora, h, use_reentrant=False)
    y.sum().backward()
    assert lora.grad is not None and torch.all(lora.grad == 2)
    assert hits == []  # nothing flowed back into the untrained early block


def test_locked_changes_at_resume():
    from zentrainer.spec import TrainSpec, locked_changes

    old = TrainSpec().to_dict()
    new = TrainSpec().to_dict()
    new["memory"]["bypass_mode"] = True
    new["optim"]["learning_rate"] = 5e-5
    new["samples"]["steps"] = 6
    assert locked_changes(old, new) == []
    new["network"]["rank"] = 32
    new["optim"]["optimizer"] = "Prodigy"
    assert locked_changes(old, new) == ["network.rank", "optim.optimizer"]


def test_preview_adapter_only_while_sampling():
    from zentrainer.harness.network import _Assisted, _Switch

    sw = _Switch()
    assist = lambda w: w + 1  # training adapter: +1
    turbo = lambda w: w + 10  # preview adapter: +10, used at 0.5
    a = _Assisted(None, assist, sw, turbo, 0.5)
    w = torch.zeros(2)
    assert torch.all(a(w) == 1)  # training: the training adapter only
    sw.on = False
    assert torch.all(a(w) == 5)  # sampling: half the turbo, no training adapter


def test_tread_routes_video_tokens_around_the_middle_blocks():
    import types as _t

    from zentrainer.harness.tread import patches

    n_blocks, N, C = 10, 40, 3
    layout = _t.SimpleNamespace(segments=[(0, 6, "text"), (6, 34, "video"), (34, 40, "audio")])
    seen = []

    def block(i):
        def run(args):  # each block adds its index and records the length it saw
            seen.append((i, args["img"].shape[0], len(args["mod_segments"]), args["rope_freqs"].shape[1]))
            return {"img": args["img"] + i}
        return run

    rep = patches(n_blocks, 0.5, torch.Generator().manual_seed(0), start=2, end=7)
    h = torch.zeros(N, C)
    segs = [(0, 6, 1), (6, 34, 0), (34, 40, 2)]
    rope = torch.zeros(1, N, 1, 2, 2, 2)
    for i in range(n_blocks):
        args = {"img": h, "mod_segments": segs, "rope_freqs": rope, "layout": layout, "t_emb": None}
        h = rep[("double_block", i)](args, {"original_block": block(i)})["img"] if ("double_block", i) in rep else block(i)(args)["img"]
    lengths = {i: n for i, n, _, _ in seen}
    assert lengths[0] == lengths[1] == N and lengths[7] == lengths[9] == N  # whole outside the window
    assert all(lengths[i] == N - 14 for i in range(2, 7))  # half the 28 video tokens routed
    assert all(r == n for _, n, _, r in seen)  # rope follows the tokens
    text_total = sum(range(n_blocks))
    assert torch.all(h[:6] == text_total) and torch.all(h[34:] == text_total)  # text/audio saw every block
    routed = h[6:34, 0]
    assert ((routed == text_total) | (routed == text_total - sum(range(2, 7)))).all()  # video: all, or skipped 2-6
    assert int((routed != text_total).sum()) == 14


def test_ema_average_applied_and_restored():
    from zentrainer.harness.ema import EMA

    p = torch.nn.Parameter(torch.zeros(3))
    ema = EMA({"a": p}, 0.9)
    for step in range(200):
        with torch.no_grad():
            p.fill_(1.0)
        ema.update(step)
    assert torch.allclose(ema.shadow["a"], torch.ones(3), atol=1e-3)
    with torch.no_grad():
        p.fill_(5.0)
    with ema.applied() as live:
        assert torch.allclose(p, torch.ones(3), atol=1e-3) and torch.all(live["a"] == 5.0)
    assert torch.all(p == 5.0)  # the live weights are back
    st = ema.state_dict()
    other = EMA({"a": torch.nn.Parameter(torch.zeros(3))}, 0.9)
    other.load_state_dict(st)
    assert torch.allclose(other.shadow["a"], ema.shadow["a"])


def test_automagic3_trains_and_keeps_its_own_lr():
    from zentrainer.harness import optim
    from zentrainer.spec import OptimSpec

    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(8, 8) * 0.1)
    target = torch.randn(8, 8)
    opt, sched = optim.build([w], OptimSpec(optimizer="Automagic3", learning_rate=1e-3, warmup_steps=10))
    first = None
    for _ in range(60):
        loss = ((w - target) ** 2).mean()
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    assert loss.item() < first
    assert optim.current_lr(opt) != 1e-3  # it steered the LR itself


def test_adapter_shape_check_skips_other_builds_layers():
    from zentrainer.harness.network import _fits

    m = torch.nn.Module()
    m.diffusion_model = torch.nn.Module()
    m.diffusion_model.proj = torch.nn.Linear(8, 96)  # a pruned build: 8 inputs, where the LoRA expects 2688
    m.diffusion_model.conv = torch.nn.Conv2d(4, 6, 3)
    assert _fits(m, "diffusion_model.proj.weight", (96, 8))
    assert not _fits(m, "diffusion_model.proj.weight", (96, 2688))
    assert _fits(m, "diffusion_model.conv.weight", (6, 4 * 9))
    assert not _fits(m, "diffusion_model.missing.weight", (6, 4))


def test_edit_pairs_scan_and_reference_conditioning(tmp_path):
    from PIL import Image

    from zentrainer.harness.data import scan
    from zentrainer.harness.edit import with_refs

    for name in ("a.png", "a_control.png", "a_control2.jpg", "b.png"):
        Image.new("RGB", (8, 8)).save(tmp_path / name)
    (tmp_path / "a.txt").write_text("make it red")
    items = {it.path.name: it for it in scan(tmp_path, warn=False)}
    assert set(items) == {"a.png", "b.png"}  # the inputs aren't trained as images of their own
    assert [c.name for c in items["a.png"].controls] == ["a_control.png", "a_control2.jpg"]
    assert items["b.png"].controls == []

    cond = [[torch.zeros(1, 4, 8), {"pooled_output": None}]]
    ref = torch.ones(16, 2, 2)
    out = with_refs(cond, [ref])
    assert out[0][1]["reference_latents"][0].shape == (1, 16, 2, 2)
    assert "reference_latents" not in cond[0][1]  # the cached caption conditioning is left alone


def test_grounded_template_keeps_the_encoders_own_system_prompt():
    from types import SimpleNamespace

    from zentrainer.harness.edit import VISION, grounded_template, instruction

    krea = "<|im_start|>system\nDescribe the image<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    clip = SimpleNamespace(tokenizer=SimpleNamespace(llama_template=krea))
    t = grounded_template(clip, 2)
    assert t.startswith("<|im_start|>system\nDescribe the image")  # not qwen3vl's default picture template
    assert t.count(VISION) == 2 and t.index(VISION) == t.index("user\n") + len("user\n")
    assert t.format("make it red").count("make it red") == 1
    assert grounded_template(SimpleNamespace(tokenizer=SimpleNamespace()), 1) is None  # a T5/CLIP encoder
    assert instruction("add a hat", "Make the following edit: {}") == "Make the following edit: add a hat"
    assert instruction("add a hat", "") == "add a hat"


def test_w8a8_kernels_match_torch():
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA (Triton kernels)")
    from zentrainer.harness.kernels.w8a8_backward import TRITON_AVAILABLE, int8_matmul, quantize_rows

    if not TRITON_AVAILABLE:
        pytest.skip("needs Triton")
    torch.manual_seed(0)
    g = torch.randn(300, 512, device="cuda", dtype=torch.bfloat16)
    s = torch.rand(512, device="cuda") + 0.5
    q = torch.randint(-127, 128, (512, 256), device="cuda", dtype=torch.int8)
    gq, step = quantize_rows(g, s, 7)
    x = g.float() * s
    assert torch.allclose(step, x.abs().amax(1) / 127, rtol=1e-5)
    assert ((gq.float() * step[:, None] - x).abs() <= step[:, None] + 1e-6).all()  # within one int8 step
    ref = (gq.float() @ q.float()) * step[:, None]
    out = int8_matmul(gq, q, step).float()
    assert ((out - ref).norm() / ref.norm()) < 5e-3


def test_compiled_blocks_are_undone():
    from zentrainer.harness.compile import compiled_blocks

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(3))

    net = Net()
    with compiled_blocks(net, True) as n:
        assert n == 3 and all("forward" in b.__dict__ for b in net.blocks)
    assert not any("forward" in b.__dict__ for b in net.blocks)  # the cached model is left as it was


def test_failed_run_frames_let_go_of_their_tensors():
    import gc
    import weakref

    from zentrainer.harness.trainer import _release_frames

    ref = None

    def step():
        nonlocal ref
        activations = torch.zeros(1000)
        ref = weakref.ref(activations)
        raise torch.OutOfMemoryError("CUDA out of memory")

    try:
        try:
            step()
        except torch.OutOfMemoryError as inner:
            raise RuntimeError("Out of VRAM at step 1") from inner
    except RuntimeError as e:
        kept = e  # ComfyUI keeps the job's exception
        _release_frames(kept)
    gc.collect()
    assert ref() is None and "step 1" in str(kept)


def test_sharpest_frame_picks_the_detailed_one(tmp_path):
    import numpy as np

    from zentrainer.harness import video

    frames = []
    rng = np.random.default_rng(0)
    for i in range(12):  # flat grey frames, one noisy (sharp) frame at index 7
        f = np.full((64, 64, 3), 0.5, dtype=np.float32)
        if i == 7:
            f = rng.random((64, 64, 3)).astype(np.float32)
        frames.append(f)
    path = tmp_path / "clip.mp4"
    video.write(path, torch.from_numpy(np.stack(frames)), 12)
    p = video.probe(path)
    assert abs(video.sharpest_frame(path, p, look=12) - 7) <= 0


def test_image_flags_rules():
    from zentrainer.dataset_stats import image_flags

    ok = {"hash": 0, "bright": 0.5, "contrast": 0.2, "sharp": 10.0, "sat": 0.3, "border": 0.0}
    looks = {
        "a": ok,
        "b": {**ok, "hash": 0b1, "bright": 0.08},              # dark, and a near-twin of a (1 bit)
        "c": {**ok, "hash": 2**63 - 1, "sharp": 1.0},          # blurry against the set's median
        "d": {**ok, "hash": 2**40 + 2**50 + 2**60 + 2**30 + 2**20 + 2**10 + 7, "border": 0.2, "sat": 0.01},
    }
    flags = image_flags(looks, {"a": 1024, "b": 1024, "c": 600, "d": 1024})
    assert "dark" in flags["b"] and "near-duplicate" in flags["b"] and "near-duplicate" in flags["a"]
    assert "blurry" in flags["c"] and "small" in flags["c"]
    assert "bars" in flags["d"] and "greyscale" in flags["d"]
