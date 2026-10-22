"""Control sets: edit training's input pictures as named layers beside a dataset."""

import io

import pytest
from PIL import Image

from zentrainer.datasets import folders as datasets
from zentrainer.harness.data import pick_controls, scan
from zentrainer.spec import SpecError, TrainSpec


def _png(color="red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def ds(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "root", lambda: tmp_path.resolve())
    datasets.create("chars")
    d = tmp_path / "chars"
    for name in ("a.png", "b.png", "a_control.png"):  # a has an old-style input beside it
        (d / name).write_bytes(_png())
    return d


def _items(d):
    return {it.path.name: it for it in scan(d, warn=False)}


def test_sets_are_read_beside_the_pictures(ds):
    for rel in ("_controls/faceswap/a.png", "_controls/faceswap/a.2.jpg", "_controls/faceswap/b.png", "_controls/depth/b.png"):
        (ds / rel).parent.mkdir(parents=True, exist_ok=True)
        (ds / rel).write_bytes(_png())
    items = _items(ds)
    assert set(items) == {"a.png", "b.png"}  # inputs never train as pictures of their own
    a, b = items["a.png"], items["b.png"]
    assert {k: [p.name for p in v] for k, v in a.control_sets.items()} == {"control": ["a_control.png"], "faceswap": ["a.png", "a.2.jpg"]}
    assert set(b.control_sets) == {"faceswap", "depth"}
    # a run's inputs: each chosen set's, in the sets' order
    assert [p.parent.name + "/" + p.name for p in pick_controls(b, ["depth", "faceswap"])] == ["depth/b.png", "faceswap/b.png"]
    assert pick_controls(a, ["depth"]) == []


def test_uploads_land_by_name_and_replace(ds):
    assert datasets.save_control("chars", "swap", "a.jpg", _png()) == "_controls/swap/a.jpg"
    assert datasets.save_control("chars", "swap", "a.2.png", _png()) == "_controls/swap/a.2.png"
    # the same input in another format replaces it rather than sitting beside it
    assert datasets.save_control("chars", "swap", "a.png", _png("blue")) == "_controls/swap/a.png"
    assert sorted(p.name for p in (ds / "_controls/swap").iterdir()) == ["a.2.png", "a.png"]
    with pytest.raises(datasets.DatasetError, match="no picture called"):
        datasets.save_control("chars", "swap", "zzz.png", _png())
    with pytest.raises(datasets.DatasetError, match="bad control set"):
        datasets.save_control("chars", "Bad Name", "a.png", _png())


def test_rename_moves_old_style_inputs_into_the_folder_layout(ds):
    assert datasets.rename_control_set("chars", "control", "faceswap") == 1
    assert not (ds / "a_control.png").exists()
    assert [p.name for p in _items(ds)["a.png"].control_sets["faceswap"]] == ["a.png"]
    datasets.create_control_set("chars", "depth")
    with pytest.raises(datasets.DatasetError, match="already"):
        datasets.rename_control_set("chars", "faceswap", "depth")
    assert datasets.items("chars")["control_sets"] == ["depth", "faceswap"]  # the empty one shows too


def test_deleting_a_picture_or_a_set_takes_its_inputs(ds):
    datasets.save_control("chars", "swap", "b.png", _png())
    datasets.delete("chars", "a.png")
    assert not (ds / "a_control.png").exists()
    assert datasets.delete_control_set("chars", "swap") == 1
    assert not (ds / "_controls/swap").exists()


def test_specs_from_before_control_sets_still_train_their_control_files():
    base = {"name": "x", "dataset": {"folder": "chars"}, "model": {"ckpt_name": "m.safetensors"}}
    old = TrainSpec.from_dict({**base, "dataset": {"folder": "chars", "edit": True}})
    assert old.dataset.control_sets == ["control"]
    new = TrainSpec.from_dict({**base, "dataset": {"folder": "chars", "control_sets": ["faceswap"]}})
    assert new.dataset.edit
    assert not TrainSpec.from_dict(base).dataset.edit
    with pytest.raises(SpecError):
        TrainSpec.from_dict({**base, "dataset": {"folder": "chars", "control_sets": ["Face Swap"]}})


# ---- channel inputs: the widened input layer (harness/wide_input.py) -----------------------------

def _moss_like(channels=4, dim=6):
    import torch

    dm = torch.nn.Module()
    dm.latent_channels = channels
    dm.video_embedder = torch.nn.Linear(channels, dim)
    model = torch.nn.Module()
    model.diffusion_model = dm
    return model


def test_a_widened_input_starts_as_the_base_layer_and_saves_whole():
    import torch

    from zentrainer.harness.wide_input import WideInput, input_layer

    model = _moss_like()
    name, layer = input_layer(model)
    assert name == "diffusion_model.video_embedder"
    wide = WideInput(layer, 2, torch.float32, "cpu")
    w, b = wide.weight(layer.weight.detach()), wide.bias(layer.bias.detach())
    assert w.shape == (6, 12)  # the noisy latent's 4 channels + 2 inputs of 4
    x = torch.randn(3, 4)
    extra = torch.randn(3, 8)
    # new columns at zero: the inputs change nothing until trained
    assert torch.allclose(torch.nn.functional.linear(torch.cat([x, extra], 1), w, b), layer(x), atol=1e-6)

    with torch.no_grad():
        wide.control.normal_()
        wide.delta.fill_(0.1)
        wide.bias_delta.fill_(-0.2)
    full_w, full_b = wide.full(layer.weight.detach(), layer.bias.detach())
    assert torch.allclose(full_w[:, :4], layer.weight + 0.1) and torch.allclose(full_w[:, 4:], wide.control)
    again = WideInput(layer, 2, torch.float32, "cpu")
    again.load(full_w, full_b, layer.weight.detach(), layer.bias.detach())
    for a, b2 in ((again.delta, wide.delta), (again.control, wide.control), (again.bias_delta, wide.bias_delta)):
        assert torch.allclose(a, b2, atol=1e-6)
    with pytest.raises(ValueError, match="input"):
        WideInput(layer, 1, torch.float32, "cpu").load(full_w, full_b, layer.weight.detach(), layer.bias.detach())


def test_only_models_with_a_latent_input_layer_widen():
    import torch

    from zentrainer.harness.wide_input import input_layer

    plain = torch.nn.Module()
    plain.diffusion_model = torch.nn.Linear(4, 4)
    assert input_layer(plain) is None
    odd = _moss_like()
    odd.diffusion_model.latent_channels = 8  # the layer isn't over the latent's channels
    assert input_layer(odd) is None


def test_an_auto_lr_floor_scales_with_a_groups_lr_multiple():
    import torch

    from zentrainer.harness import optim
    from zentrainer.spec import OptimSpec

    a, b = torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(4))
    spec = OptimSpec(optimizer="Automagic3", learning_rate=1e-4, lr_floor=0.5)
    opt, _ = optim.build([{"params": [a], "lr": 1e-4}, {"params": [b], "lr": 1e-3, "lr_mult": 10.0}], spec)
    assert [g["min_lr"] for g in opt.param_groups] == pytest.approx([5e-5, 5e-4])


def test_automagic3_cant_climb_past_a_few_times_its_start():
    import torch

    from zentrainer.harness import optim
    from zentrainer.spec import OptimSpec

    a, b = torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(4))
    spec = OptimSpec(optimizer="Automagic3", learning_rate=1e-4)
    opt, _ = optim.build([{"params": [a], "lr": 1e-4}, {"params": [b], "lr": 1e-3, "lr_mult": 10.0}], spec)
    ceil = optim.AUTOMAGIC3_CEILING
    assert [g["max_lr"] for g in opt.param_groups] == pytest.approx([1e-4 * ceil, 1e-3 * ceil])
    for _ in range(200):  # a perfectly steady gradient: every vote says "faster"
        a.grad, b.grad = torch.ones(4), torch.ones(4)
        opt.step()
    assert max(optim.current_lr(opt), *opt.get_learning_rates()) <= 1e-3 * ceil * 1.0001
