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
