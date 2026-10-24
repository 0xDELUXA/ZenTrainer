"""The face model is a setting: only the models on offer are accepted, and antelopev2's zip, which
unpacks one folder too deep, loads through a link in ZenTrainer's folder while ~/.insightface stays
exactly as it was. No InsightFace needed: its download helper is a stand-in."""

import sys
import types

import pytest
from zentrainer import paths
from zentrainer.datasets import faces


@pytest.fixture
def comfy(tmp_path, monkeypatch):
    fp = types.ModuleType("folder_paths")
    fp.get_input_directory = lambda: str(tmp_path / "comfy" / "input")
    fp.get_output_directory = lambda: str(tmp_path / "comfy" / "output")
    fp.get_user_directory = lambda: str(tmp_path / "comfy" / "user")
    fp.get_folder_paths = lambda _kind: []
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    paths.reset()
    yield tmp_path / "comfy"
    paths.reset()


def test_the_setting_takes_only_the_models_on_offer(comfy):
    assert faces.model() == faces.DEFAULT == "buffalo_l"
    paths.save_settings({"face_model": "antelopev2"})
    assert faces.model() == "antelopev2" and paths.read_settings()["folder"] == ""
    with pytest.raises(paths.SettingsError):
        paths.save_settings({"face_model": "something_else"})
    assert faces.model() == "antelopev2"
    assert {m["id"] for m in paths.describe()["face_models"]} == set(faces.MODELS)


@pytest.fixture
def insightface_home(tmp_path, monkeypatch):
    """A stand-in ~/.insightface, and InsightFace's download helper (it finds the files there)."""
    home = tmp_path / "insightface-home"
    monkeypatch.setattr(faces, "_home", lambda: home)
    storage = types.ModuleType("insightface.utils.storage")
    storage.ensure_available = lambda sub_dir, name, root: str(home / sub_dir / name)
    utils = types.ModuleType("insightface.utils")
    utils.storage = storage
    monkeypatch.setitem(sys.modules, "insightface", types.ModuleType("insightface"))
    monkeypatch.setitem(sys.modules, "insightface.utils", utils)
    monkeypatch.setitem(sys.modules, "insightface.utils.storage", storage)
    return home


def snapshot(root):
    return sorted((str(p.relative_to(root)), p.stat().st_size) for p in root.rglob("*"))


def test_a_model_laid_out_as_expected_loads_from_home(comfy, insightface_home):
    d = insightface_home / "models" / "buffalo_l"
    d.mkdir(parents=True)
    (d / "w600k_r50.onnx").write_bytes(b"x" * 10)
    assert faces.downloaded("buffalo_l") and not faces.downloaded("antelopev2")
    assert faces._root("buffalo_l") == str(insightface_home)


def test_a_nested_model_loads_through_a_link_and_home_is_untouched(comfy, insightface_home):
    inner = insightface_home / "models" / "antelopev2" / "antelopev2"
    inner.mkdir(parents=True)
    (inner / "glintr100.onnx").write_bytes(b"y" * 10)
    before = snapshot(insightface_home)
    root = faces._root("antelopev2")
    link = comfy / "user" / "zentrainer" / "insightface" / "models" / "antelopev2"
    assert root == str(comfy / "user" / "zentrainer" / "insightface")
    assert (link / "glintr100.onnx").read_bytes() == b"y" * 10
    assert faces._root("antelopev2") == root  # a second time reuses the link
    assert snapshot(insightface_home) == before
    assert faces.downloaded("antelopev2")
