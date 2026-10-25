"""paths: the ZenTrainer folder setting moves datasets and results out of ComfyUI's input/output,
the datasets setting gives datasets a folder of their own, and empty ones leave every path
exactly where it was."""

import sys
import types

import pytest
from zentrainer import paths


@pytest.fixture
def comfy(tmp_path, monkeypatch):
    fp = types.ModuleType("folder_paths")
    fp.get_input_directory = lambda: str(tmp_path / "comfy" / "input")
    fp.get_output_directory = lambda: str(tmp_path / "comfy" / "output")
    fp.get_temp_directory = lambda: str(tmp_path / "comfy" / "temp")
    fp.get_user_directory = lambda: str(tmp_path / "comfy" / "user")
    fp.get_folder_paths = lambda _kind: []
    fp.get_full_path = lambda _kind, _name: None
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    paths.reset()
    yield tmp_path / "comfy"
    paths.reset()


def _set(folder: str) -> None:
    paths.save_settings({"folder": folder})
    paths.reset()  # what a restart does


def test_default_roots_are_comfyuis_folders(comfy):
    assert paths.folder() is None
    assert paths.datasets_root() == comfy / "input"
    assert paths.output_root() == comfy / "output" / "zentrainer"
    assert paths.lora_dir() == comfy / "output" / "zentrainer" / "loras"
    assert paths.output_roots() == [comfy / "output" / "zentrainer"]
    from zentrainer.datasets import folders
    from zentrainer.harness.trainer import lora_root

    (comfy / "input").mkdir(parents=True)
    assert folders.root() == (comfy / "input").resolve()
    assert lora_root(None) == (paths.lora_dir(), paths.lora_dir())


def test_a_set_folder_moves_datasets_and_outputs(comfy, tmp_path):
    d = tmp_path / "zt"
    _set(str(d))
    assert paths.folder() == d
    assert paths.datasets_root() == d / "datasets"
    assert paths.output_root() == d / "output"
    assert paths.lora_dir() == d / "output" / "loras"
    # older runs' results stay findable where they were written
    assert paths.output_roots() == [d / "output", comfy / "output" / "zentrainer"]
    paths.ensure()
    assert (d / "datasets").is_dir() and (d / "output" / "loras").is_dir()


def test_the_folder_is_read_once_until_a_restart(comfy, tmp_path):
    assert paths.folder() is None
    paths.save_settings({"folder": str(tmp_path / "zt")})
    assert paths.folder() is None and paths.describe()["restart_needed"]
    paths.reset()
    assert paths.folder() == tmp_path / "zt" and not paths.describe()["restart_needed"]


def test_settings_round_trip_and_validation(comfy, tmp_path):
    face = {"face_model": "buffalo_l", "datasets": ""}  # the other settings, at their defaults
    assert paths.read_settings() == {"folder": "", **face}
    assert paths.save_settings({"folder": f"  {tmp_path / 'zt'}  "}) == {"folder": str(tmp_path / "zt"), **face}
    assert paths.read_settings() == {"folder": str(tmp_path / "zt"), **face}
    assert paths.save_settings({"folder": ""}) == {"folder": "", **face}
    (tmp_path / "file").write_text("x")
    for bad in ("relative/dir", str(tmp_path / "file"), str(comfy / "output"), str(comfy / "input" / "zt"), str(comfy)):
        with pytest.raises(paths.SettingsError):
            paths.validate_folder(bad)
    assert paths.read_settings() == {"folder": "", **face}  # a refused one isn't saved


def test_view_addresses_resolve_and_refuse_escapes(comfy, tmp_path):
    for f in (comfy / "input" / "cats" / "a.png", comfy / "output" / "zentrainer" / "samples" / "r1" / "s.png",
              comfy / "output" / "other.png", tmp_path / "secret.txt"):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
    assert paths.resolve_view("input", "cats", "a.png") == comfy / "input" / "cats" / "a.png"
    assert paths.resolve_view("output", "zentrainer/samples/r1", "s.png") == comfy / "output" / "zentrainer" / "samples" / "r1" / "s.png"
    assert paths.resolve_view("output", "", "other.png") == comfy / "output" / "other.png"
    assert paths.resolve_view("output", "", "missing.png") is None
    for kind, sub, name in (("input", "../..", "secret.txt"), ("output", "zentrainer/../../..", "secret.txt"),
                            ("input", "cats", "../a.png"), ("input", "", ".."), ("bogus", "", "a.png")):
        with pytest.raises(ValueError):
            paths.resolve_view(kind, sub, name)
    with pytest.raises(ValueError):
        paths.inside(comfy / "input", "../user/x")


def test_view_addresses_follow_the_folder(comfy, tmp_path):
    d = tmp_path / "zt"
    _set(str(d))
    new = d / "output" / "samples" / "r2" / "s.png"
    old = comfy / "output" / "zentrainer" / "samples" / "r1" / "s.png"
    pic = d / "datasets" / "cats" / "a.png"
    for f in (new, old, pic):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
    # sample rows keep saying zentrainer/samples/<run>: new runs land in the folder, old ones stay put
    assert paths.resolve_view("output", "zentrainer/samples/r2", "s.png") == new
    assert paths.resolve_view("output", "zentrainer/samples/r1", "s.png") == old
    assert paths.resolve_view("input", "cats", "a.png") == pic
    with pytest.raises(ValueError):
        paths.resolve_view("output", "zentrainer/..", "secret.txt")
    from zentrainer.datasets import folders

    assert folders.folder("cats") == (d / "datasets" / "cats").resolve()


def test_datasets_get_a_folder_of_their_own_and_old_ones_stay_findable(comfy, tmp_path):
    (comfy / "input" / "cats").mkdir(parents=True)  # a dataset from before, straight in input/
    (comfy / "input" / "cats" / "a.png").write_bytes(b"")
    (comfy / "input" / "clipspace").mkdir()  # ComfyUI's own folder, no pictures: not a dataset
    paths.ensure()  # at start: the settings this process runs with are read
    assert paths.save_settings({"datasets": " datasets/ "})["datasets"] == "datasets"
    assert paths.datasets_root() == comfy / "input"  # read once: in use from the next start
    info = paths.describe()
    assert info["restart_needed"] and info["datasets_elsewhere"] == [str(comfy / "input" / "cats")]
    paths.reset()
    assert paths.datasets_root() == comfy / "input" / "datasets" and (comfy / "input" / "datasets").is_dir()
    from zentrainer.datasets import folders

    assert folders.root() == (comfy / "input" / "datasets").resolve()  # the list shows only the new folder
    assert paths.find_dataset("cats") == comfy / "input" / "cats"  # but a run of "cats" still finds it
    (comfy / "input" / "datasets" / "dogs").mkdir()
    assert paths.find_dataset("dogs") == comfy / "input" / "datasets" / "dogs"
    assert paths.find_dataset("gone") == comfy / "input" / "datasets" / "gone"
    _set_ds = paths.save_settings({"datasets": str(tmp_path / "data")})["datasets"]
    paths.reset()
    assert _set_ds == str(tmp_path / "data") and paths.datasets_root() == tmp_path / "data"
    (tmp_path / "file").write_text("x")
    for bad in ("../outside", str(tmp_path / "file")):
        with pytest.raises(paths.SettingsError):
            paths.validate_datasets(bad)
    assert paths.validate_datasets(str(comfy / "input")) == ""  # ComfyUI's input itself: the default
