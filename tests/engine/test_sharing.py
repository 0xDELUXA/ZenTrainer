"""Run settings as a file to share (zentrainer/sharing.py)."""

import pytest

from zentrainer import __version__, sharing
from zentrainer.spec import TrainSpec


def _spec(**over):
    s = TrainSpec(name="alice-v1").to_dict()
    s["model"].update(kind="split", unet_name="krea2_raw_int8.safetensors", clip_name="qwen3.safetensors",
                      clip_type="krea2", vae_name="flux2-vae.safetensors")
    s["dataset"]["folder"] = "my_private_photos"
    s["output"]["lora_dir"] = "/mnt/big/loras"
    s["network"]["init_from"] = "zentrainer/old/old.safetensors"
    s["samples"]["prompts"] = ["a photo of alice", {"prompt": "edit this", "control": "my_private_photos/1.png"}]
    s["optim"]["learning_rate"] = 3e-4
    for k, v in over.items():
        s[k] = v
    return s


def test_export_leaves_out_this_machines_things():
    doc = sharing.export(_spec())
    assert doc["format"] == sharing.FORMAT and doc["zentrainer"] == __version__ and doc["name"] == "alice-v1"
    s = doc["spec"]
    assert "folder" not in s["dataset"] and "lora_dir" not in s["output"] and "init_from" not in s["network"]
    assert "control" not in s["samples"]["prompts"][1]
    assert s["optim"]["learning_rate"] == 3e-4 and s["model"]["unet_name"] == "krea2_raw_int8.safetensors"


def test_round_trip_keeps_the_settings():
    got = sharing.read(sharing.export(_spec()))
    assert got["warnings"] == [] and got["name"] == "alice-v1"
    assert got["spec"]["optim"]["learning_rate"] == 3e-4
    assert {f["field"] for f in got["files"]} == {"model.unet_name", "model.clip_name", "model.vae_name"}
    assert got["local"] == ["the dataset", "where LoRAs are saved", "a LoRA to continue from", "the edit samples' pictures"]
    # it still makes a valid run once a dataset is chosen here
    spec = got["spec"]
    spec["dataset"]["folder"] = "mine"
    assert TrainSpec.from_dict(spec).optim.learning_rate == 3e-4


def test_a_bare_spec_loads_without_the_senders_folders():
    got = sharing.read(_spec())
    assert "folder" not in got["spec"]["dataset"] and "lora_dir" not in got["spec"]["output"]


def test_newer_files_load_what_this_version_knows():
    doc = sharing.export(_spec())
    doc["version"] = sharing.VERSION + 1
    doc["spec"]["optim"]["warp_drive"] = True
    doc["spec"]["hologram"] = {"on": True}
    got = sharing.read(doc)
    assert "warp_drive" not in got["spec"]["optim"] and "hologram" not in got["spec"]
    text = " ".join(got["warnings"])
    assert "newer ZenTrainer" in text and "optim.warp_drive" in text and "hologram" in text


@pytest.mark.parametrize("doc", [[1, 2], {"format": "kohya"}, {"hello": "world"}, {"format": sharing.FORMAT}])
def test_not_a_run_file(doc):
    with pytest.raises(sharing.ShareError):
        sharing.read(doc)


def test_sample_model_folder_follows_the_model_kind():
    s = _spec()
    s["samples"]["model_name"] = "krea2_turbo.safetensors"
    rows = {f["field"]: f for f in sharing.files(s)}
    assert rows["samples.model_name"]["folder"] == "diffusion_models"
    s["model"]["kind"] = "checkpoint"
    assert {f["field"]: f for f in sharing.files(s)}["samples.model_name"]["folder"] == "checkpoints"


def test_only_what_was_left_out_is_asked_for():
    s = TrainSpec(name="plain").to_dict()  # no folders, no LoRA to continue: only the dataset
    assert sharing.read(sharing.export(s))["local"] == ["the dataset"]
