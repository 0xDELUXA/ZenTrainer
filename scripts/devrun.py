"""Run a TrainSpec through the harness directly, without the ComfyUI server.

    python scripts/devrun.py spec.json --comfy /path/to/ComfyUI [--base DIR] [--cpu]

Uses the given ComfyUI as a library (its extra_model_paths.yaml included), prints
events to stdout. For debugging the harness; the app goes through the server.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent / "engine"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--base", help="base dir for input/output/user (default: ComfyUI's)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--anomaly", action="store_true", help="torch autograd anomaly detection")
    a = ap.parse_args()
    a.spec, a.comfy = str(Path(a.spec).resolve()), str(Path(a.comfy).resolve())  # before the chdir below

    sys.argv = [sys.argv[0]] + (["--cpu"] if a.cpu else [])
    sys.path.insert(0, a.comfy)
    sys.path.insert(0, str(ENGINE))
    os.chdir(a.comfy)

    import comfy.options

    comfy.options.enable_args_parsing()
    import folder_paths
    import utils.extra_config

    yamls = [Path(a.comfy) / "extra_model_paths.yaml"]
    local_cfg = ENGINE.parent / ".zentrainer" / "local.json"  # written by `zt setup --models-from`
    if local_cfg.is_file():
        yamls += [Path(p) for p in json.loads(local_cfg.read_text()).get("extra_model_paths", [])]
    for yaml in yamls:
        if yaml.is_file():
            utils.extra_config.load_extra_path_config(str(yaml))
    if a.base:
        b = Path(a.base)
        folder_paths.set_input_directory(str(b / "input"))
        folder_paths.set_output_directory(str(b / "output"))
        folder_paths.set_user_directory(str(b / "user"))

    import torch

    from zentrainer.harness.trainer import Trainer
    from zentrainer.spec import TrainSpec

    raw = json.loads(Path(a.spec).read_text())
    spec = TrainSpec.from_dict(raw.get("spec", raw))

    def emit(event, data):
        if event != "zenkit.job":
            print(event, json.dumps(data, default=str), flush=True)

    torch.autograd.set_detect_anomaly(a.anomaly)
    status = Trainer(spec, "devrun", emit=emit).run()
    print("status:", status)


if __name__ == "__main__":
    main()
