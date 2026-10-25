"""Where things are: the repository, its managed ComfyUI, comfy.lock and this machine's settings.

The other modules read these as `repo.ROOT`, `repo.COMFY`, ... at call time, so set_root() moves
them all (the tests point zt at a throwaway clone that way).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

WINDOWS = platform.system() == "Windows"

# folders an existing ComfyUI keeps under models/
MODEL_FOLDERS = ["checkpoints", "diffusion_models", "unet", "loras", "vae", "text_encoders", "clip",
                 "clip_vision", "controlnet", "embeddings", "upscale_models", "model_patches", "detection"]


def set_root(root: Path) -> None:
    global ROOT, COMFY, LOCK, LOCAL, RUNNING, ENGINE, WEB, DIST
    ROOT = Path(root).resolve()
    COMFY = ROOT / "comfyui"  # the managed ComfyUI, pinned by comfy.lock, own .venv (gitignored)
    LOCK = ROOT / "comfy.lock"
    LOCAL = ROOT / ".zentrainer" / "local.json"  # this machine's settings (gitignored)
    RUNNING = ROOT / ".zentrainer" / "running.json"  # the port a started ComfyUI is on
    ENGINE = ROOT / "engine"
    WEB = ROOT / "web"
    DIST = WEB / "dist"


# cli/zt/repo.py -> the repository; ZT_ROOT points it elsewhere
set_root(Path(os.environ.get("ZT_ROOT") or Path(__file__).resolve().parents[2]))


def lock() -> dict:
    return json.loads(LOCK.read_text())


def local() -> dict:
    return json.loads(LOCAL.read_text()) if LOCAL.is_file() else {}


def save_local(cfg: dict) -> None:
    LOCAL.parent.mkdir(exist_ok=True)
    LOCAL.write_text(json.dumps(cfg, indent=2) + "\n")


def venv_python() -> Path:
    return COMFY / ".venv" / ("Scripts/python.exe" if WINDOWS else "bin/python")


def git(*args: str, cwd: Path | None = None) -> str:
    """A git command's output, or "" when it fails."""
    out = subprocess.run(["git", "-C", str(cwd or ROOT), *args], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""
