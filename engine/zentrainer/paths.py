"""Where ZenTrainer keeps datasets and results.

By default in ComfyUI's own folders: datasets in its input dir, LoRAs / samples / merges under
output/zentrainer/. The "ZenTrainer folder" setting moves both out of them, to
<folder>/datasets/<dataset> and <folder>/output/<loras|samples|...>. It's read once per process,
so a change takes effect when ComfyUI restarts, never under a training run. Caches and run state
stay in ComfyUI's user dir either way.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from . import safeio

DEFAULT_SETTINGS = {"folder": ""}

_UNREAD = object()
_active = _UNREAD  # the folder this process runs with (None: ComfyUI's folders)


class SettingsError(ValueError):
    pass


def settings_file() -> Path:
    import folder_paths

    return Path(folder_paths.get_user_directory()) / "zentrainer" / "settings.json"


def read_settings() -> dict:
    try:
        saved = safeio.read_json(settings_file(), {})
    except OSError:
        saved = {}
    saved = saved if isinstance(saved, dict) else {}
    return {**DEFAULT_SETTINGS, **{k: v for k, v in saved.items() if k in DEFAULT_SETTINGS}}


def folder() -> Path | None:
    """The ZenTrainer folder this process uses, or None for ComfyUI's input/output layout."""
    global _active
    if _active is _UNREAD:
        try:
            raw = str(read_settings()["folder"] or "").strip()
        except (ImportError, AttributeError):  # outside ComfyUI (tests, tools): don't pin anything
            return None
        _active = Path(raw).expanduser() if raw and os.path.isabs(os.path.expanduser(raw)) else None
    return _active


def reset() -> None:
    """Forget the folder read at start (tests)."""
    global _active
    _active = _UNREAD


def datasets_root() -> Path:
    import folder_paths

    d = folder()
    return d / "datasets" if d else Path(folder_paths.get_input_directory())


def legacy_output_root() -> Path:
    """output/zentrainer in ComfyUI's output dir: where results went before a folder was set."""
    import folder_paths

    return Path(folder_paths.get_output_directory()) / "zentrainer"


def output_root() -> Path:
    """What output/zentrainer is by default: loras/, samples/, apps/ live under it."""
    d = folder()
    return d / "output" if d else legacy_output_root()


def output_roots() -> list[Path]:
    """Where ZenTrainer results can be, newest layout first: a folder set later leaves older runs'
    samples and LoRAs where they were written."""
    d = folder()
    return [output_root(), legacy_output_root()] if d else [output_root()]


def lora_dir() -> Path:
    return output_root() / "loras"


def ensure() -> None:
    """At startup: make the folder's dirs, so a first run or upload has somewhere to go."""
    if folder():
        for p in (datasets_root(), lora_dir()):
            p.mkdir(parents=True, exist_ok=True)


# ---- the setting ----------------------------------------------------------------------------------

def validate_folder(raw: str) -> str:
    """The folder as it will be saved ("" for ComfyUI's folders); SettingsError if it can't work."""
    import folder_paths

    raw = str(raw or "").strip()
    if not raw:
        return ""
    p = Path(os.path.normpath(os.path.expanduser(raw)))
    if not p.is_absolute():
        raise SettingsError("the ZenTrainer folder must be an absolute path")
    if p.exists() and not p.is_dir():
        raise SettingsError(f"{p} exists and isn't a folder")
    rp = p.resolve()
    for label, comfy in (("input", folder_paths.get_input_directory()), ("output", folder_paths.get_output_directory())):
        c = Path(comfy).resolve()
        # inside ComfyUI's input/output is the bloat this setting is for; and <folder>/output
        # being ComfyUI's own output dir would mix the two layouts
        if rp == c or c in rp.parents or c in (rp / "datasets", rp / "output"):
            raise SettingsError(f"pick a folder outside ComfyUI's {label} folder ({c})")
    try:
        p.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=p):
            pass
    except OSError as e:
        raise SettingsError(f"can't write to {p}: {e.strerror or e}") from None
    return str(p)


def save_settings(changes: dict) -> dict:
    cur = read_settings()
    if "folder" in changes:
        cur["folder"] = validate_folder(changes["folder"])
    f = settings_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    safeio.write_json(f, cur)
    return cur


def describe() -> dict:
    """The saved setting, what this process actually uses, and whether a restart is pending."""
    saved = read_settings()
    active = folder()
    return {"settings": saved, "active": {"folder": str(active) if active else "", "datasets": str(datasets_root()),
                                          "output": str(output_root()), "loras": str(lora_dir())},
            "restart_needed": saved["folder"] != (str(active) if active else "")}


# ---- files by ComfyUI /view address ----------------------------------------------------------------

def inside(root: Path, rel: str) -> Path:
    """root/rel, refused if it climbs out of root. Lexical (like ComfyUI's /view), so a symlinked
    folder inside the root still serves."""
    root = Path(os.path.abspath(root))
    p = Path(os.path.abspath(root / rel))
    if p != root and root not in p.parents:
        raise ValueError(f"path {rel!r} is outside its folder")
    return p


def resolve_view(kind: str, subfolder: str, filename: str) -> Path | None:
    """The file a ComfyUI-style address names ({filename, subfolder, type} — how sample rows,
    dataset items and app results carry their files), wherever it lives now: type=input subfolders
    start with the dataset, type=output "zentrainer/..." is under output_roots(). None if it
    isn't there; ValueError for an address that escapes its folder."""
    import folder_paths

    if not filename or filename in (".", "..") or "/" in filename or "\\" in filename:
        raise ValueError(f"bad file name {filename!r}")
    sub = (subfolder or "").replace("\\", "/").strip("/")
    if kind == "input":
        cands = [(datasets_root(), sub), (Path(folder_paths.get_input_directory()), sub)]
    elif kind == "output":
        cands = []
        if sub == "zentrainer" or sub.startswith("zentrainer/"):
            rest = sub[len("zentrainer"):].lstrip("/")
            cands = [(r, rest) for r in output_roots()]
        cands.append((Path(folder_paths.get_output_directory()), sub))
    elif kind == "temp":
        cands = [(Path(folder_paths.get_temp_directory()), sub)]
    else:
        raise ValueError(f"bad file type {kind!r}")
    for root, rel in cands:
        p = inside(root, os.path.join(rel, filename))
        if p.is_file():
            return p
    return None
