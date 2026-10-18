"""Theme packs for the app: ZenKit's JSON format, served by the engine itself.

Sources, later wins on an id clash:
  1. engine/themes             bundled (synced from ZenKit), so no dependency on it
  2. a ComfyUI-ZenKit install  its built-ins, if present in custom_nodes
  3. <user>/zenkit/themes      the drop-in folder ZenKit also reads: one place for both
"""

from __future__ import annotations

import json
import os
from pathlib import Path

BUNDLED = Path(__file__).resolve().parent.parent / "themes"


def _zenkit_dirs() -> list[Path]:
    import folder_paths

    out = []
    for base in folder_paths.get_folder_paths("custom_nodes"):
        try:
            entries = list(Path(base).iterdir())
        except OSError:
            continue
        for d in entries:
            if (d / "zenkit_themes_api.py").is_file() and (d / "themes").is_dir():
                out.append(d / "themes")
    return out


def theme_dirs() -> list[Path]:
    import folder_paths

    dirs = [BUNDLED, *_zenkit_dirs(), Path(folder_paths.get_user_directory()) / "zenkit" / "themes"]
    seen, out = set(), []
    for d in dirs:
        real = d.resolve()
        if real not in seen and real.is_dir():
            seen.add(real)
            out.append(real)
    return out


def _resolve_css(pack: dict, folder: Path) -> None:
    """A `css` of filename ref(s) becomes inlined CSS text; refs can't leave the folder."""
    css = pack.get("css")
    refs = css if isinstance(css, list) else [css] if isinstance(css, str) else []
    if not refs or any(isinstance(r, str) and "{" in r for r in refs):
        return
    parts = []
    for ref in refs:
        if isinstance(ref, str) and ref.strip():
            try:
                parts.append((folder / os.path.basename(ref)).read_text(encoding="utf-8"))
            except OSError:
                continue
    if parts:
        pack["css"] = "\n".join(parts)
    else:
        pack.pop("css", None)


def list_themes(dirs: list[Path] | None = None) -> list[dict]:
    by_id: dict[str, dict] = {}
    for base in dirs if dirs is not None else theme_dirs():
        for folder in sorted(p for p in base.iterdir() if p.is_dir()):
            try:
                pack = json.loads((folder / "theme.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(pack, dict) and isinstance(pack.get("id"), str):
                _resolve_css(pack, folder)
                by_id.pop(pack["id"], None)  # re-insert so order follows the winning source
                by_id[pack["id"]] = pack
    return sorted(by_id.values(), key=lambda p: str(p.get("name", p["id"])).lower())
