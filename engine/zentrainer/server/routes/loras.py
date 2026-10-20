"""The LoRA library behind zenkit-ui's LoRA picker (previews, info, favorites), and merging LoRAs."""

from __future__ import annotations

import asyncio
from pathlib import Path

from aiohttp import web

from .common import _bad


def register(server) -> None:
    routes = server.routes

    # ---- the LoRA library behind zenkit-ui's LoRA picker ----------------------------------------
    def _lora_path(name: str) -> str | None:
        import folder_paths

        return folder_paths.get_full_path("loras", name) if name else None

    def _lora_rows() -> list[dict]:
        import os

        import folder_paths

        from ...loras import favorites, meta

        names = folder_paths.get_filename_list("loras")
        favs = set(favorites.get_favorites())
        dirs: dict[str, set[str]] = {}
        rows = []
        for n in names:
            path = _lora_path(n)
            folder = os.path.dirname(path) if path else ""
            if path and folder not in dirs:
                try:
                    dirs[folder] = set(os.listdir(folder))
                except OSError:
                    dirs[folder] = set()
            found = meta.find_preview(path, dirs.get(folder, set())) if path else None
            try:
                st = os.stat(path) if path else None
            except OSError:
                st = None
            name = n.replace("\\", "/")
            rows.append({"name": name, "has_preview": found is not None, "preview": found[1] if found else None,
                         "favorite": name in favs, "size": st.st_size if st else None,
                         "mtime": int(st.st_mtime) if st else None})
        return rows

    @routes.get("/zentrainer/api/loras")
    async def list_loras(_request):
        return web.json_response({"loras": await asyncio.to_thread(_lora_rows)})

    @routes.get("/zentrainer/api/lora/preview")
    async def lora_preview(request):
        """The preview as a still; raw=1 a video preview as itself; size=<px> a cached thumbnail."""
        from ...loras import meta

        found = meta.find_preview(_lora_path(request.query.get("name", "")))
        if not found:
            return web.Response(status=404)
        path, kind = found
        raw = request.query.get("raw") == "1"
        if kind == "video" and not raw:
            path = await asyncio.to_thread(meta.poster, path)
            if not path:
                return web.Response(status=404)
        try:
            size = int(request.query.get("size", "0"))
        except ValueError:
            size = 0
        if size > 0 and not raw and not path.lower().endswith(".gif"):
            path = await asyncio.to_thread(meta.thumb, path, size) or path
        return web.FileResponse(path)

    @routes.get("/zentrainer/api/lora/info")
    async def lora_info(request):
        from urllib.parse import quote

        from ...loras import favorites, meta

        name = request.query.get("name", "").replace("\\", "/")
        path = _lora_path(name)
        if not path:
            return _bad("not found", 404)
        info = await asyncio.to_thread(meta.read_info, path)
        q = "?name=" + quote(name)
        for img in info.get("images", []):  # downloaded examples served from disk; the URL stays as a fallback
            file = img.pop("file", None)
            img["remote"] = img["url"] or None
            if file:
                img["url"] = "/zentrainer/api/lora/example" + q + "&file=" + quote(file)
        found = meta.find_preview(path)
        info.update(name=name, preview=found[1] if found else None, favorite=name in favorites.get_favorites())
        return web.json_response(info)

    @routes.get("/zentrainer/api/lora/example")
    async def lora_example(request):
        from ...loras import lora_manager, meta

        path = _lora_path(request.query.get("name", ""))
        sha = await asyncio.to_thread(meta.sha256_of, path) if path else None
        file = await asyncio.to_thread(lora_manager.resolve, sha, request.query.get("file", ""))
        return web.FileResponse(file) if file else web.Response(status=404)

    @routes.get("/zentrainer/api/lora/favorites")
    async def lora_favorites(_request):
        from ...loras import favorites

        return web.json_response({"loras": favorites.get_favorites()})

    @routes.post("/zentrainer/api/lora/favorites")
    async def lora_set_favorite(request):
        from ...loras import favorites

        body = await request.json()
        name = str(body.get("name", "")).replace("\\", "/")
        if not name:
            return _bad("missing name")
        return web.json_response({"loras": favorites.toggle(name, bool(body.get("pinned", True)))})

    @routes.post("/zentrainer/api/loras/merge")
    async def merge_loras(request):
        """Merge LoRAs: {members: [{name, strength}], out_name, max_rank?, energy?}. Saves to
        output/zentrainer/loras/merged/ (a loras folder), so it loads at once."""
        import re as _re

        import folder_paths

        from ...loras import merge

        b = await request.json()
        try:
            members = [(Path(folder_paths.get_full_path_or_raise("loras", str(m["name"]))), float(m.get("strength", 1.0)))
                       for m in b.get("members", [])]
        except Exception as e:
            return _bad(f"can't find that LoRA: {e}")
        if not members:
            return _bad("add at least one LoRA")
        name = _re.sub(r"[^A-Za-z0-9_.-]+", "_", str(b.get("out_name") or "merged")).strip("._") or "merged"
        out = Path(folder_paths.get_output_directory()) / "zentrainer" / "loras" / "merged" / f"{name}.safetensors"
        if out.exists() and not b.get("overwrite"):
            return _bad(f"merged/{out.name} already exists", 409)
        try:
            stats = await asyncio.to_thread(merge.merge_files, members, out, int(b.get("max_rank") or 0),
                                            float(b.get("energy") or 0.999), overwrite_ok=bool(b.get("overwrite")))
        except Exception as e:
            return _bad(f"merge failed: {e}")
        return web.json_response({"lora_name": f"merged/{out.name}", **stats})
