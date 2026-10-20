"""This process and machine: health, themes, memory telemetry, disk use, ComfyUI's version, restart,
rebuilding the web app."""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

from aiohttp import web

from ... import __version__, state
from .. import system
from ...harness import control
from .common import WEB_DIST, _bad, _capabilities, _code_stale, _comfy_version, _exec_self, _web_state, comfy_too_old


def register(server) -> None:
    routes = server.routes

    @routes.get("/zentrainer/api/health")
    async def health(_request):
        return web.json_response({"version": __version__, "capabilities": _capabilities(), "stale": _code_stale(),
                                  "comfy_old": comfy_too_old(), **_web_state()})

    @routes.get("/zentrainer/api/themes")
    async def themes(_request):
        from ..themes import list_themes

        return web.json_response({"themes": await asyncio.to_thread(list_themes)})

    @routes.get("/zentrainer/api/system/models")
    async def loaded_models(_request):
        """Each model ComfyUI holds: its size, how much is in VRAM, per block (see residency.py)."""
        from ...models import residency

        return web.json_response({"models": await asyncio.to_thread(residency.report)})

    @routes.get("/zentrainer/api/system")
    async def get_system(request):
        """Memory samples newer than ?since=<unix time> (all buffered ones by default)."""
        try:
            since = float(request.query.get("since", 0))
        except ValueError:
            since = 0.0
        return web.json_response({"interval": system.INTERVAL, "samples": system.history(since)})

    # ---- storage --------------------------------------------------------------------------------
    @routes.get("/zentrainer/api/storage")
    async def get_storage(request):
        """The full report; ?disks=1 for just free space (cheap: the sidebar polls it)."""
        from ...runs import disk as storage

        if request.query.get("disks") == "1":
            return web.json_response({"disks": await asyncio.to_thread(storage.disks)})
        return web.json_response(await asyncio.to_thread(storage.report, state.store()))

    @routes.get("/zentrainer/api/storage/places")
    async def storage_places(_request):
        """Where a run can save its LoRAs: ComfyUI's loras folders, with free space."""
        import folder_paths

        default = Path(folder_paths.get_output_directory()) / "zentrainer" / "loras"
        base = Path(folder_paths.base_path).resolve()
        out = []
        for p in folder_paths.get_folder_paths("loras"):
            rp = Path(p).resolve()
            try:
                u = shutil.disk_usage(rp)
            except OSError:
                continue
            out.append({"path": str(rp), "free": u.free, "total": u.total, "default": rp == default.resolve(),
                        "inside_install": rp.is_relative_to(base)})
        return web.json_response({"loras": out})

    @routes.post("/zentrainer/api/storage/clean")
    async def clean_storage(request):
        """{"kind": "states", "run_id"} | {"kind": "states_all"} | {"kind": "cache", "folder"}:
        returns {"freed": bytes} ("runs" too for states_all: how many it cleaned)."""
        from ...datasets import folders as datasets
        from ...runs import disk as storage

        b = await request.json()
        try:
            if b.get("kind") == "states":
                freed = await asyncio.to_thread(storage.clean_states, state.store(), str(b.get("run_id", "")))
            elif b.get("kind") == "states_all":
                freed, runs = await asyncio.to_thread(storage.clean_all_states, state.store())
                return web.json_response({"freed": freed, "runs": runs})
            elif b.get("kind") == "cache":
                freed = await asyncio.to_thread(storage.clean_cache, state.store(), str(b.get("folder", "")))
            else:
                return _bad("kind must be states or cache")
        except (storage.StorageError, datasets.DatasetError, ValueError) as e:
            return _bad(str(e))
        except FileNotFoundError:
            return _bad("not found", 404)
        return web.json_response({"freed": freed})

    @routes.get("/zentrainer/api/comfyui")
    async def comfyui_version(request):
        """This ComfyUI's version and commit, the pin in comfy.lock, and (with ?check=1)
        how far upstream has moved. The check fetches, so it only runs when asked."""
        return web.json_response(await asyncio.to_thread(_comfy_version, request.query.get("check") == "1"))

    building = asyncio.Lock()

    @routes.post("/zentrainer/api/web/build")
    async def build_web(_request):
        """Rebuild web/dist from the source on disk: after a pull, since dist isn't in git."""
        pnpm = shutil.which("pnpm")
        if not pnpm:
            return _bad("pnpm isn't installed here: run `pnpm install && pnpm build` in web/", 501)
        async with building:
            # CI: pnpm never stops to ask (it would wait on a prompt nobody sees)
            for args in (["install", "--frozen-lockfile"], ["build"]):
                proc = await asyncio.create_subprocess_exec(pnpm, *args, cwd=WEB_DIST.parent, env={**os.environ, "CI": "1"},
                                                            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                out, _ = await proc.communicate()
                if proc.returncode:
                    return _bad(f"pnpm {args[0]} failed:\n{out.decode(errors='replace')[-2000:]}", 500)
        return web.json_response({"ok": True, **_web_state()})

    @routes.post("/zentrainer/api/restart")
    async def restart(request):
        """Restart this ComfyUI process (same interpreter and arguments)."""
        body = await request.json() if request.can_read_body else {}
        if control.active() and not body.get("force"):
            return _bad("a run is training; stop it first (or pass force)", 409)
        asyncio.get_running_loop().call_later(0.5, _exec_self)
        return web.json_response({"ok": True})
