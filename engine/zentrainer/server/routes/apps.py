"""Apps: the catalog, running one (a batch), the gallery of results, App-mode workflows to import."""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

import aiohttp
from aiohttp import web

from .common import _bad, _loopback


def register(server) -> None:
    routes = server.routes

    # ---- apps ----------------------------------------------------------------------------
    def _apps_user_root() -> Path:
        import folder_paths

        return Path(folder_paths.get_user_directory()) / "zentrainer" / "apps"

    def _batches():
        import folder_paths

        from ...apps.runtime import BatchStore

        return BatchStore(Path(folder_paths.get_user_directory()) / "zentrainer" / "app_batches")

    @routes.get("/zentrainer/api/apps")
    async def list_apps(_request):
        from ...apps import manifest

        apps, errors = manifest.list_apps(_apps_user_root())
        return web.json_response({"apps": [a.public() for a in apps], "errors": errors})

    @routes.get("/zentrainer/api/apps/{id}/defaults")
    async def app_defaults(request):
        """The selected workflow's own values ({input: value}) and which inputs it takes, so a
        form starts where the workflow does and hides what it doesn't have. ?family= picks the
        workflow of a per-model app."""
        from ...apps import compile as appcompile
        from ...apps import manifest

        apps, _ = manifest.list_apps(_apps_user_root())
        app = next((a for a in apps if a.id == request.match_info["id"]), None)
        if app is None:
            return _bad("no such app", 404)
        family = request.query.get("family") or None
        try:
            graph = manifest.pick_workflow(app, None, family) if family else app.workflows[0][1]
            values = appcompile.defaults_for(app, family)
        except manifest.AppError as e:
            return _bad(str(e))
        return web.json_response({"values": values, "inputs": app.bound_in(graph)})

    @routes.get("/zentrainer/api/apps/{id}/cover")
    async def app_cover(request):
        from ...apps import manifest

        apps, _ = manifest.list_apps(_apps_user_root())
        app = next((a for a in apps if a.id == request.match_info["id"]), None)
        cover = app and app.manifest.get("cover")
        path = (app.root / cover).resolve() if cover else None
        if not path or not path.is_relative_to(app.root.resolve()) or not path.is_file():
            return _bad("no cover", 404)
        return web.FileResponse(path)

    @routes.post("/zentrainer/api/apps/{id}/run")
    async def run_app(request):
        """{"inputs": {...}, "sweep": {input: [values]}, "client_id"}: one prompt per cell."""
        from ...apps import compile as appcompile
        from ...apps import manifest

        apps, _ = manifest.list_apps(_apps_user_root())
        app = next((a for a in apps if a.id == request.match_info["id"]), None)
        if app is None:
            return _bad("no such app", 404)
        b = await request.json()
        store = _batches()
        batch_id = store.new_id()
        try:
            cells = appcompile.compile_batch(app, b.get("inputs") or {}, b.get("sweep") or {},
                                             output_prefix=f"zentrainer/apps/{app.id}/{batch_id}")
        except manifest.AppError as e:
            return _bad(str(e))
        submitted = []
        async with aiohttp.ClientSession() as s:
            for cell in cells:
                pid = str(uuid.uuid4())
                # preview_method: ComfyUI streams sampling previews for this prompt (to client_id)
                payload = {"prompt": cell["prompt"], "prompt_id": pid,
                           "extra_data": {"zentrainer_app": app.id, "preview_method": "auto"}}
                if b.get("client_id"):
                    payload["client_id"] = b["client_id"]
                async with s.post(_loopback(server) + "/prompt", json=payload) as r:
                    result = await r.json()
                if r.status != 200:
                    return _bad("ComfyUI rejected the workflow", 422, comfy=result, submitted=len(submitted))
                submitted.append({"coords": cell["coords"], "prompt_id": pid, "save": cell["save"]})
        batch = {"batch_id": batch_id, "app": app.id, "inputs": b.get("inputs") or {}, "sweep": b.get("sweep") or {},
                 "created": time.time(), "cells": submitted}
        store.save(batch)
        return web.json_response({"batch": batch})

    async def _resolve_batch(batch: dict) -> dict:
        from ...apps.runtime import resolve

        history = {}
        queued: set[str] | None = None
        pending = [c for c in batch["cells"] if c.get("status") not in ("done", "error")]
        if pending:
            async with aiohttp.ClientSession() as s:
                for c in pending:
                    async with s.get(_loopback(server) + f"/history/{c['prompt_id']}") as r:
                        if r.status == 200:
                            history.update(await r.json())
                if any(c["prompt_id"] not in history for c in pending):
                    async with s.get(_loopback(server) + "/queue") as r:
                        q = await r.json() if r.status == 200 else {}
                    queued = {str(e[1]) for k in ("queue_running", "queue_pending") for e in q.get(k, []) if len(e) > 1}
        resolved = resolve(batch, history, queued)
        if any(c["status"] in ("done", "error") for c in resolved["cells"]) and pending:
            # persist finished cells: ComfyUI's history is in memory only
            _batches().save({k: v for k, v in resolved.items() if k not in ("done", "total", "status")})
        return resolved

    @routes.get("/zentrainer/api/gallery")
    async def gallery(request):
        """Every image the apps made, newest first: [{batch_id, prompt_id, app, created, image,
        values}] where values are what made it (the run's inputs plus that cell's sweep values).
        ?app= narrows to one app; ?limit/offset page through."""
        app_id = request.query.get("app") or None
        limit = max(1, min(int(request.query.get("limit", 200)), 1000))
        offset = max(0, int(request.query.get("offset", 0)))
        items = []
        for b in _batches().list(app_id, limit=10_000):
            if any(c.get("status") not in ("done", "error") for c in b["cells"]):
                b = await _resolve_batch(b)
            for c in b["cells"]:
                for img in c.get("images") or []:
                    items.append({"batch_id": b["batch_id"], "prompt_id": c["prompt_id"], "app": b["app"],
                                  "created": b["created"], "image": img,
                                  "values": {**(b.get("inputs") or {}), **(c.get("coords") or {})}})
        return web.json_response({"items": items[offset:offset + limit], "total": len(items)})

    @routes.delete("/zentrainer/api/gallery/{batch_id}/{prompt_id}")
    async def gallery_delete(request):
        """Delete one generation's image files and drop it from its batch."""
        import folder_paths

        store = _batches()
        try:
            batch = store.get(request.match_info["batch_id"])
        except (FileNotFoundError, ValueError):
            return _bad("no such batch", 404)
        out_root = (Path(folder_paths.get_output_directory()) / "zentrainer" / "apps").resolve()
        cells, removed = [], 0
        for c in batch["cells"]:
            if c["prompt_id"] != request.match_info["prompt_id"]:
                cells.append(c)
                continue
            for img in c.get("images") or []:
                path = (Path(folder_paths.get_output_directory()) / img.get("subfolder", "") / img["filename"]).resolve()
                if path.is_relative_to(out_root) and path.is_file():  # only app outputs, never anything else
                    path.unlink()
                    removed += 1
        if len(cells) == len(batch["cells"]):
            return _bad("no such image", 404)
        store.save({**batch, "cells": cells})
        return web.json_response({"removed": removed})

    @routes.get("/zentrainer/api/apps/batches/{batch_id}")
    async def get_batch(request):
        try:
            batch = _batches().get(request.match_info["batch_id"])
        except (FileNotFoundError, ValueError):
            return _bad("no such batch", 404)
        return web.json_response({"batch": await _resolve_batch(batch)})

    @routes.get("/zentrainer/api/apps/{id}/batches")
    async def app_batches(request):
        rows = _batches().list(request.match_info["id"], limit=int(request.query.get("limit", 12)))
        return web.json_response({"batches": [await _resolve_batch(b) for b in rows]})

    # ---- ComfyUI workflows in App mode (templates, the user's): what the importer reads -----------
    @routes.get("/zentrainer/api/wapps")
    async def list_wapps(_request):
        from ...apps import catalog

        return web.json_response({"apps": await asyncio.to_thread(catalog.list_apps)})

    @routes.get("/zentrainer/api/wapps/describe")
    async def describe_wapp(request):
        from ...apps import catalog
        from ...apps.workflow import WorkflowError

        try:
            d = await asyncio.to_thread(catalog.describe, request.query.get("id", ""))
        except catalog.AppNotFound as e:
            return _bad(f"no such app: {e}", 404)
        except WorkflowError as e:
            return _bad(str(e))
        return web.json_response(d)
