"""The standalone web app (web/dist, built by `pnpm build`); its catch-all route registers last.

A Registry/Manager install ships web/dist. A git clone fetches it on the first visit
(server/webdist.py, which the Registry pack leaves out).
"""

from __future__ import annotations

import asyncio
import logging

from aiohttp import web

from .common import WEB_DIST

try:
    from .. import webdist
except ImportError:  # the Registry pack: the app came built
    webdist = None

ROOT = WEB_DIST.parents[1]

_checked = False
_lock: asyncio.Lock | None = None


async def _ensure_once() -> None:
    global _checked, _lock
    if _checked or webdist is None:
        return
    _lock = _lock or asyncio.Lock()
    async with _lock:
        if not _checked:
            try:
                await asyncio.get_running_loop().run_in_executor(None, webdist.ensure)
            except Exception:
                logging.exception("[ZenTrainer] web app fetch failed")
            _checked = True


def _missing_page() -> str:
    why = f" ({webdist.why})" if webdist and webdist.why else ""
    return (f"ZenTrainer's web app isn't here{why}.\n\n"
            f"From {ROOT}:\n"
            "  scripts/zt web        fetches the prebuilt app (or builds it, with pnpm)\n"
            "  cd web && pnpm install && pnpm build   builds it yourself\n\n"
            "or install ZenTrainer from ComfyUI Manager / the Registry, which includes it. Then reload this page.")


def register(server) -> None:
    routes = server.routes

    # The standalone app. Built by `pnpm build` in web/; dev uses Vite's server instead.
    @routes.get("/zentrainer")
    async def app_redirect(_request):
        raise web.HTTPFound("/zentrainer/")

    @routes.get("/zentrainer/{path:.*}")
    async def app(request):
        await _ensure_once()
        rel = request.match_info["path"] or "index.html"
        target = (WEB_DIST / rel).resolve()
        if WEB_DIST not in target.parents or not target.is_file():
            target = WEB_DIST / "index.html"  # SPA fallback
        if not target.is_file():
            return web.Response(text=_missing_page(), status=503)
        return web.FileResponse(target)


_client_id: str | None = None
