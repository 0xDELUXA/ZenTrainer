"""The standalone web app (web/dist, built by `pnpm build`); its catch-all route registers last."""

from __future__ import annotations


from aiohttp import web

from .common import WEB_DIST


def register(server) -> None:
    routes = server.routes

    # The standalone app. Built by `pnpm build` in web/; dev uses Vite's server instead.
    @routes.get("/zentrainer")
    async def app_redirect(_request):
        raise web.HTTPFound("/zentrainer/")

    @routes.get("/zentrainer/{path:.*}")
    async def app(request):
        rel = request.match_info["path"] or "index.html"
        target = (WEB_DIST / rel).resolve()
        if WEB_DIST not in target.parents or not target.is_file():
            target = WEB_DIST / "index.html"  # SPA fallback
        if not target.is_file():
            return web.Response(text="ZenTrainer web app not built: run `pnpm build` in web/", status=503)
        return web.FileResponse(target)


_client_id: str | None = None
