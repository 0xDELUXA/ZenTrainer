"""The standalone web app (web/dist, built by `pnpm build`); its catch-all route registers last.

A Registry/Manager install ships web/dist. A git clone into custom_nodes doesn't (it's built
output), so the first visit fetches the prebuilt app CI keeps on the web-dist branch, with the
clone's own git: nothing to install. Also after a `git pull` that moved web/ on, when the app here
was fetched (web/dist/.source names what it was built from). A build of your own (no .source) is
left alone. Same steps as `zt web` (cli/zt/web.py), which also builds with pnpm.
"""

from __future__ import annotations

import asyncio
import io
import logging
import shutil
import subprocess
import zipfile
from pathlib import Path

from aiohttp import web

from .common import WEB_DIST

BRANCH = "web-dist"
ROOT = WEB_DIST.parents[1]

_checked = False
_lock: asyncio.Lock | None = None
_why = ""  # why there's no app, for the page that says so


def _git(*args: str) -> str:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def _wanted() -> str | None:
    """The build this checkout wants (the last commit that changed web/), or None when there's
    nothing to fetch: the app is here and current, or it's a build of your own."""
    have = (WEB_DIST / "index.html").is_file()
    stamp = WEB_DIST / ".source"
    if have and not stamp.is_file():
        return None
    want = _git("log", "-1", "--format=%H", "--", "web")
    if not want or (have and stamp.read_text().strip() == want):
        return None
    return want


def _fetch(want: str) -> bool:
    """The web-dist branch's build into web/dist, when it's the one for `want`."""
    global _why
    fetch = ["git", "-C", str(ROOT), "fetch", "--quiet", "--depth", "1", "origin", BRANCH]
    try:
        ok = subprocess.run(fetch, capture_output=True, timeout=300).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    if not ok:
        _why = "couldn't fetch the prebuilt app (no network, or no git?)"
        return False
    if _git("show", "FETCH_HEAD:SOURCE") != want:
        _why = "there's no prebuilt app for this version yet (CI builds it a few minutes after a push)"
        return False
    data = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=zip", "FETCH_HEAD"],
                          capture_output=True, timeout=300).stdout
    new = WEB_DIST.parent / ".dist-new"
    if new.exists():
        shutil.rmtree(new)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for m in z.namelist():
            if m != "SOURCE" and ".." not in Path(m).parts:
                z.extract(m, new)
    if not (new / "index.html").is_file():
        shutil.rmtree(new)
        _why = "the prebuilt app was incomplete"
        return False
    (new / ".source").write_text(want + "\n")
    old = WEB_DIST.parent / ".dist-old"  # only ever a complete build replaces web/dist
    if old.exists():
        shutil.rmtree(old)
    if WEB_DIST.exists():
        WEB_DIST.rename(old)
    new.rename(WEB_DIST)
    if old.exists():
        shutil.rmtree(old)
    return True


def _ensure() -> None:
    want = _wanted()
    if want is None:
        return
    logging.info("[ZenTrainer] fetching the prebuilt web app for this version…")
    if _fetch(want):
        logging.info("[ZenTrainer] web app ready")
    else:
        logging.warning("[ZenTrainer] web app: %s", _why)


async def _ensure_once() -> None:
    global _checked, _lock
    if _checked:
        return
    _lock = _lock or asyncio.Lock()
    async with _lock:
        if not _checked:
            try:
                await asyncio.get_running_loop().run_in_executor(None, _ensure)
            except Exception:
                logging.exception("[ZenTrainer] web app fetch failed")
            _checked = True


def _missing_page() -> str:
    why = f" ({_why})" if _why else ""
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
