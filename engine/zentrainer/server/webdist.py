"""Fetching the prebuilt web app into a git clone (see routes/webapp.py).

A git clone into custom_nodes has no web/dist (it's built output), so the first visit fetches the
prebuilt app CI keeps on the web-dist branch, with the clone's own git: nothing to install. Also
after a `git pull` that moved web/ on, when the app here was fetched (web/dist/.source names what
it was built from). A build of your own (no .source) is left alone. Same steps as `zt web`
(cli/zt/web.py), which also builds with pnpm.

Only a clone has this file: the Registry pack ships the app built and leaves it out (.comfyignore).
"""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import zipfile
from pathlib import Path

from .routes.common import WEB_DIST

BRANCH = "web-dist"
ROOT = WEB_DIST.parents[1]

why = ""  # why there's no app, for the page that says so


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
    global why
    fetch = ["git", "-C", str(ROOT), "fetch", "--quiet", "--depth", "1", "origin", BRANCH]
    try:
        ok = subprocess.run(fetch, capture_output=True, timeout=300).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    if not ok:
        why = "couldn't fetch the prebuilt app (no network, or no git?)"
        return False
    if _git("show", "FETCH_HEAD:SOURCE") != want:
        why = "there's no prebuilt app for this version yet (CI builds it a few minutes after a push)"
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
        why = "the prebuilt app was incomplete"
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


def ensure() -> None:
    want = _wanted()
    if want is None:
        return
    logging.info("[ZenTrainer] fetching the prebuilt web app for this version…")
    if _fetch(want):
        logging.info("[ZenTrainer] web app ready")
    else:
        logging.warning("[ZenTrainer] web app: %s", why)
