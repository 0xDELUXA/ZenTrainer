"""The web app (web/dist): kept, fetched prebuilt, or built with pnpm.

CI builds web/ on every push to main and keeps the result on the web-dist branch, with SOURCE
naming the last commit that changed web/ it was built from. web/dist/.source says the same for
the copy here, so it's replaced only when web/ has moved on. web/dist is built output: it's the
only thing here that's ever replaced, and only by a complete build.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import zipfile
from pathlib import Path

from . import repo, ui

BRANCH = "web-dist"


def source() -> str:
    """The last commit that changed web/: what a matching build was built from."""
    return repo.git("log", "-1", "--format=%H", "--", "web")


def built_from() -> str:
    f = repo.DIST / ".source"
    return f.read_text().strip() if f.is_file() else ""


def _swap_in(new: Path, src: str) -> None:
    """Put a complete build in place of web/dist."""
    (new / ".source").write_text(src + "\n")
    old = repo.WEB / ".dist-old"
    if old.exists():
        shutil.rmtree(old)
    if repo.DIST.exists():
        repo.DIST.rename(old)
    new.rename(repo.DIST)
    if old.exists():
        shutil.rmtree(old)


def prebuilt(want: str) -> bool:
    """Fetch the web-dist branch's build; True when it matches `want` and is in place."""
    if not repo.git("ls-remote", "--heads", "origin", BRANCH):
        return False
    fetch = ["git", "-C", str(repo.ROOT), "fetch", "--quiet", "--depth", "1", "origin", BRANCH]
    if subprocess.run(fetch, capture_output=True).returncode != 0:
        return False
    if repo.git("show", "FETCH_HEAD:SOURCE") != want:
        return False
    data = subprocess.run(["git", "-C", str(repo.ROOT), "archive", "--format=zip", "FETCH_HEAD"],
                          capture_output=True).stdout
    new = repo.WEB / ".dist-new"
    if new.exists():
        shutil.rmtree(new)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for m in z.namelist():
            if m != "SOURCE" and ".." not in Path(m).parts:
                z.extract(m, new)
    if not (new / "index.html").is_file():
        shutil.rmtree(new)
        return False
    _swap_in(new, want)
    return True


def ensure(force: bool = False) -> None:
    """web/dist up to date with web/."""
    want = source()
    have = (repo.DIST / "index.html").is_file()
    if have and not force and (not want or built_from() == want):
        return
    if shutil.which("pnpm"):
        ui.run(["pnpm", "install"], cwd=repo.WEB)
        ui.run(["pnpm", "build"], cwd=repo.WEB)
        if want and (repo.DIST / "index.html").is_file():
            (repo.DIST / ".source").write_text(want + "\n")
        return
    with ui.console.status("fetching the prebuilt web app…"):
        got = bool(want) and prebuilt(want)
    if got:
        ui.ok("web app: the prebuilt one for this version")
    elif have:
        ui.warn("web app: no build for this version yet (and no pnpm to build it); keeping the one here")
    else:
        ui.warn("web app: no prebuilt one for this version, and pnpm isn't installed to build it. "
                "Install pnpm (https://pnpm.io/installation) and run `zt setup --build-web`.")
