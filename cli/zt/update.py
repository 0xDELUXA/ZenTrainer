"""Keeping ZenTrainer itself up to date.

Only ever a fast-forward of a clean checkout: local changes, local commits or another branch's
history are left exactly as they are, and the update is skipped with a word on why.
"""

from __future__ import annotations

import subprocess

from rich.text import Text

from . import repo, ui, web

Found = tuple[str, list[str]]  # the upstream, and its new commits ("sha subject")


def check() -> Found | None:
    """The branch's upstream and the commits it has that this checkout lacks."""
    up = repo.git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if not up:
        return None
    remote = up.split("/", 1)[0]
    if subprocess.run(["git", "-C", str(repo.ROOT), "fetch", "--quiet", remote], capture_output=True).returncode != 0:
        return None
    new = repo.git("log", "--format=%h %s", f"HEAD..{up}")
    return (up, new.splitlines()) if new else None


def apply(found: Found, sync: bool = True) -> bool:
    """Fast-forward to the upstream, then (sync) bring ComfyUI and the web app along. False if skipped."""
    from . import comfy

    up, _ = found
    if repo.git("status", "--porcelain", "--untracked-files=no"):
        ui.warn("you have local changes to ZenTrainer's files: skipping the update (commit or stash them first)")
        return False
    if repo.git("rev-list", "--count", f"{up}..HEAD") not in ("", "0"):
        ui.warn(f"this branch has commits {up} doesn't: skipping the update (pull it yourself)")
        return False
    lock_before = repo.git("rev-parse", "HEAD:comfy.lock")
    try:
        ui.run(["git", "-C", repo.ROOT, "merge", "--ff-only", "--quiet", up])
    except subprocess.CalledProcessError:
        ui.warn("the update couldn't fast-forward: nothing was changed")
        return False
    ui.ok(f"up to date: {repo.git('log', '-1', '--format=%h %s')}")
    if not sync:
        return True
    if repo.venv_python().is_file() and repo.git("rev-parse", "HEAD:comfy.lock") != lock_before:
        ui.say("comfy.lock moved: bringing the managed ComfyUI along")
        comfy.setup(torch=repo.local().get("torch"))
    else:
        web.ensure()
    return True


def offer() -> None:
    """At start: say what's new, and update as the `updates` setting says (ask, always, never)."""
    mode = repo.local().get("updates", "ask")
    if mode == "never" or not (repo.ROOT / ".git").exists():
        return
    with ui.console.status("checking for updates…"):
        found = check()
    if not found:
        return
    _, commits = found
    ui.say(f"{len(commits)} new {'commit' if len(commits) == 1 else 'commits'}:")
    for c in commits[:8]:
        sha, _, subject = c.partition(" ")
        ui.console.print(Text.assemble("    ", (sha, "dim"), " ", subject))
    if len(commits) > 8:
        ui.console.print(f"    … and {len(commits) - 8} more", style="dim")
    if mode == "always":
        apply(found)
        return
    pick = ui.choose("Update now?", [("Yes", "y"), ("Not now", "n"), ("Always, without asking", "a")], "y")
    if pick == "a":
        cfg = repo.local()
        cfg["updates"] = "always"
        repo.save_local(cfg)
    if pick in ("y", "a"):
        apply(found)
