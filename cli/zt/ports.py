"""Ports: where ZenTrainer is set to run, where it actually is, and a free one to start on.

A started ComfyUI is remembered in .zentrainer/running.json, so when its port was taken and it
moved, `zt restart`, `zt runs`, `zt monitor` and the rest still find it. `port` arguments here
are a --port from the command line (None when not given).
"""

from __future__ import annotations

import json
import os
import socket
import urllib.request

from . import repo, ui


def configured(port: int | None = None) -> int:
    return port or repo.local().get("port") or repo.lock()["port"]


def alive(port: int) -> bool:
    """A ZenTrainer answers there (not just anything)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/zentrainer/api/health", timeout=1):
            return True
    except Exception:
        return False


def remember(port: int) -> None:
    repo.RUNNING.parent.mkdir(exist_ok=True)
    repo.RUNNING.write_text(json.dumps({"port": port}) + "\n")


def running(port: int | None = None) -> int | None:
    """The port a ZenTrainer answers on: the one asked for, the configured one, or where it last started."""
    ports = [configured(port)]
    if not port and repo.RUNNING.is_file():
        try:
            ports.append(int(json.loads(repo.RUNNING.read_text())["port"]))
        except (ValueError, KeyError, TypeError):
            pass
    return next((p for p in ports if alive(p)), None)


def current(port: int | None = None) -> int:
    """For talking to ZenTrainer: where it's running, else where it would run."""
    return running(port) or configured(port)


def taken(port: int) -> bool:
    """Something already listening on (or holding) the port."""
    with socket.socket() as s:
        s.settimeout(0.3)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            return True
    with socket.socket() as s:
        # a server that just stopped leaves its closed connections on the port for about a minute
        # (TIME_WAIT); ComfyUI binds through those, so the test must too. Not on Windows, where the
        # same flag would let it bind over a program still using the port.
        if os.name != "nt":
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


def free(port: int | None = None) -> int:
    """The configured port, or the next free one when another program has it. A --port given on
    the command line is kept to: it stops with a message instead of moving."""
    want = configured(port)
    if not taken(want):
        return want
    if port:
        ui.die(f"port {want} is in use by another program; pick another with --port")
    for p in range(want + 1, want + 50):
        if not taken(p):
            ui.warn(f"port {want} is taken by another program, so ZenTrainer uses {p} this time "
                    "(change the port for good in the menu's Settings)")
            return p
    ui.die(f"ports {want}–{want + 49} are all in use; pick one with --port")
