"""HTTP API, mounted on ComfyUI's own server under /zentrainer/api, one file per area.

A run is a spec in the run store plus a one-node ComfyUI job (ZenTrainSession) that
hosts the harness. The job only claims the execution lane; everything about the run
lives in the spec and is steered through /control while it trains. The job is
queued through ComfyUI's public /prompt endpoint (loopback), not its internals.
"""

from __future__ import annotations

from . import apps, datasets, host, loras, models, runs, webapp
from .common import comfy_too_old

__all__ = ["register", "comfy_too_old"]


def register(server) -> None:
    # in this order: aiohttp matches routes as registered, and the web app's catch-all is last
    for area in (host, datasets, runs, apps, loras, models, webapp):
        area.register(server)
