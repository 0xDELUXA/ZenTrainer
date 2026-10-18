"""Batches: one submission of an app (one prompt per grid cell), stored as a file and
resolved against ComfyUI's history for status and outputs."""

from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from typing import Any


class BatchStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def new_id(self) -> str:
        return time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)

    def _path(self, batch_id: str) -> Path:
        if not batch_id or "/" in batch_id or "\\" in batch_id or batch_id.startswith("."):
            raise ValueError(f"bad batch id {batch_id!r}")
        return self.root / f"{batch_id}.json"

    def save(self, batch: dict) -> None:
        self._path(batch["batch_id"]).write_text(json.dumps(batch, indent=1))

    def get(self, batch_id: str) -> dict:
        return json.loads(self._path(batch_id).read_text())

    def list(self, app_id: str | None = None, limit: int = 30) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("*.json"), reverse=True):
            try:
                b = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if app_id is None or b.get("app") == app_id:
                out.append(b)
            if len(out) >= limit:
                break
        return out


def resolve(batch: dict, history: dict[str, Any], queued: set[str] | None = None) -> dict:
    """Fill each cell's status and images from ComfyUI's /history entries. Cells already
    finished keep their stored result: ComfyUI's history doesn't survive a restart. A cell
    in neither the history nor `queued` (ComfyUI's queue, when given) was lost with a
    restart, and is marked so instead of waiting forever."""
    cells = []
    for c in batch["cells"]:
        if c.get("status") in ("done", "error"):
            cells.append(c)
            continue
        h = history.get(c["prompt_id"])
        status, images, error = "queued", [], None
        if not h and queued is not None and c["prompt_id"] not in queued:
            status, error = "error", "lost: ComfyUI restarted before it ran"
        if h:
            st = h.get("status", {})
            if st.get("status_str") == "error":
                status = "error"
                for kind, msg in st.get("messages", []):
                    if kind == "execution_error":
                        error = msg.get("exception_message")
            elif st.get("completed"):
                status = "done"
            out = h.get("outputs", {}).get(c.get("save") or "", {})
            images = [{"filename": i["filename"], "subfolder": i.get("subfolder", ""), "type": i.get("type", "output")}
                      for i in out.get("images", [])]
        cells.append({**c, "status": status, "images": images, "error": error})
    done = sum(1 for c in cells if c["status"] in ("done", "error"))
    return {**batch, "cells": cells, "done": done, "total": len(cells),
            "status": "done" if done == len(cells) else ("running" if done else "queued")}
