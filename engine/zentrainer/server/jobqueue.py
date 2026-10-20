"""The training queue: an ordered, persistent list of runs, dispatched one at a time.

Runs wait here (status "queued"), not in ComfyUI's prompt queue, so they can be
reordered, removed or held while the queue is paused, and the order survives a
restart. The worker hands the head of the queue to ComfyUI (a one-node session job)
only when no run is active, so training jobs never pile up behind each other there.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path

from .. import state

TERMINAL = {"done", "stopped", "error", "interrupted", "rejected"}
_lock = threading.Lock()


class Queue:
    def __init__(self, path: Path):
        self.path = path
        self.current: str | None = None  # dispatched to ComfyUI, not yet finished

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            data = {}
        return {"paused": bool(data.get("paused", False)), "order": list(data.get("order", []))}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.path)

    def snapshot(self) -> dict:
        with _lock:
            return {**self._read(), "current": self.current}

    def add(self, run_id: str, front: bool = False) -> None:
        with _lock:
            d = self._read()
            d["order"] = [r for r in d["order"] if r != run_id]
            d["order"].insert(0, run_id) if front else d["order"].append(run_id)
            self._write(d)

    def remove(self, run_id: str) -> bool:
        with _lock:
            d = self._read()
            if run_id not in d["order"]:
                return False
            d["order"].remove(run_id)
            self._write(d)
            return True

    def reorder(self, order: list[str]) -> None:
        """Apply a new order to the queued runs; ids not in the queue are ignored and
        queued ids missing from `order` keep their place at the end."""
        with _lock:
            d = self._read()
            known = set(d["order"])
            head = [r for r in order if r in known]
            d["order"] = head + [r for r in d["order"] if r not in head]
            self._write(d)

    def set_paused(self, paused: bool) -> None:
        with _lock:
            d = self._read()
            d["paused"] = paused
            self._write(d)

    def pop_next(self) -> str | None:
        with _lock:
            d = self._read()
            if d["paused"] or not d["order"]:
                return None
            run_id = d["order"].pop(0)
            self._write(d)
            return run_id


_queue: Queue | None = None


def get() -> Queue:
    global _queue
    if _queue is None:
        _queue = Queue(state.store().root.parent / "queue.json")
    return _queue


async def worker(dispatch, interval: float = 1.5) -> None:
    """Dispatch the next queued run whenever nothing is active. `dispatch(run_id)` is
    the coroutine that submits the session job to ComfyUI."""
    from ..harness import control

    q = get()
    store = state.store()
    while True:
        try:
            if q.current is not None:
                try:
                    status = store.meta(q.current)["status"]
                except (FileNotFoundError, ValueError):
                    status = "error"
                if status in TERMINAL and control.get(q.current) is None:
                    q.current = None
            if q.current is None and not control.active():
                run_id = q.pop_next()
                if run_id is not None:
                    q.current = run_id
                    await dispatch(run_id)
        except Exception:
            logging.exception("[ZenTrainer] queue worker")
        await asyncio.sleep(interval)
