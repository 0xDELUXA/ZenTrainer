"""Runs: create, edit while queued, resume, sample, control; presets; the training queue and its worker."""

from __future__ import annotations

import asyncio
from pathlib import Path

from aiohttp import web

from ... import presets, state
from .. import jobqueue, system
from ...harness import control
from ...spec import SpecError, TrainSpec, locked_changes
from . import common
from .common import RETRYABLE, _bad, _dequeue, _keep_later, _migrate_states, _queue_session, _remember_client


def register(server) -> None:
    routes = server.routes

    @routes.get("/zentrainer/api/runs")
    async def list_runs(_request):
        return web.json_response({"runs": state.store().list()})

    def _with_size(c: dict) -> dict:
        # a cleaned-up training state can't be resumed from: say so rather than offer it
        c = {**c, "state": c["state"] if c.get("state") and Path(c["state"]).is_file() else None}
        try:
            return {**c, "size": Path(c["path"]).stat().st_size}
        except (OSError, KeyError):
            return {**c, "size": None}  # moved or deleted since

    @routes.get("/zentrainer/api/runs/{run_id}")
    async def get_run(request):
        store, run_id = state.store(), request.match_info["run_id"]
        try:
            return web.json_response({
                "run": store.meta(run_id),
                "spec": store.spec(run_id),
                "metrics": store.read(run_id, "metrics"),
                "checkpoints": [_with_size(c) for c in store.read(run_id, "checkpoints")],
                "samples": store.read(run_id, "samples"),
                "validation": store.read(run_id, "validation"),
                "likeness": store.read(run_id, "likeness"),
                "live": control.get(run_id) is not None,
            })
        except (FileNotFoundError, ValueError):
            return _bad("run not found", 404)

    @routes.post("/zentrainer/api/runs")
    async def create_run(request):
        body = await request.json()
        try:
            spec = TrainSpec.from_dict(body.get("spec", {}))
        except (SpecError, TypeError) as e:
            return _bad(str(e))

        store = state.store()
        run_id = store.new_id()
        store.create(run_id, spec.to_dict())
        # "next" jumps the queue (starts now when idle); "end" waits its turn
        jobqueue.get().add(run_id, front=body.get("position", "next") == "next")
        _remember_client(body.get("client_id"))
        return web.json_response({"run": store.meta(run_id), "queue": jobqueue.get().snapshot()})

    @routes.put("/zentrainer/api/runs/{run_id}/spec")
    async def edit_queued_run(request):
        """Change a run's settings: {"spec"}. A run waiting in the queue keeps its place; one
        that ended before it trained a step (failed loading, out of VRAM at step one, stopped
        while queued) goes back to the front of the queue with them. A run that has trained
        keeps its spec: its checkpoints were made with it (Duplicate makes a new one)."""
        store, run_id = state.store(), request.match_info["run_id"]
        try:
            meta = store.meta(run_id)
        except (FileNotFoundError, ValueError):
            return _bad("no such run", 404)
        waiting = run_id in jobqueue.get().snapshot()["order"] and meta.get("status") == "queued"
        retry = (control.get(run_id) is None and meta.get("status") in RETRYABLE
                 and not store.read(run_id, "metrics") and not store.read(run_id, "checkpoints"))
        # a stopped run that has trained: settings the saved LoRA and optimizer state don't
        # depend on can change before it resumes (memory, samples, LR, length...)
        resume_edit = (control.get(run_id) is None and meta.get("status") in RETRYABLE and not retry
                       and bool(store.read(run_id, "checkpoints")))
        if not (waiting or retry or resume_edit):
            return _bad(f"this run is {meta.get('status')}: it can't be edited", 409)
        body = await request.json()
        try:
            spec = TrainSpec.from_dict(body.get("spec", {}))
            spec.validate()
        except (SpecError, TypeError) as e:
            return _bad(str(e))
        if resume_edit:
            # through the dataclass so fields added since it was saved compare at their defaults
            changed = locked_changes(TrainSpec.from_dict(store.spec(run_id)).to_dict(), spec.to_dict())
            if changed:
                return _bad("these can't change once a run has trained (its LoRA and optimizer state were "
                            f"made with them; Duplicate starts over): {', '.join(changed)}", 409, locked=changed)
        store.set_spec(run_id, spec.to_dict())
        if retry:
            for stream in ("samples", "validation"):
                store.rewrite(run_id, stream, [])  # its step-0 samples come again with the new settings
            store.update(run_id, message=None, final_step=None, resume_from=None, resume_state=None, status="queued")
            jobqueue.get().add(run_id, front=True)
            _remember_client(body.get("client_id"))
        return web.json_response({"run": store.meta(run_id), "requeued": retry})

    @routes.post("/zentrainer/api/runs/{run_id}/resume")
    async def resume_run(request):
        """Continue a stopped/failed run from its latest resumable checkpoint, or from
        a given one ({"checkpoint": path}); with none, it restarts from step 0."""
        store, run_id = state.store(), request.match_info["run_id"]
        try:
            meta = store.meta(run_id)
        except (FileNotFoundError, ValueError):
            return _bad("run not found", 404)
        if control.get(run_id) is not None or meta["status"] in ("queued", "loading", "training", "paused"):
            return _bad("run is already active", 409)
        body = await request.json() if request.can_read_body else {}
        from ...harness.trainer import state_for

        resumable = [c for c in store.read(run_id, "checkpoints") if state_for(c, run_id)]
        if body.get("checkpoint"):
            resumable = [c for c in resumable if c["path"] == body["checkpoint"]]
            if not resumable:
                return _bad("that checkpoint has no saved training state", 400)
        ckpt = max(resumable, key=lambda c: c["step"]) if resumable else None
        step = ckpt["step"] if ckpt else 0
        if ckpt is None and (store.read(run_id, "checkpoints") or store.read(run_id, "metrics")):
            # starting over would drop its history and retrain over its saved LoRAs and samples
            return _bad("this run has no saved training state to continue from (Resume states was off, or "
                        "they were cleaned up). Duplicate it to train again from the start.", 409)
        _keep_later(store, run_id, step)
        for stream in ("metrics", "samples", "checkpoints", "validation", "likeness"):
            store.truncate(run_id, stream, step)
        store.update(run_id, resume_from=ckpt["path"] if ckpt else None,
                     resume_state=state_for(ckpt, run_id) if ckpt else None, message=None, final_step=None,
                     status="queued")
        jobqueue.get().add(run_id, front=True)
        _remember_client(body.get("client_id"))
        return web.json_response({"run": store.meta(run_id)})

    @routes.post("/zentrainer/api/runs/{run_id}/sample")
    async def sample_run(request):
        """Sample a finished run's checkpoint: {"checkpoint"?: path (default the latest), "run_prompts"?:
        bool (default true), "tries"?: [{prompt, width?, height?, seed?}]}. Renders at that checkpoint's
        step, replacing the run's samples already there; the run keeps its status."""
        store, run_id = state.store(), request.match_info["run_id"]
        try:
            meta = store.meta(run_id)
        except (FileNotFoundError, ValueError):
            return _bad("run not found", 404)
        if control.get(run_id) is not None or meta["status"] in ("queued", "loading", "training", "paused", "sampling"):
            return _bad("run is active: use Sample now while it trains", 409)
        body = await request.json() if request.can_read_body else {}
        ckpts = [c for c in store.read(run_id, "checkpoints") if c.get("path") and Path(c["path"]).is_file()]
        if body.get("checkpoint"):
            ckpts = [c for c in ckpts if c["path"] == body["checkpoint"]]
        if not ckpts:
            return _bad("no checkpoint file to sample", 400)
        ckpt = max(ckpts, key=lambda c: (c["step"], bool(c.get("final"))))
        run_prompts = body.get("run_prompts", True)
        tries = [t for t in body.get("tries") or [] if isinstance(t, dict) and str(t.get("prompt", "")).strip()]
        if not run_prompts and not tries:
            return _bad("nothing to sample", 400)
        if run_prompts:  # a re-sample replaces the run's samples (and their likeness) at that step
            for stream in ("samples", "likeness"):
                rows = store.read(run_id, stream)
                store.rewrite(run_id, stream, [r for r in rows if r.get("step") != ckpt["step"] or r.get("manual")])
        store.update(run_id, status="queued", message=f"sampling step {ckpt['step']}", sample_job={
            "path": ckpt["path"], "step": ckpt["step"], "run_prompts": bool(run_prompts), "tries": tries,
            "prev_status": meta["status"], "prev_message": meta.get("message")})
        jobqueue.get().add(run_id, front=True)
        _remember_client(body.get("client_id"))
        return web.json_response({"run": store.meta(run_id)})

    @routes.post("/zentrainer/api/runs/{run_id}/control")
    async def control_run(request):
        """{"command": "pause"|"resume"|"stop"|"save"|"sample"|"set_lr", "args": {...}}"""
        run_id = request.match_info["run_id"]
        body = await request.json()
        ctl = control.get(run_id)
        if ctl is None:
            if body.get("command") == "stop":  # waiting in our queue, or dispatched but not started
                if jobqueue.get().remove(run_id):
                    state.store().update(run_id, status="stopped", message="removed from queue")
                    return web.json_response({"ok": True})
                return await _dequeue(server, run_id)
            return _bad("run is not training", 409)
        try:
            ctl.send(body.get("command", ""), **(body.get("args") or {}))
        except (ValueError, TypeError) as e:
            return _bad(str(e))
        if body.get("command") == "sample":  # acknowledge now; it renders at the next step boundary
            from ... import events
            events.emit(events.SAMPLE_PROGRESS, {"run_id": run_id, "phase": "requested"})
        return web.json_response({"ok": True})

    def _preset_root() -> Path:
        import folder_paths

        return Path(folder_paths.get_user_directory()) / "zentrainer" / "presets"

    @routes.get("/zentrainer/api/presets/{kind}")
    async def list_presets(request):
        try:
            return web.json_response({"presets": presets.list_presets(request.match_info["kind"], _preset_root())})
        except presets.PresetError as e:
            return _bad(str(e))

    @routes.post("/zentrainer/api/presets/{kind}")
    async def save_preset(request):
        b = await request.json()
        try:
            row = presets.save(request.match_info["kind"], str(b.get("name", "")), b.get("data"), _preset_root(),
                               str(b.get("description", "")))
        except presets.PresetError as e:
            return _bad(str(e))
        return web.json_response({"preset": row})

    @routes.delete("/zentrainer/api/presets/{kind}/{id}")
    async def delete_preset(request):
        try:
            ok = presets.delete(request.match_info["kind"], request.match_info["id"], _preset_root())
        except presets.PresetError as e:
            return _bad(str(e))
        return web.json_response({"ok": ok}) if ok else _bad("no such user preset (built-ins can't be deleted)", 404)

    @routes.get("/zentrainer/api/queue")
    async def get_queue(_request):
        q, store = jobqueue.get().snapshot(), state.store()
        def meta(rid):
            try:
                return store.meta(rid)
            except (FileNotFoundError, ValueError):
                return None
        return web.json_response({
            "paused": q["paused"],
            "current": meta(q["current"]) if q["current"] else None,
            "items": [m for m in map(meta, q["order"]) if m],
        })

    @routes.post("/zentrainer/api/queue")
    async def edit_queue(request):
        """{"action": "pause"|"resume"} or {"action": "order", "order": [...]} or
        {"action": "top"|"remove", "run_id": ...}"""
        b, q = await request.json(), jobqueue.get()
        action = b.get("action")
        if action in ("pause", "resume"):
            q.set_paused(action == "pause")
        elif action == "order":
            q.reorder([str(r) for r in b.get("order", [])])
        elif action == "top":
            q.add(str(b.get("run_id")), front=True)
        elif action == "remove":
            if q.remove(str(b.get("run_id"))):
                state.store().update(str(b["run_id"]), status="stopped", message="removed from queue")
        else:
            return _bad(f"unknown action {action!r}")
        return web.json_response(q.snapshot())

    async def _start_worker(_app):
        await asyncio.to_thread(_migrate_states)
        # runs that were mid-flight when this process last stopped can't be running now
        store, waiting = state.store(), set(jobqueue.get().snapshot()["order"])
        for m in store.list():
            job = m.get("sample_job")
            if job and (m["status"] == "sampling" or (m["status"] == "queued" and m["run_id"] not in waiting)):
                # a checkpoint was being sampled: the run itself was finished, so it goes back as it was
                store.update(m["run_id"], status=job.get("prev_status") or "done", message=job.get("prev_message"),
                             sample_job=None)
                continue
            if m["status"] in ("loading", "training", "paused") or (m["status"] == "queued" and m["run_id"] not in waiting):
                store.update(m["run_id"], status="interrupted", message="ComfyUI stopped during this run; resume it")

        async def dispatch(run_id: str):
            try:
                name = state.store().meta(run_id)["name"]
            except (FileNotFoundError, ValueError):
                return
            await _queue_session(server, run_id, name, common._client_id)
        _app["zentrainer_queue"] = asyncio.create_task(jobqueue.worker(dispatch))
        _app["zentrainer_system"] = asyncio.create_task(system.sampler())

    server.app.on_startup.append(_start_worker)
