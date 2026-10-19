"""HTTP API, mounted on ComfyUI's own server under /zentrainer/api.

A run is a spec in the run store plus a one-node ComfyUI job (ZenTrainSession) that
hosts the harness. The job only claims the execution lane; everything about the run
lives in the spec and is steered through /control while it trains. The job is
queued through ComfyUI's public /prompt endpoint (loopback), not its internals.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import time
import uuid
from pathlib import Path

import aiohttp
from aiohttp import web

from . import __version__, state
from . import datasets, jobqueue, presets, safeio, system
from .harness import control
from .spec import SpecError, TrainSpec, locked_changes

WEB_DIST = Path(__file__).resolve().parent.parent.parent / "web" / "dist"


# ended without training a step: editable, and saving puts it back in the queue
RETRYABLE = ("error", "stopped", "interrupted", "rejected")


def _keep_later(store, run_id: str, step: int) -> None:
    """Before a resume retrains past `step`, move the LoRAs and sample pictures saved after it into
    a `replaced-<time>` folder beside them: retraining writes the same file names, and those files
    are someone's results until they say otherwise."""
    import shutil

    import folder_paths

    tag = f"replaced-{time.strftime('%Y%m%d-%H%M%S')}"
    out = Path(folder_paths.get_output_directory())
    files = [Path(c["path"]) for c in store.read(run_id, "checkpoints") if c.get("step", 0) > step and c.get("path")]
    files += [out / r.get("subfolder", "") / r["filename"] for r in store.read(run_id, "samples")
              if r.get("step", 0) > step and r.get("filename") and r.get("type", "output") == "output"]
    for f in files:
        if f.is_file():
            keep = f.parent / tag / f.name
            keep.parent.mkdir(exist_ok=True)
            shutil.move(str(f), str(keep))


def _bad(message: str, status: int = 400, **extra) -> web.Response:
    return web.json_response({"error": message, **extra}, status=status)


# The oldest ComfyUI ZenTrainer is tested with (comfy.lock pins newer). Installed as a node pack
# into someone's ComfyUI, an older one would fail in odd places; this says so plainly instead.
MIN_COMFY = (0, 41, 0)


def comfy_too_old() -> str | None:
    try:
        from comfyui_version import __version__ as v
    except Exception:
        return None
    nums = tuple(int(x) for x in re.findall(r"\d+", v)[:3])
    if nums and nums < MIN_COMFY:
        return (f"ComfyUI {v} is older than ZenTrainer needs ({'.'.join(map(str, MIN_COMFY))} or newer): "
                "update ComfyUI (Manager > Update ComfyUI, or git pull)")
    return None


def register(server) -> None:
    routes = server.routes

    @routes.get("/zentrainer/api/health")
    async def health(_request):
        return web.json_response({"version": __version__, "capabilities": _capabilities(), "stale": _code_stale(),
                                  "comfy_old": comfy_too_old()})

    @routes.get("/zentrainer/api/themes")
    async def themes(_request):
        from .themes import list_themes

        return web.json_response({"themes": await asyncio.to_thread(list_themes)})

    @routes.get("/zentrainer/api/datasets")
    async def list_datasets(_request):
        return web.json_response({"datasets": await asyncio.to_thread(datasets.summary)})

    @routes.post("/zentrainer/api/datasets")
    async def create_dataset(request):
        body = await request.json()
        return await _ds(lambda: datasets.create(str(body.get("name", "")).strip()) or {"ok": True})

    @routes.get("/zentrainer/api/datasets/{name}")
    async def get_dataset(request):
        return await _ds(lambda: datasets.items(request.match_info["name"]))

    @routes.get("/zentrainer/api/datasets/{name}/thumb")
    async def thumb(request):
        import folder_paths

        name, rel = request.match_info["name"], request.query.get("path", "")
        cache = Path(folder_paths.get_user_directory()) / "zentrainer" / "thumbs"
        try:
            path = await asyncio.to_thread(datasets.thumbnail, name, rel, int(request.query.get("size", 384)), cache)
        except FileNotFoundError:
            return _bad("not found", 404)
        except (datasets.DatasetError, ValueError) as e:
            return _bad(str(e))
        return web.FileResponse(path, headers={"Cache-Control": "public, max-age=31536000, immutable"})

    @routes.put("/zentrainer/api/datasets/{name}/caption")
    async def put_caption(request):
        body = await request.json()
        name = request.match_info["name"]
        return await _ds(lambda: datasets.set_caption(name, body["path"], str(body.get("caption", "")),
                                                      set_name=str(body.get("set") or "default")) or {"ok": True})

    @routes.post("/zentrainer/api/datasets/{name}/bulk")
    async def bulk_captions(request):
        b = await request.json()
        name = request.match_info["name"]
        return await _ds(lambda: {"changed": datasets.bulk(name, b.get("op", ""), b.get("text", ""), b.get("find", ""),
                                                           set_name=str(b.get("set") or "default"))})

    @routes.put("/zentrainer/api/datasets/{name}/muted")
    async def dataset_muted(request):
        """{"set", "paths", "muted": bool}: mute or unmute pictures in a caption set."""
        b = await request.json()
        try:
            return web.json_response({"muted": datasets.set_muted(request.match_info["name"], str(b.get("set") or "default"),
                                                                  [str(p) for p in b.get("paths") or []], bool(b.get("muted", True)))})
        except FileNotFoundError:
            return _bad("dataset not found", 404)
        except datasets.DatasetError as e:
            return _bad(str(e))

    @routes.put("/zentrainer/api/datasets/{name}/settings")
    async def dataset_settings(request):
        """{"faces": bool}: this dataset's settings (faces: face report on open, in the stats, likeness)."""
        try:
            return web.json_response(datasets.put_settings(request.match_info["name"], await request.json()))
        except FileNotFoundError:
            return _bad("dataset not found", 404)

    @routes.get("/zentrainer/api/datasets/{name}/stats")
    async def dataset_stats(request):
        """The dataset's health now and over time: {"current": {metrics, scores}, "history"}; a
        snapshot is kept whenever the dataset changed since the last look."""
        from . import dataset_stats

        try:
            return web.json_response(await asyncio.to_thread(dataset_stats.snapshot, request.match_info["name"],
                                                             request.query.get("set") or "default"))
        except FileNotFoundError:
            return _bad("dataset not found", 404)

    @routes.get("/zentrainer/api/datasets/{name}/faces")
    async def dataset_faces(request):
        """The dataset's faces (faces extra): per picture, faces found, the main face's size, how
        much it looks like the rest, outliers. Cached per file; the first look takes a while."""
        from . import faces

        if not faces.available():
            return web.json_response({"available": False,
                                      "hint": "install the faces extra: scripts/zt extras faces "
                                              "(pip install insightface onnxruntime opencv-python-headless)"})
        name = request.match_info["name"]
        try:
            out = await asyncio.to_thread(faces.dataset_faces, name)
        except FileNotFoundError:
            return _bad("dataset not found", 404)
        return web.json_response({"available": True, **out})

    @routes.post("/zentrainer/api/datasets/{name}/split")
    async def split_videos(request):
        """Cut long videos into short clips: {seconds?, count? (per video, 0 = all), longer_than?}."""
        name = request.match_info["name"]
        b = await request.json() if request.can_read_body else {}
        return await _ds(lambda: datasets.split_videos(name, float(b.get("seconds") or 3.0), int(b.get("count", 20)),
                                                       float(b.get("longer_than") or 20.0)))

    @routes.get("/zentrainer/api/caption")
    async def caption_info(request):
        """Styles, models, and the recent jobs (?dataset= for one dataset's)."""
        from . import captioner

        if request.query.get("active") == "1":
            return web.json_response({"jobs": captioner.active()})
        if request.query.get("all") == "1":
            return web.json_response({"jobs": captioner.history()})
        return web.json_response({
            "styles": [{"id": k, "label": v[0], "prompt": v[1]} for k, v in captioner.STYLES.items()],
            # what goes around a style's prompt, so the panel can show exactly what the model is told
            "notes": {"video": captioner.VIDEO_NOTE, "sound": captioner.SOUND_NOTE, "subject": captioner.SUBJECT_NOTE},
            "models": await asyncio.to_thread(captioner.models),
            "hearing": await asyncio.to_thread(captioner.hearing_models),
            "senses": await asyncio.to_thread(captioner.model_senses),
            "loras": await asyncio.to_thread(captioner.te_loras),
            "lora_catalog": await asyncio.to_thread(captioner.te_lora_catalog),
            "jobs": captioner.recent(request.query.get("dataset") or None),
        })

    @routes.post("/zentrainer/api/caption/loras")
    async def add_caption_lora(request):
        """{"link", "file"?}: download a text-encoder LoRA for captioning from Hugging Face."""
        from . import captioner

        b = await request.json()
        try:
            if b.get("id"):  # a catalog entry
                return web.json_response(await asyncio.to_thread(captioner.download_te_lora, str(b["id"])))
            return web.json_response(await asyncio.to_thread(captioner.add_te_lora, str(b.get("link", "")), b.get("file")))
        except (ValueError, OSError) as e:
            return _bad(str(e))

    @routes.post("/zentrainer/api/datasets/{name}/caption")
    async def caption_dataset(request):
        """{"clip_name", "clip_type", "set", "style" | "prompt", "mode": missing|all|selected,
        "paths" (selected), "subject", "preview": n, "max_length", "replace_default"}.
        Rewriting captions that already exist in the "default" set (the one training reads
        unless told otherwise) needs replace_default: true, so it never happens by accident."""
        from . import captioner
        from .harness.data import caption_path, scan

        name, b = request.match_info["name"], await request.json()
        try:
            root = datasets.folder(name)
            set_name = datasets._set(str(b.get("set") or "default"))
        except FileNotFoundError:
            return _bad("dataset not found", 404)
        except datasets.DatasetError as e:
            return _bad(str(e))
        if not b.get("clip_name"):
            return _bad("pick a vision-language text encoder")
        mode = str(b.get("mode") or ("all" if b.get("overwrite") else "missing"))
        if mode not in captioner.MODES:
            return _bad(f"mode is one of {', '.join(captioner.MODES)}")
        paths = [str(p) for p in b.get("paths") or []]
        if mode == "selected" and not paths:
            return _bad("select some images first")
        preview = max(0, min(int(b.get("preview") or 0), 8))
        prompt = str(b.get("prompt") or captioner.STYLES.get(str(b.get("style")), captioner.STYLES["nl"])[1])
        job = captioner.CaptionJob(job_id="", dataset=name, set_name=set_name, clip_name=str(b["clip_name"]),
                                   clip_type=str(b.get("clip_type") or "auto"), prompt=prompt, mode=mode, paths=paths,
                                   subject=str(b.get("subject") or "").strip(), preview=preview)
        if not preview and set_name == "default" and not b.get("replace_default"):
            items = await asyncio.to_thread(lambda: captioner.todo(job, scan(root, warn=False)))
            existing = sum(caption_path(it.path, "default").is_file() for it in items)
            if existing:
                return _bad(f"this would rewrite {existing} existing caption{'s' if existing != 1 else ''} in the "
                            "default set", 409, replaces=existing)
        job = captioner.create(**{f: getattr(job, f) for f in ("dataset", "set_name", "clip_name", "clip_type", "prompt",
                                                               "mode", "paths", "subject", "preview")},
                               max_length=int(b.get("max_length") or 300),
                               loras=[{"name": str(lo.get("name")), "strength": float(lo.get("strength", 1.0))}
                                      for lo in (b.get("loras") or []) if lo.get("name")])
        prompt_id = str(uuid.uuid4())
        from . import jobqueue

        busy = jobqueue.get().current
        captioner.update(job, prompt_id=prompt_id, message="waits for the training run to finish" if busy else "")
        payload = {"prompt": {"cap": {"class_type": "ZenCaptionSession", "inputs": {"job_id": job.job_id},
                                      "_meta": {"title": f"caption {name}"}}},
                   "prompt_id": prompt_id, "client_id": b.get("client_id")}
        async with aiohttp.ClientSession() as s:
            async with s.post(_loopback(server) + "/prompt", json=payload) as r:
                result = await r.json()
        if r.status != 200:
            captioner.update(job, status="error", message=str(result.get("error")))
            return _bad("ComfyUI rejected the caption job", 422, comfy=result)
        return web.json_response({"job": job.public()})

    @routes.post("/zentrainer/api/datasets/{name}/segment")
    async def segment_image(request):
        """SAM3 on one image: {path, text?, positive?: [{x,y}], negative?: [{x,y}], threshold?, refine?}.
        Waits up to ~20 s for the result ({job} with status done -> GET .../segment/{id}.png);
        when the GPU is busy (a training run) it returns the queued job to poll."""
        from . import jobqueue, segment

        name = request.match_info["name"]
        b = await request.json()
        try:
            root = datasets.folder(name)
            img = (root / str(b.get("path", ""))).resolve()
            if root.resolve() not in img.parents or not img.is_file():
                return _bad("no such image in the dataset", 404)
        except FileNotFoundError:
            return _bad("dataset not found", 404)
        job = segment.create(img, text=str(b.get("text") or ""), positive=list(b.get("positive") or []),
                             negative=list(b.get("negative") or []), threshold=float(b.get("threshold") or 0.5),
                             refine=int(b.get("refine", 2)))
        if jobqueue.get().current:
            job.message = "waits for the training run to finish"
        payload = {"prompt": {"seg": {"class_type": "ZenSegmentSession", "inputs": {"job_id": job.job_id},
                                      "_meta": {"title": f"mask {name}"}}}, "prompt_id": str(uuid.uuid4())}
        async with aiohttp.ClientSession() as s:
            async with s.post(_loopback(server) + "/prompt", json=payload) as r:
                res = await r.json()
        if r.status != 200:
            return _bad("ComfyUI rejected the segment job", 422, comfy=res)
        for _ in range(100):
            if job.status not in segment.ACTIVE:
                break
            await asyncio.sleep(0.2)
        return web.json_response({"job": job.public()})

    @routes.get("/zentrainer/api/segment/{job_id}")
    async def segment_job(request):
        from . import segment

        jid = request.match_info["job_id"]
        if jid.endswith(".png"):
            p = segment.result(jid[:-4])
            return web.FileResponse(p, headers={"Cache-Control": "no-store"}) if p else _bad("no mask yet", 404)
        job = segment.get(jid)
        return web.json_response({"job": job.public()}) if job else _bad("no such job", 404)

    @routes.put("/zentrainer/api/datasets/{name}/mask")
    async def save_mask(request):
        """Write an image's mask from the editor: {path, png: data URL}; empty png deletes it."""
        name = request.match_info["name"]
        b = await request.json()
        return await _ds(lambda: datasets.save_mask(name, str(b.get("path", "")), str(b.get("png") or "")))

    @routes.post("/zentrainer/api/datasets/{name}/masks/apply")
    async def apply_masks(request):
        """One mask onto many images of its size: {paths, png, mode: cutout|replace}.
        Returns {batch (to undo), results: [{path, mask} | {path, skipped}]}."""
        name, b = request.match_info["name"], await request.json()
        paths = [str(p) for p in (b.get("paths") or [])]
        return await _ds(lambda: datasets.apply_mask(name, paths, str(b.get("png") or ""), str(b.get("mode") or "cutout")))

    @routes.post("/zentrainer/api/datasets/{name}/masks/undo")
    async def undo_masks(request):
        """Undo an apply: {batch}."""
        name, b = request.match_info["name"], await request.json()
        return await _ds(lambda: datasets.undo_masks(name, str(b.get("batch") or "")))

    @routes.get("/zentrainer/api/caption/{job_id}")
    async def caption_job(request):
        from . import captioner

        job = captioner.get(request.match_info["job_id"])
        if job is None:
            return _bad("no such job", 404)
        return web.json_response({"job": {**job.public(), "captions": dict(job.written)}})

    @routes.post("/zentrainer/api/caption/{job_id}/{action}")
    async def caption_action(request):
        """cancel: stop it (before it starts, or at the next image); undo: put back what it changed."""
        from . import captioner

        job, action = captioner.get(request.match_info["job_id"]), request.match_info["action"]
        if job is None:
            return _bad("no such job", 404)
        if action == "cancel":
            was = job.status
            captioner.cancel(job)
            if was == "queued" and job.prompt_id:
                async with aiohttp.ClientSession() as s:
                    await s.post(_loopback(server) + "/queue", json={"delete": [job.prompt_id]})
            return web.json_response({"job": job.public()})
        if action == "undo":
            try:
                n = await asyncio.to_thread(captioner.undo, job)
            except (ValueError, FileNotFoundError) as e:
                return _bad(str(e))
            return web.json_response({"job": job.public(), "restored": n})
        return _bad("action is cancel or undo", 404)

    @routes.delete("/zentrainer/api/datasets/{name}/item")
    async def delete_item(request):
        name, rel = request.match_info["name"], request.query.get("path", "")
        return await _ds(lambda: datasets.delete(name, rel) or {"ok": True})

    @routes.post("/zentrainer/api/datasets/{name}/upload")
    async def upload(request):
        name = request.match_info["name"]
        saved, errors = [], []
        reader = await request.multipart()
        async for part in reader:
            if part.filename:
                data = await part.read(decode=False)
                try:
                    saved.append(await asyncio.to_thread(datasets.save_upload, name, part.filename, bytes(data)))
                except (datasets.DatasetError, FileNotFoundError) as e:
                    errors.append(str(e))
        return web.json_response({"saved": saved, "errors": errors})

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
        from .harness.trainer import state_for

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
            from . import events
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

    @routes.get("/zentrainer/api/system/models")
    async def loaded_models(_request):
        """Each model ComfyUI holds: its size, how much is in VRAM, per block (see residency.py)."""
        from . import residency

        return web.json_response({"models": await asyncio.to_thread(residency.report)})

    @routes.get("/zentrainer/api/system")
    async def get_system(request):
        """Memory samples newer than ?since=<unix time> (all buffered ones by default)."""
        try:
            since = float(request.query.get("since", 0))
        except ValueError:
            since = 0.0
        return web.json_response({"interval": system.INTERVAL, "samples": system.history(since)})

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

    # ---- apps ----------------------------------------------------------------------------
    def _apps_user_root() -> Path:
        import folder_paths

        return Path(folder_paths.get_user_directory()) / "zentrainer" / "apps"

    def _batches():
        import folder_paths

        from .apps.runtime import BatchStore

        return BatchStore(Path(folder_paths.get_user_directory()) / "zentrainer" / "app_batches")

    @routes.get("/zentrainer/api/apps")
    async def list_apps(_request):
        from .apps import manifest

        apps, errors = manifest.list_apps(_apps_user_root())
        return web.json_response({"apps": [a.public() for a in apps], "errors": errors})

    @routes.get("/zentrainer/api/apps/{id}/defaults")
    async def app_defaults(request):
        """The selected workflow's own values ({input: value}) and which inputs it takes, so a
        form starts where the workflow does and hides what it doesn't have. ?family= picks the
        workflow of a per-model app."""
        from .apps import compile as appcompile
        from .apps import manifest

        apps, _ = manifest.list_apps(_apps_user_root())
        app = next((a for a in apps if a.id == request.match_info["id"]), None)
        if app is None:
            return _bad("no such app", 404)
        family = request.query.get("family") or None
        try:
            graph = manifest.pick_workflow(app, None, family) if family else app.workflows[0][1]
            values = appcompile.defaults_for(app, family)
        except manifest.AppError as e:
            return _bad(str(e))
        return web.json_response({"values": values, "inputs": app.bound_in(graph)})

    @routes.get("/zentrainer/api/apps/{id}/cover")
    async def app_cover(request):
        from .apps import manifest

        apps, _ = manifest.list_apps(_apps_user_root())
        app = next((a for a in apps if a.id == request.match_info["id"]), None)
        cover = app and app.manifest.get("cover")
        path = (app.root / cover).resolve() if cover else None
        if not path or not path.is_relative_to(app.root.resolve()) or not path.is_file():
            return _bad("no cover", 404)
        return web.FileResponse(path)

    @routes.post("/zentrainer/api/apps/{id}/run")
    async def run_app(request):
        """{"inputs": {...}, "sweep": {input: [values]}, "client_id"}: one prompt per cell."""
        from .apps import compile as appcompile
        from .apps import manifest

        apps, _ = manifest.list_apps(_apps_user_root())
        app = next((a for a in apps if a.id == request.match_info["id"]), None)
        if app is None:
            return _bad("no such app", 404)
        b = await request.json()
        store = _batches()
        batch_id = store.new_id()
        try:
            cells = appcompile.compile_batch(app, b.get("inputs") or {}, b.get("sweep") or {},
                                             output_prefix=f"zentrainer/apps/{app.id}/{batch_id}")
        except manifest.AppError as e:
            return _bad(str(e))
        submitted = []
        async with aiohttp.ClientSession() as s:
            for cell in cells:
                pid = str(uuid.uuid4())
                # preview_method: ComfyUI streams sampling previews for this prompt (to client_id)
                payload = {"prompt": cell["prompt"], "prompt_id": pid,
                           "extra_data": {"zentrainer_app": app.id, "preview_method": "auto"}}
                if b.get("client_id"):
                    payload["client_id"] = b["client_id"]
                async with s.post(_loopback(server) + "/prompt", json=payload) as r:
                    result = await r.json()
                if r.status != 200:
                    return _bad("ComfyUI rejected the workflow", 422, comfy=result, submitted=len(submitted))
                submitted.append({"coords": cell["coords"], "prompt_id": pid, "save": cell["save"]})
        batch = {"batch_id": batch_id, "app": app.id, "inputs": b.get("inputs") or {}, "sweep": b.get("sweep") or {},
                 "created": time.time(), "cells": submitted}
        store.save(batch)
        return web.json_response({"batch": batch})

    async def _resolve_batch(batch: dict) -> dict:
        from .apps.runtime import resolve

        history = {}
        queued: set[str] | None = None
        pending = [c for c in batch["cells"] if c.get("status") not in ("done", "error")]
        if pending:
            async with aiohttp.ClientSession() as s:
                for c in pending:
                    async with s.get(_loopback(server) + f"/history/{c['prompt_id']}") as r:
                        if r.status == 200:
                            history.update(await r.json())
                if any(c["prompt_id"] not in history for c in pending):
                    async with s.get(_loopback(server) + "/queue") as r:
                        q = await r.json() if r.status == 200 else {}
                    queued = {str(e[1]) for k in ("queue_running", "queue_pending") for e in q.get(k, []) if len(e) > 1}
        resolved = resolve(batch, history, queued)
        if any(c["status"] in ("done", "error") for c in resolved["cells"]) and pending:
            # persist finished cells: ComfyUI's history is in memory only
            _batches().save({k: v for k, v in resolved.items() if k not in ("done", "total", "status")})
        return resolved

    @routes.get("/zentrainer/api/gallery")
    async def gallery(request):
        """Every image the apps made, newest first: [{batch_id, prompt_id, app, created, image,
        values}] where values are what made it (the run's inputs plus that cell's sweep values).
        ?app= narrows to one app; ?limit/offset page through."""
        app_id = request.query.get("app") or None
        limit = max(1, min(int(request.query.get("limit", 200)), 1000))
        offset = max(0, int(request.query.get("offset", 0)))
        items = []
        for b in _batches().list(app_id, limit=10_000):
            if any(c.get("status") not in ("done", "error") for c in b["cells"]):
                b = await _resolve_batch(b)
            for c in b["cells"]:
                for img in c.get("images") or []:
                    items.append({"batch_id": b["batch_id"], "prompt_id": c["prompt_id"], "app": b["app"],
                                  "created": b["created"], "image": img,
                                  "values": {**(b.get("inputs") or {}), **(c.get("coords") or {})}})
        return web.json_response({"items": items[offset:offset + limit], "total": len(items)})

    @routes.delete("/zentrainer/api/gallery/{batch_id}/{prompt_id}")
    async def gallery_delete(request):
        """Delete one generation's image files and drop it from its batch."""
        import folder_paths

        store = _batches()
        try:
            batch = store.get(request.match_info["batch_id"])
        except (FileNotFoundError, ValueError):
            return _bad("no such batch", 404)
        out_root = (Path(folder_paths.get_output_directory()) / "zentrainer" / "apps").resolve()
        cells, removed = [], 0
        for c in batch["cells"]:
            if c["prompt_id"] != request.match_info["prompt_id"]:
                cells.append(c)
                continue
            for img in c.get("images") or []:
                path = (Path(folder_paths.get_output_directory()) / img.get("subfolder", "") / img["filename"]).resolve()
                if path.is_relative_to(out_root) and path.is_file():  # only app outputs, never anything else
                    path.unlink()
                    removed += 1
        if len(cells) == len(batch["cells"]):
            return _bad("no such image", 404)
        store.save({**batch, "cells": cells})
        return web.json_response({"removed": removed})

    @routes.get("/zentrainer/api/apps/batches/{batch_id}")
    async def get_batch(request):
        try:
            batch = _batches().get(request.match_info["batch_id"])
        except (FileNotFoundError, ValueError):
            return _bad("no such batch", 404)
        return web.json_response({"batch": await _resolve_batch(batch)})

    @routes.get("/zentrainer/api/apps/{id}/batches")
    async def app_batches(request):
        rows = _batches().list(request.match_info["id"], limit=int(request.query.get("limit", 12)))
        return web.json_response({"batches": [await _resolve_batch(b) for b in rows]})

    # ---- the LoRA library behind zenkit-ui's LoRA picker ----------------------------------------
    def _lora_path(name: str) -> str | None:
        import folder_paths

        return folder_paths.get_full_path("loras", name) if name else None

    def _lora_rows() -> list[dict]:
        import os

        import folder_paths

        from .loras import favorites, meta

        names = folder_paths.get_filename_list("loras")
        favs = set(favorites.get_favorites())
        dirs: dict[str, set[str]] = {}
        rows = []
        for n in names:
            path = _lora_path(n)
            folder = os.path.dirname(path) if path else ""
            if path and folder not in dirs:
                try:
                    dirs[folder] = set(os.listdir(folder))
                except OSError:
                    dirs[folder] = set()
            found = meta.find_preview(path, dirs.get(folder, set())) if path else None
            try:
                st = os.stat(path) if path else None
            except OSError:
                st = None
            name = n.replace("\\", "/")
            rows.append({"name": name, "has_preview": found is not None, "preview": found[1] if found else None,
                         "favorite": name in favs, "size": st.st_size if st else None,
                         "mtime": int(st.st_mtime) if st else None})
        return rows

    @routes.get("/zentrainer/api/loras")
    async def list_loras(_request):
        return web.json_response({"loras": await asyncio.to_thread(_lora_rows)})

    @routes.get("/zentrainer/api/lora/preview")
    async def lora_preview(request):
        """The preview as a still; raw=1 a video preview as itself; size=<px> a cached thumbnail."""
        from .loras import meta

        found = meta.find_preview(_lora_path(request.query.get("name", "")))
        if not found:
            return web.Response(status=404)
        path, kind = found
        raw = request.query.get("raw") == "1"
        if kind == "video" and not raw:
            path = await asyncio.to_thread(meta.poster, path)
            if not path:
                return web.Response(status=404)
        try:
            size = int(request.query.get("size", "0"))
        except ValueError:
            size = 0
        if size > 0 and not raw and not path.lower().endswith(".gif"):
            path = await asyncio.to_thread(meta.thumb, path, size) or path
        return web.FileResponse(path)

    @routes.get("/zentrainer/api/lora/info")
    async def lora_info(request):
        from urllib.parse import quote

        from .loras import favorites, meta

        name = request.query.get("name", "").replace("\\", "/")
        path = _lora_path(name)
        if not path:
            return _bad("not found", 404)
        info = await asyncio.to_thread(meta.read_info, path)
        q = "?name=" + quote(name)
        for img in info.get("images", []):  # downloaded examples served from disk; the URL stays as a fallback
            file = img.pop("file", None)
            img["remote"] = img["url"] or None
            if file:
                img["url"] = "/zentrainer/api/lora/example" + q + "&file=" + quote(file)
        found = meta.find_preview(path)
        info.update(name=name, preview=found[1] if found else None, favorite=name in favorites.get_favorites())
        return web.json_response(info)

    @routes.get("/zentrainer/api/lora/example")
    async def lora_example(request):
        from .loras import lora_manager, meta

        path = _lora_path(request.query.get("name", ""))
        sha = await asyncio.to_thread(meta.sha256_of, path) if path else None
        file = await asyncio.to_thread(lora_manager.resolve, sha, request.query.get("file", ""))
        return web.FileResponse(file) if file else web.Response(status=404)

    @routes.get("/zentrainer/api/lora/favorites")
    async def lora_favorites(_request):
        from .loras import favorites

        return web.json_response({"loras": favorites.get_favorites()})

    @routes.post("/zentrainer/api/lora/favorites")
    async def lora_set_favorite(request):
        from .loras import favorites

        body = await request.json()
        name = str(body.get("name", "")).replace("\\", "/")
        if not name:
            return _bad("missing name")
        return web.json_response({"loras": favorites.toggle(name, bool(body.get("pinned", True)))})

    # ---- storage --------------------------------------------------------------------------------
    @routes.get("/zentrainer/api/storage")
    async def get_storage(request):
        """The full report; ?disks=1 for just free space (cheap: the sidebar polls it)."""
        from . import storage

        if request.query.get("disks") == "1":
            return web.json_response({"disks": await asyncio.to_thread(storage.disks)})
        return web.json_response(await asyncio.to_thread(storage.report, state.store()))

    @routes.get("/zentrainer/api/storage/places")
    async def storage_places(_request):
        """Where a run can save its LoRAs: ComfyUI's loras folders, with free space."""
        import folder_paths

        default = Path(folder_paths.get_output_directory()) / "zentrainer" / "loras"
        base = Path(folder_paths.base_path).resolve()
        out = []
        for p in folder_paths.get_folder_paths("loras"):
            rp = Path(p).resolve()
            try:
                u = shutil.disk_usage(rp)
            except OSError:
                continue
            out.append({"path": str(rp), "free": u.free, "total": u.total, "default": rp == default.resolve(),
                        "inside_install": rp.is_relative_to(base)})
        return web.json_response({"loras": out})

    @routes.post("/zentrainer/api/storage/clean")
    async def clean_storage(request):
        """{"kind": "states", "run_id"} | {"kind": "states_all"} | {"kind": "cache", "folder"}:
        returns {"freed": bytes} ("runs" too for states_all: how many it cleaned)."""
        from . import datasets, storage

        b = await request.json()
        try:
            if b.get("kind") == "states":
                freed = await asyncio.to_thread(storage.clean_states, state.store(), str(b.get("run_id", "")))
            elif b.get("kind") == "states_all":
                freed, runs = await asyncio.to_thread(storage.clean_all_states, state.store())
                return web.json_response({"freed": freed, "runs": runs})
            elif b.get("kind") == "cache":
                freed = await asyncio.to_thread(storage.clean_cache, state.store(), str(b.get("folder", "")))
            else:
                return _bad("kind must be states or cache")
        except (storage.StorageError, datasets.DatasetError, ValueError) as e:
            return _bad(str(e))
        except FileNotFoundError:
            return _bad("not found", 404)
        return web.json_response({"freed": freed})

    # ---- ComfyUI workflows in App mode (templates, the user's): what the importer reads -----------
    @routes.get("/zentrainer/api/wapps")
    async def list_wapps(_request):
        from .apps import catalog

        return web.json_response({"apps": await asyncio.to_thread(catalog.list_apps)})

    @routes.get("/zentrainer/api/wapps/describe")
    async def describe_wapp(request):
        from .apps import catalog
        from .apps.workflow import WorkflowError

        try:
            d = await asyncio.to_thread(catalog.describe, request.query.get("id", ""))
        except catalog.AppNotFound as e:
            return _bad(f"no such app: {e}", 404)
        except WorkflowError as e:
            return _bad(str(e))
        return web.json_response(d)

    # ---- model families + downloads ------------------------------------------------------------

    @routes.get("/zentrainer/api/families")
    async def list_families(request):
        """Model families from ComfyUI's templates, with what's installed of each."""
        from . import families

        return web.json_response({"families": await asyncio.to_thread(families.list_families)})

    @routes.get("/zentrainer/api/families/{fid}")
    async def get_family(request):
        """One family; ?repo=1 adds every build in its Hugging Face repo (cached a day,
        ?refresh=1 refetches)."""
        from . import families

        fam = await asyncio.to_thread(families.get, request.match_info["fid"],
                                      request.query.get("repo") == "1", request.query.get("refresh") == "1")
        return web.json_response(fam) if fam else _bad("no such family", 404)

    @routes.get("/zentrainer/api/downloads")
    async def list_downloads(request):
        from . import downloads, families

        targets = {}
        for folder in ("diffusion_models", "text_encoders", "vae", "checkpoints"):
            try:
                targets[folder] = str(downloads.target_dir(folder))
            except ValueError:
                pass
        return web.json_response({"downloads": downloads.manager().list(), "targets": targets,
                                  "hf_token": bool(families._hf_headers())})

    @routes.post("/zentrainer/api/downloads")
    async def start_download(request):
        """Download a file the family catalog knows: {family, folder, name}."""
        from . import downloads

        body = await request.json()
        try:
            job = await asyncio.to_thread(downloads.manager().start, str(body.get("family", "")),
                                          str(body.get("folder", "")), str(body.get("name", "")))
        except (ValueError, OSError) as e:
            return _bad(str(e))
        return web.json_response(job.public())

    @routes.post("/zentrainer/api/loras/merge")
    async def merge_loras(request):
        """Merge LoRAs: {members: [{name, strength}], out_name, max_rank?, energy?}. Saves to
        output/zentrainer/loras/merged/ (a loras folder), so it loads at once."""
        import re as _re

        import folder_paths

        from . import merge

        b = await request.json()
        try:
            members = [(Path(folder_paths.get_full_path_or_raise("loras", str(m["name"]))), float(m.get("strength", 1.0)))
                       for m in b.get("members", [])]
        except Exception as e:
            return _bad(f"can't find that LoRA: {e}")
        if not members:
            return _bad("add at least one LoRA")
        name = _re.sub(r"[^A-Za-z0-9_.-]+", "_", str(b.get("out_name") or "merged")).strip("._") or "merged"
        out = Path(folder_paths.get_output_directory()) / "zentrainer" / "loras" / "merged" / f"{name}.safetensors"
        if out.exists() and not b.get("overwrite"):
            return _bad(f"merged/{out.name} already exists", 409)
        try:
            stats = await asyncio.to_thread(merge.merge_files, members, out, int(b.get("max_rank") or 0),
                                            float(b.get("energy") or 0.999), overwrite_ok=bool(b.get("overwrite")))
        except Exception as e:
            return _bad(f"merge failed: {e}")
        return web.json_response({"lora_name": f"merged/{out.name}", **stats})

    @routes.get("/zentrainer/api/adapters")
    async def list_adapters(request):
        """Training adapters (or ?kind=base: base LoRAs) for the picker: the family's known ones
        first, custom, then local files."""
        from . import adapters

        kind = "base" if request.query.get("kind") == "base" else "training"
        rows = await asyncio.to_thread(adapters.listing, request.query.get("family") or None, kind)
        return web.json_response({"adapters": rows})

    @routes.post("/zentrainer/api/adapters")
    async def add_adapter(request):
        """Add one from a Hugging Face link: {link, family?, file?, kind?}. Returns {adapter} or {choices}."""
        from . import adapters

        body = await request.json()
        try:
            out = await asyncio.to_thread(adapters.add, str(body.get("link", "")), body.get("family") or None,
                                          body.get("file") or None, "base" if body.get("kind") == "base" else "training")
        except ValueError as e:
            return _bad(str(e))
        return web.json_response(out)

    @routes.post("/zentrainer/api/adapters/download")
    async def download_adapter(request):
        from . import adapters

        body = await request.json()
        try:
            job = await asyncio.to_thread(adapters.download, str(body.get("id", "")))
        except (ValueError, OSError) as e:
            return _bad(str(e))
        return web.json_response(job.public())

    @routes.delete("/zentrainer/api/adapters")
    async def remove_adapter(request):
        from . import adapters

        return web.json_response({"removed": adapters.remove(request.query.get("id", ""))})

    @routes.delete("/zentrainer/api/downloads/{job_id}")
    async def cancel_download(request):
        from . import downloads

        ok = downloads.manager().cancel(request.match_info["job_id"])
        return web.json_response({"cancelled": ok})

    @routes.post("/zentrainer/api/downloads/clear")
    async def clear_downloads(request):
        from . import downloads

        downloads.manager().clear()
        return web.json_response({"ok": True})

    @routes.get("/zentrainer/api/comfyui")
    async def comfyui_version(request):
        """This ComfyUI's version and commit, the pin in comfy.lock, and (with ?check=1)
        how far upstream has moved. The check fetches, so it only runs when asked."""
        return web.json_response(await asyncio.to_thread(_comfy_version, request.query.get("check") == "1"))

    @routes.post("/zentrainer/api/restart")
    async def restart(request):
        """Restart this ComfyUI process (same interpreter and arguments)."""
        body = await request.json() if request.can_read_body else {}
        if control.active() and not body.get("force"):
            return _bad("a run is training; stop it first (or pass force)", 409)
        asyncio.get_running_loop().call_later(0.5, _exec_self)
        return web.json_response({"ok": True})

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
            await _queue_session(server, run_id, name, _client_id)
        _app["zentrainer_queue"] = asyncio.create_task(jobqueue.worker(dispatch))
        _app["zentrainer_system"] = asyncio.create_task(system.sampler())

    server.app.on_startup.append(_start_worker)

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


def _remember_client(client_id: str | None) -> None:
    """Session jobs report progress to the last UI client that queued something."""
    global _client_id
    if client_id:
        _client_id = client_id


def _exec_self() -> None:
    import os
    import sys

    logging.info("[ZenTrainer] restarting ComfyUI")
    sys.stdout.flush()
    os.execv(sys.executable, [sys.executable, *sys.argv])


async def _queue_session(server, run_id: str, name: str, client_id: str | None) -> web.Response:
    """Queue the one-node job that hosts the run. Its only job is to hold ComfyUI's
    execution lane; the run itself is defined by the spec in the store."""
    store = state.store()
    prompt_id = str(uuid.uuid4())
    prompt = {"session": {"class_type": "ZenTrainSession", "inputs": {"run_id": run_id},
                          "_meta": {"title": f"train {name}"}}}
    payload = {"prompt": prompt, "prompt_id": prompt_id, "extra_data": {"zentrainer_run": run_id}}
    if client_id:
        payload["client_id"] = client_id
    store.update(run_id, prompt_id=prompt_id, status="queued")
    async with aiohttp.ClientSession() as s:
        async with s.post(_loopback(server) + "/prompt", json=payload) as r:
            result = await r.json()
    if r.status != 200:
        store.update(run_id, status="rejected", message=str(result.get("error")))
        return _bad("ComfyUI rejected the run", 422, run_id=run_id, comfy=result)
    return web.json_response({"run": store.meta(run_id)})


async def _dequeue(server, run_id: str) -> web.Response:
    store = state.store()
    try:
        prompt_id = store.meta(run_id)["prompt_id"]
    except (FileNotFoundError, ValueError):
        return _bad("run not found", 404)
    async with aiohttp.ClientSession() as s:
        await s.post(_loopback(server) + "/queue", json={"delete": [prompt_id]})
    store.update(run_id, status="stopped", message="removed from queue")
    return web.json_response({"ok": True})


async def _ds(fn) -> web.Response:
    """Run a dataset operation off the event loop, mapping its errors to HTTP."""
    try:
        return web.json_response(await asyncio.to_thread(fn))
    except FileNotFoundError:
        return _bad("dataset not found", 404)
    except (datasets.DatasetError, KeyError) as e:
        return _bad(str(e))


def _migrate_states() -> None:
    """Move training-state files out of the LoRA folders (older versions saved
    <name>.state.pt beside each checkpoint, where ComfyUI lists it as a LoRA)."""

    import folder_paths

    from .harness.trainer import state_dir

    store = state.store()
    for meta in store.list():
        rid = meta["run_id"]
        rows = store.read(rid, "checkpoints")
        changed = False
        for row in rows:
            old = Path(row["path"]).with_suffix(".state.pt")
            if not row.get("state") and old.is_file():
                new = state_dir(rid) / f"{Path(row['path']).stem}.pt"
                new.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old), str(new))
                row["state"] = str(new)
                changed = True
        if changed:
            store.rewrite(rid, "checkpoints", rows)
            logging.info("[ZenTrainer] moved training state of %s out of the LoRA folder", rid)
    # strays not tied to a run row
    root = Path(folder_paths.get_output_directory()) / "zentrainer" / "loras"
    for stray in root.rglob("*.state.pt") if root.is_dir() else []:
        dest = safeio.free_name(state_dir("_orphaned") / stray.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stray), str(dest))


def _comfy_version(check: bool) -> dict:
    import json
    import subprocess

    import folder_paths

    root = Path(folder_paths.__file__).resolve().parent
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, timeout=60)  # noqa: E731
    out: dict = {"path": str(root)}
    try:
        import comfyui_version

        out["version"] = comfyui_version.__version__
    except Exception:
        pass
    head = git("rev-parse", "HEAD")
    if head.returncode == 0:
        out["commit"] = head.stdout.strip()
        out["describe"] = git("describe", "--tags", "--always").stdout.strip()
        out["dirty"] = bool(git("status", "--porcelain", "--untracked-files=no").stdout.strip())
    lock = Path(__file__).resolve().parent.parent.parent / "comfy.lock"
    if lock.is_file():
        pin = json.loads(lock.read_text())["comfyui"]
        out["pin"] = {"ref": pin["ref"], "describe": pin.get("describe")}
        out["matches_pin"] = out.get("commit") == pin["ref"]
    if check and "commit" in out:
        fetched = git("fetch", "--quiet", "origin")
        if fetched.returncode == 0:
            upstream = git("rev-parse", "origin/HEAD").stdout.strip() or git("rev-parse", "origin/master").stdout.strip()
            behind = git("rev-list", "--count", f"HEAD..{upstream}").stdout.strip()
            out["upstream"] = {"commit": upstream, "describe": git("describe", "--tags", "--always", upstream).stdout.strip(),
                               "behind": int(behind or 0)}
        else:
            out["upstream_error"] = fetched.stderr.strip()[:300]
    return out


def _loopback(server) -> str:
    host = server.address if server.address not in ("0.0.0.0", "::", "", None) else "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{server.port}"


_LOADED_AT = time.time()


def _code_stale() -> bool:
    """The engine's code on disk is newer than what this process loaded (an update or an edit):
    a restart picks it up. The sidebar offers one."""
    root = Path(__file__).resolve().parent
    return any(p.stat().st_mtime > _LOADED_AT for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _capabilities() -> dict:
    from .harness import optim

    caps: dict = {"optimizers": optim.catalog()}
    try:
        from comfy.ldm.modules.attention import REGISTERED_ATTENTION_FUNCTIONS

        # Only backends with a backward pass are offered for training.
        caps["train_attention"] = [n for n in REGISTERED_ATTENTION_FUNCTIONS if n not in ("flash", "sage", "sage3")]
        caps["infer_attention"] = list(REGISTERED_ATTENTION_FUNCTIONS)
    except Exception:
        pass
    try:
        import comfy.model_management as mm

        dev = mm.get_torch_device()
        caps["device"] = str(dev)
        caps["device_name"] = mm.get_torch_device_name(dev)
        caps["vram_total"] = mm.get_total_memory(dev)
        caps["vram_free"] = mm.get_free_memory(dev)
    except Exception:
        pass
    return caps
