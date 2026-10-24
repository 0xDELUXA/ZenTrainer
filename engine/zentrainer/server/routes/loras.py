"""The LoRA library behind zenkit-ui's LoRA picker (previews, info, favorites), merging LoRAs,
and compacting one to a size budget."""

from __future__ import annotations

import asyncio
from pathlib import Path

import threading

from aiohttp import web

from ... import paths
from .common import _bad


def register(server) -> None:
    routes = server.routes

    # ---- the LoRA library behind zenkit-ui's LoRA picker ----------------------------------------
    def _lora_path(name: str) -> str | None:
        import folder_paths

        return folder_paths.get_full_path("loras", name) if name else None

    def _lora_rows() -> list[dict]:
        import os

        import folder_paths

        from ...loras import favorites, meta

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
        from ...loras import meta

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

        from ...loras import favorites, meta

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
        from ...loras import lora_manager, meta

        path = _lora_path(request.query.get("name", ""))
        sha = await asyncio.to_thread(meta.sha256_of, path) if path else None
        file = await asyncio.to_thread(lora_manager.resolve, sha, request.query.get("file", ""))
        return web.FileResponse(file) if file else web.Response(status=404)

    @routes.get("/zentrainer/api/lora/favorites")
    async def lora_favorites(_request):
        from ...loras import favorites

        return web.json_response({"loras": favorites.get_favorites()})

    @routes.post("/zentrainer/api/lora/favorites")
    async def lora_set_favorite(request):
        from ...loras import favorites

        body = await request.json()
        name = str(body.get("name", "")).replace("\\", "/")
        if not name:
            return _bad("missing name")
        return web.json_response({"loras": favorites.toggle(name, bool(body.get("pinned", True)))})

    @routes.post("/zentrainer/api/loras/merge")
    async def merge_loras(request):
        """Merge LoRAs: {members: [{name, strength}], out_name, max_rank?, energy?}. Saves to
        output/zentrainer/loras/merged/ (a loras folder; the ZenTrainer folder's output/loras/merged/ if set), so it loads at once."""
        import re as _re

        import folder_paths

        from ...loras import merge

        b = await request.json()
        try:
            members = [(Path(folder_paths.get_full_path_or_raise("loras", str(m["name"]))), float(m.get("strength", 1.0)))
                       for m in b.get("members", [])]
        except Exception as e:
            return _bad(f"can't find that LoRA: {e}")
        if not members:
            return _bad("add at least one LoRA")
        name = _re.sub(r"[^A-Za-z0-9_.-]+", "_", str(b.get("out_name") or "merged")).strip("._") or "merged"
        out = paths.lora_dir() / "merged" / f"{name}.safetensors"
        if out.exists() and not b.get("overwrite"):
            return _bad(f"merged/{out.name} already exists", 409)
        try:
            stats = await asyncio.to_thread(merge.merge_files, members, out, int(b.get("max_rank") or 0),
                                            float(b.get("energy") or 0.999), overwrite_ok=bool(b.get("overwrite")))
        except Exception as e:
            return _bad(f"merge failed: {e}")
        return web.json_response({"lora_name": f"merged/{out.name}", **stats})

    # ---- compacting a LoRA or LoKr to a budget for the whole file (loras/compact.py) ----------------
    analyses: dict[tuple, object] = {}  # the last few, so moving the target doesn't redo the SVDs
    LEVELS = (0.9, 0.95, 0.98, 0.99, 0.995, 1.0)  # the shares the page offers, each with its size
    lock = threading.Lock()

    def _analysis(path: str, fmt: str, base: str | None):
        import os

        from ...loras import compact

        key = (path, os.path.getmtime(path), fmt, base)
        with lock:
            if key in analyses:
                return analyses[key]
        an = compact.analyze(path, fmt, base)
        with lock:
            analyses[key] = an
            while len(analyses) > 3:
                analyses.pop(next(iter(analyses)))
        return an

    def _compact_args(b: dict):
        import folder_paths

        path = _lora_path(str(b.get("name", "")).replace("\\", "/"))
        if not path:
            raise ValueError("can't find that LoRA")
        base = None
        if b.get("base"):
            folder = str(b["base"].get("folder", "diffusion_models"))
            if folder not in ("diffusion_models", "checkpoints"):
                raise ValueError("the base model is a diffusion model or a checkpoint")
            base = folder_paths.get_full_path(folder, str(b["base"].get("name", "")))
            if not base:
                raise ValueError("can't find that base model")
        fmt = "lora" if b.get("format") == "lora" else "same"
        target = b.get("target") or {}
        keep = float(target["keep"]) if "keep" in target else None
        params = int(target["params"]) if "params" in target else None
        if keep is None and params is None:
            keep = 0.99
        return path, fmt, base, keep, params, max(0, int(b.get("max_rank") or 0))

    def _precision(path: str) -> str | None:
        """The file's float precision (bf16, fp16, fp32), by the bytes in each; None if another."""
        from ...models.tensors import BYTES, read_header

        by: dict[str, int] = {}
        for t in read_header(path)[1].values():
            n = BYTES.get(t["dtype"], 0)
            for d in t["shape"]:
                n *= d
            by[t["dtype"]] = by.get(t["dtype"], 0) + n
        top = max(by, key=by.get, default=None)
        return {"F32": "fp32", "F16": "fp16", "BF16": "bf16"}.get(top or "")

    @routes.post("/zentrainer/api/loras/compact/analyze")
    async def compact_analyze(request):
        """{name, format?: same|lora, base?: {folder, name}, max_rank?, target?: {keep: 0-1} | {params}}:
        each offered level's size, and the ranks a target keeps (layer by layer)."""
        import os

        from ...loras import compact

        try:
            path, fmt, base, keep, params, max_rank = _compact_args(await request.json())
            an = await asyncio.to_thread(_analysis, path, fmt, base)
        except (ValueError, OSError, KeyError) as e:
            return _bad(str(e))

        def answer():
            ks = compact.ranks(an, keep=keep, params=params, max_rank=max_rank)
            levels = []
            for share in LEVELS:
                st = compact.summary(an, compact.ranks(an, keep=share, max_rank=max_rank))
                levels.append({"keep": share, "params": st["params"], "kept": st["kept"]})
            return {"original": {"params": an.original_params, "bytes": os.path.getsize(path), "precision": _precision(path)},
                    "levels": levels,
                    "plan": compact.summary(an, ks),
                    "names": [layer.name for layer in an.layers], "kind": sorted({layer.kind for layer in an.layers}),
                    "full_ranks": [layer.s.numel() for layer in an.layers],
                    "unsupported": an.unsupported, "base_matched": an.base_matched}

        return web.json_response(await asyncio.to_thread(answer))

    @routes.post("/zentrainer/api/loras/compact")
    async def compact_lora(request):
        """The analyze body plus {out_name, dtype?: bf16|fp16|fp32, overwrite?}. Saves to
        output/zentrainer/loras/compact/ (the ZenTrainer folder's output/loras/compact/ if set)."""
        import re as _re

        from ...loras import compact

        b = await request.json()
        try:
            path, fmt, base, keep, params, max_rank = _compact_args(b)
        except (ValueError, KeyError) as e:
            return _bad(str(e))
        name = _re.sub(r"[^A-Za-z0-9_.-]+", "_", str(b.get("out_name") or "")).strip("._")
        if name.lower().endswith(".safetensors"):
            name = name[: -len(".safetensors")]
        if not name:
            return _bad("give the compacted LoRA a name")
        out = paths.lora_dir() / "compact" / f"{name}.safetensors"
        if out.exists() and not b.get("overwrite"):
            return _bad(f"compact/{out.name} already exists", 409)

        def run():
            an = _analysis(path, fmt, base)
            ks = compact.ranks(an, keep=keep, params=params, max_rank=max_rank)
            return compact.build(path, out, an, ks, str(b.get("dtype") or "bf16"), bool(b.get("overwrite")))

        try:
            stats = await asyncio.to_thread(run)
        except FileExistsError as e:
            return _bad(str(e), 409)
        except (ValueError, OSError) as e:
            return _bad(f"compact failed: {e}")
        return web.json_response({"lora_name": f"compact/{out.name}", **stats})
