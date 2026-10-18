"""Disk use: what training and apps leave behind, and the room left where they write.

Per run: its LoRA checkpoints, its resume states (optimizer state saved beside every
checkpoint: often bigger than the LoRAs) and its samples. Per dataset: the images and the
latent / text-embedding caches training builds from them (in user/zentrainer/cache). Plus app outputs and free space on
each disk involved.

Cleaning is limited to what can't lose work: a finished run's resume states (only needed
to resume a stopped run) and dataset caches (rebuilt on the next run). Nothing while a run
is active: it may be writing either.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ACTIVE = ("queued", "loading", "training", "paused")
# resume states matter for these: they can still continue
RESUMABLE = ("stopped", "error", "interrupted", "rejected")


class StorageError(ValueError):
    pass


def size(path: Path) -> int:
    """Bytes under `path` (a file, or a folder walked without following links)."""
    try:
        if path.is_file():
            return path.stat().st_size
    except OSError:
        return 0
    total, stack = 0, [path]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _roots() -> dict[str, Path]:
    import folder_paths

    out = Path(folder_paths.get_output_directory()) / "zentrainer"
    user = Path(folder_paths.get_user_directory()) / "zentrainer"
    return {"loras": out / "loras", "samples": out / "samples", "apps": out / "apps",
            "states": user / "states", "cache": user / "cache"}


def _caches(folder: str, include_unknown: bool = False) -> list[Path]:
    """ZenTrainer's training caches (latents and conditioning) built from one dataset. Only ever
    these: other trainers' folders inside a dataset (_latent_cache/, .thumbs/...) aren't ours."""
    from .harness.data import cache_folder

    root = _roots()["cache"]
    if not root.is_dir():
        return []
    return [p for p in root.iterdir() if p.is_dir()
            and (cache_folder(p) == folder or (include_unknown and cache_folder(p) is None))]


def disks() -> list[dict]:
    """Free space on each disk ZenTrainer writes to, once per disk."""
    import folder_paths

    places = [("Outputs", folder_paths.get_output_directory()), ("Datasets", folder_paths.get_input_directory()),
              ("Run state", folder_paths.get_user_directory())]
    try:
        from .downloads import target_dir

        places.append(("Model downloads", str(target_dir("diffusion_models"))))
    except Exception:
        pass
    seen: dict[int, dict] = {}
    for label, p in places:
        try:
            dev = os.stat(p).st_dev
            if dev in seen:
                seen[dev]["labels"].append(label)
                continue
            u = shutil.disk_usage(p)
        except OSError:
            continue
        seen[dev] = {"labels": [label], "path": str(p), "total": u.total, "used": u.used, "free": u.free}
    return list(seen.values())


def report(store) -> dict:
    from . import datasets

    roots = _roots()
    runs = []
    for meta in store.list():
        rid = meta["run_id"]
        ckpts = store.read(rid, "checkpoints")
        ck_bytes = sum(size(Path(c["path"])) for c in ckpts if c.get("path"))
        states = size(roots["states"] / rid) + sum(size(Path(c["path"]).with_suffix(".state.pt")) for c in ckpts if c.get("path"))
        runs.append({"run_id": rid, "name": meta.get("name", rid), "status": meta.get("status"), "created": meta.get("created"),
                     "checkpoints": ck_bytes, "checkpoint_count": len(ckpts), "states": states,
                     "samples": size(roots["samples"] / rid)})
    runs.sort(key=lambda r: -(r["checkpoints"] + r["states"] + r["samples"]))
    sets = []
    for d in datasets.summary():
        cache = sum(size(p) for p in _caches(d["folder"]))
        sets.append({"folder": d["folder"], "count": d["images"], "images": size(datasets.root() / d["folder"]), "cache": cache})
    sets.sort(key=lambda d: -(d["images"] + d["cache"]))
    return {"disks": disks(), "runs": runs, "datasets": sets, "apps": size(roots["apps"])}


def _no_active(store) -> None:
    if any(m.get("status") in ACTIVE for m in store.list()):
        raise StorageError("a run is active: clean up once it finishes")


def clean_states(store, run_id: str) -> int:
    """Delete a finished run's resume states. Returns the bytes freed."""
    _no_active(store)
    meta = store.meta(run_id)  # raises for an unknown run
    if meta.get("status") in RESUMABLE:
        raise StorageError(f"{meta.get('name', run_id)} is {meta.get('status')}: its states are what resuming it needs")
    freed = 0
    d = _roots()["states"] / run_id
    if d.is_dir():
        freed += size(d)
        shutil.rmtree(d)
    for c in store.read(run_id, "checkpoints"):
        legacy = Path(c["path"]).with_suffix(".state.pt") if c.get("path") else None
        if legacy and legacy.is_file():
            freed += legacy.stat().st_size
            legacy.unlink()
    return freed


def clean_all_states(store) -> tuple[int, int]:
    """Every finished run's resume states. Runs that can still be resumed keep theirs.
    Returns (bytes freed, runs cleaned)."""
    _no_active(store)
    freed = runs = 0
    for meta in store.list():
        if meta.get("status") in RESUMABLE:
            continue
        n = clean_states(store, meta["run_id"])
        if n:
            freed += n
            runs += 1
    return freed, runs


def clean_cache(store, folder: str) -> int:
    """Delete the training caches built from a dataset (and any from before caches recorded their
    dataset); the next run rebuilds what it needs. Nothing inside the dataset folder is touched."""
    from . import datasets

    _no_active(store)
    datasets.folder(folder)  # must be a real dataset
    freed = 0
    for d in _caches(folder, include_unknown=True):
        freed += size(d)
        shutil.rmtree(d)
    return freed
