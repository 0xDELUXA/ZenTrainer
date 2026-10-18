"""Faces (optional: `scripts/zt extras faces`, or pip install insightface onnxruntime
opencv-python-headless): InsightFace on the CPU, so it never competes with training for VRAM.

- A dataset's face report: how many faces each picture has, how big the main one is, and how much
  it looks like the rest (cosine to the dataset's average face). Outliers stand out.
- Likeness: how much a sample's face looks like the dataset's, per sample and checkpoint, so the
  best checkpoint of a character LoRA is a number instead of a guess (Fizgig's likeness scoring).

InsightFace's pretrained models (buffalo_l, antelopev2) are for non-commercial research use only;
they download to ~/.insightface/models on first use (or are already there).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

import numpy as np

PACK = "buffalo_l"  # ArcFace R50 (WebFace600K) + RetinaFace; "antelopev2" is the stronger R100
OUTLIER = 0.35  # cosine to the dataset's average face below this: probably not the same person
_app = None
_lock = threading.RLock()  # detect() holds it while _get() builds the app


def available() -> bool:
    try:
        import cv2  # noqa: F401
        import insightface  # noqa: F401
        import onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


def _get():
    global _app
    with _lock:
        if _app is None:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name=PACK, allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            _app = app
        return _app


def detect(rgb: np.ndarray) -> list[dict]:
    """Faces in an (H, W, 3) uint8 RGB picture, largest first: bbox, detector score, the share of
    the picture's height the face spans, and its unit ArcFace embedding."""
    h = rgb.shape[0]
    with _lock:  # the onnx sessions aren't shared across threads
        faces = _get().get(np.ascontiguousarray(rgb[..., ::-1]))
    out = [{"bbox": [float(v) for v in f.bbox], "score": float(f.det_score),
            "share": float((f.bbox[3] - f.bbox[1]) / max(h, 1)), "emb": np.asarray(f.normed_embedding, np.float32)}
           for f in faces]
    return sorted(out, key=lambda f: -f["share"])


def _frames(path: Path, n: int) -> list[np.ndarray]:
    """A picture, or n frames spread over a clip, as uint8 RGB."""
    from .harness import video

    if video.is_video(path):
        p = video.probe(path)
        if p is None or not p.frames:
            return []
        idx = sorted({min(p.frames - 1, round((p.frames - 1) * (i + 0.5) / n)) for i in range(n)})
        scale = min(1.0, 1024 / max(p.width, p.height))
        px = video.read(path, idx, (max(32, round(p.width * scale)), max(32, round(p.height * scale))))
        return [(f.numpy() * 255).astype(np.uint8) for f in px]
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > 1600:
            im.thumbnail((1600, 1600))
        return [np.asarray(im)]


def main_face(path: Path, frames: int = 3) -> tuple[int, dict | None]:
    """(most faces seen in one frame, the largest face across the frames)."""
    best, most = None, 0
    for rgb in _frames(path, frames):
        fs = detect(rgb)
        most = max(most, len(fs))
        if fs and (best is None or fs[0]["share"] > best["share"]):
            best = fs[0]
    return most, best


# ---- a dataset's faces --------------------------------------------------------------------------

def _cache_dir() -> Path:
    import folder_paths

    d = Path(folder_paths.get_user_directory()) / "zentrainer" / "faces"
    d.mkdir(parents=True, exist_ok=True)
    return d


def dataset_faces(name: str, progress=None) -> dict:
    """{"items": {rel: {"faces", "share", "likeness", "outlier"}}, "with_face", "total"}; each
    picture's face is cached by its mtime and size, so a re-check only looks at what changed."""
    from . import datasets
    from .harness.data import scan

    root = datasets.folder(name)
    cache_file = _cache_dir() / f"{hashlib.sha1(name.encode()).hexdigest()[:16]}.json"
    try:
        cache = json.loads(cache_file.read_text())
    except (OSError, ValueError):
        cache = {}
    items = scan(root, warn=False)
    rows: dict[str, dict] = {}
    for k, it in enumerate(items):
        rel = str(it.path.relative_to(root))
        st = it.path.stat()
        stamp = f"{PACK}|{st.st_mtime_ns}|{st.st_size}"
        hit = cache.get(rel)
        if not hit or hit.get("stamp") != stamp:
            n, face = main_face(it.path)
            hit = {"stamp": stamp, "faces": n, "share": face["share"] if face else 0.0,
                   "emb": face["emb"].tolist() if face else None}
            cache[rel] = hit
        rows[rel] = hit
        if progress:
            progress(k + 1, len(items))
    cache = {k: v for k, v in cache.items() if k in rows}
    cache_file.write_text(json.dumps(cache))
    c = _centroid([np.asarray(r["emb"], np.float32) for r in rows.values() if r["emb"]])
    out = {}
    for rel, r in rows.items():
        like = float(np.dot(np.asarray(r["emb"], np.float32), c)) if (r["emb"] and c is not None) else None
        out[rel] = {"faces": r["faces"], "share": round(r["share"], 4),
                    "likeness": round(like, 4) if like is not None else None,
                    "outlier": like is not None and like < OUTLIER}
    return {"items": out, "with_face": sum(1 for r in rows.values() if r["emb"]), "total": len(rows), "pack": PACK}


def _centroid(embs: list[np.ndarray]) -> np.ndarray | None:
    """The dataset's average face: a unit mean, refined once without the pictures furthest from it
    (other people, a bad detection), so a few strays don't pull it."""
    if not embs:
        return None
    e = np.stack(embs)
    c = e.mean(0)
    c /= np.linalg.norm(c) + 1e-9
    keep = e[(e @ c) >= OUTLIER]
    if len(keep) >= max(2, len(e) // 2):
        c = keep.mean(0)
        c /= np.linalg.norm(c) + 1e-9
    return c


def dataset_centroid(name: str) -> np.ndarray | None:
    rows = dataset_faces(name)["items"]  # cached: cheap after the first look
    cache_file = _cache_dir() / f"{hashlib.sha1(name.encode()).hexdigest()[:16]}.json"
    cache = json.loads(cache_file.read_text())
    return _centroid([np.asarray(cache[r]["emb"], np.float32) for r in rows if cache.get(r, {}).get("emb")])


def likeness(path: Path, centroid: np.ndarray, frames: int = 4) -> float | None:
    """How much a rendered sample's main face looks like the dataset's (cosine, -1..1; same
    person is typically above 0.5): the best face in each of a few frames, averaged."""
    scores = []
    for rgb in _frames(path, frames):
        fs = detect(rgb)
        if fs:
            scores.append(max(float(np.dot(f["emb"], centroid)) for f in fs[:3]))
    return round(float(np.mean(scores)), 4) if scores else None


class LikenessWorker:
    """Scores a run's samples off the training thread (CPU), appending {"step", "index", "score"}
    rows to the run's "likeness" stream as each finishes."""

    def __init__(self, store, run_id: str, dataset: str):
        import queue

        self.store, self.run_id, self.dataset = store, run_id, dataset
        self.q: queue.Queue = queue.Queue()
        self.centroid = None
        threading.Thread(target=self._run, daemon=True, name="zentrainer-likeness").start()

    def submit(self, row: dict) -> None:
        self.q.put(row)

    def close(self) -> None:
        self.q.put(None)

    def _run(self) -> None:
        import folder_paths

        try:
            self.centroid = dataset_centroid(self.dataset)
        except Exception:
            logging.warning("[ZenTrainer] likeness: couldn't read the dataset's faces", exc_info=True)
        if self.centroid is None:
            logging.info("[ZenTrainer] likeness off: no faces in the dataset")
        while True:
            row = self.q.get()
            if row is None:
                return
            if self.centroid is None:
                continue
            try:
                path = Path(folder_paths.get_output_directory()) / row["subfolder"] / row["filename"]
                score = likeness(path, self.centroid)
                out = {"run_id": self.run_id, "step": row["step"], "index": row["index"], "score": score}
                if row.get("manual"):
                    out.update(manual=True, base=bool(row.get("base")))
                self.store.append(self.run_id, "likeness", out)
                from . import events

                events.emit(events.LIKENESS, out)
            except Exception:
                logging.warning("[ZenTrainer] likeness failed for %s", row.get("filename"), exc_info=True)
