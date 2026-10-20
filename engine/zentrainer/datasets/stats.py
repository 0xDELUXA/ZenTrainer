"""A dataset's health, measured, and kept over time.

Each look computes the metrics below; when the dataset has changed since the last snapshot (files,
sizes, captions), the new one is appended to its history, so the panel can show what an edit did
(recaptioned, pruned, cropped) and the card can draw the latest five scores as a pentagon.

    captions      share captioned (the default set), and per caption set
    consistency   how even the captions are: length spread, identical captions, very short ones
    quality       share of pictures with no image issue: training-ready size, sharp (against the set's
                  own median), sensibly exposed, no letterbox bars (per-picture flags for the view)
    uniqueness    share of pictures that aren't near-duplicates of another (64-bit difference hash)
    faces/balance with faces on (faces extra): how much the faces agree (each against the others)
                  times the share of pictures without a face issue;
                  without one, how evenly the pictures spread over aspect ratios

Scores are 0-1 (1 = good) for the pentagon; the panel shows the real numbers behind them.
"""

from __future__ import annotations

import hashlib
import json
import statistics as st
import time
from collections import Counter
from pathlib import Path

from .. import safeio

TARGET_SIDE = 768  # a picture's short side at or above this is training-ready at the usual sizes
NEAR_DUP = 6  # bits of 64 a difference hash may differ by and still count as the same picture
SIMILAR = 14  # closer than this: the same shot or a close variation (unrelated pictures sit near 32)
SOFT = 16  # the uniqueness score starts to dip below this distance to the nearest other picture


def _store_dir() -> Path:
    import folder_paths

    d = Path(folder_paths.get_user_directory()) / "zentrainer" / "dataset_stats"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key(name: str) -> str:
    return hashlib.sha1(name.encode()).hexdigest()[:16]


def _look(path: Path) -> dict | None:
    """One pass over a picture (a clip's middle frame), at 256 px: its 64-bit difference hash
    (near-identical pictures land a few bits apart) and its image statistics."""
    import numpy as np
    from PIL import Image

    from ..harness import video

    try:
        if video.is_video(path):
            p = video.probe(path)
            if p is None or not p.frames:
                return None
            scale = 256 / max(p.width, p.height)
            px = video.read(path, [p.frames // 2], (max(16, round(p.width * scale)), max(16, round(p.height * scale))))[0]
            im = Image.fromarray((px.numpy() * 255).astype("uint8"))
        else:
            im = Image.open(path)
            im.draft("RGB", (512, 512))  # JPEGs decode small
            im = im.convert("RGB")
            im.thumbnail((256, 256))
        g = im.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        hp = list(g.getdata())
        bits = 0
        for r in range(8):
            for c in range(8):
                bits = (bits << 1) | (hp[r * 9 + c] > hp[r * 9 + c + 1])
        rgb = np.asarray(im, dtype=np.float32) / 255.0
        luma = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        lap = -4 * luma[1:-1, 1:-1] + luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:]
        mx, mn = rgb.max(-1), rgb.min(-1)
        sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
        return {"hash": bits, "bright": round(float(luma.mean()), 4), "contrast": round(float(luma.std()), 4),
                "sharp": round(float(lap.var()) * 1000, 4), "sat": round(float(sat.mean()), 4),
                "border": round(_border(luma), 3)}
    except Exception:
        return None


def _border(luma) -> float:
    """The share of the picture taken by flat bars along its edges (letterboxing, frames)."""
    import numpy as np

    h, w = luma.shape
    flat = lambda a: float(a.std()) < 0.02  # noqa: E731
    top = next((i for i in range(h // 3) if not flat(luma[i])), h // 3)
    bottom = next((i for i in range(h // 3) if not flat(luma[h - 1 - i])), h // 3)
    left = next((i for i in range(w // 3) if not flat(luma[:, i])), w // 3)
    right = next((i for i in range(w // 3) if not flat(luma[:, w - 1 - i])), w // 3)
    kept = max(0, h - top - bottom) * max(0, w - left - right)
    return 1 - kept / max(1, h * w) if np.isfinite(kept) else 0.0


def _looks(name: str, items: list) -> dict[str, dict | None]:
    """Each picture's look (hash + statistics), cached by mtime and size."""
    cache_file = _store_dir() / f"{_key(name)}.looks.json"
    try:
        cache = json.loads(cache_file.read_text())
    except (OSError, ValueError):
        cache = {}
    out, keep = {}, {}
    for it in items:
        st_ = it.path.stat()
        stamp = f"{st_.st_mtime_ns}|{st_.st_size}"
        hit = cache.get(str(it.path))
        if not hit or hit[0] != stamp:
            hit = [stamp, _look(it.path)]
        keep[str(it.path)] = hit
        out[str(it.path)] = hit[1]
    safeio.write_json(cache_file, keep, indent=None)
    return out


def _nearest(looks: dict[str, dict | None]) -> dict[str, int]:
    """Each picture's distance (differing hash bits) to its closest other picture."""
    vals = [(p, lk["hash"]) for p, lk in looks.items() if lk]
    best = {p: 64 for p, _ in vals}
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            d = bin(vals[i][1] ^ vals[j][1]).count("1")
            best[vals[i][0]] = min(best[vals[i][0]], d)
            best[vals[j][0]] = min(best[vals[j][0]], d)
    return best


def _near_duplicates(looks: dict[str, dict | None], within: int = NEAR_DUP) -> set[str]:
    """The pictures with another this close (a near-identical twin by default)."""
    return {p for p, d in _nearest(looks).items() if d <= within}


def image_flags(looks: dict[str, dict | None], sides: dict[str, int]) -> dict[str, list[str]]:
    """Why each picture might need a look: dark, blown out, flat, blurry (against the set's own
    median sharpness: what's sharp depends on the content), bars, greyscale, small, near-duplicate."""
    vals = [lk for lk in looks.values() if lk]
    sharp_med = st.median(lk["sharp"] for lk in vals) if vals else 0
    sat_med = st.median(lk["sat"] for lk in vals) if vals else 0
    near = _nearest(looks)
    out = {}
    for p, lk in looks.items():
        f = []
        if lk:
            if lk["bright"] < 0.15:
                f.append("dark")
            if lk["bright"] > 0.85:
                f.append("blown out")
            if lk["contrast"] < 0.06:
                f.append("flat")
            if sharp_med and lk["sharp"] < 0.25 * sharp_med:
                f.append("blurry")
            if lk["border"] > 0.08:
                f.append("bars")
            if lk["sat"] < 0.03 and sat_med > 0.08:
                f.append("greyscale")
        if sides.get(p, TARGET_SIDE) < TARGET_SIDE:
            f.append("small")
        if near.get(p, 64) <= NEAR_DUP:
            f.append("near-duplicate")
        elif near.get(p, 64) <= SIMILAR:
            f.append("similar")
        if f:
            out[p] = f
    return out


def compute(name: str, set_name: str = "default") -> dict:
    """The dataset's metrics and 0-1 scores, now, for one caption set: its captions, and without
    the pictures muted in it."""
    from PIL import Image

    from . import folders as datasets, faces
    from ..harness import video
    from ..harness.data import muted, scan

    root = datasets.folder(name)
    off = muted(root).get(set_name, set())
    items = [it for it in scan(root, warn=False) if str(it.path.relative_to(root)) not in off]
    n = len(items)
    caps = [(it.captions.get(set_name) or "").strip() for it in items]
    captioned = [c for c in caps if c]
    words = [len(c.split()) for c in captioned]
    sets = Counter(s for it in items for s, t in it.captions.items() if t.strip())

    sides, aspects, side_of = [], [], {}
    for it in items:
        try:
            if video.is_video(it.path):
                p = video.probe(it.path)
                w, h = (p.width, p.height) if p else (0, 0)
            else:
                with Image.open(it.path) as im:
                    w, h = im.size
        except Exception:
            continue
        if w and h:
            sides.append(min(w, h))
            aspects.append(round(w / h, 1))
            side_of[str(it.path)] = min(w, h)
    dup_caps = sum(c - 1 for c in Counter(captioned).values() if c > 1)
    looks = _looks(name, items)
    nearest = _nearest(looks)
    near = sum(d <= NEAR_DUP for d in nearest.values())
    similar = sum(NEAR_DUP < d <= SIMILAR for d in nearest.values())
    # each picture pays for how close its nearest other picture is: nothing past SOFT bits apart,
    # everything when identical
    closeness = [max(0.0, (SOFT - d) / SOFT) for d in nearest.values()]
    flags = image_flags(looks, side_of)
    lv = [lk for lk in looks.values() if lk]
    bright = sorted(lk["bright"] for lk in lv)
    count = lambda tag: sum(tag in f for f in flags.values())  # noqa: E731

    m = {
        "pictures": n,
        "videos": sum(video.is_video(it.path) for it in items),
        "captioned": len(captioned),
        "caption_sets": dict(sets),
        "caption_words_median": st.median(words) if words else 0,
        "caption_words_spread": round(st.pstdev(words) / st.mean(words), 3) if len(words) > 1 and st.mean(words) else 0.0,
        "caption_short": sum(w < 5 for w in words),
        "caption_identical": dup_caps,
        "short_side_median": int(st.median(sides)) if sides else 0,
        "below_target": sum(s < TARGET_SIDE for s in sides),
        "aspect_top_share": round(max(Counter(aspects).values()) / len(aspects), 3) if aspects else 0.0,
        "near_duplicates": near,
        "similar": similar,
        "brightness_median": round(st.median(bright), 3) if bright else 0,
        "brightness_spread": round(bright[int(len(bright) * 0.9)] - bright[int(len(bright) * 0.1)], 3) if len(bright) > 2 else 0,
        "too_dark": count("dark"),
        "blown_out": count("blown out"),
        "flat": count("flat"),
        "blurry": count("blurry"),
        "bars": count("bars"),
        "greyscale": count("greyscale"),
        "saturation_median": round(st.median(lk["sat"] for lk in lv), 3) if lv else 0,
        "image_issues": sum(1 for f in flags.values() if set(f) - {"near-duplicate", "similar"}),
        "muted": len(off),
        "masked": sum(1 for it in items if it.mask),
        "pairs": sum(1 for it in items if it.controls),
    }
    face_scores = None
    if faces.available():
        report = _cached_faces(name)
        if report is not None:
            rows = report["items"].values()
            issues = sum(1 for r in rows if r["faces"] != 1 or r["outlier"])
            likes = [r["likeness"] for r in rows if r["likeness"] is not None]
            identity = report.get("identity")
            m.update(face_coverage=report["with_face"], face_issues=issues,
                     face_likeness_median=round(st.median(likes), 3) if likes else None,
                     face_likeness_min=round(min(likes), 3) if likes else None,
                     face_one_identity=bool(report.get("one_identity")))
            # how much the faces agree (each against the others: 0.2 = strangers, 0.7 = clearly one
            # person), times the share of pictures without a face issue
            agree = 0.0 if identity is None else max(0.0, min(1.0, (identity - 0.2) / 0.5))
            face_scores = agree * (1 - issues / max(1, len(report["items"])))

    def clamp(x):
        return max(0.0, min(1.0, x))

    scores = {
        "captions": clamp(len(captioned) / n) if n else 0.0,
        "consistency": clamp((1 - min(1.0, m["caption_words_spread"]) * 0.5)
                             * (1 - (dup_caps + m["caption_short"]) / max(1, len(captioned)))) if captioned else 0.0,
        # quality: training-ready size, sharp, sensibly exposed, no bars
        "quality": clamp(1 - m["image_issues"] / n) if n else 0.0,
        "uniqueness": clamp(1 - sum(closeness) / n) if n else 0.0,
    }
    if face_scores is not None:
        scores["faces"] = clamp(face_scores)
    else:
        # one aspect ratio holding every picture is fine for a fixed-size set; the score only
        # dips when one ratio dominates a set that's otherwise mixed
        scores["balance"] = clamp(1.0 if len(set(aspects)) <= 1 else 1.2 - m["aspect_top_share"])
    root_s = str(root) + "/"
    return {"metrics": m, "scores": {k: round(v, 3) for k, v in scores.items()},
            "flags": {p.replace(root_s, "", 1): f for p, f in flags.items()}}


def _cached_faces(name: str) -> dict | None:
    """The faces report when the dataset has faces on (its setting; cached per picture)."""
    from . import folders as datasets, faces

    return faces.dataset_faces(name) if datasets.get_settings(name).get("faces") else None


def _fingerprint(name: str, set_name: str) -> str:
    """What the dataset is, for this set: its files, the set's captions, masks, pairs, mutes."""
    from . import folders as datasets
    from ..harness.data import muted, scan

    root = datasets.folder(name)
    h = hashlib.sha256(json.dumps(sorted(muted(root).get(set_name, ()))).encode())
    for it in scan(root, warn=False):
        s = it.path.stat()
        h.update(f"{it.path.name}|{s.st_mtime_ns}|{s.st_size}|{it.captions.get(set_name, '')}|{bool(it.mask)}|{len(it.controls)}".encode())
    return h.hexdigest()[:20]


def _history_file(name: str, set_name: str) -> Path:
    return _store_dir() / (f"{_key(name)}.jsonl" if set_name == "default" else f"{_key(name)}.{_key(set_name)}.jsonl")


def history(name: str, set_name: str = "default") -> list[dict]:
    try:
        lines = _history_file(name, set_name).read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:  # cut short: skip it, keep the rest
            continue
    return out


def snapshot(name: str, set_name: str = "default") -> dict:
    """Compute now. A new snapshot only when the dataset changed since the last one; otherwise the
    latest is refreshed in place (faces turned on, measured again), so the history is a record of
    edits. Returns {"current", "history"}."""
    now = compute(name, set_name)
    fp = _fingerprint(name, set_name)
    past = history(name, set_name)
    row = {"at": time.time(), "set": set_name, "fingerprint": fp, "metrics": now["metrics"], "scores": now["scores"]}
    if past and past[-1].get("fingerprint", "").split("|")[0] == fp:
        row["at"] = past[-1]["at"]
        past[-1] = row
    else:
        past.append(row)
    safeio.write_text(_history_file(name, set_name), "".join(json.dumps(r) + "\n" for r in past))
    return {"current": now, "history": past[-60:]}


def latest_scores(name: str) -> dict | None:
    """The scores of the caption set looked at most recently (its history was written last), cheap,
    for the datasets list: {"scores", "set"}; None before a first look."""
    files = [f for f in _store_dir().glob(f"{_key(name)}*.jsonl")]
    if not files:
        return None
    newest = max(files, key=lambda f: f.stat().st_mtime)
    try:
        last = json.loads(newest.read_text().splitlines()[-1])
    except (OSError, ValueError, IndexError):
        return None
    return {"scores": last["scores"], "set": last.get("set", "default")}
