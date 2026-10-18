"""Dataset folders in ComfyUI's input dir: list, inspect, edit captions, upload, delete.

Every path from a request is resolved and checked to stay inside its dataset folder.
"""

from __future__ import annotations

import re
from pathlib import Path

from .harness import video
from .harness.data import IMAGE_EXTS, MASK_SUFFIXES, MEDIA_EXTS, SET_NAME, caption_path, caption_sets, find_mask, scan

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,99}$")


class DatasetError(ValueError):
    pass


def root() -> Path:
    import folder_paths

    return Path(folder_paths.get_input_directory()).resolve()


def folder(name: str, must_exist: bool = True) -> Path:
    if not _NAME.match(name or "") or name.startswith((".", "_")):
        raise DatasetError(f"bad dataset name {name!r}")
    d = (root() / name).resolve()
    if d.parent != root():
        raise DatasetError(f"bad dataset name {name!r}")
    if must_exist and not d.is_dir():
        raise FileNotFoundError(name)
    return d


def _inside(d: Path, rel: str) -> Path:
    p = (d / rel).resolve()
    if d not in p.parents:
        raise DatasetError(f"path {rel!r} is outside the dataset")
    return p


def summary() -> list[dict]:
    r = root()
    out = []
    for d in sorted(p for p in r.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        items = scan(d, warn=False)
        if not items:
            continue
        stats = [i.path.stat() for i in items if i.path.exists()]
        out.append({
            "folder": d.name,
            "images": len(items),
            "masked": sum(1 for i in items if getattr(i, "mask", None)),
            "bytes": sum(st.st_size for st in stats),
            "modified": max((st.st_mtime for st in stats), default=0),
            "videos": sum(1 for i in items if video.is_video(i.path)),
            "steps_per_epoch": sum(i.repeats for i in items),
            "captioned": sum(1 for i in items if i.caption),
            "sets": sorted({s for i in items for s in i.captions} | {"default"}, key=lambda s: (s != "default", s)),
            "thumbs": [{"subfolder": str(i.path.parent.relative_to(r)), "filename": i.path.name,
                        "path": str(i.path.relative_to(d))} for i in items[:8]],
        })
    return out


def _set(name: str) -> str:
    if name != "default" and not SET_NAME.match(name or ""):
        raise DatasetError(f"bad caption set name {name!r}: lowercase letters, digits, - and _")
    return name


_mask_stats: dict[tuple, tuple] = {}


def mask_stats(mask: Path) -> tuple[float, int, int] | None:
    """(coverage, width, height) of a mask: coverage is the mean brightness, so the share of
    the image trained at full weight. Remembered per file version."""
    from PIL import Image, ImageStat

    try:
        st = mask.stat()
    except OSError:
        return None
    key = (str(mask), st.st_mtime_ns, st.st_size)
    if key not in _mask_stats:
        try:
            with Image.open(mask) as im:
                w, h = im.size
                g = im.convert("L")
                g.thumbnail((256, 256))
                _mask_stats[key] = (ImageStat.Stat(g).mean[0] / 255, w, h)
        except OSError:
            return None
    return _mask_stats[key]


def items(name: str, caption_ext: str = ".txt") -> dict:
    from PIL import Image

    d, r = folder(name), root()
    rows = []
    for it in scan(d, caption_ext, warn=False):
        clip = video.probe(it.path) if video.is_video(it.path) else None
        if clip is not None:
            w, h = clip.width, clip.height
        else:
            try:
                with Image.open(it.path) as im:
                    w, h = im.size
            except OSError:
                w = h = None
        rows.append({
            "path": str(it.path.relative_to(d)),
            "subfolder": str(it.path.parent.relative_to(r)),  # for /view?type=input
            "filename": it.path.name,
            "caption": it.caption,
            "repeats": it.repeats,
            "width": w,
            "height": h,
            "bytes": it.path.stat().st_size,
            "mask": str(it.mask.relative_to(d)) if it.mask else None,
            "captions": it.captions,
        })
        if video.is_video(it.path):
            rows[-1].update(kind="video", fps=round(clip.fps, 3) if clip else None,
                            frames=clip.frames if clip else None, duration=round(clip.duration, 2) if clip else None)
        if it.mask and (ms := mask_stats(it.mask)):
            cover, mw, mh = ms
            rows[-1]["mask_coverage"] = round(cover, 4)
            # a mask is stretched to its image: a different shape means it won't line up
            rows[-1]["mask_mismatch"] = bool(w and h and abs(mw / mh - w / h) > 0.02)
    sets = sorted({s for r in rows for s in r["captions"]} | {"default"}, key=lambda s: (s != "default", s))
    return {"folder": name, "items": rows, "sets": sets}


def set_caption(name: str, rel: str, caption: str, caption_ext: str = ".txt", set_name: str = "default") -> None:
    img = _inside(folder(name), rel)
    if img.suffix.lower() not in MEDIA_EXTS or not img.is_file():
        raise DatasetError(f"{rel!r} is not an image or video in this dataset")
    caption_path(img, _set(set_name), caption_ext).write_text(caption.strip() + "\n", encoding="utf-8")


def bulk(name: str, op: str, text: str = "", find: str = "", caption_ext: str = ".txt", set_name: str = "default") -> int:
    """prefix / suffix / replace across every caption of one set. Returns how many changed."""
    changed = 0
    for row in items(name, caption_ext)["items"]:
        old = row["captions"].get(_set(set_name), "")
        if op == "prefix":
            new = old if old.startswith(text) else (f"{text}, {old}" if old else text)
        elif op == "suffix":
            new = old if old.endswith(text) else (f"{old}, {text}" if old else text)
        elif op == "replace":
            if not find:
                raise DatasetError("replace needs 'find'")
            new = old.replace(find, text)
        else:
            raise DatasetError(f"unknown op {op!r}; use prefix, suffix or replace")
        if new != old:
            set_caption(name, row["path"], new, caption_ext, set_name)
            changed += 1
    return changed


def delete(name: str, rel: str, caption_ext: str = ".txt") -> None:
    img = _inside(folder(name), rel)
    if img.suffix.lower() not in MEDIA_EXTS or not img.is_file():
        raise DatasetError(f"{rel!r} is not an image or video in this dataset")
    mask = find_mask(img)
    sets = caption_sets(img, caption_ext)
    img.unlink()
    for set_name in sets:
        caption_path(img, set_name, caption_ext).unlink(missing_ok=True)
    if mask:
        mask.unlink(missing_ok=True)


def create(name: str) -> None:
    d = folder(name, must_exist=False)
    if d.exists():
        raise DatasetError(f"{name!r} already exists")
    d.mkdir()


def save_upload(name: str, filename: str, data: bytes) -> str:
    """An image, a caption file, or a mask (`<stem>-masklabel.png` / `<stem>.mask.png`,
    stored as `_masks/<stem>.png`), written into the dataset folder."""
    base = Path(filename).name
    p = Path(base)
    if p.suffix.lower() not in MEDIA_EXTS | {".txt"} or base.startswith("."):
        raise DatasetError(f"{base!r}: only images, videos, masks and .txt captions")
    d = folder(name)
    if p.suffix.lower() in IMAGE_EXTS and p.stem.endswith(MASK_SUFFIXES):
        stem = next(p.stem[: -len(x)] for x in MASK_SUFFIXES if p.stem.endswith(x))
        (d / "_masks").mkdir(exist_ok=True)
        target = _inside(d, f"_masks/{stem}{p.suffix.lower()}")
    else:
        target = _inside(d, base)
    target.write_bytes(data)
    return str(target.relative_to(d))


def thumbnail(name: str, rel: str, size: int, cache: Path) -> Path:
    """A cached JPEG thumbnail (longest side `size`) of an image, a video (a frame a little way
    in, past fades from black) or a mask in the dataset."""
    import hashlib

    from PIL import Image, ImageOps

    src = _inside(folder(name), rel)
    if src.suffix.lower() not in MEDIA_EXTS or not src.is_file():
        raise DatasetError(f"{rel!r} is not an image or video in this dataset")
    size = max(32, min(int(size), 1024))
    st = src.stat()
    key = hashlib.sha1(f"{src}|{st.st_mtime_ns}|{st.st_size}|{size}".encode()).hexdigest()
    out = cache / key[:2] / f"{key}.jpg"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        if video.is_video(src):
            im = video.poster(src, 0.1)
            if im is None:
                raise DatasetError(f"{rel!r}: can't decode this video")
            im.thumbnail((size, size), Image.Resampling.LANCZOS)
            im.convert("RGB").save(out, "JPEG", quality=85)
            return out
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            im.thumbnail((size, size), Image.Resampling.LANCZOS)
            im.convert("RGB").save(out, "JPEG", quality=85)
    return out
