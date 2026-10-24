"""A dataset's analysis: the detail behind its health scores, for the Analysis page.

    buckets     where each picture lands at a training resolution (the trainer's own bucketing) and
                how much of it the crop cuts off; how many pictures each bucket gets
    captions    how often each phrase (or tag) appears across the captions; anything in most of
                them is something the LoRA will lean towards
    pictures    brightness, contrast, sharpness, size and aspect as values for histograms, and each
                picture's flags (stats.image_flags)
    duplicates  pairs of near-duplicate pictures (mirrored too), closest first

Pictures muted in the caption set are left out, as training leaves them out. Paths are relative to
the dataset folder (what the thumbnail route takes).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from . import stats

CROP_WARN = 0.15  # the crop cuts off more than this share of a picture: worth a look
FREQUENT = 0.5  # a phrase in more than this share of captions: the LoRA will lean towards it
MIN_CAPTIONS = 5  # fewer captions than this: too few for frequencies to mean much
TOP_PHRASES = 40

STOPWORDS = set("""
a an the and or but of in on at to for with from by as is are was were be been being it its this that these
those there here his her their our your my him she he they them we you i me us who whom which what where when
while into onto over under up down out off near behind beside between through during before after above below
very more most some any each other such no not only own same so than too just also can will would should could
has have had having do does did doing about against again further then once all both few nor until because if
one two image photo picture shows showing shown appears appear seen visible features featuring wearing wears
looks looking stands standing sits sitting holds holding background foreground front side left right slightly
""".split())


def kept_share(w: int, h: int, bucket: tuple[int, int]) -> float:
    """How much of a w×h picture survives the trainer's cover-and-centre-crop into `bucket`."""
    ar, br = w / h, bucket[0] / bucket[1]
    return min(ar / br, br / ar)


def buckets(sizes: dict[str, tuple[int, int]], resolution: int, step: int = 64) -> dict:
    from ..harness.data import bucket_sizes, pick_bucket

    options = bucket_sizes(resolution, step)
    rows: dict[tuple[int, int], list[str]] = {}
    cropped = []
    for rel, (w, h) in sizes.items():
        b = pick_bucket(w, h, options)
        rows.setdefault(b, []).append(rel)
        cropped.append({"path": rel, "bucket": list(b), "kept": round(kept_share(w, h, b), 4), "size": [w, h]})
    cropped.sort(key=lambda c: c["kept"])
    return {
        "resolution": resolution,
        "step": step,
        "rows": [{"size": list(b), "count": len(ps), "pictures": sorted(ps)}
                 for b, ps in sorted(rows.items(), key=lambda kv: (-len(kv[1]), kv[0]))],
        "placed": {c["path"]: {k: c[k] for k in ("bucket", "kept", "size")} for c in cropped},
        "cropped": [c for c in cropped if 1 - c["kept"] > CROP_WARN],
        "kept_median": cropped[len(cropped) // 2]["kept"] if cropped else 1.0,
        "lonely": sorted(rel for ps in rows.values() if len(ps) == 1 for rel in ps),
    }


def _is_tags(captions: list[str]) -> bool:
    """Comma-separated tags (booru style) rather than sentences: most captions split into several
    short pieces."""
    tagged = 0
    for c in captions:
        parts = [p for p in c.split(",") if p.strip()]
        if len(parts) >= 3 and sum(len(p.split()) for p in parts) / len(parts) <= 4:
            tagged += 1
    return tagged > len(captions) / 2


def _phrases(caption: str, tags: bool) -> set[str]:
    """The phrases a caption mentions, once each: its tags, or for prose its words and two- and
    three-word runs that neither start nor end on a filler word."""
    text = caption.lower()
    if tags:
        return {re.sub(r"\s+", " ", t).strip(" .") for t in text.split(",") if t.strip(" .")}
    out = set()
    for sentence in re.split(r"[.,;:!?()\[\]\n]", text):
        words = re.findall(r"[a-z0-9][a-z0-9'\-]*", sentence)
        for n in (1, 2, 3):
            for i in range(len(words) - n + 1):
                run = words[i:i + n]
                if run[0] in STOPWORDS or run[-1] in STOPWORDS or (n == 1 and len(run[0]) < 3):
                    continue
                out.add(" ".join(run))
    return out


def captions(texts: list[str], subject: str = "") -> dict:
    """How often each phrase appears across the captions (share of captions mentioning it), most
    first. A phrase whose every mention is inside a longer one is dropped for the longer one, and
    anything naming the subject (the trigger) is left out: it's meant to be in all of them."""
    texts = [t for t in (t.replace("[trigger]", " ").strip() for t in texts) if t]
    n = len(texts)
    if not n:
        return {"captions": 0, "style": None, "phrases": [], "frequent": []}
    tags = _is_tags(texts)
    counts: Counter = Counter()
    for t in texts:
        counts.update(_phrases(t, tags))
    names = {w for w in re.findall(r"[a-z0-9][a-z0-9'\-]*", subject.lower())} if subject else set()
    drop = {p for p in counts if names and names & set(p.split())}
    if not tags:
        # the most any longer phrase containing it was mentioned: as often as the phrase itself means
        # it never appears on its own
        # (a phrase naming the subject doesn't count: it's dropped, so it mustn't hide the rest)
        within: Counter = Counter()
        for longer, lc in counts.items():
            if longer in drop:
                continue
            words = longer.split()
            for size in range(1, len(words)):
                for i in range(len(words) - size + 1):
                    sub = " ".join(words[i:i + size])
                    within[sub] = max(within[sub], lc)
        drop |= {p for p, c in counts.items() if within[p] >= c}
    rows = [{"text": p, "count": c, "share": round(c / n, 3)} for p, c in counts.most_common() if p not in drop and c >= 2]
    rows = rows[:TOP_PHRASES]
    frequent = [r for r in rows if r["share"] > FREQUENT] if n >= MIN_CAPTIONS else []
    return {"captions": n, "style": "tags" if tags else "prose", "subject": subject, "phrases": rows, "frequent": frequent}


def duplicates(found: list[dict], rel: dict[str, str]) -> list[dict]:
    """stats.pairs() with paths relative to the dataset, strongest first."""
    return [{**q, "a": rel[q["a"]], "b": rel[q["b"]]} for q in found if q["a"] in rel and q["b"] in rel]


def analyze(name: str, set_name: str = "default", resolution: int = 1024) -> dict:
    from . import folders as datasets
    from ..harness.data import muted, scan

    root = datasets.folder(name)
    off = muted(root).get(set_name, set())
    items = [it for it in scan(root, warn=False) if str(it.path.relative_to(root)) not in off]
    rel = {str(it.path): str(it.path.relative_to(root)) for it in items}
    sizes = {}
    for it in items:
        wh = stats.size(it.path)
        if wh and wh[0] and wh[1]:
            sizes[rel[str(it.path)]] = wh
    looks = stats._looks(name, items)
    side_of = {str(it.path): min(sizes[rel[str(it.path)]]) for it in items if rel[str(it.path)] in sizes}
    found = stats.pairs(looks)
    flags = stats.image_flags(looks, side_of, found)
    sharp = [lk["sharp"] for lk in looks.values() if lk]
    sharp_med = sorted(sharp)[len(sharp) // 2] if sharp else 0
    pictures = []
    for it in items:
        p, lk = str(it.path), looks.get(str(it.path))
        wh = sizes.get(rel[p])
        pictures.append({
            "path": rel[p],
            "size": list(wh) if wh else None,
            "aspect": round(math.log2(wh[0] / wh[1]), 3) if wh else None,  # 0 square, +1 twice as wide
            "brightness": lk["bright"] if lk else None,
            "contrast": lk["contrast"] if lk else None,
            "sharpness": round(lk["sharp"] / sharp_med, 3) if lk and sharp_med else None,  # 1 = the set's median
            "jpeg_quality": lk.get("jpeg_q") if lk else None,
            "flags": flags.get(p, []),
        })
    subject = datasets.get_settings(name).get("caption_subject", "")
    return {
        "name": name,
        "set": set_name,
        "pictures": pictures,
        "buckets": buckets(sizes, resolution),
        "captions": captions([(it.captions.get(set_name) or "") for it in items], subject),
        "duplicates": duplicates(found, rel),
        "flag_counts": dict(Counter(f for p in pictures for f in p["flags"])),
    }
