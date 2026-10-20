"""Inputs -> prompts. Sweeps expand to a grid: one prompt per cell, each tagged with the
values that make it different, so the result can be laid out on axes."""

from __future__ import annotations

import itertools
import random
from typing import Any

from .manifest import LORA_CHAIN, MODEL_LOADERS, SWEEPABLE, App, AppError, pick_workflow, resolve_bind, titles

MAX_CELLS = 64


def _set(graph: dict, bind: str, value: Any) -> None:
    """Set the first of the bind's alternatives this workflow has (none: an optional input)."""
    alt = resolve_bind(graph, bind)
    if alt is None:
        return
    title, _, field_ = alt.partition(".")
    graph[titles(graph)[title]]["inputs"][field_] = value


def _lora_chain(graph: dict, loras: list[dict]) -> None:
    """Replace the LoRAs placeholder with a chain of N LoraLoaderModelOnly nodes (or
    remove it and rewire its consumers when N == 0)."""
    nid = titles(graph)[LORA_CHAIN]
    node = graph.pop(nid)
    upstream = node["inputs"]["model"]
    out = upstream
    for i, lo in enumerate(loras):
        new_id = f"{nid}_{i}"
        graph[new_id] = {"class_type": node["class_type"], "_meta": {"title": f"LoRA {i + 1}"},
                         "inputs": {"model": out, "lora_name": lo["name"], "strength_model": float(lo.get("strength", 1.0))}}
        out = [new_id, 0]
    for n in graph.values():
        for k, v in n["inputs"].items():
            if v == [nid, 0]:
                n["inputs"][k] = out


def _family_input(app: App) -> dict | None:
    return next((i for i in app.inputs if i["type"] == "family"), None)


def _one(app: App, values: dict[str, Any]) -> tuple[dict, str]:
    fam_in = _family_input(app)
    if fam_in is not None:
        family = values.get(fam_in["id"]) or fam_in.get("default")
        if not family:
            raise AppError(f"pick a model for {fam_in.get('label', fam_in['id'])}")
        graph = pick_workflow(app, None, str(family))
        _family_models(graph, str(family))
        layout = None
    else:
        model = values.get("model") or {}
        layout = model.get("kind") if isinstance(model, dict) else None
        graph = pick_workflow(app, layout)
    t = titles(graph)
    loras: list[dict] = []
    strength = None
    for inp in app.inputs:
        iid, typ, bind = inp["id"], inp["type"], inp.get("bind")
        v = values.get(iid, inp.get("default"))
        if isinstance(v, list) and typ not in ("loras", "resolution", "size"):
            v = v[0] if v else None  # a sweepable input given a list but not swept: first value
        if typ == "family":
            continue
        if typ == "model-file":  # a fine-tune in place of the workflow's build; empty = the workflow's
            if v:
                for b in ([bind] if isinstance(bind, str) else bind or []):
                    _set(graph, b, v)
            continue
        if typ == "model":
            if inp.get("required", True) and not isinstance(v, dict):
                raise AppError(f"pick a model for {inp.get('label', iid)}")
            for title, fields in MODEL_LOADERS.items():
                if title in t:
                    for src, dst in fields.items():
                        if v.get(src):
                            graph[t[title]]["inputs"][dst] = v[src]
            continue
        if typ == "loras":
            loras += [lo for lo in (v or []) if lo.get("name") and lo.get("on", True)]  # off rows stay listed, unapplied
            continue
        if typ == "lora":
            if v:
                loras.append({"name": v, "strength": 1.0})
            continue
        if bind == f"{LORA_CHAIN}.strength":
            strength = float(v)
            continue
        if typ == "seed" and (v is None or int(v) < 0):
            v = random.randint(0, 2**32 - 1)
        if typ in ("resolution", "size"):
            if v is None:
                continue  # keep the workflow's own size
            w, h = v
            _set(graph, bind[0], int(w))
            _set(graph, bind[1], int(h))
            continue
        if bind and v is not None:
            _set(graph, bind, v)
    if strength is not None:
        for lo in loras:
            lo["strength"] = strength
    if LORA_CHAIN in titles(graph):
        _lora_chain(graph, loras)
    return graph, layout or ""


def _family_models(graph: dict, family: str) -> None:
    """The workflow's own files where installed, else an installed build of the family."""
    from ..models import families
    from .catalog import _pick_models

    fam = families._all().get(family)
    if fam is not None:
        _pick_models(graph, fam)


def defaults_for(app: App, family: str | None) -> dict[str, Any]:
    """{input id: the selected workflow's own value} for inputs bound to one field, so the
    form starts where the workflow does (8 steps for a turbo model, 50 for a base one)."""
    graph = pick_workflow(app, None, family) if family else app.workflows[0][1]
    if family:
        _family_models(graph, family)  # the files a run would actually use
    t = titles(graph)
    out: dict[str, Any] = {}
    for inp in app.inputs:
        binds = [resolve_bind(graph, b) for b in ([inp["bind"]] if isinstance(inp.get("bind"), str) else inp.get("bind") or [])]
        binds = [b for b in binds if b and "." in b]
        if not binds:
            continue
        vals = []
        for b in binds:
            title, _, field_ = b.partition(".")
            v = graph[t[title]]["inputs"].get(field_)
            vals.append(None if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int) else v)  # a link: no value
        out[inp["id"]] = vals if inp["type"] in ("resolution", "size") else vals[0]
    return out


def compile_batch(app: App, inputs: dict[str, Any], sweep: dict[str, list] | None = None,
                  output_prefix: str = "zentrainer/apps") -> list[dict]:
    """[{coords: {input: value}, prompt: graph, save: node id}], one per grid cell."""
    sweep = {k: v for k, v in (sweep or {}).items() if isinstance(v, list) and v}
    for k in sweep:
        if app.input(k).get("type") not in SWEEPABLE or not app.input(k).get("sweep"):
            raise AppError(f"{app.input(k).get('label', k)} can't be swept")
    keys = list(sweep)
    combos = list(itertools.product(*(sweep[k] for k in keys))) or [()]
    if len(combos) > MAX_CELLS:
        raise AppError(f"that's {len(combos)} images; the limit is {MAX_CELLS} per batch")
    save_title = next((o["from"] for o in app.manifest.get("outputs", [])), "Save")
    cells = []
    for combo in combos:
        coords = dict(zip(keys, combo))
        graph, _ = _one(app, {**inputs, **coords})
        save = titles(graph).get(save_title)
        if save and "filename_prefix" in graph[save]["inputs"]:
            graph[save]["inputs"]["filename_prefix"] = output_prefix
        cells.append({"coords": coords, "prompt": graph, "save": save})
    return cells
