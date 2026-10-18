"""ComfyUI workflows that carry App mode (`extra.linearData`): the templates made that
way, and any workflow a user saved in App mode. These are what the app importer starts from:
`describe` resolves each marked widget to the API input it feeds (via `workflow`), and the
role helpers (`roles`, `_fold_switches`, `_prune`) find and tidy a text-to-image graph's
prompt, size, seed and sampler, which is how the Text to Image app's per-model workflows were
first made from ComfyUI's templates. `_pick_models` points a graph's loaders at installed
files, which per-model apps use at run time.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .workflow import describe_spec, exposed_inputs, input_spec, to_api

LOADERS = {"UNETLoader": ("diffusion_models", "unet_name"), "CLIPLoader": ("text_encoders", "clip_name"),
           "VAELoader": ("vae", "vae_name"), "CheckpointLoaderSimple": ("checkpoints", "ckpt_name")}


class AppNotFound(LookupError):
    pass


# ---- ComfyUI's node definitions --------------------------------------------------------------

_info: tuple[float, dict] | None = None


def object_info() -> dict:
    """What /object_info returns, built in-process; cached a few seconds (file lists change)."""
    global _info
    if _info and time.time() - _info[0] < 10:
        return _info[1]
    import nodes
    from comfy_api.internal import _ComfyNodeInternal

    def node_info(name: str) -> dict:  # mirrors server.py's /object_info
        cls = nodes.NODE_CLASS_MAPPINGS[name]
        if issubclass(cls, _ComfyNodeInternal):
            return cls.GET_NODE_INFO_V1()
        it = cls.INPUT_TYPES()
        return {"input": it, "input_order": {k: list(v) for k, v in it.items()}, "output": cls.RETURN_TYPES,
                "output_name": getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES), "name": name,
                "display_name": nodes.NODE_DISPLAY_NAME_MAPPINGS.get(name, name),
                "output_node": bool(getattr(cls, "OUTPUT_NODE", False))}

    out = {}
    for name in list(nodes.NODE_CLASS_MAPPINGS):
        try:
            out[name] = node_info(name)
        except Exception:  # a broken custom node shouldn't take every app down
            continue
    info = json.loads(json.dumps(out, default=str))
    _info = (time.time(), info)
    return info


# ---- sources ---------------------------------------------------------------------------------

def _templates_dir() -> Path | None:
    from ..families import _templates_dir as d

    return d()


def _workflows_dir() -> Path:
    import folder_paths

    return Path(folder_paths.get_user_directory()) / "default" / "workflows"


def _linear(wf: dict) -> dict | None:
    ld = (wf.get("extra") or {}).get("linearData")
    return ld if isinstance(ld, dict) and ld.get("inputs") else None


def _uses_api_nodes(wf: dict) -> bool:
    return '"api_node"' in json.dumps(wf) or any(str(n.get("type", "")).endswith("Node") and "api" in str(n.get("type", "")).lower()
                                                 for n in wf.get("nodes", []))


def list_apps() -> list[dict]:
    apps: list[dict] = []
    root = _templates_dir()
    if root and (root / "index.json").is_file():
        for cat in json.loads((root / "index.json").read_text(encoding="utf-8")):
            for t in cat.get("templates", []):
                path = root / f"{t['name']}.json"
                try:
                    wf = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if not _linear(wf):
                    continue
                apps.append({"id": f"template/{t['name']}", "name": t.get("title") or t["name"], "source": "template",
                             "media": t.get("mediaType", "image"), "description": t.get("description", ""),
                             "thumbnail": f"/templates/{t['name']}-1.{t.get('mediaSubtype', 'webp')}",
                             "api": t["name"].startswith("api_") or t.get("openSource") is False,
                             "date": t.get("date")})
    wdir = _workflows_dir()
    if wdir.is_dir():
        for path in sorted(wdir.rglob("*.json")):
            try:
                wf = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not _linear(wf):
                continue
            rel = path.relative_to(wdir).as_posix()
            apps.append({"id": f"workflow/{rel}", "name": path.stem, "source": "workflow", "media": "image",
                         "description": f"Your workflow {rel}", "api": _uses_api_nodes(wf),
                         "date": time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime))})
    return apps


def _load(app_id: str) -> dict:
    kind, _, rest = app_id.partition("/")
    if kind == "template":
        root = _templates_dir()
        path = (root / f"{rest}.json") if root else None
        if not rest or "/" in rest or rest.startswith(".") or not path or not path.is_file():
            raise AppNotFound(app_id)
    elif kind == "workflow":
        wdir = _workflows_dir().resolve()
        path = (wdir / rest).resolve()
        if not path.is_relative_to(wdir) or path.suffix != ".json" or not path.is_file():
            raise AppNotFound(app_id)
    else:
        raise AppNotFound(app_id)
    return json.loads(path.read_text(encoding="utf-8"))


# ---- Text to Image: inputs by role -------------------------------------------------------------

def _src(prompt: dict, v: Any) -> str | None:
    return v[0] if isinstance(v, list) and len(v) == 2 and str(v[0]) in prompt else None


def _find_back(prompt: dict, start: str, want: tuple[str, ...], via: tuple[str, ...], depth: int = 6,
               ok=lambda node, name: True) -> tuple[str, str] | None:
    """Walk back (breadth first) from node `start` through inputs named in `via` (any, if
    empty) to the nearest node with an unlinked input named in `want`: (node id, input)."""
    seen, frontier = set(), [start]
    for _ in range(depth):
        nxt = []
        for nid in frontier:
            if nid in seen or nid not in prompt:
                continue
            seen.add(nid)
            inputs = prompt[nid]["inputs"]
            for w in want:
                if w in inputs and _src(prompt, inputs[w]) is None and ok(prompt[nid], w):
                    return nid, w
            nxt += [s for k, v in inputs.items() if (k in via or not via) and (s := _src(prompt, v))]
        frontier = nxt
    return None


def _text_source(node: dict, name: str) -> bool:
    """A prompt's text: a text field, or a string primitive's value (not a boolean's)."""
    return name != "value" or node["class_type"].startswith("PrimitiveString")


def _upstream(prompt: dict, start: str) -> set[str]:
    seen, todo = set(), [start]
    while todo:
        nid = todo.pop()
        if nid in seen or nid not in prompt:
            continue
        seen.add(nid)
        todo += [s for v in prompt[nid]["inputs"].values() if (s := _src(prompt, v))]
    return seen


def _enhancer(prompt: dict, text_node: str) -> tuple[str, str] | None:
    """A switch on the prompt's path whose "on" branch runs an LLM (TextGenerate): the
    template's prompt enhancer, as (node, input) holding its on/off value."""
    for nid in _upstream(prompt, text_node):
        n = prompt[nid]
        if "switch" not in n["inputs"] or "on_true" not in n["inputs"]:
            continue
        on = _src(prompt, n["inputs"]["on_true"])
        if on and any(prompt[x]["class_type"] == "TextGenerate" for x in _upstream(prompt, on)):
            sw = _src(prompt, n["inputs"]["switch"])
            if sw is None:
                return nid, "switch"
            if "value" in prompt[sw]["inputs"] and _src(prompt, prompt[sw]["inputs"]["value"]) is None:
                return sw, "value"
    return None


def _samplers(prompt: dict) -> list[str]:
    return [nid for nid, n in prompt.items() if any(k in n["inputs"] for k in ("positive", "guider")) and
            any(k in n["inputs"] for k in ("latent_image", "latent"))]


def roles(prompt: dict) -> dict[str, tuple[str, str]]:
    """{role: (node id, input)} for prompt, negative, width, height, seed, steps, cfg."""
    out: dict[str, tuple[str, str]] = {}
    sams = _samplers(prompt)
    if not sams:
        return out
    s = sams[0]
    ins = prompt[s]["inputs"]
    cond_via = ("positive", "negative", "conditioning", "conditioning_1", "conditioning_to", "cond", "guider")
    for role, start in (("prompt", "positive"), ("negative", "negative")):
        origin = _src(prompt, ins.get(start)) or (_src(prompt, ins.get("guider")) if "guider" in ins else None)
        if origin:
            if "guider" in ins and start in prompt[origin]["inputs"]:  # CFGGuider: follow its own positive/negative
                origin = _src(prompt, prompt[origin]["inputs"][start]) or origin
            enc = _find_back(prompt, origin, ("text",), cond_via, ok=lambda n, k: True) or (origin, "text")
            hit = _find_back(prompt, enc[0], ("text", "prompt", "value"), (), depth=8, ok=_text_source)
            if hit and (role == "prompt" or hit != out.get("prompt")):
                out[role] = hit
                if role == "prompt" and (en := _enhancer(prompt, enc[0])):
                    out["enhance"] = en
    lat = _src(prompt, ins.get("latent_image") or ins.get("latent"))
    if lat:
        for k in ("width", "height"):
            hit = _find_back(prompt, lat, (k,), ())
            if hit:
                out[k] = hit
        if "width" not in out:  # sized by a selector node instead (ResolutionSelector)
            for k in ("aspect_ratio", "megapixels"):
                hit = _find_back(prompt, lat, (k,), ())
                if hit:
                    out[k] = hit
    for role, names in (("seed", ("seed", "noise_seed")), ("steps", ("steps",)), ("cfg", ("cfg",))):
        hit = _find_back(prompt, s, names, ("noise", "sigmas", "guider", "model"), depth=3)
        if hit:
            out[role] = hit
    return out


ROLE_LABELS = {"prompt": "Prompt", "negative": "Negative prompt", "enhance": "Prompt enhancer", "width": "Width",
               "height": "Height", "aspect_ratio": "Aspect ratio", "megapixels": "Megapixels", "seed": "Seed",
               "steps": "Steps", "cfg": "CFG"}


def _api(wf: dict, info: dict) -> dict:
    """API format as is (our own workflows may be saved that way), else converted."""
    return wf if "nodes" not in wf else to_api(wf, info)


def _fold_switches(prompt: dict, keep: set[str]) -> None:
    """Replace each switch whose value is fixed (a literal, or an unlinked boolean primitive)
    with its active branch. Switches and primitives in `keep` (bound to a control) stay."""
    for nid in [n for n, v in prompt.items() if v["class_type"] == "ComfySwitchNode" and n not in keep]:
        ins = prompt[nid]["inputs"]
        sw = ins.get("switch")
        src = _src(prompt, sw)
        if src is not None:
            if src in keep or "value" not in prompt[src]["inputs"] or _src(prompt, prompt[src]["inputs"]["value"]) is not None:
                continue
            sw = prompt[src]["inputs"]["value"]
        if not isinstance(sw, bool):
            continue
        branch = ins.get("on_true" if sw else "on_false")
        del prompt[nid]
        for n in prompt.values():
            for k, v in n["inputs"].items():
                if isinstance(v, list) and len(v) == 2 and str(v[0]) == nid:
                    n["inputs"][k] = branch


def _prune(prompt: dict, outputs: list[str]) -> None:
    """Drop every node that doesn't feed one of `outputs`."""
    live: set[str] = set()
    for o in outputs:
        live |= _upstream(prompt, o)
    for nid in [n for n in prompt if n not in live]:
        del prompt[nid]


# ---- describe ------------------------------------------------------------------------

def _input_row(key: str, prompt: dict, nid: str, name: str, info: dict, label: str, role: str | None = None) -> dict:
    node = prompt[nid]
    spec = input_spec(info.get(node["class_type"], {}), name, node["inputs"])
    return {"key": key, "node_id": nid, "input": name, "class_type": node["class_type"], "role": role,
            "label": label, "title": node.get("_meta", {}).get("title") or node["class_type"],
            "value": node["inputs"].get(name), "spec": describe_spec(spec)}


def describe(app_id: str) -> dict:
    info = object_info()
    wf = _load(app_id)
    prompt = to_api(wf, info)
    ex = exposed_inputs(wf, info)
    rows = []
    for e in ex["inputs"]:
        key = f"{e['node_id']}.{e['input']}"
        if e["node_id"] in prompt and not e["linked"] and not any(r["key"] == key for r in rows):
            row = _input_row(key, prompt, e["node_id"], e["input"], info, e.get("label") or e["input"])
            row["targets"] = [t for t in e.get("targets") or [] if t != [e["node_id"], e["input"]]]
            rows.append(row)
    outputs = [i for o in ex["outputs"] for i in o["node_ids"]]
    return {"id": app_id, "inputs": rows, "outputs": outputs, "unresolved": ex.get("unresolved", [])}


def _is_output(node: dict) -> bool:
    return node["class_type"].startswith("SaveImage") or node["class_type"] == "PreviewImage"


def _set(prompt: dict, nid: str, name: str, value: Any) -> None:
    if nid in prompt:
        prompt[nid]["inputs"][name] = value


def _pick_models(prompt: dict, fam: Any) -> None:
    """Loaders keep the template's files when installed; otherwise an installed build of the
    same family (a distilled one for the diffusion model: these are for generating)."""
    from .. import families

    index = families._installed_index()
    for n in prompt.values():
        spec = LOADERS.get(n["class_type"])
        if not spec:
            continue
        folder, key = spec
        want = str(n["inputs"].get(key, "")).replace("\\", "/").rsplit("/", 1)[-1]
        if want in index.get(folder, {}):
            n["inputs"][key] = index[folder][want]
            continue
        files = [f for f in fam.files.get(folder, []) if f.name in index.get(folder, {})]
        files.sort(key=lambda f: not f.distilled)
        if files:
            n["inputs"][key] = index[folder][files[0].name]


